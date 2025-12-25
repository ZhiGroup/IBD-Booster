/**
 * Forward PBWT with Error Correction for P-smoother
 *
 * This file implements the forward pass of the P-smoother algorithm.
 * It processes sites from start to end, using both forward and reverse
 * PBWT divergence to identify IBS blocks for error correction.
 *
 * Key concepts:
 *   - Link array: Each haplotype has (hap_id, fwd_block, rev_block)
 *   - Radix sort by (fwd_block, rev_block) groups haplotypes into IBS blocks
 *   - Within each block, minority alleles are corrected to majority
 *   - Circular gap buffer delays output to allow accumulated corrections
 *
 * Parallelization:
 *   - Chromosome is divided into overlapping windows
 *   - Each window has a warmup phase (to build PBWT state) and active phase
 *   - Only the active phase produces output; warmup establishes correct state
 *
 * Based on P-smoother by Degui Zhi and Shaojie Zhang.
 */

#include "psmoother.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>

using namespace std;

// Flattened link array: link[i*3 + 0] = hap_id, link[i*3 + 1] = fwd_block, link[i*3 + 2] = rev_block
// This improves cache locality vs vector<vector<int>>

// Helper: counting sort for radix sort on flattened link array
// Uses pre-allocated count and output arrays to avoid allocation
static void countingSortFlat(
    vector<int>& link,      // Flattened M x 3 array
    int col,                // Column to sort by (0, 1, or 2)
    int M,
    vector<int>& count,     // Pre-allocated, size M+2
    vector<int>& output)    // Pre-allocated, size M*3
{
    // Clear count array
    fill(count.begin(), count.end(), 0);

    // Count occurrences
    for (int i = 0; i < M; ++i) {
        int key = link[i * 3 + col];
        count[key + 1]++;
    }

    // Cumulative sum (prefix sum)
    for (int i = 1; i <= M + 1; ++i) {
        count[i] += count[i - 1];
    }

    // Build output array (stable sort)
    for (int i = 0; i < M; ++i) {
        int key = link[i * 3 + col];
        int pos = count[key]++;
        output[pos * 3 + 0] = link[i * 3 + 0];
        output[pos * 3 + 1] = link[i * 3 + 1];
        output[pos * 3 + 2] = link[i * 3 + 2];
    }

    // Copy back
    memcpy(link.data(), output.data(), M * 3 * sizeof(int));
}

// Process a block of haplotypes for error correction (uses flattened link array)
// Standard version for single-threaded path
// IMPORTANT: gap_ptr is the current circular buffer position - must use (gap_ptr + k) % G indexing
static int processBlockFlat(
    const vector<int>& link,  // Flattened M x 3 array
    int start, int end,
    const vector<vector<uint8_t>>& orig_gap,
    vector<vector<uint8_t>>& edit_gap,
    int gap_size,
    int gap_ptr,  // Current circular buffer position
    int min_width,
    double rho)
{
    int blockSize = end - start;
    if (blockSize < min_width) return 0;

    int local_corrections = 0;

    // Count alleles in each gap position (using circular buffer indexing)
    vector<int> zero_count(gap_size), one_count(gap_size);
    for (int j = start; j < end; ++j) {
        int id = link[j * 3 + 0];
        for (int k = 0; k < gap_size; ++k) {
            int buf_idx = (gap_ptr + k) % gap_size;  // Circular buffer index
            if (orig_gap[buf_idx][id] == 0) ++zero_count[k];
            else ++one_count[k];
        }
    }

    // Error correct: if minority allele frequency < rho, correct to majority
    for (int k = 0; k < gap_size; ++k) {
        if (min(zero_count[k], one_count[k]) <= blockSize * rho) {
            uint8_t correctAllele = (zero_count[k] < one_count[k]) ? 1 : 0;
            int buf_idx = (gap_ptr + k) % gap_size;  // Circular buffer index
            for (int j = start; j < end; ++j) {
                int id = link[j * 3 + 0];
                if (edit_gap[buf_idx][id] != correctAllele) {
                    edit_gap[buf_idx][id] = correctAllele;
                    local_corrections++;
                }
            }
        }
    }
    return local_corrections;
}

// Process block with pre-allocated counting arrays (avoids per-call allocation)
// IMPORTANT: Counts from orig_gap (original values), writes corrections to edit_gap
// IMPORTANT: gap_ptr is the current circular buffer position - must use (gap_ptr + k) % G indexing
static int processBlockPrealloc(
    const vector<int>& link,  // Flattened M x 3 array
    int start, int end,
    const vector<vector<uint8_t>>& orig_gap,  // Original values for counting
    vector<vector<uint8_t>>& edit_gap,        // Where to write corrections
    vector<bool>& gap_modified,
    vector<int>& zero_count,  // Pre-allocated, size G
    vector<int>& one_count,   // Pre-allocated, size G
    int G,
    int gap_ptr,  // Current circular buffer position
    int min_width,
    double rho)
{
    int blockSize = end - start;
    if (blockSize < min_width) return 0;

    int local_corrections = 0;

    // Clear counts (faster than reallocating)
    for (int k = 0; k < G; ++k) {
        zero_count[k] = 0;
        one_count[k] = 0;
    }

    // Count alleles from ORIGINAL gap using circular buffer indexing
    for (int k = 0; k < G; ++k) {
        int buf_idx = (gap_ptr + k) % G;  // Circular buffer index
        for (int j = start; j < end; ++j) {
            int id = link[j * 3 + 0];
            if (orig_gap[buf_idx][id] == 0) ++zero_count[k];
            else ++one_count[k];
        }
    }

    // Check each gap position for correction
    for (int k = 0; k < G; ++k) {
        if (min(zero_count[k], one_count[k]) <= blockSize * rho) {
            uint8_t correctAllele = (zero_count[k] < one_count[k]) ? 1 : 0;
            int buf_idx = (gap_ptr + k) % G;  // Circular buffer index

            // Apply corrections to edit_gap
            for (int j = start; j < end; ++j) {
                int id = link[j * 3 + 0];
                if (edit_gap[buf_idx][id] != correctAllele) {
                    edit_gap[buf_idx][id] = correctAllele;
                    gap_modified[buf_idx] = true;
                    local_corrections++;
                }
            }
        }
    }
    return local_corrections;
}

// Window structure for parallel forward PBWT
struct ForwardWindow {
    int warmup_start;   // Start of warm-up phase (no corrections)
    int store_start;    // Start of active phase (corrections made)
    int store_end;      // End of active phase
};

// Create overlapping windows for forward PBWT
static vector<ForwardWindow> createForwardWindows(int N, int nthreads, int L, int G) {
    if (N == 0 || nthreads <= 0) return {};

    vector<ForwardWindow> windows;

    // Overlap should be at least L sites for PBWT block accuracy
    int overlap = max(L * 2, 100);

    // Usable sites for processing (need G sites ahead for gap buffer)
    int usable_sites = N - G;
    if (usable_sites <= 0) {
        // Too few sites, single window
        windows.push_back({0, 0, usable_sites - 1});
        return windows;
    }

    // Calculate step size
    int step = max((usable_sites - overlap) / nthreads, 1);

    int store_start = 0;
    for (int w = 0; w < nthreads && store_start < usable_sites; ++w) {
        ForwardWindow win;

        if (w == 0) {
            win.warmup_start = 0;
        } else {
            win.warmup_start = max(0, store_start - overlap);
        }

        win.store_start = store_start;

        if (w == nthreads - 1) {
            // Last window goes to end
            win.store_end = usable_sites - 1;
        } else {
            win.store_end = min(store_start + step + overlap / 2, usable_sites - 1);
        }

        windows.push_back(win);
        store_start = win.store_end + 1;

        if (store_start >= usable_sites) break;
    }

    return windows;
}

// Thread function for forward PBWT window
static void runForwardWindowThread(
    int window_id,
    const ForwardWindow& window,
    vector<vector<uint8_t>>& hap_data,  // Shared, but each thread writes to disjoint portion
    const vector<int>& reverse_pre,
    const vector<int>& reverse_div,
    int M, int N, int L, int G,
    int min_width, double rho,
    atomic<int>& total_corrections,
    atomic<int>& total_blocks,
    bool verbose)
{
    int local_corrections = 0;
    int local_blocks = 0;

    // Forward PBWT state
    vector<int> pre(M), div(M);
    iota(pre.begin(), pre.end(), 0);
    fill(div.begin(), div.end(), 0);

    vector<int> a(M), b(M), d(M), e(M);
    vector<int> block(M), rBlock(M);

    // Pre-allocate flattened link array and sort buffers (avoids allocation per site)
    vector<int> link(M * 3);           // Flattened: [hap_id, fwd_block, rev_block] x M
    vector<int> sort_count(M + 2);     // For counting sort
    vector<int> sort_output(M * 3);    // For counting sort output

    // Pre-allocate counting arrays for block processing (avoids per-block allocation)
    vector<int> zero_count(G), one_count(G);

    // Gap buffers - MUST have separate orig and edit buffers (matching original P-smoother)
    // orig_gap: used for counting alleles (never modified)
    // edit_gap: used for writing corrections
    vector<vector<uint8_t>> orig_gap(G, vector<uint8_t>(M));
    vector<vector<uint8_t>> edit_gap(G, vector<uint8_t>(M));
    int gap_ptr = 0;

    // Track which sites in the gap buffer have been modified
    vector<bool> gap_modified(G, false);

    // Track corrections for deferred write-back
    vector<pair<int, vector<uint8_t>>> corrections;  // (site, corrected_data)

    // Current gap buffer site indices (circular buffer of G sites)
    vector<int> gap_sites(G);
    for (int g = 0; g < G; ++g) {
        int site_idx = window.warmup_start + g;
        gap_sites[g] = site_idx;
        if (site_idx >= 0 && site_idx < N) {
            for (int i = 0; i < M; ++i) {
                orig_gap[g][i] = hap_data[site_idx][i];
                edit_gap[g][i] = hap_data[site_idx][i];
            }
        }
    }

    // Process sites from warmup_start to store_end
    for (int site = window.warmup_start; site <= window.store_end && site + G < N; ++site) {
        bool in_active_phase = (site >= window.store_start);
        int current_gap_site = gap_sites[gap_ptr];

        // Step 1: Update forward PBWT using original alleles from orig_gap
        int u = 0, v = 0;
        int p = site + 1, q = site + 1;
        for (int i = 0; i < M; ++i) {
            int id = pre[i];
            if (div[i] > p) p = div[i];
            if (div[i] > q) q = div[i];

            if (orig_gap[gap_ptr][id] == 0) {
                a[u] = id;
                d[u] = p;
                ++u;
                p = 0;
            } else {
                b[v] = id;
                e[v] = q;
                ++v;
                q = 0;
            }
        }
        for (int i = 0; i < u; ++i) {
            pre[i] = a[i];
            div[i] = d[i];
        }
        for (int i = 0; i < v; ++i) {
            pre[u + i] = b[i];
            div[u + i] = e[i];
        }

        // Step 2: Save site from PREVIOUS iteration BEFORE overwriting
        // Must save ALL sites in active phase (not just modified ones) to match single-threaded
        if (in_active_phase) {
            if (current_gap_site >= window.store_start && current_gap_site <= window.store_end) {
                corrections.emplace_back(current_gap_site, vector<uint8_t>(edit_gap[gap_ptr].begin(), edit_gap[gap_ptr].end()));
            }
        }

        // Step 3: Advance gap buffer BEFORE correction (matching original P-smoother)
        int next_site = site + G;
        if (next_site < N) {
            gap_sites[gap_ptr] = next_site;
            for (int i = 0; i < M; ++i) {
                orig_gap[gap_ptr][i] = hap_data[next_site][i];
                edit_gap[gap_ptr][i] = hap_data[next_site][i];
            }
            gap_modified[gap_ptr] = false;
        }
        gap_ptr = (gap_ptr + 1) % G;

        // Step 4: Error correction on NEW gap contents
        // IMPORTANT: Corrections must happen during WARMUP too, so gap buffer state is correct
        // at window boundaries. We just don't save/count warmup corrections.
        // NOTE: Original P-smoother increments site BEFORE computing rsite and block IDs
        // So we use (site + 1) to match that behavior
        int effective_site = site + 1;
        int rsite = (N - 1) - effective_site - G;
        if (rsite >= 0 && rsite < N) {
            size_t roffset = (size_t)rsite * M;

            // Initialize rBlock from reverse PBWT
            int rid = 0;
            for (int i = 0; i < M; ++i) {
                int rDiv = reverse_div[roffset + i];
                rDiv = (N - 1) - rDiv;
                if (rDiv < effective_site + (G - 1) + L) ++rid;
                rBlock[reverse_pre[roffset + i]] = rid;
            }

            // Initialize block from forward PBWT
            int fid = 0;
            for (int i = 0; i < M; ++i) {
                if (div[i] > effective_site - L) ++fid;
                block[pre[i]] = fid;
            }

            // Algorithm 2 - Block matching (flattened link array)
            for (int i = 0; i < M; ++i) {
                link[i * 3 + 0] = i;
                link[i * 3 + 1] = block[i];
                link[i * 3 + 2] = rBlock[i];
            }

            // Radix sort by reverse block, then forward block (using pre-allocated buffers)
            countingSortFlat(link, 2, M, sort_count, sort_output);
            countingSortFlat(link, 1, M, sort_count, sort_output);

            // Process blocks (using pre-allocated counting arrays)
            // Corrections always happen, but only count stats during active phase
            int start = 0;
            for (int i = 1; i < M; ++i) {
                if (link[i * 3 + 1] != link[(i-1) * 3 + 1] || link[i * 3 + 2] != link[(i-1) * 3 + 2]) {
                    int corr = processBlockPrealloc(link, start, i, orig_gap, edit_gap, gap_modified,
                                                     zero_count, one_count, G, gap_ptr, min_width, rho);
                    if (in_active_phase) {
                        local_corrections += corr;
                        local_blocks++;
                    }
                    start = i;
                }
            }
            int corr = processBlockPrealloc(link, start, M, orig_gap, edit_gap, gap_modified,
                                             zero_count, one_count, G, gap_ptr, min_width, rho);
            if (in_active_phase) {
                local_corrections += corr;
                local_blocks++;
            }
        }
    }

    // Save remaining sites in gap buffer
    // For the LAST window (store_end >= N - G - 1), we need to also save the final G sites
    // which are beyond store_end but still need to be output
    // Must save ALL sites (not just modified ones) to match single-threaded behavior
    bool is_last_window = (window.store_end >= N - G - 1);
    int effective_store_end = is_last_window ? (N - 1) : window.store_end;

    for (int j = 0; j < G; ++j) {
        int buf_idx = (gap_ptr + j) % G;
        int site = gap_sites[buf_idx];
        if (site >= window.store_start && site <= effective_store_end && site >= 0 && site < N) {
            corrections.emplace_back(site, vector<uint8_t>(edit_gap[buf_idx].begin(), edit_gap[buf_idx].end()));
        }
    }

    // Apply all corrections to hap_data (thread-safe: each thread writes to disjoint sites)
    for (const auto& corr : corrections) {
        int site = corr.first;
        for (int i = 0; i < M; ++i) {
            hap_data[site][i] = corr.second[i];
        }
    }

    // Accumulate stats
    total_corrections += local_corrections;
    total_blocks += local_blocks;

    if (verbose) {
        cerr << "[P-smoother] fwdPBWT Window " << window_id << " complete: "
             << "sites [" << window.store_start << ", " << window.store_end << "], "
             << local_corrections << " corrections, " << local_blocks << " blocks" << endl;
    }
}

// Note: Member functions countingSort and processBlock are no longer used
// All sorting now uses countingSortFlat with pre-allocated buffers
// All block processing uses processBlockFlat with flattened link array

// Run forward PBWT with error correction
void PSmoother::runForwardPBWT(std::vector<std::vector<uint8_t>>& hap_data) {
    if (params.verbose) {
        cerr << "[P-smoother] Running forward PBWT with error correction";
        if (params.nthreads > 1) {
            cerr << " with " << params.nthreads << " threads";
        }
        cerr << "..." << endl;
    }

    int L = params.length;
    int G = max(params.gap, 1);

    if (params.nthreads <= 1) {
        // Single-threaded path (original code)
        vector<int> pre(M), div(M);
        iota(pre.begin(), pre.end(), 0);
        fill(div.begin(), div.end(), 0);

        vector<int> a(M), b(M), d(M), e(M);
        vector<int> block(M), rBlock(M);

        // Pre-allocate flattened link array and sort buffers
        vector<int> link(M * 3);
        vector<int> sort_count(M + 2);
        vector<int> sort_output(M * 3);

        vector<vector<uint8_t>> orig_gap(G, vector<uint8_t>(M));
        vector<vector<uint8_t>> edit_gap(G, vector<uint8_t>(M));
        int gap_ptr = 0;

        for (int g = 0; g < G && g < N; ++g) {
            for (int i = 0; i < M; ++i) {
                orig_gap[g][i] = hap_data[g][i];
                edit_gap[g][i] = hap_data[g][i];
            }
        }

        for (int site = 0; site + G < N; ++site) {
            // Step 1: Write output from current gap position
            for (int i = 0; i < M; ++i) {
                hap_data[site][i] = edit_gap[gap_ptr][i];
            }

            // Step 2: Update forward PBWT using original alleles
            int u = 0, v = 0;
            int p = site + 1, q = site + 1;
            for (int i = 0; i < M; ++i) {
                int id = pre[i];
                if (div[i] > p) p = div[i];
                if (div[i] > q) q = div[i];

                if (orig_gap[gap_ptr][id] == 0) {
                    a[u] = id;
                    d[u] = p;
                    ++u;
                    p = 0;
                } else {
                    b[v] = id;
                    e[v] = q;
                    ++v;
                    q = 0;
                }
            }
            for (int i = 0; i < u; ++i) {
                pre[i] = a[i];
                div[i] = d[i];
            }
            for (int i = 0; i < v; ++i) {
                pre[u + i] = b[i];
                div[u + i] = e[i];
            }

            // Step 3: Advance gap buffer BEFORE correction (matching original P-smoother)
            int next_site = site + G;
            if (next_site < N) {
                for (int i = 0; i < M; ++i) {
                    orig_gap[gap_ptr][i] = hap_data[next_site][i];
                    edit_gap[gap_ptr][i] = hap_data[next_site][i];
                }
            }
            gap_ptr = (gap_ptr + 1) % G;

            // Step 4: Now correct the NEW gap contents (after advancing)
            // NOTE: Original P-smoother increments site BEFORE computing rsite and block IDs
            // So we use (site + 1) to match that behavior
            int effective_site = site + 1;
            int rsite = (N - 1) - effective_site - G;
            if (rsite >= 0 && rsite < N) {
                size_t roffset = (size_t)rsite * M;

                int rid = 0;
                for (int i = 0; i < M; ++i) {
                    int rDiv = reverse_div[roffset + i];
                    rDiv = (N - 1) - rDiv;
                    if (rDiv < effective_site + (G - 1) + L) ++rid;
                    rBlock[reverse_pre[roffset + i]] = rid;
                }

                int fid = 0;
                for (int i = 0; i < M; ++i) {
                    if (div[i] > effective_site - L) ++fid;
                    block[pre[i]] = fid;
                }

                // Fill flattened link array
                for (int i = 0; i < M; ++i) {
                    link[i * 3 + 0] = i;
                    link[i * 3 + 1] = block[i];
                    link[i * 3 + 2] = rBlock[i];
                }

                // Radix sort using pre-allocated buffers
                countingSortFlat(link, 2, M, sort_count, sort_output);
                countingSortFlat(link, 1, M, sort_count, sort_output);

                int start = 0;
                for (int i = 1; i < M; ++i) {
                    if (link[i * 3 + 1] != link[(i-1) * 3 + 1] || link[i * 3 + 2] != link[(i-1) * 3 + 2]) {
                        int corr = processBlockFlat(link, start, i, orig_gap, edit_gap, G, gap_ptr, params.width, params.rho);
                        corrections_count += corr;
                        if (i - start >= params.width) blocks_processed++;
                        start = i;
                    }
                }
                int corr = processBlockFlat(link, start, M, orig_gap, edit_gap, G, gap_ptr, params.width, params.rho);
                corrections_count += corr;
                if (M - start >= params.width) blocks_processed++;
            }

            if (params.verbose && (site % params.checkpoint == 0)) {
                cerr << "[P-smoother] Forward PBWT checkpoint: site " << site
                     << ", corrections: " << corrections_count << endl;
            }
        }

        for (int j = 0; j < G && (N - G + j) >= 0 && (N - G + j) < N; ++j) {
            int site = N - G + j;
            int buf_idx = (gap_ptr + j) % G;
            for (int i = 0; i < M; ++i) {
                hap_data[site][i] = edit_gap[buf_idx][i];
            }
        }
    } else {
        // Multi-threaded path
        vector<ForwardWindow> windows = createForwardWindows(N, params.nthreads, L, G);
        int actual_threads = static_cast<int>(windows.size());

        if (params.verbose) {
            cerr << "[P-smoother] Created " << actual_threads << " forward windows" << endl;
        }

        atomic<int> total_corrections(0);
        atomic<int> total_blocks(0);

        vector<thread> threads;
        for (int w = 0; w < actual_threads; ++w) {
            threads.emplace_back(runForwardWindowThread,
                               w, cref(windows[w]),
                               ref(hap_data),
                               cref(reverse_pre), cref(reverse_div),
                               M, N, L, G,
                               params.width, params.rho,
                               ref(total_corrections), ref(total_blocks),
                               params.verbose);
        }

        for (auto& t : threads) {
            t.join();
        }

        corrections_count = total_corrections.load();
        blocks_processed = total_blocks.load();
    }

    if (params.verbose) {
        cerr << "[P-smoother] Forward PBWT complete. "
             << corrections_count << " corrections in "
             << blocks_processed << " blocks." << endl;
    }
}
