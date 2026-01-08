/**
 * Reverse PBWT for P-smoother
 *
 * This file implements the reverse pass of the P-smoother algorithm.
 * It processes sites from end to start (backwards), computing the
 * reverse divergence array that will be used by the forward pass
 * to identify IBS blocks.
 *
 * The reverse PBWT is conceptually identical to the forward PBWT,
 * but processes sites in reverse order. This gives us divergence
 * values that indicate how far back (towards the end of the chromosome)
 * each pair of adjacent haplotypes match.
 *
 * Combined with forward divergence, we can identify contiguous IBS
 * blocks that span multiple sites in both directions.
 *
 * Parallelization uses overlapping windows similar to the forward pass.
 *
 * Based on P-smoother by Degui Zhi and Shaojie Zhang.
 */

#include "psmoother.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <thread>
#include <algorithm>
#include <cmath>

using namespace std;

// Window structure for parallel rPBWT
struct ReverseWindow {
    int start_site;     // Lowest site index to STORE (in original coordinates)
    int end_site;       // Highest site index to STORE (in original coordinates)
    int warmup_end;     // Site index to start warmup (beyond end_site)
};

// Create overlapping windows for reverse PBWT
// Windows are defined in terms of original site coordinates
// Processing goes from high site indices to low (backwards)
static vector<ReverseWindow> createReverseWindows(int N, int nthreads, int L) {
    if (N == 0 || nthreads <= 0) return {};

    vector<ReverseWindow> windows;

    // Overlap should be at least L sites (P-smoother block length)
    // This ensures divergence arrays have accurate values at boundaries
    int overlap = max(L * 2, 100);  // Conservative overlap

    // Calculate step size (sites per window, excluding overlap)
    int step = max((N - overlap) / nthreads, 1);

    // Create windows from high sites to low sites
    // Window 0 handles highest sites, Window nthreads-1 handles lowest
    int end_site = N - 1;

    for (int w = 0; w < nthreads && end_site >= 0; ++w) {
        ReverseWindow win;
        win.end_site = end_site;

        if (w == nthreads - 1) {
            // Last window goes to site 0
            win.start_site = 0;
        } else {
            // Calculate start with overlap for next window
            win.start_site = max(0, end_site - step - overlap + 1);
        }

        // Warmup extends beyond end_site (but capped at N-1)
        // For first window (highest sites), no warmup needed
        if (w == 0) {
            win.warmup_end = win.end_site;
        } else {
            win.warmup_end = min(N - 1, win.end_site + overlap);
        }

        windows.push_back(win);

        // Next window's end is overlapping with this window's start
        end_site = win.start_site + overlap - 1;
        if (end_site < 0) break;
    }

    return windows;
}

// Thread function to run reverse PBWT on a window
static void runReverseWindowThread(
    int window_id,
    const ReverseWindow& window,
    const vector<vector<uint8_t>>& hap_data,
    vector<int>& reverse_pre,
    vector<int>& reverse_div,
    int M,
    int N,
    int store_start,    // First site index this window should STORE
    int store_end,      // Last site index this window should STORE
    bool verbose)
{
    // PBWT state arrays
    vector<int> pre(M), div(M);
    iota(pre.begin(), pre.end(), 0);  // Initialize: [0, 1, 2, ..., M-1]
    fill(div.begin(), div.end(), 0);  // Initialize: all zeros

    vector<int> a(M), b(M), d(M), e(M);  // Temporary arrays for update

    // Process sites in REVERSE order (from warmup_end down to start_site)
    for (int site = window.warmup_end; site >= window.start_site; --site) {
        int reverse_site_idx = (N - 1) - site;  // Convert to reverse index

        // PBWT update algorithm
        int u = 0, v = 0;
        int p = reverse_site_idx + 1;
        int q = reverse_site_idx + 1;

        for (int i = 0; i < M; ++i) {
            int id = pre[i];
            if (div[i] > p) p = div[i];
            if (div[i] > q) q = div[i];

            // Get allele at this site for haplotype id
            uint8_t allele = hap_data[site][id];

            if (allele == 0) {
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

        // Update prefix and divergence arrays
        for (int i = 0; i < u; ++i) {
            pre[i] = a[i];
            div[i] = d[i];
        }
        for (int i = 0; i < v; ++i) {
            pre[u + i] = b[i];
            div[u + i] = e[i];
        }

        // Only store results if site is in our storage range
        if (site >= store_start && site <= store_end) {
            size_t offset = (size_t)reverse_site_idx * M;
            for (int i = 0; i < M; ++i) {
                reverse_pre[offset + i] = pre[i];
                reverse_div[offset + i] = div[i];
            }
        }
    }

    if (verbose) {
        cerr << "[P-smoother] rPBWT Window " << window_id << " complete: "
             << "processed sites [" << window.start_site << ", " << window.warmup_end << "], "
             << "stored [" << store_start << ", " << store_end << "]" << endl;
    }
}

// Run reverse PBWT on in-memory haplotype data
// hap_data[site][haplotype] = 0 or 1
// Stores results in reverse_pre and reverse_div (flattened arrays)
void PSmoother::runReversePBWT(const std::vector<std::vector<uint8_t>>& hap_data) {
    if (params.verbose) {
        cerr << "[P-smoother] Running reverse PBWT on " << N << " sites, " << M << " haplotypes";
        if (params.nthreads > 1) {
            cerr << " with " << params.nthreads << " threads";
        }
        cerr << "..." << endl;
    }

    // Allocate storage for reverse PBWT arrays
    // Stored as flattened [site][M] arrays
    reverse_pre.resize((size_t)N * M);
    reverse_div.resize((size_t)N * M);

    if (params.nthreads <= 1) {
        // Single-threaded path (original sequential code)
        vector<int> pre(M), div(M);
        iota(pre.begin(), pre.end(), 0);
        fill(div.begin(), div.end(), 0);

        vector<int> a(M), b(M), d(M), e(M);

        for (int site = N - 1; site >= 0; --site) {
            int reverse_site_idx = (N - 1) - site;

            int u = 0, v = 0;
            int p = reverse_site_idx + 1;
            int q = reverse_site_idx + 1;

            for (int i = 0; i < M; ++i) {
                int id = pre[i];
                if (div[i] > p) p = div[i];
                if (div[i] > q) q = div[i];

                uint8_t allele = hap_data[site][id];

                if (allele == 0) {
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

            size_t offset = (size_t)reverse_site_idx * M;
            for (int i = 0; i < M; ++i) {
                reverse_pre[offset + i] = pre[i];
                reverse_div[offset + i] = div[i];
            }

            if (params.verbose && (reverse_site_idx % params.checkpoint == 0)) {
                cerr << "[P-smoother] rPBWT checkpoint: " << reverse_site_idx << " sites processed" << endl;
            }
        }
    } else {
        // Multi-threaded path with windowing
        vector<ReverseWindow> windows = createReverseWindows(N, params.nthreads, params.length);
        int actual_threads = static_cast<int>(windows.size());

        if (params.verbose) {
            cerr << "[P-smoother] Created " << actual_threads << " reverse windows" << endl;
        }

        // Calculate storage boundaries for each window
        // Each window stores a non-overlapping portion
        vector<int> store_start(actual_threads), store_end(actual_threads);

        for (int w = 0; w < actual_threads; ++w) {
            if (w == 0) {
                // First window stores from its end down to midpoint with next
                store_end[w] = windows[w].end_site;
                if (actual_threads > 1) {
                    store_start[w] = (windows[w].start_site + windows[w + 1].end_site) / 2 + 1;
                } else {
                    store_start[w] = windows[w].start_site;
                }
            } else if (w == actual_threads - 1) {
                // Last window stores from midpoint with previous down to 0
                store_end[w] = store_start[w - 1] - 1;
                store_start[w] = 0;
            } else {
                // Middle windows store from midpoint with previous to midpoint with next
                store_end[w] = store_start[w - 1] - 1;
                store_start[w] = (windows[w].start_site + windows[w + 1].end_site) / 2 + 1;
            }

            // Clamp to valid ranges
            store_start[w] = max(0, min(store_start[w], N - 1));
            store_end[w] = max(0, min(store_end[w], N - 1));
        }

        // Launch worker threads for windows 0 to actual_threads-2
        // Main thread will handle the last window
        vector<thread> threads;
        for (int w = 0; w < actual_threads - 1; ++w) {
            threads.emplace_back(runReverseWindowThread,
                               w, cref(windows[w]),
                               cref(hap_data),
                               ref(reverse_pre), ref(reverse_div),
                               M, N,
                               store_start[w], store_end[w],
                               params.verbose);
        }

        // Main thread handles the last window
        int last_w = actual_threads - 1;
        runReverseWindowThread(
            last_w, cref(windows[last_w]),
            cref(hap_data),
            ref(reverse_pre), ref(reverse_div),
            M, N,
            store_start[last_w], store_end[last_w],
            params.verbose);

        // Wait for worker threads
        for (auto& t : threads) {
            t.join();
        }
    }

    if (params.verbose) {
        cerr << "[P-smoother] Reverse PBWT complete." << endl;
    }
}
