/**
 * IBD-Booster: Fast IBD Segment Detection with P-smoother Integration
 *
 * This program detects Identity-By-Descent (IBD) segments from phased VCF files
 * using the Positional Burrows-Wheeler Transform (PBWT) algorithm.
 *
 * Pipeline:
 *   1. Read phased VCF file into 2D haplotype array
 *   2. Run P-smoother for haplotype error correction
 *   3. Apply minMac filter and pack to bit-packed format
 *   4. Transpose to haplotype-major layout for cache-efficient extension
 *   5. Run PBWT-based seed collection and extension to find IBD segments
 *
 * Key optimizations:
 *   - P-smoother integration for genotyping error correction
 *   - Haplotype-major data layout for cache-friendly extension
 *   - Multi-threaded windowed processing with overlapping windows
 *   - Seed merging to eliminate redundant extension work
 *   - Striped deduplication for thread-safe segment output
 *
 */

#include "../include/pbwt.hpp"
#include "../include/vcf_utils.hpp"
#include "../include/psmoother.hpp"
#include "../include/feature_extractor.hpp"
#include "../include/xgb_predictor.hpp"
#include "../include/nn_predictor.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <utility>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <array>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>  // For madvise() huge page hints
#elif defined(_WIN32)
#include <windows.h>
#endif

// Simple parallel_for using std::thread (replaces OpenMP to avoid thread pool conflicts)
// Main thread participates in work to use exactly nthreads total
template<typename Func>
void parallel_for(int start, int end, int nthreads, Func&& func) {
    if (nthreads <= 1 || end - start <= 1) {
        for (int i = start; i < end; ++i) {
            func(i);
        }
        return;
    }

    int range = end - start;
    int chunk = (range + nthreads - 1) / nthreads;
    std::vector<std::thread> threads;
    threads.reserve(nthreads - 1);  // Main thread handles one chunk

    // Spawn nthreads-1 worker threads for chunks 0 to nthreads-2
    for (int t = 0; t < nthreads - 1; ++t) {
        int t_start = start + t * chunk;
        int t_end = std::min(t_start + chunk, end);
        if (t_start >= end) break;

        threads.emplace_back([t_start, t_end, &func]() {
            for (int i = t_start; i < t_end; ++i) {
                func(i);
            }
        });
    }

    // Main thread handles the last chunk
    int main_start = start + (nthreads - 1) * chunk;
    int main_end = end;
    for (int i = main_start; i < main_end; ++i) {
        func(i);
    }

    // Join worker threads
    for (auto& th : threads) {
        th.join();
    }
}

// SIMD for accelerated haplotype comparison
// AVX-512 (512 bits = 512 sites at a time) > AVX2 (256 bits) > 64-bit scalar
#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define USE_AVX512 1
#define USE_AVX2 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#endif

// Pin a thread to a specific CPU core for better cache locality
inline void pinThreadToCore(std::thread& t, int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "[WARN] Failed to pin thread to core " << core_id << "\n";
    }
#elif defined(_WIN32)
    DWORD_PTR mask = 1ULL << core_id;
    HANDLE handle = t.native_handle();
    if (SetThreadAffinityMask(handle, mask) == 0) {
        std::cerr << "[WARN] Failed to pin thread to core " << core_id << "\n";
    }
#endif
}

/**
 * Seed: Represents a candidate IBD segment before extension.
 * Seeds are collected during PBWT traversal when two haplotypes share
 * a long match in the PBWT prefix array.
 */
struct Seed {
    int hap1;        // First haplotype index
    int hap2;        // Second haplotype index
    int start_site;  // Starting marker index
    int end_site;    // Ending marker index (inclusive)
};

// Haplotype-major data layout for cache-friendly extension
// HapMajorData is defined in vcf_utils.hpp
// This function transposes site-major to haplotype-major layout (parallelized)
void transposeToHapMajor(HapMajorData& hap_major, const HapMetadata& meta,
                         const HapData& site_major, int nthreads = 1) {
    hap_major.n_haps = meta.n_haps;
    hap_major.n_sites = meta.n_sites;
    hap_major.bytes_per_hap = (meta.n_sites + 7) / 8;
    const int dense_stride = meta.dense_stride;

    // Allocate haplotype-major storage
    hap_major.data.resize((size_t)hap_major.n_haps * hap_major.bytes_per_hap, 0);

    // Request huge pages for this large allocation (2GB+ array)
    // Works when THP is set to 'madvise' or 'always' - no sudo required
    // Reduces TLB misses: 500K page table entries (4KB) -> 1K entries (2MB)
#ifdef __linux__
    madvise(hap_major.data.data(), hap_major.data.size(), MADV_HUGEPAGE);
#endif

    // Transpose: for each haplotype, gather bits from all sites
    // Parallelized by haplotype - each writes to different memory region (no race)
    uint8_t* data_ptr = hap_major.data.data();
    const uint8_t* src_ptr = site_major.data();
    const int local_n_sites = hap_major.n_sites;
    const int local_bytes_per_hap = hap_major.bytes_per_hap;

    parallel_for(0, hap_major.n_haps, nthreads, [=](int hap) {
        uint8_t* hap_ptr = data_ptr + (size_t)hap * local_bytes_per_hap;
        const int hap_byte = hap >> 3;
        const int hap_bit = hap & 7;

        for (int site = 0; site < local_n_sites; ++site) {
            // Get bit from site-major
            const uint8_t* site_ptr = src_ptr + (size_t)site * dense_stride;
            uint8_t bit = (site_ptr[hap_byte] >> hap_bit) & 1;
            // Set bit in haplotype-major
            if (bit) {
                hap_ptr[site >> 3] |= (1 << (site & 7));
            }
        }
    });
}

// Merge overlapping seeds for the same haplotype pair
// This significantly reduces extension work by eliminating redundant seeds
inline void mergeSeeds(std::vector<Seed>& seeds) {
    if (seeds.size() <= 1) return;

    // Sort by (min(hap1,hap2), max(hap1,hap2), start_site) for consistent ordering
    std::sort(seeds.begin(), seeds.end(), [](const Seed& a, const Seed& b) {
        int a_h1 = std::min(a.hap1, a.hap2);
        int a_h2 = std::max(a.hap1, a.hap2);
        int b_h1 = std::min(b.hap1, b.hap2);
        int b_h2 = std::max(b.hap1, b.hap2);
        if (a_h1 != b_h1) return a_h1 < b_h1;
        if (a_h2 != b_h2) return a_h2 < b_h2;
        return a.start_site < b.start_site;
    });

    // Merge overlapping/adjacent seeds for same hap pair
    size_t write_idx = 0;
    for (size_t i = 1; i < seeds.size(); ++i) {
        Seed& curr = seeds[write_idx];
        const Seed& next = seeds[i];

        // Same haplotype pair? (order-independent comparison)
        int curr_h1 = std::min(curr.hap1, curr.hap2);
        int curr_h2 = std::max(curr.hap1, curr.hap2);
        int next_h1 = std::min(next.hap1, next.hap2);
        int next_h2 = std::max(next.hap1, next.hap2);

        if (curr_h1 == next_h1 && curr_h2 == next_h2 &&
            next.start_site <= curr.end_site + 1) {
            // Merge: extend current seed's end
            curr.end_site = std::max(curr.end_site, next.end_site);
        } else {
            // Different pair or non-overlapping: move to next slot
            ++write_idx;
            if (write_idx != i) {
                seeds[write_idx] = next;
            }
        }
    }

    seeds.resize(write_idx + 1);
}

/**
 * Find true divergence point by scanning backward using hap_major layout.
 * This is a simplified version of nextStartHapMajor without gap handling,
 * used for fast true_dmin computation during seed collection.
 *
 * @return The site index where haplotypes first differ (true_dmin)
 */
inline int findTrueDmin(int hap1, int hap2, int dmin,
                        const HapMajorData& hap_major)
{
    if (dmin <= 0) return 0;

    const int bytes_per_hap = hap_major.bytes_per_hap;
    const uint8_t* __restrict h1_ptr = hap_major.data.data() + (size_t)hap1 * bytes_per_hap;
    const uint8_t* __restrict h2_ptr = hap_major.data.data() + (size_t)hap2 * bytes_per_hap;

    int m = dmin - 1;

    // 64-bit word scanning backward
    while (m >= 64) {
        int word_start_byte = (m - 63) >> 3;
        if (((m - 63) & 7) == 0) {
            uint64_t w1, w2;
            memcpy(&w1, h1_ptr + word_start_byte, 8);
            memcpy(&w2, h2_ptr + word_start_byte, 8);
            uint64_t xor_word = w1 ^ w2;
            if (xor_word == 0) {
                m -= 64;
                continue;
            }
            // Found mismatch - return position after it
            int highest_mismatch = 63 - __builtin_clzll(xor_word);
            return (m - 63) + highest_mismatch + 1;
        }
        break;
    }

    // Bit-by-bit for remainder
    while (m >= 0) {
        int byte_idx = m >> 3;
        int bit_idx = m & 7;
        if ((h1_ptr[byte_idx] ^ h2_ptr[byte_idx]) >> bit_idx & 1) {
            return m + 1;  // Mismatch at m, so true_dmin is m+1
        }
        --m;
    }

    return 0;
}

/**
 * Collect IBD seeds at a single site using PBWT divergence array.
 *
 * The PBWT maintains haplotypes in sorted order by their prefix. Haplotypes
 * with small divergence values (D[i]) share long matches with neighbors.
 * This function identifies pairs of haplotypes that:
 *   1. Share a match of at least minSeedMatch cM
 *   2. Span at least minMarkers markers
 *   3. Have different alleles at the current site (i.e., the match ends here)
 *
 * @param st          Current PBWT state (A=permutation, D=divergence)
 * @param site_idx    Current marker index
 * @param LastMarker  Index of the last marker
 * @param seeds       Output vector to append seeds to
 * @param meta        VCF metadata
 * @param hap_data    Bit-packed haplotype data
 * @param genPos      Genetic positions in cM
 * @param minSeedMatch Minimum seed length in cM
 * @param minMarkers  Minimum number of markers for a seed
 */
void collectSeedsSimple(
    const PBWTState& st,
    int site_idx,
    int LastMarker,
    std::vector<Seed>& seeds,
    const HapMetadata &meta,
    const HapData &hap_data,
    const HapMajorData& hap_major,  // For fast true_dmin computation
    const std::vector<double>& genPos,
    double minSeedMatch,
    int minMarkers)
{
    const int M = st.A.size();
    int i0 = 0;
    int na = 0;
    int nb = 0;

    // Thread-local caches for cache-friendly O(n²) pair processing
    // OPTIMIZATION: Cache incrementally during counting to avoid redundant get_hap() calls
    thread_local std::vector<uint8_t> alleles_cache;
    thread_local std::vector<int32_t> d_cache;      // Cache D values
    thread_local std::vector<int32_t> hapid_cache;  // Cache haplotype IDs
    if ((int)alleles_cache.size() < M) {
        alleles_cache.resize(M);
        d_cache.resize(M);
        hapid_cache.resize(M);
    }

    // Precompute constants for this site
    const int seed_end = site_idx - 1;
    const double genPos_seed_end = (site_idx > 0) ? genPos[seed_end] : 0.0;
    const double genPos_site_idx = genPos[site_idx];
    const bool is_last_marker = (site_idx == LastMarker);

    // Normal PBWT seed collection loop
    // OPTIMIZATION: Cache alleles/D/hapid incrementally during counting (single get_hap per haplotype)
    for (int i = 0; i < M; ++i) {
        int hap_id = st.A[i];
        int divergence = st.D[i];

        // Flush block when divergence exceeds threshold (simple marker count check)
        if (divergence > site_idx - minMarkers) {
            int block_size = i - i0;

            if (block_size >= 2 && ((na && nb) || is_last_marker)) {
                // Data already cached incrementally - process pairs directly
                for (int ia = 0; ia < block_size; ++ia) {
                    int dmin = 0;
                    uint8_t a1 = alleles_cache[ia];
                    int hap1 = hapid_cache[ia];

                    // Precompute threshold: dmin must be <= site_idx - minMarkers for enough markers
                    const int dmin_threshold = site_idx - minMarkers;

                    for (int ib = ia + 1; ib < block_size; ++ib) {
                        int d_val = d_cache[ib];
                        if (d_val > dmin) dmin = d_val;

                        // Early break: once dmin exceeds threshold, all subsequent ib will also fail
                        if (dmin > dmin_threshold) break;

                        uint8_t a2 = alleles_cache[ib];

                        if (a1 != a2) {
                            double length = genPos_seed_end - genPos[dmin];
                            if (length >= minSeedMatch)
                                seeds.emplace_back(Seed{hap1, hapid_cache[ib], dmin, seed_end});
                        } else if (is_last_marker) {
                            double length = genPos_site_idx - genPos[dmin];
                            if (length >= minSeedMatch)
                                seeds.emplace_back(Seed{hap1, hapid_cache[ib], dmin, site_idx});
                        }
                    }
                }
            }
            na = nb = 0;
            i0 = i;
        }

        // Cache + count in one operation (eliminates redundant get_hap call)
        int local_idx = i - i0;
        uint8_t allele = get_hap(site_idx, hap_id, meta, hap_data);
        hapid_cache[local_idx] = hap_id;
        alleles_cache[local_idx] = allele;
        d_cache[local_idx] = divergence;
        if (allele == 0) na++;
        else nb++;
    }

    // Process final block - data already cached incrementally
    int final_block_size = M - i0;
    if (final_block_size >= 2 && ((na && nb) || is_last_marker)) {
        // Precompute threshold: dmin must be <= site_idx - minMarkers for enough markers
        const int dmin_threshold = site_idx - minMarkers;

        for (int ia = 0; ia < final_block_size; ++ia) {
            int dmin = 0;
            uint8_t a1 = alleles_cache[ia];
            int hap1 = hapid_cache[ia];

            for (int ib = ia + 1; ib < final_block_size; ++ib) {
                int d_val = d_cache[ib];
                if (d_val > dmin) dmin = d_val;

                // Early break: once dmin exceeds threshold, all subsequent ib will also fail
                if (dmin > dmin_threshold) break;

                uint8_t a2 = alleles_cache[ib];

                if (a1 != a2) {
                    double length = genPos_seed_end - genPos[dmin];
                    if (length >= minSeedMatch)
                        seeds.emplace_back(Seed{hap1, hapid_cache[ib], dmin, seed_end});
                } else if (is_last_marker) {
                    double length = genPos_site_idx - genPos[dmin];
                    if (length >= minSeedMatch)
                        seeds.emplace_back(Seed{hap1, hapid_cache[ib], dmin, site_idx});
                }
            }
        }
    }
}

/**
 * SegmentKey: Unique identifier for an IBD segment for deduplication.
 * Overlapping windows may produce duplicate segments; this key
 * ensures each unique segment is output exactly once.
 */
struct SegmentKey {
    int hap1, hap2, start_bp, end_bp;

    bool operator==(const SegmentKey& other) const {
        return hap1 == other.hap1 && hap2 == other.hap2 &&
               start_bp == other.start_bp && end_bp == other.end_bp;
    }
};

// Fast hash for SegmentKey using FNV-1a style mixing
struct SegmentKeyHash {
    size_t operator()(const SegmentKey& k) const noexcept {
        // Pack into 128 bits and mix - much faster than 4 separate std::hash calls
        size_t h = 14695981039346656037ULL;  // FNV offset basis
        h ^= static_cast<size_t>(k.hap1);
        h *= 1099511628211ULL;  // FNV prime
        h ^= static_cast<size_t>(k.hap2);
        h *= 1099511628211ULL;
        h ^= static_cast<size_t>(k.start_bp);
        h *= 1099511628211ULL;
        h ^= static_cast<size_t>(k.end_bp);
        h *= 1099511628211ULL;
        return h;
    }
};

/**
 * BinarySegment: Compact binary format for IBD segments.
 * Stores site indices directly to avoid bp->site conversion on read.
 * 28 bytes vs ~45 bytes text, plus no parsing overhead.
 */
#pragma pack(push, 1)
struct BinarySegment {
    int32_t hap1;       // Combined haplotype index (sample * 2 + hap)
    int32_t hap2;
    int32_t start_idx;  // Site index (direct, no bp lookup needed)
    int32_t end_idx;
    int32_t start_bp;   // Keep bp for final output
    int32_t end_bp;
    float length_cm;
};
#pragma pack(pop)
static_assert(sizeof(BinarySegment) == 28, "BinarySegment must be 28 bytes");

/**
 * Sort Segments.bin by (hap1, hap2) for cache-friendly feature extraction.
 * Segments sharing haplotypes will be processed together.
 *
 * For files that fit in memory: in-memory sort
 * For large files: external merge sort (TODO)
 */
// Comparator for BinarySegment sorting by (hap1, hap2)
inline bool segmentLess(const BinarySegment& a, const BinarySegment& b) {
    if (a.hap1 != b.hap1) return a.hap1 < b.hap1;
    return a.hap2 < b.hap2;
}

// Entry for k-way merge heap
struct MergeEntry {
    BinarySegment seg;
    int file_idx;
    bool operator>(const MergeEntry& other) const {
        return !segmentLess(seg, other.seg);  // For min-heap
    }
};

/**
 * External merge sort for large binary segment files.
 * Phase 1: Split into sorted chunks that fit in memory
 * Phase 2: K-way merge of sorted chunks
 */
bool sortBinarySegments(const std::string& input_file, const std::string& output_file,
                        size_t max_memory_bytes = 4ULL * 1024 * 1024 * 1024) {
    // Get file size
    FILE* fin = fopen(input_file.c_str(), "rb");
    if (!fin) {
        std::cerr << "[ERROR] Cannot open " << input_file << " for reading\n";
        return false;
    }
    fseek(fin, 0, SEEK_END);
    long long file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    size_t n_segments = file_size / sizeof(BinarySegment);
    std::cerr << "[Sort] " << n_segments << " segments (" << file_size / (1024*1024) << " MB)\n";

    // Calculate chunk size (segments that fit in memory)
    size_t segments_per_chunk = max_memory_bytes / sizeof(BinarySegment);
    size_t n_chunks = (n_segments + segments_per_chunk - 1) / segments_per_chunk;

    auto total_start = std::chrono::steady_clock::now();

    if (n_chunks == 1) {
        // File fits in memory - simple in-memory sort
        std::vector<BinarySegment> segments(n_segments);
        size_t read = fread(segments.data(), sizeof(BinarySegment), n_segments, fin);
        fclose(fin);

        if (read != n_segments) {
            std::cerr << "[ERROR] Read only " << read << " of " << n_segments << " segments\n";
            return false;
        }

        std::cerr << "[Sort] In-memory sort...\n";
        std::sort(segments.begin(), segments.end(), segmentLess);

        FILE* fout = fopen(output_file.c_str(), "wb");
        if (!fout) {
            std::cerr << "[ERROR] Cannot open " << output_file << " for writing\n";
            return false;
        }
        fwrite(segments.data(), sizeof(BinarySegment), n_segments, fout);
        fclose(fout);

        auto total_end = std::chrono::steady_clock::now();
        auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
        std::cerr << "[Sort] Completed in " << total_sec << " seconds\n";
        return true;
    }

    // ========== EXTERNAL MERGE SORT ==========
    std::cerr << "[Sort] External sort: " << n_chunks << " chunks of ~"
              << segments_per_chunk << " segments each\n";

    // Phase 1: Create sorted chunk files
    std::vector<std::string> chunk_files;
    std::vector<BinarySegment> chunk_buffer(segments_per_chunk);

    for (size_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
        size_t remaining = n_segments - chunk_idx * segments_per_chunk;
        size_t chunk_size = std::min(segments_per_chunk, remaining);

        size_t read = fread(chunk_buffer.data(), sizeof(BinarySegment), chunk_size, fin);
        if (read != chunk_size) {
            std::cerr << "[ERROR] Failed to read chunk " << chunk_idx << "\n";
            fclose(fin);
            return false;
        }

        // Sort this chunk
        std::sort(chunk_buffer.begin(), chunk_buffer.begin() + chunk_size, segmentLess);

        // Write to temp file
        std::string chunk_file = input_file + ".tmp." + std::to_string(chunk_idx);
        FILE* fchunk = fopen(chunk_file.c_str(), "wb");
        if (!fchunk) {
            std::cerr << "[ERROR] Cannot create temp file " << chunk_file << "\n";
            fclose(fin);
            return false;
        }
        fwrite(chunk_buffer.data(), sizeof(BinarySegment), chunk_size, fchunk);
        fclose(fchunk);
        chunk_files.push_back(chunk_file);
    }
    fclose(fin);
    chunk_buffer.clear();
    chunk_buffer.shrink_to_fit();

    // Phase 2: K-way merge
    std::cerr << "[Sort] Merging " << n_chunks << " sorted chunks...\n";

    // Open all chunk files
    std::vector<FILE*> chunk_fps(n_chunks);
    for (size_t i = 0; i < n_chunks; ++i) {
        chunk_fps[i] = fopen(chunk_files[i].c_str(), "rb");
        if (!chunk_fps[i]) {
            std::cerr << "[ERROR] Cannot open chunk file " << chunk_files[i] << "\n";
            return false;
        }
    }

    // Output file
    FILE* fout = fopen(output_file.c_str(), "wb");
    if (!fout) {
        std::cerr << "[ERROR] Cannot open " << output_file << " for writing\n";
        return false;
    }

    // Initialize min-heap with first segment from each chunk
    std::priority_queue<MergeEntry, std::vector<MergeEntry>, std::greater<MergeEntry>> heap;

    for (size_t i = 0; i < n_chunks; ++i) {
        MergeEntry entry;
        if (fread(&entry.seg, sizeof(BinarySegment), 1, chunk_fps[i]) == 1) {
            entry.file_idx = static_cast<int>(i);
            heap.push(entry);
        }
    }

    // Output buffer for efficiency
    const size_t OUT_BUFFER_SIZE = 100000;
    std::vector<BinarySegment> out_buffer;
    out_buffer.reserve(OUT_BUFFER_SIZE);
    size_t merged_count = 0;

    // Merge
    while (!heap.empty()) {
        MergeEntry top = heap.top();
        heap.pop();

        out_buffer.push_back(top.seg);
        if (out_buffer.size() >= OUT_BUFFER_SIZE) {
            fwrite(out_buffer.data(), sizeof(BinarySegment), out_buffer.size(), fout);
            out_buffer.clear();
        }

        // Read next segment from same file
        MergeEntry next;
        if (fread(&next.seg, sizeof(BinarySegment), 1, chunk_fps[top.file_idx]) == 1) {
            next.file_idx = top.file_idx;
            heap.push(next);
        }
    }

    // Flush remaining
    if (!out_buffer.empty()) {
        fwrite(out_buffer.data(), sizeof(BinarySegment), out_buffer.size(), fout);
    }

    fclose(fout);

    // Close and delete temp files
    for (size_t i = 0; i < n_chunks; ++i) {
        fclose(chunk_fps[i]);
        std::remove(chunk_files[i].c_str());
    }

    auto total_end = std::chrono::steady_clock::now();
    auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    std::cerr << "[Sort] External sort completed in " << total_sec << " seconds\n";

    return true;
}

/**
 * Convert TSV segments to binary format for fast processing.
 * This runs as part of the augmentation phase.
 *
 * TSV format: sample1\thap1\tsample2\thap2\tstart_bp\tend_bp\tlength_cm
 * Binary format: BinarySegment (28 bytes, includes site indices)
 *
 * @param tsv_file     Path to input TSV file
 * @param bin_file     Path to output binary file
 * @param positions    VCF bp positions for bp->site index conversion
 * @return             Number of segments converted, or -1 on error
 */
int64_t convertTsvToBinary(const std::string& tsv_file, const std::string& bin_file,
                            const std::vector<int>& positions) {
    FILE* fin = fopen(tsv_file.c_str(), "r");
    if (!fin) {
        std::cerr << "[ERROR] Cannot open " << tsv_file << " for reading\n";
        return -1;
    }

    FILE* fout = fopen(bin_file.c_str(), "wb");
    if (!fout) {
        std::cerr << "[ERROR] Cannot open " << bin_file << " for writing\n";
        fclose(fin);
        return -1;
    }

    // Buffer for writing (reduces syscalls)
    const size_t BUFFER_SIZE = 16384;
    std::vector<BinarySegment> buffer;
    buffer.reserve(BUFFER_SIZE);

    int64_t count = 0;
    char line[256];

    auto start = std::chrono::steady_clock::now();

    while (fgets(line, sizeof(line), fin)) {
        int sample1, hap1_idx, sample2, hap2_idx, start_bp, end_bp;
        double length_cm;

        if (sscanf(line, "%d\t%d\t%d\t%d\t%d\t%d\t%lf",
                   &sample1, &hap1_idx, &sample2, &hap2_idx,
                   &start_bp, &end_bp, &length_cm) != 7) {
            continue;  // Skip malformed lines
        }

        // Convert sample+hap to combined haplotype index
        int hap1 = sample1 * 2 + hap1_idx;
        int hap2 = sample2 * 2 + hap2_idx;

        // Binary search for site indices
        auto start_it = std::lower_bound(positions.begin(), positions.end(), start_bp);
        auto end_it = std::lower_bound(positions.begin(), positions.end(), end_bp);

        int start_idx = static_cast<int>(start_it - positions.begin());
        int end_idx = static_cast<int>(end_it - positions.begin());

        // Adjust end_idx to be the actual site at or before end_bp
        if (end_idx > 0 && (end_it == positions.end() || *end_it > end_bp)) {
            end_idx--;
        }

        BinarySegment seg;
        seg.hap1 = hap1;
        seg.hap2 = hap2;
        seg.start_idx = start_idx;
        seg.end_idx = end_idx;
        seg.start_bp = start_bp;
        seg.end_bp = end_bp;
        seg.length_cm = static_cast<float>(length_cm);

        buffer.push_back(seg);
        count++;

        if (buffer.size() >= BUFFER_SIZE) {
            fwrite(buffer.data(), sizeof(BinarySegment), buffer.size(), fout);
            buffer.clear();
        }
    }

    // Flush remaining
    if (!buffer.empty()) {
        fwrite(buffer.data(), sizeof(BinarySegment), buffer.size(), fout);
    }

    fclose(fin);
    fclose(fout);

    auto end = std::chrono::steady_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cerr << "[TSV->Binary] Converted " << count << " segments in " << sec << " seconds\n";

    return count;
}

// Striped deduplication to reduce mutex contention with multiple threads
// Each stripe has its own mutex, so threads accessing different stripes don't block
class StripedDedup {
private:
    static constexpr int NUM_STRIPES = 256;  // Increased from 64 to reduce contention
    std::array<std::mutex, NUM_STRIPES> stripe_mtx;
    std::array<std::unordered_set<SegmentKey, SegmentKeyHash>, NUM_STRIPES> stripe_sets;
    SegmentKeyHash hasher;

public:
    StripedDedup() = default;

    void reserve(size_t total_capacity) {
        size_t per_stripe = (total_capacity + NUM_STRIPES - 1) / NUM_STRIPES;
        for (int i = 0; i < NUM_STRIPES; ++i) {
            stripe_sets[i].reserve(per_stripe);
        }
    }

    // Returns true if key was NEW (inserted), false if duplicate
    bool tryInsert(const SegmentKey& key) {
        size_t h = hasher(key);
        int stripe = h & (NUM_STRIPES - 1);  // Fast modulo for power of 2

        std::lock_guard<std::mutex> lock(stripe_mtx[stripe]);
        return stripe_sets[stripe].insert(key).second;
    }
};

// Per-thread output buffer for both binary and TSV segments
class OutputBuffer {
private:
    static const size_t BUFFER_THRESHOLD = 8192;  // ~8K segments
    std::vector<BinarySegment> bin_buffer;
    std::string tsv_buffer;
    FILE* bin_file;
    FILE* tsv_file;
    std::mutex& file_mtx;
    std::atomic<size_t>& segment_count;
    StripedDedup& dedup;
    size_t local_count = 0;

public:
    OutputBuffer(FILE* bin, FILE* tsv, std::mutex& mtx, std::atomic<size_t>& count, StripedDedup& dedup_ref)
        : bin_file(bin), tsv_file(tsv), file_mtx(mtx), segment_count(count), dedup(dedup_ref) {
        bin_buffer.reserve(BUFFER_THRESHOLD * 3 / 2);
        tsv_buffer.reserve(BUFFER_THRESHOLD * 50);  // ~50 bytes per TSV line
    }

    // Write segment to both buffers
    void writeBinary(const BinarySegment& seg) {
        bin_buffer.push_back(seg);

        // Also format TSV line
        char line[128];
        int len = snprintf(line, sizeof(line), "%d\t%d\t%d\t%d\t%d\t%d\t%.6f\n",
                          seg.hap1 / 2, seg.hap1 & 1, seg.hap2 / 2, seg.hap2 & 1,
                          seg.start_bp, seg.end_bp, static_cast<double>(seg.length_cm));
        tsv_buffer.append(line, len);

        local_count++;

        if (bin_buffer.size() >= BUFFER_THRESHOLD) {
            flush();
        }
    }

    void flush() {
        if (!bin_buffer.empty()) {
            std::lock_guard<std::mutex> lock(file_mtx);
            fwrite(bin_buffer.data(), sizeof(BinarySegment), bin_buffer.size(), bin_file);
            fwrite(tsv_buffer.data(), 1, tsv_buffer.size(), tsv_file);
            bin_buffer.clear();
            tsv_buffer.clear();
        }
        if (local_count > 0) {
            segment_count.fetch_add(local_count, std::memory_order_relaxed);
            local_count = 0;
        }
    }

    ~OutputBuffer() {
        flush();
    }
};

/**
 * SeedQueue: Thread-safe bounded queue for passing seed batches between
 * PBWT producer threads and seed extension consumer threads.
 */
class SeedQueue {
private:
    std::queue<std::vector<Seed>> queue;
    std::mutex mtx;
    std::condition_variable cv;
    const size_t capacity;
    bool finished_producing = false;

public:
    explicit SeedQueue(size_t cap) : capacity(cap) {}

    // Try to add seeds to queue (non-blocking, returns false if full)
    bool offer(std::vector<Seed>&& seeds) {
        std::unique_lock<std::mutex> lock(mtx);
        if (queue.size() >= capacity) {
            return false;  // Queue full
        }
        queue.push(std::move(seeds));
        cv.notify_one();
        return true;
    }

    // Poll with timeout
    bool poll(std::vector<Seed>& out, int timeout_ms = 50) {
        std::unique_lock<std::mutex> lock(mtx);
        if (cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [this] { return !queue.empty() || finished_producing; })) {
            if (!queue.empty()) {
                out = std::move(queue.front());
                queue.pop();
                return true;
            }
        }
        return false;
    }

    void markFinished() {
        std::unique_lock<std::mutex> lock(mtx);
        finished_producing = true;
        cv.notify_all();
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }
};

/**
 * Collect IBD seeds for multi-threaded windowed processing.
 *
 * Similar to collectSeedsSimple but handles window boundaries:
 * - Seeds starting before collectionStart are skipped (handled by prior window)
 * - Only the last window performs close-out at the chromosome end
 *
 * @param st              Current PBWT state
 * @param site_idx        Current marker index
 * @param seeds           Output vector for seeds
 * @param meta            VCF metadata
 * @param hap_data        Bit-packed haplotype data
 * @param genPos          Genetic positions in cM
 * @param minSeedMatch    Minimum seed length in cM
 * @param minMarkers      Minimum number of markers
 * @param maxIbsStart     Maximum divergence for IBS block membership
 * @param windowStart     First marker of this window (for warmup)
 * @param collectionStart First marker where seeds should be collected
 * @param windowEnd       Last marker of this window
 * @param isLastWindow    True if this is the final window (handles close-out)
 */
void collectSeeds(
    const PBWTState& st,
    int site_idx,
    std::vector<Seed>& seeds,
    const HapMetadata &meta,
    const HapData &hap_data,
    const HapMajorData& hap_major,  // For fast true_dmin computation
    const std::vector<double>& genPos,
    double minSeedMatch,
    int minMarkers,
    int maxIbsStart,
    int windowStart,
    int collectionStart,
    int windowEnd,
    bool isLastWindow)
{
    const int M = st.A.size();
    int i0 = 0;
    int na = 0;
    int nb = 0;

    // Thread-local caches for cache-friendly O(n²) pair processing
    // OPTIMIZATION: Cache alleles/D/hapid incrementally during counting to avoid
    // redundant get_hap() calls (was reading each allele twice: once for counting, once for caching)
    thread_local std::vector<uint8_t> alleles_cache;
    thread_local std::vector<int32_t> d_cache;      // Cache D values (avoid st.D[] lookups in inner loop)
    thread_local std::vector<int32_t> hapid_cache;  // Cache haplotype IDs (avoid st.A[] lookups)
    if ((int)alleles_cache.size() < M) {
        alleles_cache.resize(M);
        d_cache.resize(M);
        hapid_cache.resize(M);
    }

    // Precompute constants for this site
    const int seed_end_const = site_idx - 1;
    const bool is_closeout = isLastWindow && site_idx == windowEnd;

    // Normal PBWT seed collection loop
    // Java uses: if (d[j] <= maxIbsStart) to detect matching blocks
    // This accounts for genetic distance, not just marker count
    // OPTIMIZATION: Cache alleles/D/hapid incrementally during counting (single get_hap per haplotype)
    for (int i = 0; i < M; ++i) {
        int hap_id = st.A[i];
        int divergence = st.D[i];

        // Flush block when divergence exceeds maxIbsStart (matching Java's logic)
        // Java: if (d[j] <= maxIbsStart) means in block; else exit block
        if (divergence > maxIbsStart) {
            int block_size = i - i0;

            // PHASE 1 OPTIMIZATION: Early exit for trivial block (size < 2)
            // Process if polymorphic (na && nb) OR last window close-out
            if (block_size >= 2 && ((na && nb) || is_closeout)) {
                // Data already cached incrementally - process pairs directly using cached values
                for (int ia = 0; ia < block_size; ++ia) {
                    int dmin = 0;
                    uint8_t a1 = alleles_cache[ia];
                    int hap1 = hapid_cache[ia];

                    for (int ib = ia + 1; ib < block_size; ++ib) {
                        int d_val = d_cache[ib];
                        if (d_val > dmin) dmin = d_val;
                        uint8_t a2 = alleles_cache[ib];
                        int hap2 = hapid_cache[ib];

                        // Normalize haplotype order for consistent dedup across windows
                        int h1 = hap1, h2 = hap2;
                        if (h1 > h2) std::swap(h1, h2);

                        if (a1 != a2) {
                            // Mismatch: seed ends at site BEFORE the mismatch
                            int seed_end = seed_end_const;

                            // Check range first (seed must end at or after collectionStart-1)
                            bool in_range = (seed_end >= collectionStart) ||
                                           (seed_end == collectionStart - 1 && collectionStart > 0);
                            if (!in_range) continue;

                            // OPTIMIZATION: Early reject ONLY when dmin is NOT truncated
                            // When dmin > windowStart, we know dmin is accurate
                            // When dmin <= windowStart, dmin might be truncated, true_dmin could be much smaller
                            if (dmin > windowStart) {
                                double prelim_length = genPos[seed_end] - genPos[dmin];
                                if (prelim_length < minSeedMatch || dmin > maxIbsStart) continue;
                            }

                            // is_new_seed check: prevent duplicate detection across overlapping windows
                            // Java: ibsStart>windowStart || windowStart==0 || alleles differ at ibsStart-1
                            // TIMING FIX: Since we check BEFORE pbwt_step, our dmin is from site_idx-1 state.
                            // Java checks AFTER pbwt update, where D values could be site_idx+1.
                            // For mismatch seeds (a1 != a2), the upcoming pbwt_step would set divergence,
                            // so we use site_idx as the effective dmin for the is_new_seed check.
                            // This makes (site_idx > windowStart) true within the collection window.
                            // NOTE: get_hap calls here are for dmin-1, a DIFFERENT site - cannot be cached
                            int effective_dmin_for_newcheck = std::max(dmin, site_idx);
                            bool is_new_seed = (windowStart == 0) || (effective_dmin_for_newcheck > windowStart) ||
                                (dmin > 0 && get_hap(dmin - 1, h1, meta, hap_data) !=
                                             get_hap(dmin - 1, h2, meta, hap_data));
                            if (!is_new_seed) continue;

                            // Compute true_dmin using fast hap_major scan (only when truncated)
                            int true_dmin = (dmin <= windowStart && windowStart > 0)
                                          ? findTrueDmin(h1, h2, dmin, hap_major)
                                          : dmin;

                            double length = genPos[seed_end] - genPos[true_dmin];
                            int num_markers = seed_end - true_dmin + 1;
                            if (length >= minSeedMatch && num_markers >= minMarkers && true_dmin <= maxIbsStart) {
                                seeds.emplace_back(Seed{h1, h2, true_dmin, seed_end});
                            }
                        } else if (is_closeout) {
                            // Close-out ONLY for last window - no more overlapping windows to catch these seeds
                            // Intermediate windows: seeds continue into overlapping windows

                            // OPTIMIZATION: Early reject ONLY when dmin is NOT truncated
                            if (dmin > windowStart) {
                                double prelim_length = genPos[site_idx] - genPos[dmin];
                                if (prelim_length < minSeedMatch || dmin > maxIbsStart) continue;
                            }

                            // Compute true_dmin using fast hap_major scan (only when truncated)
                            int true_dmin = (dmin <= windowStart && windowStart > 0)
                                          ? findTrueDmin(h1, h2, dmin, hap_major)
                                          : dmin;

                            // For close-out, use same timing adjustment as mismatch seeds.
                            // After Java's PBWT update, ibsStart would be site_idx+1.
                            // Using max(dmin, site_idx) approximates this behavior.
                            int effective_dmin_for_newcheck = std::max(dmin, site_idx);
                            bool is_new_seed = (windowStart == 0) || (effective_dmin_for_newcheck > windowStart) ||
                                (dmin > 0 && get_hap(dmin - 1, h1, meta, hap_data) !=
                                             get_hap(dmin - 1, h2, meta, hap_data));

                            double length = genPos[site_idx] - genPos[true_dmin];
                            int num_markers = site_idx - true_dmin + 1;
                            if (is_new_seed && length >= minSeedMatch && num_markers >= minMarkers && true_dmin <= maxIbsStart) {
                                seeds.emplace_back(Seed{h1, h2, true_dmin, site_idx});
                            }
                        }
                    }
                }
            }
            na = nb = 0;
            i0 = i;
        }

        // Cache + count in one operation (eliminates redundant get_hap call)
        // Previously: counted alleles here, then re-read them during block flush
        int local_idx = i - i0;
        uint8_t allele = get_hap(site_idx, hap_id, meta, hap_data);
        hapid_cache[local_idx] = hap_id;
        alleles_cache[local_idx] = allele;
        d_cache[local_idx] = divergence;
        if (allele == 0) na++;
        else nb++;
    }

    // Process any remaining block - data already cached incrementally
    int final_block_size = M - i0;

    // PHASE 1 OPTIMIZATION: Early exit for trivial final block
    // Process if polymorphic (na && nb) OR last window close-out
    if (final_block_size >= 2 && ((na && nb) || is_closeout)) {
        // Use cached values directly (no separate caching loop needed)
        for (int ia = 0; ia < final_block_size; ++ia) {
            int dmin = 0;
            uint8_t a1 = alleles_cache[ia];
            int hap1 = hapid_cache[ia];

            for (int ib = ia + 1; ib < final_block_size; ++ib) {
                int d_val = d_cache[ib];
                if (d_val > dmin) dmin = d_val;
                uint8_t a2 = alleles_cache[ib];
                int hap2 = hapid_cache[ib];

                // Normalize haplotype order for consistent dedup across windows
                int h1 = hap1, h2 = hap2;
                if (h1 > h2) std::swap(h1, h2);

                if (a1 != a2) {
                    // Mismatch: seed ends at site BEFORE the mismatch
                    int seed_end = seed_end_const;

                    // Check range first
                    bool in_range = (seed_end >= collectionStart) ||
                                   (seed_end == collectionStart - 1 && collectionStart > 0);
                    if (!in_range) continue;

                    // OPTIMIZATION: Early reject ONLY when dmin is NOT truncated
                    if (dmin > windowStart) {
                        double prelim_length = genPos[seed_end] - genPos[dmin];
                        if (prelim_length < minSeedMatch || dmin > maxIbsStart) continue;
                    }

                    // is_new_seed check with timing fix (same as above)
                    // NOTE: get_hap calls for dmin-1 cannot be cached (different site)
                    int effective_dmin_for_newcheck = std::max(dmin, site_idx);
                    bool is_new_seed = (windowStart == 0) || (effective_dmin_for_newcheck > windowStart) ||
                        (dmin > 0 && get_hap(dmin - 1, h1, meta, hap_data) !=
                                     get_hap(dmin - 1, h2, meta, hap_data));
                    if (!is_new_seed) continue;

                    // Compute true_dmin using fast hap_major scan (only when truncated)
                    int true_dmin = (dmin <= windowStart && windowStart > 0)
                                  ? findTrueDmin(h1, h2, dmin, hap_major)
                                  : dmin;

                    double length = genPos[seed_end] - genPos[true_dmin];
                    int num_markers = seed_end - true_dmin + 1;
                    if (length >= minSeedMatch && num_markers >= minMarkers && true_dmin <= maxIbsStart) {
                        seeds.emplace_back(Seed{h1, h2, true_dmin, seed_end});
                    }
                } else if (is_closeout) {
                    // Close-out ONLY for last window
                    // OPTIMIZATION: Early reject ONLY when dmin is NOT truncated
                    if (dmin > windowStart) {
                        double prelim_length = genPos[site_idx] - genPos[dmin];
                        if (prelim_length < minSeedMatch || dmin > maxIbsStart) continue;
                    }

                    // Compute true_dmin using fast hap_major scan (only when truncated)
                    int true_dmin = (dmin <= windowStart && windowStart > 0)
                                  ? findTrueDmin(h1, h2, dmin, hap_major)
                                  : dmin;

                    // Close-out with timing adjustment (same as mismatch seeds)
                    int effective_dmin_for_newcheck = std::max(dmin, site_idx);
                    bool is_new_seed = (windowStart == 0) || (effective_dmin_for_newcheck > windowStart) ||
                        (dmin > 0 && get_hap(dmin - 1, h1, meta, hap_data) !=
                                     get_hap(dmin - 1, h2, meta, hap_data));

                    double length = genPos[site_idx] - genPos[true_dmin];
                    int num_markers = site_idx - true_dmin + 1;
                    if (is_new_seed && length >= minSeedMatch && num_markers >= minMarkers && true_dmin <= maxIbsStart) {
                        seeds.emplace_back(Seed{h1, h2, true_dmin, site_idx});
                    }
                }
            }
        }
    }
}

static bool is_vcf(const std::string& path) {
    auto n = path.size();
    return (n >= 4  && path.compare(n-4, 4, ".vcf")    == 0) ||
           (n >= 7  && path.compare(n-7, 7, ".vcf.gz") == 0);
}

// Fast inline allele comparison - avoids function call overhead and redundant computations
// Precomputed byte_offset and bit_shift for both haplotypes
// NOTE: Monomorphic site check removed - sites are pre-filtered by minMac during VCF loading
// Uses direct stride calculation instead of site_offsets lookup (all sites are dense)
inline bool allelesDifferFast(int site_idx,
                               int byte_off1, int bit_shift1,
                               int byte_off2, int bit_shift2,
                               const HapMetadata& meta,
                               const HapData& hap_data)
{
    const uint8_t* site_ptr = hap_data.data() + (size_t)site_idx * meta.dense_stride;
    uint8_t val1 = (site_ptr[byte_off1] >> bit_shift1) & 1;
    uint8_t val2 = (site_ptr[byte_off2] >> bit_shift2) & 1;
    return val1 != val2;
}

/**
 * Extend a seed backwards to find the true IBD segment start.
 *
 * Scans backwards from the seed start, allowing gaps up to maxGap bp.
 * Uses haplotype-major layout for cache efficiency (sequential memory access
 * for each haplotype rather than strided access across all haplotypes).
 *
 * Optimizations:
 *   - 64-bit word scanning to skip 64 matching sites at once
 *   - XOR-based comparison (branchless for matches)
 *   - Memory prefetching for backward scan
 *
 * @return New start position, or -1 if a preceding seed would cover this segment
 */
inline int nextStartHapMajor(int hap1, int hap2, int start,
              const std::vector<double>& genPos,
              const HapMetadata& meta,
              const HapMajorData& hap_major,
              double minSeedMatch,
              double maxGap,
              int minMarkers,
              double minExtend,
              int mM1)
{
    if (__builtin_expect(start < 2 || maxGap < 0, 0)) {
        return start;
    }

    const int* __restrict bp_positions = meta.vcf_bp_positions.data();
    const int bytes_per_hap = hap_major.bytes_per_hap;

    // Get pointers to each haplotype's data
    const uint8_t* __restrict h1_ptr = hap_major.data.data() + (size_t)hap1 * bytes_per_hap;
    const uint8_t* __restrict h2_ptr = hap_major.data.data() + (size_t)hap2 * bytes_per_hap;

    int m = start - 1;
    int firstMismatchPos = bp_positions[m];
    int firstMatch = start - 2;

    // Prefetch the memory region we'll be scanning (backwards)
    if (m >= 512) {
        __builtin_prefetch(h1_ptr + ((m - 512) >> 3), 0, 3);
        __builtin_prefetch(h2_ptr + ((m - 512) >> 3), 0, 3);
    }

#ifdef USE_AVX512
    // AVX-512: 512-bit scanning (512 sites at a time) - 8x faster than 64-bit
    while (m >= 512) {
        if (((m - 511) & 7) == 0) {  // Byte-aligned
            int word_start_byte = (m - 511) >> 3;
            __m512i v1 = _mm512_loadu_si512((const __m512i*)(h1_ptr + word_start_byte));
            __m512i v2 = _mm512_loadu_si512((const __m512i*)(h2_ptr + word_start_byte));
            __m512i xor_result = _mm512_xor_si512(v1, v2);

            if (_mm512_test_epi64_mask(xor_result, xor_result) == 0) {
                m -= 512;  // All 512 bits match, skip ahead
                continue;
            }

            // Has mismatches - find the highest set bit position
            alignas(64) uint64_t words[8];
            _mm512_store_si512((__m512i*)words, xor_result);

            for (int i = 7; i >= 0; --i) {
                if (words[i] != 0) {
                    int highest_mismatch = 63 - __builtin_clzll(words[i]);
                    m = (m - 511) + i * 64 + highest_mismatch;
                    break;
                }
            }
        }
        break;
    }
#endif

#ifdef USE_AVX2
    // AVX2: 256-bit scanning (256 sites at a time) - 4x faster than 64-bit
    while (m >= 256) {
        // Check alignment: we want bits [m-255, m] to be in 32 consecutive bytes
        if (((m - 255) & 7) == 0) {  // Byte-aligned
            int word_start_byte = (m - 255) >> 3;
            __m256i v1 = _mm256_loadu_si256((const __m256i*)(h1_ptr + word_start_byte));
            __m256i v2 = _mm256_loadu_si256((const __m256i*)(h2_ptr + word_start_byte));
            __m256i xor_result = _mm256_xor_si256(v1, v2);

            if (_mm256_testz_si256(xor_result, xor_result)) {
                m -= 256;  // All 256 bits match, skip ahead
                continue;
            }

            // Has mismatches - find the highest set bit position
            // Extract as 4 x 64-bit words (words[3] contains highest sites [m-63, m])
            alignas(32) uint64_t words[4];
            _mm256_store_si256((__m256i*)words, xor_result);

            // Check from highest to lowest word
            for (int i = 3; i >= 0; --i) {
                if (words[i] != 0) {
                    int highest_mismatch = 63 - __builtin_clzll(words[i]);
                    m = (m - 255) + i * 64 + highest_mismatch;
                    break;
                }
            }
        }
        break;  // Not aligned or found mismatch, switch to finer scanning
    }
#endif

    // 64-bit word scanning: skip 64 matching sites at once when aligned
    // Fallback for non-AVX2 or when fewer than 256 sites remain
    while (m >= 64) {
        // Check alignment: we want bits [m-63, m] to be in one 64-bit word
        int word_start_byte = (m - 63) >> 3;
        if (((m - 63) & 7) == 0) {  // Byte-aligned
            uint64_t w1, w2;
            memcpy(&w1, h1_ptr + word_start_byte, 8);
            memcpy(&w2, h2_ptr + word_start_byte, 8);
            uint64_t xor_word = w1 ^ w2;
            if (xor_word == 0) {
                m -= 64;  // Skip 64 matching sites
                continue;
            }
            // Has mismatches - find highest set bit and process from there
            int highest_mismatch = 63 - __builtin_clzll(xor_word);
            m = (m - 63) + highest_mismatch;  // Jump to the mismatch position
        }
        break;  // Not aligned or found mismatch, switch to bit-by-bit
    }

    // Bit-by-bit loop for remaining sites or after finding mismatch
    while (m > 0) {
        --m;
        const int byte_idx = m >> 3;
        const int bit_idx = m & 7;
        // XOR the bytes, then extract the bit
        if (__builtin_expect((h1_ptr[byte_idx] ^ h2_ptr[byte_idx]) >> bit_idx & 1, 0)) {
            // Mismatch found (rare case)
            if ((firstMismatchPos - bp_positions[m]) > maxGap) {
                ++m;
                break;
            } else if (m > 0) {
                firstMatch = m - 1;
            }
        }
    }

    const double len = genPos[firstMatch] - genPos[m];
    if (len >= minSeedMatch && (firstMatch - m) >= (minMarkers - 1)) {
        return -1;  // Skip seed - preceding seed exists
    }
    return (len < minExtend || (firstMatch - m) < mM1) ? start : m;
}

/**
 * Extend a seed forwards to find the true IBD segment end.
 *
 * Scans forward from the seed end, allowing gaps up to maxGap bp.
 * Uses the same optimizations as nextStartHapMajor (64-bit word scanning,
 * XOR-based comparison, prefetching).
 *
 * @return New end position, extended as far as allowed by gap constraints
 */
inline int nextEndHapMajor(int hap1, int hap2, int end,
              const std::vector<double>& genPos,
              const HapMetadata& meta,
              const HapMajorData& hap_major,
              double minSeedMatch,
              double maxGap,
              int minMarkers,
              double minExtend,
              int LastMarker,
              int mM1)
{
    if (__builtin_expect(end > (LastMarker - 2) || maxGap < 0, 0)) {
        return end;
    }

    const int* __restrict bp_positions = meta.vcf_bp_positions.data();
    const int bytes_per_hap = hap_major.bytes_per_hap;

    // Get pointers to each haplotype's data
    const uint8_t* __restrict h1_ptr = hap_major.data.data() + (size_t)hap1 * bytes_per_hap;
    const uint8_t* __restrict h2_ptr = hap_major.data.data() + (size_t)hap2 * bytes_per_hap;

    int m = end + 1;
    int firstMismatchPos = bp_positions[m];
    int firstMatch = end + 2;

    // Prefetch the memory region we'll be scanning (forwards)
    if (m + 512 < LastMarker) {
        __builtin_prefetch(h1_ptr + ((m + 512) >> 3), 0, 3);
        __builtin_prefetch(h2_ptr + ((m + 512) >> 3), 0, 3);
    }

#ifdef USE_AVX512
    // AVX-512: 512-bit scanning (512 sites at a time) - 8x faster than 64-bit
    while (m + 512 <= LastMarker) {
        if ((m & 7) == 0) {  // m is byte-aligned
            int word_start_byte = m >> 3;
            __m512i v1 = _mm512_loadu_si512((const __m512i*)(h1_ptr + word_start_byte));
            __m512i v2 = _mm512_loadu_si512((const __m512i*)(h2_ptr + word_start_byte));
            __m512i xor_result = _mm512_xor_si512(v1, v2);

            if (_mm512_test_epi64_mask(xor_result, xor_result) == 0) {
                m += 512;  // All 512 bits match, skip ahead
                continue;
            }

            // Has mismatches - find the lowest set bit position
            alignas(64) uint64_t words[8];
            _mm512_store_si512((__m512i*)words, xor_result);

            for (int i = 0; i < 8; ++i) {
                if (words[i] != 0) {
                    int lowest_mismatch = __builtin_ctzll(words[i]);
                    m = m + i * 64 + lowest_mismatch;
                    break;
                }
            }
        }
        break;
    }
#endif

#ifdef USE_AVX2
    // AVX2: 256-bit scanning (256 sites at a time) - 4x faster than 64-bit
    while (m + 256 <= LastMarker) {
        // Check alignment: we want bits [m, m+255] to be in 32 consecutive bytes
        if ((m & 7) == 0) {  // m is byte-aligned
            int word_start_byte = m >> 3;
            __m256i v1 = _mm256_loadu_si256((const __m256i*)(h1_ptr + word_start_byte));
            __m256i v2 = _mm256_loadu_si256((const __m256i*)(h2_ptr + word_start_byte));
            __m256i xor_result = _mm256_xor_si256(v1, v2);

            if (_mm256_testz_si256(xor_result, xor_result)) {
                m += 256;  // All 256 bits match, skip ahead
                continue;
            }

            // Has mismatches - find the lowest set bit position
            // Extract as 4 x 64-bit words (words[0] contains lowest sites [m, m+63])
            alignas(32) uint64_t words[4];
            _mm256_store_si256((__m256i*)words, xor_result);

            // Check from lowest to highest word
            for (int i = 0; i < 4; ++i) {
                if (words[i] != 0) {
                    int lowest_mismatch = __builtin_ctzll(words[i]);
                    m = m + i * 64 + lowest_mismatch;
                    break;
                }
            }
        }
        break;  // Not aligned or found mismatch, switch to finer scanning
    }
#endif

    // 64-bit word scanning: skip 64 matching sites at once when aligned
    // Fallback for non-AVX2 or when fewer than 256 sites remain
    while (m + 64 <= LastMarker) {
        // Check alignment: we want bits [m, m+63] to be in one 64-bit word
        if ((m & 7) == 0) {  // m is byte-aligned
            int word_start_byte = m >> 3;
            uint64_t w1, w2;
            memcpy(&w1, h1_ptr + word_start_byte, 8);
            memcpy(&w2, h2_ptr + word_start_byte, 8);
            uint64_t xor_word = w1 ^ w2;
            if (xor_word == 0) {
                m += 64;  // Skip 64 matching sites
                continue;
            }
            // Has mismatches - find lowest set bit and jump there
            int lowest_mismatch = __builtin_ctzll(xor_word);
            m = m + lowest_mismatch;  // Jump to the mismatch position
        }
        break;  // Not aligned or found mismatch, switch to bit-by-bit
    }

    // Bit-by-bit loop for remaining sites or after finding mismatch
    while (m < LastMarker) {
        ++m;
        const int byte_idx = m >> 3;
        const int bit_idx = m & 7;
        // XOR the bytes, then extract the bit
        if (__builtin_expect((h1_ptr[byte_idx] ^ h2_ptr[byte_idx]) >> bit_idx & 1, 0)) {
            // Mismatch found (rare case)
            if ((bp_positions[m] - firstMismatchPos) > maxGap) {
                --m;
                break;
            } else if (m < LastMarker) {
                firstMatch = m + 1;
            }
        }
    }

    const double len = genPos[m] - genPos[firstMatch];
    return (len < minExtend || (m - firstMatch) < mM1) ? end : m;
}

// ORIGINAL: Site-major version (kept for fallback/comparison)
// PHASE 2 OPTIMIZATION: inline to reduce function call overhead (millions of calls)
inline int nextStart( int hap1, int hap2, int start,
              const std::vector<double>& genPos,
              const HapMetadata& meta,
              const HapData& hap_data,
              double minSeedMatch,
              double maxGap,
              double minMarkers,
              double minExtend)
{
    if (start<2 || maxGap < 0) {
            return start;
    }

    // Precompute byte offsets and bit shifts (constant for this hap pair)
    const int byte_off1 = hap1 >> 3;
    const int bit_shift1 = hap1 & 7;
    const int byte_off2 = hap2 >> 3;
    const int bit_shift2 = hap2 & 7;

    // Precompute pointers for faster access
    const uint8_t* hap_ptr = hap_data.data();
    const int* bp_positions = meta.vcf_bp_positions.data();
    const int dense_stride = meta.dense_stride;  // All sites are dense (minMac filtered)

    int mM1 = std::floor((minExtend/minSeedMatch) * minMarkers) - 1;

    int m = start - 1;
    int firstMismatchPos = bp_positions[m];
    int firstMatch = start-2;

    // Simple optimized loop - direct stride calculation (faster than site_offsets lookup)
    // NOTE: All sites are dense because minMac filtering removes monomorphic sites
    while (m > 0) {
        --m;
        const uint8_t* site_ptr = hap_ptr + (size_t)m * dense_stride;
        uint8_t val1 = (site_ptr[byte_off1] >> bit_shift1) & 1;
        uint8_t val2 = (site_ptr[byte_off2] >> bit_shift2) & 1;
        if (val1 != val2) {
            if ((firstMismatchPos - bp_positions[m]) > maxGap) {
                ++m;
                break;
            } else if (m > 0) {
                firstMatch = m - 1;
            }
        }
    }
    double len = (genPos[firstMatch] - genPos[m]);
    if (len>=minSeedMatch && (firstMatch - m)>=(minMarkers-1)) {
        // skip seed since preceding seed exists for extended segment
        return -1;
    }
    else {
        return (len<minExtend || (firstMatch-m)<(mM1)) ? start : m;
    }
}

// PHASE 2 OPTIMIZATION: inline to reduce function call overhead (285M calls!)
inline int nextEnd( int hap1, int hap2, int end,
              const std::vector<double>& genPos,
              const HapMetadata& meta,
              const HapData& hap_data,
              double minSeedMatch,
              double maxGap,
              double minMarkers,
              double minExtend,
              int LastMarker)
{
    if (end >(LastMarker-2) || maxGap<0) {
            return end;
    }

    // Precompute byte offsets and bit shifts (constant for this hap pair)
    const int byte_off1 = hap1 >> 3;
    const int bit_shift1 = hap1 & 7;
    const int byte_off2 = hap2 >> 3;
    const int bit_shift2 = hap2 & 7;

    // Precompute pointers for faster access
    const uint8_t* hap_ptr = hap_data.data();
    const int* bp_positions = meta.vcf_bp_positions.data();
    const int dense_stride = meta.dense_stride;  // All sites are dense (minMac filtered)

    int mM1 = std::floor((minExtend/minSeedMatch) * minMarkers) - 1;
    int m = end + 1;
    int firstMismatchPos = bp_positions[m];
    int firstMatch = end + 2;

    // Simple optimized loop - direct stride calculation (faster than site_offsets lookup)
    // NOTE: All sites are dense because minMac filtering removes monomorphic sites
    while (m < LastMarker) {
        ++m;
        const uint8_t* site_ptr = hap_ptr + (size_t)m * dense_stride;
        uint8_t val1 = (site_ptr[byte_off1] >> bit_shift1) & 1;
        uint8_t val2 = (site_ptr[byte_off2] >> bit_shift2) & 1;
        if (val1 != val2) {
            if ((bp_positions[m] - firstMismatchPos) > maxGap) {
                --m;
                break;
            } else if (m < LastMarker) {
                firstMatch = m + 1;
            }
        }
    }
    double len = (genPos[m] - genPos[firstMatch]);
    return (len<minExtend || (m-firstMatch)<(mM1)) ? end : m;
}

// Window structure for parallel PBWT
struct Window {
    int start_site;
    int end_site;
};

// Create overlapping windows for parallel processing
std::vector<Window> createOverlappingWindows(
    const std::vector<double>& genPos,
    double min_seed_cm,
    int min_markers,
    int n_threads)
{
    int n_sites = genPos.size();
    if (n_sites == 0 || n_threads <= 0) return {};

    std::vector<Window> windows;

    // Calculate step size in cM
    double total_length_cm = genPos[n_sites - 1] - genPos[0];
    double step_cm = std::max((total_length_cm - min_seed_cm) / n_threads, 1e-6);

    int start = 0;
    double target_end_cm = genPos[start] + min_seed_cm + step_cm;
    int end = start;
    while (end < n_sites && genPos[end] < target_end_cm) end++;
    while (end < n_sites && end < min_markers) end++;

    // Create windows with minSeed overlap
    // Note: actual thread count may be 1-2 less than requested due to chromosome length
    int index = 0;
    while (index < n_threads - 1 && end < n_sites - 1) {
        // Calculate next window's start
        double overlap_cm = genPos[end] - min_seed_cm;
        int next_start = start;
        while (next_start < end && genPos[next_start] < overlap_cm) next_start++;

        // If exact match, increment by 1
        if (next_start < end && std::abs(genPos[next_start] - overlap_cm) < 1e-9) {
            next_start++;
        }

        // Ensure minimum overlap
        while (next_start > 0 &&
               ((end - (next_start - 1) + 1) < min_markers ||
                genPos[end] - genPos[next_start - 1] < min_seed_cm)) {
            next_start--;
        }

        // Account for warm-up when checking for gaps
        double next_warmup_target = genPos[next_start] + min_seed_cm;
        int next_collection_start = next_start;
        while (next_collection_start < n_sites && genPos[next_collection_start] < next_warmup_target) {
            next_collection_start++;
        }
        int next_min_markers_end = std::min(next_start + min_markers - 1, n_sites - 1);
        next_collection_start = std::max(next_collection_start, next_min_markers_end);

        // If there would be a gap between current window's end and next window's collection start,
        // extend current window to cover the gap
        if (next_collection_start > end + 1) {
            end = next_collection_start - 1;
        }

        windows.push_back({start, end});
        index++;

        start = next_start;

        // Next end: current end + step
        target_end_cm = genPos[end] + step_cm;
        while (end < n_sites && genPos[end] < target_end_cm) end++;
    }

    // Last window
    windows.push_back({start, n_sites - 1});

    return windows;
}

// Process seeds from queue (extend + write to file)
// Uses haplotype-major layout for cache-friendly extension
void processSeedBatch(
    const std::vector<Seed>& seeds,
    const std::vector<double>& genPos,
    const HapMetadata& meta,
    const HapMajorData& hap_major,
    double MIN_SEED_CM,
    double MIN_OUTPUT_CM,
    double MAX_GAP,
    int MIN_MARKERS,
    double MIN_EXTEND,
    int LastMarker,
    OutputBuffer& output_buffer)
{
    // Precompute mM1 once for all seeds in batch (avoids repeated floor/division)
    const int mM1 = static_cast<int>(std::floor((MIN_EXTEND / MIN_SEED_CM) * MIN_MARKERS)) - 1;

    for (const Seed& seed : seeds) {
        int hap1 = seed.hap1;
        int hap2 = seed.hap2;
        int ext_start = seed.start_site;
        int ext_end = seed.end_site;

        // Extend backwards using fast 64-bit scan first, then gap-tolerant extension
        int prevStart = hap_major.extendMatchBackward64(hap1, hap2, ext_start);
        int nStart = nextStartHapMajor(hap1, hap2, prevStart, genPos, meta, hap_major,
                              MIN_SEED_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, mM1);

        while (nStart >= 0 && nStart < prevStart) {
            prevStart = nStart;
            nStart = nextStartHapMajor(hap1, hap2, prevStart, genPos, meta, hap_major,
                              MIN_SEED_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, mM1);
        }

        if (nStart >= 0) {
            // Extend forwards using fast 64-bit word scanning (replaces bit-by-bit loop)
            int inclEnd = hap_major.extendMatchForward64(hap1, hap2, ext_end, LastMarker);
            int prevInclEnd = inclEnd;
            int nextInclEnd = nextEndHapMajor(hap1, hap2, prevInclEnd, genPos, meta, hap_major,
                                      MIN_SEED_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, LastMarker, mM1);
            while (nextInclEnd > prevInclEnd) {
                prevInclEnd = nextInclEnd;
                nextInclEnd = nextEndHapMajor(hap1, hap2, prevInclEnd, genPos, meta, hap_major,
                                      MIN_SEED_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, LastMarker, mM1);
            }

            // Write segment to buffer (use MIN_OUTPUT_CM for final filtering)
            double length_cm = genPos[nextInclEnd] - genPos[nStart];
            if (length_cm >= MIN_OUTPUT_CM && (hap1 >> 1) != (hap2 >> 1)) {
                BinarySegment seg;
                seg.hap1 = hap1;
                seg.hap2 = hap2;
                seg.start_idx = nStart;
                seg.end_idx = nextInclEnd;
                seg.start_bp = meta.vcf_bp_positions[nStart];
                seg.end_bp = meta.vcf_bp_positions[nextInclEnd];
                seg.length_cm = static_cast<float>(length_cm);
                output_buffer.writeBinary(seg);
            }
        }
    }
}

// Timing counters (atomic for thread-safe accumulation)
struct TimingStats {
    std::atomic<uint64_t> pbwt_ns{0};       // PBWT step time
    std::atomic<uint64_t> collect_ns{0};    // Seed collection time
    std::atomic<uint64_t> extend_ns{0};     // Seed extension time
    std::atomic<uint64_t> dedup_ns{0};      // Deduplication time
    std::atomic<uint64_t> seeds_collected{0};
    std::atomic<uint64_t> seeds_extended{0};
};

// Worker function for each window thread
void runWindowThread(
    int window_id,
    int start_site,
    int end_site,
    const HapMetadata& meta,
    const HapData& hap_data,
    const HapMajorData& hap_major,  // Haplotype-major for cache-friendly extension
    const std::vector<double>& genPos,
    double MIN_SEED_CM,
    double MIN_OUTPUT_CM,
    double MAX_GAP,
    int MIN_MARKERS,
    double MIN_EXTEND,
    int LastMarker,
    SeedQueue& seedQ,
    std::atomic<int>& finished_count,
    int n_windows,
    FILE* bin_file,
    FILE* tsv_file,
    std::mutex& file_mtx,
    std::atomic<size_t>& segment_count,
    StripedDedup& dedup,
    bool isLastWindow,
    TimingStats& timing)
{
    const size_t SEED_LIST_THRESHOLD = 65536;
    int n_haps = meta.n_haps;

    // Create per-thread output buffer (writes both binary and TSV)
    OutputBuffer output_buffer(bin_file, tsv_file, file_mtx, segment_count, dedup);

    // Initialize PBWT state for this window
    PBWTState st(n_haps);

    // Initialize divergence array to windowStart for correct boundary filtering
    for (int i = 0; i < n_haps; ++i) {
        st.D[i] = start_site;
    }

    std::vector<uint8_t> site_buffer(n_haps);
    std::vector<Seed> seedList;
    seedList.reserve(SEED_LIST_THRESHOLD);

    // Warm-up: advance PBWT without collecting seeds until MIN_SEED_CM distance
    double target_cm = genPos[start_site] + MIN_SEED_CM;
    int collection_start = start_site;

    while (collection_start <= end_site && genPos[collection_start] < target_cm) {
        collection_start++;
    }

    // Ensure minimum marker distance
    int min_warm_up_end = std::min(start_site + MIN_MARKERS - 1, end_site);
    collection_start = std::max(collection_start, min_warm_up_end);

    // Warm-up: advance PBWT from start_site to collection_start WITHOUT collecting
    for (int m = start_site; m < collection_start && m <= end_site; ++m) {
        get_site(m, meta, hap_data, site_buffer);
        pbwt_step(m, site_buffer, st);
    }

    bool use_seed_queue = false;

    // Run PBWT for this window
    int flush_count = 0;
    size_t total_seeds = 0;
    int maxIbsStart = start_site;

    // Thread-local timing accumulators
    uint64_t local_pbwt_ns = 0;
    uint64_t local_collect_ns = 0;
    uint64_t local_extend_ns = 0;
    size_t local_seeds_collected = 0;
    size_t local_seeds_extended = 0;

    for (int site_idx = collection_start; site_idx <= end_site; ++site_idx) {
        // Check if we should switch to queue mode
        if (!use_seed_queue && finished_count.load(std::memory_order_acquire) > 0) {
            use_seed_queue = true;
        }

        size_t seeds_before = seedList.size();

        // Update maxIbsStart to ensure minimum seed length
        while (maxIbsStart + 1 < site_idx &&
               genPos[site_idx] - genPos[maxIbsStart + 1] >= MIN_SEED_CM &&
               site_idx - maxIbsStart >= MIN_MARKERS) {
            maxIbsStart++;
        }

        // Time seed collection (includes get_site)
        auto t1 = std::chrono::high_resolution_clock::now();
        get_site(site_idx, meta, hap_data, site_buffer);
        collectSeeds(st, site_idx, seedList, meta, hap_data, hap_major, genPos,
                    MIN_SEED_CM, MIN_MARKERS, maxIbsStart, start_site, collection_start, end_site,
                    isLastWindow);
        auto t2 = std::chrono::high_resolution_clock::now();
        local_collect_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        size_t seeds_generated = seedList.size() - seeds_before;
        total_seeds += seeds_generated;
        local_seeds_collected += seeds_generated;

        // Time PBWT step
        auto t3 = std::chrono::high_resolution_clock::now();
        pbwt_step(site_idx, site_buffer, st);
        auto t4 = std::chrono::high_resolution_clock::now();
        local_pbwt_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();

        // Flush seeds to queue when threshold reached
        if (seedList.size() > SEED_LIST_THRESHOLD) {
            std::vector<Seed> to_flush = std::move(seedList);
            seedList.clear();
            seedList.reserve(SEED_LIST_THRESHOLD);

            size_t batch_size = to_flush.size();

            // If not using queue or queue full, process inline
            // Note: offer() only moves data if it succeeds; on failure, to_flush is still valid
            if (!use_seed_queue || !seedQ.offer(std::move(to_flush))) {
                // Not using queue OR queue full - process inline
                // (If offer failed, to_flush still contains data since move only happens on success)
                auto te1 = std::chrono::high_resolution_clock::now();
                processSeedBatch(to_flush, genPos, meta, hap_major,
                               MIN_SEED_CM, MIN_OUTPUT_CM,
                               MAX_GAP, MIN_MARKERS, MIN_EXTEND, LastMarker,
                               output_buffer);
                auto te2 = std::chrono::high_resolution_clock::now();
                local_extend_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(te2 - te1).count();
                local_seeds_extended += batch_size;
            }
            // else: successfully queued, helper thread will process
            flush_count++;
        }
    }

    // Process remaining seeds
    if (!seedList.empty()) {
        auto te1 = std::chrono::high_resolution_clock::now();
        processSeedBatch(seedList, genPos, meta, hap_major,
                        MIN_SEED_CM, MIN_OUTPUT_CM,
                        MAX_GAP, MIN_MARKERS, MIN_EXTEND, LastMarker,
                        output_buffer);
        auto te2 = std::chrono::high_resolution_clock::now();
        local_extend_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(te2 - te1).count();
        local_seeds_extended += seedList.size();
    }

    // Accumulate to global timing stats
    timing.pbwt_ns += local_pbwt_ns;
    timing.collect_ns += local_collect_ns;
    timing.extend_ns += local_extend_ns;
    timing.seeds_collected += local_seeds_collected;
    timing.seeds_extended += local_seeds_extended;

    // Mark this thread as finished
    finished_count++;

    // Help consume remaining seeds from queue
    std::vector<Seed> batch;
    while (finished_count.load() < n_windows || !seedQ.empty()) {
        if (seedQ.poll(batch)) {
            processSeedBatch(batch, genPos, meta, hap_major,
                           MIN_SEED_CM, MIN_OUTPUT_CM,
                           MAX_GAP, MIN_MARKERS, MIN_EXTEND, LastMarker,
                           output_buffer);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "IBD-Booster: Fast IBD segment detection with P-smoother integration\n\n";
        std::cerr << "Usage: " << argv[0] << " <input.vcf|input.vcf.gz> <genetic_map.map> [options]\n\n";
        std::cerr << "Options:\n";
        std::cerr << "  --threads=N       Number of threads (default: 1)\n";
        std::cerr << "  --min-output=F    Minimum cM length for IBD output segments (default: 2.0)\n";
        std::cerr << "  --no-psmoother    Skip P-smoother (for pre-smoothed files)\n";
        std::cerr << "  --ps-length=N     P-smoother block length (default: 20)\n";
        std::cerr << "  --ps-width=N      P-smoother minimum block width (default: 20)\n";
        std::cerr << "  --ps-gap=N        P-smoother gap size (default: 1)\n";
        std::cerr << "  --ps-rho=F        P-smoother error rate threshold (default: 0.05)\n";
        std::cerr << "\nOutput format (tab-separated):\n";
        std::cerr << "  sample1_idx  hap1  sample2_idx  hap2  start_bp  end_bp  length_cM\n";
        return 1;
    }

    const std::string vcf_path = argv[1];
    const std::string map_path = argv[2];

    // Parse command-line options
    int nthreads = 1;
    double min_output = 2.0;  // Minimum cM length for IBD output segments
    bool skip_psmoother = false;
    PSmootherParams ps_params;  // P-smoother parameters (defaults: L=20, W=20, G=1, rho=0.05)

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--threads=") == 0) {
            nthreads = std::stoi(arg.substr(10));
        } else if (arg.find("--min-output=") == 0) {
            min_output = std::stod(arg.substr(13));
        } else if (arg == "--no-psmoother") {
            skip_psmoother = true;
        } else if (arg.find("--ps-length=") == 0) {
            ps_params.length = std::stoi(arg.substr(12));
        } else if (arg.find("--ps-width=") == 0) {
            ps_params.width = std::stoi(arg.substr(11));
        } else if (arg.find("--ps-gap=") == 0) {
            ps_params.gap = std::stoi(arg.substr(9));
        } else if (arg.find("--ps-rho=") == 0) {
            ps_params.rho = std::stod(arg.substr(9));
        }
    }
    // Pass thread count to P-smoother for parallel processing
    ps_params.nthreads = nthreads;
    ps_params.verbose = false;  // Disable detailed P-smoother output

    // Thread configuration for augmentation phase
    int extract_threads = nthreads;
    int xgb_threads = nthreads;

    std::cerr << "[INFO] Using " << nthreads << " thread(s)\n";
    std::cerr << "[INFO] min-output=" << min_output << " cM\n";
    if (skip_psmoother) {
        std::cerr << "[INFO] P-smoother disabled (--no-psmoother)\n";
    } else {
        std::cerr << "[INFO] P-smoother (L=" << ps_params.length
                  << " W=" << ps_params.width << " G=" << ps_params.gap
                  << " rho=" << ps_params.rho << ")\n";
    }
    if (!is_vcf(vcf_path)) {
        std::cerr << "Error: only .vcf or .vcf.gz are supported.\n";
        return 1;
    }

    auto start_time = std::chrono::steady_clock::now();
    HapMetadata meta;
    HapData hap_data;          // flat memory storage for all sites
    int minMac = 2;

    // Flow: VCF → 2D → P-smoother → pack (with minMac filter) → IBD detection
    std::vector<std::vector<uint8_t>> hap_2d;

    auto vcf_start = std::chrono::steady_clock::now();
    try {
        meta = read_vcf_to_2d(vcf_path, hap_2d);
    } catch (const std::exception& e) {
        std::cerr << "ERROR during VCF reading: " << e.what() << "\n";
        return 1;
    }
    auto vcf_end = std::chrono::steady_clock::now();

    // Correction locations from P-smoother (persists for feature extraction)
    std::vector<std::pair<int, int>> correction_locations;

    // Run P-smoother for haplotype error correction (unless disabled)
    if (!skip_psmoother) {
        auto ps_start = std::chrono::steady_clock::now();
        PSmoother smoother(meta.n_haps, meta.n_sites, ps_params);
        int corrections = smoother.smooth(hap_2d);
        std::cerr << "[P-smoother] Correction locations tracked: "
                  << smoother.getCorrections().size() << std::endl;
        // Copy corrections before smoother goes out of scope
        correction_locations = smoother.getCorrections();
        auto ps_end = std::chrono::steady_clock::now();
        double ps_time = std::chrono::duration<double>(ps_end - ps_start).count();

        // Check if file appears already smoothed (correction rate < 0.1%)
        // P-smoother is not idempotent - running it on already-smoothed data produces spurious corrections
        // due to PBWT block boundaries shifting after the first smoothing pass
        double total_alleles = (double)meta.n_haps * meta.n_sites;
        double correction_rate = (double)corrections / total_alleles;
        const double ALREADY_SMOOTHED_THRESHOLD = 0.001;  // 0.1%

        if (corrections > 0 && correction_rate < ALREADY_SMOOTHED_THRESHOLD) {
            // Reload original data - file was already smoothed
            std::cerr << "[INFO] File appears already smoothed (correction rate " << (correction_rate * 100) << "% < 0.1%), reloading original\n";
            hap_2d.clear();
            meta = read_vcf_to_2d(vcf_path, hap_2d);
            corrections = 0;
        }

        // Write smoothed VCF only if meaningful corrections were made (async to overlap with IBD detection)
        if (corrections > 0) {
            // Generate output filename in current directory: /path/to/input.vcf.gz -> input_smooth.vcf.gz
            std::string smoothed_vcf = vcf_path;
            // Extract just the filename (remove directory path)
            size_t last_slash = smoothed_vcf.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                smoothed_vcf = smoothed_vcf.substr(last_slash + 1);
            }
            // Insert _smooth before .vcf extension
            size_t ext_pos = smoothed_vcf.rfind(".vcf");
            if (ext_pos != std::string::npos) {
                smoothed_vcf.insert(ext_pos, "_smooth");
            } else {
                smoothed_vcf += "_smooth.vcf.gz";
            }

            // Write smoothed VCF synchronously (before freeing hap_2d)
            write_smoothed_vcf(vcf_path, smoothed_vcf, hap_2d, meta, minMac);
        }
    }

    // Recompute MAC (needed for minMac filter)
    recompute_mac(hap_2d, meta, nthreads);

    // Pack to bit-packed format (applies minMac filter)
    pack_haplotypes(hap_2d, hap_data, meta, minMac, nthreads);

    // Free 2D data (no longer needed after packing)
    hap_2d.clear();
    hap_2d.shrink_to_fit();

    const int32_t n_samples = meta.n_samples;
    const int32_t n_sites   = meta.n_sites;

    // Create haplotype-major layout for cache-friendly extension
    HapMajorData hap_major;
    transposeToHapMajor(hap_major, meta, hap_data, nthreads);

    // Read genetic map and interpolate positions
    GeneticMap gmap = readGeneticMap(map_path);
    std::vector<int> vcf_bp = meta.vcf_bp_positions;
    std::vector<double> genPos = interpolateGeneticPositions(gmap, vcf_bp);

    // Parameters
    double MIN_SEED_CM = 2.0;    // Minimum cM for seed collection (fixed, matches HapIBD default)
    double MIN_OUTPUT_CM = min_output;  // Minimum cM for output filtering (user-configurable)
    double MAX_GAP = 1000;
    double MIN_EXTEND = std::min(1.0, MIN_SEED_CM);
    int MIN_MARKERS = 100;
    int LastMarker = n_sites - 1;

    // Open output files (both binary for ML pipeline and TSV for human inspection)
    FILE* bin_file = fopen("subset_5k_Segments.bin", "wb");
    FILE* tsv_file = fopen("subset_5k_Segments.tsv", "w");
    if (!bin_file || !tsv_file) {
        std::cerr << "[ERROR] Cannot open output files for writing\n";
        return 1;
    }

    // Shared resources for parallel execution
    const size_t QUEUE_CAPACITY = 16;
    SeedQueue seedQ(QUEUE_CAPACITY);
    std::atomic<int> finished_count(0);
    std::atomic<size_t> segment_count(0);
    std::mutex file_mtx;

    auto pbwt_start = std::chrono::steady_clock::now();

    if (nthreads == 1) {
        // Single-threaded path with batch processing
        const size_t SEED_LIST_THRESHOLD = 65536;

        int n_haps = n_samples * 2;
        PBWTState st(n_haps);
        std::vector<uint8_t> site_buffer(n_haps);
        std::vector<Seed> seeds;
        seeds.reserve(SEED_LIST_THRESHOLD);

        // Output buffers for single-threaded (both binary and TSV)
        std::vector<BinarySegment> seg_buffer;
        std::string tsv_buffer;
        seg_buffer.reserve(8192);
        tsv_buffer.reserve(8192 * 50);
        auto flushSegBuffer = [&]() {
            if (!seg_buffer.empty()) {
                fwrite(seg_buffer.data(), sizeof(BinarySegment), seg_buffer.size(), bin_file);
                fwrite(tsv_buffer.data(), 1, tsv_buffer.size(), tsv_file);
                seg_buffer.clear();
                tsv_buffer.clear();
            }
        };

        // Precompute mM1 once (avoids repeated floor/division in hot loop)
        const int mM1 = static_cast<int>(std::floor((MIN_EXTEND / MIN_SEED_CM) * MIN_MARKERS)) - 1;

        // Lambda to process and extend a batch of seeds using haplotype-major layout
        auto processBatch = [&]() {
            for (const auto &seed : seeds) {
                int hap1 = seed.hap1;
                int hap2 = seed.hap2;
                int ext_start = seed.start_site;
                int ext_end = seed.end_site;

                // Extend backwards using fast 64-bit scan first, then gap-tolerant extension
                int prevStart = hap_major.extendMatchBackward64(hap1, hap2, ext_start);
                int nStart = nextStartHapMajor(hap1, hap2, prevStart, genPos, meta, hap_major,
                                       MIN_SEED_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, mM1);

                while (nStart >= 0 && nStart < prevStart) {
                    prevStart = nStart;
                    nStart = nextStartHapMajor(hap1, hap2, prevStart, genPos, meta, hap_major,
                                       MIN_SEED_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, mM1);
                }

                if (nStart >= 0) {
                    // Extend forwards using fast 64-bit word scanning (replaces bit-by-bit loop)
                    int inclEnd = hap_major.extendMatchForward64(hap1, hap2, ext_end, LastMarker);
                    int prevInclEnd = inclEnd;
                    int nextInclEnd = nextEndHapMajor(hap1, hap2, prevInclEnd, genPos, meta, hap_major,
                                              MIN_SEED_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, LastMarker, mM1);
                    while (nextInclEnd > prevInclEnd) {
                        prevInclEnd = nextInclEnd;
                        nextInclEnd = nextEndHapMajor(hap1, hap2, prevInclEnd, genPos, meta, hap_major,
                                              MIN_SEED_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, LastMarker, mM1);
                    }

                    // Write segment to buffer (use MIN_OUTPUT_CM for final filtering)
                    double length_cm = genPos[nextInclEnd] - genPos[nStart];
                    if (length_cm >= MIN_OUTPUT_CM && (hap1 >> 1) != (hap2 >> 1)) {
                        BinarySegment seg;
                        seg.hap1 = hap1;
                        seg.hap2 = hap2;
                        seg.start_idx = nStart;
                        seg.end_idx = nextInclEnd;
                        seg.start_bp = meta.vcf_bp_positions[nStart];
                        seg.end_bp = meta.vcf_bp_positions[nextInclEnd];
                        seg.length_cm = static_cast<float>(length_cm);
                        seg_buffer.push_back(seg);

                        // Also format TSV line
                        char line[128];
                        int len = snprintf(line, sizeof(line), "%d\t%d\t%d\t%d\t%d\t%d\t%.6f\n",
                                          hap1 / 2, hap1 & 1, hap2 / 2, hap2 & 1,
                                          seg.start_bp, seg.end_bp, length_cm);
                        tsv_buffer.append(line, len);

                        segment_count++;
                        if (seg_buffer.size() >= 8192) {
                            flushSegBuffer();
                        }
                    }
                }
            }
            seeds.clear();
        };

        // Simple single-threaded loop
        for (int32_t site_idx = 0; site_idx < n_sites; ++site_idx) {
            get_site(site_idx, meta, hap_data, site_buffer);
            collectSeedsSimple(st, site_idx, LastMarker, seeds, meta, hap_data, hap_major, genPos, MIN_SEED_CM, MIN_MARKERS);
            pbwt_step(site_idx, site_buffer, st);

            // Process batch when threshold reached (improves cache locality)
            if (seeds.size() >= SEED_LIST_THRESHOLD) {
                processBatch();
            }
        }

        // Process remaining seeds
        if (!seeds.empty()) {
            processBatch();
        }

        // Flush remaining segments
        flushSegBuffer();
        fclose(bin_file);
        fclose(tsv_file);

        auto end_time = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        std::cout << "\nStatistics\n";
        std::cout << "  samples          :  " << n_samples << "\n";
        std::cout << "  markers          :  " << n_sites << "\n";
        std::cout << "  IBD segments     :  " << segment_count.load() << "\n";
        std::cout << "  IBD segs/sample  :  "
                << (n_samples > 0 ? double(segment_count.load()) / n_samples : 0.0) << "\n";
        std::cout << "Wallclock Time:  " << total_elapsed << " s\n";
    } else {
        // Parallel path (nthreads > 1)
        // Create windows
        std::vector<Window> windows = createOverlappingWindows(genPos, MIN_SEED_CM, MIN_MARKERS, nthreads);

        // Striped deduplication for reduced mutex contention with multiple threads
        StripedDedup dedup;
        dedup.reserve(300000000);  // Reserve for ~300M segments total

        // Timing statistics
        TimingStats timing;

        // Use actual number of windows created (may be less than requested)
        int actual_threads = static_cast<int>(windows.size());

        // Launch worker threads for windows 0 to actual_threads-2
        // Main thread will handle the last window
        std::vector<std::thread> threads;
        for (int w = 0; w < actual_threads - 1; ++w) {
            bool isLastWindow = false;
            threads.emplace_back(runWindowThread,
                               w, windows[w].start_site, windows[w].end_site,
                               std::cref(meta), std::cref(hap_data), std::cref(hap_major),
                               std::cref(genPos),
                               MIN_SEED_CM, MIN_OUTPUT_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, LastMarker,
                               std::ref(seedQ), std::ref(finished_count), actual_threads,
                               bin_file, tsv_file, std::ref(file_mtx), std::ref(segment_count),
                               std::ref(dedup),
                               isLastWindow,
                               std::ref(timing));
            // Pin thread to core w to prevent migration and improve cache locality
            pinThreadToCore(threads.back(), w);
        }

        // Main thread handles the last window
        int last_w = actual_threads - 1;
        runWindowThread(
            last_w, windows[last_w].start_site, windows[last_w].end_site,
            std::cref(meta), std::cref(hap_data), std::cref(hap_major),
            std::cref(genPos),
            MIN_SEED_CM, MIN_OUTPUT_CM, MAX_GAP, MIN_MARKERS, MIN_EXTEND, LastMarker,
            std::ref(seedQ), std::ref(finished_count), actual_threads,
            bin_file, tsv_file, std::ref(file_mtx), std::ref(segment_count),
            std::ref(dedup),
            true,  // isLastWindow
            std::ref(timing));

        // Wait for worker threads to complete
        for (auto& t : threads) {
            t.join();
        }

        seedQ.markFinished();

        fclose(bin_file);
        fclose(tsv_file);

        auto end_time = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        std::cout << "\nStatistics\n";
        std::cout << "  samples          :  " << n_samples << "\n";
        std::cout << "  markers          :  " << n_sites << "\n";
        std::cout << "  IBD segments     :  " << segment_count.load() << "\n";
        std::cout << "  IBD segs/sample  :  "
                << (n_samples > 0 ? double(segment_count.load()) / n_samples : 0.0) << "\n";
        std::cout << "Wallclock Time:  " << total_elapsed << " s\n";
    }

    // ============================================================
    // Segment Augmentation Phase
    // ============================================================
    //
    // Uses machine learning to filter false-positive IBD segments.
    //
    // Feature extraction: For each segment, divide into 10 chunks and extract:
    //   - Physical length (bp)
    //   - Genetic length (cM)
    //   - Number of mismatches (allele differences between haplotypes)
    //   - Number of P-smoother corrections
    // Total: 40 features per segment (10 chunks × 4 features)
    //
    // Two-pass global normalization (StandardScaler):
    //   Pass 1: Compute global mean and std across ALL segments
    //   Pass 2: Normalize each segment using global stats, then predict
    //
    // This ensures consistent normalization regardless of batch size.
    // ============================================================

    std::cout << "[Augmentation] Starting segment augmentation...\n";
    auto augment_start = std::chrono::steady_clock::now();

    // Create feature extractor using data still in memory from detection phase
    // NOTE: hap_major layout enables fast XOR+popcount mismatch counting
    FeatureExtractorParams fe_params;
    fe_params.n_chunks = 10;   // 10 chunks per segment
    fe_params.verbose = false;
    FeatureExtractor feature_extractor(meta, hap_major, genPos, correction_locations, fe_params);

    const size_t BATCH_SIZE = 100000;  // Process segments in batches to limit memory
    const int n_features = 40;         // 10 chunks × 4 features per chunk

    // Open binary segments file for reading
    FILE* seg_file = fopen("subset_5k_Segments.bin", "rb");
    if (!seg_file) {
        std::cerr << "[ERROR] Cannot open subset_5k_Segments.bin for reading\n";
        return 1;
    }

    // Get total segment count from file size
    fseek(seg_file, 0, SEEK_END);
    size_t file_size = ftell(seg_file);
    size_t total_segments = file_size / sizeof(BinarySegment);
    fseek(seg_file, 0, SEEK_SET);

    // Buffers (reused for both passes)
    std::vector<BinarySegment> read_buffer(BATCH_SIZE);
    std::vector<IBDSegment> segment_batch;
    segment_batch.reserve(BATCH_SIZE);
    std::vector<float> batch_features(BATCH_SIZE * n_features);

    // ============================================================
    // Pass 1: Compute global statistics (mean and std for each feature)
    // ============================================================
    std::cout << "[Augmentation] Computing global statistics...\n";
    auto stats_start = std::chrono::steady_clock::now();

    // Use double for accumulation to avoid floating-point precision loss
    std::vector<double> global_sum(n_features, 0.0);
    std::vector<double> global_sum_sq(n_features, 0.0);
    size_t total_count = 0;

    size_t segments_read = 0;
    while (segments_read < total_segments) {  // Process ALL segments for global statistics
        size_t to_read = std::min(BATCH_SIZE, total_segments - segments_read);
        size_t n_read = fread(read_buffer.data(), sizeof(BinarySegment), to_read, seg_file);
        if (n_read == 0) break;

        // Convert BinarySegment to IBDSegment
        segment_batch.clear();
        for (size_t i = 0; i < n_read; ++i) {
            const BinarySegment& bs = read_buffer[i];
            IBDSegment seg;
            seg.hap1 = bs.hap1;
            seg.hap2 = bs.hap2;
            seg.start_idx = bs.start_idx;
            seg.end_idx = bs.end_idx;
            seg.start_bp = bs.start_bp;
            seg.end_bp = bs.end_bp;
            seg.length_cm = bs.length_cm;
            segment_batch.push_back(seg);
        }

        // Extract features
        feature_extractor.extractBatchFlat(segment_batch, batch_features.data(), extract_threads);

        // Accumulate sum and sum² for each feature
        for (size_t i = 0; i < n_read; ++i) {
            for (int f = 0; f < n_features; ++f) {
                double val = batch_features[i * n_features + f];
                global_sum[f] += val;
                global_sum_sq[f] += val * val;
            }
        }
        total_count += n_read;
        segments_read += n_read;

        // Progress
        double progress = static_cast<double>(segments_read) / total_segments;
        std::cout << "\r[Computing statistics] " << std::fixed << std::setprecision(1)
                  << (progress * 100.0) << "%" << std::flush;
    }
    std::cout << "\n";

    // Compute global mean and std: mean = sum/n, std = sqrt(sum²/n - mean²)
    std::vector<float> global_mean(n_features);
    std::vector<float> global_std(n_features);
    for (int f = 0; f < n_features; ++f) {
        global_mean[f] = static_cast<float>(global_sum[f] / total_count);
        double variance = (global_sum_sq[f] / total_count) - (global_mean[f] * global_mean[f]);
        global_std[f] = static_cast<float>(std::sqrt(std::max(0.0, variance)));
        if (global_std[f] < 1e-8f) global_std[f] = 1.0f;  // Avoid division by zero
    }

    auto stats_end = std::chrono::steady_clock::now();
    double stats_sec = std::chrono::duration<double>(stats_end - stats_start).count();

    // ============================================================
    // Pass 2: Normalize features and run XGBoost prediction
    // ============================================================
    std::cout << "[Augmentation] Running prediction...\n";
    auto predict_start = std::chrono::steady_clock::now();

    // Reset file position for second pass
    fseek(seg_file, 0, SEEK_SET);

    // Load XGBoost model
    std::string model_path = "../Reproducibility/models/xgb_ibd_augmentation.json";
    XGBPredictor predictor(model_path, xgb_threads);
    if (!predictor.isLoaded()) {
        std::cerr << "[ERROR] Failed to load XGBoost model\n";
        return 1;
    }

    // Load Neural Network model (custom 5-layer MLP, weights exported from PyTorch)
    // Uses the same 40 normalized features as XGBoost; outputs to a separate file
    std::string nn_model_path = "../Reproducibility/models/nn_weights.bin";
    NNPredictor nn_predictor(nn_model_path);
    if (!nn_predictor.isLoaded()) {
        std::cerr << "[ERROR] Failed to load NN weights from " << nn_model_path << "\n";
        return 1;
    }
    std::cout << "[Augmentation] NN model loaded from " << nn_model_path << "\n";

    // Open output files
    FILE* out_file = fopen("subset_5k_predicted_segments_xgb.txt", "w");
    if (!out_file) {
        std::cerr << "[ERROR] Cannot open subset_5k_predicted_segments_xgb.txt for writing\n";
        fclose(seg_file);
        return 1;
    }

    FILE* nn_out_file = fopen("subset_5k_predicted_segments_nn.txt", "w");
    if (!nn_out_file) {
        std::cerr << "[ERROR] Cannot open subset_5k_predicted_segments_nn.txt for writing\n";
        fclose(seg_file);
        fclose(out_file);
        return 1;
    }

    // Extract chromosome from first site's VCF fixed fields
    std::string chrom = "NA";
    if (!meta.vcf_fixed_fields.empty()) {
        const std::string& first_fixed = meta.vcf_fixed_fields[0];
        size_t tab_pos = first_fixed.find('\t');
        if (tab_pos != std::string::npos) {
            chrom = first_fixed.substr(0, tab_pos);
        }
    }

    // Timing accumulators
    uint64_t total_extract_ns = 0;
    uint64_t total_predict_ns = 0;
    uint64_t total_nn_predict_ns = 0;
    size_t positive_predictions = 0;
    size_t nn_positive_predictions = 0;

    segments_read = 0;
    while (segments_read < total_segments) {
        size_t to_read = std::min(BATCH_SIZE, total_segments - segments_read);
        size_t n_read = fread(read_buffer.data(), sizeof(BinarySegment), to_read, seg_file);
        if (n_read == 0) break;

        // Convert BinarySegment to IBDSegment
        segment_batch.clear();
        for (size_t i = 0; i < n_read; ++i) {
            const BinarySegment& bs = read_buffer[i];
            IBDSegment seg;
            seg.hap1 = bs.hap1;
            seg.hap2 = bs.hap2;
            seg.start_idx = bs.start_idx;
            seg.end_idx = bs.end_idx;
            seg.start_bp = bs.start_bp;
            seg.end_bp = bs.end_bp;
            seg.length_cm = bs.length_cm;
            segment_batch.push_back(seg);
        }
        segments_read += n_read;

        // Extract features
        auto ext_start = std::chrono::high_resolution_clock::now();
        feature_extractor.extractBatchFlat(segment_batch, batch_features.data(), extract_threads);
        auto ext_end = std::chrono::high_resolution_clock::now();
        total_extract_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(ext_end - ext_start).count();

        // Normalize with global mean/std
        for (size_t i = 0; i < n_read; ++i) {
            for (int f = 0; f < n_features; ++f) {
                batch_features[i * n_features + f] =
                    (batch_features[i * n_features + f] - global_mean[f]) / global_std[f];
            }
        }

        // Run XGBoost prediction
        auto pred_start = std::chrono::high_resolution_clock::now();
        std::vector<float> predictions = predictor.predictBatch(batch_features.data(), n_read, n_features);
        auto pred_end = std::chrono::high_resolution_clock::now();
        total_predict_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(pred_end - pred_start).count();

        // Run NN prediction on same normalized features
        auto nn_pred_start = std::chrono::high_resolution_clock::now();
        std::vector<float> nn_predictions = nn_predictor.predictBatch(batch_features.data(), n_read, n_features);
        auto nn_pred_end = std::chrono::high_resolution_clock::now();
        total_nn_predict_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(nn_pred_end - nn_pred_start).count();

        // Write kept segments to separate output files for each model.
        // Both models output P(false positive): < 0.5 → keep (true IBD), ≥ 0.5 → filter.
        // Format-once optimization: format the output line once per segment, then
        // conditionally append to each model's buffer. This avoids redundant snprintf
        // calls which dominated runtime when formatting was done independently.
        std::string out_buffer;
        out_buffer.reserve(predictions.size() * 80);
        std::string nn_out_buffer;
        nn_out_buffer.reserve(nn_predictions.size() * 80);
        char line[256];
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool xgb_keep = predictions[i] < 0.5f;
            bool nn_keep = nn_predictions[i] < 0.5f;
            if (xgb_keep || nn_keep) {
                const IBDSegment& seg = segment_batch[i];
                int sample1 = seg.hap1 / 2;
                int hap1 = (seg.hap1 % 2) + 1;
                int sample2 = seg.hap2 / 2;
                int hap2 = (seg.hap2 % 2) + 1;
                const std::string& id1 = meta.sampleIDs[sample1];
                const std::string& id2 = meta.sampleIDs[sample2];
                int len = snprintf(line, sizeof(line), "%s\t%d\t%s\t%d\t%s\t%d\t%d\t%.6f\n",
                        id1.c_str(), hap1, id2.c_str(), hap2,
                        chrom.c_str(), seg.start_bp, seg.end_bp, seg.length_cm);
                if (xgb_keep) {
                    positive_predictions++;
                    out_buffer.append(line, len);
                }
                if (nn_keep) {
                    nn_positive_predictions++;
                    nn_out_buffer.append(line, len);
                }
            }
        }
        fwrite(out_buffer.data(), 1, out_buffer.size(), out_file);
        fwrite(nn_out_buffer.data(), 1, nn_out_buffer.size(), nn_out_file);

        // Progress bar
        double progress = static_cast<double>(segments_read) / total_segments;
        int bar_width = 40;
        int filled = static_cast<int>(progress * bar_width);
        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < filled) std::cout << "=";
            else if (i == filled) std::cout << ">";
            else std::cout << " ";
        }
        double batch_ext_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ext_end - ext_start).count();
        double batch_pred_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pred_end - pred_start).count();
        std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% "
                  << "(ext:" << batch_ext_ms << "ms pred:" << batch_pred_ms << "ms)" << std::flush;
    }
    std::cout << "\n";

    fclose(seg_file);
    fclose(out_file);
    fclose(nn_out_file);

    auto augment_end = std::chrono::steady_clock::now();
    double wallclock_sec = std::chrono::duration<double>(augment_end - augment_start).count();
    double extract_sec = total_extract_ns / 1e9;
    double inference_sec = total_predict_ns / 1e9;
    double nn_inference_sec = total_nn_predict_ns / 1e9;

    size_t segments_filtered = segments_read - positive_predictions;
    double kept_pct = 100.0 * positive_predictions / segments_read;
    double filter_pct = 100.0 * segments_filtered / segments_read;

    size_t nn_segments_filtered = segments_read - nn_positive_predictions;
    double nn_kept_pct = 100.0 * nn_positive_predictions / segments_read;
    double nn_filter_pct = 100.0 * nn_segments_filtered / segments_read;

    std::cout << "\n[Segment Augmentation Results]\n";
    std::cout << "  segments processed:  " << segments_read << "\n";
    std::cout << "  --- XGBoost ---\n";
    std::cout << "  segments kept:       " << positive_predictions << " (" << kept_pct << "%)\n";
    std::cout << "  segments filtered:   " << segments_filtered << " (" << filter_pct << "%)\n";
    std::cout << "  output file:         subset_5k_predicted_segments_xgb.txt\n";
    std::cout << "  --- Neural Network ---\n";
    std::cout << "  segments kept:       " << nn_positive_predictions << " (" << nn_kept_pct << "%)\n";
    std::cout << "  segments filtered:   " << nn_segments_filtered << " (" << nn_filter_pct << "%)\n";
    std::cout << "  output file:         subset_5k_predicted_segments_nn.txt\n";
    std::cout << "  --- Timing ---\n";
    std::cout << "  wall clock time:     " << wallclock_sec << " s\n";
    std::cout << "  global stats time:   " << stats_sec << " s\n";
    std::cout << "  feature extraction:  " << extract_sec << " s (" << (segments_read / extract_sec) << " seg/sec)\n";
    std::cout << "  XGB inference:       " << inference_sec << " s (" << (segments_read / inference_sec) << " seg/sec)\n";
    std::cout << "  NN inference:        " << nn_inference_sec << " s (" << (segments_read / nn_inference_sec) << " seg/sec)\n";

}
