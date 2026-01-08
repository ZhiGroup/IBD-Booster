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
 * Based on hap-IBD algorithm by Browning & Browning.
 * P-smoother based on work by Degui Zhi and Shaojie Zhang.
 */

#include "../include/pbwt.hpp"
#include "../include/vcf_utils.hpp"
#include "../include/psmoother.hpp"
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
// Instead of site-major (all haps for site 0, all haps for site 1, ...),
// store haplotype-major (all sites for hap 0, all sites for hap 1, ...)
// This dramatically improves cache locality during extension.
struct HapMajorData {
    std::vector<uint8_t> data;  // Packed bits: hap-major layout
    int n_haps;
    int n_sites;
    int bytes_per_hap;  // (n_sites + 7) / 8

    // Transpose from site-major to haplotype-major
    // Parallelized by haplotype (each haplotype writes to its own memory region - no race)
    void transpose(const HapMetadata& meta, const HapData& site_major, int nthreads = 1) {
        n_haps = meta.n_haps;
        n_sites = meta.n_sites;
        bytes_per_hap = (n_sites + 7) / 8;
        const int dense_stride = meta.dense_stride;

        // Allocate haplotype-major storage
        data.resize((size_t)n_haps * bytes_per_hap, 0);

        // Transpose: for each haplotype, gather bits from all sites
        // Parallelized by haplotype - each writes to different memory region (no race)
        uint8_t* data_ptr = data.data();
        const uint8_t* src_ptr = site_major.data();
        const int local_n_sites = n_sites;
        const int local_bytes_per_hap = bytes_per_hap;

        parallel_for(0, n_haps, nthreads, [=](int hap) {
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

    // Get allele for haplotype h at site s
    inline uint8_t get(int hap, int site) const {
        return (data[(size_t)hap * bytes_per_hap + (site >> 3)] >> (site & 7)) & 1;
    }

    // Fast check if two haplotypes match at a site (XOR-based, single comparison)
    inline bool allelesMatch(int hap1, int hap2, int site) const {
        const int byte_idx = site >> 3;
        const int bit_idx = site & 7;
        return !((data[(size_t)hap1 * bytes_per_hap + byte_idx] ^
                  data[(size_t)hap2 * bytes_per_hap + byte_idx]) >> bit_idx & 1);
    }

    // Get pointer to haplotype's data (for block operations)
    inline const uint8_t* hapPtr(int hap) const {
        return data.data() + (size_t)hap * bytes_per_hap;
    }

    // Fast 64-bit backward scan to find first mismatch position (scanning backwards)
    // Returns the first matching position (inclusive), or 0 if matches to beginning
    // Much faster than bit-by-bit allelesMatch() loop
    inline int extendMatchBackward64(int hap1, int hap2, int start) const {
        if (start <= 0) return start;

        const uint8_t* __restrict h1_ptr = data.data() + (size_t)hap1 * bytes_per_hap;
        const uint8_t* __restrict h2_ptr = data.data() + (size_t)hap2 * bytes_per_hap;

        int m = start - 1;

        // 64-bit word scanning backwards: skip 64 matching sites at once when byte-aligned
        while (m >= 63) {
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
                // Has mismatches - find highest set bit (last mismatch in this word)
                int highest_mismatch = 63 - __builtin_clzll(xor_word);
                return (m - 63) + highest_mismatch + 1;  // Return first matching position after mismatch
            }
            break;  // Not aligned, switch to bit-by-bit
        }

        // Bit-by-bit for remaining sites or unaligned start
        while (m >= 0) {
            const int byte_idx = m >> 3;
            const int bit_idx = m & 7;
            if ((h1_ptr[byte_idx] ^ h2_ptr[byte_idx]) >> bit_idx & 1) {
                return m + 1;  // Mismatch at m, return m+1 as first match
            }
            --m;
        }

        return 0;  // Matched all the way to beginning
    }

    // Fast 64-bit forward scan to find first mismatch position
    // Returns the last matching position (inclusive), or LastMarker if matches to end
    // Much faster than bit-by-bit allelesMatch() loop
    inline int extendMatchForward64(int hap1, int hap2, int start, int LastMarker) const {
        if (start >= LastMarker) return start;

        const uint8_t* __restrict h1_ptr = data.data() + (size_t)hap1 * bytes_per_hap;
        const uint8_t* __restrict h2_ptr = data.data() + (size_t)hap2 * bytes_per_hap;

        int m = start + 1;

        // 64-bit word scanning: skip 64 matching sites at once when byte-aligned
        while (m + 64 <= LastMarker) {
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
                // Has mismatches - find lowest set bit (first mismatch)
                int lowest_mismatch = __builtin_ctzll(xor_word);
                return m + lowest_mismatch - 1;  // Return last matching position
            }
            break;  // Not aligned, switch to bit-by-bit
        }

        // Bit-by-bit for remaining sites or unaligned start
        while (m <= LastMarker) {
            const int byte_idx = m >> 3;
            const int bit_idx = m & 7;
            if ((h1_ptr[byte_idx] ^ h2_ptr[byte_idx]) >> bit_idx & 1) {
                return m - 1;  // Mismatch at m, return m-1 as last match
            }
            ++m;
        }

        return LastMarker;  // Matched all the way to end
    }
};

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

// Per-thread output buffer with striped deduplication for reduced contention
class OutputBuffer {
private:
    static const size_t BUFFER_THRESHOLD = 1 << 18;  // 256 KB
    std::string buffer;
    FILE* output_file;
    std::mutex& file_mtx;
    std::atomic<size_t>& segment_count;
    StripedDedup& dedup;
    size_t local_count = 0;  // Thread-local counter to reduce atomic contention

public:
    OutputBuffer(FILE* file, std::mutex& mtx, std::atomic<size_t>& count, StripedDedup& dedup_ref)
        : output_file(file), file_mtx(mtx), segment_count(count), dedup(dedup_ref) {
        buffer.reserve(BUFFER_THRESHOLD * 3 / 2);
    }

    // Write segment data to buffer
    bool write(const char* data, size_t len, const SegmentKey& key) {
        buffer.append(data, len);
        local_count++;  // Thread-local, no atomic overhead

        // Flush when buffer exceeds threshold
        if (buffer.size() >= BUFFER_THRESHOLD) {
            flush();
        }
        return true;
    }

    void flush() {
        if (!buffer.empty()) {
            std::lock_guard<std::mutex> lock(file_mtx);
            fwrite(buffer.data(), 1, buffer.size(), output_file);
            buffer.clear();
        }
        // Batch update global counter
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
    char line_buffer[256];  // Thread-local line buffer

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

            // Write segment to buffer with deduplication (use MIN_OUTPUT_CM for final filtering)
            double length_cm = genPos[nextInclEnd] - genPos[nStart];
            if (length_cm >= MIN_OUTPUT_CM && (hap1 >> 1) != (hap2 >> 1)) {
                // Normalize haplotype order for consistent dedup across windows
                // (PBWT order differs between windows, so same pair might be (5,10) vs (10,5))
                // Use full haplotype IDs, not sample IDs, to distinguish different hap pairs
                int h1 = std::min(hap1, hap2);
                int h2 = std::max(hap1, hap2);
                SegmentKey key = {h1, h2,
                                 meta.vcf_bp_positions[nStart],
                                 meta.vcf_bp_positions[nextInclEnd]};
                int len = snprintf(line_buffer, sizeof(line_buffer),
                                  "%d\t%d\t%d\t%d\t%d\t%d\t%.6f\n",
                                  hap1 / 2, hap1 & 1, hap2 / 2, hap2 & 1,
                                  meta.vcf_bp_positions[nStart],
                                  meta.vcf_bp_positions[nextInclEnd],
                                  length_cm);
                output_buffer.write(line_buffer, len, key);
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
    FILE* output_file,
    std::mutex& file_mtx,
    std::atomic<size_t>& segment_count,
    StripedDedup& dedup,
    bool isLastWindow,
    TimingStats& timing)  // Added timing parameter
{
    const size_t SEED_LIST_THRESHOLD = 65536;
    int n_haps = meta.n_haps;

    // Create per-thread output buffer
    OutputBuffer output_buffer(output_file, file_mtx, segment_count, dedup);

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
                processSeedBatch(to_flush, genPos, meta, hap_major, MIN_SEED_CM, MIN_OUTPUT_CM,
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
        processSeedBatch(seedList, genPos, meta, hap_major, MIN_SEED_CM, MIN_OUTPUT_CM,
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
            processSeedBatch(batch, genPos, meta, hap_major, MIN_SEED_CM, MIN_OUTPUT_CM,
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

    // Run P-smoother for haplotype error correction (unless disabled)
    if (!skip_psmoother) {
        auto ps_start = std::chrono::steady_clock::now();
        PSmoother smoother(meta.n_haps, meta.n_sites, ps_params);
        int corrections = smoother.smooth(hap_2d);
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
    hap_major.transpose(meta, hap_data, nthreads);

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

    // Open output file
    FILE* output_file = fopen("segments2.tsv", "w");
    if (!output_file) {
        std::cerr << "[ERROR] Cannot open output file\n";
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
        char line_buffer[256];  // For formatted output

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

                    // Write segment to file (use MIN_OUTPUT_CM for final filtering)
                    double length_cm = genPos[nextInclEnd] - genPos[nStart];
                    if (length_cm >= MIN_OUTPUT_CM && (hap1 >> 1) != (hap2 >> 1)) {
                        int len = snprintf(line_buffer, sizeof(line_buffer),
                                          "%d\t%d\t%d\t%d\t%d\t%d\t%.6f\n",
                                          hap1 / 2, hap1 & 1, hap2 / 2, hap2 & 1,
                                          meta.vcf_bp_positions[nStart],
                                          meta.vcf_bp_positions[nextInclEnd],
                                          length_cm);
                        fwrite(line_buffer, 1, len, output_file);
                        segment_count++;
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

        fclose(output_file);

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
                               output_file, std::ref(file_mtx), std::ref(segment_count),
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
            output_file, std::ref(file_mtx), std::ref(segment_count),
            std::ref(dedup),
            true,  // isLastWindow
            std::ref(timing));

        // Wait for worker threads to complete
        for (auto& t : threads) {
            t.join();
        }

        seedQ.markFinished();

        fclose(output_file);

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
}
