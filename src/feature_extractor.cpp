/**
 * Feature Extractor Implementation
 *
 * Extracts features from IBD segments for ML-based classification.
 * For each segment, divides into N chunks and computes per-chunk features:
 *   - Physical length (bp)
 *   - Genetic length (cM)
 *   - Mismatch count (allele differences between haplotypes)
 *   - Correction count (P-smoother corrections in this region)
 *
 * Optimizations:
 *   - HapMajorData layout for fast XOR+popcount mismatch counting
 *   - Per-haplotype correction index for O(1) lookup
 *   - OpenMP parallelization for batch extraction
 */

#include "feature_extractor.hpp"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

FeatureExtractor::FeatureExtractor(
    const HapMetadata& meta,
    const HapMajorData& hap_major,
    const std::vector<double>& genPos,
    const std::vector<std::pair<int, int>>& corrections,
    const FeatureExtractorParams& params)
    : meta(meta), hap_major(hap_major), genPos(genPos), corrections(corrections), params(params)
{
    // Build per-haplotype correction index for O(1) lookup instead of O(log 29M) binary search
    // corrections is vector of (site, hap) pairs
    if (!corrections.empty()) {
        cerr << "[FeatureExtractor] Building per-haplotype correction index..." << endl;
        for (const auto& corr : corrections) {
            corrections_by_hap[corr.second].push_back(corr.first);  // hap -> site
        }
        // Sort each haplotype's correction sites for potential binary search within
        for (auto& kv : corrections_by_hap) {
            sort(kv.second.begin(), kv.second.end());
        }
        cerr << "[FeatureExtractor] Index built: " << corrections_by_hap.size()
             << " haplotypes with corrections" << endl;
    }
}

int FeatureExtractor::bpToSiteIndex(int bp) const {
    // Binary search for the site index corresponding to bp position
    const auto& positions = meta.vcf_bp_positions;
    auto it = lower_bound(positions.begin(), positions.end(), bp);

    if (it == positions.end()) {
        return static_cast<int>(positions.size()) - 1;
    }

    return static_cast<int>(it - positions.begin());
}

// OPTIMIZED: Single-pass corrections counting for all chunks
// Instead of 10 binary searches, uses 1 binary search + 1 scan
void FeatureExtractor::countCorrectionsAllChunks(
    int hap1, int hap2,
    int seg_start, int seg_end,
    const int* chunk_boundaries,
    int n_chunks,
    int* chunk_counts) const
{
    // Initialize counts to zero
    for (int i = 0; i < n_chunks; ++i) {
        chunk_counts[i] = 0;
    }

    if (corrections.empty()) return;

    // ONE binary search to find first correction >= segment start
    auto it = lower_bound(corrections.begin(), corrections.end(),
                          make_pair(seg_start, 0));

    // ONE scan through all corrections in segment range
    while (it != corrections.end() && it->first <= seg_end) {
        // Check if this correction is for one of our haplotypes
        if (it->second == hap1 || it->second == hap2) {
            // Find which chunk this correction belongs to
            int site = it->first;
            // Binary search could be used, but linear search is fine for 10 chunks
            for (int c = 0; c < n_chunks; ++c) {
                if (site >= chunk_boundaries[c] && site < chunk_boundaries[c + 1]) {
                    chunk_counts[c]++;
                    break;
                }
            }
        }
        ++it;
    }
}

// OPTIMIZED: Mismatch counting for all chunks using efficient byte-level XOR+popcount
// Uses HapMajorData::countMismatches for each chunk (which uses 64-bit word processing)
void FeatureExtractor::countMismatchesAllChunks(
    int hap1, int hap2,
    int seg_start, int seg_end,
    const int* chunk_boundaries,
    int n_chunks,
    int* chunk_counts) const
{
    if (seg_start >= seg_end) {
        for (int i = 0; i < n_chunks; ++i) {
            chunk_counts[i] = 0;
        }
        return;
    }

    // Use efficient byte-level countMismatches for each chunk
    for (int c = 0; c < n_chunks; ++c) {
        int chunk_start = chunk_boundaries[c];
        int chunk_end = chunk_boundaries[c + 1];
        chunk_counts[c] = hap_major.countMismatches(hap1, hap2, chunk_start, chunk_end);
    }
}

SegmentFeatures FeatureExtractor::extract(const IBDSegment& segment) const {
    SegmentFeatures result;
    result.segment = segment;
    result.features.resize(4 * params.n_chunks, 0.0);

    // Get site indices for segment bounds
    int seg_start_idx = segment.start_idx;
    int seg_end_idx = segment.end_idx;

    // Number of sites in segment
    int n_sites = seg_end_idx - seg_start_idx + 1;

    if (n_sites <= 0) {
        return result;
    }

    // Single loop: compute chunk bounds and features together
    // Single-pass corrections: one binary search + scan for whole segment

    // Find first correction >= segment start (ONE binary search for all chunks)
    auto corr_it = lower_bound(corrections.begin(), corrections.end(),
                               make_pair(seg_start_idx, 0));

    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        int chunk_start = seg_start_idx + (chunk * n_sites) / params.n_chunks;
        int chunk_end = seg_start_idx + ((chunk + 1) * n_sites) / params.n_chunks;

        if (chunk_start >= chunk_end) {
            continue;
        }

        // Feature 1: Physical length (bp)
        double phys_len = static_cast<double>(
            meta.vcf_bp_positions[chunk_end - 1] - meta.vcf_bp_positions[chunk_start]);

        // Feature 2: Genetic length (cM)
        double gen_len = genPos[chunk_end - 1] - genPos[chunk_start];

        // Feature 3: Mismatches (use efficient byte-level XOR+popcount)
        int mm_count = hap_major.countMismatches(segment.hap1, segment.hap2,
                                                  chunk_start, chunk_end);

        // Feature 4: Corrections (scan from current position, single-pass across all chunks)
        int corr_count = 0;
        while (corr_it != corrections.end() && corr_it->first < chunk_end) {
            if (corr_it->second == segment.hap1 || corr_it->second == segment.hap2) {
                corr_count++;
            }
            ++corr_it;
        }

        // Store features
        int base_idx = chunk * 4;
        result.features[base_idx + 0] = phys_len;
        result.features[base_idx + 1] = gen_len;
        result.features[base_idx + 2] = static_cast<double>(mm_count);
        result.features[base_idx + 3] = static_cast<double>(corr_count);
    }

    return result;
}

vector<SegmentFeatures> FeatureExtractor::extractBatch(const vector<IBDSegment>& segments) const {
    vector<SegmentFeatures> results;
    results.reserve(segments.size());

    for (const auto& seg : segments) {
        results.push_back(extract(seg));
    }

    if (params.verbose) {
        cerr << "[FeatureExtractor] Extracted features for " << segments.size() << " segments" << endl;
    }

    return results;
}

vector<string> FeatureExtractor::getFeatureNames() const {
    vector<string> names;
    names.reserve(4 * params.n_chunks);

    for (int i = 0; i < params.n_chunks; ++i) {
        names.push_back("phys_len_" + to_string(i));
        names.push_back("gen_len_" + to_string(i));
        names.push_back("n_mismatches_" + to_string(i));
        names.push_back("n_corrections_" + to_string(i));
    }

    return names;
}

void FeatureExtractor::extractBatchFlat(const vector<IBDSegment>& segments,
                                         float* out_features,
                                         int n_threads) const {
    const int n_segments = static_cast<int>(segments.size());
    const int n_features = 4 * params.n_chunks;

    if (n_segments == 0) return;

    // Set OpenMP thread count
    omp_set_num_threads(n_threads);

    // Simple parallel loop - each segment is independent
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_segments; ++idx) {
        const IBDSegment& seg = segments[idx];
        float* out = out_features + idx * n_features;

        const int seg_start_idx = seg.start_idx;
        const int seg_end_idx = seg.end_idx;
        const int n_sites = seg_end_idx - seg_start_idx + 1;

        if (n_sites <= 0) {
            // Fill with zeros
            for (int f = 0; f < n_features; ++f) {
                out[f] = 0.0f;
            }
            continue;
        }

        // Precompute all chunk boundaries once
        int chunk_bounds[11];  // 10 chunks = 11 boundaries
        for (int c = 0; c <= params.n_chunks; ++c) {
            chunk_bounds[c] = seg_start_idx + (c * n_sites) / params.n_chunks;
        }

        // Single-pass mismatch counting
        const uint8_t* h1_ptr = hap_major.hapPtr(seg.hap1);
        const uint8_t* h2_ptr = hap_major.hapPtr(seg.hap2);
        int mm_counts[10] = {0};

        const int start_byte = seg_start_idx >> 3;
        const int end_byte = (seg_end_idx - 1) >> 3;

        for (int byte_idx = start_byte; byte_idx <= end_byte; ++byte_idx) {
            uint8_t xored = h1_ptr[byte_idx] ^ h2_ptr[byte_idx];
            if (xored == 0) continue;  // Fast skip for matching bytes

            int bit_start = (byte_idx == start_byte) ? (seg_start_idx & 7) : 0;
            int bit_end = (byte_idx == end_byte) ? ((seg_end_idx - 1) & 7) : 7;

            for (int bit = bit_start; bit <= bit_end; ++bit) {
                if ((xored >> bit) & 1) {
                    int site = (byte_idx << 3) + bit;
                    for (int c = 0; c < params.n_chunks; ++c) {
                        if (site >= chunk_bounds[c] && site < chunk_bounds[c + 1]) {
                            mm_counts[c]++;
                            break;
                        }
                    }
                }
            }
        }

        // Correction counting - O(1) hash lookup per haplotype
        int corr_counts[10] = {0};

        auto it1 = corrections_by_hap.find(seg.hap1);
        if (it1 != corrections_by_hap.end()) {
            for (int site : it1->second) {
                if (site >= seg_start_idx && site < seg_end_idx) {
                    for (int c = 0; c < params.n_chunks; ++c) {
                        if (site >= chunk_bounds[c] && site < chunk_bounds[c + 1]) {
                            corr_counts[c]++;
                            break;
                        }
                    }
                }
            }
        }

        auto it2 = corrections_by_hap.find(seg.hap2);
        if (it2 != corrections_by_hap.end()) {
            for (int site : it2->second) {
                if (site >= seg_start_idx && site < seg_end_idx) {
                    for (int c = 0; c < params.n_chunks; ++c) {
                        if (site >= chunk_bounds[c] && site < chunk_bounds[c + 1]) {
                            corr_counts[c]++;
                            break;
                        }
                    }
                }
            }
        }

        // Store all features
        for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
            const int chunk_start = chunk_bounds[chunk];
            const int chunk_end = chunk_bounds[chunk + 1];
            const int base_idx = chunk * 4;

            if (chunk_start >= chunk_end) {
                out[base_idx + 0] = 0.0f;
                out[base_idx + 1] = 0.0f;
                out[base_idx + 2] = 0.0f;
                out[base_idx + 3] = 0.0f;
            } else {
                out[base_idx + 0] = static_cast<float>(
                    meta.vcf_bp_positions[chunk_end - 1] - meta.vcf_bp_positions[chunk_start]);
                out[base_idx + 1] = static_cast<float>(
                    genPos[chunk_end - 1] - genPos[chunk_start]);
                out[base_idx + 2] = static_cast<float>(mm_counts[chunk]);
                out[base_idx + 3] = static_cast<float>(corr_counts[chunk]);
            }
        }
    }

}
