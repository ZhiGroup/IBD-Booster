/**
 * Feature Extractor for IBD Segment Augmentation
 *
 * Extracts features from IBD segments for ML-based classification.
 * Based on constructDataset.py from the original IBD-Booster implementation.
 *
 * For each segment, divides it into N chunks and extracts per-chunk features:
 *   - Physical length (bp)
 *   - Genetic length (cM)
 *   - Number of mismatches (allele differences between haplotypes)
 *   - Number of P-smoother corrections
 */

#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <vector>
#include <cstdint>
#include <utility>
#include <unordered_map>
#include "vcf_utils.hpp"

/**
 * IBDSegment: Represents a detected IBD segment
 */
struct IBDSegment {
    int hap1;           // First haplotype index
    int hap2;           // Second haplotype index
    int start_idx;      // Start site index (into vcf_bp_positions)
    int end_idx;        // End site index (inclusive)
    int start_bp;       // Start position in base pairs
    int end_bp;         // End position in base pairs
    double length_cm;   // Segment length in centiMorgans
};

/**
 * SegmentFeatures: Feature vector for a single segment
 */
struct SegmentFeatures {
    IBDSegment segment;                 // Original segment info
    std::vector<double> features;       // Feature vector (4 * n_chunks values)

    // Individual feature access (for n_chunks chunks, each has 4 features)
    // Index layout: [phys_len_0, gen_len_0, mismatches_0, corrections_0,
    //                phys_len_1, gen_len_1, mismatches_1, corrections_1, ...]
};

/**
 * FeatureExtractorParams: Configuration for feature extraction
 */
struct FeatureExtractorParams {
    int n_chunks = 10;      // Number of chunks to divide each segment into
    bool verbose = false;   // Enable verbose output
};

/**
 * FeatureExtractor: Extracts features from IBD segments
 *
 * Uses HapMajorData for fast mismatch counting via XOR + popcount (~60x faster)
 *
 * Usage:
 *   FeatureExtractor extractor(meta, hap_major, genPos, corrections, params);
 *   SegmentFeatures features = extractor.extract(segment);
 */
class FeatureExtractor {
public:
    /**
     * Constructor
     * @param meta        VCF metadata (positions, sample info)
     * @param hap_major   Haplotype-major layout data (for fast XOR+popcount mismatch counting)
     * @param genPos      Genetic positions (cM) for each site
     * @param corrections Sorted vector of (site, hap) correction pairs from P-smoother
     * @param params      Feature extraction parameters
     */
    FeatureExtractor(
        const HapMetadata& meta,
        const HapMajorData& hap_major,
        const std::vector<double>& genPos,
        const std::vector<std::pair<int, int>>& corrections,
        const FeatureExtractorParams& params = FeatureExtractorParams()
    );

    /**
     * Extract features from a single segment
     * @param segment  IBD segment to extract features from
     * @return         SegmentFeatures containing the feature vector
     */
    SegmentFeatures extract(const IBDSegment& segment) const;

    /**
     * Extract features from multiple segments
     * @param segments  Vector of IBD segments
     * @return          Vector of SegmentFeatures
     */
    std::vector<SegmentFeatures> extractBatch(const std::vector<IBDSegment>& segments) const;

    /**
     * Extract features for batch into contiguous float matrix (for XGBoost)
     * @param segments      Vector of IBD segments
     * @param out_features  Output float array [n_segments × 40], row-major
     *                      Must be pre-allocated: segments.size() * getFeatureCount()
     * @param n_threads     Number of threads for parallel extraction
     */
    void extractBatchFlat(const std::vector<IBDSegment>& segments,
                          float* out_features,
                          int n_threads = 1) const;

    /**
     * Get the number of features per segment
     * @return  Number of features (4 * n_chunks)
     */
    int getFeatureCount() const { return 4 * params.n_chunks; }

    /**
     * Get feature names for CSV header
     * @return  Vector of feature names
     */
    std::vector<std::string> getFeatureNames() const;

private:
    const HapMetadata& meta;
    const HapMajorData& hap_major;
    const std::vector<double>& genPos;
    const std::vector<std::pair<int, int>>& corrections;
    FeatureExtractorParams params;

    // Per-haplotype correction index: hap_id -> sorted list of corrected sites
    // Built once at construction, eliminates O(log 29M) binary search per segment
    std::unordered_map<int, std::vector<int>> corrections_by_hap;

    /**
     * Find site index for a given base pair position using binary search
     * @param bp  Base pair position
     * @return    Site index (or closest site)
     */
    int bpToSiteIndex(int bp) const;

    /**
     * OPTIMIZED: Single-pass corrections counting for all chunks
     * Instead of 10 binary searches, uses 1 binary search + 1 scan
     * @param hap1              First haplotype index
     * @param hap2              Second haplotype index
     * @param seg_start         Segment start site index
     * @param seg_end           Segment end site index
     * @param chunk_boundaries  Array of n_chunks+1 boundary indices
     * @param n_chunks          Number of chunks
     * @param chunk_counts      Output array of correction counts per chunk
     */
    void countCorrectionsAllChunks(int hap1, int hap2, int seg_start, int seg_end,
                                   const int* chunk_boundaries, int n_chunks,
                                   int* chunk_counts) const;

    /**
     * OPTIMIZED: Single-pass mismatch counting for all chunks
     * Instead of 10 separate passes, uses 1 pass with chunk tracking
     * @param hap1              First haplotype index
     * @param hap2              Second haplotype index
     * @param seg_start         Segment start site index
     * @param seg_end           Segment end site index
     * @param chunk_boundaries  Array of n_chunks+1 boundary indices
     * @param n_chunks          Number of chunks
     * @param chunk_counts      Output array of mismatch counts per chunk
     */
    void countMismatchesAllChunks(int hap1, int hap2, int seg_start, int seg_end,
                                  const int* chunk_boundaries, int n_chunks,
                                  int* chunk_counts) const;
};

#endif // FEATURE_EXTRACTOR_HPP
