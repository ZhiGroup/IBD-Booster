/**
 * P-smoother: Haplotype Error Correction using Bidirectional PBWT
 *
 * Based on the P-smoother algorithm by Degui Zhi and Shaojie Zhang.
 * Modified for in-memory operation with IBD-Booster.
 *
 * Algorithm:
 *   1. Run reverse PBWT (backwards through sites) to compute divergence
 *   2. Run forward PBWT with error correction using both forward and
 *      reverse divergence to identify IBS blocks
 *   3. Within each IBS block, if minority allele frequency < rho,
 *      correct minority alleles to the majority allele
 *
 * This corrects genotyping errors that would otherwise cause false
 * IBD segment breaks.
 */

#ifndef PSMOOTHER_HPP
#define PSMOOTHER_HPP

#include <vector>
#include <cstdint>
#include <string>

/**
 * PSmootherParams: Configuration parameters for P-smoother.
 */
struct PSmootherParams {
    int length = 20;       // L: PBWT block length in sites
    int width = 20;        // W: Minimum haplotypes in block for correction
    int gap = 1;           // G: Gap size for buffered correction
    double rho = 0.05;     // Error rate threshold: correct if freq < rho
    int checkpoint = 100000;  // Progress reporting interval
    bool verbose = true;      // Enable detailed output
    int nthreads = 1;         // Number of threads for parallel processing
};

/**
 * PSmoother: Main class for haplotype error correction.
 *
 * Usage:
 *   PSmoother smoother(n_haps, n_sites, params);
 *   int corrections = smoother.smooth(hap_data);
 */
class PSmoother {
public:
    PSmoother(int n_haps, int n_sites, const PSmootherParams& params = PSmootherParams());

    // Main entry point: smooth haplotype data in-place
    // hap_data[site][haplotype] = allele (0 or 1)
    // Returns number of corrections made
    int smooth(std::vector<std::vector<uint8_t>>& hap_data);

    // Get statistics
    int getCorrectionsCount() const { return corrections_count; }
    int getBlocksProcessed() const { return blocks_processed; }

private:
    int M;  // number of haplotypes
    int N;  // number of sites
    PSmootherParams params;

    // Statistics
    int corrections_count = 0;
    int blocks_processed = 0;

    // Reverse PBWT data (computed by runReversePBWT)
    // Stored as flattened [site * M + i] for memory efficiency
    std::vector<int> reverse_pre;  // flattened [N][M]
    std::vector<int> reverse_div;  // flattened [N][M]

    // Run reverse PBWT (backwards through sites), store results
    void runReversePBWT(const std::vector<std::vector<uint8_t>>& hap_data);

    // Run forward PBWT with error correction
    void runForwardPBWT(std::vector<std::vector<uint8_t>>& hap_data);
};

#endif // PSMOOTHER_HPP
