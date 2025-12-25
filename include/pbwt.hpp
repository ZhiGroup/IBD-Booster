/**
 * Positional Burrows-Wheeler Transform (PBWT) for IBD Detection
 *
 * The PBWT maintains haplotypes in sorted order by their prefixes (alleles
 * at sites 0..k). The key insight is that haplotypes sharing long identical
 * prefixes are adjacent in the sorted order, making it efficient to find
 * long matches.
 *
 * Key arrays:
 *   - A[i]: The haplotype at position i in the sorted order
 *   - D[i]: The divergence point - the first site where A[i] differs from A[i-1]
 *
 * Haplotypes with small D[i] values share long matches with their neighbors.
 * This is the basis for efficient IBD seed detection.
 *
 * Reference: Durbin, R. (2014). Efficient haplotype matching and storage
 * using the positional Burrows-Wheeler transform (PBWT).
 */

#ifndef PBWT_HPP
#define PBWT_HPP

#include <cstdint>
#include <vector>

/**
 * PBWTState: Maintains the state of the PBWT algorithm.
 *
 * A and D are updated at each site via pbwt_step().
 * A0/A1 and D0/D1 are temporary buffers used during the update.
 */
struct PBWTState {
    std::vector<int32_t> A;     // Permutation: A[i] = haplotype at sorted position i
    std::vector<int32_t> D;     // Divergence: D[i] = first differing site between A[i] and A[i-1]
    std::vector<int32_t> A0;    // Temp buffer for haplotypes with allele 0
    std::vector<int32_t> A1;    // Temp buffer for haplotypes with allele 1
    std::vector<int32_t> D0;    // Temp divergence for allele 0 haplotypes
    std::vector<int32_t> D1;    // Temp divergence for allele 1 haplotypes

    explicit PBWTState(int32_t n_haps);
};

/**
 * Perform one PBWT update step for a single site.
 *
 * Updates A and D arrays based on the alleles at the current site.
 * After this step, A is sorted by prefixes through site_idx.
 *
 * @param site_idx  Current site index (used for divergence updates)
 * @param site      Alleles for all haplotypes at this site [n_haps]
 * @param st        PBWT state to update
 */
void pbwt_step(
    int site_idx,
    const std::vector<uint8_t>& site,
    PBWTState& st
);

#endif // PBWT_HPP