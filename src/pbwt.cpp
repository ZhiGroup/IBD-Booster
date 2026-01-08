/**
 * PBWT Step Implementation
 *
 * This file implements the core PBWT update operation. Each call to
 * pbwt_step() processes one site, updating the A (permutation) and
 * D (divergence) arrays.
 *
 * The algorithm partitions haplotypes by their allele (0 or 1) at the
 * current site, maintaining relative order within each partition.
 * The divergence array tracks where neighboring haplotypes first differ.
 */

#include "pbwt.hpp"
#include <cstring>
#include <algorithm>

/// Initialize PBWT state with identity permutation
PBWTState::PBWTState(int32_t n_haps) {
    A.resize(n_haps);
    D.resize(n_haps);
    A0.resize(n_haps);
    A1.resize(n_haps);
    D0.resize(n_haps);
    D1.resize(n_haps);
    
    // Initialize A with haplotype indices and D with zeros
    for (int i = 0; i < n_haps; ++i) {
        A[i] = i;
        D[i] = 0;
    }
}

void pbwt_step(
    int site_idx,
    const std::vector<uint8_t>& site,
    PBWTState& st
) {
    const int n = (int)st.A.size();
    int count0 = 0, count1 = 0;

    // Initialize divergence trackers for each allele group
    int p0 = site_idx + 1, p1 = site_idx + 1;

    for (int r = 0; r < n; ++r) {
        int hap = st.A[r];
        uint8_t allele = site[hap];
        int k = st.D[r];

        // Update all divergence trackers with current divergence
        if (k > p0) p0 = k;
        if (k > p1) p1 = k;

        if (allele == 0) {
            st.A0[count0] = hap;
            st.D0[count0] = p0;  // assign divergence
            p0 = 0;  // reset tracker for this allele
            ++count0;
        } else {
            st.A1[count1] = hap;
            st.D1[count1] = p1;  // assign divergence
            p1 = 0;  // reset tracker for this allele
            ++count1;
        }
    }

    // Merge partitions back into main arrays
    if (count0 > 0) {
        std::copy(st.A0.begin(), st.A0.begin() + count0, st.A.begin());
        std::copy(st.D0.begin(), st.D0.begin() + count0, st.D.begin());
    }
    if (count1 > 0) {
        std::copy(st.A1.begin(), st.A1.begin() + count1, st.A.begin() + count0);
        std::copy(st.D1.begin(), st.D1.begin() + count1, st.D.begin() + count0);
    }
}