/**
 * P-Smoother Main Class Implementation
 * 
 * Based on P-smoother: efficient PBWT smoothing of large haplotype panels
 *   Yue W., Naseri A., Wang V., Shakya P., Zhang S., Zhi D.
 *   Bioinformatics Advances, 2022, Volume 2, Issue 1, vbac045
 *   https://doi.org/10.1093/bioadv/vbac045
 *   GitHub: https://github.com/ZhiGroup/P-smoother
 * 
 * Description:
 *   Adapted for in-memory operation with the IBD caller, this class provides 
 *   efficient PBWT-based smoothing of large haplotype panels for downstream analyses
 *   such as identity-by-descent (IBD) detection.
 */

#include "psmoother.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>

using namespace std;

PSmoother::PSmoother(int n_haps, int n_sites, const PSmootherParams& params)
    : M(n_haps), N(n_sites), params(params) {
}

int PSmoother::smooth(std::vector<std::vector<uint8_t>>& hap_data) {
    auto start_time = chrono::high_resolution_clock::now();

    // Validate input dimensions
    if ((int)hap_data.size() != N) {
        cerr << "[P-smoother] ERROR: hap_data has " << hap_data.size()
             << " sites, expected " << N << endl;
        return -1;
    }
    if (N > 0 && (int)hap_data[0].size() != M) {
        cerr << "[P-smoother] ERROR: hap_data has " << hap_data[0].size()
             << " haplotypes, expected " << M << endl;
        return -1;
    }

    // Step 1: Run reverse PBWT
    runReversePBWT(hap_data);

    // Step 2: Run forward PBWT with error correction
    runForwardPBWT(hap_data);

    // Sort correction locations by (site, haplotype) for efficient range queries
    std::sort(correction_locations.begin(), correction_locations.end());

    if (params.verbose) {
        cerr << "[P-smoother] Tracked " << correction_locations.size()
             << " correction locations" << endl;
    }

    // Free reverse PBWT memory (no longer needed)
    reverse_pre.clear();
    reverse_pre.shrink_to_fit();
    reverse_div.clear();
    reverse_div.shrink_to_fit();

    auto end_time = chrono::high_resolution_clock::now();
    int total_time = static_cast<int>(chrono::duration<double>(end_time - start_time).count());

    cerr << "[P-smoother] Completed in " << total_time << "s ("
         << corrections_count << " corrections)" << endl;

    return corrections_count;
}


