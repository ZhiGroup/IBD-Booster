// P-smoother main class implementation
// Based on P-smoother by Degui Zhi and Shaojie Zhang
// Modified for in-memory operation with cpp-HapIBD

#include "psmoother.hpp"
#include <iostream>
#include <chrono>

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
