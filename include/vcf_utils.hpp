/**
 * VCF Utilities for IBD-Booster
 *
 * This module handles VCF file I/O and haplotype data management:
 *   - Reading phased VCF files into memory
 *   - Bit-packed storage for efficient memory usage
 *   - Genetic map interpolation
 *   - P-smoother integration (2D format for correction, then repack)
 */

#ifndef VCF_UTILS_HPP
#define VCF_UTILS_HPP

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

/**
 * SiteType: Storage strategy for a variant site.
 *   - DENSE: Bit-packed storage for all haplotypes (common sites)
 *   - SPARSE: Same as DENSE (reserved for future sparse encoding)
 *   - MONO_REF: All haplotypes carry reference allele (no storage needed)
 *   - MONO_ALT: All haplotypes carry alternate allele (no storage needed)
 */
enum class SiteType : uint8_t { DENSE = 0, SPARSE = 1, MONO_REF = 2, MONO_ALT = 3 };

/**
 * HapMetadata: Metadata about haplotypes and variant sites.
 * Stores positions, sample IDs, and indexing information for bit-packed data.
 */
struct HapMetadata {
    int32_t n_samples;                         // Number of samples in VCF
    int32_t n_sites;                           // Number of variant sites (after filtering)
    int32_t n_haps;                            // Number of haplotypes (2 * n_samples for diploid)
    int32_t dense_stride;                      // Bytes per site in bit-packed format: (n_haps + 7) / 8
    std::vector<int> vcf_bp_positions;         // Physical position (bp) for each site
    std::vector<size_t> site_offsets;          // Byte offset into hap_data for each site
    std::vector<SiteType> site_types;          // Storage type for each site
    std::vector<int> site_mac;                 // Minor allele count (for post-smoothing filter)
    std::vector<uint8_t> isDiploid;            // Per-sample ploidy flag
    std::vector<std::string> sampleIDs;        // Sample identifiers from VCF header

    // For fast smoothed VCF writing (avoids re-parsing input VCF)
    std::string vcf_header;                    // Full VCF header (all # lines)
    std::vector<std::string> vcf_fixed_fields; // Fixed fields per site (CHROM...FORMAT\t)
};

/// Bit-packed haplotype data: one bit per haplotype per site (site-major layout)
using HapData = std::vector<uint8_t>;

// Instead of site-major (all haps for site 0, all haps for site 1, ...),
// store haplotype-major (all sites for hap 0, all sites for hap 1, ...)
// This dramatically improves cache locality during extension.
struct HapMajorData {
    std::vector<uint8_t> data;  // Packed bits: hap-major layout
    int n_haps;
    int n_sites;
    int bytes_per_hap;  // (n_sites + 7) / 8

    HapMajorData() : n_haps(0), n_sites(0), bytes_per_hap(0) {}

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

    /**
     * Count mismatches between two haplotypes in a site range using XOR + popcount
     * ~60x faster than per-site allelesDiffer() calls
     * @param hap1       First haplotype index
     * @param hap2       Second haplotype index
     * @param start_idx  Start site index (inclusive)
     * @param end_idx    End site index (exclusive)
     * @return           Number of mismatches
     */
    inline int countMismatches(int hap1, int hap2, int start_idx, int end_idx) const {
        if (start_idx >= end_idx) return 0;

        const uint8_t* __restrict h1_ptr = hapPtr(hap1);
        const uint8_t* __restrict h2_ptr = hapPtr(hap2);

        int count = 0;
        int start_byte = start_idx >> 3;
        int end_byte = (end_idx - 1) >> 3;
        int start_bit = start_idx & 7;
        int end_bit = (end_idx - 1) & 7;

        if (start_byte == end_byte) {
            // All bits in same byte
            uint8_t mask = ((1u << (end_bit - start_bit + 1)) - 1) << start_bit;
            uint8_t xored = (h1_ptr[start_byte] ^ h2_ptr[start_byte]) & mask;
            return __builtin_popcount(xored);
        }

        // First partial byte
        if (start_bit != 0) {
            uint8_t mask = ~((1u << start_bit) - 1);  // bits from start_bit to 7
            uint8_t xored = (h1_ptr[start_byte] ^ h2_ptr[start_byte]) & mask;
            count += __builtin_popcount(xored);
            start_byte++;
        }

        // Full 64-bit words in the middle
        while (start_byte + 8 <= end_byte) {
            uint64_t w1, w2;
            memcpy(&w1, h1_ptr + start_byte, 8);
            memcpy(&w2, h2_ptr + start_byte, 8);
            count += __builtin_popcountll(w1 ^ w2);
            start_byte += 8;
        }

        // Remaining full bytes
        while (start_byte < end_byte) {
            count += __builtin_popcount(h1_ptr[start_byte] ^ h2_ptr[start_byte]);
            start_byte++;
        }

        // Last partial byte
        uint8_t mask = (1u << (end_bit + 1)) - 1;  // bits 0 to end_bit
        uint8_t xored = (h1_ptr[end_byte] ^ h2_ptr[end_byte]) & mask;
        count += __builtin_popcount(xored);

        return count;
    }

    // Fast 64-bit backward scan to find first mismatch position (scanning backwards)
    // Returns the first matching position (inclusive), or 0 if matches to beginning
    // Much faster than bit-by-bit allelesMatch() loop
    inline int extendMatchBackward64(int hap1, int hap2, int start) const {
        if (start <= 0) return start;

        const uint8_t* __restrict h1_ptr = hapPtr(hap1);
        const uint8_t* __restrict h2_ptr = hapPtr(hap2);

        int m = start - 1;

        while (m >= 63) {
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
                int highest_mismatch = 63 - __builtin_clzll(xor_word);
                return (m - 63) + highest_mismatch + 1;
            }
            break;
        }

        while (m >= 0) {
            const int byte_idx = m >> 3;
            const int bit_idx = m & 7;
            if ((h1_ptr[byte_idx] ^ h2_ptr[byte_idx]) >> bit_idx & 1) {
                return m + 1;
            }
            --m;
        }

        return 0;
    }

    // Fast 64-bit forward scan to find first mismatch position
    // Returns the last matching position (inclusive), or LastMarker if matches to end
    // Much faster than bit-by-bit allelesMatch() loop
    inline int extendMatchForward64(int hap1, int hap2, int start, int LastMarker) const {
        if (start >= LastMarker) return start;

        const uint8_t* __restrict h1_ptr = hapPtr(hap1);
        const uint8_t* __restrict h2_ptr = hapPtr(hap2);

        int m = start + 1;

        while (m + 64 <= LastMarker) {
            if ((m & 7) == 0) {
                int word_start_byte = m >> 3;
                uint64_t w1, w2;
                memcpy(&w1, h1_ptr + word_start_byte, 8);
                memcpy(&w2, h2_ptr + word_start_byte, 8);
                uint64_t xor_word = w1 ^ w2;
                if (xor_word == 0) {
                    m += 64;
                    continue;
                }
                int lowest_mismatch = __builtin_ctzll(xor_word);
                return m + lowest_mismatch - 1;
            }
            break;
        }

        while (m <= LastMarker) {
            const int byte_idx = m >> 3;
            const int bit_idx = m & 7;
            if ((h1_ptr[byte_idx] ^ h2_ptr[byte_idx]) >> bit_idx & 1) {
                return m - 1;
            }
            ++m;
        }

        return LastMarker;
    }
};

/// Genetic map: bp positions and corresponding cM positions
struct GeneticMap {
    std::vector<int> bp_vec;     // Physical positions (bp)
    std::vector<double> cm_vec;  // Genetic positions (cM)
};

/// Read VCF directly to bit-packed format (legacy, not used with P-smoother)
HapMetadata read_vcf(const std::string& vcf_file, const std::string& bin_file, int minMac, HapData& hap_data);

/// Read genetic map file (PLINK format: chrom, strand, cM, bp)
GeneticMap readGeneticMap(const std::string &filename);

/// Interpolate genetic positions (cM) for VCF marker positions
std::vector<double> interpolateGeneticPositions(
    const GeneticMap &map,
    const std::vector<int> &vcf_bp_positions
);

/// Get allele for a single haplotype at a single site
inline bool get_hap(int site_idx, int hap_idx,
                    const HapMetadata& meta,
                    const HapData& hap_data)
{
    SiteType stype = meta.site_types[site_idx];
    size_t offset = meta.site_offsets[site_idx];

    switch (stype) {
        case SiteType::MONO_REF:
            return 0;
        case SiteType::MONO_ALT:
            return 1;
        case SiteType::DENSE:
        {
            const uint8_t* site_ptr = hap_data.data() + offset;
            return (site_ptr[hap_idx >> 3] >> (hap_idx & 7)) & 1;
        }
        case SiteType::SPARSE:
        {
            const uint8_t* site_ptr = hap_data.data() + offset;
            return (site_ptr[hap_idx >> 3] >> (hap_idx & 7)) & 1;
        }
        default:
            throw std::runtime_error("Unknown site type in get_hap()");
    }
}

/// Check if two haplotypes have different alleles at a site (optimized)
inline bool allelesDiffer(int site_idx, int hap1, int hap2,
                          const HapMetadata &meta,
                          const HapData &hap_data)
{
    SiteType stype = meta.site_types[site_idx];

    // Fast path: monomorphic sites - all haplotypes have same allele
    if (__builtin_expect(stype == SiteType::MONO_REF || stype == SiteType::MONO_ALT, 0)) {
        return false;
    }

    // DENSE/SPARSE: Read site data once and compare
    size_t offset = meta.site_offsets[site_idx];
    const uint8_t* site_ptr = hap_data.data() + offset;

    uint8_t val1 = (site_ptr[hap1 >> 3] >> (hap1 & 7)) & 1;
    uint8_t val2 = (site_ptr[hap2 >> 3] >> (hap2 & 7)) & 1;

    return val1 != val2;
}


/// Extract all haplotype alleles for a single site into a buffer
inline void get_site(int site_idx,
                     const HapMetadata& meta,
                     const HapData& hap_data,
                     std::vector<uint8_t>& site_buffer)
{
    SiteType stype = meta.site_types[site_idx];
    size_t offset = meta.site_offsets[site_idx];
    int n_haps = meta.n_haps;

    if ((int)site_buffer.size() != n_haps) site_buffer.assign(n_haps, 0);

    switch (stype) {
        case SiteType::MONO_REF:
            std::fill(site_buffer.begin(), site_buffer.end(), 0);
            break;
        case SiteType::MONO_ALT:
            std::fill(site_buffer.begin(), site_buffer.end(), 1);
            break;
        case SiteType::DENSE:
        {
            const uint8_t* site_ptr = hap_data.data() + offset;
            for (int s = 0; s < n_haps; ++s) {
                site_buffer[s] = (site_ptr[s >> 3] >> (s & 7)) & 1;
            }
            break;
        }
        case SiteType::SPARSE:
        {
            const uint8_t* site_ptr = hap_data.data() + offset;
            for (int s = 0; s < n_haps; ++s) {
                site_buffer[s] = (site_ptr[s >> 3] >> (s & 7)) & 1;
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown site type in get_site_column");
    }
}

/// Extract all site alleles for a single haplotype into a buffer
inline void get_hap_col(int hap_idx,
                        const HapMetadata& meta,
                        const HapData& hap_data,
                        std::vector<uint8_t>& hap_buffer)
{
    int n_sites = meta.n_sites;
    
    if ((int)hap_buffer.size() != n_sites) hap_buffer.assign(n_sites, 0);

    for (int site_idx = 0; site_idx < n_sites; ++site_idx) {
        SiteType stype = meta.site_types[site_idx];
        
        switch (stype) {
            case SiteType::MONO_REF:
                hap_buffer[site_idx] = 0;
                break;
            case SiteType::MONO_ALT:
                hap_buffer[site_idx] = 1;
                break;
            case SiteType::DENSE:
            {
                size_t offset = meta.site_offsets[site_idx];
                const uint8_t* site_ptr = hap_data.data() + offset;
                hap_buffer[site_idx] = (site_ptr[hap_idx >> 3] >> (hap_idx & 7)) & 1;
                break;
            }
            case SiteType::SPARSE:
            {
                size_t offset = meta.site_offsets[site_idx];
                const uint8_t* site_ptr = hap_data.data() + offset;
                hap_buffer[site_idx] = (site_ptr[hap_idx >> 3] >> (hap_idx & 7)) & 1;
                break;
            }
            default:
                throw std::runtime_error("Unknown site type in get_hap_col");
        }
    }
}

// ============================================================
// P-smoother integration helpers
// ============================================================

// Read VCF directly to unpacked 2D vector format (for P-smoother)
// Reads ALL biallelic sites (no minMac filtering) so P-smoother operates on full data
// Returns hap_2d[site][haplotype] = 0 or 1
// Also stores MAC per site in meta.site_mac for later filtering
HapMetadata read_vcf_to_2d(
    const std::string& vcf_file,
    std::vector<std::vector<uint8_t>>& hap_2d);

// Recompute MAC values from hap_2d after P-smoother has modified alleles
// Must be called before pack_haplotypes to ensure filtering uses post-smoothed counts
void recompute_mac(
    const std::vector<std::vector<uint8_t>>& hap_2d,
    HapMetadata& meta,
    int nthreads = 1);

// Pack 2D vector to bit-packed HapData format (after P-smoother)
// Applies minMac filter during packing - only sites with MAC >= minMac are kept
// Updates meta to reflect filtered sites
void pack_haplotypes(
    const std::vector<std::vector<uint8_t>>& hap_2d,
    HapData& hap_data,
    HapMetadata& meta,
    int minMac,
    int nthreads = 1);

// Write smoothed haplotypes to VCF (preserves original header and variant info)
void write_smoothed_vcf(
    const std::string& input_vcf,
    const std::string& output_vcf,
    const std::vector<std::vector<uint8_t>>& hap_2d,
    const HapMetadata& meta,
    int minMac);

#endif // VCF_UTILS_HPP
