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

/// Bit-packed haplotype data: one bit per haplotype per site
using HapData = std::vector<uint8_t>;

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
    HapMetadata& meta);

// Pack 2D vector to bit-packed HapData format (after P-smoother)
// Applies minMac filter during packing - only sites with MAC >= minMac are kept
// Updates meta to reflect filtered sites
void pack_haplotypes(
    const std::vector<std::vector<uint8_t>>& hap_2d,
    HapData& hap_data,
    HapMetadata& meta,
    int minMac);

// Write smoothed haplotypes to VCF (preserves original header and variant info)
void write_smoothed_vcf(
    const std::string& input_vcf,
    const std::string& output_vcf,
    const std::vector<std::vector<uint8_t>>& hap_2d,
    const HapMetadata& meta,
    int minMac);

#endif // VCF_UTILS_HPP
