#include "vcf_utils.hpp"
#include <htslib/vcf.h>
#include <htslib/hts.h>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <thread>

// Simple parallel_for using std::thread (replaces OpenMP to avoid thread pool conflicts)
// Main thread participates in work to use exactly nthreads total
template<typename Func>
static void parallel_for(int start, int end, int nthreads, Func&& func) {
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


HapMetadata read_vcf(const std::string& vcf_file,
                     const std::string& bin_file,
                     int minMac,
                     HapData& hap_data)
{
    htsFile* fp = hts_open(vcf_file.c_str(), "r");
    if (!fp) throw std::runtime_error("Cannot open VCF: " + vcf_file);

    bcf_hdr_t* hdr = vcf_hdr_read(fp);
    if (!hdr) {
        hts_close(fp);
        throw std::runtime_error("Cannot read VCF header: " + vcf_file);
    }

    int n_samples = bcf_hdr_nsamples(hdr);
    if (n_samples <= 0) {
        bcf_hdr_destroy(hdr);
        hts_close(fp);
        throw std::runtime_error("VCF contains no samples.");
    }
    std::vector<std::string> sample_ids;
    for (int i = 0; i < n_samples; ++i) {
        const char* name = hdr->samples[i];
        if (name)
            sample_ids.emplace_back(name);
        else
            sample_ids.emplace_back("UNKNOWN");
    }

    bcf1_t* rec = bcf_init();
    int32_t* gt_arr = nullptr;
    int ngt_arr = 0;
    int n_haps = n_samples * 2;
    int dense_stride = (n_haps + 7) / 8;

    HapMetadata meta;
    meta.sampleIDs = std::move(sample_ids);
    meta.n_samples = n_samples;
    meta.n_haps = n_haps;
    meta.dense_stride = dense_stride;

    // bin_file parameter kept for API compatibility but no longer written
    (void)bin_file;

    std::vector<uint8_t> site_buffer(dense_stride, 0);
    int32_t n_sites = 0;
    meta.isDiploid.assign(n_samples, false);

    while (bcf_read(fp, hdr, rec) == 0) {
        bcf_unpack(rec, BCF_UN_STR);
        if (rec->n_allele > 2) continue; // skip multiallelic

        int ngt = bcf_get_genotypes(hdr, rec, &gt_arr, &ngt_arr);
        if (ngt < 0) continue;

        // Single pass: count alleles AND pack bits simultaneously
        std::fill(site_buffer.begin(), site_buffer.end(), 0);
        int ref_count = 0, alt_count = 0;

        for (int s = 0; s < n_samples; ++s) {
            int idx0 = 2*s, idx1 = 2*s + 1;
            int g0 = bcf_gt_is_missing(gt_arr[idx0]) ? 0 : bcf_gt_allele(gt_arr[idx0]);
            int g1 = bcf_gt_is_missing(gt_arr[idx1]) ? 0 : bcf_gt_allele(gt_arr[idx1]);

            // Count alleles
            if (g0 == 0) ++ref_count; else ++alt_count;
            if (g1 == 0) ++ref_count; else ++alt_count;

            // Pack bits (do this always - we'll discard if filtered)
            if (g0 != 0) site_buffer[(2*s) >> 3] |= (1u << ((2*s) & 7));
            if (g1 != 0) site_buffer[(2*s + 1) >> 3] |= (1u << ((2*s + 1) & 7));

            // Mark diploid
            if (!meta.isDiploid[s] && (bcf_gt_is_phased(gt_arr[idx1]) || !bcf_gt_is_missing(gt_arr[idx1]))) {
                meta.isDiploid[s] = true;
            }
        }

        // Minimum allele count filter
        int mac = std::min(ref_count, alt_count);
        if (mac < minMac) continue;

        // Determine site type
        SiteType stype;
        if (alt_count == 0) stype = SiteType::MONO_REF;
        else if (ref_count == 0) stype = SiteType::MONO_ALT;
        else if (alt_count <= n_samples / 32) stype = SiteType::SPARSE;
        else stype = SiteType::DENSE;

        meta.site_types.push_back(stype);
        meta.vcf_bp_positions.push_back(rec->pos + 1);
        meta.site_offsets.push_back(hap_data.size());

        // Store site into hap_data (bit-packed for DENSE and SPARSE; no bytes for MONO_*)
        if (stype == SiteType::DENSE || stype == SiteType::SPARSE) {
            // site_buffer already populated above
            hap_data.insert(hap_data.end(), site_buffer.begin(), site_buffer.end());
        }
        // MONO_REF and MONO_ALT: no payload needed

        ++n_sites;
    }

    bcf_destroy(rec);
    free(gt_arr);
    bcf_hdr_destroy(hdr);
    hts_close(fp);

    meta.n_sites = n_sites;

    std::cerr << "[INFO] Read VCF complete: " << n_samples
              << " samples, " << n_sites << " sites.\n";

    return meta;
}

GeneticMap readGeneticMap(const std::string &filename) {
    GeneticMap map;
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open genetic map file: " + filename);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string chrom, strand;
        double cm;
        int bp;
        if (!(iss >> chrom >> strand >> cm >> bp)) {
            throw std::runtime_error("Malformed line in genetic map: " + line);
        }
        map.bp_vec.push_back(bp);
        map.cm_vec.push_back(cm);
    }
    return map;
}

std::vector<double> interpolateGeneticPositions(
        const GeneticMap &map,
        const std::vector<int> &vcf_bp_positions)
{
    std::vector<double> genPos;
    genPos.reserve(vcf_bp_positions.size());

    const double minEndCmDist = 5.0;  // Minimum cM window for extrapolation (matches HapIBD)
    int mapSizeM1 = map.bp_vec.size() - 1;

    for (int site_bp : vcf_bp_positions) {
        auto it = std::lower_bound(map.bp_vec.begin(), map.bp_vec.end(), site_bp);
        int index = std::distance(map.bp_vec.begin(), it);

        if (it != map.bp_vec.end() && *it == site_bp) {
            // Exact match - use map value directly
            genPos.push_back(map.cm_vec[index]);
        } else if (index == 0) {
            // Position is BEFORE first map entry - extrapolate using min 5 cM window
            // Find a point at least minEndCmDist away from first point
            int bIndex = 1;
            for (int i = 1; i <= mapSizeM1; ++i) {
                if (map.cm_vec[i] - map.cm_vec[0] >= minEndCmDist) {
                    bIndex = i;
                    break;
                }
                bIndex = std::min(i, mapSizeM1);
            }
            int aIndex = 0;
            double x = site_bp;
            double a = map.bp_vec[aIndex];
            double b = map.bp_vec[bIndex];
            double fa = map.cm_vec[aIndex];
            double fb = map.cm_vec[bIndex];
            double y = fa + ((double)(x - a) / (double)(b - a)) * (fb - fa);
            genPos.push_back(y);
        } else if (index == map.bp_vec.size()) {
            // Position is AFTER last map entry - extrapolate using min 5 cM window
            // Find a point at least minEndCmDist before the last point
            int aIndex = mapSizeM1 - 1;
            for (int i = mapSizeM1 - 1; i >= 0; --i) {
                if (map.cm_vec[mapSizeM1] - map.cm_vec[i] >= minEndCmDist) {
                    aIndex = i;
                    break;
                }
                aIndex = std::max(i, 0);
            }
            int bIndex = mapSizeM1;
            double x = site_bp;
            double a = map.bp_vec[aIndex];
            double b = map.bp_vec[bIndex];
            double fa = map.cm_vec[aIndex];
            double fb = map.cm_vec[bIndex];
            double y = fa + ((double)(x - a) / (double)(b - a)) * (fb - fa);
            genPos.push_back(y);
        } else {
            // Normal interpolation between two map positions
            int aIndex = index - 1;
            int bIndex = index;
            double x = site_bp;
            double a = map.bp_vec[aIndex];
            double b = map.bp_vec[bIndex];
            double fa = map.cm_vec[aIndex];
            double fb = map.cm_vec[bIndex];
            double y = fa + ((double)(x - a) / (double)(b - a)) * (fb - fa);
            genPos.push_back(y);
        }
    }
    return genPos;
}

// ============================================================
// P-smoother integration helpers
// ============================================================

// Fast custom line reader - avoids std::getline overhead
static char* fast_getline(FILE* fp, char* buf, size_t buf_size, size_t& line_len) {
    size_t pos = 0;
    int c;
    while ((c = fgetc_unlocked(fp)) != EOF && c != '\n') {
        if (pos < buf_size - 1) buf[pos++] = (char)c;
    }
    if (pos == 0 && c == EOF) return nullptr;
    buf[pos] = '\0';
    line_len = pos;
    return buf;
}

HapMetadata read_vcf_to_2d(
    const std::string& vcf_file,
    std::vector<std::vector<uint8_t>>& hap_2d)
{
    // Check if file is gzipped
    bool is_gzipped = (vcf_file.size() >= 3 &&
                       vcf_file.compare(vcf_file.size() - 3, 3, ".gz") == 0);

    FILE* fp = nullptr;
    if (is_gzipped) {
        std::string cmd = "zcat '" + vcf_file + "'";
        fp = popen(cmd.c_str(), "r");
    } else {
        fp = fopen(vcf_file.c_str(), "r");
    }
    if (!fp) throw std::runtime_error("Cannot open VCF: " + vcf_file);

    // Large read buffer for FILE*
    std::vector<char> file_buf(32 * 1024 * 1024);
    setvbuf(fp, file_buf.data(), _IOFBF, file_buf.size());

    // Line buffer - large enough for ~500K samples * 4 chars each
    size_t line_buf_size = 4 * 1024 * 1024;  // 4MB initial
    std::vector<char> line_buf(line_buf_size);

    HapMetadata meta;
    size_t line_len = 0;

    // Read header lines
    bool have_first_data_line = false;
    while (fast_getline(fp, line_buf.data(), line_buf_size, line_len)) {
        if (line_len == 0) continue;
        char* line = line_buf.data();

        if (line[0] == '#') {
            meta.vcf_header.append(line, line_len);
            meta.vcf_header += '\n';

            if (line_len > 1 && line[1] != '#') {
                // #CHROM line - count samples by counting tabs after column 9
                int tab_count = 0;
                char* p = line;
                char* sample_start = nullptr;

                while (*p) {
                    if (*p == '\t') {
                        if (tab_count >= 9 && sample_start) {
                            meta.sampleIDs.emplace_back(sample_start, p - sample_start);
                        }
                        ++tab_count;
                        sample_start = p + 1;
                    }
                    ++p;
                }
                // Last sample (no trailing tab)
                if (tab_count >= 9 && sample_start && *sample_start) {
                    meta.sampleIDs.emplace_back(sample_start);
                }
            }
        } else {
            have_first_data_line = true;  // First data line is in buffer
            break;
        }
    }

    int n_samples = static_cast<int>(meta.sampleIDs.size());
    if (n_samples <= 0) throw std::runtime_error("VCF contains no samples.");

    int n_haps = n_samples * 2;
    int dense_stride = (n_haps + 7) / 8;

    meta.n_samples = n_samples;
    meta.n_haps = n_haps;
    meta.dense_stride = dense_stride;
    meta.isDiploid.assign(n_samples, true);  // Assume all diploid for phased VCF

    // Reallocate line buffer if needed for large sample counts
    size_t needed_size = (size_t)n_samples * 4 + 4096;
    if (needed_size > line_buf_size) {
        line_buf_size = needed_size;
        line_buf.resize(line_buf_size);
    }

    // Reserve space
    hap_2d.clear();
    hap_2d.reserve(20000);
    meta.vcf_fixed_fields.reserve(20000);
    meta.site_mac.reserve(20000);
    meta.site_types.reserve(20000);
    meta.vcf_bp_positions.reserve(20000);

    int32_t n_sites = 0;

    // Process first data line (already read) and subsequent lines
    if (!have_first_data_line) {
        if (is_gzipped) pclose(fp); else fclose(fp);
        meta.n_sites = 0;
        std::cerr << "[INFO] Read VCF to 2D complete: " << n_samples
                  << " samples, 0 sites (no data lines found).\n";
        return meta;
    }

    do {
        if (line_len == 0) continue;
        char* line = line_buf.data();

        // Find the 9 tabs to separate fixed fields from genotypes
        int tab_count = 0;
        char* gt_start = nullptr;
        char* alt_start = nullptr;
        char* alt_end = nullptr;
        char* pos_start = nullptr;
        char* pos_end = nullptr;

        char* p = line;
        while (*p && tab_count < 9) {
            if (*p == '\t') {
                ++tab_count;
                if (tab_count == 1) pos_start = p + 1;
                else if (tab_count == 2) pos_end = p;
                else if (tab_count == 4) alt_start = p + 1;
                else if (tab_count == 5) alt_end = p;
                else if (tab_count == 9) gt_start = p + 1;
            }
            ++p;
        }

        if (tab_count < 9 || !gt_start) continue;

        // NOTE: Original P-smoother does NOT filter multiallelic sites - it processes all sites
        // and treats any non-'0' allele as '1'. We do the same to match the original behavior.
        // Multiallelic filtering happens later during pack_haplotypes with minMac.

        // Store fixed fields (up to and including the tab after FORMAT)
        meta.vcf_fixed_fields.emplace_back(line, gt_start - line);

        // Parse position
        int bp_pos = 0;
        for (char* c = pos_start; c < pos_end; ++c) {
            bp_pos = bp_pos * 10 + (*c - '0');
        }
        meta.vcf_bp_positions.push_back(bp_pos);

        // Pre-allocate site alleles vector
        hap_2d.emplace_back(n_haps);
        std::vector<uint8_t>& site_alleles = hap_2d.back();

        // Parse genotypes - optimized for "X|Y\t" pattern
        int ref_count = 0, alt_count = 0;
        p = gt_start;

        for (int s = 0; s < n_samples; ++s) {
            // Fast parse: expect "X|Y" or "X/Y" format
            char c0 = *p;
            int g0 = (c0 != '0' && c0 != '.') ? 1 : 0;
            p += 2;  // Skip allele and separator
            char c1 = *p;
            int g1 = (c1 != '0' && c1 != '.') ? 1 : 0;

            site_alleles[2*s] = g0;
            site_alleles[2*s + 1] = g1;

            ref_count += (2 - g0 - g1);
            alt_count += (g0 + g1);

            // Skip to next sample
            while (*p && *p != '\t') ++p;
            if (*p) ++p;
        }

        // Store MAC
        meta.site_mac.push_back(std::min(ref_count, alt_count));

        // Determine site type
        SiteType stype;
        if (alt_count == 0) stype = SiteType::MONO_REF;
        else if (ref_count == 0) stype = SiteType::MONO_ALT;
        else if (alt_count <= n_samples / 32) stype = SiteType::SPARSE;
        else stype = SiteType::DENSE;
        meta.site_types.push_back(stype);

        ++n_sites;

    } while (fast_getline(fp, line_buf.data(), line_buf_size, line_len));

    if (is_gzipped) pclose(fp); else fclose(fp);

    meta.n_sites = n_sites;

    return meta;
}

// Recompute MAC values from hap_2d after P-smoother has modified alleles
// This ensures filtering uses post-smoothed allele counts
void recompute_mac(
    const std::vector<std::vector<uint8_t>>& hap_2d,
    HapMetadata& meta,
    int nthreads)
{
    int n_sites = static_cast<int>(hap_2d.size());
    int n_haps = meta.n_haps;

    if (n_sites != meta.n_sites) {
        std::cerr << "[WARNING] recompute_mac: hap_2d size (" << n_sites
                  << ") != meta.n_sites (" << meta.n_sites << ")\n";
    }

    meta.site_mac.resize(n_sites);

    parallel_for(0, n_sites, nthreads, [&](int site) {
        int alt_count = 0;
        for (int h = 0; h < n_haps; ++h) {
            if (hap_2d[site][h] != 0) ++alt_count;
        }
        int ref_count = n_haps - alt_count;
        meta.site_mac[site] = std::min(ref_count, alt_count);
    });
}

void pack_haplotypes(
    const std::vector<std::vector<uint8_t>>& hap_2d,
    HapData& hap_data,
    HapMetadata& meta,
    int minMac,
    int nthreads)
{
    int n_sites_all = meta.n_sites;
    int n_haps = meta.n_haps;
    int dense_stride = meta.dense_stride;

    // Step 1: Filter sites by minMac and build new metadata
    std::vector<int> filtered_indices;  // Maps filtered index -> original index
    std::vector<int> new_bp_positions;
    std::vector<SiteType> new_site_types;

    for (int site = 0; site < n_sites_all; ++site) {
        if (meta.site_mac[site] >= minMac) {
            filtered_indices.push_back(site);
            new_bp_positions.push_back(meta.vcf_bp_positions[site]);
            new_site_types.push_back(meta.site_types[site]);
        }
    }

    int n_sites_filtered = static_cast<int>(filtered_indices.size());

    // Step 2: Update metadata with filtered sites
    meta.n_sites = n_sites_filtered;
    meta.vcf_bp_positions = std::move(new_bp_positions);
    meta.site_types = std::move(new_site_types);
    meta.site_mac.clear();  // No longer needed after filtering
    meta.site_mac.shrink_to_fit();

    // Step 3: Pre-compute site offsets and total size for filtered sites
    meta.site_offsets.resize(n_sites_filtered);
    size_t total_size = 0;
    for (int i = 0; i < n_sites_filtered; ++i) {
        meta.site_offsets[i] = total_size;
        SiteType stype = meta.site_types[i];
        if (stype == SiteType::DENSE || stype == SiteType::SPARSE) {
            total_size += dense_stride;
        }
    }

    // Step 4: Pre-allocate hap_data
    hap_data.clear();
    hap_data.resize(total_size, 0);

    // Step 5: Pack filtered sites in parallel
    parallel_for(0, n_sites_filtered, nthreads, [&](int i) {
        int orig_site = filtered_indices[i];
        SiteType stype = meta.site_types[i];
        if (stype == SiteType::DENSE || stype == SiteType::SPARSE) {
            size_t offset = meta.site_offsets[i];
            // Pack bits directly into hap_data
            for (int h = 0; h < n_haps; ++h) {
                if (hap_2d[orig_site][h] != 0) {
                    hap_data[offset + (h >> 3)] |= (1u << (h & 7));
                }
            }
        }
    });
}

// Write smoothed haplotypes to VCF - OPTIMIZED VERSION
// Uses pre-stored header and fixed fields - no VCF re-read needed!
// Writes ALL biallelic sites (no minMac filter) to match original P-smoother behavior
void write_smoothed_vcf(
    const std::string& input_vcf,
    const std::string& output_vcf,
    const std::vector<std::vector<uint8_t>>& hap_2d,
    const HapMetadata& meta,
    int minMac)
{
    (void)input_vcf;  // No longer needed - we use stored data
    (void)minMac;     // No longer used - write all sites

    int n_samples = meta.n_samples;
    int n_sites = static_cast<int>(hap_2d.size());

    // Check that we have stored VCF data
    if (meta.vcf_header.empty() || meta.vcf_fixed_fields.size() != (size_t)n_sites) {
        throw std::runtime_error("VCF header/fixed fields not stored in metadata. Cannot write smoothed VCF.");
    }

    // Pre-allocate genotype buffer: "0|0\t0|1\t1|0\t...\n"
    // Each sample takes 4 chars (X|X + tab or newline)
    size_t gt_buf_size = (size_t)n_samples * 4;
    std::vector<char> gt_buffer(gt_buf_size);

    // Pre-fill with tab separators and pipe characters
    for (int s = 0; s < n_samples; ++s) {
        size_t pos = (size_t)s * 4;
        gt_buffer[pos + 1] = '|';
        gt_buffer[pos + 3] = (s < n_samples - 1) ? '\t' : '\n';
    }

    // Open output with FILE* for faster raw writes
    FILE* out_fp = fopen(output_vcf.c_str(), "w");
    if (!out_fp) throw std::runtime_error("Cannot open output VCF: " + output_vcf);

    // Set large write buffer (64MB)
    std::vector<char> out_buffer(64 * 1024 * 1024);
    setvbuf(out_fp, out_buffer.data(), _IOFBF, out_buffer.size());

    // Write header (already includes newlines)
    fwrite(meta.vcf_header.data(), 1, meta.vcf_header.size(), out_fp);

    // Write each site
    for (int site_idx = 0; site_idx < n_sites; ++site_idx) {
        // Write fixed fields (already includes trailing tab)
        const std::string& fixed = meta.vcf_fixed_fields[site_idx];
        fwrite(fixed.data(), 1, fixed.size(), out_fp);

        // Build genotype string in buffer (fast: just fill in allele chars)
        const std::vector<uint8_t>& site_data = hap_2d[site_idx];
        for (int s = 0; s < n_samples; ++s) {
            size_t pos = (size_t)s * 4;
            gt_buffer[pos] = '0' + site_data[2*s];
            gt_buffer[pos + 2] = '0' + site_data[2*s + 1];
        }

        // Write entire genotype buffer at once
        fwrite(gt_buffer.data(), 1, gt_buf_size, out_fp);
    }

    fclose(out_fp);

    std::cerr << "[P-smoother] Wrote smoothed VCF: " << output_vcf
              << " (" << n_sites << " sites)\n";
}