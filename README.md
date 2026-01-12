# IBD-Booster
Fast IBD segment detection from phased VCF files, integrating P-smoother error correction and seed-and-extend IBD calling following Browning et al. (hap-IBD), implemented in C++. IBD-Booster uses the positional Burrows-Wheeler transform (PBWT) algorithm for efficient IBD detection.

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

## Usage

```bash
./IBD-Booster <input.vcf.gz> <genetic_map.map> [options]
```

## Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--threads=N` | Number of threads | 1 |
| `--min-output=F` | Minimum cM length for IBD output segments | 2.0 |
| `--no-psmoother` | Skip P-smoother (for pre-smoothed files) | disabled |
| `--ps-length=N` | P-smoother block length | 20 |
| `--ps-width=N` | P-smoother minimum block width | 20 |
| `--ps-gap=N` | P-smoother gap size | 1 |
| `--ps-rho=F` | P-smoother error rate threshold | 0.05 |

## Output Format

IBD segments are written to `segments2.tsv` in tab-separated format:

| Column | Description |
|--------|-------------|
| 1 | Sample 1 index (0-based) |
| 2 | Sample 1 haplotype (0 or 1) |
| 3 | Sample 2 index (0-based) |
| 4 | Sample 2 haplotype (0 or 1) |
| 5 | Segment start position (bp) |
| 6 | Segment end position (bp) |
| 7 | Segment length (cM) |

Example output:
```
0	0	1	1	10000	50000	2.500000
0	1	2	0	15000	75000	3.200000
```

This indicates:
- Sample 0 haplotype 0 shares IBD with Sample 1 haplotype 1 from bp 10000-50000 (2.5 cM)
- Sample 0 haplotype 1 shares IBD with Sample 2 haplotype 0 from bp 15000-75000 (3.2 cM)





