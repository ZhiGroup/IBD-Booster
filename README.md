# IBD-Booster

Fast IBD segment detection from phased VCF files using PBWT with integrated P-smoother error correction.

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

## Parameters

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--threads=N` | Number of threads | 1 |
| `--ps-length=N` | P-smoother block length | 20 |
| `--ps-width=N` | P-smoother minimum block width | 20 |
| `--ps-gap=N` | P-smoother gap size | 1 |
| `--ps-rho=F` | P-smoother error rate threshold | 0.05 |

### IBD Detection Parameters (defaults)

| Parameter | Description | Default |
|-----------|-------------|---------|
| min-seed | Minimum cM length of seed IBS segment | 2.0 |
| max-gap | Maximum base-pair gap for extending seed segments | 1000 |
| min-extend | Minimum cM length of IBS segment that can extend a seed | min(1.0, min-seed) |
| min-markers | Minimum number of markers in seed/extension segments | 100 |
| min-mac | Minimum minor allele count (markers below threshold excluded) | 2 |

## Output

Writes IBD segments to stdout in tab-separated format.
