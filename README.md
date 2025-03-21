# IBDBoost

#### to start, run the pipeline.sh script

./pipeline.sh <desired name stem> <AFR population size> <EUR population size> <maf frequency cutoff> <genotyping error> <genetic map> <nCPUs for ground truth extraction> <random seed>
ex: ./pipeline.sh test 200 200 0.05 0.001 data/genetic_map_GRCh38_chr20.txt 10 1

run src/findFalseIBDs.sh with the hap-IBD results and the hap-IBD + P_smoother results
format: src/findFalseIBDs.sh <reported output> <ground truth segments> <centimorgan cutoff> <threshold for falseness>
creates reported "true" and "false" segments, along with reporting accuracy, power, extraction

format: python src/constructDataset.py <unsmoothed vcf> <smoothed vcf> <ground truth segments> <"true" reported segments> <"false" reported segments> <n_chunks> <rate_map_file>  <output file name>

#### to get the boosted segments, run:
python src/segmentAugmentation.py <constructed_dataset


Dependencies
Python
- tskit
- msprime
- numpy
- tqdm
- stdpopsim
- pytorch
- XGBoost
- pandas

Other
- bcftools



