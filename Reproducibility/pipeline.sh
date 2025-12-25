#!/bin/bash
ts_name=$1
afr_pop_size=$2
eur_pop_size=$3
maf_cutoff=$4
genotyping_error=$5
rate_map_file=$6
n_cpus_gt_extraction=$7
random_seed=$8

# inputs needed
# tree sequence name, population size, maf cutoff, genotyping error, recombination rate map file, and the number of cpus you want to use for ground truth extraction



sim_std_out=$(python src/simulation.py $ts_name $afr_pop_size $eur_pop_size $maf_cutoff $genotyping_error $random_seed) # runs simulation, then captures output string containing the name of the processed file


# ------------------------------------
# there are a few different things printed from stdout, this captures the last thing printed (the processed vcf file)
delimiter=" "

# Save the original IFS
original_ifs=$IFS
# Set IFS to the delimiter
IFS=$delimiter

# Split the string and iterate over each part
for part in $sim_std_out; do
    last_part=$part
done

# Restore the original IFS
IFS=$original_ifs
# ------------------------------------
processed_vcf=$last_part 

default_cm_cutoff=1.7

python src/ibdInference.py $processed_vcf $rate_map_file # runs hap-IBD and P-smoother

python extractGT.py $ts_name $rate_map_file $default_cm_cutoff $n_cpus_gt_extraction # a little convoluted, but this is a python script to generate the shell script to extract the ground truth segments
./gt_extraction.sh

# now you have the ground truth segments, hap-ibd output, and hap-ibd + p-smoother output, and can use the files in the construction_dataset_files folder

# you may want to run this in background since the IBD extraction may take a little while
