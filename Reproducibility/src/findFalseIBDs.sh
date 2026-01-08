#!/bin/bash
hap_input=$1
gt_segments_input=$2
cm_cutoff=$3
false_ibd_cutoff=$4


# finds segments that have more and less than {false_ibd_cutoff} overlap with a ground truth segment
python src/findFalseIBD.py $hap_input $gt_segments_input $cm_cutoff $false_ibd_cutoff
python src/calculateStatistics.py $hap_input $gt_segments_input $cm_cutoff
