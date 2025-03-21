import sys
from typing import List, Dict
import os
import csv
from tqdm import tqdm
import numpy as np
from convertHapMapToPLINK import convert

def loadVCFData(vcf_file_path: str):
    """loads in the vcf data into a matrix, creates a dictionary mapping the sample ID to the corresponding index in the vcf, and the list of variable sites

    Args:
        vcf_file_path (str): file path to VCF file to be loaded into memory

    Returns:
        tuple(List[List], Dict, list): returns VCF matrix, ids dictionary, and list of variable sites
    """
    f = open(vcf_file_path, 'r')
    vcf_data = f.readlines()
    f.close()
    i = 0 # going to use this to determine where to cutoff header, can be different based on what has been done via bcf tools
    while vcf_data[i][0:6] != "#CHROM":
        i += 1
    vcf_data = vcf_data[i:]
    vcf_data = [i.strip().split() for i in vcf_data]
    ids_dict = {vcf_data[0][i]: i for i in range(9, len(vcf_data[0]))} # used to determine which column index to access based on sample id
    sites_array = [int(vcf_data[i][1]) for i in range(1, len(vcf_data))]
    return vcf_data, ids_dict, sites_array


class Segment:
    # class to represent segment with some methods to help with splitting by site
    def __init__(self, indiv1: str, hap1:int , indiv2: str, hap2: int, start: int, end: int):
        self.indiv1 = indiv1
        self.hap1 = hap1
        self.indiv2 = indiv2
        self.hap2 = hap2
        self.start = start
        self.end = end
        self.len_cm = None
        self.physical_length = self.end - self.start
        self.sites = []
    
    def __binSearchForStart(self, bp_start: int, sites_array: List[int]) -> int:
        """
        binary search for starting point when iterating through vcf
        """
        l, r = 0, len(sites_array) - 1
        while l < r:
            mid = (l + r) // 2
            if sites_array[mid] == bp_start:
                return mid
            elif sites_array[mid] > bp_start:
                r = mid - 1
            else:
                l = mid + 1
        return l

    
    def getAllSegmentSites(self, vcf_data: List[List[str]], sites: List[int]):
        # this function finds all of the sites in the VCF that are within the bounds of the segment 
        m = self.__binSearchForStart(self.start, sites) + 1 # add one to skip first line fo vcf data
        segment_sites = {} # these indices correspond to the vcf matrix
        while m < len(vcf_data) - 1 and int(vcf_data[m][1]) <= self.end:
            segment_sites[m] = int(vcf_data[m][1])
            m += 1
        return segment_sites

    
    def __repr__(self):
        return f"sample1: {self.indiv1}\nhaplotype1: {self.hap1}\nsample2 {self.indiv2}\nhaplotype2: {self.hap2}\nstart: {self.start}\nend: {self.end}\ngenetic length: {self.len_cm}"
    

def processSegment(input_string: str) -> Segment:
    """
    create segment object from string
    """
    lst = input_string.strip().split()
    hap_id1 = int(lst[0])
    hap_id2 = int(lst[1])
    start = int(float(lst[2]))
    end = int(float(lst[3]))
    haplotype1 = hap_id1 % 2
    haplotype2 = hap_id2 % 2
    tsk_id1 = "tsk_" + str(hap_id1 // 2)
    tsk_id2 = "tsk_" + str(hap_id2 // 2)
    return Segment(tsk_id1, haplotype1, tsk_id2, haplotype2, start, end)

def generateSegmentChunks(seg: Segment, n_chunks: int, vcf: List[List], sites: List[int], offset: int):
    """_summary_

    Args:
        seg (Segment): segment object
        n_chunks (int): number of desired chunks
        vcf (List[List]): VCF Matrix
        sites (List[int]): list of variable sites in VCF
        offset (int): index of the first site in the segment

    Returns:
        List[tuple[int]]: list of chunked site indices, stored as a tuple with the first site and last site within
                          the chunk as a non-inclusive interval
    """
    n_sites = len(seg.getAllSegmentSites(vcf, sites))
    starts = [i * (n_sites // (n_chunks - 1)) + offset for i in range(n_chunks)]
    exclusive_chunk_intervals = []
    for i in range(len(starts) - 1):
        exclusive_chunk_intervals.append((starts[i], starts[i + 1]))
    exclusive_chunk_intervals.append((starts[-1], min(n_sites + offset, sites[-1] - 1)))
    return exclusive_chunk_intervals

def getMismatches(segment: Segment, vcf_data: List[List], site_interval: "tuple[int]", ids_dict: Dict):
    n_mismatches = 0
    sample_idx1 = ids_dict[segment.indiv1]
    sample_idx2 = ids_dict[segment.indiv2]
    for i in range(site_interval[0], site_interval[1]):
        allele1 = vcf_data[i][sample_idx1].split("|")[segment.hap1]
        allele2 = vcf_data[i][sample_idx2].split("|")[segment.hap2]
        if allele1 != allele2:
            n_mismatches += 1
    return n_mismatches

def getCorrections(segment: Segment, vcf_data_unsmooth: List[List], vcf_data_smooth: List[List], site_interval: "tuple[int]", ids_dict: Dict):
    n_corrections = 0
    sample_idx1 = ids_dict[segment.indiv1]
    sample_idx2 = ids_dict[segment.indiv2]
    for i in range(site_interval[0], site_interval[1]):
        vcf_val1 = vcf_data_unsmooth[i][sample_idx1].split("|")[segment.hap1]
        vcf_val2 = vcf_data_unsmooth[i][sample_idx2].split("|")[segment.hap2]
        vcf_val_s1 = vcf_data_smooth[i][sample_idx1].split("|")[segment.hap1]
        vcf_val_s2 = vcf_data_smooth[i][sample_idx2].split("|")[segment.hap2]
        if vcf_val1 != vcf_val_s1:
            n_corrections += 1
        if vcf_val2 != vcf_val_s2:
            n_corrections += 1
    return n_corrections

def generateFeatureNames(n_chunks):
    f_names = ["id1", "hap1", "id2", "hap2", "start", "end"]
    for i in range(n_chunks):
        f_names.append(f"phys_len_{i}")
        f_names.append(f"gen_len_{i}")
        f_names.append(f"n_mismatches_{i}")
        f_names.append(f"n_corrections_{i}")
    f_names.append("classification")
    return f_names

def getGeneticPostion(site, rate_map_dict):
    return rate_map_dict[site]

def createRateMapDict(rate_map_fp):
    f = open(rate_map_fp)
    data = f.readlines()
    f.close()
    rate_map_dict = {}
    for line in data:
        line = line.strip().split()
        rate_map_dict[int(line[0])] = float(line[1])
    return rate_map_dict

def interpolate_plink_rate_map(vcf_file, plink_rate_map_file):
        _started = False
        f = open(vcf_file)
        sites = []
        for line in f.readlines():
            if _started:
                vals = line.split("\t")
                sites.append(int(vals[1]))
            elif ("#CHROM" in line[0:1000]):
                _started = True
        f.close()
        f = open(plink_rate_map_file)
        output_file = os.path.splitext(plink_rate_map_file)[0] + ".interpolated" + os.path.splitext(plink_rate_map_file)[1]
        f_o = open(output_file, 'w+')
        xp = []
        yp = []
        lines = f.readlines()
        #check rate map type (HapMapII or PLINK)
        first = lines[0].split()
        if first[0][0:3] == "chr":
            for line in lines:
                vals = line.split()
                xp.append(int(vals[1]))
                yp.append(float(vals[3]))
        else:
            for line in lines:
                vals = line.split()
                xp.append(int(vals[3]))
                yp.append(float(vals[2]))
        _output_vals = np.interp(sites, xp, yp)
        for i in range(len(_output_vals)):
            f_o.write(str(i) + "\t" + str(_output_vals[i]) + "\n")
        f.close()
        f_o.close()
        return output_file

def checkRateMapType(file_path):
    f = open(file_path)
    first = f.readline().strip().split("\t")
    f.close()
    if first[0][0:3] == "chr":
        return "HapMap"
    else:
        return "PLINK"

def main():
    vcf_fp = sys.argv[1]
    vcf_smooth_fp = sys.argv[2]
    gt_segments_fp = sys.argv[3]
    true_reported_segments_fp = sys.argv[4]
    false_reported_segments_fp = sys.argv[5]
    n_chunks = int(sys.argv[6])
    rate_map_file = sys.argv[7]
    output_file = sys.argv[8]


    rate_map_is_plink = True
    if checkRateMapType(rate_map_file) == "HapMap":
        rate_map_is_plink = False
    if rate_map_is_plink == False:
        rate_map_file = convert(rate_map_file)
    rate_map_file = interpolate_plink_rate_map(vcf_fp, rate_map_file)
    rate_map_dict = createRateMapDict(rate_map_file)
    
    
    vcf, ids_dict, sites = loadVCFData(vcf_fp)
    vcf_s, _, _ = loadVCFData(vcf_smooth_fp)
    field_names = generateFeatureNames(n_chunks)
    with open(output_file, "w+", newline = "\n") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(field_names)
        f = open(gt_segments_fp)
        for line in tqdm(f.readlines()):
            seg = processSegment(line)
            segment_sites = seg.getAllSegmentSites(vcf_s, sites)
            offset = list(segment_sites.keys())[0]
            seg_intervals = generateSegmentChunks(seg, n_chunks, vcf_s, sites, offset)
            data = [seg.indiv1, seg.hap1, seg.indiv2, seg.hap2, seg.start, seg.end]
            
            for interval in seg_intervals:
                if interval[0] == interval[1]: # check for empty interval, should only happen at the end of a segment if the number of sites is divisible by n_chunks - 1 
                    for i in range(4):
                        data.append(0)
                    continue
                physical_start = int(vcf[interval[0]][1])
                physical_end = int(vcf[interval[1]][1])
                genetic_start = getGeneticPostion(interval[0], rate_map_dict)
                genetic_end = getGeneticPostion(interval[1] - 1, rate_map_dict)
                physical_length = physical_end - physical_start
                genetic_length = genetic_end - genetic_start
                n_mismatches = getMismatches(seg, vcf_s, interval, ids_dict)
                n_corrections = getCorrections(seg, vcf, vcf_s, interval, ids_dict)
                data.append(physical_length)
                data.append(genetic_length)
                data.append(n_mismatches)
                data.append(n_corrections)
            data.append(1)
            writer.writerow(data)
            
        f.close()
        f = open(true_reported_segments_fp)
        for line in tqdm(f.readlines()):
            seg = processSegment(line)
            segment_sites = seg.getAllSegmentSites(vcf_s, sites)
            offset = list(segment_sites.keys())[0]
            seg_intervals = generateSegmentChunks(seg, n_chunks, vcf_s, sites, offset)
            data = [seg.indiv1, seg.hap1, seg.indiv2, seg.hap2, seg.start, seg.end]
            for interval in seg_intervals:
                if interval[0] == interval[1]:
                    for i in range(4):
                        data.append(0)
                    continue
                physical_start = int(vcf[interval[0]][1])
                physical_end = int(vcf[interval[1]][1])
                genetic_start = getGeneticPostion(interval[0], rate_map_dict)
                genetic_end = getGeneticPostion(interval[1] - 1, rate_map_dict)
                physical_length = physical_end - physical_start
                genetic_length = genetic_end - genetic_start
                n_mismatches = getMismatches(seg, vcf_s, interval, ids_dict)
                n_corrections = getCorrections(seg, vcf, vcf_s, interval, ids_dict)
                data.append(physical_length)
                data.append(genetic_length)
                data.append(n_mismatches)
                data.append(n_corrections)
            data.append(2)
            writer.writerow(data)
        f.close()
        f = open(false_reported_segments_fp)
        for line in tqdm(f.readlines()):
            seg = processSegment(line)
            segment_sites = seg.getAllSegmentSites(vcf_s, sites)
            offset = list(segment_sites.keys())[0]
            seg_intervals = generateSegmentChunks(seg, n_chunks, vcf_s, sites, offset)
            data = [seg.indiv1, seg.hap1, seg.indiv2, seg.hap2, seg.start, seg.end]
            for interval in seg_intervals:
                if interval[0] == interval[1]:
                    for i in range(4):
                        data.append(0)
                    continue
                physical_start = int(vcf[interval[0]][1])
                physical_end = int(vcf[interval[1]][1])
                genetic_start = getGeneticPostion(interval[0], rate_map_dict)
                genetic_end = getGeneticPostion(interval[1] - 1, rate_map_dict)
                physical_length = physical_end - physical_start
                genetic_length = genetic_end - genetic_start
                n_mismatches = getMismatches(seg, vcf_s, interval, ids_dict)
                n_corrections = getCorrections(seg, vcf, vcf_s, interval, ids_dict)
                data.append(physical_length)
                data.append(genetic_length)
                data.append(n_mismatches)
                data.append(n_corrections)
            data.append(3)
            writer.writerow(data)
        f.close()

if __name__ == "__main__":
    main()



    
    

    
