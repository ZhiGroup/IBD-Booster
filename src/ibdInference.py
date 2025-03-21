import os
import sys
import subprocess
import time
import numpy as np
from convertHapMapToPLINK import convert
from PSmootherEvaluation import *




class hap_ibd:
    def __init__(self, path_to_jar, min_seed = 2.0, max_gap = 1000, min_extend = 1.0, min_output = 2.0, min_markers = 100, min_mac = 2, nthreads = 12, exclude_samples = None):
        self.path_to_jar = path_to_jar

        self.min_seed = min_seed
        self.max_gap = max_gap
        self.min_extend = min_extend
        self.min_output = min_output
        self.min_markers = min_markers
        self.min_mac = min_mac
        self.nthreads = nthreads
        self.exclude_samples = exclude_samples
        self.out = None
    def run(self, vcf_file, plink_rate_map_file, out):
        """pass in the path to the vcf file, the path to the PLINK rate map file, and the desired name for the ibd output files

        """
        self.out = out
        command = ["java", "-jar", f"{self.path_to_jar}", f"gt={vcf_file}", f"map={plink_rate_map_file}", f"out={out}", f"min-seed={self.min_seed}",
            f"max-gap={self.max_gap}", f"min-extend={self.min_extend}", f"min-output={self.min_output}", f"min-markers={self.min_markers}",
            f"min-mac={self.min_mac}", f"nthreads={self.nthreads}"]
        if self.exclude_samples != None:
            command.append(f"excludesamples={self.exclude_samples}")
        subprocess.run(command)

class p_smoother:
    def __init__(self, path_to_shell_script, length = 20, width = 20, gap = 1, rho = 0.05, checkpoint = 100000):
        self.path_to_shell_script = path_to_shell_script
        self.length = length
        self.width = width
        self.gap = gap
        self.rho = rho
        self.checkpoint = checkpoint

    def run(self, vcf_file, plink_rate_map_file):
        interpolated_map = self.interpolate_plink_rate_map(vcf_file, plink_rate_map_file)
        command = [f"./{self.path_to_shell_script}", "--inputVCF", f"{vcf_file}", 
                   "--map", f"{interpolated_map}", 
                   "--length", f"{self.length}", 
                   "--width", f"{self.width}", 
                   "--rho", f"{self.rho}", 
                   "--gap", f"{self.gap}", 
                   f"--checkpoint", f"{self.checkpoint}"]
        subprocess.run(command)
        command2 = ["rm", f"{interpolated_map}"]
        subprocess.run(command2)

    def interpolate_plink_rate_map(self, vcf_file, plink_rate_map_file):
        _started = False
        s = time.time()
        f = open(vcf_file)
        sites = []
        for line in f:
            if _started:
                vals = line[0:1000].split()
                sites.append(int(vals[1]))
            elif ("#CHROM" in line[0:1000]):
                _started = True
        f.close()
        e = time.time()
        print(f"{e - s} seconds to read recombination rate map")
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
        s = time.time()
        _output_vals = np.interp(sites, xp, yp)
        e = time.time()
        print(f"{e - s} seconds to interpolate recombination rate map")
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
    """
    usage:
    processed_vcf_name: will be taken care of for you by the shell script, but is the output of the simulation script
    rate_map_file: the plink or HapMapII recombination rate map file (HapMap likely will work better)
    """
    processed_vcf_name = os.path.splitext(sys.argv[1])[0]
    rate_map_file = sys.argv[2]
    rate_map_is_plink = True
    print(processed_vcf_name)
    if checkRateMapType(rate_map_file) == "HapMap":
        rate_map_is_plink = False
    ps_obj = p_smoother("src/P-smoother.sh")
    print("Running P-smoother...")
    s = time.time()
    ps_obj.run(f"{processed_vcf_name}.vcf", rate_map_file)
    e = time.time()
    print(f"{e - s} seconds to run P-smoother")
    hap_ibd_obj = hap_ibd("src/hap-ibd.jar")
    if rate_map_is_plink == False:
        rate_map_file = convert(rate_map_file)
    print("Running hap-IBD...")
    hap_ibd_obj.run(f"{processed_vcf_name}.smooth.vcf", rate_map_file, "hap_ibd_p_smoother_results")
    hap_ibd_obj.run(f"{processed_vcf_name}.vcf", rate_map_file, "hap_ibd_results")
    if rate_map_is_plink == False:
        subprocess.run(["rm", rate_map_file])
    subprocess.run(["gunzip", "hap_ibd_p_smoother_results.ibd.gz"])
    subprocess.run(["gunzip", "hap_ibd_results.ibd.gz"])
    


if __name__ == "__main__":
    main()
