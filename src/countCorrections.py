import sys

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
    return vcf_data

def countCorrections(vcf, smooth_vcf):
    n_corrections = 0
    for i in range(1, len(vcf)):
        for j in range(9, len(vcf[0])):
            vcf_val = vcf[i][j].split("|")
            vcf_smooth_val = smooth_vcf[i][j].split("|")
            if vcf_val[0] != vcf_smooth_val[0]:
                n_corrections += 1
            if vcf_val[1] != vcf_smooth_val[1]:
                n_corrections += 1
    return n_corrections


def main():
    unsmooth_vcf_fp = sys.argv[1]
    smooth_vcf_fp = sys.argv[2]
    vcf = loadVCFData(unsmooth_vcf_fp)
    vcf_s = loadVCFData(smooth_vcf_fp)
    n_corrections = countCorrections(vcf, vcf_s)
    print(n_corrections)
    exit(0)

if __name__ == "__main__":
    main()