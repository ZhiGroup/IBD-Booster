import sys
from tqdm import tqdm

def loadVCFData(vcf_file_path: str):

    f = open(vcf_file_path, 'r')
    vcf_data = f.readlines()
    f.close()
    i = 0 # going to use this to determine where to cutoff header, can be different based on what has been done via bcf tools
    while vcf_data[i][0:6] != "#CHROM":
        i += 1
    vcf_data = vcf_data[i:]
    vcf_data = [i.strip().split() for i in vcf_data]
    # ids_dict = {vcf_data[0][i]: i for i in range(9, len(vcf_data[0]))} # used to determine which column index to access based on sample id
    # sites_array = [int(vcf_data[i][1]) for i in range(1, len(vcf_data))]
    return vcf_data



def getGenotype(vcf, individual, ids_dict, site_idx):
    site_offset = 1
    indiv_idx = ids_dict[f"tsk_{individual}"]
    if site_idx - 1 > len(vcf):
        return f"ERROR at site: {site_idx - 1}"
    return vcf[site_idx - site_offset][indiv_idx]


# need to account for: correction and correct correction, correction and incorrect correction,
# no correction and correct no correction, no correction and incorrect no correction

def getCorrectCorrections(original_vcf, vcf_w_error, vcf_w_error_smooth):
    corrections = 0
    non_corrections = 0
    errors = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in tqdm(range(1, len(original_vcf))):
        for j in range(9, len(original_vcf[0])):
            vcf_w_error_vals = vcf_w_error[i][j].split("|")
            vcf_w_error_smooth_vals = vcf_w_error_smooth[i][j].split("|")
            original_vcf_vals = original_vcf[i][j].split("|")
            if original_vcf_vals[0] != vcf_w_error_vals[0]:
                errors += 1
            if original_vcf_vals[1] != vcf_w_error_vals[1]:
                errors += 1
            if vcf_w_error_smooth_vals[0] != vcf_w_error_vals[0]:
                corrections += 1
                if original_vcf_vals[0] == vcf_w_error_smooth_vals[0]:
                    tp += 1
                else:
                    fp += 1
            if vcf_w_error_smooth_vals[1] != vcf_w_error_vals[1]:
                corrections += 1
                if original_vcf_vals[1] == vcf_w_error_smooth_vals[1]:
                    tp += 1
                else:
                    fp += 1
            if vcf_w_error_smooth_vals[0] == vcf_w_error_vals[0]:
                non_corrections += 1
                if original_vcf_vals[0] == vcf_w_error_smooth_vals[0]:
                    tn += 1
                else:
                    fn += 1
            if vcf_w_error_smooth_vals[1] == vcf_w_error_vals[1]:
                non_corrections += 1
                if original_vcf_vals[1] == vcf_w_error_smooth_vals[1]:
                    tn += 1
                else:
                    fn += 1



    return errors, corrections, non_corrections, tp, fp, tn, fn






def run(original_vcf_fp, vcf_w_error_fp, smooth_vcf_fp):
    original_vcf = loadVCFData(original_vcf_fp)
    vcf_w_error = loadVCFData(vcf_w_error_fp)
    smooth_vcf = loadVCFData(smooth_vcf_fp)
    assert len(original_vcf) == len(smooth_vcf) == len(vcf_w_error)
    errors, corrections, non_corrections, tp, fp, tn, fn = getCorrectCorrections(original_vcf, vcf_w_error, smooth_vcf)
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = "NA"
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = "NA"
    print(f"Number of errors: {errors}")
    print(f"Number of corrections made: {corrections}")
    print(f"Number of non_corrections: {non_corrections}")
    print(f"Number of errors corrected: {tp}")
    print(f"Number of non-errors corrected: {fp}")
    print(f"Number of missed corrections: {fn}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")