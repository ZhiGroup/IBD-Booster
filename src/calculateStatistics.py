import sys
import statistics
import subprocess
from filterSegments import *
from formatSegments import *

class IBDSegment():
    
    def __init__(self,_index1,_index2,_start,_end):
        self.index1= _index1
        self.index2 = _index2
        self.start = _start
        self.end = _end
        self.interval = [self.start, self.end]
    
    def __lt__(self,other):
        if (self.index1 < other.index1): return True
        if (self.index1 == other.index1):
            if (self.index2 < other.index2): return True
        return False
    
    def __gt__(self,other):
        if (self.index1 > other.index1): return True
        if (self.index1 == other.index1):
            if self.index2 > other.index2: return True
        return False
    
    def __le__ (self,other):
        if (self.index1 < other.index1): return True
        if (self.index1 == other.index1):
            return self.index2 <= other.index2
        return False
    
    def __eq__(self,other):
        if (self is None and other is None): return True
        if (other is None): return False
        if (self.index1 == other.index1 and self.index2 == other.index2):
            return True
        return False
    
    def __str__(self):
        str(self.index1) + '\t' + str(self.index2) + '\t' + str(self.start) + '\t' + str(self.end) 
            
    def __repr__(self):
        return self.__str__()
    

def strToIBDObj(_str):
    _index1 = -1
    _index2 = -1
    _start = -1
    _end = -1
    if (_str == None or _str == ''):
        return IBDSegment(_index1, _index2, _start, _end)
    _str = _str.replace('\n','')
    vals = _str.split()
    
    _index1 = int(vals[0])
    _index2 = int(vals[1])
    _start = float(vals[2])
    _end = float(vals[3])
    return IBDSegment(_index1, _index2, _start, _end)



def get_coverage(_target, _query):
    '''
    Returns the proportion of the covered _target interval by the _query
    '''
    return float (max(0, min(_target[1], _query[1]) - max(_target[0], _query[0])))/ ((_query[1]-_query[0]))

def merge_intervals(intervals):
    if len(intervals) <= 1:
        return intervals
    merged =[]
    for i in intervals:
        if not merged or merged[-1][-1] < i[0]:
            merged.append(i)
        else:
                merged[-1][-1] = max(merged[-1][-1], i[-1])
    return merged

def compute_length_accuracy(gt_path, rp_path):
    """
    assumes the reported segments and ground truth segments are sorted in order their ids
    reports average best coverage by a ground truth segment the reported segments
    """
    f_g = open(gt_path, 'r')
    f_r = open(rp_path, 'r')
    line_g = f_g.readline()
    line_r = f_r.readline()
    gt_obj = strToIBDObj(line_g)
    rp_obj = strToIBDObj(line_r)

    coverage_array = []
    _ibd_r = [] # intermediary list to store segments that are shared between gt and reported
    while rp_obj.index1 != -1:
        while (gt_obj < rp_obj) and (gt_obj.index1 != -1): # ignore segments that aren't shared between samples
            r = f_g.readline()
            gt_obj = strToIBDObj(r)
        while (gt_obj == rp_obj) and (gt_obj.index1 != -1): # start adding ibd-segments 
            _ibd_r.append(gt_obj) # add segments shared between reported and gt
            gt_obj = strToIBDObj(f_g.readline())

        max_cov = 0
        for ir in _ibd_r:
            cov = get_coverage([ir.start, ir.end], [rp_obj.start, rp_obj.end])
            if cov > max_cov:
                max_cov = cov # get best coverage for a given reported segment 
        coverage_array.append(max_cov)
        line_r = f_r.readline()
        gt_obj_new = strToIBDObj(line_r)
        if (not gt_obj_new == rp_obj):
            _ibd_r = []
        
        rp_obj =gt_obj_new
    f_g.close()
    f_r.close()
    return statistics.mean(coverage_array) # return average best coverage


def compute_accuracy(gt_path,rp_path):
    
    f_r = open (rp_path)
    f_gt = open(gt_path)
    line_r = f_r.readline()
    line_g = f_gt.readline()
    ra_obj = strToIBDObj(line_r)
    gt_obj = strToIBDObj(line_g)
    num_covered = 0.0
    num_not_covered = 0
    _ibd_r = []
    #print "start"
    num_v = 0.0
    
    
    while ra_obj.index1 != -1:    
        while(gt_obj < ra_obj and gt_obj.index1 != -1):
            r = f_gt.readline()
            gt_obj = strToIBDObj(r)
        while (gt_obj == ra_obj and gt_obj.index1!=-1):
            _ibd_r.append(gt_obj)
            gt_obj = strToIBDObj(f_gt.readline())
            #print _ibd_r
        _proportion_sum = 0
        found = False
        for ir in _ibd_r:
            _proportion_sum = get_coverage([ir.start,ir.end],[ra_obj.start,ra_obj.end])
            if (_proportion_sum >= 0.5):
                num_covered = num_covered + 1
                found = True
                break
        if (found == False):
            num_not_covered +=1
        num_v +=1
        line_r = f_r.readline()
        gt_obj_new = strToIBDObj(line_r)
        if (not gt_obj_new == ra_obj):
            _ibd_r = []
        
        ra_obj =gt_obj_new
            
    f_r.close()
    f_gt.close()
    
    return num_covered/num_v


def compute_power(gt_path,rp_path):
    
    f_g = open (gt_path)
    f_r = open(rp_path)
    line_g = f_g.readline()
    line_r = f_r.readline()
    
    gt_obj = strToIBDObj(line_g)
    r_obj = strToIBDObj(line_r)
   
    
    _ibd_r = []
    num_v = 0
    total_sum = 0.0
    while gt_obj.index1 != -1:
        
        while(r_obj < gt_obj and r_obj.index1 != -1):
            r = f_r.readline()
            r_obj = strToIBDObj(r)
        while (r_obj == gt_obj and r_obj.index1!=-1):
            _ibd_r.append(r_obj)
            r_obj = strToIBDObj(f_r.readline())
        _proportion_sum = 0
        for ir in _ibd_r:
            _proportion_sum += get_coverage([ir.start,ir.end],[gt_obj.start,gt_obj.end])

        total_sum += _proportion_sum
        num_v +=1
        line_g = f_g.readline()
        gt_obj_new = strToIBDObj(line_g)

        if (not gt_obj_new == gt_obj):
            _ibd_r = []
        
        gt_obj =gt_obj_new
            
    f_g.close()
    f_r.close()


    return total_sum/num_v

def compute_accumulative_power(gt_path, rp_path):
    f_g = open (gt_path)
    f_r = open(rp_path)
    line_g = f_g.readline()
    line_r = f_r.readline()
    
    gt_obj = strToIBDObj(line_g)
    r_obj = strToIBDObj(line_r)
   
    
    _ibd_r = []
    num_v = 0
    total_sum = 0.0
    while gt_obj.index1 != -1:
        
        while(r_obj < gt_obj and r_obj.index1 != -1):
            r = f_r.readline()
            r_obj = strToIBDObj(r)
        while (r_obj == gt_obj and r_obj.index1!=-1):
            _ibd_r.append(r_obj)
            r_obj = strToIBDObj(f_r.readline())
        _proportion_sum = 0
        intervals = [i.interval for i in _ibd_r]
        merged_intervals = merge_intervals(intervals)
        for mi in merged_intervals:
            _proportion_sum += get_coverage(mi, gt_obj.interval)

        total_sum += _proportion_sum
        num_v +=1
        line_g = f_g.readline()
        gt_obj_new = strToIBDObj(line_g)

        if (not gt_obj_new == gt_obj):
            _ibd_r = []
        
        gt_obj =gt_obj_new
            
    f_g.close()
    f_r.close()
    return total_sum/num_v



def main():
    hap_output = sys.argv[1]
    gt_file = sys.argv[2]
    cm_cutoff = float(sys.argv[3])
    filter_gt(gt_file, "tmp1_gt", cm_cutoff) # filter according to cutoff
    filter_hap(hap_output, "tmp1_hap", cm_cutoff) # filter according to cutoff
    formatSegments("tmp1_hap", "tmp1_gt", "tmp2_hap", "tmp2_gt") # put segments in form samp1 samp2 start end
    with open("tmp3_hap", "w+") as file1:
        subprocess.run(["sort", "-g", "-k1", "-k2", "tmp2_hap"], stdout = file1) # sort by samp id
    with open("tmp3_gt", "w+") as file2:
        subprocess.run(["sort", "-g", "-k1", "-k2", "tmp2_gt"], stdout = file2) # sort by samp id
    # maf = extract_maf_frequency(hap_output)
    l_acc = compute_length_accuracy("tmp3_gt", "tmp3_hap")
    acc = compute_accuracy("tmp3_gt", "tmp3_hap")
    power = compute_power("tmp3_gt", "tmp3_hap")
    a_pow = compute_accumulative_power("tmp3_gt", "tmp3_hap")
    print(f"Accuracy: {acc}")
    print(f"Length Accuracy: {l_acc}")
    print(f"Power: {power}")
    print(f"Accumulative Power: {a_pow}")
    subprocess.run(["rm", "tmp1_hap", "tmp2_hap", "tmp3_hap", "tmp1_gt", "tmp2_gt", "tmp3_gt"])

if __name__ == "__main__":
    main()