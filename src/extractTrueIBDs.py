import numpy as np
import msprime
import tskit
import sys
import time

ts_path = sys.argv[1]
start_index = int(sys.argv[2])
end_index = int(sys.argv[3])
genetic_map_file = sys.argv[4]
min_cutoff = float(sys.argv[5])

s = time.time()

ts = tskit.load(ts_path)
id_start = 0
id_end = ts.num_samples - 1




xp = []
yp = []
f = open(genetic_map_file)
for line in f:
	vals = line.split()
	xp.append(int(vals[1]))
	yp.append(float(vals[len(vals)-1]))
f.close()

max_val = xp[len(xp)-1]
_sites = np.arange(0,max_val+1)		
gen_map =np.interp(_sites,xp,yp)
	
f_o = open("subsets_ibd_" + str(start_index) + "_" + str(end_index) + ".txt",'w+')




trees_iter = ts.trees()
tree = next(trees_iter)

mrca_last = [['0' for k in range(id_end+1)] for l in range(id_end+1)]
last_left = [[0 for k in range(id_end+1)] for l in range(id_end+1)]

for i in range(start_index,end_index):
	for j in range(start_index+1,id_end):
		mrca = tree.mrca(i, j)
		mrca_last[i][j] = mrca
		mrca_last[j][i] = mrca
		last_left[i][j] = tree.interval[0]
		last_left[j][i] = tree.interval[0]

min_tree_subsample = 5000

last_tree_pos = 0
for tree in trees_iter:
	if (tree.interval[0] -  last_tree_pos) < min_tree_subsample:
		continue
	#print (tree.interval[0])
	last_tree_pos = tree.interval[0]
	for i in range(start_index,end_index):
		for j in range(i+1,id_end):
			mrca = tree.mrca(i, j)
			if (mrca_last[i][j] != mrca):
				left = tree.interval[0]
				#if (left - last_left[i][j] < min_cutoff):
				genomic_end = int(round(left))
				genomic_start = int(last_left[i][j])
				gen_end = 0
				gen_start = 0
				if (genomic_end > len(gen_map)):
					gen_end = gen_map[len(gen_map)-1]
				else:
					gen_end = gen_map[genomic_end]
				if (genomic_start > len(gen_map)):
					gen_start = gen_map[len(gen_map)-1]
				else:
					gen_start = gen_map[genomic_start]
					
				if (gen_end - gen_start < min_cutoff):
				
					last_left[i][j] = left
					last_left[j][i] =left
					mrca_last[i][j] = mrca
					mrca_last[j][i] = mrca
					continue
				f_o.write(str(i) + "\t" + str(j) + "\t" +\
				str(genomic_start)+ "\t" +\
				str(genomic_end) + "\t" + str(gen_start)+ "\t" + str(gen_end) +  "\n")
				last_left[i][j] = left
				last_left[j][i] =left
				mrca_last[i][j] = mrca
				mrca_last[j][i] = mrca

for i in range(start_index,end_index):
	for j in range(i+1,id_end):
	
		genomic_start = int(round(last_left[i][j]))
		genomic_end = int(ts.sequence_length)
		gen_end = 0
		gen_start = 0
		if (genomic_end > len(gen_map)):
			gen_end = gen_map[len(gen_map)-1]
		else:
			gen_end = gen_map[genomic_end]
		if (genomic_start > len(gen_map)):
			gen_start = gen_map[len(gen_map)-1]
		else:
			gen_start = gen_map[genomic_start]
	
		if gen_end - gen_start >= min_cutoff:
			f_o.write(str(i) + "\t" + str(j) + "\t" +\
			str(genomic_start)+ "\t" +\
			str(genomic_end) + "\t" + str(gen_start)+ "\t" + str(gen_end) +  "\n")

f_o.close()
					
e = time.time()
print(f"{e - s} seconds for ground truth extraction")

	
