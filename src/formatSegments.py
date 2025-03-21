def formatSegments(hap_ibd_res, gt_file, output_hap, output_gt):

	f = open(hap_ibd_res)
	f_o = open(output_hap ,'w+')
	for line in f:
		vals = line.split()
		id1 = vals[0].replace("tsk_","")
		hap1 = vals[1]
		id2 = vals[2].replace("tsk_","")
		hap2 = vals[3]
		start = vals[5]
		end = vals[6]
		id1_f1 = int(id1) * 2 + int(hap1) -1
		id1_f2 = int(id2) * 2 + int(hap2) -1
		len_cm = vals[7]
		f_o.write(str(id1_f1) + "\t" + str(id1_f2) + "\t" + start + "\t" + end + "\t" + len_cm + "\n")
	f.close()
	f_o.close()

	f = open(gt_file )
	f_o = open(output_gt ,'w+')
	for line in f:
		if ("individual_1_id" in line): continue
		vals = line.split()
		id1 = vals[0].replace("tsk_","")
		hap1 = vals[1]
		id2 = vals[1].replace("tsk_","")
		start = vals[2]
		end = vals[3]
		id1_f1 = int(id1) #  * 2 + int(hap1) -1
		id1_f2 = int(id2) #  * 2 + int(hap2) -1
		gen_start = float(vals[4])
		gen_end = float(vals[5])
		len_cm = gen_end - gen_start
		f_o.write(str(id1_f1) + "\t" + str(id1_f2) + "\t" + start + "\t" + end + "\t" + str(len_cm) + "\n")
	f.close()
	f_o.close()
