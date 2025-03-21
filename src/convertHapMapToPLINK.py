import os

def convert(hap_map_file):
    f_in = open(hap_map_file)
    split = os.path.splitext(hap_map_file)
    fp_out = split[0] + "_converted" + split[1]
    f_out = open(fp_out, 'w+')
    for line in f_in:
        data = line.split()
        chr_data = data[0].split("chr")[1]
        g_d = data[3]
        if float(g_d) == 0.0:
            g_d = "0"
        bp = data[1]
        f_out.write(f"{chr_data} . {g_d} {bp}\n")
    f_in.close()
    f_out.close()
    return fp_out




    