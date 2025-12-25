
def filter_hap(file_in: str, file_out: str, cutoff: float) -> None:
    f_in = open(file_in, 'r')
    f_out = open(file_out, 'w+')
    for line in f_in.readlines():
        vals = line.strip().split()
        cm_length = float(vals[7])
        if cm_length >= cutoff:
            f_out.write(line)
    f_in.close()
    f_out.close()

def filter_gt(file_in: str, file_out: str, cutoff: float) -> None:
    f_in = open(file_in, 'r')
    f_out = open(file_out, 'w+')
    for line in f_in.readlines():
        vals = line.strip().split()
        start = float(vals[4])
        end = float(vals[5])
        if (end - start)  >= cutoff:
            f_out.write(line)
    f_in.close()
    f_out.close()