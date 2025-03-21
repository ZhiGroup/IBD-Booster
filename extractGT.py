import sys
import tskit
import subprocess


def generate_chunks(n_cores, n_samples):
    """
    Generates cutoffs to use multiple cores for extracting ground truth segments
    """
    chunk_size = n_samples // n_cores
    chunks = [((i * chunk_size), (i * chunk_size + chunk_size)) for i in range(n_cores)]
    return chunks




def main():
    ts_path = sys.argv[1]
    if ts_path[len(ts_path) - 6:] != ".trees":
        ts_path += ".trees"
    genetic_map_file = sys.argv[2]
    min_cutoff = float(sys.argv[3])
    n_cpus = int(sys.argv[4])
    ts = tskit.load(ts_path)
    n_samples = ts.num_samples
    indices = generate_chunks(n_cpus, n_samples)
    f = open("gt_extraction.sh", "w+")
    for i in range(len(indices)):
        string = f"nohup python src/extractTrueIBDs.py {ts_path} {indices[i][0]} {indices[i][1]} {genetic_map_file} {min_cutoff} &\n"
        f.write(string)
        f.write(f"PID{i}=$!\n")
    f.write("\n")
    for i in range(len(indices)):
        f.write(f"wait $PID{i}\n")
    f.write("cat subset* > gt_segments.txt\n")
    f.write("rm subset*\n")
    f.close()
    subprocess.run(["chmod", "u+x", "gt_extraction.sh"])
    

if __name__ == "__main__":
    main()
