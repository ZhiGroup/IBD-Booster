import random
from collections import defaultdict


class error_imputer:
    def __init__(self, input_fp, error_rates, output_fp):
        self.input_fp = input_fp
        self.error_rates = error_rates
        self.output_fp = output_fp
        f = open(self.input_fp)
        self.vcf_data = f.readlines()
        f.close()
        # remove header information
        j = 0
        while self.vcf_data[j][0] == "#":
            j += 1
        self.header_info = self.vcf_data[0:j]
        self.vcf_data = self.vcf_data[j:]
        self.metadata = [i.strip().split("\t")[0:9] for i in self.vcf_data]
        self.vcf_data = [i.strip().split("\t")[9:] for i in self.vcf_data]
        self.n_sites = len(self.vcf_data)
        self.n_samples = len(self.vcf_data[0])
        self.error_loci = defaultdict()
        self.individuals = self.header_info[-1].split("\t")
    
    def flip_allele(self, allele):
        if allele == "0":
            return "1"
        else:
            return "0"

    def add_error(self):
        for i in range(0, len(self.vcf_data[0])):
            seed_val = ((43 * i) * 11) // 7
            random.seed(seed_val)
            error_rate = random.choice(self.error_rates)
            n_sites_flipped = int(self.n_sites * error_rate)
            sites_to_flip = random.sample(range(self.n_sites), n_sites_flipped)
            self.error_loci[self.individuals[i + 9]] = sites_to_flip # added 9 bc I do not get rid of metadata titles in individuals
            for j in sites_to_flip:
                allele_to_flip = random.choice([0, 1])
                vals = self.vcf_data[j][i].split("|")
                if allele_to_flip == 0:
                    vals[0] = self.flip_allele(vals[0])
                else:
                    vals[1] = self.flip_allele(vals[1])
                self.vcf_data[j][i] = f"{vals[0]}|{vals[1]}"
    
    def write_to_file(self):
        f = open(self.output_fp, 'w+')
        for line in self.header_info:
            f.write(line)
        for i in range(len(self.vcf_data)):
            tmp = self.metadata[i] + self.vcf_data[i]
            tmp = "\t".join(tmp) + "\n"
            f.write(tmp)
        f.close()
        f = open("error_loci.txt", 'w')
        for key in self.error_loci.keys():
            f.write(key)
            for site in self.error_loci[key]:
                f.write("\t")
                f.write(str(site))
            f.write("\n")
        f.close()

    

def implant_error(vcf_filepath: str, error_rates, output_file: str):
    obj = error_imputer(vcf_filepath, error_rates, output_file)
    obj.add_error()
    obj.write_to_file()



