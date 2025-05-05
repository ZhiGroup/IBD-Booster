import os
import sys
import subprocess
import time


import torch
import torch.nn as nn
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler


from ibdInference import hap_ibd, p_smoother
from constructDataset import constructBoostedDataset
from formatSegments import formatSegments



class Net(nn.Module):
    """
    Neural Network for filtering of segments
    """
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(40, 36)
        self.activation1 = nn.LeakyReLU()
        self.hidden_layer2 = nn.Linear(36, 18)
        self.activation2 = nn.LeakyReLU()
        self.hidden_layer3 = nn.Linear(18, 9)
        self.activation3 = nn.LeakyReLU()
        self.hidden_layer4 = nn.Linear(9, 3)
        self.activation4 = nn.LeakyReLU()
        self.output = nn.Linear(3, 1)
        self.activation_output = nn.Sigmoid()
    def forward(self, x):
        x = self.activation1(self.hidden_layer1(x))
        x = self.activation2(self.hidden_layer2(x))
        x = self.activation3(self.hidden_layer3(x))
        x = self.activation4(self.hidden_layer4(x))
        x = self.activation_output(self.output(x))
        return x

def df_to_file(df: pd.DateOffset, output_file: str, col: str) -> None:
    """
    writes df to a file for downstream use
    """
    f = open(output_file, 'w')
    for i in range(len(df)):
        row = df.iloc[i]
        cm_columns = [i for i in df.columns if i[0:7] == "gen_len"]
        cm_len = round(sum(row[cm_columns].values), 3)
        if row[col] == 0:
            f.write(f"{row['id1']}\t{row['hap1'] + 1}\t{row['id2']}\t{row['hap2'] + 1}\t20\t{row['start']}\t{row['end']}\t{cm_len}\n")
    f.close()

def augmentSegments(df: pd.DataFrame):
    """
    this function loads in the the neural network and XGBoost models, then filters them according to the
    predictions made by each model, returning the original dataframe with 2 new columns (1 for each prediction)
    """
    exclude = ["id1", "hap1", "id2", "hap2", "start", "end", "classification", "hap_idx1", "hap_idx2"]
    X = df[[i for i in df.columns if i not in exclude]]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
    model = Net()
    model.load_state_dict(torch.load("models/nn_model_state_dict.model", weights_only=True))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgb_ibd_augmentation.json")
    y_pred_xgb = pd.Series(xgb_model.predict(X))

    X_tens = torch.tensor(X.values, dtype = torch.float32)
    y_pred_nn = pd.Series((model(X_tens) > 0.5).int().flatten())

    df = df.reset_index().drop('index', axis = "columns")
    df['xgb_pred'] = y_pred_xgb
    df['nn_pred'] = y_pred_nn
    return df




   
def formatSegmentsBoosted(hap_ibd_res_fp: str, output_hap: str) -> None:
	"""This function condenses the ground truth segments and the hap-IBD reported segments to the same format

	Args:
		hap_ibd_res_fp (str): file path to input hap-IBD results
		gt_file (str): file path to input ground truth segment results
		output_hap (str): desired output path for hap-IBD results
		output_gt (str): desired output path for ground truth segments
	"""

	f = open(hap_ibd_res_fp)
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


def main():
    vcf_file = sys.argv[1]
    rate_map_file = sys.argv[2]
    dataset_file = sys.argv[3]
    base_path = os.path.splitext(vcf_file)[0]
    s = time.time()
    ps_obj = p_smoother("src/P-smoother.sh")
    print("Running P-smoother...")
    ps_obj.run(f"{vcf_file}", rate_map_file)
    hap_ibd_obj = hap_ibd("src/hap-ibd.jar")
    hap_ibd_obj.run(f"{base_path}.smooth.vcf", rate_map_file, "hap_ibd_p_smoother_results")
    subprocess.run(["gunzip", "hap_ibd_p_smoother_results.ibd.gz"])
    print("constructing boosted dataset...")
    formatSegmentsBoosted("hap_ibd_p_smoother_results.ibd", "hap_ibd_p_smoother_results_formatted.ibd")
    constructBoostedDataset(vcf_file, f"{base_path}.smooth.vcf", rate_map_file, "hap_ibd_p_smoother_results_formatted.ibd", 10, dataset_file)
    
    df = pd.read_csv(dataset_file)
    df = augmentSegments(df)
    df_to_file(df, "boosted_segments_xgb.txt", "xgb_pred")
    df_to_file(df, "boosted_segments_nn.txt", "nn_pred")
    e = time.time()
    time_elapsed = e - s
    print(f"total time elapsed: {time_elapsed:.3f}")
    subprocess.run(["rm", "hap_ibd_p_smoother_results_formatted.ibd", f"{base_path}.smooth.vcf", "hap_ibd_p_smoother_results.hbd.gz", "hap_ibd_p_smoother_results.log"])



    

    


if __name__ == "__main__":
    main()
