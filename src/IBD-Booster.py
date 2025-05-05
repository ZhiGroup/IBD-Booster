import os
import sys
import subprocess
from typing import Tuple


import torch
import torch.nn as nn
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler


from ibdInference import hap_ibd, p_smoother, checkRateMapType
from convertHapMapToPLINK import convert as convert_hap_map_to_plink
from constructDataset import constructDataset

    


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


def augmentSegments(df: pd.DataFrame):
    """
    this function loads in the the neural network and XGBoost models, then filters them according to the
    predictions made by each model, returning the original dataframe with 2 new columns (1 for each prediction)
    """
    X, _ = process_df(df)
    model = Net()
    model.load_state_dict(torch.load("../models/nn_model_state_dict.model", weights_only=True))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("../models/xgb_ibd_augmentation.json")
    y_pred_xgb = pd.Series(xgb_model.predict(X))

    X_tens = torch.tensor(X.values, dtype = torch.float32)
    y_pred_nn = pd.Series((model(X_tens) > 0.5).int())
    df['classification'] = df['classification'] - 2
    df = df.reset_index().drop('index', axis = "columns")
    df['xgb_pred'] = y_pred_xgb
    df['nn_pred'] = y_pred_nn
    return df


def process_df(df: pd.DataFrame) -> Tuple[pd.Dataframe, pd.DataFrame]: 
    """
    simple processing of the input file: removal of non-feature columns and ground truth segments (if included)
    """
    exclude = ["id1", "hap1", "id2", "hap2", "start", "end", "classification", "hap_idx1", "hap_idx2"]
    df['classification'] = df['classification'] - 2
    X = df[[i for i in df.columns if i not in exclude]]
    y = df['classification']
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
    return X, y

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


   



def main():
    vcf_file = sys.argv[1]
    rate_map_file = sys.argv[2]
    dataset_file = sys.argv[3]
    base_path = os.path.splitext(vcf_file)[0]
    if checkRateMapType(rate_map_file) == "HapMap":
        rate_map_is_plink = False
    else:
        rate_map_is_plink = True
    if rate_map_is_plink:
        rate_map_file = convert_hap_map_to_plink(rate_map_file)
    ps_obj = p_smoother("src/P-smoother.sh")
    print("Running P-smoother...")
    ps_obj.run(f"{vcf_file}", rate_map_file)
    if rate_map_is_plink == False:
        rate_map_file = convert_hap_map_to_plink(rate_map_file)
    hap_ibd_obj = hap_ibd("src/hap-ibd.jar")
    hap_ibd_obj.run(f"{base_path}.smooth.vcf", rate_map_file, "hap_ibd_p_smoother_results")
    hap_ibd_obj.run(f"{vcf_file}", rate_map_file, "hap_ibd_results")
    if rate_map_is_plink == False:
        subprocess.run(["rm", rate_map_file])
    subprocess.run(["gunzip", "hap_ibd_p_smoother_results.ibd.gz"])
    subprocess.run(["gunzip", "hap_ibd_results.ibd.gz"])
    constructDataset(vcf_file, f"{base_path}.smooth.vcf", "hap_ibd_p_smoother_results.ibd", f"hap_ibd_results.ibd", 10, rate_map_file, dataset_file)
    df = pd.read_csv(dataset_file)
    df = augmentSegments(df)
    df_to_file(df, "ibd_boosted_segments.txt", "xgb_pred")
    df_to_file(df, "ibd_boosted_segments_nn.txt", "nn_pred")



    

    


if __name__ == "__main__":
    main()
