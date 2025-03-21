import sys
import torch
import torch.nn as nn
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler



class Net(nn.Module):
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


def augmentSegments(df):
    X, y = process_df(df)
    model = Net()
    model.load_state_dict(torch.load("../models/nn_model_state_dict.model", weights_only=True))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("../models/xgb_ibd_augmentation.json")
    y_pred_xgb = pd.Series(xgb_model.predict(X))

    X_tens = torch.tensor(X.values, dtype = torch.float32)
    y_pred_nn = pd.Series((model(X_tens) > 0.5).int())
    df = df[df['classification'] != 1]
    df['classification'] = df['classification'] - 2
    df = df.reset_index().drop('index', axis = "columns")
    df['xgb_pred'] = y_pred_xgb
    df['nn_pred'] = y_pred_nn
    return df


def process_df(df):
    exclude = ["id1", "hap1", "id2", "hap2", "start", "end", "classification", "hap_idx1", "hap_idx2"]
    df = df[df.classification != 1]
    df['classification'] = df['classification'] - 2
    X = df[[i for i in df.columns if i not in exclude]]
    y = df['classification']
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
    return X, y

def df_to_file(df, output_file, col):
    f = open(output_file, 'w')
    for i in range(len(df)):
        row = df.iloc[i]
        cm_columns = [i for i in df.columns if i[0:7] == "gen_len"]
        cm_len = round(sum(row[cm_columns].values), 3)
        if row[col] == 0:
            f.write(f"{row['id1']}\t{row['hap1'] + 1}\t{row['id2']}\t{row['hap2'] + 1}\t20\t{row['start']}\t{row['end']}\t{cm_len}\n")
    f.close()

def main():
    dataset = sys.argv[1]
    augmented_df = augmentSegments(dataset)
    df_to_file(augmented_df, "predicted_segments_nn.txt", 'nn_pred')
    df_to_file(augmented_df, "predicted_segments_xgb.txt", "xgb_pred")


if __name__ == "__main__":
    main()
    
    