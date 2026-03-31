#!/usr/bin/env python

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

pd.options.mode.chained_assignment = None

class Config:
    def __init__(self, args):
        self.setup_paths(args)
        self.seed = args.seed if args.seed is not None else np.random.randint(0, 10000)
        self.test_ratio = args.test_ratio
        self.age_splitpoint = args.age_splitpoint

    def setup_paths(self, args):
        self.raw_data_path = os.path.join("..", "data", "rawdata", "DATA_ses-01_2024-12-09.csv")
        self.res_data_path = args.result_path if args.result_path is not None else os.path.join(
            "..", "derivatives", "2025-09-17 No feature engineering", "no split", 
            "2025-10-07_original_age-0_sex-0_tsr-0.0 (2025-09-17_original_age-0_sex-0_XGBM)", 
            "[table] combined results.csv"
        )
        self.out_dir = os.path.join("..", "outputs")
        self.out_suffix = args.out_suffix if args.out_suffix is not None else datetime.now().strftime("%y%m%d")
        self.out_path = os.path.join(self.out_dir, f"train-test_ids_{self.out_suffix}.json")
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=None, 
                        help="The value used to initialize all random number generator.")
    parser.add_argument("-tr", "--test_ratio", type=float, default=0.3, 
                        help="The ratio of the test set. (default: 0.3)")
    parser.add_argument("-as", "--age_splitpoint", type=int, default=45,
                        help="Participants younger than this age will be assigned to the 'Y' set; older or equal to this age will be assigned to the 'O' set. (default: 45)")
    parser.add_argument("-rp", "--result_path", type=str, default=None, 
                        help="Path of the result data to be used (*combined results.csv).")
    parser.add_argument("-sfx", "--out_suffix", type=str, default=None,
                        help="Suffix of the output filename. (if not specified, use YYMMDD)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config(args)
    
    raw_data = pd.read_csv(config.raw_data_path)
    raw_data = raw_data.rename(columns={
        "BASIC_INFO_ID": "Subj", 
        "BASIC_INFO_AGE": "Age", 
        "BASIC_INFO_SEX": "Sex"
    })
    raw_data["Sex"] = raw_data["Sex"].map({1: "M", 2: "F"})
    raw_data["AgeGroup"] = np.where(raw_data["Age"] < config.age_splitpoint, "Y", "O")
    raw_data["Group"] = raw_data["AgeGroup"] + raw_data["Sex"]
    raw_data = raw_data.loc[:, ["Subj", "Group"]]

    res_data = pd.read_csv(config.res_data_path)
    res_data = res_data.query("Type == 'ALL'")
    res_data["Subj"] = res_data["SID"].map(lambda x: f"sub-{x:04d}")
    res_data["abs_PADAC"] = res_data["CorrectedPAD"].abs()
    mean_padac = res_data["abs_PADAC"].mean()
    res_data = res_data.query(f"abs_PADAC <= {mean_padac}").loc[:, ["Subj", "Age"]]

    data = pd.merge(res_data, raw_data, how="left", on="Subj")
    
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=config.test_ratio, random_state=config.seed
    )
    trainval_idx, test_idx = next(sss.split(data["Subj"], data["Group"]))
    split_with_ids = {
        "Train": data.iloc[trainval_idx]["Subj"].tolist(), 
        "Test": data.iloc[test_idx]["Subj"].tolist()
    }

    with open(config.out_path, 'w', encoding='utf-8') as f:
        json.dump(split_with_ids, f, ensure_ascii=False)

    print(f"\nTrain-val ({len(trainval_idx)}) and Test ({len(test_idx)}) IDs (Total: {data.shape[0]}) are assigned and saved to:\n{config.out_path}\n")

if __name__ == "__main__":
    main()