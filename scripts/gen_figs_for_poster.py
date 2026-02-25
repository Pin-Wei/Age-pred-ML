#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

import seaborn as sns
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.getcwd(), "..", "src"))
from utils import basic_Q_features, ST_features
from plotting import plot_real_pred_age, plot_comparison_bars, plot_cormat

sys.path.append(os.path.join(os.getcwd(), "..", "scripts"))
from gen_derivatives import Constants, Description, load_model_results, force_agesex_cols
from compare_versions import Config, define_arguments, compare_pad_values

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

## ------------------------------------------------------------------------------------------------------------

def load_result_df(custom_bar_x_lab):
    result_DF_list = []

    for version, input_folder in config.input_folders.items():
        desc_json_path = os.path.join(input_folder, "description.json")
        with open(desc_json_path, 'r', errors='ignore') as f:
            desc_json = json.load(f)
        desc = Description(desc_json, const, args)

        config.result_path = os.path.join(input_folder, "results_{}_{}.json")
        result_DF, _, _ = load_model_results(
            config, desc, const, 
            output_path=os.path.join(out_path, f"results_DF_{version}.csv"), 
            overwrite=args.overwrite
        )
        result_DF["Version"] = version
        result_DF["VerType"] = result_DF["Version"] + "_" + result_DF["Type"]

        result_DF.rename(columns={"PredictedAgeDifference": "PAD", "CorrectedPAD": "PADAC",}, inplace=True)
        result_DF["PAD_abs"] = result_DF["PAD"].abs()
        result_DF["PADAC_abs"] = result_DF["PADAC"].abs()

        if args.custom_bar_x_lab is not None:
            data_DF = pd.read_csv(config.raw_data_path)
            data_DF["SID"] = data_DF["BASIC_INFO_ID"].map(lambda x: x.replace("sub-0", ""))
            temp_DF = force_agesex_cols(DF=result_DF, DF2=data_DF)
            temp_DF.rename(columns={custom_bar_x_lab: "BarGroup"}, inplace=True)
            result_DF = pd.merge(
                result_DF, temp_DF.loc[:, ["SID", "BarGroup"]], on="SID", how="left"
            )
                
        if "Sex" not in result_DF.columns:
            result_DF["Sex"] = ""
            result_DF.replace({"AgeGroup": {"le-44": "Y", "ge-45": "O"}}, inplace=True)
            result_DF.replace({"BarGroup": {"le-44": "Y", "ge-45": "O"}}, inplace=True)
        else:
            result_DF.replace({"AgeGroup": {"all": "", "le-44": "Y", "ge-45": "O"}}, inplace=True)
            result_DF.replace({"BarGroup": {"all": "", "le-44": "Y", "ge-45": "O"}}, inplace=True)
        
        result_DF["Group"] = result_DF["AgeGroup"] + result_DF["Sex"]
        result_DF_list.append(result_DF)

    result_DF = pd.concat(result_DF_list, ignore_index=True)
    result_DF.drop_duplicates(inplace=True)

    return desc, result_DF

def compare_approachs(result_DF, feature_orientations, version_list, args=None):
    different_participants = getattr(args, "different_participants", False)
    stats_DF_list = []

    for pad_type in ["PAD", "PADAC"]:
        for ori_name in feature_orientations:
            for group in ["Y", "O"]:
                for ver_1, ver_2 in itertools.combinations(version_list, 2):
                    stats_results = compare_pad_values(
                        V1_abs=result_DF.query(
                            f"Type == '{ori_name}' & BarGroup == '{group}' & Version == '{ver_1}'"
                        )[f"{pad_type}_abs"], 
                        V2_abs=result_DF.query(
                            f"Type == '{ori_name}' & BarGroup == '{group}' & Version == '{ver_2}'"
                        )[f"{pad_type}_abs"], 
                        independent=True if different_participants else False
                    )
                    stats_results.insert(0, "Type", ori_name)
                    stats_results.insert(1, "Group", group)
                    stats_results.insert(2, "PAD_type", pad_type)
                    stats_results.insert(3, "V1", ver_1)
                    stats_results.insert(4, "V2", ver_2)
                    stats_DF_list.append(stats_results)
        
    return pd.concat(stats_DF_list, ignore_index=True)

def compare_modalities(result_DF, feature_orientations, version_list, args=None):
    different_participants = getattr(args, "different_participants", False)
    stats_DF_list = []

    for version in version_list:
        for group in ["Y", "O"]:
            for pad_type in ["PAD", "PADAC"]:
                for ori_1, ori_2 in itertools.combinations(feature_orientations, 2):
                    stats_results = compare_pad_values(
                        V1_abs=result_DF.query(
                            f"Version == '{version}' & BarGroup == '{group}' & Type == '{ori_1}'"
                        )[f"{pad_type}_abs"], 
                        V2_abs=result_DF.query(
                            f"Version == '{version}' & BarGroup == '{group}' & Type == '{ori_2}'"
                        )[f"{pad_type}_abs"], 
                        independent=True if different_participants else False
                    )
                    stats_results.insert(0, "Version", version)
                    stats_results.insert(1, "Group", group)
                    stats_results.insert(2, "PAD_type", pad_type)
                    stats_results.insert(3, "V1", ori_1)
                    stats_results.insert(4, "V2", ori_2)
                    stats_DF_list.append(stats_results)
                    
    return pd.concat(stats_DF_list, ignore_index=True)

def make_df_for_cormat(result_DF, data_DF):
    DF_Y = result_DF.query("Version == 'ByAge' & AgeGroup == 'Y'")
    DF_O = result_DF.query("Version == 'ByAge' & AgeGroup == 'O'")
    DF_Un = result_DF.query("Version == 'Undivided'")
    data_DF_interested = data_DF.loc[:, ["SID"] + basic_q_features + st_features]

    wide_df_dict = {}
    for pad_type in ["PAD", "PADAC"]:
        wide_df_dict[pad_type] = {}            
        for group_name, sub_df in zip(["Y", "O", "Y&O"], [DF_Y, DF_O, DF_Un]):
            sub_df = (
                sub_df.loc[:, ["Type", "SID", pad_type]]
                .pivot(index="SID", columns="Type", values=pad_type)
                .reset_index()
                .rename(columns={"index": "SID"})
            )
            wide_df_dict[pad_type][group_name] = sub_df.merge(data_DF_interested, on="SID", how="left")

    return wide_df_dict

if __name__ == "__main__":
    const = Constants()
    basic_q_features = basic_Q_features()
    st_features = ST_features()

    for model_type in ["ElasticNet", "CART", "RF", "XGBM", "LGBM"]:
        
        for test_or_all, by_age_out_path, undivided_out_path in zip(
            ["Testing", "Entire"], 
            [f"2025-09-17_original_sex-0_{model_type}", 
             f"2025-09-17_original_sex-0 (2025-09-17_original_sex-0_{model_type})"], 
            [f"2025-09-17_original_age-0_sex-0_{model_type}", 
             f"2025-10-07_original_age-0_sex-0_tsr-0.0 (2025-09-17_original_age-0_sex-0_{model_type})"]
        ):
            out_path = os.path.join("..", "derivatives", "2025-11-12 Psychonomic poster", model_type, test_or_all)
            os.makedirs(out_path, exist_ok=True)

            ## Define arguments
            sys.argv = [
                "compare_versions.py", 
                "-v", "0", 
                "-ba", by_age_out_path, 
                "-un", undivided_out_path, 
                "-cbg", "0"
            ]
            args = define_arguments()
            args.ignore_all = False
            args.overwrite = False
            custom_bar_x_lab = "AgeGroup"

            config = Config(args)    
            version_list = list(config.input_folders.keys()) # ["ByAge", "Undivided"]

            ## Load results:
            desc, result_DF = load_result_df(custom_bar_x_lab)
            feature_orientations = result_DF["Type"].unique()

            melted_result_DF = (
                result_DF
                .loc[:, ["Version", "Type", "BarGroup", "SID", "PAD_abs", "PADAC_abs"]]
                .melt(
                    id_vars=["Version", "Type", "BarGroup", "SID"], 
                    value_vars=["PAD_abs", "PADAC_abs"], 
                    var_name="PAD_type"
                )
            )

            ## Scatter plots:
            for version in version_list:
                for ori_name in feature_orientations:
                    df_temp = result_DF.query(f"Version == '{version}' & Type == '{ori_name}'")
                    df1 = df_temp.loc[:, ["Age", "PredictedAge"]]
                    df1.insert(1, "pad_type", "Raw")
                    df2 = df_temp.loc[:, ["Age", "CorrectedPredictedAge"]]
                    df2.insert(1, "pad_type", "AC")
                    df2.rename(columns={"CorrectedPredictedAge": "PredictedAge"}, inplace=True)
                    plot_real_pred_age(
                        DF=pd.concat([df1, df2], axis=0), 
                        y_lab="PredictedAge", 
                        color_hue="pad_type", 
                        color_dict={"Raw": "#00BFFF", "AC": "#FF1493"}, 
                        output_path= os.path.join(out_path, f"scatter_real_pred_age_{version}_{ori_name}.png"),
                        overwrite=args.overwrite
                    )

            ## Bar plots:
            ## Test if an age-stratified approach improves prediction accuracy:
            stats_versions_DF = compare_approachs(result_DF, feature_orientations, version_list, args=args)
            stats_versions_DF.to_csv(os.path.join(out_path, "compare_versions_stats.csv"), index=False)

            for pad_type in ["PAD", "PADAC"]:
                plot_comparison_bars(
                    result_DF=melted_result_DF.query(f"PAD_type == '{pad_type}_abs'"), 
                    stats_DF=stats_versions_DF.query(f"PAD_type == '{pad_type}'"), 
                    out_file=os.path.join(out_path, f"compare_versions_{pad_type}_bars.png"), 
                    fig_type="approach_per_pad", 
                    overwrite=args.overwrite
                )

            # melted_result_DF["GroupxPAD"] = melted_result_DF["BarGroup"].astype(str) + "_" + melted_result_DF["var"]
            # stats_versions_DF["GroupxPAD"] = stats_versions_DF["Group"] + "_" + stats_versions_DF["PAD_type"]
            # plot_comparison_bars( ... , fig_type="version_both_pads")

            ## Compare models trained using features in specific modalities:
            stats_modalities_DF = compare_modalities(result_DF, feature_orientations, version_list, args=args)
            stats_modalities_DF.to_csv(os.path.join(out_path, "compare_modalities_stats.csv"), index=False)
            
            melted_result_DF["VerxGroup"] = melted_result_DF["Version"] + "_" + melted_result_DF["BarGroup"].astype(str)

            for pad_type in ["PAD", "PADAC"]:
                plot_comparison_bars(
                    result_DF=melted_result_DF.query(f"PAD_type == '{pad_type}_abs'"), 
                    stats_DF=stats_modalities_DF.query(f"PAD_type == '{pad_type}'"), 
                    out_file=os.path.join(out_path, f"compare_modalities_{pad_type}_bars.png"), 
                    fig_type="modalities_per_pad"
                )

            ## Correlation matrices:
            data_DF = pd.read_csv(config.raw_data_path)
            data_DF["SID"] = data_DF["BASIC_INFO_ID"].map(lambda x: x.replace("sub-0", ""))

            wide_df_dict = make_df_for_cormat(result_DF, data_DF)

            for pad_type in ["PAD", "PADAC"]:
                for group_name in ["Y", "O", "Y&O"]:

                    ## Among PAC(AC) values:
                    plot_cormat(
                        DF=wide_df_dict[pad_type][group_name], 
                        targ_cols=sorted(desc.feature_oris),  
                        annot_fs=16, 
                        figsize=(4, 3), 
                        output_path=os.path.join(out_path, f"cormat_{pad_type}_{group_name}.png"), 
                        overwrite=args.overwrite
                    )

                    ## Correlation with questionnaire / standardized test features:
                    for y_title, y_cols, fig_size in zip(
                            ["questionnaire", "standardized test"], 
                            [basic_q_features, st_features], 
                            [(12, 11), (8, 7)]
                        ):
                        plot_cormat(
                            DF=wide_df_dict[pad_type][group_name], 
                            targ_cols=sorted(desc.feature_oris), 
                            corrwith_cols=y_cols, 
                            rename_ycols=True, 
                            y_title=y_title, 
                            annot_fs=18, 
                            ticks_fs=20, 
                            fig_size=fig_size, 
                            output_path=os.path.join(out_path, f"cormat_{pad_type}_{y_title}_{group_name}.png"), 
                            overwrite=args.overwrite
                        )
