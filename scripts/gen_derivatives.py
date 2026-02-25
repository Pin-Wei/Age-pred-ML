#!/usr/bin/python

import os
import sys
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import pingouin as pg

sys.path.append(os.path.join(os.getcwd(), "..", "src"))
from plotting import (
    plot_real_pred_age, plot_corr_with_stats, plot_multiple_corr_with_stats, plot_real_syn_data, 
    plot_pad_bars, plot_feature_importances, plot_cormat, plot_feature_sunburst, 
    plot_color_legend
)
from utils import basic_Q_features, ST_features, domain_approach_mapping_dict

## Classes: ===========================================================================

class Config:
    def __init__(self, args):
        self.source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.js_code_path = os.path.join(self.source_path, "src", "output_desc_viewer.js")
        self.html_template_path = os.path.join(self.source_path, "src", "output_desc_viewer.html") 

        self.folder = args.folder
        self.input_folder = os.path.join(self.source_path, "outputs", self.folder)
        self.desc_path    = os.path.join(self.input_folder, "description.json")
        self.datmat_path  = os.path.join(self.input_folder, "prepared_data.csv")
        self.result_path  = os.path.join(self.input_folder, "results_{}_{}.json")
        self.feature_path = os.path.join(self.input_folder, "features_{}_{}.csv")

        self.output_folder              = os.path.join(self.source_path, "derivatives", self.folder)
        self.html_desc_outpath          = os.path.join(self.output_folder, "description.html")
        self.model_results_outpath      = os.path.join(self.output_folder, "[table] combined results.csv")
        self.model_infos_outpath        = os.path.join(self.output_folder, "[table] model types and feature numbers.csv")
        self.subj_infos_outpath         = os.path.join(self.output_folder, "[table] median and std of ages{}.csv")
        self.pad_cormat_data_outpath    = os.path.join(self.output_folder, "[database] PAD values with interested features.csv")
        self.pad_pairwise_corr_outpath  = os.path.join(self.output_folder, "[table] pairwise correlations between {}.xlsx")
        # self.feature_df_outpath         = os.path.join(self.output_folder, "[table] selected features.csv")
        self.feature_importance_outpath = os.path.join(self.output_folder, "[bar] {} feature importances ('{}').png")
        self.pred_vs_real_age_outpath   = os.path.join(self.output_folder, "[scatter] predicted vs real age ({}).png")
        self.residual_regress_outpath   = os.path.join(self.output_folder, "[regress] {}PAD x real age.png")
        self.real_vs_synth_data_outpath = os.path.join(self.output_folder, "[hist] distributions of real and synthetic data in '{}'.png")
        self.feature_cormat_outpath     = os.path.join(self.output_folder, "[cormat] between features of {} data in '{}'.png")
        self.pad_barplot_outpath        = os.path.join(self.output_folder, "[bar] PAD values{}.png")
        self.pad_cormat_outpath         = os.path.join(self.output_folder, "[cormat] between {} in '{}'{}.png")
        self.pad_scatter_outpath        = os.path.join(self.output_folder, "[scatter] between {} in '{}' ({} × {}).png")
        # self.sunburst_outpath           = os.path.join(self.output_folder, "[pie] {}.png")
        self.pred_type_legend_outpath   = os.path.join(self.output_folder, "[color legend] prediction Before & After correction.png")
        self.pad_type_legend_outpath    = os.path.join(self.output_folder, "[color legend] PAD & PAD_ac.png")

class Constants:
    def __init__(self):
        self.data_synth_methods = [
            "SMOTENC", "CTGAN", "TVAE"
        ]
        self.test_result_cols = [
            "Age", "PredictedAge", "PredictedAgeDifference", "CorrectedPAD", "CorrectedPredictedAge"
        ]
        self.train_result_cols = [
            "TrainingSubjID", "TrainingAge", "TrainingPredAge", "TrainingPAD", "TrainingPADAC", "TrainingCorPredAge"
        ]
        self.model_info_cols = [
            "Model", "NumberOfFeatures"
        ]
        self.pad_bar_y_lims = [
            {"STR": 9, "BEH": 11, "FUN": 14, "ST": np.nan, "ALL": 11}, 
            {"STR": 14, "BEH": 14, "FUN": 14, "ST": 14, "ALL": 14}
        ][1]
        
class Description:
    def __init__(self, desc_json, const, args):
        # for k, v in desc_json.items():
        #     snake_k = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", k).lower() # add underscores before capital letters that follow lowercase letters
        #     setattr(self, snake_k, v)

        self.age_group_labels = desc_json["AgeGroups"]
        self.age_breaks = (
            [ 0 ] + [ int(x.split("-")[1]) for x in self.age_group_labels[:-1] ] + [ np.inf ]
        )
        self.sep_sex = bool(desc_json["SexSeparated"]) 
        self.label_cols = (
            ["Sex", "AgeGroup"] if self.sep_sex 
            else ["AgeGroup"]
        )
        self.label_list = (
            list(itertools.product(self.age_group_labels, ["M", "F"])) if self.sep_sex
            else self.age_group_labels
        )
        self.grouping_col = (
            "AgeSex" if self.sep_sex 
            else "AgeGroup"
        )
        self.feature_orientations = (
            ["STRUCTURE", "BEH", "FUNCTIONAL"] if args.ignore_all
            else desc_json["FeatureOrientations"]
        )
        if any( char.isdigit() for char in self.feature_orientations[0] ): 
            self.feature_oris = self.feature_orientations
        else:
            self.feature_oris = [ o[:3] for o in self.feature_orientations ]
        self.traintest = (
            True if desc_json["TestsetRatio"] != 0 
            else False
        )
        self.sid_name = (
            "TestingSubjID" if self.traintest 
            else "SubjID"
        )
        self.data_synthetized = (
            True if desc_json["DataBalancingMethod"] in const.data_synth_methods
            else False
        )
        self.use_pretrained = (
            True if desc_json["UsePretrainedModels"] is not None 
            else False
        )

class ColorDicts:
    def __init__(self): 
        self.sex = {
            "M": "#3399FF", 
            "F": "#FF9933"
        }
        self.train_test = {
            "Train": "#FF1493", 
            "Test":  "#00BFFF"
        }
        self.real_synth = {
            "Real":      "#FF9933", 
            "Synthetic": "#3399FF"
        }
        self.pred_type = {
            "Raw": "#00BFFF", 
            "AC":  "#FF1493" # age-corrected
        }
        self.pad_bars = {
            "PAD":    "#00BFFF", # "#219EBC",
            "PAD_ac": "#FF1493"  # "#FB8500" 
        }
        self.sunburst = {
            "STRUCTURE": {
            ## Parents
                "GM":         "#305973",  # Deep slate blue
                "WM":         "#6FAE9E",  # Deep teal
                "NULL":       "#6A4F83",  # Deep muted violet
            ## Children
                "VOLUME":     "#A5D9D4",  # Light teal
                "ThickAvg":   "#B2C8DF",  # Icy blue-gray
                "ThickStd":   "#B7B1D8",  # Muted lavender
                "FA":         "#F1D4B8"   # Soft peach blush
            },
            "BEH": {
            ## Parents
                "MEMORY":     "#2E7D7E",  # Deep teal
                "MOTOR":      "#E66A28",  # Deep orange
                "LANGUAGE":   "#6A4F83",  # Deep muted violet
            ## Children
                "EXCLUSION":  "#7FC5C0",  # Light teal
                "OSPAN":      "#A5D9D4",  # Lighter teal
                "MST":        "#D6ECEA",  # Lightest teal
                "GFORCE":     "#FFA46B",  # Light orange
                "GOFITTS":    "#FFBD85",  # Lighter orange
                "BILPRESS":   "#FFDAB3",  # Lightest orange
                "SPEECHCOMP": "#9D7FBF",  # Light muted violet
                "WORDNAME":   "#C3B2DB"   # Lightest muted violet
            },
            "FUNCTIONAL": {
            ## Parents
                "MRI":        "#3A4F7A",  # Deep indigo blue
                "EEG":        "#6FAE9E",  # Deep teal
            ## Children
                "MEMORY":     "#A5D9D4",  # Light teal
                "MOTOR":      "#FFA46B",  # Light orange
                "LANGUAGE":   "#C3B2DB"   # Light muted violet
            },
            "ALL": {
            ## Parents
                "STR":        "#4E5D6C",  # Deep gray blue
                "MRI":        "#3A4F7A",  # Deep indigo blue
                "EEG":        "#6FAE9E",  # Deep teal
                "BEH":        "#E67E22",  # Deep orange
            ## Children
                "GM":         "#B2C8DF",  # Pale icy blue-gray
                "WM":         "#CBD5C0",  # Soft pale sage green
                "NULL":       "#9AA0B5",  # Slate gray
                "MEMORY":     "#A5C9B3",  # Calm mint green
                "MOTOR":      "#F6A96C",  # Balanced warm orange
                "LANGUAGE":   "#B7B1D8"   # Muted lavender
            }
        }

## Functions: =========================================================================

def define_arguments():
    parser = argparse.ArgumentParser(
        description="""
        The dataset was divided into groups according to the gender and age range of the 
        participants, and different models were then trained based on the data collected 
        from three orientations (structural, behavioral, and functional) in these groups.
        
        This script generates the derivatives of the model outputs from the folder specified by the '-f' argument, 
        and the generated files will be saved in a folder with the same name as the '-f' argument under the 'derivatives' folder.
        """
    )
    parser.add_argument("-f", "--folder", type=str, required=True, 
                        help="The folder name of the model outputs.")
    parser.add_argument("-padac", "--use_padac", action="store_true", default=False,
                        help="Use age-corrected PAD values for correlation analysis.")
    parser.add_argument("-cbg", "--custom_bar_x_lab", type=int, default=None,
                        help="Use custom grouping strategy for PAD bar plots (0: AgeGroup; 1: AgeSex).")
    parser.add_argument("-ia", "--ignore_all", action="store_true", default=False,
                        help="Ignore 'All' feature orientations.")
    parser.add_argument("-o", "--overwrite", action="store_true", default=False, 
                        help="Overwrite if the output files already exist.")
    parser.add_argument("-pa", "--p_adjust", action="store_true", default=False,
                        help="Use adjusted p-values for pairwise correlations.")
    
    return parser.parse_args()

def generate_html(config, input_json, overwrite=False): 
    '''
    Show the description of the output in a web browser.
    '''
    if not os.path.exists(config.html_desc_outpath) or overwrite:
        with open(config.html_template_path, "r", encoding="utf-8") as f:
            html_template = f.read()
        with open(config.js_code_path, "r") as f:
            js_code = f.read()
        json_str = json.dumps(input_json, indent=2) # indent=2 for better readability
        html = (
            html_template
            .replace('__JSON_DATA__', json_str)
            .replace('__JS_CODE__', js_code)
        )
        with open(config.html_desc_outpath, "w", encoding="utf-8") as f:
            f.write(html)
            
        print(f"\nHTML file with description viewer is saved to:\n{config.html_desc_outpath}")

def load_data_matrix(config, desc):
    '''
    Load the data matrix that is used to train the models.
    '''
    if desc.data_synthetized:
        data_DF = pd.read_csv(config.datmat_path.replace(".csv", " (marked).csv"))
    else:
        data_DF = pd.read_csv(config.datmat_path)

    data_DF["SID"] = data_DF["ID"].map(lambda x: x.replace("sub-0", ""))
    data_DF.drop(columns=["ID"], inplace=True)
    data_DF["AGE_GROUP"] = pd.cut(
        data_DF["BASIC_INFO_AGE"], bins=desc.age_breaks, labels=desc.age_group_labels
    )
    if desc.sep_sex:
        data_DF["SEX"] = data_DF["BASIC_INFO_SEX"].replace({1: "M", 2: "F"})

    return data_DF

def load_model_results(config, desc, const, output_path, overwrite=False):
    '''
    Load model results of training and testing, combine them if needed, 
    and save the result dataframe to file.
    '''
    def _process_results(results, cols, label, ori_name, sep_sex=desc.sep_sex):
        processed_results = pd.DataFrame({ 
            k: v for k, v in results.items() 
            if k in cols and v is not None
        })
        if any(char.isdigit() for char in ori_name):
            processed_results.insert(0, "Type", ori_name)
        else:
            processed_results.insert(0, "Type", ori_name[:3])
        if sep_sex:
            age_group, sex = label
            if age_group in ["all", "le-44", "ge-45"]:
                age_group = {"all": "", "le-44": "Y", "ge-45": "O"}[age_group]
            processed_results.insert(2, "AgeGroup", age_group)
            processed_results.insert(2, "Sex", sex)
        else:
            if label in ["all", "le-44", "ge-45"]:
                label = {"all": "", "le-44": "Y", "ge-45": "O"}[label]
            processed_results.insert(2, "AgeGroup", label) 

        return processed_results

    def _combine_train_test_results(train_results_DF, result_DF):
        train_results_DF.rename(columns={
            "TrainingAge"       : "Age", 
            "TrainingPredAge"   : "PredictedAge", 
            "TrainingPAD"       : "PredictedAgeDifference", 
            "TrainingPADAC"     : "CorrectedPAD", 
            "TrainingCorPredAge": "CorrectedPredictedAge"
        }, inplace=True)
        test_results_DF = result_DF.copy(deep=True)
        test_results_DF.insert(1, "TrainTest", "Test")
        test_results_DF.drop(columns=["Model", "NumberOfFeatures"], inplace=True)
        train_results_DF.insert(1, "TrainTest", "Train")
        combined_results_DF = pd.concat([train_results_DF, test_results_DF])
        combined_results_DF.sort_values(by=["Type"] + desc.label_cols + ["TrainTest"], ascending=False)

        return combined_results_DF

    main_results_list, train_results_list = [], []
    selected_features = { o: {} for o in desc.feature_oris }

    for label in desc.label_list:
        group_name = "_".join(label) if desc.sep_sex else label

        for ori_name in desc.feature_orientations:
            result_path = config.result_path.format(group_name, ori_name)

            if os.path.exists(result_path):
                print(f"Loading {os.path.basename(result_path)} ...")
                with open(result_path, "r", errors="ignore") as f:
                    results = json.load(f)

                main_cols = [desc.sid_name] + const.test_result_cols + const.model_info_cols
                main_results = _process_results(results, main_cols, label, ori_name)
                main_results_list.append(main_results)

                if desc.traintest:
                    train_results = _process_results(results, const.train_result_cols, label, ori_name)
                    train_results_list.append(train_results)

                if any(char.isdigit() for char in ori_name):
                    selected_features[ori_name][group_name] = results["FeatureNames"]
                else:
                    selected_features[ori_name[:3]][group_name] = results["FeatureNames"]

    result_DF = pd.concat(main_results_list, ignore_index=True)
    result_DF.insert(1, "SID", result_DF[desc.sid_name].map(lambda x: x.replace("sub-0", "")))
    result_DF.drop(columns=[desc.sid_name], inplace=True)

    if desc.traintest: 
        train_results_DF = pd.concat(train_results_list, ignore_index=True)
        train_results_DF.insert(1, "SID", train_results_DF["TrainingSubjID"].map(lambda x: x.replace("sub-0", "")))
        train_results_DF.drop(columns=["TrainingSubjID"], inplace=True)
        combined_results_DF = _combine_train_test_results(train_results_DF, result_DF)
    else:
        combined_results_DF = result_DF

    if not os.path.exists(output_path) or overwrite:
        combined_results_DF.to_csv(output_path, index=False)
        print(f"\nResults are saved to:\n{output_path}")

    return result_DF, combined_results_DF, selected_features

def save_model_infos(DF, desc, output_path, overwrite=False):
    '''
    Save selected model types and number of features to file.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        DF["Info"] = DF.apply(lambda x: f"{x['Model']} ({x['NumberOfFeatures']})", axis=1)
        (
            DF.loc[:, desc.label_cols + ["Type", "Info"]]
            .drop_duplicates() 
            .pivot(index=desc.label_cols, columns="Type", values="Info")
            .loc[:, desc.feature_oris] # re-order
            .iloc[::-1] # reverse rows
            .to_csv(output_path)
        )
        print(f"\nModel infos are saved to:\n{output_path}")

def save_subj_infos(DF, desc, output_path, overwrite=False):
    '''
    Aggregate the number of participants alongside the median and STDs of their age
    for each group to file.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        (
            DF.loc[:, ["SID", "Age"] + desc.label_cols]
            .drop_duplicates("SID")
            .groupby(desc.label_cols)["Age"]
            .agg(["count", "median", "std"])
            .rename(columns = {"count": "N", "median": "Median", "std": "STD"})
            .reset_index()
            .to_csv(output_path, index=False)
        )
        print(f"\nSubject infos are saved to:\n{output_path}")

def force_agesex_cols(DF, DF2):
    '''
    Mandatory addition of columns to separate gender and age groups
    for customized x-axis labels in bar plots.
    <Inputs>
    - DF: the 'model results' DataFrame containing 'SID' and 'Age' columns.
    - DF2: the 'subject info' DataFrame containing 'SID' and 'BASIC_INFO_SEX' columns.
    '''
    if "SID" not in DF.columns or "Age" not in DF.columns:
        raise ValueError("The input DataFrame must contain 'SID' and 'Age' columns.")
    if "SID" not in DF2.columns or "BASIC_INFO_SEX" not in DF2.columns:
        raise ValueError("The second DataFrame must contain 'SID' and 'BASIC_INFO_SEX' columns.")

    DF = DF.copy(deep=True)                     
    age_group_labels = ["le-44", "ge-45"] # no other options for now
    age_breaks = [0, 44, np.inf]
    DF["AgeGroup"] = pd.cut(
        DF["Age"], bins=age_breaks, labels=age_group_labels
    )
    DF2 = DF2.loc[:, ["SID", "BASIC_INFO_SEX"]].copy(deep=True)
    DF2["Sex"] = DF2["BASIC_INFO_SEX"].replace({1: "M", 2: "F"})
    DF2.drop(columns=["BASIC_INFO_SEX"], inplace=True)
    if "Sex" in DF.columns:
        DF.drop(columns=["Sex"], inplace=True) # force re-creation
    DF = DF.merge(DF2, left_on="SID", right_on="SID", how="left")  
    DF["AgeSex"] = DF["AgeGroup"].astype(str) + "_" + DF["Sex"]

    return DF

def make_long_result_DF(DF, desc):
    '''
    Modify and transform the 'model results' DataFrame into long format.
    '''
    long_DF = (
        DF.rename(columns={
            "PredictedAgeDifference": "PAD", 
            "CorrectedPAD": "PAD_ac"
        })
        .loc[:, ["SID", "Age"] + desc.label_cols + ["Type", "PAD", "PAD_ac"]]
        .melt(
            id_vars = ["SID", "Age"] + desc.label_cols + ["Type"], 
            value_vars = ["PAD", "PAD_ac"], 
            var_name = "PAD_type", 
            value_name = "PAD_value"
        )
        .sort_values(
            by=desc.label_cols, 
            ascending=False
        )
    )
    long_DF["PAD_abs_value"] = long_DF["PAD_value"].abs()
    if desc.sep_sex:
        long_DF["AgeSex"] = long_DF["AgeGroup"] + "_" + long_DF["Sex"]

    return long_DF

def make_grouped_result_DF(DF, desc, pad_type):
    '''
    Make a dictionary containing a wide-format result_DF for each group
    to be used in correlation analysis.
    '''
    DF = DF[DF["PAD_type"] == pad_type] 
    grouped_result_DF = {}

    # for label in desc.label_list: 
    #     group_name = "_".join(label) if desc.sep_sex else label
    renamed_labels = DF["AgeSex"].unique() if desc.sep_sex else DF["AgeGroup"].unique()

    for group_name in renamed_labels:
        if desc.sep_sex:
            sub_df = DF.query("AgeSex == @group_name")
        else:
            sub_df = DF.query("AgeGroup == @group_name")

        grouped_result_DF[group_name] = (
            sub_df.loc[:, ["SID", "Type", "PAD_value"]]
            .pivot(index="SID", columns="Type", values="PAD_value")
            .reset_index()
            .rename(columns={"index": "SID"})
        )

    return grouped_result_DF

def calc_pairwise_corr(DF_dict, targ_cols, desc, output_path,    
                       output_cols=['X', 'Y', 'n', 'r', 'CI95%', 'p-unc', 'p-corr'], 
                       sort_by='p-unc', p_adj='bonf', creating=True, overwrite=False):
    '''
    Calculate pairwise correlation between target features for each group 
    and save to an Excel file.
    '''
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            creating = False

    pw_corr_list = []
    for group_name, DF in DF_dict.items():
        if creating:
            pw_corr = (
                pg.pairwise_corr(DF.loc[:, targ_cols], padjust=p_adj)
                .sort_values(by=sort_by)[output_cols]
            )
            if not os.path.exists(output_path):
                pw_corr.to_excel(output_path, sheet_name=group_name, index=False)
            else:
                with pd.ExcelWriter(output_path, mode='a') as writer: 
                    pw_corr.to_excel(writer, sheet_name=group_name, index=False)
        else:
            pw_corr = pd.read_excel(output_path, sheet_name=group_name)

        pw_corr.insert(0, desc.grouping_col, group_name)
        pw_corr_list.append(pw_corr)

    if creating:
        print(f"\nPairwise comparisons is saved to:\n{output_path}")

    return pd.concat(pw_corr_list, ignore_index=True)

def make_feature_DF(ori_name, feature_list, domain_approach_mapping):
    '''
    Make a DataFrame containing the domain and approach labels for each feature, 
    as well as the proportion information of each domain and approach.
    '''
    dict_list = []
    for feature in feature_list:

        if "MRI" in feature:
            dat_type, task, method, cond, hemi, region, var_name = feature.split("_")
            
            if dat_type == "STRUCTURE":
                if ori_name == "ALL": 
                    dict_list.append({
                        "domain": cond, # "GM", "WM", or "NULL", 
                        "approach": "STR", 
                        "feature": feature
                    })
                else: 
                    dict_list.append({
                        "domain": var_name, # "VOLUME", "ThickAvg", or "FA"
                        "approach": cond, # "GM", "WM", or "NULL"
                        "feature": feature
                    })
            else: # "FUNCTIONAL"
                dict_list.append({
                    # "domain": dat_type, # "MEMORY", "MOTOR", or "LANGUAGE"
                    "domain": task, # "SPEECHCOMP", "WORDNAME", "MST", or "GFORCE"
                    "approach": "MRI", 
                    "feature": feature
                })

        elif "EEG" in feature:
            dat_type, task, method, cond, hemi, region, time_freq_var = feature.split("_")[:7]
            
            dict_list.append({
                # "domain": dat_type, # "MEMORY", "MOTOR", or "LANGUAGE"
                "domain": task, # "EXCLUSION", "OSPAN", "GOFITTS", or "BILPRESS"
                "approach": "EEG", 
                "feature": feature
            })

        elif "BEH" in feature: 
            dat_type, task, method, cond, var_name = feature.split("_")

            if ori_name == "ALL": 
                dict_list.append({
                    "domain": dat_type, # "MEMORY", "MOTOR", or "LANGUAGE"
                    "approach": "BEH", 
                    "feature": feature
                })
            else: 
                dict_list.append({
                    "domain": task, 
                    "approach": dat_type, # "MEMORY", "MOTOR", or "LANGUAGE"
                    "feature": feature
                })

    feature_DF = pd.DataFrame(dict_list)
    approach_num = feature_DF["approach"].value_counts()
    approach_pr = feature_DF["approach"].value_counts(normalize=True) * 100
    within_approach_domain_num = feature_DF.groupby("approach")["domain"].value_counts()   
    within_approach_domain_pr = feature_DF.groupby("approach")["domain"].value_counts(normalize=True) * 100
    feature_DF["approach_num"] = feature_DF["approach"].map(approach_num)
    feature_DF["approach_and_num"] = feature_DF.apply(
        lambda x: f"{x['approach']} ({x['approach_num']})", axis=1)
    feature_DF["approach_pr"] = feature_DF["approach"].map(approach_pr)
    feature_DF["approach_and_pr"] = feature_DF.apply(
        lambda x: f"{x['approach']}<br>({x['approach_pr']:.1f}%)" if x['approach_pr'] > 7 
        else f"{x['approach']} ({x['approach_pr']:.1f}%)", axis=1)
    feature_DF["domain_num"] = feature_DF.apply(
        lambda x: within_approach_domain_num[x["approach"]][x["domain"]], axis=1)
    feature_DF["domain_and_num"] = feature_DF.apply(
        lambda x: f"{x['domain']} ({x['domain_num']})", axis=1)  
    feature_DF["domain_pr"] = feature_DF.apply(
        lambda x: within_approach_domain_pr[x["approach"]][x["domain"]], axis=1)
    feature_DF["overall_pr"] = feature_DF.apply(
        lambda x: within_approach_domain_pr[x["approach"]][x["domain"]] * x["approach_pr"], axis=1)
    feature_DF["domain_and_pr"] = feature_DF.apply(
        lambda x: f"{x['domain']}<br>({x['domain_pr']:.1f}%)" if x['overall_pr'] > 300 
        else f"{x['domain']} ({x['domain_pr']:.1f}%)", axis=1)
    
    return feature_DF

## Main function: ---------------------------------------------------------------------

def main():
    args = define_arguments()
    config = Config(args)
    const = Constants()
    color_dicts = ColorDicts()
    os.makedirs(config.output_folder, exist_ok=True)

    with open(config.desc_path, 'r', errors='ignore') as f:
        desc_json = json.load(f)

    desc = Description(desc_json, const, args)
    generate_html(config, desc_json, overwrite=args.overwrite)
    
    basic_q_features = basic_Q_features()
    st_features = ST_features()
    # domain_approach_mapping = domain_approach_mapping_dict()

    pad_type = "PAD_ac" if args.use_padac else "PAD"
    suffix = " (ignore 'All')" if args.ignore_all else ""

    ## Load the data matrix that is used to train the models:
    data_DF = load_data_matrix(config, desc)

    ## Load model results:
    result_DF, combined_results_DF, selected_features = load_model_results(
        config, desc, const, 
        output_path=config.model_results_outpath,
        overwrite=args.overwrite
    )

    ## Save some general information: 
    save_model_infos(
        DF=result_DF.copy(deep=True), 
        desc=desc, 
        output_path=config.model_infos_outpath,
        overwrite=args.overwrite
    )    

    save_subj_infos(
        DF=result_DF, 
        desc=desc, 
        output_path=config.subj_infos_outpath.format(""),
        overwrite=args.overwrite
    )

    if desc.traintest:
        save_subj_infos(
            DF=combined_results_DF.query("TrainTest == 'Train'"), 
            desc=desc, 
            output_path=config.subj_infos_outpath.format(" (training set)"),
            overwrite=args.overwrite
        )
    
    ## Plot feature importances for each model:
    if not desc.use_pretrained: 
        for ori_name in desc.feature_orientations: 
            for label in desc.label_list: 
                group_name = "_".join(label) if desc.sep_sex else label
                feature_importances = pd.read_csv(
                    config.feature_path.format(group_name, ori_name), header=None
                )
                feature_importances.sort_values(by=1, ascending=False, key=abs, inplace=True)
                fw = len(feature_importances) * 0.3
                new_group_name = group_name.replace("le-44", "Y").replace("ge-45", "O")
                plot_feature_importances(
                    feature_importances=feature_importances, 
                    x_lim=None, # since those values of LGBM are greater than the [-1, 1] range
                    fig_size=(8, fw), 
                    output_path=config.feature_importance_outpath.format(ori_name[:3], new_group_name), 
                    overwrite=args.overwrite
                )
    
    ## Plot the relationship between real and predicted ages:
    for ori_name in desc.feature_oris:           
        df_temp = combined_results_DF.query("Type == @ori_name")
        df1 = df_temp.loc[:, ["Age", "PredictedAge"]]
        df1.insert(1, "pad_type", "Raw")
        df2 = df_temp.loc[:, ["Age", "CorrectedPredictedAge"]]
        df2.insert(1, "pad_type", "AC")
        df2.rename(columns={"CorrectedPredictedAge": "PredictedAge"}, inplace=True)

        plot_real_pred_age(
            DF=pd.concat([df1, df2], axis=0), 
            y_lab="PredictedAge", 
            color_hue="pad_type", 
            color_dict=color_dicts.pred_type, 
            output_path= config.pred_vs_real_age_outpath.format(ori_name),
            overwrite=args.overwrite
        )
    
    plot_multiple_corr_with_stats(
        DF=combined_results_DF, 
        x_lab="Age", 
        y_lab="CorrectedPAD", 
        z_lab="Type", 
        output_path=config.residual_regress_outpath.format("corrected "),
        overwrite=args.overwrite
    )

    plot_color_legend(
        color_dict=color_dicts.pred_type, 
        output_path=config.pred_type_legend_outpath,
        fig_size=(1.5, .5), n_cols=2, box_size=.5, overwrite=args.overwrite
    )

    ## If the data was synthetized, compare real and synthetic data for each group:
    if desc.data_synthetized: 
        for label in desc.label_list: 
            group_name = "_".join(label) if desc.sep_sex else label
            new_group_name = group_name.replace("le-44", "Y").replace("ge-45", "O")

            if desc.sep_sex:
                age_group, sex = label
                sub_datmat = data_DF.query("AGE_GROUP == @age_group & SEX == @sex")
            else:
                sub_datmat = data_DF.query("AGE_GROUP == @label")

            ## Get the targeted features
            targeted_features = []
            num_f_per_ori = 2
            for ori_name in desc.feature_oris:
                for x in range(num_f_per_ori):
                    targ_f = selected_features[ori_name][group_name][x]
                    targeted_features.append(targ_f)

            ## Check if the synthetic data matches the original one in distribution:
            plot_real_syn_data(
                DF=sub_datmat, 
                group_name=group_name, 
                targeted_features=targeted_features, 
                n_rows=len(desc.feature_orientations), 
                n_cols=num_f_per_ori, 
                color_dict=color_dicts.real_synth, 
                output_path=config.real_vs_synth_data_outpath.format(new_group_name), 
                overwrite=args.overwrite
            )

            ## Compate the correlations among selected features between real and synthetic data:
            for S_or_R in ["Synthetic", "Real"]: 
                plot_cormat(
                    DF=sub_datmat.query("R_S == @S_or_R"), 
                    targ_cols=targeted_features, 
                    shorter_xcol_names=True, 
                    figsize=(12, 4), 
                    output_path=config.feature_cormat_outpath.format(S_or_R, new_group_name), 
                    overwrite=args.overwrite
                )

    ## Modify and transform results_DF into long format:
    long_result_DF = make_long_result_DF(result_DF, desc)

    ## If custom grouping strategy is specified for PAD bar plots:
    fname_add = ""
    if args.custom_bar_x_lab is not None:
        bar_df = force_agesex_cols(DF=long_result_DF, DF2=data_DF)
        custom_bar_x_lab = ["AgeGroup", "AgeSex"][args.custom_bar_x_lab]
        fname_add = f" (grouped by '{custom_bar_x_lab}')"

    ## Plot PAD bars: 
    plot_pad_bars(
        DF=long_result_DF if args.custom_bar_x_lab is None else bar_df, 
        x_lab=desc.grouping_col if args.custom_bar_x_lab is None else custom_bar_x_lab, 
        one_or_many="many", 
        color_dict = color_dicts.pad_bars, 
        output_path=config.pad_barplot_outpath.format(suffix + fname_add), 
        overwrite=args.overwrite
    )

    for ori_name in desc.feature_oris: 
        plot_pad_bars(
            DF=long_result_DF.query("Type == @ori_name") if args.custom_bar_x_lab is None else bar_df.query("Type == @ori_name"), 
            x_lab=desc.grouping_col if args.custom_bar_x_lab is None else custom_bar_x_lab, 
            one_or_many="one", 
            color_dict = color_dicts.pad_bars, 
            output_path=config.pad_barplot_outpath.format(fname_add + f" ({ori_name})"), 
            y_max=const.pad_bar_y_lims[ori_name] if ori_name in const.pad_bar_y_lims.keys() else None,
            overwrite=args.overwrite
        )

    plot_color_legend(
        color_dict=color_dicts.pad_bars, 
        output_path=config.pad_type_legend_outpath,
        fig_size=(2, .5), n_cols=2, box_size=0.5, overwrite=args.overwrite
    )

    ## Correlation analyses: 
    grouped_result_DF = make_grouped_result_DF(long_result_DF, desc, pad_type)
    pad_with_interested_features = []

    for group_name in grouped_result_DF.keys():
        sub_result_df = grouped_result_DF[group_name]
        sub_result_df = sub_result_df.merge(
            data_DF.loc[:, ["SID"] + basic_q_features + st_features], 
            on="SID", how="left"
        )
        pad_with_interested_features.append(sub_result_df)

        ## Correlations among PAD values:
        plot_cormat(
            DF=sub_result_df, 
            targ_cols=sorted(desc.feature_oris), 
            yr=90, 
            output_path=config.pad_cormat_outpath.format(pad_type, group_name, suffix), 
            figsize=(3, 3) if len(desc.feature_orientations) <= 3 else (4, 4),
            overwrite=args.overwrite
        )

        ## Correlations between PAD values and standardized test features: 
        plot_cormat(
            DF=sub_result_df, 
            targ_cols=st_features, 
            corrwith_cols=sorted(desc.feature_oris), 
            output_path=config.pad_cormat_outpath.format(f"{pad_type} & standardized features", group_name, f" (N={len(sub_result_df)})"), 
            figsize=(8, 6),
            overwrite=args.overwrite
        )

        ## Correlations between PAD values and questionnaire features:
        plot_cormat(
            DF=sub_result_df, 
            targ_cols=basic_q_features, 
            corrwith_cols=sorted(desc.feature_oris), 
            output_path=config.pad_cormat_outpath.format(f"{pad_type} & questionnaire features", group_name, f" (N={len(sub_result_df)})"), 
            figsize=(8, 10),
            overwrite=args.overwrite
        )

    pad_with_interested_features = pd.concat(pad_with_interested_features)
    pad_with_interested_features.to_csv(config.pad_cormat_data_outpath, index=False)

    ## Calculate pairwise correlations between PAD values:
    corr_DF = calc_pairwise_corr(
        DF_dict=grouped_result_DF, 
        targ_cols=sorted(desc.feature_oris), 
        desc=desc, 
        output_path=config.pad_pairwise_corr_outpath.format(pad_type), 
        overwrite=args.overwrite
    )

    ## Plot correlations with statistics:
    try:
        for group_name, sub_DF in grouped_result_DF.items():
            sub_corr_df = corr_DF[corr_DF[desc.grouping_col] == group_name]

            for t1, t2 in [("BEH", "FUN"), ("BEH", "STR"), ("FUN", "STR")]:            
                corr_table = sub_corr_df.query("X == @t1 & Y == @t2")
                if corr_table.empty:
                    corr_table = sub_corr_df.query("X == @t2 & Y == @t1")

                plot_corr_with_stats(
                    DF=sub_DF, 
                    corr_table=corr_table, 
                    x_lab=t1, 
                    y_lab=t2, 
                    p_apply="p-corr" if args.p_adjust else "p-unc", 
                    output_path=config.pad_scatter_outpath.format(pad_type, group_name, t1, t2), 
                    overwrite=args.overwrite
                )

    except Exception as e:
        print(f"\nError in plot_corr_with_stats: {e}")

## Finally: ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
    print("\nDone!\n")

## Incomplete: ========================================================================

# output_path = output_path.replace(".png", " (num).png")
# parent_col=["approach_and_num", "approach_and_pr"][0]
# label_col=["domain_and_num", "domain_and_pr"][0]

    # ## Prepare a smaller, non-duplicatie dataframe:
    # model_info_DF = (
    #     result_DF.loc[:, desc.label_cols + ["Type", "Model", "NumberOfFeatures"]]
    #     .drop_duplicates()
    # )

    # ## Make a dataframe with hierachical labels for selected features:
    # feature_DF_list = [] # to concat and save to .csv

    # for ori_name in desc.feature_orientations: 

    #     ## One dataframe per group:
    #     feature_DF_dict = {
    #         group_name: make_feature_DF(
    #             ori_name, feature_list, domain_approach_mapping
    #         ) 
    #         for group_name, feature_list in selected_features[ori_name].items()
    #     }

    #     for group_name, feature_DF in feature_DF_dict.items(): 
    #         feature_df = pd.DataFrame(feature_DF)
    #         feature_df.insert(0, "Type", ori_name[:3])
    #         feature_df.insert(1, "Group", group_name)
    #         feature_DF_list.append(feature_df)

    #     ## Prepare annotation for each subplot:
    #     subplot_annots, fig_titles = {}, {}
    #     for label in desc.label_list: 
    #         if desc.sep_sex:
    #             age_group, sex = label
    #             group_name = f"{age_group}_{sex}"
    #             model_info = model_info_DF.query(
    #                 "Type == @ori_name & AgeGroup == @age_group & Sex == @sex"
    #             )
    #         else:
    #             group_name = label
    #             model_info = model_info_DF.query(
    #                 "Type == @ori_name & AgeGroup == @group_name"
    #             )
    #         model_type = model_info['Model'].iloc[0]
    #         n_features = model_info['NumberOfFeatures'].iloc[0]
    #         subplot_annots[group_name] = f"{group_name} ({model_type} - {n_features})" 
    #         fig_titles[group_name] = f"{model_type} ({n_features})"

    #     ## One set of sunburst charts per feature type:
    #     fp = os.path.join(config.output_folder, config.sunburst_outpath.replace("<FeatureType>", ori_name[:3]))
    #     plot_many_feature_sunbursts(
    #         feature_DF_dict=feature_DF_dict, 
    #         parent_col=parent_col, 
    #         label_col=label_col, 
    #         color_dict=color_dicts.sunburst[ori_name], 
    #         fig_title=f"{ori_name[:3]}", 
    #         subplot_annots=subplot_annots, 
    #         one_or_many="many", 
    #         output_path=fp, 
    #         overwrite=args.overwrite
    #     )
        
    #     ## One sunburst chart per group:
    #     for group_name, feature_DF in feature_DF_dict.items():
    #         plot_feature_sunburst(
    #             feature_DF=feature_DF, 
    #             parent_col=parent_col, 
    #             label_col=label_col, 
    #             color_dict=color_dicts.sunburst[ori_name], 
    #             fig_title=fig_titles[group_name], 
    #             one_or_many="one", 
    #             output_path=fp.replace(".png", f" ({group_name}).png"), 
    #             overwrite=args.overwrite
    #         )

    # ## Save the feature dataframe:
    # output_path = os.path.join(config.output_folder, config.feature_df_outpath)
    # if not os.path.exists(os.path.dirname(output_path)) or args.overwrite:
    #     feature_DF_long = pd.concat(feature_DF_list)
    #     feature_DF_long.rename(columns={
    #         "feature"     : "Feature", 
    #         "domain"      : "Level_2", 
    #         "domain_num"  : "L2_num", 
    #         "approach"    : "Level_1", 
    #         "approach_num": "L1_num"
    #     }, inplace=True)
    #     feature_DF_long = feature_DF_long.loc[:, [
    #         "Type", "Group", "Level_1", "L1_num", "Level_2", "L2_num" # , "Feature"
    #     ]]
    #     feature_DF_long.drop_duplicates(inplace=True)
    #     feature_DF_long.to_csv(output_path, index=False)
    #     print(f"\nFeature dataframe saved to:\n{output_path}")