#!/usr/bin/env python

# Usage: python predict_age_ML.py 

import os
import sys
import json
import pickle
import shutil
import argparse
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

import optuna
import optunahub
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.join(os.getcwd(), "..", "src"))
from data_preparing import DataManager
from feature_engineering import FeatureReducer, FeatureSelector, BorutaSelector
from utils import ST_features, platform_features

## Classes: ===========================================================================

class Constants:
    def __init__(self, args):
        ## The age groups defined by different methods:
        self.age_groups = { 
            "no-cut": {
                "all": (0, np.inf)
            }, 
            "cut-40-41": {
                "le-40" : ( 0, 40),    # less than or equal to
                "ge-41" : (41, np.inf) # greater than or equal to
            }, 
            "cut-44-45": {
                "le-44" : ( 0, 44),
                "ge-45" : (45, np.inf) 
            }, 
            "cut-8-wais": {
                "le-24": ( 0, 24), 
                "25-29": (25, 29), 
                "30-34": (30, 34),
                "35-44": (35, 44), 
                "45-54": (45, 54), 
                "55-64": (55, 64), 
                "65-69": (65, 69), 
                "ge-70": (70, np.inf)
            }, 
            "every-5-yrs": {
                "le-24": ( 0, 24), 
                "25-29": (25, 29), 
                "30-34": (30, 34), 
                "35-39": (35, 39), 
                "40-44": (40, 44), 
                "45-49": (45, 49), 
                "50-54": (50, 54), 
                "55-59": (55, 59), 
                "60-64": (60, 64), 
                "65-69": (65, 69), 
                "70-74": (70, 74), 
                "ge-75": (75, np.inf)
            }, 
            "elderly": {
                "ge-60": (60, np.inf)
            }
        }

        ## The correspondence between domains and approaches:
        if args.standardize_tests:
            st_features = ST_features()
            st_features_to_drop = [
                [f"LANGUAGE_ST_SCALED_{f1}", f"MEMORY_ST_SCALED_{f2}", f"MOTOR_ST_SCALED_{f3}"]
                for (f1, f2, f3) in list(product(
                    ["SIMILARITY", "VOCABULARY", "INFORMATION", "SUM"], 
                    ["AudImm", "VisImm", "ImmMem", "WorMem", "LogMemI", "FacI", "VerPairI", "FamPicI", "LetNumSeq", "SpaForward", "SpaBackward", "SpaTotal"], 
                    ["FineMotor", "Balance", "ProcessingSpeed"]
                ))
            ]
            self.domain_approach_mapping = {}
            for x, (f1, f2, f3) in enumerate(st_features_to_drop):
                self.domain_approach_mapping[f"ST-{x+1}"] = {
                    "domains": [ f for f in st_features if f not in (f1, f2, f3) ], 
                    "approaches": [""]
                }
        elif args.platform_features:
            pf = platform_features()
            self.domain_approach_mapping = {}
            for domain in ["MOTOR", "MEMORY", "LANGUAGE"]:
                self.domain_approach_mapping[domain] = {
                    "domains": [ f for f in pf if f.startswith(domain) ],
                    "approaches": [""]
                }
        else:
            self.domain_approach_mapping = { 
                "STRUCTURE": {
                    "domains": ["STRUCTURE"],
                    "approaches": ["MRI"]
                },
                "BEH": {
                    "domains": ["MOTOR", "MEMORY", "LANGUAGE"],
                    "approaches": ["BEH"]
                },
                "FUNCTIONAL": {
                    "domains": ["MOTOR", "MEMORY", "LANGUAGE"],
                    "approaches": ["EEG", "MRI"]
                }
            }
            self.all_domain_approach_mapping = {
                "ALL": {
                    "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
                    "approaches": ["MRI", "BEH", "EEG"]
                }
            }
            if not args.no_all_mappings:
                self.domain_approach_mapping.update(self.all_domain_approach_mapping)

            elif args.only_all_mapping:
                self.domain_approach_mapping = self.all_domain_approach_mapping

        ## The names of models to evaluate:
        self.model_names = [ 
            "ElasticNet", 
            "CART", # DecisionTreeRegressor
            "RF",   # RandomForestRegressor
            "XGBM", # xgb.XGBRegressor
            "LGBM"  # lgb.LGBMRegressor
        ]

        ## The number of participants in each balanced group:
        self.balance_to_num = {
            "cut-8-wais": {
                "CTGAN": 60, 
                "TVAE": 60,
                "SMOTENC": 60, 
                "downsample": 15, 
                "bootstrap": 15
            }, 
            "cut-44-45": {
                "CTGAN": 60*4, 
                "TVAE": 60*4, 
                "SMOTENC": 60*4, 
                "downsample": 15*4, 
                "bootstrap": 15*4
            }
        }
        ## Number of parallel threads to use:
        self.n_jobs = 16

class Config:
    def __init__(self, args, constants):
        self.source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.raw_data_path = os.path.join(self.source_path, "data", "rawdata", "DATA_ses-01_2024-12-09.csv")
        self.inclusion_data_path = os.path.join(self.source_path, "data", "rawdata", "InclusionList_ses-01.csv")
        self.syndata_folder = os.path.join(self.source_path, "data", "syndata")
        if args.pretrained_model_folder is not None:
            self.pretrained_model_folder = os.path.join(self.source_path, "outputs", args.pretrained_model_folder)
        self = self._setup_vars(args, constants)
        self = self._setup_out_folder(args)
        self.logging_outpath                 = os.path.join(self.out_folder, "log.txt")
        self.description_outpath             = os.path.join(self.out_folder, "description.json")
        self.prepared_data_outpath           = os.path.join(self.out_folder, "prepared_data.csv")
        self.split_ids_outpath               = os.path.join(self.out_folder, "train_and_test_ids.json")
        self.scaler_outpath_format           = os.path.join(self.out_folder, "scaler_{}.pkl")
        self.selector_outpath_format         = os.path.join(self.out_folder, "selector_{}_{}_{}.pkl")
        self.reducer_outpath_format          = os.path.join(self.out_folder, "reducer_{}_{}.pkl")
        self.pc_loadings_outpath_format      = os.path.join(self.out_folder, "PC_loadings_{}_{}.xlsx")
        self.rc_loadings_outpath_format      = os.path.join(self.out_folder, "RC_loadings_{}_{}.xlsx")
        self.dropped_features_outpath_format = os.path.join(self.out_folder, "dropped_features_{}_{}.txt")
        self.model_outpath_format            = os.path.join(self.out_folder, "models_{}_{}_{}.pkl")
        self.features_outpath_format         = os.path.join(self.out_folder, "features_{}_{}.csv")
        self.training_results_outpath_format = os.path.join(self.out_folder, "training_results_{}_{}.json")
        self.results_outpath_format          = os.path.join(self.out_folder, "results_{}_{}.json")
        self.failure_record_outpath          = os.path.join(self.out_folder, "failure_record.txt")

    def _setup_vars(self, args, constants):
        self.seed = args.seed if args.seed is not None else np.random.randint(0, 10000)
        self.by_gender = [False, True][args.by_gender]
        self.age_method = ["no-cut", "cut-40-41", "cut-44-45", "cut-8-wais", "elderly"][args.age_method]
        self.age_bin_labels = list(constants.age_groups[self.age_method].keys())
        self.age_boundaries = list(constants.age_groups[self.age_method].values())
        self.age_correction_method = ["Zhang et al. (2023)", "Beheshti et al. (2019)", None][args.age_correction_method]
        self.age_correction_groups = ["cut-8-wais", "every-5-yrs"][0]
        self.pad_age_groups = list(constants.age_groups[self.age_correction_groups].keys())
        self.pad_age_breaks = [ 0 ] + [ x for _, x in list(constants.age_groups[self.age_correction_groups].values()) ] 
        
        self.balancing_method = [None, "CTGAN", "TVAE", "SMOTENC", "downsample", "bootstrap"][args.balancing_method]
        self.balancing_groups = ["cut-8-wais", "cut-44-45"][args.balancing_groups]
        self.balance_to_num = None
        if self.balancing_method is not None:
            if args.sample_size is None:
                self.balance_to_num = constants.balance_to_num[self.balancing_groups][self.balancing_method]
            else:
                self.balance_to_num = args.sample_size

        if args.testset_ratio is not None:
            self.split_with_ids = None
            self.testset_ratio = args.testset_ratio
        elif args.split_with_ids is not None:
            with open(args.split_with_ids, 'r') as f:
                self.split_with_ids = json.load(f)
            self.testset_ratio = len(self.split_with_ids["Test"]) / (len(self.split_with_ids["Train"]) + len(self.split_with_ids["Test"]))
        else:
            raise ValueError("Please specify either --testset_ratio or --split_with_ids")

        if args.feature_selection:
            self.feature_transformer = "selector"
            self.fs_thresh_method = ["fixed_threshold", "explained_ratio"][args.fs_thresh_method]
        elif args.apply_pca: 
            self.feature_transformer = "reducer"
        else:
            self.feature_transformer = None

        if args.training_model is not None:
            self.included_models = [ constants.model_names[args.training_model] ]
        else:
            self.included_models = constants.model_names

        return self

    def _setup_out_folder(self, args):
        prefix = datetime.today().strftime('%Y-%m-%d')

        if args.use_prepared_data:
            balancing_method = os.path.basename(os.path.dirname(args.use_prepared_data)).split("_")[0]
            prefix += f"_{balancing_method.lower()}"
        elif self.balancing_method is not None:
            prefix += f"_{self.balancing_method.lower()}"
        else:
            prefix += "_original"

        if args.age_method == 0:
            prefix += "_age-0"

        if args.by_gender == 0:
            prefix += "_sex-0"

        # prefix += f"_tsr-{self.testset_ratio:.1f}"

        if args.standardize_tests:
            prefix += "_ST"

        if args.pretrained_model_folder is not None:
            self.out_folder = os.path.join(self.source_path, "outputs", f"{prefix} ({args.pretrained_model_folder})")
            # self.out_folder = os.path.join(self.source_path, "outputs", f"{prefix}_pre-trained")
        elif args.training_model is not None:
            model_name = self.included_models[0]
            self.out_folder = os.path.join(self.source_path, "outputs", f"{prefix}_{model_name}")
        else:
            self.out_folder = os.path.join(self.source_path, "outputs", prefix)

        if args.overwrite and os.path.exists(self.out_folder):
            shutil.rmtree(self.out_folder)

        while os.path.exists(self.out_folder): # make sure the output folder does not exist:
            self.out_folder = self.out_folder + "+"

        os.makedirs(self.out_folder)

        return self

## Functions: =========================================================================

def convert_np_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert np.ndarray to list
    elif isinstance(obj, np.generic): 
        return obj.item()   # Convert np.generic to scalar
    elif isinstance(obj, pd.Series):
        return obj.tolist() # Convert pd.Series to list
    elif isinstance(obj, list):
        return [ convert_np_types(i) for i in obj ] 
    elif isinstance(obj, dict):
        return { k: convert_np_types(v) for k, v in obj.items() } 
    else:
        return obj
    
def initialization():
    '''
    Parse command line arguments,
    define the config and constants objects, 
    make the output folder, 
    set external random seed, 
    initialize the logger, 
    copy the script file to the output folder, 
    and save the description of the current execution as a JSON file.
    '''
    def _define_arguments():
        parser = argparse.ArgumentParser(description="")
        ## General:
        parser.add_argument("-o", "--overwrite", action="store_true", default=False, 
                            help="Overwrite the output folder if it already exists.")
        parser.add_argument("-i", "--ignore", type=int, default=0, 
                            help="Ignore the first N iterations (in case the script was interrupted by an accident and you don't want to start from the beginning).")
        parser.add_argument("-s", "--seed", type=int, default=9865, 
                            help="The value used to initialize all random number generator.")
        
        ## Data preparation:
        parser.add_argument("-dat", "--prepare_data_and_exit", action="store_true", default=False, 
                            help="Only make the data, do not train any model.")
        parser.add_argument("-upd", "--use_prepared_data", type=str, default=None, 
                            help="File path of the data to be used (.csv).")
        
        ## Balancing data such that all groups have the same number of participants:
        parser.add_argument("-bm", "--balancing_method", type=int, default=0, 
                            help="The method to balance the data (0: None, 1: CTGAN, 2: TVAE, 3: SMOTENC, 4: downsample, 5: bootstrap).")
        parser.add_argument("-bg", "--balancing_groups", type=int, default=0, 
                            help="The groups to be balanced.")
        parser.add_argument("-n", "--sample_size", type=int, default=None, 
                            help="The number of participants to up- or down-sample to. (if None, use the default number of participants per group set in constants.balance_to_num).")
        
        ## Separate data into groups:
        parser.add_argument("-age", "--age_method", type=int, default=2, 
                            help="The method to define age groups (0: 'no-cut', 1: 'cut-40-41', 2: 'cut-44-45', 3: 'cut-8-wais', 4: 'eldery').")
        parser.add_argument("-sex", "--by_gender", type=int, default=0, 
                            help="Whether to separate the data by gender (0: False, 1: True).")
        
        ## Split data into training and testing sets:
        parser.add_argument("-sid", "--split_with_ids", type=str, # default=None, 
                            default=os.path.join(os.getcwd(), "..", "outputs", "train_and_test_ids.json"), 
                            help="File path of a dictionary (.json) containing two lists of IDs to be used for splitting the data into training and testing sets. " +
                                 "If provided, the testset_ratio will be determined by its length.")
        parser.add_argument("-tsr", "--testset_ratio", type=float, default=None, 
                            help="The ratio of the testing set. " +
                                 "If provided, the split_with_ids will be ignored.")
        
        ## Feature groups to be included:
        parser.add_argument("-nam", "--no_all_mappings", action="store_true", default=False, 
                            help="Not to include 'All' domain-approach mappings for feature selection.")
        parser.add_argument("-oam", "--only_all_mapping", action="store_true", default=False, 
                            help="Include only 'All' domain-approach mappings for feature selection.")
        parser.add_argument("-st", "--standardize_tests", action="store_true", default=False, 
                            help="Use only the standardized test scores as features.")
        parser.add_argument("-pf", "--platform_features", action="store_true", default=False, 
                            help="Use platform features as features.")
        
        ## Feature engineering:
        parser.add_argument("--apply_pca", action="store_true", default=False, 
                            help="Apply PCA for feature reduction.")
        parser.add_argument("-rhcf", "--remove_highly_correlated_features", action="store_true", default=False, 
                            help="Remove highly correlated features before feature reduction (PCA).")
        parser.add_argument("-fs", "--feature_selection", action="store_true", default=False, 
                            help="Perform feature selection. This will overwrite the default setting of using PCA-based feature reduction.")
        parser.add_argument("-fstm", "--fs_thresh_method", type=int, default=1, 
                            help="The method to determine the threshold for feature selection (0: 'threshold', 1: 'explained_ratio').")
        parser.add_argument("-fst", "--fs_threshold", type=float, default=None, 
                            help="Features whose absolute importance is below this value will be removed.")
        parser.add_argument("-fsr", "--fs_explained_ratio", type=float, default=None, 
                            help="Stop selecting features when cumulative feature importance reaches this value.")
        parser.add_argument("-mfn", "--max_feature_num", type=int, default=None, 
                            help="The maximum number of features to be selected.")

        ## Model training:
        parser.add_argument("-pmf", "--pretrained_model_folder", type=str, default=None, 
                            help="The folder where the pre-trained model files (.pkl) are stored.")
        parser.add_argument("-m", "--training_model", type=int, default=None, 
                            help="The type of the model to be used for training (0: 'ElasticNet', 1: 'RF', 2: 'CART', 3: 'XGBM', 4: 'LGBM').")

        ## Age correction:
        parser.add_argument("-acm", "--age_correction_method", type=int, default=0, 
                            help="The method to correct age (0: 'Zhang et al. (2023)', 1: 'Beheshti et al. (2019)', 2: None).")
        
        return parser.parse_args()

    def _save_description(args, config, constants, logger):
        '''
        Save the description of the current execution as a JSON file.
        '''
        desc = {
            "Seed": config.seed, 
            "UsePreparedData": args.use_prepared_data, 
            "RawDataVersion": config.raw_data_path if args.use_prepared_data is None else "--", 
            "InclusionFileVersion": config.inclusion_data_path if args.use_prepared_data is None else "--", 
            "DataBalancingMethod": config.balancing_method, 
            "BalancingGroups": config.balancing_groups, 
            "NumPerBalancedGroup": config.balance_to_num
        }

        if args.use_prepared_data: 
            file_dir, _ = os.path.split(args.use_prepared_data)
            preceding_dir, folder = os.path.split(file_dir)
            if preceding_dir == config.syndata_folder:
                balancing_method, balancing_groups, balance_to_num = folder.split("_")[:3]
                desc["DataBalancingMethod"] = balancing_method
                desc["BalancingGroups"] = balancing_groups
                desc["NumPerBalancedGroup"] = balance_to_num
            else:
                try:
                    with open(os.path.join(file_dir, "description.json"), 'r', errors='ignore') as f:
                        desc_json = json.load(f)
                    desc["DataBalancingMethod"] = desc_json["DataBalancingMethod"]
                    desc["BalancingGroups"] = desc_json["BalancingGroups"]
                    desc["NumPerBalancedGroup"] = desc_json["NumPerBalancedGroup"]
                except:
                    pass # no description file

        if not args.prepare_data_and_exit: 
            desc.update({
                "SexSeparated": config.by_gender, 
                "AgeGroups": config.age_bin_labels, 
                "FeatureOrientations": list(constants.domain_approach_mapping.keys()), 
                "SkippedIterationNum": args.ignore, 
                "TestsetRatio": config.testset_ratio, 
                "UsePredefinedSplit": config.split_with_ids, 
                "UsePretrainedModels": args.pretrained_model_folder, 
                "IncludedOptimizationModels": config.included_models if args.pretrained_model_folder is None else "Depend on the previous results", 
                "AgeCorrectionMethod": config.age_correction_method, 
                "AgeCorrectionGroups": config.pad_age_groups
            })

            if config.feature_transformer == "reducer":
                desc.update({
                    "FeatureTransformer": "PCA reducer",
                    "RemoveHighlyCorrelatedFeatures": args.remove_highly_correlated_features
                })

            elif config.feature_transformer == "selector": 
                desc.update({
                    "FeatureTransformer": "Auto selector", 
                    "FSThresholdMethod": config.fs_thresh_method, 
                    "FSThreshold": args.fs_threshold, 
                    "FSExplainedRatio": args.fs_explained_ratio, 
                    "MaxFeatureNum": args.max_feature_num
                })

        ## Save the description to a JSON file:
        desc = convert_np_types(desc)
        with open(config.description_outpath, 'w', encoding='utf-8') as f:
            json.dump(desc, f, ensure_ascii=False)

        logger.info("The description of the current execution is saved :-)")

    args = _define_arguments() 
    constants = Constants(args) 
    config = Config(args, constants) 

    ## Set external random seed:
    random.seed(config.seed) 
    np.random.seed(config.seed) 
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    ## Setup logger:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=config.logging_outpath)
    logging.info(f"\nStart at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger = logging.getLogger()

    ## Copy the current Python script to the output folder:
    shutil.copyfile(src=os.path.abspath(__file__), dst=os.path.join(config.out_folder, os.path.basename(__file__)))
    logger.info("The current python script is copied to the output folder :-)")

    ## Save the description of the current execution as a JSON file:
    _save_description(args, config, constants, logger)

    return args, constants, config, logger

def filter_features_preliminary(DF, domains, approaches, ori_name, args): 
    '''
    Filter features by domain and approach, according to data collection orientations.
    '''
    if args.standardize_tests or args.platform_features:
        included_features = [ col for col in DF.columns if any( d == col for d in domains ) ]
    else:
        included_features = [ col for col in DF.columns if any( d in col for d in domains ) and any( a in col for a in approaches ) ]
    
    if ori_name == "FUNCTIONAL": # exclude "STRUCTURE" features
        included_features = [ col for col in included_features if "STRUCTURE" not in col ]
    elif ori_name == None: 
        included_features = ["ID", "BASIC_INFO_AGE", "BASIC_INFO_SEX"] + included_features

    return DF.loc[:, included_features]

def prepare_dataset(args, constants, config, logger):
    '''
    If path to the prepared data is provided, load it. 
    Otherwise, prepare data with DataManager (balancing if requested).
    Save it in either case.
    Before returning, perform preliminary feature filtering.
    '''
    if args.use_prepared_data is not None: 
        logger.info("Loading the prepared dataset ...")
        DF_prepared = pd.read_csv(args.use_prepared_data)
        
    else:
        logger.info("Loading and processing the raw dataset ...")
        data_manager = DataManager(
            raw_data_path=config.raw_data_path, 
            inclusion_data_path=config.inclusion_data_path
        )
        data_manager.load_data()
        DF_prepared = data_manager.DF

        if config.balancing_method is not None:
            logger.info(f"Making balanced datasets using '{config.balancing_method}' method ...")
            
            balancing_outdir = os.path.join(config.syndata_folder, f"{config.balancing_method.lower()}_{config.balancing_groups}_{config.balance_to_num}")
            while os.path.exists(balancing_outdir):
                balancing_outdir = balancing_outdir + "+"
            os.makedirs(balancing_outdir)

            data_manager.make_data_balanced(
                method=config.balancing_method, 
                balancing_groups=config.balancing_groups,
                age_bin_dict=constants.age_groups["cut-8-wais"], 
                balance_to_num=config.balance_to_num, 
                seed=config.seed, 
                out_dir=balancing_outdir
            )
            DF_prepared = data_manager.DF_balanced

    logger.info("Saving the prepared dataset ...")
    DF_prepared.to_csv(config.prepared_data_outpath, index=False) 

    ## If user only wants to prepare the data:
    if args.prepare_data_and_exit:
        logger.info("Data preparation is completed. Exiting the program ...")
        sys.exit(0)

    logger.info("Preliminary feature filtering based on domains and approaches ...")
    all_domains = [ 
        d
        for d_dict in constants.domain_approach_mapping.values()
        for d in d_dict["domains"] # which is a list
    ]
    all_approaches = [
        a
        for a_dict in constants.domain_approach_mapping.values()
        for a in a_dict["approaches"]
    ]
    DF_prepared = filter_features_preliminary(
        DF_prepared, 
        domains=all_domains,  
        approaches=all_approaches, 
        ori_name=None, 
        args=args
    )

    return DF_prepared

def divide_and_preprocess_data(DF, args, config, logger):
    '''
    Divide the dataset into groups (+ define their labels), 
    separately preprocess them (including splitting into training and testing sets), 
    and return a dictionary of preprocessed datasets. 
    '''
    def _divide_dataset(DF, config, divide_by_gender):
        '''
        Divide the dataset into groups based on participants' age (and gender).
        '''
        if divide_by_gender:
            group_name_list = [ f"{age_group}_{sex}" for age_group, sex 
                in list(product(config.age_bin_labels, ["M", "F"])) 
            ] 
            sub_DF_list = [
                DF[(DF["BASIC_INFO_AGE"].between(lb, ub)) & (DF["BASIC_INFO_SEX"] == sex)].reset_index(drop=True) 
                for (lb, ub), sex in list(product(config.age_boundaries, [1, 2]))
            ]
        else:
            group_name_list = config.age_bin_labels
            sub_DF_list = [
                DF[DF["BASIC_INFO_AGE"].between(lb, ub)].reset_index(drop=True) 
                for lb, ub in config.age_boundaries
            ]

        return group_name_list, sub_DF_list

    def _preprocess_divided_datasets(X, y, ids, split_with_ids, testset_ratio, trained_scaler, seed):
        '''
        1. Remove features (data columns) with too many missing observations
        2. Fill missing values (with column-wise median)
        3. Split data into training and testing sets
        4. Perform feature scaling (scaler should be trained only on the training set)
        '''
        def __remove_outlier_features(X):
            n_subjs = len(X)
            na_rates = pd.Series(X.isnull().sum() / n_subjs)
            Q1 = na_rates.quantile(.25)
            Q3 = na_rates.quantile(.75)
            IQR = Q3 - Q1
            outliers = na_rates[na_rates > (Q3 + IQR * 1.5)]

            return X.drop(columns=outliers.index)
        
        def __feature_scaling(X, scaler, trained_or_not):
            if trained_or_not == "not_trained":
                return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            else: # "trained"
                return pd.DataFrame(scaler.transform(X), columns=X.columns)
            
        def __get_normed_train_test(X, y, ids, split_with_ids, testset_ratio, trained_scaler, seed):
            if trained_scaler is None:
                scaler = StandardScaler() # to have zero mean and unit variance
                trained_or_not = "not_trained"
            else:
                scaler = trained_scaler
                trained_or_not = "trained"

            if testset_ratio == 0: # no testing set, use the whole dataset for training
                X_train_scaled = __feature_scaling(X, scaler, trained_or_not)
                y_train, id_train = y, ids
                X_test_scaled, y_test, id_test = pd.DataFrame(), pd.Series(), pd.Series()

            else: 
                if split_with_ids is not None: # split the dataset based on pre-defined IDs
                    idx_train, idx_test = ids.isin(split_with_ids["Train"]), ids.isin(split_with_ids["Test"])
                    X_train, y_train, id_train = X[idx_train], y[idx_train], ids[idx_train]
                    X_test, y_test, id_test = X[idx_test], y[idx_test], ids[idx_test]
                
                else: # split the dataset based on the given ratio
                    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                        X, y, ids, test_size=testset_ratio, random_state=seed
                    )
                X_train_scaled = __feature_scaling(X_train, scaler, trained_or_not)
                X_test_scaled = __feature_scaling(X_test, scaler, "trained")

            return X_train_scaled, y_train, id_train, X_test_scaled, y_test, id_test, scaler
        
        X_cleaned = __remove_outlier_features(X)
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(imputer.fit_transform(X_cleaned), columns=X_cleaned.columns)
        X_train_scaled, y_train, id_train, X_test_scaled, y_test, id_test, scaler = __get_normed_train_test(X_imputed, y, ids, split_with_ids, testset_ratio, trained_scaler, seed)

        return {
            "X_train": X_train_scaled, 
            "X_test": X_test_scaled, 
            "y_train": y_train, 
            "y_test": y_test, 
            "id_train": id_train.tolist() if isinstance(id_train, pd.Series) else id_train, 
            "id_test": id_test.tolist() if isinstance(id_test, pd.Series) else id_test, 
            "scaler": scaler
        }

    group_name_list, sub_DF_list = _divide_dataset(DF, config, args.by_gender)
    preprocessed_data_dicts = {}
    train_ids, test_ids = [], []    

    for group_name, sub_DF in zip(group_name_list, sub_DF_list):
        if sub_DF.empty:
            logger.warning(f"Oh no! Data subset of the '{group_name}' group is empty :-S")
            preprocessed_data_dicts[group_name] = None

        else:
            logger.info(f"Preprocessing data subset of the {group_name} group ...")

            ## Load pre-trained scaler (if should be used):
            if args.pretrained_model_folder is not None:
                logger.info("Loading pre-trained scaler ...")
                try:
                    with open(os.path.join(config.pretrained_model_folder, f"scaler_{group_name}.pkl"), 'rb') as f:
                        trained_scaler = pickle.load(f)
                    
                except FileNotFoundError:
                    logger.warning("Pre-trained scaler not found, build a new MinMaxScaler.")
                    trained_scaler = None
            else:
                trained_scaler = None
    
            ## Preprocessing:
            preprocessed_data_dicts[group_name] = _preprocess_divided_datasets(
                X=sub_DF.drop(columns=["ID", "BASIC_INFO_AGE"]), 
                y=sub_DF["BASIC_INFO_AGE"], 
                ids=sub_DF["ID"], 
                split_with_ids=config.split_with_ids, 
                testset_ratio=config.testset_ratio, 
                trained_scaler=trained_scaler, 
                seed=config.seed
            ) 
            train_ids.append(preprocessed_data_dicts[group_name]["id_train"])
            test_ids.append(preprocessed_data_dicts[group_name]["id_test"])

            ## Save the scaler to the output folder:
            with open(config.scaler_outpath_format.format(group_name), 'wb') as f:
                pickle.dump(preprocessed_data_dicts[group_name]["scaler"], f)

    ## Save the IDs of participants in the training and testing sets (ungrouped):
    logger.info("Saving the IDs of participants in the training and testing sets ...")
    split_with_ids = {
        "Train": sum(train_ids, []), # ensure a flat list
        "Test": sum(test_ids, [])
    }
    with open(config.split_ids_outpath, 'w', encoding='utf-8') as f:
        json.dump(split_with_ids, f, ensure_ascii=False)

    return preprocessed_data_dicts

def use_pretrained_model(X, group_name, ori_name, config, logger):
    '''
    Load a pre-trained model (and the corresponding feature reducer/selector) from the specified folder, 
    and return the model, the feature reducer/selector, and the transformed dataset.
    '''
    saved_json = os.path.join(config.pretrained_model_folder, f"results_{group_name}_{ori_name}.json")
    with open(saved_json, 'r', encoding='utf-8') as f:
        saved_results = json.load(f)

    best_model_name = saved_results["Model"]
    saved_model = os.path.join(config.pretrained_model_folder, f"models_{group_name}_{ori_name}_{best_model_name}.pkl")
    with open(saved_model, 'rb') as f:
        trained_model = pickle.load(f)

    reducer_path = os.path.join(config.pretrained_model_folder, os.path.basename(config.reducer_outpath_format.format(group_name, ori_name)))
    selector_path = os.path.join(config.pretrained_model_folder, os.path.basename(config.selector_outpath_format.format(group_name, ori_name, best_model_name)))

    if os.path.exists(reducer_path):
        logger.info("PCA reducer is found, loading and applying it ...")
        with open(reducer_path, 'rb') as f:
            reducer = pickle.load(f)
        X_transformed = reducer.transform(X)
        config.feature_transformer = "reducer"
        selector = None

    elif os.path.exists(selector_path):
        logger.info("Feature selector is found, loading and applying it ...")
        with open(selector_path, 'rb') as f: 
            selector = pickle.load(f)
        X_transformed = selector.transform(X)
        config.feature_transformer = "selector"
        reducer = None

    else:
        logger.info("Original features are used in the pre-trained model.")
        used_features = saved_results["FeatureNames"]
        X_transformed = X.loc[:, used_features]
        config.feature_transformer = None
        reducer, selector = None, None

        if list(X_transformed.columns) != used_features:
            raise ValueError("The model was trained on different features than the ones used now.")

    return best_model_name, trained_model, reducer, selector, X_transformed

def train_and_save_pca(X, group_name, ori_name, args, config, logger):
    '''
    Train a PCA model with procedures specified in FeatureReducer
    and save the model (+ its derivatives).
    '''
    reducer = FeatureReducer(seed=config.seed, n_iter=100)
    reducer.fit(X=X, rhcf=args.remove_highly_correlated_features)

    logger.info("Saving the loading matrixs ...")
    pc_xlsx = config.pc_loadings_outpath_format.format(group_name, ori_name)
    with pd.ExcelWriter(pc_xlsx, engine="openpyxl", mode="w") as writer:
        for f in reducer.loadings.keys():
            reducer.loadings[f].to_excel(writer, sheet_name=f)
    
    rc_xlsx = config.rc_loadings_outpath_format.format(group_name, ori_name)
    with pd.ExcelWriter(rc_xlsx, engine="openpyxl", mode="w") as writer:
        for f in reducer.rotated_loadings.keys():
            reducer.rotated_loadings[f].to_excel(writer, sheet_name=f)

    logger.info("Saving the feature reducer object ...")
    reducer_path = config.reducer_outpath_format.format(group_name, ori_name)
    with open(reducer_path, 'wb') as f:
        pickle.dump(reducer, f)

    logger.info("Saving the dropped features ...")
    with open(config.dropped_features_outpath_format.format(group_name, ori_name), 'w') as f:
        f.write("\n".join(reducer.dropped_features))

    return reducer

def train_and_evaluate_models(X, y, model_names, select_features, selector_outpath, seed, args, config, logger, 
                              n_trials=50, n_jobs=16):
    '''
    For each model type,
    1. Define which hyperparameters to optimize and the objective function in _optimize_objective(), 
       and find the best params over n_trials using Optuna.
    ** The estimator of the objective function
       is the pipeline (feature selector + model) built in _build_pipeline().
    2. Re-evaluate the pipeline with the best params across 5-fold CV, 
       and store the results (trained model, mean MAE, etc.) in a dictionary.
    '''
    def _build_pipeline(params, model_name, select_features, seed, n_jobs=16):
        '''
        If select_features is True, build a sequence of data transformers (a feature_selector) with an final predictor (the model).
        Otherwise, just build a the model.
        '''
        if model_name == "ElasticNet":
            model = ElasticNet(
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                random_state=seed
            )
        elif model_name == "CART":
            model = DecisionTreeRegressor(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=seed
            )
        elif model_name == "RF":
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=seed,
                n_jobs=n_jobs
            )
        elif model_name == "XGBM":
            model = XGBRegressor(
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                random_state=seed,
                n_jobs=n_jobs
            )
        elif model_name == "LGBM":
            model = LGBMRegressor(
                metric='mae',
                num_leaves=params["num_leaves"],
                learning_rate=params["learning_rate"],
                feature_fraction=params["feature_fraction"],
                bagging_fraction=params["bagging_fraction"],
                bagging_freq=params["bagging_freq"],
                min_child_samples=params["min_child_samples"],
                random_state=seed,
                n_jobs=n_jobs
            )

        if select_features:
            # if model_name in ["RF", "XGBM", "LGBM"]: # ensemble tree-based models:
            #     feature_selector = BorutaSelector(
            #         estimator=model, 
            #         n_estimators="auto", 
            #         verbose=0, 
            #         random_state=seed
            #     )
            # else: # ElasticNet or CART
            feature_selector = FeatureSelector(
                model_name=model_name, 
                model=model, 
                seed=seed, 
                thresh_method=params["fs_thresh_method"], 
                threshold=params["fs_threshold"], 
                explained_ratio=params["fs_explained_ratio"],
                max_feature_num=params["fs_max_feature_num"], 
                n_jobs=n_jobs
            )

            return Pipeline([("feature_selector", feature_selector), 
                             ("regressor", model)])
        else:
            return Pipeline([("regressor", model)])

    def _optimize_objective(trial, X, y, model_name, select_features, seed, args, config, n_jobs=16): 
        '''
        Objective function for hyperparameter optimization, 
        which is the average MAE scores across 5-fold cross validation.
        '''
        if model_name == "ElasticNet":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)
            }
        elif model_name == "CART":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)
            }
        elif model_name == "RF":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                "max_depth": trial.suggest_int("max_depth", 2, 32)
            }
        elif model_name == "XGBM":
            params = {
                "max_depth": trial.suggest_int("max_depth", 1, 9),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0)
            }
        elif model_name == "LGBM":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
            }

        if select_features: 
            feature_params = {
                "fs_thresh_method": config.fs_thresh_method, 
                "fs_threshold": None, 
                "fs_explained_ratio": None, 
                "fs_max_feature_num": None
            }
            if feature_params["fs_thresh_method"] == "fixed_threshold": 
                if args.fs_threshold is None:
                    feature_params["fs_threshold"] = trial.suggest_float("fs_threshold", 1e-5, 0.01)
                else:
                    feature_params["fs_threshold"] = args.fs_threshold

            elif feature_params["fs_thresh_method"] == "explained_ratio": 
                if args.fs_explained_ratio is None:
                    feature_params["fs_explained_ratio"] = trial.suggest_float("fs_explained_ratio", 0.9, 1)
                else:
                    feature_params["fs_explained_ratio"] = args.fs_explained_ratio
                
            params.update(feature_params)
            for k, v in feature_params.items():
                trial.set_user_attr(k, v)

        pipline = _build_pipeline(
            params, model_name, select_features, seed, n_jobs
        )
        neg_mae_scores = cross_val_score(
            estimator=pipline, X=X, y=y, 
            cv=KFold(n_splits=5, shuffle=True, random_state=seed), 
            scoring="neg_mean_absolute_error", 
            n_jobs=n_jobs, 
            verbose=1
        )
        return -1 * np.mean(neg_mae_scores)

    ## Check input:
    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame."
    assert isinstance(y, pd.Series) or isinstance(y, np.ndarray), "y should be a pandas Series or a numpy array."
    assert X.shape[0] == y.shape[0], "X and y should have the same number of rows."
    assert (not np.any(np.isnan(X))) and (not np.any(np.isnan(y))), "X and y should not contain NaN values."
    assert (not np.any(np.isinf(X))) and (not np.any(np.isinf(y))), "X and y should not contain infinite values."

    results = {}

    for model_name in model_names:
        logger.info(f"Optimizing hyperparameters for {model_name} ...")

        ## Initialize the model with the best hyperparameters:
        module = optunahub.load_module(package="samplers/auto_sampler")
        study = optuna.create_study(
            direction='minimize', 
            sampler=module.AutoSampler(seed=seed) # automatically selects an algorithm internally, see: https://medium.com/optuna/autosampler-automatic-selection-of-optimization-algorithms-in-optuna-1443875fd8f9
        )
        study.optimize(
            lambda trial: _optimize_objective(trial, X, y, model_name, select_features, seed, args, config, n_jobs),
            n_trials=n_trials, show_progress_bar=True
        )
        logger.info("Parameter optimization is completed :-)")

        ## Train and evaluate the model across 5-fold CV:
        logger.info("Evaluating the model with the best hyperparameters ...")
        best_params = study.best_params
        best_params.update(study.best_trial.user_attrs)
        best_pipeline = _build_pipeline(
            best_params, model_name, select_features, seed, n_jobs
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        mae_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            best_pipeline.fit(X_train, y_train)
            y_pred = best_pipeline.predict(X_test)
            if np.isnan(y_pred).any(): 
                raise ValueError("Model prediction contains NaN values.")
            mae_scores.append(mean_absolute_error(y_test, y_pred))
        
        ## Save feature selector if exists:
        if select_features:
            selector = best_pipeline.named_steps["feature_selector"]                
            with open(selector_outpath.replace("<model_name>", model_name), "wb") as f:
                pickle.dump(selector, f)
            X_ = selector.transform(X)
        else:
            selector = None
            X_ = X

        ## Storing results:
        trained_model = best_pipeline.named_steps["regressor"]

        if hasattr(trained_model, "coef_"):
            feature_importance = pd.Series(trained_model.coef_, index=X_.columns)
            # feature_importance.sort_values(ascending=False, key=abs, inplace=True)
        elif hasattr(trained_model, "feature_importances_"):
            feature_importance = pd.Series(trained_model.feature_importances_, index=X_.columns)
            # feature_importance.sort_values(ascending=False, key=abs, inplace=True)
        else:
            raise ValueError("Model does not have a 'coef_' or 'feature_importances_' attribute.")

        results[model_name] = {
            "trained_model": trained_model, 
            "feature_selector": selector, 
            "selected_features": list(feature_importance.index), 
            "feature_importances": feature_importance,
            "cv_scores": mae_scores, 
            "mae_mean": np.mean(mae_scores), 
            "mae_std": np.std(mae_scores)
        }
        results[model_name].update(best_params)

    return results

def save_model_and_results(results, group_name, ori_name, config, logger):
    '''
    Find the best-performing model and its corresponding features (+ importances), 
    and save results for both the best model and the other models.
    '''
    best_model_name = min(results, key=lambda x: results[x]["mae_mean"])
    trained_model = results[best_model_name]["trained_model"]
    selected_features = results[best_model_name]["selected_features"]

    logger.info("Saving the best-performing model (.pkl) ...")
    model_outpath = config.model_outpath_format.format(group_name, ori_name, best_model_name)
    with open(model_outpath, 'wb') as f:
        pickle.dump(trained_model, f)

    logger.info("Saving the best selected features and their importances ...")
    features_outpath = config.features_outpath_format.format(group_name, ori_name, "") #, results[best_model_name]["fs_method"])
    results[best_model_name]["feature_importances"].to_csv(
        features_outpath, header=False
    )

    logger.info("Saving other models to the 'other models' folder ...")
    embedded_outpath = os.path.join(config.out_folder, "other models")
    os.makedirs(embedded_outpath, exist_ok=True)
    
    for model_name, model_result in results.items():
        if model_name != best_model_name: 
            model_outpath = os.path.join(embedded_outpath, os.path.basename(config.model_outpath_format.format(group_name, ori_name, model_name)))
            with open(model_outpath, 'wb') as f:
                pickle.dump(model_result["trained_model"], f)

    logger.info("Saving results for all models ...")
    model_results = { 
        model_name: { 
            k: v for k, v in res.items() if k not in ["trained_model", "feature_selector"]
        } for model_name, res in results.items()
    }
    model_results = convert_np_types(model_results)
    model_results_outpath = config.training_results_outpath_format.format(group_name, ori_name)
    with open(model_results_outpath, 'w') as f:
        json.dump(model_results, f)

    return best_model_name, trained_model, selected_features

def perform_age_correction(train_or_test, y, y_pred, pads, config, correction_ref=None):
    '''
    Perform age correction using either method described in Zhang et al. (2023) 
    or the method described in Zhang et al. (2022).
    '''
    def _generate_correction_ref(age, pad, age_groups, age_breaks): 
        '''
        Cut data into age groups based on the given age bin edges,
        calculate the mean and standard deviation of PAD for each age group, 
        and use this as the reference table for age correction.
        '''
        DF = pd.DataFrame({"Age": age, "PAD": pad})
        DF["Group"] = pd.cut(DF["Age"], bins=age_breaks, labels=age_groups)
        correction_ref = DF.groupby("Group", observed=True)["PAD"].agg(['mean', 'std']).reset_index()
        
        return correction_ref.rename(columns={"mean": "PAD_mean", "std": "PAD_std"})  

    def _apply_age_correction(true_ages, predictions, correction_ref, age_groups, age_breaks):
        '''
        Determine the age group of the current sample, calculate their PAD, 
        and apply age correction by subtracting group mean and dividing by group STD.
        '''
        corrected_predictions = []

        for pred, true_age in zip(predictions, true_ages):
            age_label = pd.cut([true_age], bins=age_breaks, labels=age_groups)[0]
            pad = pred - true_age

            ## Get the mean and standard deviation of PAD for the age group:
            if age_label in list(correction_ref["Group"]):
                pad_mean = correction_ref.query("Group == @age_label")["PAD_mean"].values[0]
                pad_std = correction_ref.query("Group == @age_label")["PAD_std"].values[0]
            else:
                ## If the age group is not in the reference table, use the mean and standard deviation of all samples:
                pad_mean = correction_ref["PAD_mean"].mean()
                pad_std = correction_ref["PAD_std"].mean()

            if pad_std < 1e-6: # Handle the case where the std of the PAD is almost 0
                padac = pad - pad_mean
            else:
                padac = (pad - pad_mean) / pad_std # The age-corrected PAD

            corrected_predictions.append(pred - padac)

        return np.array(corrected_predictions)

    def _model_age_related_bias(real_ages, offsets):
        reg  = LinearRegression().fit(
            X=np.array(real_ages).reshape(-1, 1), y=np.array(offsets)
        )
        return {"intercept": reg.intercept_, "slope": reg.coef_}

    def _calc_bias_free_offsets(reg, predicted_ages):
        return predicted_ages * reg["slope"] + reg["intercept"]
    
    if config.age_correction_method == "Zhang et al. (2023)":
        if train_or_test == "train":
            correction_ref = _generate_correction_ref(
                age=y, 
                pad=pads, 
                age_groups=config.pad_age_groups, 
                age_breaks=config.pad_age_breaks
            )
        corrected_y_pred = _apply_age_correction(
            true_ages=y, 
            predictions=y_pred, 
            correction_ref=correction_ref, 
            age_groups=config.pad_age_groups, 
            age_breaks=config.pad_age_breaks
        )
        corrected_y_pred = pd.Series(corrected_y_pred, index=y.index)
        padac = corrected_y_pred - y

    elif config.age_correction_method == "Beheshti et al. (2019)":
        if train_or_test == "train":
            correction_ref = _model_age_related_bias(y, pads)
        padac = _calc_bias_free_offsets(correction_ref, y_pred)
        corrected_y_pred = y - padac

    return corrected_y_pred, padac, correction_ref

## Main function: ---------------------------------------------------------------------

def main():
    args, constants, config, logger = initialization()

    ## Prepare the dataset (the program may exit here if args.prepare_data_and_exit is True):
    DF_prepared = prepare_dataset(args, constants, config, logger)

    ## Divide the dataset into groups, define their labels, and separately perform preprocessing:
    preprocessed_data_dicts = divide_and_preprocess_data(DF_prepared, args, config, logger)

    logger.info("Preparation steps are completed, start training the models ...")    
    record_if_failed = [] # a list of strings to record the failed processing
    iter = 0 # iteration counter for <group> and <feature orientations>

    for group_name, data_dict in preprocessed_data_dicts.items():
        logger.info(f"Group: '{group_name}'")

        if data_dict is None: 
            logger.warning("Data subset of the current group is empty!!\nUnable to train models, skipping ...")
            record_if_failed.append(f"Entire {group_name}.")
            pass

        for ori_name, ori_content in constants.domain_approach_mapping.items():
            logger.info(f"Feature orientation: {ori_name}")

            if iter < args.ignore: 
                logger.info(f"Skipping {iter}-th iteration :-O")
                iter += 1
                pass

            iter += 1

            logger.info("Filtering features based on the domain and approach ...")
            X_train_included = filter_features_preliminary(
                data_dict["X_train"], 
                domains=ori_content["domains"], 
                approaches=ori_content["approaches"], 
                ori_name=ori_name, 
                args=args
            )

            if X_train_included.empty: 
                logger.warning("No features are included for the current orientation, the definition may be wrong!!")
                logger.warning("Unable to train models, pass this iteration ...")
                record_if_failed.append(f"{ori_name} of {group_name}.")
                pass

            if args.pretrained_model_folder is not None:
                print(f"Applying pre-trained {ori_name} model on {group_name} :-)")
                logger.info("Using the pre-trained model :-P")
                best_model_name, trained_model, reducer, selector, X_train_transformed = use_pretrained_model(
                    X_train_included, group_name, ori_name, config, logger
                )
                selected_features = list(X_train_transformed.columns)
                
            else: # do what supposed to be done
                if config.feature_transformer == "reducer":
                    logger.info("Performing dimensionality reduction ...")
                    reducer = train_and_save_pca(X_train_included, group_name, ori_name, args, config, logger)
                    X_train_transformed = reducer.transform(X=X_train_included)

                logger.info("Training different models and finding the best one ...")
                results = train_and_evaluate_models(
                    X=X_train_transformed if config.feature_transformer == "reducer" else X_train_included, 
                    y=data_dict["y_train"], 
                    model_names=config.included_models, 
                    select_features=args.feature_selection, 
                    selector_outpath=config.selector_outpath_format.format(group_name, ori_name, "<model_name>"),
                    seed=config.seed, 
                    args=args, 
                    config=config, 
                    logger=logger, 
                    n_jobs=constants.n_jobs
                )
                best_model_name, trained_model, selected_features = save_model_and_results(
                    results, group_name, ori_name, config, logger
                )
                    
            logger.info("Applying the best model to the training set ...")
            if args.pretrained_model_folder is not None:
                y_pred_train = trained_model.predict(X_train_transformed)
            elif config.feature_transformer == "selector":
                selector = results[best_model_name]["feature_selector"] 
                X_train_selected = selector.transform(X_train_included)
                y_pred_train = trained_model.predict(X_train_selected)
            elif config.feature_transformer == "reducer":
                y_pred_train = trained_model.predict(X_train_transformed)
            else:
                X_train_selected = X_train_included.loc[:, selected_features] 
                y_pred_train = trained_model.predict(X_train_selected)

            y_pred_train = pd.Series(y_pred_train, index=data_dict["y_train"].index)
            pad_train = y_pred_train - data_dict["y_train"]

            if config.age_correction_method:
                logger.info("Applying age-correction to the training set ...")
                corrected_y_pred_train, padac_train, correction_ref = perform_age_correction(
                    "train", data_dict["y_train"], y_pred_train, pad_train, config
                )
            else: 
                logger.info("No age-correction is applied ...")
                corrected_y_pred_train, padac_train, correction_ref = [], [], {}
            
            if data_dict["X_test"].empty: 
                logger.info("No testing set ...")
                save_results = {
                    "Model": best_model_name, 
                    "NumberOfSubjs": len(data_dict["id_train"]), 
                    "SubjID": list(data_dict["id_train"]), 
                    "Note": "Train and test sets are the same.", 
                    "Age": list(data_dict["y_train"]), 
                    "PredictedAge": list(y_pred_train), 
                    "PredictedAgeDifference": list(pad_train), 
                    "CorrectedPAD": list(padac_train), 
                    "CorrectedPredictedAge": list(corrected_y_pred_train), 
                    "AgeCorrectionTable": correction_ref.to_dict(orient='records') if type(correction_ref) is not dict else correction_ref, 
                    "NumberOfFeatures": len(selected_features), 
                    "FeatureNames": list(selected_features) 
                }

            else:
                logger.info("Applying the best model to the testing set ...")
                if config.feature_transformer == "selector":
                    X_test_selected = selector.transform(data_dict["X_test"])
                    y_pred_test = trained_model.predict(X_test_selected)
                elif config.feature_transformer == "reducer":
                    X_test_transformed = reducer.transform(data_dict["X_test"])
                    y_pred_test = trained_model.predict(X_test_transformed)
                else:
                    X_test_selected = data_dict["X_test"].loc[:, selected_features]
                    y_pred_test = trained_model.predict(X_test_selected)

                y_pred_test = pd.Series(y_pred_test, index=data_dict["y_test"].index)
                pad = y_pred_test - data_dict["y_test"]

                if args.age_correction_method in [0, 1]:
                    logger.info("Applying age-correction to the testing set ...")
                    corrected_y_pred_test, padac, _ = perform_age_correction(
                        train_or_test="test", 
                        y=data_dict["y_test"], 
                        y_pred=y_pred_test, 
                        pads=pad, 
                        config=config, 
                        correction_ref=correction_ref
                    )
                else:
                    logger.info("Again, no age-correction is applied ...")
                    corrected_y_pred_test, padac = [], []

                save_results = {
                    "Model": best_model_name, 
                    "NumberOfTraining": len(data_dict["id_train"]), 
                    "TrainingSubjID": list(data_dict["id_train"]), 
                    "TrainingAge": list(data_dict["y_train"]), 
                    "TrainingPredAge": list(y_pred_train), 
                    "TrainingPAD": list(pad_train), 
                    "TrainingPADAC": list(padac_train), 
                    "TrainingCorPredAge": list(corrected_y_pred_train), 
                    "NumberOfTesting": len(data_dict["id_test"]), 
                    "TestingSubjID": list(data_dict["id_test"]), 
                    "Age": list(data_dict["y_test"]), 
                    "PredictedAge": list(y_pred_test), 
                    "PredictedAgeDifference": list(pad), 
                    "CorrectedPAD": list(padac), 
                    "CorrectedPredictedAge": list(corrected_y_pred_test), 
                    "AgeCorrectionTable": correction_ref.to_dict(orient='records') if type(correction_ref) is not dict else correction_ref, 
                    "NumberOfFeatures": len(selected_features), 
                    "FeatureNames": list(selected_features)
                }

            logger.info("Saving the modeling results ...")
            save_results = convert_np_types(save_results)
            results_outpath = config.results_outpath_format.format(group_name, ori_name)
            with open(results_outpath, 'w', encoding='utf-8') as f:
                json.dump(save_results, f, ensure_ascii=False)

    logger.info("Saving the record of failed processing ...")
    with open(config.failure_record_outpath, 'w') as f:
        f.write("\n".join(record_if_failed))    

    logger.info(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
if __name__ == "__main__":
    main()
    print("\nGood job, bye!\n")