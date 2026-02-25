#!/usr/bin/env python

import os
import sys
import json
import shutil
import argparse
import logging
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm
import joblib
import optuna
import optunahub
import torch
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.join(os.getcwd(), "..", "src"))
from utils import Platform_features

class Config:
    def __init__(self, args):
        self.setup_paths(args)
        self.setup_vars(args)

    def setup_paths(self, args):
        self.root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.raw_data_path = os.path.join(self.root_dir, "data", "rawdata", "DATA_ses-01_2025-05-29.csv")
        self.inclusion_data_path = os.path.join(self.root_dir, "data", "rawdata", "InclusionList_ses-01.csv")

        self.output_dir = os.path.join(self.root_dir, "outputs", f"{datetime.today().strftime('%Y-%m-%d')}_distill")
        if not getattr(args, "overwrite", False):
            while os.path.exists(self.output_dir):
                self.output_dir += "+"
        os.makedirs(self.output_dir, exist_ok=True)
        self.config_path = os.path.join(self.output_dir, "config.json")
        self.log_path = os.path.join(self.output_dir, "log.txt")
        self.prepared_data_path = os.path.join(self.output_dir, "prepared_data.csv")
        self.scaler_path = os.path.join(self.output_dir, "scaler.joblib")
        self.imputer_path = os.path.join(self.output_dir, "imputer.joblib")        
        self.feature_removed_path = os.path.join(self.output_dir, "feature_removed.txt")
        self.feature_candidates_path = os.path.join(self.output_dir, "feature_candidates.txt")
        self.feature_available_path = os.path.join(self.output_dir, "feature_available.txt")
        self.mdl_hyperparams_path_template = os.path.join(self.output_dir, "mdl_hyperparams_{}.json")
        self.model_path_template = os.path.join(self.output_dir, "model_{}.joblib")
        self.bias_path_template = os.path.join(self.output_dir, "bias_{}.json")
        self.predictions_path = os.path.join(self.output_dir, "predictions.csv")
        self.performance_path = os.path.join(self.output_dir, "performance.csv")

    def setup_vars(self, args):
        self.remove_features_with_excessive_na = False
        self.test_ratio = args.test_ratio
        self.k_folds = args.k_folds
        self.stratify_by = "wais_age_groups"
        self.teacher_model_name = "XGBoost"
        self.student_model_name = "ElasticNet"
        self.opt_trial_num = 100
        self.distill_alpha = args.distill_alpha

        self.label_smoothing = ~args.no_label_smoothing
        if self.label_smoothing:
            self.y_range = (15, 85)
            self.y_bin_step = 1
            self.y_sigma = 1.0
            self.device = "cpu"
            self.performance_metric = "KL_Loss"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.performance_metric = "MAE"
        
        self.seed = np.random.randint(0, 1000) if args.seed is None else args.seed
        self.n_jobs = 10
        
class Constants:
    def __init__(self, config):
        self.id_cols = [
            "ID", "Age", "Sex", "AgeGroup", "AgeSexGroup"
        ]
        self.platform_features = Platform_features()
        self.setup_age_groups(config)
        if config.label_smoothing:
            self.setup_label_smoothing_frame(config)

    def setup_age_groups(self, config):
        self.age_groups = {
            "wais_age_groups": {
                "le-24": ( 0, 24), 
                "25-29": (25, 29), 
                "30-34": (30, 34),
                "35-44": (35, 44), 
                "45-54": (45, 54), 
                "55-64": (55, 64), 
                "65-69": (65, 69), 
                "ge-70": (70, np.inf)
            }
        }[config.stratify_by]
        self.age_group_boundaries = [0] + [ upper for _, upper in self.age_groups.values() ]

    def setup_label_smoothing_frame(self, config):
        y_min, y_max = config.y_range
        self.y_bins = np.arange(y_min, y_max + config.y_bin_step, config.y_bin_step)
        self.y_bin_centers = (self.y_bins[1:] + self.y_bins[:-1]) / 2

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", default=False, 
                        help="Set to True to overwrite the output directory.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, 
                        help="Set to True to show logging in the terminal.")
    parser.add_argument("-s", "--seed", type=int, default=None, 
                        help="The value used to initialize all random number generator.")
    parser.add_argument("-te", "--test_ratio", type=float, default=0.2, 
                        help="The ratio of the test set to the whole dataset.")
    parser.add_argument("-kf", "--k_folds", type=int, default=5, 
                        help="The number of folds for cross-validation.")
    parser.add_argument("-ta", "--distill_alpha", type=float, default=0.7,
                        help="The alpha value for distillation loss, between 0 and 1.")
    parser.add_argument("-ptm", "--pretrained_teacher_model", type=str, default=None,
                        help="The path to the pretrained teacher model. If provided, the script will skip training.")
    parser.add_argument("-oth", "--opt_teacher_hyperparams", type=str, default=None,
                        help="The path to the optimized hyperparameters for teacher model.")
    parser.add_argument("-osh", "--opt_student_hyperparams", type=str, default=None,
                        help="The path to the optimized hyperparameters for student model.")
    parser.add_argument("-nls", "--no_label_smoothing", action="store_true", default=False, 
                        help="Set to True to disable label smoothing.")
    args = parser.parse_args()

    return args

def setup_logger(args, config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(config.log_path)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if args.verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger

def save_config(config):
    config_dict = {
        k.upper(): v for k, v in config.__dict__.items() 
        if not any(s in k for s in ["path", "dir"])
        or k in ["raw_data_path", "inclusion_data_path"]
        # and k not in ["n_jobs", "device"]
    }
    with open(config.config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False)

def prepare_datasets(config, const, logger):
    '''
    Step-1. Load data and apply inclusion criteria
    Step-2. Assign "AgeSexGroup" to each participant based on their Age and Sex
    Step-3. Split data into train-val/testing sets with stratified sampling
    Step-4. Remove features with excessive missing values (optional)
    Step-5. Scale features to have zero mean and unit variance
    Step-6. Split training and validation sets for each fold with stratified sampling
    '''
    def _load_data(config):
        raw_data = pd.read_csv(config.raw_data_path)
        raw_data.rename(columns={
            "BASIC_INFO_ID": "ID", 
            "BASIC_INFO_AGE": "Age", 
            "BASIC_INFO_SEX": "Sex", 
        }, inplace=True) 

        inclusion_data = pd.read_csv(config.inclusion_data_path)
        inclusion_data = inclusion_data.query("MRI == 1") 

        DF = pd.merge(raw_data, inclusion_data[["ID"]], on="ID", how='inner') 
        DF.dropna(subset=["ID", "Age", "Sex"], inplace=True)
        DF["Sex"] = DF["Sex"].map({1: "M", 2: "F"})
        return DF

    def _add_group_cols(DF, const):
        DF["AgeGroup"] = pd.cut(
            DF["Age"], 
            bins=const.age_group_boundaries, 
            labels=list(const.age_groups.keys())
        )
        DF["AgeSexGroup"] = DF["AgeGroup"].astype(str) + "_" + DF["Sex"].astype(str)
        return DF

    def _split_trainval_test(DF, config, group_col="AgeSexGroup"):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=config.test_ratio, random_state=config.seed
        )
        trainval_idx, test_idx = next(sss.split(DF.index, DF[group_col]))
        return (
            DF.iloc[trainval_idx, :].reset_index(drop=True), 
            DF.iloc[test_idx, :].reset_index(drop=True)
        )

    def _get_feature_init(DF, const):
        return [
            f for f in DF.columns
            if not any( kw in f for kw in ["RESTING_", "BASIC_", "_ST_"] )
            and f not in const.id_cols
        ]
        
    def _get_outlier_features(DF, feature_init):
        X = DF.loc[:, feature_init]
        n_subjs = len(X)
        na_rates = pd.Series(X.isnull().sum() / n_subjs, index=X.columns)
        Q1 = na_rates.quantile(.25)
        Q3 = na_rates.quantile(.75)
        IQR = Q3 - Q1
        thres = Q3 + IQR * 1.5
        outliers = na_rates[na_rates > thres]
        return outliers.index, thres
    
    def _feature_scaling(DF_trainval, DF_test, feature_candidates):
        scaler = StandardScaler()
        scaler.fit(DF_trainval.loc[:, feature_candidates])
        DF_trainval.loc[:, feature_candidates] = scaler.transform(DF_trainval.loc[:, feature_candidates])
        DF_test.loc[:, feature_candidates] = scaler.transform(DF_test.loc[:, feature_candidates])
        return scaler, DF_trainval, DF_test

    def _split_train_val(DF, config, group_col="AgeSexGroup"):
        skf = StratifiedKFold(
            n_splits=config.k_folds, shuffle=True, random_state=config.seed
        )
        for n, (tr_idx, va_idx) in enumerate(skf.split(DF.index, DF[group_col])):
            DF.loc[DF.index[tr_idx], [f"Fold_{n+1}"]] = "Train"
            DF.loc[DF.index[va_idx], [f"Fold_{n+1}"]] = "Val"
        return DF

    ## Step-1 and 2:
    logger.info("Loading data ...")
    DF = _load_data(config)
    DF = _add_group_cols(DF, const)

    ## Step-3:
    logger.info("Splitting data into train-val/test sets ...")
    DF_trainval, DF_test = _split_trainval_test(DF, config)

    ## Step-4:
    if config.remove_features_with_excessive_na:
        logger.info("Finding features with excessive missing values in train-val set ...")
        feature_init = _get_feature_init(DF_trainval, const) 
        feature_outliers, thres = _get_outlier_features(DF_trainval, feature_init)
        feature_candidates = list(set(feature_init) - set(feature_outliers))
        logger.info(f"Number of outlier features: {len(feature_outliers)} (missing rate > {thres:.3f})")
        
        with open(config.feature_removed_path, "w", encoding="utf-8") as f:
            f.write("\n".join(feature_outliers))
        logger.info(f"Names of removed features are saved in '{os.path.basename(config.feature_removed_path)}'.")

        logger.info("Removing outlier features ...")
        DF_trainval = DF_trainval.drop(columns=feature_outliers)
        DF_test = DF_test.drop(columns=feature_outliers)
        logger.info(f"Number of remaining features: {len(feature_candidates)}")
    else:
        feature_candidates = _get_feature_init(DF_trainval, const)
        logger.info(f"Number of features: {len(feature_candidates)}")

    with open(config.feature_candidates_path, "w", encoding="utf-8") as f:
        f.write("\n".join(feature_candidates))
    logger.info(f"Names of candidate features are saved in '{os.path.basename(config.feature_candidates_path)}'.")
    
    ## Step-5:
    logger.info("Scaling features ...")
    scaler, DF_trainval, DF_test = _feature_scaling(DF_trainval, DF_test, feature_candidates)
    joblib.dump(scaler, config.scaler_path)
    logger.info(f"Scaler is saved to '{os.path.basename(config.scaler_path)}'")

    ## Step-6:
    logger.info("Assigning train/validation sets for each fold ...")
    DF_trainval = _split_train_val(DF_trainval, config)

    ## Finally:
    DF_to_save = (
        pd.concat([
            DF_trainval, 
            DF_test.assign(**{f"Fold_{n+1}": "Test" for n in range(config.k_folds)})
        ], axis=0)
    )
    ordered_cols = (
        ["ID", "Age", "Sex"] + 
        [ f"Fold_{n+1}" for n in range(config.k_folds)] + 
        feature_candidates
    )
    DF_to_save.loc[:, ordered_cols].to_csv(config.prepared_data_path, index=False)
    logger.info(f"Train-val and test datasets are saved to '{os.path.basename(config.prepared_data_path)}'")

    return DF_trainval, DF_test, feature_candidates

def label_smoothing(y, config, const):
    cdfs = norm.cdf(const.y_bins, loc=y, scale=config.y_sigma)
    probs = cdfs[1:] - cdfs[:-1]    
    return probs / probs.sum() # to ensure sum(probs) == 1

def inverse_label_smoothing(probs, const):
    return np.matmul(probs, const.y_bin_centers)

def kl_loss(probs_pred, probs_true, eps=1e-12):
    probs_pred = np.clip(probs_pred, eps, None)
    probs_true = np.clip(probs_true, eps, None)
    probs_pred /= probs_pred.sum(axis=1, keepdims=True)
    return np.mean(
        np.sum(probs_true * np.log(probs_true / probs_pred), axis=1)
    )

def build_model(model_name, y_size, params, config):
    if model_name == "XGBoost":
        if y_size > 1:
            kwargs = {
                "num_target": y_size, 
                "multi_strategy": "multi_output_tree", 
                "custom_metric": kl_loss
            }
        else:
            kwargs = {
                "eval_metric": "mae",
            }
        return xgb.XGBRegressor( # see: https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html
            learning_rate=params["learning_rate"], 
            min_split_loss=params["min_split_loss"], 
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"], 
            subsample=params["subsample"], 
            colsample_bytree=params["colsample_bytree"], 
            reg_alpha=params["reg_alpha"], 
            reg_lambda=params["reg_lambda"], 
            grow_policy=params["grow_policy"], 
            tree_method="hist", 
            n_estimators=2000, # use a high upper limit and control through early stopping
            early_stopping_rounds=30, # stop if the validation MAE (or other metric) does not improve for 30 rounds
            device=config.device, 
            n_jobs=config.n_jobs, 
            random_state=config.seed, 
            **kwargs
        ) 
    elif model_name == "ElasticNet":
        model = ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            random_state=config.seed
        )
        if y_size > 1:
            return MultiOutputRegressor(model)
        else:
            return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
def train_model(model_name, model, X_train, y_train, X_val, y_val):
    y_train = np.vstack(y_train.to_numpy())
    y_val = np.vstack(y_val.to_numpy())
    
    if model_name == "XGBoost":
        model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            verbose=False
        )
    elif model_name == "ElasticNet":
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def optimize_hyperparameters(X, y, grp, model_name, config, logger):
    def _objective(trial, X, y, grp, model_name, config, logger):
        if model_name == "XGBoost":
            params = { # https://xgboost.readthedocs.io/en/stable/parameter.html
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "min_split_loss": trial.suggest_float("min_split_loss", 0.0, 5.0), # the minimum loss reduction required to perform a split
                "max_depth": trial.suggest_int("max_depth", 2, 10), 
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20, log=True), # the minimum sum of instance weight (Hessian) required in a child (leaf) node for a split to occur
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), # the proportion of samples used during training of each tree
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0), # the proportion of features used during training of each tree
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True), # L1 regularization term on weights
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True), # L2 regularization term on weights
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            }
        elif model_name == "ElasticNet":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)
            }
        
        y_size = y.iloc[0].shape[0] if config.label_smoothing else 1
        model = build_model(model_name, y_size, params, config)
        skf = StratifiedKFold(
            n_splits=config.k_folds, shuffle=True, random_state=config.seed
        )
        performance_records = []
        for n, (tr_idx, va_idx) in enumerate(skf.split(X.index, grp)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            model = train_model(model_name, model, X_tr, y_tr, X_va, y_va)
            y_va_pred = model.predict(X_va)

            if y_size > 1:
                y_va = np.vstack(y_va.to_numpy())
                v = kl_loss(y_va_pred, y_va)
            else:
                v = mean_absolute_error(y_va_pred, y_va)                
            performance_records.append(v)
            logger.info(f"Trial {trial.number}, Fold {n+1}: {config.performance_metric} = {v:.3f}")

            trial.report(np.mean(performance_records), n) # report intermediate objective value
            if trial.should_prune():
                logger.info(f"Trial {trial.number} is pruned.")
                raise optuna.TrialPruned()
            
        return np.mean(performance_records)

    logger.info(f"Model: {model_name}")
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Target size per sample: {y.iloc[0].shape[0] if config.label_smoothing else 1}")
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(
        direction="minimize", 
        sampler=module.AutoSampler(seed=config.seed), # automatically selects the best sampler, see: https://medium.com/optuna/autosampler-automatic-selection-of-optimization-algorithms-in-optuna-1443875fd8f9
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5) # prunes unpromising trials, see: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna.pruners.MedianPruner
    )
    study.optimize(
        lambda trial: _objective(trial, X, y, grp, model_name, config, logger), 
        n_trials=config.opt_trial_num, 
        show_progress_bar=True
    )
    best_params = study.best_params
    logger.info(f"Optimization is done. The best mean {config.performance_metric} is {study.best_value:.3f} in trial {study.best_trial.number}.")

    return best_params

def trainval_teacher(DF_trainval, feature_full, target_col, params, prediction_records, performance_records, config, const, logger):
    '''
    Step-1. Train teacher model on training set of each fold and save the model
    Step-2. Evaluate teacher model on validation set of each fold, save predictions and calculate performance metric (MAE or KL Loss)
    Step-3. Select the best fold based on performance metric and load the corresponding teacher model
    Step-4. Calculate age-related bias on the best fold and save the bias correction parameters
    '''
    for n in range(config.k_folds):
        DF_train = DF_trainval.query(f"Fold_{n+1} == 'Train'")
        DF_val = DF_trainval.query(f"Fold_{n+1} == 'Val'")

        ## Step-1:
        logger.info(f"Training teacher model (fold {n+1}/{config.k_folds}) ...") 
        y_size = DF_train[target_col].iloc[0].shape[0] if config.label_smoothing else 1    
        teacher_model = build_model(config.teacher_model_name, y_size, params, config)
        teacher_model = train_model(
            model_name=config.teacher_model_name, 
            model=teacher_model, 
            X_train=DF_train.loc[:, feature_full], 
            y_train=DF_train[target_col], 
            X_val=DF_val.loc[:, feature_full], 
            y_val=DF_val[target_col]
        )
        joblib.dump(teacher_model, config.model_path_template.format(f"teacher_fold-{n+1}"))
        
        ## Step-2:
        logger.info("Evaluating teacher model ...")
        y_va_pred = teacher_model.predict(DF_val.loc[:, feature_full])
        
        if config.label_smoothing:
            v = kl_loss(y_va_pred, np.vstack(DF_val[target_col].to_numpy()))
            y_va_pred_age = inverse_label_smoothing(y_va_pred, const)
        else:
            v = mean_absolute_error(y_va_pred, DF_val["Age"])
        logger.info(f"Fold {n+1}: {config.performance_metric} = {v:.3f}")

        prediction_records.append(pd.DataFrame({
            "Model": "Teacher",
            "Stage": "Val", 
            "Fold": n+1, 
            "ID": DF_val["ID"].tolist(),
            "Real_Age": DF_val["Age"].tolist(),
            "Pred_Age": y_va_pred.tolist() if not config.label_smoothing else y_va_pred_age
        }, index=DF_val.index))

        performance_records.append(pd.DataFrame({
            "Model": "Teacher",
            "Stage": "Val", 
            "Fold": n+1, 
            config.performance_metric: v, 
            f"{config.performance_metric}_2": ""
        }, index=[0]))

    ## Step-3:
    df1 = pd.concat(performance_records, ignore_index=True).query("Model == 'Teacher' & Stage == 'Val'")
    best_fold_teacher = df1.loc[df1[config.performance_metric].idxmin(), "Fold"]
    logger.info(f"Best fold: {best_fold_teacher}")
    teacher_model = joblib.load(config.model_path_template.format(f"teacher_fold-{best_fold_teacher}"))

    ## Step-4:
    df2 = pd.concat(prediction_records, ignore_index=True).query(f"Model == 'Teacher' & Stage == 'Val' & Fold == {best_fold_teacher}")
    slope_te, const_te = model_age_related_bias(df2["Real_Age"], df2["Pred_Age"])
    with open(config.bias_path_template.format("teacher"), "w") as f:
        json.dump({"slope": slope_te, "intercept": const_te}, f)
    logger.info(f"Age-related bias correction for teacher model: slope = {slope_te:.3f}, intercept = {const_te:.3f}")

    return teacher_model, slope_te, const_te, performance_records, prediction_records

def trainval_student(DF_trainval, feature_full, target_col, feature_available, params, teacher_model, prediction_records, performance_records, config, const, logger):
    '''
    Similar to training and validation of teacher model, but with the following differences:
    1. The input features are limited to those available on the platform (feature_available).
    2. The training target is a weighted average of the teacher's prediction and the real age.
    '''
    for n in range(config.k_folds):
        logger.info(f"Training student model (fold {n+1}/{config.k_folds}) ...")
        y_size = DF_trainval[target_col].iloc[0].shape[0] if config.label_smoothing else 1
        student_model = build_model(config.student_model_name, y_size, params, config)
        
        DF_train = DF_trainval.query(f"Fold_{n+1} == 'Train'")
        y_tr_teacher = teacher_model.predict(DF_train.loc[:, feature_full])
        y_tr = y_tr_teacher * config.distill_alpha + DF_train[target_col] * (1 - config.distill_alpha)
            
        DF_val = DF_trainval.query(f"Fold_{n+1} == 'Val'")
        y_va_teacher = teacher_model.predict(DF_val.loc[:, feature_full])
        y_va = y_va_teacher * config.distill_alpha + DF_val[target_col] * (1 - config.distill_alpha)
        
        student_model = train_model(
            model_name=config.student_model_name, 
            model=student_model, 
            X_train=DF_train.loc[:, feature_available], 
            y_train=y_tr, 
            X_val=DF_val.loc[:, feature_available], 
            y_val=y_va
        )
        joblib.dump(student_model, config.model_path_template.format(f"student_fold-{n+1}"))

        logger.info("Evaluating student model ...")
        y_va_pred = student_model.predict(DF_val.loc[:, feature_available])

        if config.label_smoothing:
            v = kl_loss(y_va_pred, np.vstack(DF_val[target_col].to_numpy()))
            v2 = kl_loss(y_va_pred, np.vstack(y_va.to_numpy()))
            y_va_pred_age = inverse_label_smoothing(y_va_pred, const)
        else:
            v = mean_absolute_error(y_va_pred, DF_val["Age"])
            v2 = mean_absolute_error(y_va_pred, y_va)
        logger.info(f"Fold {n+1}: {config.performance_metric} = {v:.3f}, {v2:.3f} (real-vs-prediction, teacher-vs-student)")

        prediction_records.append(pd.DataFrame({
            "Model": "Student",
            "Stage": "Val", 
            "Fold": n+1, 
            "ID": DF_val["ID"].tolist(),
            "Real_Age": DF_val["Age"].tolist(), 
            "Pred_Age": y_va_pred.tolist() if config.label_smoothing else y_va_pred_age.tolist()
        }, index=DF_val.index))

        performance_records.append(pd.DataFrame({
            "Model": "Student",
            "Stage": "Val", 
            "Fold": n+1, 
            f"{config.performance_metric}": v, 
            f"{config.performance_metric}_2": v2
        }, index=[0]))

    df3 = pd.concat(performance_records, ignore_index=True).query("Model == 'Student' & Stage == 'Val'")
    best_fold_student = df3.loc[df3[f"{config.performance_metric}_2"].idxmin(), "Fold"]
    logger.info(f"Best fold: {best_fold_student}")
    student_model = joblib.load(config.model_path_template.format(f"student_fold-{best_fold_student}"))

    df4 = pd.concat(prediction_records, ignore_index=True).query(f"Model == 'Student' & Stage == 'Val' & Fold == {best_fold_student}")
    slope_st, const_st = model_age_related_bias(df4["Real_Age"], df4["Pred_Age"])
    logger.info(f"Age-related bias correction: slope = {slope_st:.3f}, intercept = {const_st:.3f}")

    return student_model, slope_st, const_st, performance_records, prediction_records

def model_age_related_bias(y_true, y_pred):
    slope, intercept = np.polyfit(y_true, y_pred, deg=1) 
    return slope, intercept 

def apply_bias_correction(y_pred, slope, intercept):
    return (y_pred - intercept) / slope

def main():
    args = define_args()
    config = Config(args)
    const = Constants(config)
    logger = setup_logger(args, config)
    prediction_records, performance_records = [], []

    save_config(config)
    logger.info(f"Configuration is saved to '{os.path.basename(config.config_path)}'.")

    shutil.copyfile(src=os.path.abspath(__file__), dst=os.path.join(config.output_dir, os.path.basename(__file__)))
    logger.info("The current python script is copied to the output folder.")

    logger.info("Preparing data ...")
    DF_trainval, DF_test, feature_candidates = prepare_datasets(config, const, logger)

    feature_available = [ f for f in const.platform_features if f in feature_candidates ]
    with open(config.feature_available_path, "w", encoding="utf-8") as f:
        f.write("\n".join(feature_available))
    logger.info(f"Names of features available on the platform ({len(feature_available)}) are saved in '{os.path.basename(config.feature_available_path)}'.")

    if config.label_smoothing:
        logger.info("Applying label smoothing to the training targets ...")
        DF_trainval.loc[:, "Age_Smoothed"] = DF_trainval["Age"].apply(lambda y: label_smoothing(y, config, const))
        DF_test.loc[:, "Age_Smoothed"] = DF_test["Age"].apply(lambda y: label_smoothing(y, config, const))
        target_col = "Age_Smoothed"
    else:
        target_col = "Age"

    ## Teacher model -----------------------------------------------------------------

    if args.pretrained_teacher_model is not None:
        logger.info(f"Loading pretrained teacher model from '{args.pretrained_teacher_model}' ...")
        teacher_model = joblib.load(args.pretrained_teacher_model)
        
        pretrained_model_dir = os.path.dirname(args.pretrained_teacher_model)
        bias_file = os.path.join(pretrained_model_dir, "bias_teacher.json")
        with open(bias_file, "r") as f:
            bias_info = json.load(f)
        slope_te, const_te = bias_info["slope"], bias_info["intercept"]
    else:
        if args.opt_teacher_hyperparams is not None:
            logger.info(f"Loading optimized hyperparameters for teacher model from '{args.opt_teacher_hyperparams}' ...")
            with open(args.opt_teacher_hyperparams, "r") as f:
                hyperparams_teacher = json.load(f)
        else:
            logger.info("Optimizing hyperparameters for teacher model ...")
            hyperparams_teacher = optimize_hyperparameters(
                X=DF_trainval.loc[:, feature_candidates],
                y=DF_trainval[target_col], 
                grp=DF_trainval["AgeSexGroup"], 
                model_name=config.teacher_model_name, 
                config=config, 
                logger=logger
            )
            with open(config.mdl_hyperparams_path_template.format(f"teacher"), "w") as f:
                json.dump(hyperparams_teacher, f)

        teacher_model, slope_te, const_te, performance_records, prediction_records = trainval_teacher(
            DF_trainval, feature_candidates, target_col, hyperparams_teacher, prediction_records, performance_records, config, const, logger
        )

    X_test_full = DF_test.loc[:, feature_candidates]
    y_test_pred_teacher = teacher_model.predict(X_test_full)
    if config.label_smoothing:
        v = kl_loss(y_test_pred_teacher, np.vstack(DF_test[target_col].to_numpy()))
        y_test_pred_age_teacher = inverse_label_smoothing(y_test_pred_teacher, const)
    else:
        v = mean_absolute_error(y_test_pred_teacher, DF_test["Age"])

    prediction_records.append(pd.DataFrame({
        "Model": "Teacher",
        "Stage": "Test", 
        "ID": DF_test["ID"].tolist(),
        "Real_Age": DF_test["Age"].tolist(), 
        "Pred_Age": y_test_pred_teacher.tolist() if config.label_smoothing else y_test_pred_age_teacher.tolist(), 
        "Pred_Age_Cor": apply_bias_correction(y_test_pred_teacher, slope_te, const_te).tolist()
    }, index=DF_test.index))

    performance_records.append(pd.DataFrame({
        "Model": "Teacher",
        "Stage": "Test", 
        "Fold": "", 
        config.performance_metric: v, 
        f"{config.performance_metric}_2": ""
    }, index=[0]))
    logger.info(f"{config.performance_metric} of teacher model on test set = {v:.3f}")

    ## Student model -----------------------------------------------------------------

    imputer = SimpleImputer(strategy="median")
    DF_trainval.loc[:, feature_available] = imputer.fit_transform(DF_trainval.loc[:, feature_available])
    joblib.dump(imputer, config.imputer_path)

    if args.opt_student_hyperparams is not None:
        logger.info(f"Loading optimized hyperparameters for student model from '{args.opt_student_hyperparams}' ...")
        with open(args.opt_student_hyperparams, "r") as f:
            hyperparams_student = json.load(f)
    else:
        logger.info("Optimizing hyperparameters for student model ...")
        hyperparams_student = optimize_hyperparameters(
            X=DF_trainval.loc[:, feature_available],
            y=DF_trainval[target_col], 
            grp=DF_trainval["AgeSexGroup"], 
            model_name=config.student_model_name, 
            config=config, 
            logger=logger
        )
        with open(config.mdl_hyperparams_path_template.format(f"student"), "w") as f:
            json.dump(hyperparams_student, f)

    student_model, slope_st, const_st, performance_records, prediction_records = trainval_student(
        DF_trainval, feature_candidates, feature_available, hyperparams_student, teacher_model, prediction_records, performance_records, config, logger
    )

    X_test_platform = imputer.transform(DF_test.loc[:, feature_available])
    y_test_pred_student = student_model.predict(X_test_platform)
    if config.label_smoothing:
        v = kl_loss(y_test_pred_student, np.vstack(DF_test[target_col].to_numpy()))
        y_test_pred_student = inverse_label_smoothing(y_test_pred_student, const)
    else:
        v = mean_absolute_error(y_test_pred_student, DF_test["Age"])

    prediction_records.append(pd.DataFrame({
        "Model": "Student",
        "Stage": "Test", 
        "ID": DF_test["ID"].tolist(),
        "Real_Age": DF_test["Age"].tolist(), 
        "Pred_Age": y_test_pred_student.tolist(), 
        "Pred_Age_Cor": apply_bias_correction(y_test_pred_student, slope_st, const_st).tolist()
    }, index=DF_test.index))
    
    performance_records.append(pd.DataFrame({
        "Model": "Student",
        "Stage": "Test",
        "Fold": "", 
        config.performance_metric: v, 
        f"{config.performance_metric}_2": ""
    }, index=[0]))
    logger.info(f"{config.performance_metric} of student model on test set = {v:.3f}")

    ## Finally -----------------------------------------------------------------------
    
    prediction_df = pd.concat(prediction_records, ignore_index=True)
    performance_df = pd.concat(performance_records, ignore_index=True)

    for idx in performance_df.index:
        model = performance_df.loc[idx, "Model"]
        stage = performance_df.loc[idx, "Stage"]
        fold = performance_df.loc[idx, "Fold"]
        sub_df = prediction_df.query(f"Model == '{model}' & Stage == '{stage}' & Fold == '{fold}'")
        y_true = sub_df["Real_Age"].tolist()
        y_pred = sub_df["Pred_Age"].tolist()
        r2 = r2_score(y_true, y_pred)
        performance_df.loc[idx, f"R2_{model}"] = r2

    prediction_df.to_csv(config.predictions_path, index=False)    
    performance_df.to_csv(config.performance_path, index=False)
    logger.info("Predictions and performance of teacher and student models on test set are saved.")

if __name__ == "__main__":
    main()
