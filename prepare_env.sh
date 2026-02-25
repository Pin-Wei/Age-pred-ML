#!/bin/bash

set -e

echo "--- Initializing Environment ---"

mamba create -p /home/aclexp/mambaforge/envs/quanta python=3.10 -y

source $(conda info --base)/etc/profile.d/conda.sh
conda activate quanta

echo "Installing PyTorch, Torchvision, and CUDA toolkit..."
mamba install -y -c pytorch -c nvidia pytorch torchvision

echo "Installing Data Science stack (Scikit-Learn, XGBoost, LightGBM)..."
mamba install -y -c conda-forge \
	ipython \
    numpy \
	pandas \
    scipy \
    statsmodels \
	pingouin \
    scikit-learn \
	imbalanced-learn \
	matplotlib \
    seaborn \
	cmaes \
    optuna \
	optunahub \
    xgboost \
    lightgbm \
    shap \
	boruta_py \
    sdv \
    ctgan \
    ffmpeg \
    requests \
    tqdm

mamba install -y -c ets factor_analyzer

echo "--- Setup Complete! ---"
echo "Use 'conda activate quanta' to start."