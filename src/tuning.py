import optuna
import sys
from pathlib import Path
import json
PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))
from src.modeling import *
from src.data_split import *

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

with open("../data/features/features_100.json", "r", encoding="utf-8") as f:
    selected_features = json.load(f)

train_df = pd.read_csv('../data/train_test/holdout/train_full.csv', index_col='id', parse_dates=['timestamp'])

time_folds = get_time_folds()

def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 2000),
        'max_depth':        trial.suggest_int('max_depth', 3, 9),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma':            trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'eval_metric':      "rmse",
        'tree_method':      "hist",
        'objective':        'reg:squarederror',
        'enable_categorical':True,
        'early_stopping_rounds':200,
        'random_state':     42,
        'n_jobs':           -1,
    }

    model = XGBRegressor(**params)
    xgboost_quality, oof_pred_xgb, xgb_models, features_import_xgb, xgb_features = run_boosting_cv(
    train_df=train_df,
    time_folds=time_folds,
    split_fold=split_fold,
    model_factory=model,
    model_name="xgboost",
    booster="xgboost",
    preprocessing_func=prep_with_features_eng,
    selected_features=selected_features,
    )
    return xgboost_quality['mean_rmsle'].item()

def lgbm_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 2000),
        'num_leaves':       trial.suggest_int('num_leaves', 15, 127),
        'max_depth':        trial.suggest_int('max_depth', 3, 12),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain':   trial.suggest_float('min_split_gain', 0, 1.0),
        'objective':        'regression',
        'min_child_samples': 20,
        'random_state':     42,
        'n_jobs':           -1,
        'verbose':          -1,
    }

    model = LGBMRegressor(**params)
    lightgbm_quality, oof_pred_lightgbm, lightgbm_models, features_import_light, lgb_features = run_boosting_cv(
    train_df=train_df,
    time_folds=time_folds,
    split_fold=split_fold,
    model_factory=model,
    model_name=f"lightgbm",
    booster="lightgbm",
    preprocessing_func=prep_with_features_eng,
    selected_features=selected_features,
    )
    return lightgbm_quality['mean_rmsle'].item()

def catboost_objective(trial):
    params = {
        'iterations':      trial.suggest_int('iterations', 200, 2000),
        'depth':           trial.suggest_int('depth', 3, 10),
        'learning_rate':   trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'l2_leaf_reg':     trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0, 10.0),
        'border_count':    trial.suggest_int('border_count', 32, 255),
        'early_stopping_rounds':200,
        'eval_metric':       "RMSE",
        'loss_function':   'RMSE',
        'random_seed':     42,
        'verbose':         0,
    }

    model = CatBoostRegressor(**params)
    cat_quality, oof_pred_cat, cat_models, features_import_cat, cat_features = run_boosting_cv(
        train_df=train_df,
        time_folds=time_folds,
        split_fold=split_fold,
        model_factory=model,
        model_name=f"catboost",
        booster="catboost",
        preprocessing_func=prep_with_features_eng,
        selected_features=selected_features,
    )
    return cat_quality['mean_rmsle'].item()

def tune_busting(name):
    if name == 'catboost':
        study_cat = optuna.create_study(direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=456))
        study_cat.optimize(catboost_objective, n_trials=50, show_progress_bar=True)
        best_params = study_cat.best_params    
        best_rmsle = study_cat.best_value
    elif name == 'lightgbm':
        study_lgb = optuna.create_study(direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=456))
        study_lgb.optimize(lgbm_objective, n_trials=50, show_progress_bar=True)
        best_params = study_lgb.best_params    
        best_rmsle = study_lgb.best_value
    elif name == 'xgboost':
        study_xgb = optuna.create_study(direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=456))
        study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
        best_params = study_xgb.best_params    
        best_rmsle = study_xgb.best_value
    else:
        raise ValueError(
            "name должен быть: 'lightgbm', 'xgboost' или 'catboost'"
        )
    return best_params, best_rmsle
