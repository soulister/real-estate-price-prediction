import optuna
import json
import pandas as pd

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.modeling import run_boosting_cv
from src.data_split import get_time_folds, split_fold
from src.preprocessing import prep_with_features_eng


def _run_xgb_objective(trial, train_df, time_folds, selected_features):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric": "rmse",
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "enable_categorical": True,
        "early_stopping_rounds": 200,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = XGBRegressor(**params)
    quality, _, _, _, _ = run_boosting_cv(
        train_df=train_df,
        time_folds=time_folds,
        split_fold=split_fold,
        model_factory=model,
        model_name="xgboost",
        booster="xgboost",
        preprocessing_func=prep_with_features_eng,
        selected_features=selected_features,
    )
    return quality["mean_rmsle"].item()


def _run_lgbm_objective(trial, train_df, time_folds, selected_features):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 1.0),
        "objective": "regression",
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = LGBMRegressor(**params)
    quality, _, _, _, _ = run_boosting_cv(
        train_df=train_df,
        time_folds=time_folds,
        split_fold=split_fold,
        model_factory=model,
        model_name="lightgbm",
        booster="lightgbm",
        preprocessing_func=prep_with_features_eng,
        selected_features=selected_features,
    )
    return quality["mean_rmsle"].item()


def _run_catboost_objective(trial, train_df, time_folds, selected_features):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 2000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "early_stopping_rounds": 200,
        "eval_metric": "RMSE",
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": 0,
    }
    model = CatBoostRegressor(**params)
    quality, _, _, _, _ = run_boosting_cv(
        train_df=train_df,
        time_folds=time_folds,
        split_fold=split_fold,
        model_factory=model,
        model_name="catboost",
        booster="catboost",
        preprocessing_func=prep_with_features_eng,
        selected_features=selected_features,
    )
    return quality["mean_rmsle"].item()


def tune_boosting(
    name: str,
    train_df: pd.DataFrame,
    selected_features: list,
    n_trials: int = 50,
    seed: int = 456,
):
    time_folds = get_time_folds()
    objectives = {
        "xgboost": _run_xgb_objective,
        "lightgbm": _run_lgbm_objective,
        "catboost": _run_catboost_objective,
    }
    if name not in objectives:
        raise ValueError("name должен быть: 'lightgbm', 'xgboost' или 'catboost'")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: objectives[name](
            trial, train_df, time_folds, selected_features
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study.best_params, study.best_value


def tune_all_models(
    train_df: pd.DataFrame,
    selected_features: list,
    params_paths: dict,
    n_trials: int = 50,
    seed: int = 456,
):
    results = {}
    for name in ("catboost", "lightgbm", "xgboost"):
        best_params, best_rmsle = tune_boosting(
            name=name,
            train_df=train_df,
            selected_features=selected_features,
            n_trials=n_trials,
            seed=seed,
        )
        results[name] = {"params": best_params, "rmsle": best_rmsle}
        with open(params_paths[name], "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)
    return results


# обратная совместимость с ноутбуками
def tune_busting(name, train_df, selected_features, n_trials=50, seed=456):
    return tune_boosting(name, train_df, selected_features, n_trials, seed)
