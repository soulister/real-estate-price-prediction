import lightgbm
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
from pathlib import Path

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.preprocessing import (
    prep_with_features_eng,
    preprocess_train_val,
    fit_cat_categories,
    save_cat_categories,
)
from src.data_split import split_fold, get_time_folds
from src.metrics import rmsle

def _catboost_string_encoding(X_train, X_val):
    cat_features = X_train.select_dtypes(include="category").columns.tolist()
    X_train = X_train.copy()
    X_val = X_val.copy()
    for col in cat_features:
        X_train[col] = X_train[col].astype(str).fillna("__MISSING__")
        X_val[col] = X_val[col].astype(str).fillna("__MISSING__")
    return X_train, X_val, cat_features


def run_boosting_cv(
    train_df,
    time_folds,
    split_fold,
    model_factory,
    model_name,
    booster,
    preprocessing_func,
    selected_features=None,
):
    oof_pred = []
    rmsle_by_folds = []
    features_import = []
    models = []
    model_features = []

    for fold in time_folds:
        train, val = split_fold(df=train_df, fold=fold)

        with_features_eng = preprocessing_func.__name__ == "prep_with_features_eng"
        train, val = preprocess_train_val(
            train, val, with_features_eng=with_features_eng
        )

        if selected_features is None:
            X_train = train.drop("price_doc", axis=1)
            X_val = val.drop("price_doc", axis=1)
        else:
            X_train = train[selected_features]
            X_val = val[selected_features]

        y_train = train["price_doc"]
        y_val = val["price_doc"]

        val_ids = val.index.copy()

        cat_features = X_train.select_dtypes(
            include="category"
        ).columns.to_list()

        model = model_factory

        if booster == "catboost":
            X_train, X_val, cat_features = _catboost_string_encoding(X_train, X_val)

        if booster == "lightgbm":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                categorical_feature=cat_features,
                callbacks=[
                    lightgbm.early_stopping(stopping_rounds=200),
                    lightgbm.log_evaluation(period=200),
                ],
            )

            pred_log = model.predict(
                X_val,
                num_iteration=model.best_iteration_
            )

            feature_importance = model.feature_importances_

        elif booster == "xgboost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=200,
            )

            pred_log = model.predict(X_val)

            feature_importance = model.feature_importances_

        elif booster == "catboost":
            model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )

            pred_log = model.predict(X_val)

            feature_importance = model.get_feature_importance()

        else:
            raise ValueError(
                "booster должен быть: 'lightgbm', 'xgboost' или 'catboost'"
            )

        pred = np.expm1(pred_log)

        score = rmsle(np.expm1(y_val), pred)
        error = pred - np.expm1(y_val)

        rmsle_by_folds.append(score)

        models.append({
            "booster": booster,
            "fold": fold["fold"],
            "model": model,
            "features": X_train.columns.to_list()
        })

        features_import.append(feature_importance)
        model_features.append(X_train.columns.to_list())

        fold_oof = pd.DataFrame({
            "id": val_ids,
            "fold": fold["fold"],
            "val_block": fold["val_block"],
            "y_true": np.expm1(y_val).values,
            f"{model_name}_pred": pred,
            f"{model_name}_error": error
        })

        oof_pred.append(fold_oof)

    quality = pd.DataFrame([{
        "model_name": model_name,
        "fold_1": rmsle_by_folds[0],
        "fold_2": rmsle_by_folds[1],
        "fold_3": rmsle_by_folds[2],
    }])

    quality["mean_rmsle"] = quality[
        ["fold_1", "fold_2", "fold_3"]
    ].mean(axis=1)

    quality["rmsle_std"] = quality[
        ["fold_1", "fold_2", "fold_3"]
    ].std(axis=1)

    oof_pred = pd.concat(oof_pred)

    return quality, oof_pred, models, features_import, model_features


def prepare_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list,
):
    X_train = prep_with_features_eng(train_df)
    cat_categories = fit_cat_categories(X_train)
    X_test = prep_with_features_eng(test_df, cat_categories=cat_categories)
    y_train = X_train["price_doc"]
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    cat_features = X_train.select_dtypes(include="category").columns.tolist()
    return X_train, X_test, y_train, cat_categories, cat_features


def _apply_catboost_encoding(X_train, X_test, cat_features):
    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()
    for col in cat_features:
        X_train_cat[col] = X_train_cat[col].astype(str).fillna("__MISSING__")
        X_test_cat[col] = X_test_cat[col].astype(str).fillna("__MISSING__")
    return X_train_cat, X_test_cat


def create_xgb_model(xgb_params: dict) -> XGBRegressor:
    return XGBRegressor(
        **xgb_params,
        eval_metric="rmse",
        tree_method="hist",
        objective="reg:squarederror",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )


def create_lgb_model(lgb_params: dict) -> LGBMRegressor:
    return LGBMRegressor(
        **lgb_params,
        objective="regression",
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
    )


def create_cat_model(cat_params: dict, verbose: int = 200) -> CatBoostRegressor:
    return CatBoostRegressor(
        **cat_params,
        eval_metric="RMSE",
        loss_function="RMSE",
        random_seed=42,
        verbose=verbose,
    )


def train_final_models(
    X_train,
    y_train,
    X_test,
    cat_features,
    params: dict,
    verbose: int = 200,
):
    model_xgb = create_xgb_model(params["xgb"])
    model_xgb.fit(X_train, y_train, verbose=verbose)

    X_train_cat, X_test_cat = _apply_catboost_encoding(X_train, X_test, cat_features)
    model_cat = create_cat_model(params["cat"], verbose=verbose)
    model_cat.fit(X_train_cat, y_train, cat_features=cat_features)

    model_lgb = create_lgb_model(params["lgb"])
    model_lgb.fit(
        X_train,
        y_train,
        eval_metric="rmse",
        categorical_feature=cat_features,
    )

    return {
        "xgb": model_xgb,
        "lgb": model_lgb,
        "cat": model_cat,
        "X_test_cat": X_test_cat,
    }


def predict_base_models(models: dict, X_test, X_test_cat=None) -> dict:
    xgb_pred_log = models["xgb"].predict(X_test)
    lgb_pred_log = models["lgb"].predict(X_test)
    if X_test_cat is None:
        X_test_cat = models.get("X_test_cat", X_test)
    cat_pred_log = models["cat"].predict(X_test_cat)
    return {
        "xgboost_100_tuned_pred": xgb_pred_log,
        "lightgbm_100_tuned_pred": lgb_pred_log,
        "catboost_100_tuned_pred": cat_pred_log,
    }


def save_boosting_models(models: dict, model_paths: dict) -> None:
    Path(model_paths["xgb"]).parent.mkdir(parents=True, exist_ok=True)
    models["xgb"].save_model(model_paths["xgb"])
    with open(model_paths["lgb"], "w", encoding="utf-8") as f:
        f.write(models["lgb"].booster_.model_to_string())
    models["cat"].save_model(model_paths["cat"])


def load_boosting_models(model_paths: dict, params: dict):
    model_xgb = XGBRegressor()
    model_xgb.load_model(model_paths["xgb"])

    with open(model_paths["lgb"], encoding="utf-8") as f:
        model_str = f.read()
    booster = lgb.Booster(model_str=model_str)
    model_lgb = create_lgb_model(params["lgb"])
    model_lgb._Booster = booster
    model_lgb.fitted_ = True

    model_cat = CatBoostRegressor()
    model_cat.load_model(model_paths["cat"])

    return {"xgb": model_xgb, "lgb": model_lgb, "cat": model_cat}


def build_oof_predictions(
    train_df: pd.DataFrame,
    selected_features: list,
    params: dict,
    model_names: dict | None = None,
):
    model_names = model_names or {
        "xgb": "xgboost_100_tuned",
        "lgb": "lightgbm_100_tuned",
        "cat": "catboost_100_tuned",
    }
    time_folds = get_time_folds()
    oof_frames = []

    specs = [
        ("xgb", "xgboost", create_xgb_model(params["xgb"])),
        ("lgb", "lightgbm", create_lgb_model(params["lgb"])),
        ("cat", "catboost", create_cat_model(params["cat"], verbose=0)),
    ]

    for key, booster, model in specs:
        _, oof_pred, _, _, _ = run_boosting_cv(
            train_df=train_df,
            time_folds=time_folds,
            split_fold=split_fold,
            model_factory=model,
            model_name=model_names[key],
            booster=booster,
            preprocessing_func=prep_with_features_eng,
            selected_features=selected_features,
        )
        oof_frames.append(oof_pred.reset_index(drop=True))

    oof = oof_frames[0][["id", "fold", "val_block", "y_true"]]
    for frame in oof_frames:
        pred_col = [c for c in frame.columns if c.endswith("_pred")][0]
        err_col = [c for c in frame.columns if c.endswith("_error")][0]
        oof = oof.merge(
            frame[["id", pred_col, err_col]],
            on="id",
            how="left",
        )
    return oof


def build_holdout_stack_predictions(
    dev_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    selected_features: list,
    params: dict,
):
    holdout_proc = prep_with_features_eng(holdout_df)
    y_holdout = holdout_proc["price_doc"]

    X_train, X_holdout, y_train, _, cat_features = prepare_train_test(
        dev_df,
        holdout_df,
        selected_features,
    )
    models = train_final_models(
        X_train, y_train, X_holdout, cat_features, params, verbose=0
    )
    preds = predict_base_models(models, X_holdout, models["X_test_cat"])
    return pd.DataFrame({**preds, "y_true": y_holdout.values})