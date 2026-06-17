import lightgbm
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import *
from src.data_split import *
from src.metrics import *


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

        train, val = preprocessing_func(train), preprocessing_func(val)

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