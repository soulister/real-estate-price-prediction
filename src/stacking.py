import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge

PRED_COLS = [
    "catboost_100_tuned_pred",
    "xgboost_100_tuned_pred",
    "lightgbm_100_tuned_pred",
]


def build_oof_stacking_file(
    stack_df: pd.DataFrame,
    holdout_stack: pd.DataFrame | None = None,
) -> pd.DataFrame:
    train_part = np.log1p(stack_df[PRED_COLS + ["y_true"]])
    if holdout_stack is None:
        return train_part
    holdout_part = holdout_stack[PRED_COLS + ["y_true"]]
    return pd.concat([train_part, holdout_part])


def fit_meta_model(oof_stacking: pd.DataFrame, alpha: float) -> Ridge:
    meta_model = Ridge(alpha=alpha)
    meta_model.fit(oof_stacking[PRED_COLS], oof_stacking["y_true"])
    return meta_model


def predict_stacking(meta_model: Ridge, test_stack_df: pd.DataFrame) -> np.ndarray:
    stack_pred_log = meta_model.predict(test_stack_df[PRED_COLS])
    return np.expm1(stack_pred_log)


def save_meta_model(meta_model: Ridge, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_model, path)


def load_meta_model(path: str | Path) -> Ridge:
    return joblib.load(path)


def load_json_params(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_submission(submission: pd.DataFrame, test_ids: pd.Series) -> None:
    if len(submission) != len(test_ids):
        raise ValueError(
            f"Размер сабмита ({len(submission)}) не совпадает с test ({len(test_ids)})"
        )
    expected_ids = test_ids.reset_index(drop=True)
    actual_ids = submission["id"].reset_index(drop=True)
    if not actual_ids.equals(expected_ids):
        raise ValueError("Порядок id в сабмите не совпадает с test.csv")
