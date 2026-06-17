import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import load_config, get_path
from src.data_split import add_time_blocks, get_dev_holdout_split
from src.preprocessing import save_cat_categories
from src.modeling import (
    prepare_train_test,
    train_final_models,
    predict_base_models,
    save_boosting_models,
    load_boosting_models,
    build_oof_predictions,
    build_holdout_stack_predictions,
    _apply_catboost_encoding,
)
from src.stacking import (
    PRED_COLS,
    build_oof_stacking_file,
    fit_meta_model,
    predict_stacking,
    save_meta_model,
    load_meta_model,
    load_json_params,
    check_submission,
)


def load_raw_data(config: dict):
    train_path = get_path(config, "paths", "train")
    test_path = get_path(config, "paths", "test")
    macro_path = get_path(config, "paths", "macro")

    train_df = pd.read_csv(
        train_path, index_col="id", parse_dates=["timestamp"], low_memory=False
    )
    test_df = pd.read_csv(
        test_path, index_col="id", parse_dates=["timestamp"], low_memory=False
    )
    macro_df = pd.read_csv(macro_path, parse_dates=["timestamp"])
    test_ids = pd.read_csv(test_path, parse_dates=["timestamp"])["id"]

    train_df = train_df.merge(macro_df, on="timestamp", how="left")
    test_df = test_df.merge(macro_df, on="timestamp", how="left")
    return train_df, test_df, test_ids


def load_selected_features(config: dict) -> list:
    with open(get_path(config, "paths", "features"), encoding="utf-8") as f:
        return json.load(f)


def load_model_params(config: dict) -> dict:
    return {
        "xgb": load_json_params(get_path(config, "paths", "params", "xgb")),
        "lgb": load_json_params(get_path(config, "paths", "params", "lgb")),
        "cat": load_json_params(get_path(config, "paths", "params", "cat")),
        "alpha": load_json_params(get_path(config, "paths", "params", "alpha"))["alpha"],
    }


def get_model_paths(config: dict) -> dict:
    return {
        "xgb": get_path(config, "paths", "models", "xgb"),
        "lgb": get_path(config, "paths", "models", "lgb"),
        "cat": get_path(config, "paths", "models", "cat"),
        "meta": get_path(config, "paths", "models", "meta"),
    }


def save_submission(submission: pd.DataFrame, config: dict) -> Path:
    out_dir = get_path(config, "paths", "submissions_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config["training"]["submission_name"]
    submission.to_csv(out_path, index=False)
    return out_path


def load_dev_holdout_frames(config: dict, train_df: pd.DataFrame):
    train_full_path = get_path(config, "paths", "train_full")
    holdout_path = get_path(config, "paths", "holdout")

    if train_full_path.exists() and holdout_path.exists():
        dev_df = pd.read_csv(
            train_full_path, index_col="id", parse_dates=["timestamp"]
        )
        holdout_df = pd.read_csv(
            holdout_path, index_col="id", parse_dates=["timestamp"]
        )
        return dev_df, holdout_df

    blocks_df = add_time_blocks(train_df.reset_index())
    dev_df, holdout_df = get_dev_holdout_split(blocks_df)
    return dev_df.set_index("id"), holdout_df.set_index("id")


def run_predict(config: dict):
    train_df, test_df, test_ids = load_raw_data(config)
    selected_features = load_selected_features(config)
    params = load_model_params(config)
    model_paths = get_model_paths(config)

    X_train, X_test, _, cat_categories, cat_features = prepare_train_test(
        train_df, test_df, selected_features
    )
    save_cat_categories(
        cat_categories,
        get_path(config, "paths", "preprocessing", "cat_categories"),
    )

    models = load_boosting_models(model_paths, params)
    _, X_test_cat = _apply_catboost_encoding(X_train, X_test, cat_features)

    test_stack = pd.DataFrame(predict_base_models(models, X_test, X_test_cat))
    meta_model = load_meta_model(model_paths["meta"])
    stack_pred = predict_stacking(meta_model, test_stack)

    submission = pd.DataFrame({"id": test_ids, "price_doc": stack_pred})
    check_submission(submission, test_ids)
    out_path = save_submission(submission, config)
    print(f"predict: submission saved to {out_path}")


def run_retrain(config: dict):
    train_df, test_df, test_ids = load_raw_data(config)
    selected_features = load_selected_features(config)
    params = load_model_params(config)
    model_paths = get_model_paths(config)
    verbose = config["training"]["verbose"]

    X_train, X_test, y_train, cat_categories, cat_features = prepare_train_test(
        train_df, test_df, selected_features
    )
    save_cat_categories(
        cat_categories,
        get_path(config, "paths", "preprocessing", "cat_categories"),
    )

    models = train_final_models(
        X_train, y_train, X_test, cat_features, params, verbose=verbose
    )
    save_boosting_models(models, model_paths)

    if config["training"]["rebuild_oof_on_retrain"]:
        dev_df, holdout_df = load_dev_holdout_frames(config, train_df)

        stack_df = build_oof_predictions(dev_df, selected_features, params)
        stack_df = stack_df[["y_true"] + PRED_COLS]
        holdout_stack = build_holdout_stack_predictions(
            dev_df, holdout_df, selected_features, params
        )
        oof_stacking = build_oof_stacking_file(stack_df, holdout_stack)
    else:
        oof_path = get_path(config, "paths", "oof_stacking")
        oof_stacking = pd.read_csv(oof_path, index_col=0)

    oof_out = get_path(config, "paths", "oof_stacking")
    oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof_stacking.to_csv(oof_out)

    meta_model = fit_meta_model(oof_stacking, params["alpha"])
    save_meta_model(meta_model, model_paths["meta"])

    test_stack = pd.DataFrame(
        predict_base_models(models, X_test, models["X_test_cat"])
    )
    stack_pred = predict_stacking(meta_model, test_stack)
    submission = pd.DataFrame({"id": test_ids, "price_doc": stack_pred})
    check_submission(submission, test_ids)
    out_path = save_submission(submission, config)
    print(f"retrain: models and submission saved, submission -> {out_path}")


def run_retune(config: dict):
    from src.tuning import tune_all_models

    train_full_path = get_path(config, "paths", "train_full")
    if train_full_path.exists():
        train_df = pd.read_csv(
            train_full_path, index_col="id", parse_dates=["timestamp"]
        )
    else:
        train_df, _, _ = load_raw_data(config)
        blocks_df = add_time_blocks(train_df.reset_index())
        train_df, _ = get_dev_holdout_split(blocks_df)
        train_df = train_df.set_index("id")

    selected_features = load_selected_features(config)
    params_paths = {
        "xgboost": get_path(config, "paths", "params", "xgb"),
        "lightgbm": get_path(config, "paths", "params", "lgb"),
        "catboost": get_path(config, "paths", "params", "cat"),
    }
    retune_cfg = config["retune"]
    results = tune_all_models(
        train_df=train_df,
        selected_features=selected_features,
        params_paths=params_paths,
        n_trials=retune_cfg["n_trials"],
        seed=retune_cfg["seed"],
    )
    for name, result in results.items():
        print(f"retune {name}: mean_rmsle={result['rmsle']:.6f}")
    run_retrain(config)


def run(config: dict | None = None):
    config = config or load_config()
    mode = config["mode"]
    print(f"pipeline mode: {mode}")

    if mode == "predict":
        run_predict(config)
    elif mode == "retrain":
        run_retrain(config)
    elif mode == "retune":
        run_retune(config)
    else:
        raise ValueError("mode должен быть: predict, retrain или retune")


if __name__ == "__main__":
    run()
