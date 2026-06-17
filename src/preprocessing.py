import pandas as pd
import numpy as np
import json
from pathlib import Path

def timestamp_prep(df: pd.DataFrame):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df = df.drop('timestamp', axis=1)
    if 'time_block' in df.columns:
        df = df.drop('time_block', axis=1)
    
    return df

def features_engennering(df: pd.DataFrame):
    df["living_ratio"] = df["life_sq"] / df["full_sq"]
    df["kitchen_ratio"] = df["kitch_sq"] / df["full_sq"]
    df["non_living_sq"] = df["full_sq"] - df["life_sq"]
    df["non_living_ratio"] = df["non_living_sq"] / df["full_sq"]

    df["sq_per_room"] = df["full_sq"] / df["num_room"]
    df["life_sq_per_room"] = df["life_sq"] / df["num_room"]

    df["rel_floor"] = df["floor"] / df["max_floor"]
    df["is_first_floor"] = (df["floor"] == 1).astype(int)
    df["is_top_floor"] = (df["floor"] == df["max_floor"]).astype(int)
    df["floor_from_top"] = df["max_floor"] - df["floor"]

    df["house_age"] = df["year"] - df["build_year"]

    return df


def anomalies_deleting(df: pd.DataFrame):
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.drop_duplicates()
    for i in ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 'state', 'raion_popul']:
        df_cleaned.loc[df_cleaned[i] == 0, i] = np.nan

    df_cleaned.loc[df_cleaned['max_floor'] < df_cleaned['floor'], ['max_floor', 'floor']] = np.nan
    df_cleaned.loc[df_cleaned['life_sq'] > df_cleaned['full_sq'], ['life_sq', 'full_sq']] = np.nan
    df_cleaned.loc[df_cleaned['kitch_sq'] > df_cleaned['full_sq'], ['kitch_sq', 'full_sq']] = np.nan
    df_cleaned.loc[df_cleaned['kitch_sq'] > df_cleaned['life_sq'], ['kitch_sq', 'life_sq']] = np.nan
    df_cleaned.loc[(df_cleaned['build_year'] > 2015) | (df_cleaned['build_year'] < 1800), ['build_year']] = np.nan
    df_cleaned.loc[(df_cleaned['full_sq'] > 1000) | (df_cleaned['full_sq'] < 10), ['full_sq']] = np.nan
    df_cleaned.loc[df_cleaned['child_on_acc_pre_school'] == '#!', 'child_on_acc_pre_school'] = np.nan
    macro_cat_columns = ['child_on_acc_pre_school', 'modern_education_share', 'old_education_build_share']
    for i in macro_cat_columns:
        df_cleaned[i] = df_cleaned[i].str.replace(',', '.').astype('float')
    
    return df_cleaned

def log_target(df: pd.DataFrame, target = 'price_doc'):
    if 'price_doc' in df.columns:
        df[target] = np.log1p(df[target])

    return df

def fit_cat_categories(df: pd.DataFrame) -> dict:
    cat_cols = df.select_dtypes(include=['object', 'datetime', 'category']).columns
    categories = {}
    for col in cat_cols:
        if df[col].dtype.name == 'category':
            categories[col] = df[col].cat.categories
        else:
            categories[col] = df[col].astype('category').cat.categories
    return categories


def cat_features_encoding(
    df: pd.DataFrame,
    cat_categories: dict | None = None,
):
    cat_cols = df.select_dtypes(include=['object', 'datetime']).columns
    for col in cat_cols:
        if cat_categories is None:
            df[col] = df[col].astype('category')
        else:
            df[col] = pd.Categorical(df[col], categories=cat_categories[col])
    return df


def preprocess_train_val(train: pd.DataFrame, val: pd.DataFrame, with_features_eng: bool):
    train = anomalies_deleting(train)
    val = anomalies_deleting(val)
    train = timestamp_prep(train)
    val = timestamp_prep(val)
    train = log_target(train)
    val = log_target(val)
    if with_features_eng:
        train = features_engennering(train)
        val = features_engennering(val)
    cat_categories = fit_cat_categories(train)
    train = cat_features_encoding(train)
    val = cat_features_encoding(val, cat_categories)
    return train, val


def base_preprocessing(df: pd.DataFrame, cat_categories: dict | None = None):
    df = anomalies_deleting(df=df)
    df = timestamp_prep(df=df)
    df = log_target(df=df)
    df = cat_features_encoding(df=df, cat_categories=cat_categories)

    return df

def prep_with_features_eng(df: pd.DataFrame, cat_categories: dict | None = None):
    df = anomalies_deleting(df=df)
    df = timestamp_prep(df=df)
    df = log_target(df=df)
    df = features_engennering(df=df)
    df = cat_features_encoding(df=df, cat_categories=cat_categories)
    return df


def save_cat_categories(categories: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {col: categories[col].tolist() for col in categories}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4)


def load_cat_categories(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {col: pd.Index(values) for col, values in data.items()}

