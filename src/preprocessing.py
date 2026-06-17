import pandas as pd
import numpy as np

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

def cat_features_encoding(df: pd.DataFrame):
    cat_features = df.select_dtypes(include=['object', 'datetime'])
    for i in cat_features:
        df[i] = df[i].astype('category')
    return df

def base_preprocessing(df: pd.DataFrame):
    df = anomalies_deleting(df=df)
    df = timestamp_prep(df=df)
    df = log_target(df=df)
    df = cat_features_encoding(df=df)

    return df

def prep_with_features_eng(df: pd.DataFrame):
    df = anomalies_deleting(df=df)
    df = timestamp_prep(df=df)
    df = log_target(df=df)
    df = features_engennering(df=df)
    df = cat_features_encoding(df=df)
    return df
    
