import pandas as pd
import numpy as np

def timestamp_prep(df: pd.DataFrame):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.year
    df = df.drop('timestamp', axis=1)
    df = df.drop('time_block', axis=1)
    
    return df

def anomalies_deleting(df: pd.DataFrame):
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.drop_duplicates()
    for i in ['price_doc', 'full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 'state', 'raion_popul']:
        df_cleaned.loc[df_cleaned[i] == 0, i] = np.nan

    df_cleaned.loc[df_cleaned['max_floor'] < df_cleaned['floor'], ['max_floor', 'floor']] = np.nan
    df_cleaned.loc[df_cleaned['life_sq'] > df_cleaned['full_sq'], ['life_sq', 'full_sq']] = np.nan
    df_cleaned.loc[df_cleaned['kitch_sq'] > df_cleaned['full_sq'], ['kitch_sq', 'full_sq']] = np.nan
    df_cleaned.loc[df_cleaned['kitch_sq'] > df_cleaned['life_sq'], ['kitch_sq', 'life_sq']] = np.nan
    df_cleaned.loc[df_cleaned['kitch_sq'] > df_cleaned['life_sq'], ['kitch_sq', 'life_sq']] = np.nan
    df_cleaned.loc[(df_cleaned['build_year'] > 2015) | (df_cleaned['build_year'] < 1800), ['build_year']] = np.nan
    df_cleaned.loc[(df_cleaned['full_sq'] > 1000) | (df_cleaned['full_sq'] < 10), ['full_sq']] = np.nan
    df_cleaned.loc[df_cleaned['child_on_acc_pre_school'] == '#!', 'child_on_acc_pre_school'] = np.nan
    macro_cat_columns = ['child_on_acc_pre_school', 'modern_education_share', 'old_education_build_share']
    for i in macro_cat_columns:
        df_cleaned[i] = df_cleaned[i].str.replace(',', '.').astype('float')
    
    return df_cleaned

def log_target(df: pd.DataFrame, target = 'price_doc'):
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