import pandas as pd
import os
import argparse

colomn_names = ['project', 'parent_hashes', 'commit_hash', 'author_name',
                'author_email', 'author_date', 'author_date_unix_timestamp',
                'commit_message', 'la', 'ld', 'fileschanged', 'nf', 'ns', 'nd',
                'entropy', 'ndev', 'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp',
                'classification', 'fix', 'is_buggy_commit']
feature_name = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
label_name = ["is_buggy_commit"]


def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df


def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df


def load_data(base_path: str, baseline_name: str):
    pkl_test = pd.read_pickle(os.path.join(base_path, baseline_name, "features_test.pkl"))

    pkl_train = pd.read_pickle(os.path.join(base_path, baseline_name, "features_train.pkl"))
    pkl_train = convert_dtype_dataframe(pkl_train, feature_name)
    pkl_test = convert_dtype_dataframe(pkl_test, feature_name)
    pkl_train = replace_value_dataframe(pkl_train)
    pkl_test = replace_value_dataframe(pkl_test)

    X_train, y_train = pkl_train[feature_name if baseline_name != 'la' else ['la']].values, pkl_train[
        label_name].values.flatten()
    X_test, y_test = pkl_test[feature_name if baseline_name != 'la' else ['la']].values, pkl_test[
        label_name].values.flatten()

    return X_train, y_train, X_test, y_test


def load_test_dataframe(base_path: str, baseline_name: str):
    pkl_test = pd.read_pickle(os.path.join(base_path, baseline_name, "features_test.pkl"))
    pkl_test = convert_dtype_dataframe(pkl_test, feature_name)
    if 'jitline' in baseline_name:
        pkl_test = pkl_test.sort_values(by='commit_hash')
    # effort
    result_df = pd.DataFrame()
    result_df['commit_id'] = pkl_test['commit_hash']
    result_df['LOC'] = pkl_test['la'] + pkl_test['ld']

    result_df['label'] = pkl_test['is_buggy_commit']

    return result_df


def load_deepjit_test_dataframe(result_df):
    pkl_test = pd.read_pickle(os.path.join("data", 'deeper', "features_test.pkl"))
    pkl_test = convert_dtype_dataframe(pkl_test, feature_name)

    # merge according to commit_hash
    result_df = pd.merge(result_df, pkl_test, left_on='commit_id', right_on='commit_hash', how='left')
    # effort
    result_df['LOC'] = result_df['la'] + result_df['ld']

    result_df['label'] = result_df['is_buggy_commit']

    return result_df


