import pandas as pd
from typing import Tuple

def preprocess_dataframe(df: pd.DataFrame, target_col: str = "price"):
    """
    Basic preprocessing:
    - drop rows with NA
    - get_dummies for categorical columns (drop_first=True)
    - separate X and y
    Returns: X, y, feature_columns
    """
    df = df.dropna().copy()
    # ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    df_processed = pd.get_dummies(df, drop_first=True)
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    feature_cols = X.columns.tolist()
    return X, y, feature_cols
