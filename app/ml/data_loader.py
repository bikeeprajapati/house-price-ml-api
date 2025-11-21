import pandas as pd

def load_csv(path: str):
    """
    Load CSV into pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df
