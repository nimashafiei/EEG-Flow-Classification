import pandas as pd

def rest_center(game_df: pd.DataFrame, rest_df: pd.DataFrame):
    """Center each row by the subject's REST feature mean (per feature)."""
    mu = rest_df.mean(axis=0)
    return game_df - mu, rest_df - mu
