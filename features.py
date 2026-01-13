import pandas as pd

def create_features(df):
    # 展示タイム / ST 順位や枠番そのまま
    df["展示タイム順位"] = df["展示タイム"].rank(method="min")
    df["展示ST順位"] = df["展示ST"].rank(method="min")
    return df