import pandas as pd
import numpy as np

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def build_features(df: pd.DataFrame, stadium: int = 1, race_no: int = 1) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    # --- 文字列→数値化（必要なら） ---
    for c in x.columns:
        if c in ["racer_name"]:
            continue
        if x[c].dtype == "object":
            x[c] = _num(x[c])

    # --- 追加6列（モデルが欲しがるやつ） ---
    if "race_no" not in x.columns:
        x["race_no"] = race_no
    else:
        x["race_no"] = race_no

    if "stadium" not in x.columns:
        x["stadium"] = stadium
    else:
        x["stadium"] = stadium

    if "lane" not in x.columns and "racer_boat_number" in x.columns:
        x["lane"] = x["racer_boat_number"]

    if "exh_st_rank" not in x.columns and "racer_start_timing" in x.columns:
        x["exh_st_rank"] = x["racer_start_timing"].rank(method="min")

    for c in ["f_penalty", "l_penalty"]:
        if c not in x.columns:
            x[c] = 0

    # --- 使わない列（文字列）を落とす ---
    x = x.drop(columns=["racer_name"], errors="ignore")

    # ✅ 超重要：label列は予測時に絶対いらないので落とす
    x = x.drop(columns=[c for c in x.columns if str(c).startswith("label_")], errors="ignore")

    # 欠損埋め
    x = x.fillna(0)

    return x

# 互換用（app.py が create_features を探す場合に備える）
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
