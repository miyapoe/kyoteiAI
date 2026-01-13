# features.py
import pandas as pd
import numpy as np

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def build_features(df_raw: pd.DataFrame, stadium: int = 1, race_no: int = 1) -> pd.DataFrame:
    """
    df_raw(JSON取得結果) -> モデル入力用特徴量
    - 予測時に必要な列を必ず作る
    - label_* は推論では不要なので落とす
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    # --- 推論では不要（学習ラベル）を落とす ---
    for c in ["label_1st", "label_2nd", "label_3rd"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # --- 必須の追加6列（モデルが要求してたやつ） ---
    df["race_no"] = int(race_no)
    df["stadium"] = int(stadium)

    # lane は艇番でOK（1〜6）
    if "lane" not in df.columns:
        if "racer_boat_number" in df.columns:
            df["lane"] = df["racer_boat_number"]
        else:
            df["lane"] = np.arange(1, len(df) + 1)

    # exh_st_rank：展示STの順位（小さい方が良い）
    if "racer_start_timing" in df.columns:
        df["exh_st_rank"] = _num(df["racer_start_timing"]).rank(method="min", ascending=True)
    else:
        df["exh_st_rank"] = 0

    # f_penalty / l_penalty：F/L回数をそのまま入れる（最低限）
    df["f_penalty"] = _num(df["racer_flying_count"]) if "racer_flying_count" in df.columns else 0
    df["l_penalty"] = _num(df["racer_late_count"]) if "racer_late_count" in df.columns else 0

    # --- 数値化（モデルに入れる列が object だと事故るので全部数値に寄せる） ---
    for c in df.columns:
        if c in ["racer_name"]:  # 文字列は使わない
            continue
        if df[c].dtype == "object":
            df[c] = _num(df[c])

    # --- 使わない文字列列を落とす ---
    if "racer_name" in df.columns:
        df = df.drop(columns=["racer_name"])

    # --- 欠損埋め ---
    df = df.fillna(0)

    return df

# 旧名互換（app側で create_features を探すことがあるため）
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
