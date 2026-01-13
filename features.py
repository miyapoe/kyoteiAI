# features.py
from __future__ import annotations

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# utils
# ------------------------------------------------------------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _ensure_col(df: pd.DataFrame, col: str, default=np.nan):
    if col not in df.columns:
        df[col] = default
    return df

def _rank_asc(series: pd.Series) -> pd.Series:
    # 小さいほど良い（展示タイムなど）
    return series.rank(method="min", ascending=True)

def _rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(method="min", ascending=False)

def _safe_div(a, b):
    b = np.where(np.asarray(b) == 0, np.nan, b)
    return np.asarray(a) / b

def _load_feature_names(path_candidates=("feature_names.csv", "models/feature_names.csv")):
    for p in path_candidates:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                df = pd.read_csv(p)
                # 1列だけ or "feature"列想定
                if "feature" in df.columns:
                    feats = df["feature"].astype(str).tolist()
                else:
                    feats = df.iloc[:, 0].astype(str).tolist()
                feats = [f for f in feats if f and f != "nan"]
                if feats:
                    return feats
            except Exception:
                pass
    return None

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def build_features(
    df_raw: pd.DataFrame,
    stadium: int = 1,
    race_no: int = 1,
) -> pd.DataFrame:
    """
    Streamlit側から呼ぶ想定。
    - df_raw: scraper(fetch_race_json) の戻り（6艇×列）
    - stadium, race_no: モデルに必要なら入れる
    返り値: モデル入力用の特徴量DF（行=艇）
    """
    if df_raw is None or len(df_raw) == 0:
        return pd.DataFrame()

    x = df_raw.copy()

    # --- 日本語→英語へ寄せる（保険）---
    rename_map = {
        "展示タイム": "racer_exhibition_time",
        "展示ST": "racer_start_timing",
        "チルト": "racer_tilt_adjustment",
        "体重増減": "racer_weight_adjustment",
        "艇番": "racer_boat_number",
        "選手名": "racer_name",
    }
    for ja, en in rename_map.items():
        if ja in x.columns and en not in x.columns:
            x = x.rename(columns={ja: en})

    # --- 絶対ほしい基本（艇ごとに差が出るもの）---
    _ensure_col(x, "racer_boat_number", np.nan)
    _ensure_col(x, "racer_number", np.nan)  # 選手登録番号（あれば強い）
    _ensure_col(x, "racer_exhibition_time", np.nan)
    _ensure_col(x, "racer_start_timing", np.nan)
    _ensure_col(x, "racer_tilt_adjustment", np.nan)
    _ensure_col(x, "racer_weight_adjustment", np.nan)

    # 成績/能力系（programs側に入ってるはず）
    base_cols = [
        "racer_national_top_1_percent",
        "racer_national_top_2_percent",
        "racer_national_top_3_percent",
        "racer_local_top_1_percent",
        "racer_local_top_2_percent",
        "racer_local_top_3_percent",
        "racer_assigned_motor_top_2_percent",
        "racer_assigned_motor_top_3_percent",
        "racer_assigned_boat_top_2_percent",
        "racer_assigned_boat_top_3_percent",
        "racer_average_start_timing",
        "racer_flying_count",
        "racer_late_count",
        "racer_age",
        "racer_weight",
    ]
    for c in base_cols:
        _ensure_col(x, c, np.nan)

    # 気象（レース共通。艇差は出ないが補助にはなる）
    wx_cols = ["wind", "wind_direction", "wave", "weather_code", "temperature", "water_temperature"]
    for c in wx_cols:
        _ensure_col(x, c, np.nan)

    # --- 数値化 ---
    num_like = (
        ["racer_boat_number", "racer_number",
         "racer_exhibition_time", "racer_start_timing",
         "racer_tilt_adjustment", "racer_weight_adjustment"]
        + base_cols
        + wx_cols
    )
    for c in num_like:
        x[c] = _to_num(x[c])

    # --- 艇番（lane） & レース情報 ---
    # model期待名が lane / stadium / race_no っぽいので必ず作る
    x["lane"] = x["racer_boat_number"].copy()
    x["stadium"] = int(stadium)
    x["race_no"] = int(race_no)

    # --- 罰則っぽい特徴（期待名が f_penalty / l_penalty の場合の埋め）---
    # “飛び/出遅れ”の回数があれば反映、無ければ0
    x["f_penalty"] = x["racer_flying_count"].fillna(0)
    x["l_penalty"] = x["racer_late_count"].fillna(0)

    # --- ランク系（艇差が出る重要特徴）---
    # 展示タイムは小さいほど良い
    x["exh_time_rank"] = _rank_asc(x["racer_exhibition_time"])
    # 展示STは小さい（速い）ほど良いがマイナスが入る時もあるので、そのまま小さい順でOK
    x["exh_st_rank"] = _rank_asc(x["racer_start_timing"])

    # --- 差分系（平均との差）---
    # “全艇同じ”になりにくい補助特徴（展示や成績が入ってる前提）
    def add_diff(col: str, out_col: str):
        m = x[col].mean(skipna=True)
        x[out_col] = x[col] - (0 if pd.isna(m) else m)

    add_diff("racer_exhibition_time", "exh_time_diff")
    add_diff("racer_start_timing", "exh_st_diff")
    add_diff("racer_average_start_timing", "avg_st_diff")
    add_diff("racer_weight", "weight_diff")
    add_diff("racer_age", "age_diff")

    # --- 率のまとめ（強さの合成）---
    # national/local のtop率は “小さいほど強い” ではなく “率が高いほど弱い” なので注意。
    # ここでは「上位率が高いほど強い」前提の入力ならそのまま、逆なら学習側に合わせる必要あり。
    x["national_top3_sum"] = (
        x["racer_national_top_1_percent"].fillna(0)
        + x["racer_national_top_2_percent"].fillna(0)
        + x["racer_national_top_3_percent"].fillna(0)
    )
    x["local_top3_sum"] = (
        x["racer_local_top_1_percent"].fillna(0)
        + x["racer_local_top_2_percent"].fillna(0)
        + x["racer_local_top_3_percent"].fillna(0)
    )
    x["motor_top_sum"] = x["racer_assigned_motor_top_2_percent"].fillna(0) + x["racer_assigned_motor_top_3_percent"].fillna(0)
    x["boat_top_sum"] = x["racer_assigned_boat_top_2_percent"].fillna(0) + x["racer_assigned_boat_top_3_percent"].fillna(0)

    # --- 最終：モデルが期待する列に合わせる ---
    feat_names = _load_feature_names()
    feat = x.select_dtypes(include=["number"]).copy()

    if feat_names is not None:
        # 欠けてる列は0で追加、余分列は捨てる、順番も合わせる
        for c in feat_names:
            if c not in feat.columns:
                feat[c] = 0
        feat = feat[feat_names]
    else:
        # feature_names.csv が無い場合の安全策（数値列全部）
        pass

    # --- 欠損は最後に0埋め ---
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    return feat


# 旧名互換（app側で create_features を探すことがあるため）
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
