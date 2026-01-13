# features.py
import pandas as pd
import numpy as np

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    scraper(JSON)が返す df を前提に特徴量を作る
    - 英語キー(推奨): racer_exhibition_time, racer_start_timing, racer_tilt_adjustment
    - 旧日本語キーにも一応対応
    """
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    # --- カラム名のゆらぎ吸収（日本語→英語へ寄せる）---
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

    # --- 数値化（無ければNaNの列を作る）---
    need_cols = [
        "racer_exhibition_time",
        "racer_start_timing",
        "racer_tilt_adjustment",
        "racer_weight_adjustment",
        "wind",
        "wind_direction",
        "wave",
        "weather_code",
        "temperature",
        "water_temperature",
        # 成績系（programs側）
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
        "racer_weight",
        "racer_age",
        "racer_flying_count",
        "racer_late_count",
        "racer_average_start_timing",
    ]
    for c in need_cols:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = _num(x[c])

    # --- 便利な順位特徴（展示タイムがある時だけ）---
    # 小さい方が良いので rank(ascending=True)
    x["exh_time_rank"] = x["racer_exhibition_time"].rank(method="min", ascending=True)
    x["st_rank"] = x["racer_start_timing"].rank(method="min", ascending=True)

    # --- 欠損埋め（LightGBMはNaNでも動くが、安定のため0埋め）---
    feat = x.select_dtypes(include=["number"]).copy()
    feat = feat.fillna(0)

    return feat
