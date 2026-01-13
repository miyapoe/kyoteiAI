# features.py
import pandas as pd
import numpy as np

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def build_features(
    df: pd.DataFrame,
    stadium: int | None = None,
    race_no: int | None = None,
) -> pd.DataFrame:
    """
    scraper(JSON)が返す df を前提に特徴量を作る。
    返り値は「学習時に使う想定の数値特徴量のみ（NaNは0埋め）」。

    追加で欲しい列（missing対策）:
      - race_no, stadium, lane, exh_st_rank, f_penalty, l_penalty
    """
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    # --- カラム名ゆらぎ吸収（日本語→英語）---
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

    # --- 数値化（無ければNaN列を作る）---
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

    # --- 順位特徴（小さい方が良い）---
    x["exh_time_rank"] = x["racer_exhibition_time"].rank(method="min", ascending=True)
    x["st_rank"] = x["racer_start_timing"].rank(method="min", ascending=True)

    # ✅ missing対策 6列を必ず作る -----------------
    # race_no / stadium は app から渡すのが正解（渡ってこなければ 0）
    x["race_no"] = int(race_no) if race_no is not None else 0
    x["stadium"] = int(stadium) if stadium is not None else 0

    # lane（枠）＝艇番でOK（なければ連番）
    if "racer_boat_number" in x.columns:
        x["lane"] = _num(x["racer_boat_number"])
    else:
        x["lane"] = np.arange(1, len(x) + 1)

    # exh_st_rank（展示ST順位）：小さい方が良い。欠損は最後扱いにする
    st_tmp = x["racer_start_timing"].copy()
    st_tmp = st_tmp.fillna(st_tmp.max() + 1 if pd.notna(st_tmp.max()) else 999)
    x["exh_st_rank"] = st_tmp.rank(method="min", ascending=True)

    # f_penalty / l_penalty：とりあえず「持ち」= 1/0（学習に合わせたければ後で調整）
    x["f_penalty"] = (x["racer_flying_count"].fillna(0) > 0).astype(int)
    x["l_penalty"] = (x["racer_late_count"].fillna(0) > 0).astype(int)
    # ---------------------------------------------

    # --- 数値だけにして0埋め ---
    feat = x.select_dtypes(include=["number"]).copy()
    feat = feat.fillna(0)

    return feat

# 互換用（appがcreate_featuresをimportしてても動くように）
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
