# features.py
# JSONスクレイプ(df_raw) -> モデル入力(df_feat) を作る
# 目的:
# - 列名ゆらぎ吸収（日本語/英語）
# - 数値化
# - モデルが期待する追加列を必ず作る（race_no, stadium, lane, exh_st_rank, exh_time_rank, f_penalty, l_penalty）
# - 予測時は label_* は落とす（学習時だけ keep_labels=True で残せる）

from __future__ import annotations

import pandas as pd
import numpy as np


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# 日本語→英語の寄せ（必要なら増やしてOK）
_RENAME_MAP = {
    "艇番": "racer_boat_number",
    "選手名": "racer_name",
    "選手番号": "racer_number",
    "展示タイム": "racer_exhibition_time",
    "展示ST": "racer_start_timing",
    "チルト": "racer_tilt_adjustment",
    "体重増減": "racer_weight_adjustment",
    "風": "wind",
    "風向": "wind_direction",
    "波": "wave",
    "天候": "weather_code",
    "気温": "temperature",
    "水温": "water_temperature",
}

# モデル入力として数値化したい（無い場合は0で作る）
_NUMERIC_BASE_COLS = [
    # 出走表(programs)系
    "racer_boat_number",
    "racer_number",
    "racer_class_number",
    "racer_branch_number",
    "racer_birthplace_number",
    "racer_age",
    "racer_weight",
    "racer_flying_count",
    "racer_late_count",
    "racer_average_start_timing",
    "racer_national_top_1_percent",
    "racer_national_top_2_percent",
    "racer_national_top_3_percent",
    "racer_local_top_1_percent",
    "racer_local_top_2_percent",
    "racer_local_top_3_percent",
    "racer_assigned_motor_number",
    "racer_assigned_motor_top_2_percent",
    "racer_assigned_motor_top_3_percent",
    "racer_assigned_boat_number",
    "racer_assigned_boat_top_2_percent",
    "racer_assigned_boat_top_3_percent",
    # 気象(previews)系（scraperで全行に付与される想定）
    "wind",
    "wind_direction",
    "wave",
    "weather_code",
    "temperature",
    "water_temperature",
    # 展示(previews)系
    "racer_exhibition_time",
    "racer_start_timing",
    "racer_tilt_adjustment",
    "racer_weight_adjustment",
]

_LABEL_COLS = ["label_1st", "label_2nd", "label_3rd"]


def build_features(
    df_raw: pd.DataFrame,
    stadium: int | None = None,
    race_no: int | None = None,
    keep_labels: bool = False,
) -> pd.DataFrame:
    """
    予測用の特徴量を作るメイン関数（app.py から呼ぶ想定）

    - stadium / race_no は app 側の入力を渡してOK（Noneなら推定/0）
    - keep_labels=False なら label_* は必ず削除（予測で混ざると事故る）
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    # 1) 列名ゆらぎ吸収（日本語→英語）
    for ja, en in _RENAME_MAP.items():
        if ja in df.columns and en not in df.columns:
            df = df.rename(columns={ja: en})

    # 2) 必要列を確実に用意して数値化
    for c in _NUMERIC_BASE_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = _to_num(df[c])

    # 3) 追加6列（あなたのモデルが期待してたやつ）
    # race_no / stadium
    if "race_no" not in df.columns:
        df["race_no"] = int(race_no) if race_no is not None else 0
    else:
        df["race_no"] = _to_num(df["race_no"]).fillna(int(race_no) if race_no is not None else 0)

    if "stadium" not in df.columns:
        df["stadium"] = int(stadium) if stadium is not None else 0
    else:
        df["stadium"] = _to_num(df["stadium"]).fillna(int(stadium) if stadium is not None else 0)

    # lane = 進入コース相当（基本は艇番で代用）
    if "lane" not in df.columns:
        if "racer_boat_number" in df.columns:
            df["lane"] = df["racer_boat_number"]
        else:
            df["lane"] = 0
    df["lane"] = _to_num(df["lane"])

    # 展示ST順位（モデル名: exh_st_rank）
    if "exh_st_rank" not in df.columns:
        if "racer_start_timing" in df.columns:
            # 小さいほど良い（STが早いほど上位）
            df["exh_st_rank"] = df["racer_start_timing"].rank(method="min", ascending=True)
        else:
            df["exh_st_rank"] = 0
    df["exh_st_rank"] = _to_num(df["exh_st_rank"])

    # 展示タイム順位（モデル名: exh_time_rank） ← 今 missing になってたやつ
    if "exh_time_rank" not in df.columns:
        if "racer_exhibition_time" in df.columns:
            # 小さいほど良い（展示タイムが速いほど上位）
            df["exh_time_rank"] = df["racer_exhibition_time"].rank(method="min", ascending=True)
        else:
            df["exh_time_rank"] = 0
    df["exh_time_rank"] = _to_num(df["exh_time_rank"])

    # 罰則系（無ければ0固定）
    for c in ["f_penalty", "l_penalty"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = _to_num(df[c]).fillna(0)

    # 4) 文字列系は落とす（racer_name など）
    # 予測の入力に文字列が混ざると事故るので確実に排除
    drop_str = []
    for c in df.columns:
        if df[c].dtype == "object":
            drop_str.append(c)
    if drop_str:
        df = df.drop(columns=drop_str, errors="ignore")

    # 5) label_* は予測時は必ず落とす（学習時だけ keep_labels=True）
    if not keep_labels:
        df = df.drop(columns=_LABEL_COLS, errors="ignore")

    # 6) 欠損は0埋め（LightGBMはNaNでも動くけど安定させる）
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 7) 最終的に数値だけにする
    df_feat = df.select_dtypes(include=["number"]).copy()

    # 8) 並び（艇番順）
    if "racer_boat_number" in df_feat.columns:
        df_feat = df_feat.sort_values("racer_boat_number").reset_index(drop=True)

    return df_feat


# app.py が create_features を探しに来るケースがあるので互換用に用意
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
