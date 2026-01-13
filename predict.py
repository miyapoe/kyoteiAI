# predict.py
# JSONで取った「出走表＋展示＋気象」から特徴量を作り、
# 1着/2着/3着の3モデル(model1-3.pkl)で三連単の上位候補を出す

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _to_float(series: pd.Series) -> pd.Series:
    """数値化（失敗はNaN）"""
    return pd.to_numeric(series, errors="coerce")


def _safe_rank(values: pd.Series, ascending: bool = True) -> pd.Series:
    """順位（欠損は最後扱い）"""
    v = _to_float(values)
    # rankはNaNがNaNになるので、最後に大きい値を入れて順位付け
    fill = v.max() + 9999 if ascending else v.min() - 9999
    v2 = v.fillna(fill)
    return v2.rank(method="min", ascending=ascending)


def _zscore(values: pd.Series) -> pd.Series:
    v = _to_float(values)
    m = v.mean()
    s = v.std()
    if pd.isna(s) or s == 0:
        return (v - m).fillna(0.0)
    return ((v - m) / s).fillna(0.0)


def _get_model_feature_names(model) -> list[str] | None:
    """
    LightGBM/Sklearnモデルから学習時の特徴量名を取得
    - sklearn: feature_names_in_
    - lightgbm.LGBM*: feature_name_
    - lightgbm.Booster: feature_name()
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_name") and callable(getattr(model, "feature_name")):
        try:
            return list(model.feature_name())
        except Exception:
            pass
    return None


def _align_features_to_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    モデルが要求する特徴量に合わせて
    - 足りない列は0で追加
    - 余計な列は落とす
    - 列順を揃える
    """
    names = _get_model_feature_names(model)
    if not names:
        # 特徴量名が取れないモデルは、そのまま渡す
        return X

    X2 = X.copy()
    for c in names:
        if c not in X2.columns:
            X2[c] = 0.0

    X2 = X2[names]
    return X2


def _predict_score_per_boat(model, X: pd.DataFrame) -> np.ndarray:
    """
    1艇ごとの「勝つっぽさ」スコアを出す
    - classifier なら predict_proba の1列目(=positive)を使う
    - regression/booster なら predict を使う
    """
    # sklearn系
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
        # 変な形の時は最後の列
        return proba[:, -1].astype(float)

    # lightgbm Booster / regressor
    pred = model.predict(X)
    pred = np.asarray(pred)
    if pred.ndim == 2:
        # 2Dなら最後の列を採用（念のため）
        pred = pred[:, -1]
    return pred.astype(float)


# -----------------------------
# Feature Engineering
# -----------------------------
def build_features(df_raw: pd.DataFrame, weather: dict | None = None) -> pd.DataFrame:
    """
    df_raw: 1レース6艇ぶんのDataFrame（scraper.fetch_race_json()が返す想定）
    weather: {"wind":.., "wind_direction":.., "wave":.., "weather_code":.., "temperature":.., "water_temperature":..}
    """
    df = df_raw.copy()

    # 必須キー（無いと後で困るので補完）
    if "racer_boat_number" not in df.columns:
        # 旧キーがある場合のフォールバック（念のため）
        if "boat" in df.columns:
            df["racer_boat_number"] = df["boat"]
        elif "艇" in df.columns:
            df["racer_boat_number"] = df["艇"]

    # 数値化しやすい列は明示的にfloatへ
    num_cols = [
        # 展示
        "racer_exhibition_time",
        "racer_start_timing",
        "racer_tilt_adjustment",
        "racer_weight",
        "racer_weight_adjustment",
        # 成績/率
        "racer_national_top_1_percent",
        "racer_national_top_2_percent",
        "racer_national_top_3_percent",
        "racer_local_top_1_percent",
        "racer_local_top_2_percent",
        "racer_local_top_3_percent",
        # モーター/ボート
        "racer_assigned_motor_top_2_percent",
        "racer_assigned_motor_top_3_percent",
        "racer_assigned_boat_top_2_percent",
        "racer_assigned_boat_top_3_percent",
        "racer_assigned_motor_number",
        "racer_assigned_boat_number",
        # 属性
        "racer_age",
        "racer_flying_count",
        "racer_late_count",
        "racer_average_start_timing",
        "racer_class_number",
        "racer_branch_number",
        "racer_birth_place_number",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = _to_float(df[c])

    # あると強い“レース内順位”系
    if "racer_exhibition_time" in df.columns:
        df["exh_time_rank"] = _safe_rank(df["racer_exhibition_time"], ascending=True)  # 速いほど上位
        df["exh_time_z"] = _zscore(df["racer_exhibition_time"])

    if "racer_start_timing" in df.columns:
        # STは小さい(=早い)ほど良いが、マイナスもある
        df["exh_st_rank"] = _safe_rank(df["racer_start_timing"], ascending=True)
        df["exh_st_z"] = _zscore(df["racer_start_timing"])

    # コース(枠)のダミー（艇番=枠として扱う）
    if "racer_boat_number" in df.columns:
        df["lane"] = _to_float(df["racer_boat_number"]).fillna(0).astype(int)
        for i in range(1, 7):
            df[f"lane_{i}"] = (df["lane"] == i).astype(int)

    # ペナルティ系
    if "racer_flying_count" in df.columns:
        df["f_penalty"] = df["racer_flying_count"].fillna(0)
    else:
        df["f_penalty"] = 0.0

    if "racer_late_count" in df.columns:
        df["l_penalty"] = df["racer_late_count"].fillna(0)
    else:
        df["l_penalty"] = 0.0

    # 気象を全艇に付与
    w = weather or {}
    df["wind"] = float(w.get("wind", 0) if w.get("wind", 0) is not None else 0)
    df["wind_direction"] = int(w.get("wind_direction", 0) if w.get("wind_direction", 0) is not None else 0)
    df["wave"] = float(w.get("wave", 0) if w.get("wave", 0) is not None else 0)
    df["weather_code"] = int(w.get("weather_code", 0) if w.get("weather_code", 0) is not None else 0)
    df["temperature"] = float(w.get("temperature", 0) if w.get("temperature", 0) is not None else 0)
    df["water_temperature"] = float(w.get("water_temperature", 0) if w.get("water_temperature", 0) is not None else 0)

    # 風向はカテゴリ扱い（簡単ダミー）
    for d in range(0, 9):  # 0〜8くらいまで来る想定（来なければ0）
        df[f"wind_dir_{d}"] = (df["wind_direction"] == d).astype(int)

    # 使う特徴量だけ抽出（文字列は落とす）
    # ※ ここは後で「モデルの要求列」に合わせて自動調整されるので、
    #   多少多めに残してOK
    keep = []
    for c in df.columns:
        if c in ("racer_name", "race_title", "race_subtitle"):
            continue
        if df[c].dtype == "object":
            continue
        keep.append(c)

    feat = df[keep].copy()

    # 欠損埋め（LightGBMはNaNもいけるが、今回は安定重視）
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return feat


# -----------------------------
# Trifecta prediction
# -----------------------------
def predict_trifecta(
    model1,
    model2,
    model3,
    df_feat: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    model1: 1着モデル
    model2: 2着モデル
    model3: 3着モデル
    df_feat: build_features() の戻り（6行想定）
    top_n: 上位いくつ出すか
    """
    if df_feat is None or df_feat.empty:
        raise ValueError("df_feat is empty")

    # 舟番（艇番）を取りたいので保持
    if "racer_boat_number" not in df_feat.columns:
        raise KeyError("df_feat must contain 'racer_boat_number'")

    boats = df_feat["racer_boat_number"].astype(int).tolist()

    # モデルに合わせて列を調整
    X1 = _align_features_to_model(model1, df_feat)
    X2 = _align_features_to_model(model2, df_feat)
    X3 = _align_features_to_model(model3, df_feat)

    # 予測スコア（艇ごと）
    p1 = _predict_score_per_boat(model1, X1)
    p2 = _predict_score_per_boat(model2, X2)
    p3 = _predict_score_per_boat(model3, X3)

    # 0〜1っぽくない場合もあるので、最低0にクリップ
    p1 = np.clip(p1, 0.0, None)
    p2 = np.clip(p2, 0.0, None)
    p3 = np.clip(p3, 0.0, None)

    # 舟番→index のマップ
    idx = {b: i for i, b in enumerate(boats)}

    rows = []
    for a, b, c in itertools.permutations(boats, 3):
        ia, ib, ic = idx[a], idx[b], idx[c]
        score = float(p1[ia] * p2[ib] * p3[ic])
        rows.append((a, b, c, score, float(p1[ia]), float(p2[ib]), float(p3[ic])))

    df_pred = pd.DataFrame(
        rows,
        columns=["1着", "2着", "3着", "score", "p1_1着", "p2_2着", "p3_3着"],
    ).sort_values("score", ascending=False)

    # 上位抽出
    df_pred = df_pred.head(int(top_n)).reset_index(drop=True)

    return df_pred
