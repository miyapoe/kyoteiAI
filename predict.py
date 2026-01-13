# predictor.py
import pandas as pd
import numpy as np

def _to_numeric_safe(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    JSON取得DF（出走表＋展示＋気象）→ モデル入力用の特徴量DF
    学習時に使っていた列名と一致していない可能性があるので、
    ここは“最低限”＆“落ちない”作りにしています。
    """
    df = df_raw.copy()

    # よくある数値列を数値化
    num_cols = [
        "racer_boat_number",
        "racer_number",
        "racer_weight",
        "racer_average_start_timing",
        "racer_exhibition_time",
        "racer_start_timing",
        "racer_tilt_adjustment",
        "wind",
        "wave",
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
    ]
    df = _to_numeric_safe(df, num_cols)

    # 展示の順位特徴量（あれば）
    if "racer_exhibition_time" in df.columns:
        df["exh_time_rank"] = df["racer_exhibition_time"].rank(method="min")
    if "racer_start_timing" in df.columns:
        df["exh_st_rank"] = df["racer_start_timing"].rank(method="min")

    # 文字列列はそのまま残しておいて、後でdrop/dummy化する
    return df


def _get_feature_names(model):
    """
    joblib/pickleで保存された LightGBM の形式差異に対応
    - Booster
    - LGBMClassifier / LGBMRanker / LGBMRegressor
    """
    # Booster 直
    if hasattr(model, "feature_name"):
        try:
            return list(model.feature_name())
        except Exception:
            pass

    # sklearn wrapper
    if hasattr(model, "booster_"):
        try:
            return list(model.booster_.feature_name())
        except Exception:
            pass

    # wrapperの feature_name_ がある場合
    if hasattr(model, "feature_name_"):
        try:
            return list(model.feature_name_)
        except Exception:
            pass

    raise ValueError("モデルから特徴量名を取得できません（feature_name が無い）")


def _align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    学習時の特徴量名に合わせて
    - 足りない列を0で追加
    - 余計な列は削除
    - 列順も合わせる
    """
    feat_names = _get_feature_names(model)

    # 足りない列を追加
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0

    # 余計な列を削除＆順序合わせ
    X = X[feat_names]
    return X


def predict_123(model1, model2, model3, df_feat: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    1着/2着/3着モデルの確率を使って三連単の上位を返す
    - model1: 1着
    - model2: 2着
    - model3: 3着
    """
    import itertools

    # 予測に不要な列を落とす（選手名など）
    X = df_feat.copy()
    drop_like = ["racer_name", "race_date", "race_title"]
    X = X.drop(columns=[c for c in drop_like if c in X.columns], errors="ignore")

    # カテゴリ/文字列をダミー化（列ズレ対策）
    X = pd.get_dummies(X, dummy_na=True)

    # 各モデルに合わせて列を整形
    X1 = _align_to_model(X.copy(), model1)
    X2 = _align_to_model(X.copy(), model2)
    X3 = _align_to_model(X.copy(), model3)

    # 予測確率（sklearn wrapperなら predict_proba、Boosterなら predict）
    def proba(m, Xin):
        if hasattr(m, "predict_proba"):
            return m.predict_proba(Xin)
        return m.predict(Xin)

    p1 = proba(model1, X1)
    p2 = proba(model2, X2)
    p3 = proba(model3, X3)

    # クラスの並びが「1号艇=0, 2号艇=1, ...」で学習されている前提
    boats = df_feat["racer_boat_number"].astype(int).tolist()
    idx_map = {b: (b - 1) for b in boats}  # 1→0, 2→1...

    results = []
    for a, b, c in itertools.permutations(boats, 3):
        ia, ib, ic = idx_map[a], idx_map[b], idx_map[c]
        # 各艇の“その艇がその着順になる確率”を掛け算（近似スコア）
        score = float(p1[ia][a-1] * p2[ib][b-1] * p3[ic][c-1])
        results.append({"三連単": f"{a}-{b}-{c}", "score": score})

    out = pd.DataFrame(results).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return out
