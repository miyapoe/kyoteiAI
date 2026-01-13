# predict.py
import os
import joblib
import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def _is_nonempty_file(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1024  # 1KB以上を「中身あり」扱い


def load_models():
    """
    優先順：
      1) model1-3.txt (LightGBM Booster)
      2) model1-3.pkl (joblib)
    戻り値: (m1, m2, m3, info)
    """
    # --- txt ---
    if lgb is not None:
        txts = ["model1.txt", "model2.txt", "model3.txt"]
        if all(_is_nonempty_file(p) for p in txts):
            try:
                m1 = lgb.Booster(model_file=txts[0])
                m2 = lgb.Booster(model_file=txts[1])
                m3 = lgb.Booster(model_file=txts[2])
                return m1, m2, m3, "LightGBM Booster (.txt)"
            except Exception as e:
                # txt が壊れてるなど
                return None, None, None, f"txt load failed: {e}"

    # --- pkl（非推奨：互換問題が起きやすい）---
    pkls = ["model1.pkl", "model2.pkl", "model3.pkl"]
    if all(os.path.exists(p) for p in pkls):
        try:
            m1 = joblib.load(pkls[0])
            m2 = joblib.load(pkls[1])
            m3 = joblib.load(pkls[2])
            return m1, m2, m3, "joblib (.pkl)"
        except Exception as e:
            return None, None, None, f"pkl load failed: {e}"

    return None, None, None, "no models"


def _model_feature_names(model):
    # Booster
    if hasattr(model, "feature_name"):
        try:
            return list(model.feature_name())
        except Exception:
            pass
    # sklearn API
    if hasattr(model, "feature_name_"):
        return list(getattr(model, "feature_name_"))
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    return None


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    # Booster
    if hasattr(model, "predict") and "lightgbm.basic" in str(type(model)).lower():
        return np.asarray(model.predict(X))
    # sklearn-like
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return np.asarray(p)[:, 1]
    # fallback
    return np.asarray(model.predict(X))


def _align_X_to_model(model, X: pd.DataFrame) -> pd.DataFrame:
    feats = _model_feature_names(model)
    if feats is None:
        # そのまま（モデル側が順序依存なら事故るが、情報が無いので）
        return X.copy()

    X2 = X.copy()
    # 欠けてる列は 0 埋め
    for c in feats:
        if c not in X2.columns:
            X2[c] = 0
    # 余分列は無視して並べ替え
    X2 = X2[feats]
    return X2


def predict_trifecta(model1, model2, model3, df_feat: pd.DataFrame, df_raw: pd.DataFrame | None = None, top_n: int = 10):
    """
    df_feat: 6艇分の特徴量（行=艇）
    df_raw: あると艇番・選手名を表示に付与
    """
    if df_feat is None or df_feat.empty:
        raise ValueError("df_feat is empty")

    X1 = _align_X_to_model(model1, df_feat)
    X2 = _align_X_to_model(model2, df_feat)
    X3 = _align_X_to_model(model3, df_feat)

    p1 = _predict_proba(model1, X1)
    p2 = _predict_proba(model2, X2)
    p3 = _predict_proba(model3, X3)

    n = len(df_feat)
    if n < 3:
        raise ValueError("need at least 3 rows (boats)")

    # 表示用：艇番・名前
    boat_nums = None
    names = None
    if df_raw is not None:
        if "racer_boat_number" in df_raw.columns:
            boat_nums = df_raw["racer_boat_number"].tolist()
        if "racer_name" in df_raw.columns:
            names = df_raw["racer_name"].tolist()

    # 3連単：順列
    rows = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                score = float(p1[i] * p2[j] * p3[k])
                rows.append((i, j, k, score))

    rows.sort(key=lambda x: x[3], reverse=True)
    rows = rows[:top_n]

    out = []
    for i, j, k, s in rows:
        if boat_nums is not None:
            a = boat_nums[i]
            b = boat_nums[j]
            c = boat_nums[k]
        else:
            a, b, c = i + 1, j + 1, k + 1

        rec = {
            "1着": int(a),
            "2着": int(b),
            "3着": int(c),
            "score": s,
            "p1": float(p1[i]),
            "p2": float(p2[j]),
            "p3": float(p3[k]),
        }
        if names is not None:
            rec["1着_選手"] = names[i]
            rec["2着_選手"] = names[j]
            rec["3着_選手"] = names[k]
        out.append(rec)

    return pd.DataFrame(out)
