# predict.py
import os
import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def _candidate_paths(names):
    """ルート直下 / models/ の両方を探す"""
    out = []
    for n in names:
        out.append(n)
        out.append(os.path.join("models", n))
    return out


def _pick_existing(paths):
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def load_models():
    """
    優先順：
      1) model1-3.txt (LightGBM Booster)
      2) （非推奨）pkl は基本読まない（Python差で壊れやすい）

    戻り値: (m1, m2, m3, info)
    """
    if lgb is None:
        return None, None, None, "lightgbm is not installed"

    # txt を探す
    m1p = _pick_existing(_candidate_paths(["model1.txt"]))
    m2p = _pick_existing(_candidate_paths(["model2.txt"]))
    m3p = _pick_existing(_candidate_paths(["model3.txt"]))

    if not (m1p and m2p and m3p):
        return None, None, None, "model*.txt not found (put in repo root or models/)"

    try:
        m1 = lgb.Booster(model_file=m1p)
        m2 = lgb.Booster(model_file=m2p)
        m3 = lgb.Booster(model_file=m3p)
        return m1, m2, m3, f"LightGBM Booster (.txt) [{m1p}, {m2p}, {m3p}]"
    except Exception as e:
        return None, None, None, f"txt load failed: {e}"


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


def _predict_proba_booster(model, X: pd.DataFrame) -> np.ndarray:
    # LightGBM Booster: binaryなら確率、multiなら各クラス確率になる
    p = model.predict(X)
    p = np.asarray(p)
    # multi-class の場合は「class=1」を返す等、用途に合わせる必要あり
    if p.ndim == 2 and p.shape[1] >= 2:
        return p[:, 1]
    return p.reshape(-1)


def _align_X_to_model(model, X: pd.DataFrame) -> pd.DataFrame:
    feats = _model_feature_names(model)
    X2 = X.copy()

    if feats is None:
        # 情報が取れないならそのまま（事故る可能性はある）
        return X2

    # 欠けてる列は 0 埋め
    for c in feats:
        if c not in X2.columns:
            X2[c] = 0

    # 余分列は捨てて並べ替え
    X2 = X2[feats]
    return X2


def predict_trifecta(model1, model2, model3, df_feat: pd.DataFrame, df_raw: pd.DataFrame | None = None, top_n: int = 10):
    """
    df_feat: 6艇分の特徴量（行=艇）
    df_raw: あると艇番・選手名を表示に付与
    """
    if df_feat is None or df_feat.empty:
        raise ValueError("df_feat is empty")

    # 念のため数値に寄せる
    df_feat = df_feat.select_dtypes(include=["number"]).copy()
    if df_feat.empty:
        raise ValueError("df_feat has no numeric columns")

    X1 = _align_X_to_model(model1, df_feat)
    X2 = _align_X_to_model(model2, df_feat)
    X3 = _align_X_to_model(model3, df_feat)

    p1 = _predict_proba_booster(model1, X1)
    p2 = _predict_proba_booster(model2, X2)
    p3 = _predict_proba_booster(model3, X3)

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
    rows = rows[: int(top_n)]

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
