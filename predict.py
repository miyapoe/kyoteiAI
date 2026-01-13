# predict.py
# --------------------------------------------
# LightGBM 1着/2着/3着モデルで「三連単」を作る
#  - model1.txt / model2.txt / model3.txt (推奨: LightGBM Booster)
#  - model1.pkl / model2.pkl / model3.pkl (互換で壊れやすいので非推奨)
#
# 重要:
#  - 学習時の特徴量名と推論時の特徴量名がズレると、欠けた列が0埋めされて
#    「6艇ぜんぶ同じ入力」→「予測ぜんぶ同じ/極小」になりがちです。
# --------------------------------------------

from __future__ import annotations

import os
import joblib
import pandas as pd
import numpy as np

LAST_ALIGN = {}

def get_last_align():
    return LAST_ALIGN

try:
    import lightgbm as lgb
except Exception:
    lgb = None


# -----------------------------
# Utils
# -----------------------------
def _is_nonempty_file(path: str, min_bytes: int = 1024) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


def _model_feature_names(model):
    """
    モデルが覚えてる特徴量名を取る（取れない場合 None）
    """
    # LightGBM Booster
    if hasattr(model, "feature_name"):
        try:
            return list(model.feature_name())
        except Exception:
            pass

    # sklearn-like (LGBMClassifier 等)
    if hasattr(model, "feature_name_"):
        try:
            return list(getattr(model, "feature_name_"))
        except Exception:
            pass
    if hasattr(model, "feature_names_in_"):
        try:
            return list(getattr(model, "feature_names_in_"))
        except Exception:
            pass

    return None


def _align_X_to_model(model, X: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    学習時の列(feats)に合わせて並べ替え＋足りない列を0埋め＋余分列を捨てる
    """
    feats = _model_feature_names(model)
    if feats is None:
        # どうしようもないのでそのまま
        if verbose:
            print("[align] feats=None -> use X as-is")
        return X.copy()

    X2 = X.copy()

    missing = [c for c in feats if c not in X2.columns]
    hit = len(feats) - len(missing)

    # 欠け列は0
    for c in missing:
        X2[c] = 0

    # 余分列は落として順序合わせ
    X2 = X2[feats]

    if verbose:
        print(
            f"[align] model_feats={len(feats)} hit={hit} missing={len(missing)} "
            f"sample_missing={missing[:10]}"
        )
        # 行ごとに同じになってないか軽く診断
        nunique_min = X2.nunique().min() if len(X2.columns) else None
        print(f"[align] X2 shape={X2.shape} nunique_min={nunique_min}")

    return X2


def _predict_proba_binary(model, X: pd.DataFrame) -> np.ndarray:
    """
    1次元の確率っぽい値を返す（Booster / sklearn どちらも対応）
    """
    # LightGBM Booster
    if lgb is not None and isinstance(model, lgb.Booster):
        p = model.predict(X)
        return np.asarray(p).reshape(-1)

    # sklearn-like
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = np.asarray(p)
        # 2値分類なら [:,1]
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].reshape(-1)
        return p.reshape(-1)

    # fallback: predict をそのまま
    p = model.predict(X)
    return np.asarray(p).reshape(-1)


def _safe_float_arr(a, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    # 確率っぽく 0~1 の範囲外があれば軽くクリップ（モデル次第）
    a = np.clip(a, 0.0, 1.0)
    # 0だらけで掛け算が死ぬのを避けたいなら eps を足す（好み）
    # a = np.maximum(a, eps)
    return a


# -----------------------------
# Model loader
# -----------------------------
def load_models(
    base_dir: str = ".",
    prefer_txt: bool = True,
) -> tuple[object | None, object | None, object | None, str]:
    """
    優先順:
      1) model1-3.txt (LightGBM Booster)  ※推奨
      2) model1-3.pkl (joblib)            ※互換問題が起きやすい

    戻り値: (model1, model2, model3, info)
    """
    txts = [os.path.join(base_dir, f"model{i}.txt") for i in (1, 2, 3)]
    pkls = [os.path.join(base_dir, f"model{i}.pkl") for i in (1, 2, 3)]

    # --- txt ---
    if prefer_txt and lgb is not None:
        if all(_is_nonempty_file(p) for p in txts):
            try:
                m1 = lgb.Booster(model_file=txts[0])
                m2 = lgb.Booster(model_file=txts[1])
                m3 = lgb.Booster(model_file=txts[2])
                return m1, m2, m3, "LightGBM Booster (.txt)"
            except Exception as e:
                return None, None, None, f"txt load failed: {e}"

    # --- pkl ---
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in pkls):
        try:
            m1 = joblib.load(pkls[0])
            m2 = joblib.load(pkls[1])
            m3 = joblib.load(pkls[2])
            return m1, m2, m3, "joblib (.pkl)"
        except Exception as e:
            return None, None, None, f"pkl load failed: {e}"

    # txtがpreferでなくても、pklが無いならtxtも見る
    if (not prefer_txt) and lgb is not None:
        if all(_is_nonempty_file(p) for p in txts):
            try:
                m1 = lgb.Booster(model_file=txts[0])
                m2 = lgb.Booster(model_file=txts[1])
                m3 = lgb.Booster(model_file=txts[2])
                return m1, m2, m3, "LightGBM Booster (.txt)"
            except Exception as e:
                return None, None, None, f"txt load failed: {e}"

    return None, None, None, "no models"


# -----------------------------
# Trifecta prediction
# -----------------------------
def predict_trifecta(
    model1,
    model2,
    model3,
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
    top_n: int = 10,
    verbose_align: bool = False,
) -> pd.DataFrame:
    """
    6艇分の特徴量(df_feat)から三連単(順序つき)を作る

    - model1: 1着になる確率モデル
    - model2: 2着になる確率モデル
    - model3: 3着になる確率モデル

    df_raw を渡すと、艇番/選手名を結果に付与する
    """
    if df_feat is None or df_feat.empty:
        raise ValueError("df_feat is empty")

    # 行=艇 の前提
    n = len(df_feat)
    if n < 3:
        raise ValueError("need at least 3 rows (boats)")

    # モデルが欲しい列に揃える
    X1 = _align_X_to_model(model1, df_feat, verbose=verbose_align)
    X2 = _align_X_to_model(model2, df_feat, verbose=verbose_align)
    X3 = _align_X_to_model(model3, df_feat, verbose=verbose_align)

    # 予測
    p1 = _safe_float_arr(_predict_proba_binary(model1, X1))
    p2 = _safe_float_arr(_predict_proba_binary(model2, X2))
    p3 = _safe_float_arr(_predict_proba_binary(model3, X3))

    # 表示用: 艇番・選手名
    boat_nums = None
    names = None
    if df_raw is not None:
        if "racer_boat_number" in df_raw.columns:
            boat_nums = df_raw["racer_boat_number"].tolist()
        if "racer_name" in df_raw.columns:
            names = df_raw["racer_name"].tolist()

    # 三連単（順列）
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
        a = int(boat_nums[i]) if boat_nums is not None else i + 1
        b = int(boat_nums[j]) if boat_nums is not None else j + 1
        c = int(boat_nums[k]) if boat_nums is not None else k + 1

        rec = {
            "1着": a,
            "2着": b,
            "3着": c,
            "score": s,
            "p1": float(p1[i]),
            "p2": float(p2[j]),
            "p3": float(p3[k]),
        }
        if names is not None and len(names) == n:
            rec["1着_選手"] = names[i]
            rec["2着_選手"] = names[j]
            rec["3着_選手"] = names[k]
        out.append(rec)

    return pd.DataFrame(out)
