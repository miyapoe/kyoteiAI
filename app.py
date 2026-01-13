# app.py
import os
import streamlit as st
import pandas as pd

from scraper import fetch_race_json

# features.py 側の関数名ブレに耐える
build_features = None
try:
    # まずは推奨：build_features（あなたが今使ってる想定）
    from features import build_features as _bf  # type: ignore
    build_features = _bf
except Exception:
    try:
        # 旧：create_features しか無い場合
        from features import create_features as _cf  # type: ignore
        build_features = _cf
    except Exception:
        build_features = None

# 予測モジュール
from predict import load_models, predict_trifecta


# -----------------------------
# Align診断用
# -----------------------------
def _get_model_feature_names(m):
    # LightGBM Booster
    if hasattr(m, "feature_name"):
        try:
            return list(m.feature_name())
        except Exception:
            pass
    # sklearn系
    if hasattr(m, "feature_names_in_"):
        return list(getattr(m, "feature_names_in_"))
    if hasattr(m, "feature_name_"):
        return list(getattr(m, "feature_name_"))
    return None


st.set_page_config(page_title="競艇AI（JSON取得 + LightGBM予測）", layout="wide")
st.title("🚤 競艇AI（JSON取得 + LightGBM予測）")
st.caption("出走表(programs)・展示/気象(previews)を JSON から取得して表示。モデルがあれば三連単予測もします。")

# -----------------------------
# Inputs
# -----------------------------
c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

with c1:
    race_date = st.text_input("開催日（YYYYMMDD）", value="20260112")

with c2:
    stadium = st.number_input(
        "場コード（race_stadium_number）", min_value=1, max_value=30, value=1, step=1
    )

with c3:
    race_no = st.number_input("レース番号（1〜12）", min_value=1, max_value=12, value=1, step=1)

with c4:
    top_n = st.slider("表示件数（予測）", min_value=5, max_value=30, value=10, step=1)

# -----------------------------
# Model load (safe)
# -----------------------------
with st.expander("📦 モデルファイルチェック", expanded=False):
    for fn in ["model1.txt", "model2.txt", "model3.txt", "model1.pkl", "model2.pkl", "model3.pkl"]:
        st.write(
            f"- {fn}: exists={os.path.exists(fn)} size={os.path.getsize(fn) if os.path.exists(fn) else 0}"
        )

model1, model2, model3, model_info = load_models()
if model1 is None or model2 is None or model3 is None:
    st.warning("※ モデル未読込（データ取得のみ動作）")
else:
    st.success(f"✅ モデル読込OK: {model_info}")

# -----------------------------
# Run
# -----------------------------
if st.button("取得＆予測", width="stretch"):
    with st.spinner("データ取得中..."):
        try:
            df_raw, weather = fetch_race_json(race_date, int(stadium), int(race_no))
        except Exception as e:
            st.error(f"❌ 取得失敗: {e}")
            st.stop()

    if df_raw is None or df_raw.empty:
        st.error("❌ データが空です（場コード/日付/レースが違う or APIにデータが無い可能性）")
        st.stop()

    st.success("✅ 取得成功")

    st.subheader("📋 出走表＋展示（取得データ）")
    st.dataframe(df_raw, width="stretch", hide_index=True)

    st.subheader("🔎 df_raw.columns（確認用）")
    st.write(list(df_raw.columns))

    st.subheader("🌤 気象")
    st.json(weather)

    # -----------------------------
    # Feature engineering
    # -----------------------------
    with st.spinner("特徴量作成中..."):
        if build_features is None:
            # 最低限フォールバック（数値列だけ）
            df_feat = df_raw.select_dtypes(include=["number"]).copy()
        else:
            # ★重要：余計な引数を渡さない（TypeError防止）
            # build_features が (df, stadium=..., race_no=...) を受け取れる実装なら渡す
            try:
                df_feat = build_features(df_raw, stadium=int(stadium), race_no=int(race_no))  # type: ignore
            except TypeError:
                # build_features(df) / create_features(df) 型しかない場合
                df_feat = build_features(df_raw)  # type: ignore

    if df_feat is None or df_feat.empty:
        st.error("❌ 特徴量が空です（features.py の処理を確認してください）")
        st.stop()

    # -----------------------------
    # Align診断（ここが「消えてた」原因：このブロックが無かった）
    # -----------------------------
    with st.expander("🪵 align診断（モデル特徴量と一致してる？）", expanded=False):
        if model1 is None:
            st.info("モデル未読込なので診断できません")
        else:
            feats = _get_model_feature_names(model1) or []
            cols = set(df_feat.columns)

            hit = [f for f in feats if f in cols]
            missing = [f for f in feats if f not in cols]

            nunique_min = int(df_feat.nunique(dropna=False).min()) if not df_feat.empty else 0

            st.json({
                "model_feats": len(feats),
                "hit": len(hit),
                "missing": len(missing),
                "sample_missing": missing[:20],
                "nunique_min": nunique_min,
            })

            if missing:
                st.warning("missing がある＝モデルが期待する特徴量が足りないので、0埋めされて精度が落ちやすいです。")

    st.subheader("🧪 特徴量（先頭）")
    st.dataframe(df_feat.head(10), width="stretch", hide_index=True)

    # -----------------------------
    # Predict
    # -----------------------------
    if model1 is None or model2 is None or model3 is None:
        st.error("❌ モデルが読み込めていないため予測できません（model1-3 を配置してください）")
        st.stop()

    with st.spinner("LightGBM予測中..."):
        try:
            df_pred = predict_trifecta(
                model1, model2, model3,
                df_feat,
                df_raw=df_raw,
                top_n=int(top_n)
            )
        except Exception as e:
            st.error(f"❌ 予測失敗: {e}")
            st.stop()

    st.subheader("🎯 三連単予測（スコア上位）")
    st.dataframe(df_pred, width="stretch", hide_index=True)
    st.caption("※ score は 1着×2着×3着 の確率を掛け合わせた近似スコアです。")
