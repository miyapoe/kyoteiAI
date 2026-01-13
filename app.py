# app.py
import os
import streamlit as st
import pandas as pd
from predict import load_models, predict_trifecta, get_last_align

from scraper import fetch_race_json

# features.py 側の関数名ブレに耐える
try:
    from features import create_features as build_features
except Exception:
    try:
        from features import build_features  # type: ignore
    except Exception:
        build_features = None

from predict import load_models, predict_trifecta


# -----------------------------
# Page
# -----------------------------
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
    stadium = st.number_input("場コード（race_stadium_number）", min_value=1, max_value=30, value=1, step=1)

with c3:
    race_no = st.number_input("レース番号（1〜12）", min_value=1, max_value=12, value=1, step=1)

with c4:
    top_n = st.slider("表示件数（予測）", min_value=5, max_value=30, value=10, step=1)


# -----------------------------
# Helpers
# -----------------------------
def _file_info(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    return f"exists size={os.path.getsize(path)}"


# -----------------------------
# Model file check
# -----------------------------
with st.expander("📦 モデルファイルチェック", expanded=False):
    candidates = [
        "model1.txt", "model2.txt", "model3.txt",
        "models/model1.txt", "models/model2.txt", "models/model3.txt",
        "model1.pkl", "model2.pkl", "model3.pkl",
        "models/model1.pkl", "models/model2.pkl", "models/model3.pkl",
        "feature_names.csv", "models/feature_names.csv",
    ]
    for fn in candidates:
        st.write(f"- {fn}: {_file_info(fn)}")


# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_models_cached():
    return load_models()

model1, model2, model3, model_info = _load_models_cached()

if model1 is None or model2 is None or model3 is None:
    st.warning(f"※ モデル未読込（データ取得のみ動作） / info: {model_info}")
else:
    st.success(f"✅ モデル読込OK: {model_info}")


# -----------------------------
# Run
# -----------------------------
if st.button("取得＆予測", use_container_width=True):
    # ---- Fetch ----
    with st.spinner("データ取得中..."):
        try:
            # keywordではなく位置引数で固定（race_dateのkeywordズレ事故回避）
            df_raw, weather = fetch_race_json(race_date, int(stadium), int(race_no))
        except Exception as e:
            st.error(f"❌ 取得失敗: {e}")
            st.stop()

    if df_raw is None or df_raw.empty:
        st.error("❌ データが空です（場コード/日付/レースが違う or APIにデータが無い可能性）")
        st.stop()

    st.success("✅ 取得成功")

    st.subheader("📋 出走表＋展示（取得データ）")
    st.dataframe(df_raw, use_container_width=True, hide_index=True)

    st.subheader("🌤 気象")
    st.json(weather)

    # ---- Features ----
    with st.spinner("特徴量作成中..."):
        if build_features is None:
            df_feat = df_raw.select_dtypes(include=["number"]).copy()
        else:
            df_feat = build_features(df_raw)

    if df_feat is None or df_feat.empty:
        st.error("❌ 特徴量が空です（features.py の処理を確認）")
        st.stop()

    # 念のため数値のみ
    df_feat = df_feat.select_dtypes(include=["number"]).copy()

    st.subheader("🧪 特徴量（先頭）")
    st.dataframe(df_feat.head(10), use_container_width=True, hide_index=True)

    # ---- Predict ----
    if model1 is None or model2 is None or model3 is None:
        st.error("❌ モデルが読み込めていないため予測できません（model1-3.txt を配置してください）")
        st.stop()

    st.info("🔎 デバッグ: verbose_align=True（モデル特徴量と一致しているかLogsに出します）")

    with st.spinner("LightGBM予測中..."):
        try:
            df_pred = predict_trifecta(
                model1, model2, model3,
                df_feat,
                df_raw=df_raw,
                top_n=int(top_n),
                verbose_align=True,   # ★これが重要
            )
            st.subheader("🪵 align診断（モデル特徴量と一致してる？）")
            st.json(get_last_align())
        except Exception as e:
            st.error(f"❌ 予測失敗: {e}")
            st.stop()

    st.subheader("🎯 三連単予測（スコア上位）")
    st.dataframe(df_pred, use_container_width=True, hide_index=True)
    st.caption("※ score は 1着×2着×3着 の確率を掛け合わせた近似スコアです。")
