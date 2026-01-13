# app.py
import streamlit as st
import joblib
import pandas as pd

from scraper import scrape_race_json
from predictor import build_features, predict_123

st.set_page_config(page_title="競艇AI（LightGBM）", layout="wide")
st.title("🚤 競艇AI（LightGBM予測）")
st.caption("JSON（出走表＋展示＋気象）→ 特徴量 → LightGBMで三連単スコア上位を表示")

# 入力
c1, c2, c3, c4 = st.columns(4)
with c1:
    date = st.text_input("開催日（YYYYMMDD）", "20260112")
with c2:
    stadium_no = st.number_input("場コード（API側）", 1, 24, 1)
with c3:
    race_no = st.number_input("レース番号", 1, 12, 1)
with c4:
    top_n = st.slider("表示件数", 5, 60, 10)

# モデル読み込み
@st.cache_resource
def load_models():
    m1 = joblib.load("model1.pkl")
    m2 = joblib.load("model2.pkl")
    m3 = joblib.load("model3.pkl")
    return m1, m2, m3

try:
    model1, model2, model3 = load_models()
except Exception as e:
    st.error(f"モデル読み込み失敗: {e}")
    st.stop()

if st.button("📥 取得＆予測", use_container_width=True):
    with st.spinner("データ取得中..."):
        try:
            df_raw = scrape_race_json(date, int(stadium_no), int(race_no))
        except Exception as e:
            st.error(f"取得失敗: {e}")
            st.stop()

    if df_raw.empty:
        st.error("❌ データが空です（場コード/日付/レース番号が違う可能性）")
        st.stop()

    st.success("✅ 取得成功")

    # 見やすい表示
    show_cols = [
    "racer_boat_number",
    "racer_name",
    "racer_number",
    "racer_weight",
    "wind",
    "wave",
    "temperature",
    "water_temperature",
]
    cols = [c for c in show_cols if c in df_raw.columns]
    st.subheader("📋 取得データ")
    st.dataframe(df_raw[cols] if cols else df_raw, use_container_width=True, hide_index=True)

    # 特徴量
    df_feat = build_features(df_raw)

    # 予測
    with st.spinner("LightGBM予測中..."):
        try:
            df_pred = predict_123(model1, model2, model3, df_feat, top_n=top_n)
        except Exception as e:
            st.error(f"予測失敗: {e}")
            st.stop()

    st.subheader("🎯 三連単予測（スコア上位）")
    st.dataframe(df_pred, use_container_width=True, hide_index=True)

    st.caption("※ scoreは確率の掛け算による近似スコアです（学習時のクラス設計に依存します）。")
