# app.py
import streamlit as st
import pandas as pd

from scraper import fetch_race_json
from features import create_features
from predict import predict_trifecta

st.set_page_config(page_title="競艇AI（JSON取得＋LightGBM予測）", layout="wide")
st.title("🚤 競艇AI（JSON取得＋LightGBM予測）")

c1, c2, c3 = st.columns(3)
with c1:
    race_date = st.text_input("開催日（YYYYMMDD）", "20260112")
with c2:
    stadium = st.number_input("場コード（※ここは1）", min_value=1, max_value=24, value=1, step=1)
with c3:
    race_no = st.number_input("レース番号", min_value=1, max_value=12, value=1, step=1)

if st.button("取得 & 予測"):
    with st.spinner("データ取得中..."):
        try:
            df_raw, weather = fetch_race_json(
                race_date=race_date,
                stadium=int(stadium),
                race_no=int(race_no),
            )
        except Exception as e:
            st.error(f"❌ 取得失敗: {e}")
            st.stop()

    if df_raw is None or df_raw.empty:
        st.error("❌ データが空です（場コード/日付/レース番号が違う可能性）")
        st.stop()

    st.success("✅ 取得成功")

    st.subheader("📋 取得データ（出走表＋展示）")
    show_cols = [
        "racer_boat_number",
        "racer_name",
        "racer_number",
        "racer_weight",
        "racer_exhibition_time",
        "racer_start_timing",
        "racer_tilt_adjustment",
    ]
    cols = [c for c in show_cols if c in df_raw.columns]
    st.dataframe(df_raw[cols] if cols else df_raw, use_container_width=True, hide_index=True)

    st.subheader("🌤
