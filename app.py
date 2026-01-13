# app.py
import streamlit as st
import pandas as pd
from scraper import scrape_race_json

st.set_page_config(page_title="競艇AI（JSON版）", layout="wide")

st.title("🚤 競艇AI（JSON取得・表示）")
st.caption("出走表＋展示＋風・波をJSON APIから取得")

# ---- 入力 ----
col1, col2, col3 = st.columns(3)

with col1:
    date = st.text_input("開催日（YYYYMMDD）", "20260112")

with col2:
    stadium_no = st.number_input(
        "場コード（※ ここは 1 を入れる）",
        min_value=1,
        max_value=24,
        value=1,
        step=1,
    )

with col3:
    race_no = st.number_input("レース番号", 1, 12, 1)

# ---- 実行 ----
if st.button("取得する"):
    with st.spinner("データ取得中..."):
        try:
            df = scrape_race_json(date, stadium_no, race_no)
        except Exception as e:
            st.error(f"取得失敗: {e}")
            st.stop()

    if df.empty:
        st.error("❌ データが空です（場コード or 日付が違う）")
        st.stop()

    st.success("✅ 取得成功")

    show_cols = [
        "racer_boat_number",
        "racer_name",
        "racer_number",
        "racer_weight",
        "racer_exhibition_time",
        "racer_start_timing",
        "racer_tilt_adjustment",
        "wind",
        "wave",
        "temperature",
        "water_temperature",
    ]

    cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
