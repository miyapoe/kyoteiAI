import streamlit as st
import pandas as pd

# 自作モジュール
from scraper import fetch_race_json
from features import create_features
from predict import predict_trifecta

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="競艇AI（JSON版）", layout="wide")
st.title("🚤 競艇AI（JSON取得＋LightGBM予測）")

col1, col2, col3 = st.columns(3)

with col1:
    race_date = st.text_input("開催日（YYYYMMDD）", "20260112")

with col2:
    stadium_code = st.number_input("場コード（※ここは1）", min_value=1, max_value=24, value=1)

with col3:
    race_no = st.number_input("レース番号", min_value=1, max_value=12, value=1)

# -----------------------------
# 実行
# -----------------------------
if st.button("取得 & 予測"):
    with st.spinner("データ取得中..."):
        try:
            # JSON取得
            df_raw, weather = fetch_race_json(
                race_date=race_date,
                stadium_code=stadium_code,
                race_no=race_no
            )

            # 空チェック
            if df_raw is None or df_raw.empty:
                st.error("❌ データが空です（レース未開催 or API不正）")
                st.stop()

            st.success("✅ 取得成功")

        except Exception as e:
            st.error(f"❌ 取得失敗: {e}")
            st.stop()

    # -----------------------------
    # 表示（出走表）
    # -----------------------------
    st.subheader("📋 出走表")
    st.dataframe(df_raw, use_)
