# app.py
import streamlit as st
import pandas as pd

from scraper import scrape_race_json

# ------------------------------
# ページ設定
# ------------------------------
st.set_page_config(
    page_title="競艇AI（JSON取得版）",
    page_icon="🚤",
    layout="wide"
)

st.title("🚤 競艇AI（JSON取得版）")
st.caption("出走表を公式JSON APIから取得して表示します")

# ------------------------------
# 入力UI
# ------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    date = st.text_input(
        "開催日（YYYYMMDD）",
        value="20260112",
        help="例: 20260112"
    )

with col2:
    stadium_no = st.number_input(
        "場コード（例: 20）",
        min_value=1,
        max_value=24,
        value=20
    )

with col3:
    race_no = st.number_input(
        "レース番号（1〜12）",
        min_value=1,
        max_value=12,
        value=1
    )

st.divider()

# ------------------------------
# 実行ボタン
# ------------------------------
if st.button("📥 取得 & 表示", use_container_width=True):

    with st.spinner("JSONデータ取得中..."):
        try:
            df_raw = scrape_race_json(
                date=date,
                stadium_no=int(stadium_no),
                race_no=int(race_no)
            )
        except Exception as e:
            st.error(f"❌ 取得エラー: {e}")
            st.stop()

    # --------------------------
    # 空データチェック（超重要）
    # --------------------------
    if df_raw.empty:
        st.error(
            "❌ データが取得できませんでした。\n\n"
            "・開催日が違う\n"
            "・場コード / レース番号が存在しない\n"
            "・まだ出走表が公開されていない\n"
        )
        st.stop()

    # --------------------------
    # 表示
    # --------------------------
    st.success("✅ 出走表データ取得成功！")

    st.subheader("📋 出走表（JSON）")
    st.dataframe(
        df_raw,
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"行数: {len(df_raw)}（通常6艇）")

# ------------------------------
# フッター
# ------------------------------
st.divider()
st.caption("Powered by boatrace open JSON API")
