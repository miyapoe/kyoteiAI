import streamlit as st
import pandas as pd
import joblib
from scraper import scrape_race_data
from feature_engineer import create_features
from predictor import predict_trifecta

st.title("競艇AI予測アプリ")

# --- 入力 ---
place = st.selectbox("場コード（例: 20）", ["20", "21", "22", "23", "24"])
race_no = st.number_input("レース番号（1〜12）", min_value=1, max_value=12, value=1)
race_date = st.date_input("日付を選択")

# --- 実行ボタン ---
if st.button("予測する"):

    with st.spinner("データ取得中..."):
        try:
            df_raw = scrape_race_data(place, race_no, race_date)
        except Exception as e:
            st.error(f"データ取得に失敗しました: {e}")
            st.stop()

    st.success("データ取得成功！")

    st.write("📋 取得したデータ")
    st.dataframe(df_raw)

    # --- 特徴量作成 ---
    df_feat = create_features(df_raw)

    # --- モデル読み込み & 予測 ---
    with st.spinner("予測中..."):
        try:
            model = joblib.load("model/lgbm_trifecta.pkl")
            preds = predict_trifecta(model, df_feat)
        except Exception as e:
            st.error(f"予測に失敗しました: {e}")
            st.stop()

    st.success("予測完了！")

    st.write("🎯 予測三連単（確率上位）")
    st.dataframe(preds.head(10))