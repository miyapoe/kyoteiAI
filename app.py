# app.py
import os
import streamlit as st
import joblib
import pandas as pd

from scraper import fetch_race_json
from predict import build_features, predict_trifecta


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="競艇AI（JSON取得＋LightGBM予測）", layout="wide")
st.title("🚤 競艇AI（JSON取得＋LightGBM予測）")


# -----------------------------
# Sidebar / Inputs
# -----------------------------
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    race_date = st.text_input("開催日（YYYYMMDD）", value="20260112")

with col2:
    # あなたのUIの通り「ここは1」前提にしているなら default=1
    stadium = st.number_input("場コード（※ここは1）", min_value=1, max_value=30, value=1, step=1)

with col3:
    race_no = st.number_input("レース番号", min_value=1, max_value=12, value=1, step=1)

with col4:
    top_n = st.slider("表示件数（予測）", min_value=1, max_value=30, value=10, step=1)


# -----------------------------
# Utils
# -----------------------------
def _validate_date(s: str) -> bool:
    return isinstance(s, str) and len(s) == 8 and s.isdigit()


@st.cache_resource
def load_models_debug():
    """モデル存在確認→ロード。失敗したら例外を上へ投げる"""
    paths = ["model1.pkl", "model2.pkl", "model3.pkl"]

    st.write("### 📦 モデルファイルチェック")
    for p in paths:
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else None
        st.write(f"- `{p}` exists={exists} size={size}")

    m1 = joblib.load(paths[0])
    m2 = joblib.load(paths[1])
    m3 = joblib.load(paths[2])
    return m1, m2, m3


# -----------------------------
# Model load (A: show real error)
# -----------------------------
st.info("※ まずモデルを読み込みます（失敗時は詳細を表示して停止します）")

try:
    model1, model2, model3 = load_models_debug()
    st.success("✅ モデル読込OK")
except Exception as e:
    st.error("❌ モデルロード失敗（詳細）")
    st.exception(e)
    st.stop()


# -----------------------------
# Run
# -----------------------------
if st.button("取得＆予測", use_container_width=True):
    if not _validate_date(race_date):
        st.error("❌ 開催日は YYYYMMDD（8桁数字）で入力してください")
        st.stop()

    # ---- Fetch ----
    with st.spinner("データ取得中..."):
        try:
            # 位置引数で呼ぶ（キーワード引数のズレ事故を避ける）
            df_raw, weather = fetch_race_json(race_date, int(stadium), int(race_no))
        except Exception as e:
            st.error(f"❌ 取得失敗: {e}")
            st.exception(e)
            st.stop()

    if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
        st.error("❌ データが空です（場コード/日付/レースが違う可能性）")
        st.stop()

    st.success("✅ 取得成功")

    # ---- Show fetched ----
    st.subheader("📋 出走表＋展示（取得データ）")
    try:
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
    except Exception:
        # もしdf_rawがDataFrameでないケースでも表示できるよう保険
        st.write(df_raw)

    st.subheader("🌤 気象")
    st.json(weather)

    # ---- Feature ----
    with st.spinner("特徴量作成中..."):
        try:
            df_feat = build_features(df_raw, weather=weather)
        except TypeError:
            # build_features が weather 引数を取らない場合に備えてフォールバック
            df_feat = build_features(df_raw)
        except Exception as e:
            st.error("❌ 特徴量作成失敗")
            st.exception(e)
            st.stop()

    st.subheader("🧪 特徴量（先頭）")
    st.dataframe(df_feat.head(), use_container_width=True, hide_index=True)

    # ---- Predict ----
    with st.spinner("LightGBM予測中..."):
        try:
            df_pred = predict_trifecta(model1, model2, model3, df_feat, top_n=int(top_n))
        except Exception as e:
            st.error("❌ 予測失敗（詳細）")
            st.exception(e)
            st.stop()

    st.subheader("🎯 三連単予測（スコア上位）")
    st.dataframe(df_pred, use_container_width=True, hide_index=True)
    st.caption("※ scoreは各着順モデルの確率を掛け合わせた近似スコアです。")
