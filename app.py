# app.py
import itertools
import joblib
import pandas as pd
import streamlit as st

from scraper import fetch_race_json

st.set_page_config(page_title="競艇AI（JSON取得＋LightGBM予測）", layout="wide")
st.title("🚤 競艇AI（JSON取得＋LightGBM予測）")

# -----------------------------
# UI
# -----------------------------
c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
with c1:
    race_date = st.text_input("開催日（YYYYMMDD）", "20260112")
with c2:
    stadium = st.number_input("場コード（※ここは1）", min_value=1, max_value=24, value=1, step=1)
with c3:
    race_no = st.number_input("レース番号", min_value=1, max_value=12, value=1, step=1)
with c4:
    top_n = st.slider("表示件数（予測）", 5, 60, 10)

# -----------------------------
# Model load
# -----------------------------
@st.cache_resource
def load_models():
    m1 = joblib.load("model1.pkl")
    m2 = joblib.load("model2.pkl")
    m3 = joblib.load("model3.pkl")
    return m1, m2, m3

def _get_feature_names(model):
    # Booster
    if hasattr(model, "feature_name"):
        return list(model.feature_name())
    # sklearn wrapper
    if hasattr(model, "booster_"):
        return list(model.booster_.feature_name())
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    raise ValueError("モデルから特徴量名を取得できません")

def _align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    feat = _get_feature_names(model)
    for c in feat:
        if c not in X.columns:
            X[c] = 0
    return X[feat]

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 数値化（存在する列だけ）
    num_cols = [
        "racer_boat_number",
        "racer_number",
        "racer_weight",
        "racer_exhibition_time",
        "racer_start_timing",
        "racer_tilt_adjustment",
        "wind",
        "wave",
        "temperature",
        "water_temperature",
        "racer_average_start_timing",
        "racer_national_top_1_percent",
        "racer_national_top_2_percent",
        "racer_national_top_3_percent",
        "racer_local_top_1_percent",
        "racer_local_top_2_percent",
        "racer_local_top_3_percent",
        "racer_assigned_motor_top_2_percent",
        "racer_assigned_motor_top_3_percent",
        "racer_assigned_boat_top_2_percent",
        "racer_assigned_boat_top_3_percent",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 展示順位特徴量
    if "racer_exhibition_time" in df.columns:
        df["exh_time_rank"] = df["racer_exhibition_time"].rank(method="min")
    if "racer_start_timing" in df.columns:
        df["exh_st_rank"] = df["racer_start_timing"].rank(method="min")

    return df

def proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.predict(X)

def predict_trifecta(model1, model2, model3, df_feat: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    # 予測用X（文字列列も含む → dummiesで展開）
    X = df_feat.copy()
    # 予測に邪魔な列があれば落とす
    drop_cols = [c for c in ["racer_name"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    X = pd.get_dummies(X, dummy_na=True)

    X1 = _align_to_model(X.copy(), model1)
    X2 = _align_to_model(X.copy(), model2)
    X3 = _align_to_model(X.copy(), model3)

    p1 = proba(model1, X1)
    p2 = proba(model2, X2)
    p3 = proba(model3, X3)

    boats = df_feat["racer_boat_number"].astype(int).tolist()

    # 前提：クラスが1〜6（インデックス0〜5）に対応してる想定
    def getp(p, boat_num):
        idx = boat_num - 1
        if idx < 0 or idx >= p.shape[1]:
            return 0.0
        # 行も同じ並び（艇番順）前提：念のため boat_num-1 行を参照
        ridx = boat_num - 1
        if ridx < 0 or ridx >= p.shape[0]:
            ridx = 0
        return float(p[ridx][idx])

    rows = []
    for a, b, c in itertools.permutations(boats, 3):
        score = getp(p1, a) * getp(p2, b) * getp(p3, c)
        rows.append({"三連単": f"{a}-{b}-{c}", "score": score})

    out = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return out

# -----------------------------
# Run
# -----------------------------
try:
    model1, model2, model3 = load_models()
except Exception as e:
    st.error(f"モデル読み込み失敗: {e}")
    st.stop()

if st.button("取得＆予測", use_container_width=True):
    with st.spinner("データ取得中..."):
        try:
            df_raw, weather = fetch_race_json(race_date=race_date, stadium=int(stadium), race_no=int(race_no))
        except Exception as e:
            st.error(f"❌ 取得失敗: {e}")
            st.stop()

    if df_raw is None or df_raw.empty:
        st.error("❌ データが空です（場コード/日付/レースが違う可能性）")
        st.stop()

    st.success("✅ 取得成功")

    st.subheader("📋 出走表＋展示（取得データ）")
    st.dataframe(df_raw, use_container_width=True, hide_index=True)

    st.subheader("🌤 気象")
    st.json(weather)

    with st.spinner("特徴量作成中..."):
        df_feat = build_features(df_raw)

    with st.spinner("LightGBM予測中..."):
        try:
            df_pred = predict_trifecta(model1, model2, model3, df_feat, top_n=top_n)
        except Exception as e:
            st.error(f"❌ 予測失敗: {e}")
            st.stop()

    st.subheader("🎯 三連単予測（スコア上位）")
    st.dataframe(df_pred, use_container_width=True, hide_index=True)
    st.caption("※ scoreは各着順モデルの確率を掛け合わせた近似スコアです。")
