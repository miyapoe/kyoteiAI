import streamlit as st
import pandas as pd
import pickle

# ✅ 新 scraper.py（JSON API版）を使う
from scraper import scrape_race_json

st.set_page_config(page_title="競艇AI 予測", layout="wide")
st.title("🚤 競艇AI（JSON取得版）")

st.caption("出走表・展示情報・気象を JSON API から取得して表示します。モデルがあれば予測も実行します。")

# -----------------------------
# 入力
# -----------------------------
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.2])
with col1:
    hd = st.text_input("開催日（YYYYMMDD）", value="20260112")
with col2:
    jcd = st.text_input("場コード（例: 20）", value="20")
with col3:
    rno = st.text_input("レース番号（例: 1）", value="1")
with col4:
    top_n = st.slider("表示件数（予測）", 5, 50, 10)

st.divider()

# -----------------------------
# モデル読み込み（任意）
# -----------------------------
@st.cache_resource
def load_models():
    """
    model1.pkl / model2.pkl / model3.pkl がある場合のみロード。
    無い・壊れている場合は None を返す。
    """
    try:
        m1 = pickle.load(open("model1.pkl", "rb"))
        m2 = pickle.load(open("model2.pkl", "rb"))
        m3 = pickle.load(open("model3.pkl", "rb"))
        return m1, m2, m3
    except Exception:
        return None, None, None

model1, model2, model3 = load_models()

# -----------------------------
# 特徴量作成（最低限）
# -----------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    まず動くこと優先の最低限特徴量。
    列が無い場合も落ちないように安全に処理します。
    """
    out = df.copy()

    # 数値変換（入っていない/文字列でも落ちない）
    for c in ["display_time", "start_timing", "wind_speed", "wave_height", "temperature", "weight"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # ランク特徴（展示タイム・展示ST）
    if "display_time" in out.columns:
        out["display_time_rank"] = out["display_time"].rank(method="min")
    else:
        out["display_time_rank"] = pd.NA

    if "start_timing" in out.columns:
        out["start_timing_rank"] = out["start_timing"].rank(method="min")
    else:
        out["start_timing_rank"] = pd.NA

    # 風向（wind_angle）は数値ならそのまま、無ければ欠損
    if "wind_angle" in out.columns:
        out["wind_angle"] = pd.to_numeric(out["wind_angle"], errors="coerce")
    else:
        out["wind_angle"] = pd.NA

    # 予測に使わない列を落とす用のメモ（必要なら追加）
    return out

# -----------------------------
# 予測（三連単スコア上位）
# -----------------------------
def predict_trifecta(df_feat: pd.DataFrame, m1, m2, m3, top_n: int = 10) -> pd.DataFrame:
    """
    m1: 1着 多クラス（6クラス）
    m2: 2着 多クラス（6クラス）
    m3: 3着 多クラス（6クラス）
    という想定で、三連単をスコア化して上位を返します。

    ※モデルの学習時の特徴量と列が一致していないと動きません。
    """
    import itertools

    # モデル入力のために、文字列列を落とし、ダミー化
    X = df_feat.copy()
    drop_cols = [c for c in ["name", "weather", "rank"] if c in X.columns]
    X = X.drop(columns=drop_cols, errors="ignore")
    X = pd.get_dummies(X)

    # 予測確率（shape: [6,6]）
    p1 = m1.predict(X)
    p2 = m2.predict(X)
    p3 = m3.predict(X)

    # entry_no（1〜6）を基準に三連単作成
    # dfは entry_no 昇順の想定
    entries = df_feat["entry_no"].tolist()

    results = []
    for i, j, k in itertools.permutations(range(len(entries)), 3):
        # ここは学習時のクラス順に依存。一般に 0=1号艇 ... 5=6号艇 を想定
        score = p1[i][entries[i]-1] * p2[j][entries[j]-1] * p3[k][entries[k]-1]
        results.append({
            "三連単": f"{entries[i]}-{entries[j]}-{entries[k]}",
            "score": score
        })

    out = pd.DataFrame(results).sort_values("score", ascending=False).head(top_n)
    return out

# -----------------------------
# 実行
# -----------------------------
if st.button("📥 取得＆表示（＋予測）", use_container_width=True):
    with st.spinner("データ取得中..."):
        try:
            df_raw = scrape_race_json(date=hd, jcd=jcd, rno=rno)
        except Exception as e:
            st.error(f"取得エラー: {e}")
            st.stop()

    st.subheader("📋 取得データ（出走表＋展示＋気象＋結果）")
    st.dataframe(df_raw, use_container_width=True)

    if df_raw.empty:
        st.error("❌ データが空でした。日付/場/レース番号が存在するか確認してください。")
        st.stop()

    # 気象（レース共通）が入っていれば上部に表示
    weather_cols = ["weather", "wind_speed", "wind_angle", "wave_height", "temperature"]
    if all(c in df_raw.columns for c in weather_cols):
        w = df_raw.iloc[0]
        st.info(
            f"🌤 天候: {w.get('weather')} / 風速: {w.get('wind_speed')} "
            f"/ 風向(角度): {w.get('wind_angle')} / 波高: {w.get('wave_height')} / 気温: {w.get('temperature')}"
        )

    # 特徴量
    df_feat = create_features(df_raw)

    st.subheader("🧩 特徴量（最低限）")
    show_cols = [c for c in ["entry_no", "name", "display_time", "start_timing", "display_time_rank", "start_timing_rank",
                             "wind_speed", "wind_angle", "wave_height", "temperature", "weight", "rank"] if c in df_feat.columns]
    st.dataframe(df_feat[show_cols] if show_cols else df_feat, use_container_width=True)

    # 予測（モデルがある場合のみ）
    if model1 is None or model2 is None or model3 is None:
        st.warning("⚠️ model1.pkl / model2.pkl / model3.pkl が読み込めないため、予測はスキップします。")
        st.caption("まずはデータ取得・整形が成功していることを確認してください。モデルを差し替えると予測が動きます。")
        st.stop()

    with st.spinner("予測中..."):
        try:
            df_pred = predict_trifecta(df_feat, model1, model2, model3, top_n=top_n)
        except Exception as e:
            st.error(f"予測エラー: {e}")
            st.stop()

    st.subheader("🎯 予測（スコア上位）")
    st.dataframe(df_pred, use_container_width=True)

    st.caption("※ score は確率の近似スコアです（モデル学習の特徴量・クラス順が一致している必要があります）。")
else:
    st.caption("左の入力を設定して「取得＆表示（＋予測）」を押してください。")
