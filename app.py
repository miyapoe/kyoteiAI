# app.py
import os
import streamlit as st
import pandas as pd

from scraper import fetch_race_json

# features.py 側の関数名ブレに耐える
build_features = None
try:
    from features import build_features as _bf  # type: ignore
    build_features = _bf
except Exception:
    try:
        from features import create_features as _cf  # type: ignore
        build_features = _cf
    except Exception:
        build_features = None

# 予測モジュール
from predict import load_models, predict_trifecta


# -----------------------------
# Align診断用
# -----------------------------
def _get_model_feature_names(m):
    # LightGBM Booster
    if hasattr(m, "feature_name"):
        try:
            return list(m.feature_name())
        except Exception:
            pass
    # sklearn系
    if hasattr(m, "feature_names_in_"):
        return list(getattr(m, "feature_names_in_"))
    if hasattr(m, "feature_name_"):
        return list(getattr(m, "feature_name_"))
    return None


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
# Model load
# -----------------------------
with st.expander("📦 モデルファイルチェック", expanded=False):
    for fn in ["model1.txt", "model2.txt", "model3.txt", "model1.pkl", "model2.pkl", "model3.pkl"]:
        st.write(f"- {fn}: exists={os.path.exists(fn)} size={os.path.getsize(fn) if os.path.exists(fn) else 0}")

model1, model2, model3, model_info = load_models()
if model1 is None or model2 is None or model3 is None:
    st.warning(f"※ モデル未読込（データ取得のみ動作） / {model_info}")
else:
    st.success(f"✅ モデル読込OK: {model_info}")

# -----------------------------
# Run
# -----------------------------
if st.button("取得＆予測", width="stretch"):
    # 1) 取得
    with st.spinner("データ取得中..."):
        try:
            df_raw, weather = fetch_race_json(race_date, int(stadium), int(race_no))
        except Exception as e:
            st.error(f"❌ 取得失敗: {e}")
            st.stop()

    if df_raw is None or df_raw.empty:
        st.error("❌ データが空です（場コード/日付/レースが違う or APIにデータが無い可能性）")
        st.stop()

    st.success("✅ 取得成功")

    st.subheader("📋 出走表＋展示（取得データ）")
    st.dataframe(df_raw, width="stretch", hide_index=True)

    with st.expander("🔎 df_raw.columns（確認用）", expanded=False):
        st.write(list(df_raw.columns))

    st.subheader("🌤 気象")
    st.json(weather)

    # 2) 特徴量
    with st.spinner("特徴量作成中..."):
        if build_features is None:
            df_feat = df_raw.select_dtypes(include=["number"]).copy()
        else:
            # build_features が stadium/race_no を受け取れるなら渡す
            try:
                df_feat = build_features(df_raw, stadium=int(stadium), race_no=int(race_no))  # type: ignore
            except TypeError:
                df_feat = build_features(df_raw)  # type: ignore

    if df_feat is None or df_feat.empty:
        st.error("❌ 特徴量が空です（features.py の処理を確認してください）")
        st.stop()

    # 追加6列チェック（超重要）
    need6 = ["race_no", "stadium", "lane", "exh_st_rank", "f_penalty", "l_penalty"]
    with st.expander("🧩 追加6列の確認（df_feat）", expanded=False):
        missing6 = [c for c in need6 if c not in df_feat.columns]
        st.write({"missing6": missing6})
        show_cols = [c for c in need6 if c in df_feat.columns]
        if show_cols:
            st.dataframe(df_feat[show_cols], width="stretch", hide_index=True)

    # align診断
    with st.expander("🪵 align診断（モデル特徴量と一致してる？）", expanded=False):
        if model1 is None:
            st.info("モデル未読込なので診断できません")
        else:
            feats = _get_model_feature_names(model1) or []
            cols = set(df_feat.columns)

            hit = [f for f in feats if f in cols]
            missing = [f for f in feats if f not in cols]
            nunique_min = int(df_feat.nunique(dropna=False).min()) if not df_feat.empty else 0

            st.json({
                "model_feats": len(feats),
                "hit": len(hit),
                "missing": len(missing),
                "sample_missing": missing[:20],
                "nunique_min": nunique_min,
            })

            if missing:
                st.warning("missing がある＝モデルが期待する特徴量が足りないので、0埋めになり精度が落ちやすいです。")

    st.subheader("🧪 特徴量（先頭）")
    st.dataframe(df_feat.head(10), width="stretch", hide_index=True)

    # 3) 予測（モデル未読込ならここで止める）
    if model1 is None or model2 is None or model3 is None:
        st.error("❌ モデルが読み込めていないため予測できません（model1.txt〜model3.txt を配置してください）")
        st.stop()

    with st.spinner("LightGBM予測中..."):
        try:
            df_pred = predict_trifecta(
                model1, model2, model3,
                df_feat=df_feat,
                df_raw=df_raw,
                top_n=int(top_n)
            )
        except Exception as e:
            st.error(f"❌ 予測失敗: {e}")
            st.stop()

    st.subheader("🎯 三連単予測（スコア上位）")
    st.dataframe(df_pred, width="stretch", hide_index=True)
    st.caption("※ score は 1着×2着×3着 の確率を掛け合わせた近似スコアです。")

    # 4) p1/p2/p3 が全部同じ問題のデバッグ
    with st.expander("🧪 p1/p2/p3 デバッグ（同じ？）", expanded=False):
        # predict_trifecta が返すdfに p1/p2/p3 が入ってる前提（あなたのpredictは入れてる）
        cols = [c for c in ["p1", "p2", "p3", "score"] if c in df_pred.columns]
        if cols:
            st.dataframe(df_pred[cols].head(20), width="stretch", hide_index=True)
        else:
            st.info("df_pred に p1/p2/p3 が無いので表示できません（predict.py を確認）")
