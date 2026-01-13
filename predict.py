import pandas as pd

def predict_trifecta(df_feat):
    """
    仮の予測関数（まずは動作確認用）
    """
    df = df_feat.copy()

    # ダミー結果
    df["score"] = range(len(df), 0, -1)

    result = df[["racer_boat_number", "score"]].sort_values(
        "score", ascending=False
    )

    return result
