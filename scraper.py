# scraper.py
# ------------------------------------------------------------
# Boatrace Open API (boatraceopenapi.github.io) v2 対応スクレイパ（完成版）
#
# できること：
# - 指定日/指定場/指定レースの「出走表(programs) + 展示/気象(previews) + 結果(results)」を1艇1行で結合
# - 指定日/指定場の「全レース(1〜12R)」を結合して返す（学習用）
# - results が無い（未確定）場合でも落ちない（rank/label は NaN）
#
# 改善点（旧→完成版）：
# - fetch_day_all_races() は daily JSON を 1回だけ取得して 12R を高速生成（36リクエスト→最大3）
# - results の構造揺れ（list/dict）に強い
# - merge 前に racer_boat_number を Int64 に統一してサイレント不一致を防止
# - 学習向けに object→numeric を強めに（coerce）つつ、文字列列は保持
#
# 使い方：
#   from scraper import fetch_race_json, fetch_day_all_races
#   df, meta = fetch_race_json("20260112", stadium=1, race_no=1)
#   df_all = fetch_day_all_races("20260112", stadium=1)
# ------------------------------------------------------------

from __future__ import annotations

import requests
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

BASE = "https://boatraceopenapi.github.io"


# -----------------------------
# Low-level
# -----------------------------
def _get_json(url: str, session: Optional[requests.Session] = None, timeout: int = 25) -> dict:
    s = session or requests
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get_v2_daily(kind: str, yyyymmdd: str, session: Optional[requests.Session] = None) -> dict:
    """
    kind: "programs" / "previews" / "results"
    yyyymmdd: "20260112"
    """
    yyyy = yyyymmdd[:4]
    url = f"{BASE}/{kind}/v2/{yyyy}/{yyyymmdd}.json"
    return _get_json(url, session=session)


def _as_list(v: Any) -> List[Any]:
    """
    list/dict/None を「レースブロックの配列」として扱える形に寄せる。
    - list: そのまま
    - dict: values を配列化（順序は保証されないが _find_race は条件一致で拾う）
    - その他: []
    """
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return list(v.values())
    return []


def _find_race(items: Any, stadium: int, race_no: int) -> Optional[dict]:
    """
    items: root["programs"] / root["previews"] / root["results"] の list/dict/None
    stadium: race_stadium_number
    race_no: race_number
    """
    for x in _as_list(items):
        if isinstance(x, dict) and x.get("race_stadium_number") == stadium and x.get("race_number") == race_no:
            return x
    return None


def _boats_to_df(boats: Any) -> pd.DataFrame:
    """
    - programs: boats=list
    - previews: boats=dict {"1": {...}, ...}
    - results : boats=dict or list の可能性あり
    """
    if boats is None:
        return pd.DataFrame()
    if isinstance(boats, list):
        return pd.DataFrame(boats)
    if isinstance(boats, dict):
        return pd.DataFrame(list(boats.values()))
    return pd.DataFrame()


def _ensure_boat_number(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    艇番列を racer_boat_number に統一
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    if "racer_boat_number" in df.columns:
        return df
    for k in ["boat_number", "lane", "艇番"]:
        if k in df.columns:
            return df.rename(columns={k: "racer_boat_number"})
    return df


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _to_int64(series: pd.Series) -> pd.Series:
    """
    数値化して Int64（nullable int）へ
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _extract_rank_from_results(res_race: dict) -> pd.DataFrame:
    """
    results のレースブロックから艇番ごとの着順(rank)を作る。
    返り値: columns=["racer_boat_number","rank"]
    """
    if not isinstance(res_race, dict):
        return pd.DataFrame(columns=["racer_boat_number", "rank"])

    # 1) boats内に着順が入っている場合
    df_res_boats = _ensure_boat_number(_boats_to_df(res_race.get("boats")))

    rank_col = None
    for c in ["racer_rank", "rank", "arrival", "finish", "racer_arrival", "racer_place"]:
        if c in df_res_boats.columns:
            rank_col = c
            break

    if not df_res_boats.empty and "racer_boat_number" in df_res_boats.columns and rank_col:
        out = df_res_boats[["racer_boat_number", rank_col]].copy()
        out = out.rename(columns={rank_col: "rank"})
        out["racer_boat_number"] = _to_int64(out["racer_boat_number"])
        out["rank"] = _to_num(out["rank"])
        return out[["racer_boat_number", "rank"]]

    # 2) arrival_order（順位配列）がある場合（キー揺れ対応）
    arrival = res_race.get("arrival_order") or res_race.get("arrivalOrder") or res_race.get("arrival")
    # arrival が ["1","2","3","4","5","6"] みたいな艇番順列の可能性
    if isinstance(arrival, list) and len(arrival) >= 3:
        rows = []
        for idx, b in enumerate(arrival, start=1):
            bn = _to_num(b)
            rows.append({"racer_boat_number": bn, "rank": idx})
        out = pd.DataFrame(rows)
        out["racer_boat_number"] = _to_int64(out["racer_boat_number"])
        out["rank"] = _to_num(out["rank"])
        return out[["racer_boat_number", "rank"]]

    # 3) それっぽいキーを総当たり（保険）
    for key in ["ranks", "rankings", "order"]:
        v = res_race.get(key)
        if isinstance(v, dict):
            try:
                out = pd.DataFrame([{"racer_boat_number": k, "rank": v_} for k, v_ in v.items()])
                out["racer_boat_number"] = _to_int64(out["racer_boat_number"])
                out["rank"] = _to_num(out["rank"])
                return out[["racer_boat_number", "rank"]]
            except Exception:
                pass

    return pd.DataFrame(columns=["racer_boat_number", "rank"])


def _extract_trifecta_payout(res_race: dict) -> Optional[float]:
    """
    results から三連単払い戻し(trifecta payout)を拾えれば返す。
    データ構造が揺れるので見つけられない場合は None。
    """
    if not isinstance(res_race, dict):
        return None

    # ありがちなキー候補を探す（存在する時だけ拾う）
    candidates = [
        "trifecta_payout",
        "trifectaPayOut",
        "payout_trifecta",
        "sanrentan_payout",
        "sanrentan",
    ]
    for k in candidates:
        if k in res_race:
            try:
                v = _to_num(res_race.get(k))
                return float(v) if pd.notna(v) else None
            except Exception:
                return None

    # 払戻テーブルっぽいところを探索（保険）
    for k in ["payouts", "payoff", "payoffs", "refunds"]:
        pv = res_race.get(k)
        if isinstance(pv, list):
            for item in pv:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("bet_type") or item.get("type") or "")
                if any(t in name for t in ["三連単", "trifecta", "3連単", "sanrentan"]):
                    amt = item.get("payout") or item.get("amount") or item.get("payoff")
                    v = _to_num(amt)
                    return float(v) if pd.notna(v) else None

    return None


def _add_labels_from_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    rank 列から label_1st/2nd/3rd を作る（学習用）
    """
    out = df.copy()
    if "rank" not in out.columns:
        out["rank"] = np.nan

    out["rank"] = _to_num(out["rank"])

    out["label_1st"] = (out["rank"] == 1).astype("Int64")
    out["label_2nd"] = (out["rank"] == 2).astype("Int64")
    out["label_3rd"] = (out["rank"] == 3).astype("Int64")

    # rankがNaNの行（未確定）では label も NaN にしておく（学習で除外しやすい）
    mask_nan = out["rank"].isna()
    out.loc[mask_nan, ["label_1st", "label_2nd", "label_3rd"]] = pd.NA

    return out


def _merge_on_boat_number(df_left: pd.DataFrame, df_right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    """
    racer_boat_number を Int64 に寄せてから merge（サイレント不一致防止）
    """
    if df_left is None or df_left.empty:
        return df_left
    if df_right is None or df_right.empty:
        return df_left

    if "racer_boat_number" not in df_left.columns or "racer_boat_number" not in df_right.columns:
        return df_left

    df_left = df_left.copy()
    df_right = df_right.copy()
    df_left["racer_boat_number"] = _to_int64(df_left["racer_boat_number"])
    df_right["racer_boat_number"] = _to_int64(df_right["racer_boat_number"])
    return df_left.merge(df_right, on="racer_boat_number", how=how)


def _fetch_day_roots(race_date: str, session: Optional[requests.Session] = None) -> Tuple[dict, dict, dict]:
    """
    daily json を 1回ずつ取得（results は無ければ {}）
    """
    prog_root = _get_v2_daily("programs", race_date, session=session)
    prev_root = _get_v2_daily("previews", race_date, session=session)
    try:
        res_root = _get_v2_daily("results", race_date, session=session)
    except Exception:
        res_root = {}
    return prog_root, prev_root, res_root


def _build_race_df_from_roots(
    race_date: str,
    stadium: int,
    race_no: int,
    prog_root: dict,
    prev_root: dict,
    res_root: dict,
) -> Tuple[pd.DataFrame, dict]:
    """
    取得済み roots から 1レース分を組み立てる（高速化の中核）
    """
    programs = prog_root.get("programs", [])
    previews = prev_root.get("previews", [])
    results = res_root.get("results", [])

    prog_race = _find_race(programs, stadium, race_no)
    prev_race = _find_race(previews, stadium, race_no)
    res_race = _find_race(results, stadium, race_no)

    if prog_race is None:
        return pd.DataFrame(), {}

    # ---- base: programs boats ----
    df_prog = _ensure_boat_number(_boats_to_df(prog_race.get("boats")))
    if df_prog.empty:
        return pd.DataFrame(), {}

    # boat_number を先に固定（merge安全）
    if "racer_boat_number" in df_prog.columns:
        df_prog["racer_boat_number"] = _to_int64(df_prog["racer_boat_number"])

    # ---- meta (weather etc.) from previews ----
    meta: Dict[str, Any] = {}
    if isinstance(prev_race, dict):
        meta = {
            "wind": prev_race.get("race_wind"),
            "wind_direction": prev_race.get("race_wind_direction_number"),
            "wave": prev_race.get("race_wave"),
            "weather_code": prev_race.get("race_weather_number"),
            "temperature": prev_race.get("race_temperature"),
            "water_temperature": prev_race.get("race_water_temperature"),
        }
        for k, v in meta.items():
            df_prog[k] = v

        # ---- merge previews boats (exhibition/st/tilt...) ----
        df_prev = _ensure_boat_number(_boats_to_df(prev_race.get("boats")))
        keep_prev = [
            "racer_boat_number",
            "racer_exhibition_time",   # 展示タイム
            "racer_start_timing",      # 展示ST
            "racer_tilt_adjustment",   # チルト
            "racer_weight_adjustment", # 体重増減（あれば）
        ]
        df_prev = df_prev[[c for c in keep_prev if c in df_prev.columns]].copy()
        df_prog = _merge_on_boat_number(df_prog, df_prev, how="left")

    # ---- merge results (rank/finish) ----
    trifecta_payout = None
    if isinstance(res_race, dict):
        df_rank = _extract_rank_from_results(res_race)
        if not df_rank.empty:
            df_prog = _merge_on_boat_number(df_prog, df_rank, how="left")
        else:
            df_prog["rank"] = np.nan
        trifecta_payout = _extract_trifecta_payout(res_race)
    else:
        df_prog["rank"] = np.nan

    # ---- labels ----
    df_prog = _add_labels_from_rank(df_prog)

    # ---- add ids ----
    df_prog["race_date"] = race_date
    df_prog["stadium"] = int(stadium)
    df_prog["race_no"] = int(race_no)

    if trifecta_payout is not None:
        df_prog["trifecta_payout"] = trifecta_payout
        meta["trifecta_payout"] = trifecta_payout
    else:
        df_prog["trifecta_payout"] = np.nan

    # ---- sort ----
    if "racer_boat_number" in df_prog.columns:
        df_prog["racer_boat_number"] = _to_int64(df_prog["racer_boat_number"])
        df_prog = df_prog.sort_values("racer_boat_number").reset_index(drop=True)

    return df_prog, meta


# -----------------------------
# Public API
# -----------------------------
def fetch_race_json(race_date: str, stadium: int, race_no: int) -> Tuple[pd.DataFrame, dict]:
    """
    v2 programs/previews/results から
    - 出走表(programs)
    - 展示/気象(previews)
    - 結果(results)
    を統合して返す

    return:
      df_all: 1艇=1行 DataFrame（出走+展示+気象+結果列）
      meta:   気象などレース共通情報 dict
    """
    with requests.Session() as session:
        prog_root, prev_root, res_root = _fetch_day_roots(race_date, session=session)
        return _build_race_df_from_roots(
            race_date=race_date,
            stadium=stadium,
            race_no=race_no,
            prog_root=prog_root,
            prev_root=prev_root,
            res_root=res_root,
        )


def fetch_day_all_races(race_date: str, stadium: int) -> pd.DataFrame:
    """
    指定日・指定場(stadium)の 1R〜12R を回して結合して返す（学習用）
    - daily json を 1回ずつ取得して高速化
    - results が無い日/レースは rank/label が NaN のまま入る
      → train.py 側で label がある行だけ使えばOK
    """
    with requests.Session() as session:
        prog_root, prev_root, res_root = _fetch_day_roots(race_date, session=session)

    all_df: List[pd.DataFrame] = []
    for rno in range(1, 13):
        try:
            df_r, _ = _build_race_df_from_roots(
                race_date=race_date,
                stadium=stadium,
                race_no=rno,
                prog_root=prog_root,
                prev_root=prev_root,
                res_root=res_root,
            )
            if df_r is not None and not df_r.empty:
                all_df.append(df_r)
        except Exception:
            continue

    if not all_df:
        return pd.DataFrame()

    df = pd.concat(all_df, ignore_index=True)

    # -----------------------------
    # 学習向けの軽い整形
    # -----------------------------
    # 残したい文字列列（必要に応じて増やしてOK）
    keep_text_cols = {
        "racer_name",
        "racer_family_name",
        "racer_first_name",
        "stadium_name",
        "race_name",
    }

    # 数値化を強めに（object→numeric/coerce）
    for c in df.columns:
        if c in keep_text_cols:
            continue
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # boat_number は Int64 を再保証
    if "racer_boat_number" in df.columns:
        df["racer_boat_number"] = _to_int64(df["racer_boat_number"])

    return df
