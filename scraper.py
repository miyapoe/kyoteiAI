# scraper.py
import requests
import pandas as pd

BASE = "https://boatraceopenapi.github.io"

# -----------------------------
# Low-level
# -----------------------------
def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.json()

def _get_v2_daily(kind: str, yyyymmdd: str) -> dict:
    """
    kind: "programs" / "previews" / "results"
    yyyymmdd: "20260112"
    """
    yyyy = yyyymmdd[:4]
    url = f"{BASE}/{kind}/v2/{yyyy}/{yyyymmdd}.json"
    return _get_json(url)

def _find_race(items: list, stadium: int, race_no: int):
    """
    items: root["programs"] / root["previews"] / root["results"] の配列
    stadium: race_stadium_number
    race_no: race_number
    """
    for x in items:
        if isinstance(x, dict) and x.get("race_stadium_number") == stadium and x.get("race_number") == race_no:
            return x
    return None

def _boats_to_df(boats):
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

def _ensure_boat_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    艇番列を racer_boat_number に統一
    """
    if df.empty:
        return df
    if "racer_boat_number" in df.columns:
        return df
    for k in ["boat_number", "lane", "艇番"]:
        if k in df.columns:
            return df.rename(columns={k: "racer_boat_number"})
    return df

# -----------------------------
# Public API
# -----------------------------
def fetch_race_json(race_date: str, stadium: int, race_no: int):
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
    # ---- get daily json ----
    prog_root = _get_v2_daily("programs", race_date)
    prev_root = _get_v2_daily("previews", race_date)
    res_root  = _get_v2_daily("results",  race_date)

    programs = prog_root.get("programs", [])
    previews = prev_root.get("previews", [])
    results  = res_root.get("results",  [])

    # ---- find target race blocks ----
    prog_race = _find_race(programs, stadium, race_no)
    prev_race = _find_race(previews, stadium, race_no)
    res_race  = _find_race(results,  stadium, race_no)

    if prog_race is None:
        return pd.DataFrame(), {}

    # ---- base: programs boats ----
    df_prog = _boats_to_df(prog_race.get("boats"))
    df_prog = _ensure_boat_number(df_prog)

    # ---- meta (weather etc.) from previews ----
    meta = {}
    if isinstance(prev_race, dict):
        meta = {
            "wind": prev_race.get("race_wind"),
            "wind_direction": prev_race.get("race_wind_direction_number"),
            "wave": prev_race.get("race_wave"),
            "weather_code": prev_race.get("race_weather_number"),
            "temperature": prev_race.get("race_temperature"),
            "water_temperature": prev_race.get("race_water_temperature"),
        }
        # 全艇へ付与
        for k, v in meta.items():
            df_prog[k] = v

    # ---- merge previews boats (exhibition/st/tilt...) ----
    if isinstance(prev_race, dict):
        df_prev = _boats_to_df(prev_race.get("boats"))
        df_prev = _ensure_boat_number(df_prev)

        keep_prev = [
            "racer_boat_number",
            "racer_exhibition_time",   # 展示タイム
            "racer_start_timing",      # 展示ST
            "racer_tilt_adjustment",   # チルト
            "racer_weight_adjustment", # 体重増減（あれば）
        ]
        df_prev = df_prev[[c for c in keep_prev if c in df_prev.columns]].copy()

        if "racer_boat_number" in df_prog.columns and "racer_boat_number" in df_prev.columns:
            df_prog = df_prog.merge(df_prev, on="racer_boat_number", how="left")

    # ---- merge results (rank/finish) ----
    # resultsの構造はデータによって揺れるので、boatsとarrival_order両方を見に行く
    if isinstance(res_race, dict):
        # 1) boats内に着順が入っている場合
        df_res_boats = _boats_to_df(res_race.get("boats"))
        df_res_boats = _ensure_boat_number(df_res_boats)

        # 着順カラム候補
        rank_col = None
        for c in ["racer_rank", "rank", "arrival", "finish", "racer_arrival"]:
            if c in df_res_boats.columns:
                rank_col = c
                break

        if not df_res_boats.empty and "racer_boat_number" in df_res_boats.columns and rank_col:
            df_rank = df_res_boats[["racer_boat_number", rank_col]].copy()
            df_rank = df_rank.rename(columns={rank_col: "rank"})
            df_rank["rank"] = pd.to_numeric(df_rank["rank"], errors="coerce")
            df_prog = df_prog.merge(df_rank, on="racer_boat_number", how="left")

        # 2) arrival_order がある場合（上書き/補完）
        arrival = res_race.get("arrival_order") or res_race.get("arrivalOrder")
        if isinstance(arrival, list) and len(arrival) > 0 and "racer_boat_number" in df_prog.columns:
            # arrivalが艇番配列の想定： ["1","2","3",...]
            rank_map = {}
            for i, v in enumerate(arrival):
                try:
                    b = int(v)
                    rank_map[b] = i + 1
                except Exception:
                    continue
            if rank_map:
                df_prog["rank"] = df_prog["racer_boat_number"].map(rank_map)

        # 3) 3連単払戻（あればmetaに入れる）
        # データ構造が違うので安全に拾うだけ
        meta["trifecta_payout"] = res_race.get("trifecta_payout") or res_race.get("trifectaPayout")

    # ---- sort & label ----
    if "racer_boat_number" in df_prog.columns:
        df_prog = df_prog.sort_values("racer_boat_number").reset_index(drop=True)

    # rankからラベル列（学習用）
    if "rank" in df_prog.columns:
        df_prog["label_1st"] = (df_prog["rank"] == 1).astype(int)
        df_prog["label_2nd"] = (df_prog["rank"] == 2).astype(int)
        df_prog["label_3rd"] = (df_prog["rank"] == 3).astype(int)
    else:
        df_prog["label_1st"] = 0
        df_prog["label_2nd"] = 0
        df_prog["label_3rd"] = 0

    return df_prog, meta


def fetch_day_all_races(race_date: str, stadium: int):
    """
    その日のその場の 1R〜12R をまとめて取って、学習/検証用に使える形で返す。
    return: df_all（行=レース×艇）
    """
    dfs = []
    for rno in range(1, 13):
        df_r, meta = fetch_race_json(race_date, stadium, rno)
        if df_r is None or df_r.empty:
            continue
        df_r["race_date"] = race_date
        df_r["race_no"] = rno
        df_r["stadium"] = stadium
        dfs.append(df_r)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
