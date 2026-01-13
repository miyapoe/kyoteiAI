# scraper.py
import requests
import pandas as pd

def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def _find_race(items: list, stadium: int, race_no: int):
    for x in items:
        if isinstance(x, dict) and x.get("race_stadium_number") == stadium and x.get("race_number") == race_no:
            return x
    return None

def _boats_to_df(boats):
    """
    programs: boats が list のことが多い
    previews: boats が dict {"1": {...}, ...} のことが多い
    """
    if boats is None:
        return pd.DataFrame()
    if isinstance(boats, list):
        return pd.DataFrame(boats)
    if isinstance(boats, dict):
        return pd.DataFrame(list(boats.values()))
    return pd.DataFrame()

def fetch_race_json(race_date: str, stadium: int, race_no: int):
    """
    race_date: YYYYMMDD
    stadium: API側の場コード（あなたの例は 1）
    race_no: 1-12

    return:
      df_raw: 1艇=1行 DataFrame（出走表＋展示）
      weather: dict（風/波/気温など）
    """
    yyyy = race_date[:4]
    # v2 で統一（あなたのスクショは previews/v2 が動いてる）
    url_prog = f"https://boatraceopenapi.github.io/programs/v2/{yyyy}/{race_date}.json"
    url_prev = f"https://boatraceopenapi.github.io/previews/v2/{yyyy}/{race_date}.json"

    prog_root = _get_json(url_prog)
    prev_root = _get_json(url_prev)

    prog_items = prog_root.get("programs", [])
    prev_items = prev_root.get("previews", [])

    prog = _find_race(prog_items, stadium, race_no)
    if prog is None:
        return pd.DataFrame(), {}

    df_prog = _boats_to_df(prog.get("boats"))
    if df_prog.empty:
        return pd.DataFrame(), {}

    # 出走表側の艇番キーを統一
    if "racer_boat_number" not in df_prog.columns:
        for k in ["boat_number", "lane", "艇番"]:
            if k in df_prog.columns:
                df_prog = df_prog.rename(columns={k: "racer_boat_number"})
                break

    weather = {}
    prev = _find_race(prev_items, stadium, race_no)

    if prev is not None:
        # 気象（レース共通）
        weather = {
            "race_wind": prev.get("race_wind"),
            "race_wind_direction_number": prev.get("race_wind_direction_number"),
            "race_wave": prev.get("race_wave"),
            "race_weather_number": prev.get("race_weather_number"),
            "race_temperature": prev.get("race_temperature"),
            "race_water_temperature": prev.get("race_water_temperature"),
        }

        # 展示（艇別）
        df_prev = _boats_to_df(prev.get("boats"))

        if not df_prev.empty:
            if "racer_boat_number" not in df_prev.columns:
                for k in ["boat_number", "lane", "艇番"]:
                    if k in df_prev.columns:
                        df_prev = df_prev.rename(columns={k: "racer_boat_number"})
                        break

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

    # 表示しやすい並び
    if "racer_boat_number" in df_prog.columns:
        df_prog = df_prog.sort_values("racer_boat_number").reset_index(drop=True)

    return df_prog, weather
