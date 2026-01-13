# scraper.py
import requests
import pandas as pd

BASE = "https://boatraceopenapi.github.io"

def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def _get_json_fallback(urls) -> dict:
    last = None
    for u in urls:
        try:
            return _get_json(u)
        except Exception as e:
            last = e
            continue
    raise last

def _find_race(items: list, stadium: int, race_no: int):
    for x in items:
        if isinstance(x, dict) and x.get("race_stadium_number") == stadium and x.get("race_number") == race_no:
            return x
    return None

def _boats_to_df(boats):
    # programs: list / previews: dict {"1": {...}, ...} の両対応
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
    stadium: race_stadium_number（あなたのスクショでは 1）
    race_no: 1-12
    return: (df_raw, weather_dict)
    df_raw = 出走表(programs) + 展示(previews) + 気象(previews)
    """

    yyyy = race_date[:4]

    # v2があれば v2、無ければ v1へフォールバック
    programs_urls = [
        f"{BASE}/programs/v2/{yyyy}/{race_date}.json",
        f"{BASE}/programs/v1/{race_date}.json",
    ]
    previews_urls = [
        f"{BASE}/previews/v2/{yyyy}/{race_date}.json",
        f"{BASE}/previews/v1/{race_date}.json",
    ]

    prog_root = _get_json_fallback(programs_urls)
    prev_root = _get_json_fallback(previews_urls)

    programs = prog_root.get("programs", [])
    previews = prev_root.get("previews", [])

    prog_race = _find_race(programs, stadium, race_no)
    prev_race = _find_race(previews, stadium, race_no)

    if prog_race is None:
        return pd.DataFrame(), {}

    df_prog = _boats_to_df(prog_race.get("boats"))

    # 出走表の必須キー（艇番）が無い場合に備えてリネーム保険
    if "racer_boat_number" not in df_prog.columns:
        for k in ["boat_number", "艇番", "lane"]:
            if k in df_prog.columns:
                df_prog = df_prog.rename(columns={k: "racer_boat_number"})
                break

    weather = {}
    if prev_race is not None:
        df_prev = _boats_to_df(prev_race.get("boats"))

        if "racer_boat_number" not in df_prev.columns:
            for k in ["boat_number", "艇番", "lane"]:
                if k in df_prev.columns:
                    df_prev = df_prev.rename(columns={k: "racer_boat_number"})
                    break

        # 展示情報（あれば）
        keep_prev = [
            "racer_boat_number",
            "racer_exhibition_time",   # 展示タイム
            "racer_start_timing",      # 展示ST
            "racer_tilt_adjustment",   # チルト
            "racer_weight_adjustment", # 体重増減（あれば）
        ]
        df_prev = df_prev[[c for c in keep_prev if c in df_prev.columns]].copy()

        # 気象（レース共通）
        weather = {
            "wind": prev_race.get("race_wind"),
            "wind_direction": prev_race.get("race_wind_direction_number"),
            "wave": prev_race.get("race_wave"),
            "weather_code": prev_race.get("race_weather_number"),
            "temperature": prev_race.get("race_temperature"),
            "water_temperature": prev_race.get("race_water_temperature"),
        }

        # df_prog に気象を全行へ付与
        for k, v in weather.items():
            df_prog[k] = v

        # 展示を艇番で結合
        if "racer_boat_number" in df_prog.columns and "racer_boat_number" in df_prev.columns:
            df_prog = df_prog.merge(df_prev, on="racer_boat_number", how="left")

    # 並び
    if "racer_boat_number" in df_prog.columns:
        df_prog = df_prog.sort_values("racer_boat_number").reset_index(drop=True)

    return df_prog, weather
