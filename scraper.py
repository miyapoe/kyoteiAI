# scraper.py
import requests
import pandas as pd

def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def _find_race_block(items: list, stadium_no: int, race_no: int):
    """
    items: programs/previews の配列
    stadium_no と race_no が一致するレース dict を返す
    """
    for p in items:
        if not isinstance(p, dict):
            continue
        st = p.get("race_stadium_number")
        rn = p.get("race_number")
        if st == stadium_no and rn == race_no:
            return p
    return None

def _boats_to_df(boats):
    """
    boats が list でも dict でも DataFrame にする
    - programs: boats が list
    - previews: boats が dict {"1": {...}, "2": {...}} の形式
    """
    if boats is None:
        return pd.DataFrame()

    if isinstance(boats, list):
        return pd.DataFrame(boats)

    if isinstance(boats, dict):
        # dictの値を配列化
        vals = []
        for _, v in boats.items():
            if isinstance(v, dict):
                vals.append(v)
        return pd.DataFrame(vals)

    return pd.DataFrame()

def scrape_race_json(date: str, stadium_no: int, race_no: int) -> pd.DataFrame:
    """
    date: YYYYMMDD
    stadium_no: このAPI上の race_stadium_number（あなたのスクショでは 1）
    race_no: 1-12
    return: 出走表(programs) + 展示(previews) + 気象(previews) を 1艇=1行で返す
    """
    yyyy = date[:4]
    url_prog = f"https://boatraceopenapi.github.io/programs/v2/{yyyy}/{date}.json"
    url_prev = f"https://boatraceopenapi.github.io/previews/v2/{yyyy}/{date}.json"

    prog_root = _get_json(url_prog)
    prev_root = _get_json(url_prev)

    prog_items = prog_root.get("programs", [])
    prev_items = prev_root.get("previews", [])

    race_prog = _find_race_block(prog_items, stadium_no, race_no)
    if race_prog is None:
        return pd.DataFrame()

    # --- 出走表(艇) ---
    df_prog = _boats_to_df(race_prog.get("boats"))

    # 出走表側の艇番を統一
    if "racer_boat_number" not in df_prog.columns:
        for k in ["boat_number", "lane", "艇番"]:
            if k in df_prog.columns:
                df_prog = df_prog.rename(columns={k: "racer_boat_number"})
                break

    # --- 展示＋気象 ---
    race_prev = _find_race_block(prev_items, stadium_no, race_no)
    if race_prev is not None:
        df_prev = _boats_to_df(race_prev.get("boats"))

        # 展示側の艇番を統一
        if "racer_boat_number" not in df_prev.columns:
            for k in ["boat_number", "lane", "艇番"]:
                if k in df_prev.columns:
                    df_prev = df_prev.rename(columns={k: "racer_boat_number"})
                    break

        # 必要列だけ残してマージ（キーは艇番）
        keep_prev = [
            "racer_boat_number",
            "racer_exhibition_time",   # 展示タイム
            "racer_start_timing",      # 展示ST
            "racer_tilt_adjustment",   # チルト
            "racer_weight_adjustment", # 体重増減（あれば）
        ]
        df_prev = df_prev[[c for c in keep_prev if c in df_prev.columns]].copy()

        # 気象（レース共通）を全艇へ
        weather_cols = {
            "race_wind": "wind_speed",
            "race_wind_direction_number": "wind_direction",
            "race_wave": "wave_height",
            "race_weather_number": "weather_code",
            "race_temperature": "temperature",
            "race_water_temperature": "water_temperature",
        }
        for src, dst in weather_cols.items():
            if src in race_prev:
                df_prog[dst] = race_prev.get(src)

        # マージ
        if "racer_boat_number" in df_prog.columns and "racer_boat_number" in df_prev.columns:
            df_prog = df_prog.merge(df_prev, on="racer_boat_number", how="left")

    # 並び
    if "racer_boat_number" in df_prog.columns:
        df_prog = df_prog.sort_values("racer_boat_number")

    return df_prog.reset_index(drop=True)
