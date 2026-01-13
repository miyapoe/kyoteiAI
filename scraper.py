import requests
import pandas as pd

def fetch_json(url):
    res = requests.get(url)
    res.raise_for_status()
    return res.json()

def scrape_race_json(date: str, jcd: str, rno: str):
    # API URL
    url_prog = f"https://boatraceopenapi.github.io/programs/v1/{date}.json"
    url_prev = f"https://boatraceopenapi.github.io/previews/v1/{date}.json"
    url_resu = f"https://boatraceopenapi.github.io/results/v2/{date}.json"

    # JSON取得
    programs = fetch_json(url_prog).get(jcd, {}).get(rno, [])
    previews = fetch_json(url_prev).get(jcd, {}).get(rno, {})
    results = fetch_json(url_resu).get(jcd, {}).get(rno, {})

    # --- 出走表 (DataFrame化) ---
    df_prog = pd.DataFrame(programs)

    # --- 展示タイム・ST ---
    df_disp = pd.DataFrame(previews.get("display_time", []))
    df_st = pd.DataFrame(previews.get("start_timing", []))

    # --- 気象データ ---
    weather_data = {
        "weather": previews.get("weather"),
        "wind_angle": previews.get("wind_angle"),
        "wind_speed": previews.get("wind_speed"),
        "wave_height": previews.get("wave_height"),
        "temperature": previews.get("temperature")
    }
    df_prog = df_prog.assign(**weather_data)

    # --- 展示タイム/ST をマージ ---
    df_disp.rename(columns={"entry": "entry_no", "time": "display_time"}, inplace=True)
    df_st.rename(columns={"entry": "entry_no", "start": "start_timing"}, inplace=True)

    df_merged = df_prog.merge(df_disp, on="entry_no", how="left")
    df_merged = df_merged.merge(df_st, on="entry_no", how="left")

    # --- 着順を追加 ---
    arrival = results.get("arrival_order", [])
    entry_order = {int(val): i+1 for i, val in enumerate(arrival[:3])}
    df_merged["rank"] = df_merged["entry_no"].map(entry_order)

    return df_merged.sort_values("entry_no").reset_index(drop=True)
