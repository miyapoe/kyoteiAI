# scraper.py
import requests
import pandas as pd

def scrape_race_json(date: str, stadium_no: int, race_no: int) -> pd.DataFrame:
    yyyy = date[:4]
    url = f"https://boatraceopenapi.github.io/programs/v2/{yyyy}/{date}.json"

    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    programs = data.get("programs", [])
    if not programs:
        return pd.DataFrame()

    # ★ ここが重要：配列から条件一致を探す
    race = next(
        (
            p for p in programs
            if p.get("race_stadium_number") == stadium_no
            and p.get("race_number") == race_no
        ),
        None
    )

    if race is None:
        return pd.DataFrame()

    boats = race.get("boats", [])
    if not boats:
        return pd.DataFrame()

    df = pd.DataFrame(boats)

    # よく使う列だけ整理（全部残してもOK）
    keep = [
        "racer_boat_number",
        "racer_name",
        "racer_number",
        "racer_age",
        "racer_weight",
        "racer_average_start_timing",
        "racer_national_top_1_percent",
        "racer_national_top_2_percent",
        "racer_national_top_3_percent",
        "racer_assigned_motor_number",
        "racer_assigned_boat_number",
    ]
    return df[[c for c in keep if c in df.columns]]
