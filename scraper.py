# scraper.py
import requests

def fetch_race_json(race_date: str, stadium: int, race_no: int):
    url = f"https://boatraceopenapi.github.io/programs/v1/{race_date}.json"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()

    races = data["programs"]
    for r in races:
        if r["race_stadium_number"] == stadium and r["race_number"] == race_no:
            return r

    return None
