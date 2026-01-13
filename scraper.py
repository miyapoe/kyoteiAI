# scraper.py
import requests

def fetch_race_json(date: str, stadium: int):
    """
    date: YYYYMMDD
    stadium: 場コード（ここでは 1 固定）
    """
    url = f"https://boatraceopenapi.github.io/programs/v1/{date}.json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()
