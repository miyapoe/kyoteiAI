import requests
from bs4 import BeautifulSoup
import pandas as pd

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def scrape_race_data(jcd, rno, hd):
    # 出走表ページ
    race_url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={hd}"
    res = requests.get(race_url)
    soup = BeautifulSoup(res.text, "html.parser")
    rows = soup.select("table.is-fs12 tbody tr")

    data = []
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) == 0:
            continue
        data.append({
            "枠番": safe_float(cols[0]),
            "選手名": cols[1],
            "体重": safe_float(cols[5]),
            "F数": safe_float(cols[6]),
            "L数": safe_float(cols[7]),
            "平均ST": safe_float(cols[8]),
            "勝率": safe_float(cols[9]),
            "2連率": safe_float(cols[10]),
            "3連率": safe_float(cols[11])
        })

    df = pd.DataFrame(data)

    # 直前情報
    info_url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?hd={hd}&jcd={jcd}&rno={rno}"
    res2 = requests.get(info_url)
    soup2 = BeautifulSoup(res2.text, "html.parser")
    info_rows = soup2.select("table tbody tr")
    ex_data = []
    for row in info_rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) > 0:
            ex_data.append({
                "展示タイム": safe_float(cols[1]),
                "展示ST": safe_float(cols[2]),
                "チルト": safe_float(cols[3])
            })
    if len(ex_data) == len(df):
        for i, row in enumerate(ex_data):
            df.at[i, "展示タイム"] = row["展示タイム"]
            df.at[i, "展示ST"] = row["展示ST"]
            df.at[i, "チルト"] = row["チルト"]

    return df