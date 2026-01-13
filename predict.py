import pandas as pd
import itertools
import requests
from bs4 import BeautifulSoup

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def get_odds(jcd, rno, hd):
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd}&hd={hd}"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    odds = {}
    rows = soup.select("table.odds3t1 tbody tr")
    for row in rows:
        tds = row.find_all("td")
        for i in range(0, len(tds), 3):
            combo = tds[i].get_text(strip=True).replace("－","-")
            odd = safe_float(tds[i+1].get_text(strip=True))
            if combo and odd:
                odds[combo] = odd
    return odds

def predict_trifecta(df, m1, m2, m3, top_n):
    X = df.drop(columns=["選手名"], errors="ignore")
    pred1 = m1.predict(X)
    pred2 = m2.predict(X)
    pred3 = m3.predict(X)
    results = []
    for i,j,k in itertools.permutations(range(len(df)), 3):
        score = pred1[i][0] * pred2[j][1] * pred3[k][2]
        combo = f"{int(df.iloc[i]['枠番'])}-{int(df.iloc[j]['枠番'])}-{int(df.iloc[k]['枠番'])}"
        results.append({"三連単": combo, "score": score})
    odds = get_odds(df.iloc[0]["場コード"], df.iloc[0]["レース番号"], df.iloc[0]["日付"])
    df_out = pd.DataFrame(results)
    df_out["odds"] = df_out["三連単"].map(odds)
    df_out["期待値"] = df_out["score"] * df_out["odds"]
    df_out = df_out.dropna().sort_values("期待値", ascending=False)
    return df_out.head(top_n)