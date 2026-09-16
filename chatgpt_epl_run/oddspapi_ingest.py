from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path
import requests
import pandas as pd

BASE_URL = "https://api.oddspapi.io/v4"
SPORT_ID = 10
EPL_TOURNAMENT_ID = 17
TARGET_LINES = {7.5, 8.5, 11.5, 12.5}
MARKET_NAME = "Corners - Over Under Full Time"
DEFAULT_BOOKMAKERS = "pinnacle,bet365,sbobet"
OUT = Path("chatgpt_epl_run/odds")
OUT.mkdir(parents=True, exist_ok=True)


def api_key() -> str:
    key = os.environ.get("ODDSPAPI_KEY")
    if not key:
        raise RuntimeError("Set ODDSPAPI_KEY in the environment/GitHub Actions secret.")
    return key


def get(path: str, **params):
    params = {**params, "apiKey": api_key()}
    r = requests.get(f"{BASE_URL}/{path}", params=params, timeout=180)
    r.raise_for_status()
    return r.json()


def iter_windows(start: date, end: date, days: int = 10):
    cur = start
    while cur <= end:
        hi = min(end, cur + timedelta(days=days-1))
        yield cur, hi
        cur = hi + timedelta(days=1)


def fetch_epl_fixtures(start: str, end: str) -> pd.DataFrame:
    s = date.fromisoformat(start); e = date.fromisoformat(end)
    rows = []
    for lo, hi in iter_windows(s, e):
        payload = get("fixtures", sportId=SPORT_ID, **{"from": lo.isoformat(), "to": hi.isoformat()})
        for f in payload:
            if int(f.get("tournamentId", -1)) == EPL_TOURNAMENT_ID:
                rows.append(f)
        time.sleep(0.1)
    return pd.DataFrame(rows).drop_duplicates(subset=["fixtureId"] if rows else None)


def market_catalog():
    payload = get("markets", sportId=SPORT_ID)
    target = []
    for m in payload:
        if m.get("marketName") != MARKET_NAME:
            continue
        try:
            line = float(m.get("handicap"))
        except (TypeError, ValueError):
            continue
        if line not in TARGET_LINES:
            continue
        target.append(m)
    return target


def outcome_name_map(catalog):
    mapping = {}
    for m in catalog:
        mid = str(m["marketId"])
        outcomes = m.get("outcomes") or []
        if isinstance(outcomes, dict):
            iterable = outcomes.items()
        else:
            iterable = [(str(o.get("outcomeId")), o) for o in outcomes]
        for oid, o in iterable:
            name = o.get("outcomeName") or o.get("name") or o.get("label")
            if name:
                mapping[(mid, str(oid))] = str(name)
    return mapping


def flatten_history(fixture, history, catalog, bookmakers):
    market_by_id = {str(m["marketId"]): m for m in catalog}
    outcome_names = outcome_name_map(catalog)
    rows = []
    books = history.get("bookmakers", {}) if isinstance(history, dict) else {}
    for book, bdata in books.items():
        markets = (bdata or {}).get("markets", {})
        for mid, mdata in markets.items():
            if str(mid) not in market_by_id:
                continue
            line = float(market_by_id[str(mid)]["handicap"])
            outcomes = (mdata or {}).get("outcomes", {})
            for oid, odata in outcomes.items():
                side = outcome_names.get((str(mid), str(oid))) or str(oid)
                players = (odata or {}).get("players", {})
                snaps = players.get("0", [])
                if isinstance(snaps, dict):
                    snaps = [snaps]
                for snap in snaps:
                    price = snap.get("price")
                    if price is None:
                        continue
                    rows.append({
                        "fixture_id": fixture.get("fixtureId"),
                        "commence_time": fixture.get("startTime"),
                        "home_team": fixture.get("participant1Name"),
                        "away_team": fixture.get("participant2Name"),
                        "bookmaker": book,
                        "market_id": mid,
                        "line": line,
                        "side": side,
                        "recorded_at": snap.get("createdAt") or snap.get("updatedAt"),
                        "decimal_odds": float(price),
                        "limit": snap.get("limit"),
                    })
    return rows


def download_history(start: str, end: str, bookmakers: str = DEFAULT_BOOKMAKERS):
    fixtures = fetch_epl_fixtures(start, end)
    catalog = market_catalog()
    fixtures.to_csv(OUT / "epl_fixtures.csv", index=False)
    pd.DataFrame(catalog).to_json(OUT / "corner_market_catalog.json", orient="records", indent=2)
    rows = []
    for _, f in fixtures.iterrows():
        if not f.get("hasOdds", True):
            continue
        hist = get("historical-odds", fixtureId=f["fixtureId"], bookmakers=bookmakers)
        rows.extend(flatten_history(f.to_dict(), hist, catalog, bookmakers))
        # historical endpoint is free but still rate-limited; keep requests gentle.
        time.sleep(4.7)
    odds = pd.DataFrame(rows)
    odds.to_csv(OUT / "epl_corner_odds_history.csv", index=False)
    return odds


if __name__ == "__main__":
    # Full historical corner coverage availability depends on when each bookmaker began carrying the market.
    # Start at May 2023 so it aligns with our first OOS season and expand as coverage allows.
    download_history("2023-05-01", "2026-05-31")
