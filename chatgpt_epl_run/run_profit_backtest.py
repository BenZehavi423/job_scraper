from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from betting_engine import backtest_singles, devig_two_way, expected_value

RESULTS = Path("chatgpt_epl_run/results")
ODDS = Path("chatgpt_epl_run/odds/epl_corner_odds_history.csv")
PRED = RESULTS / "target_market_predictions.csv"
OUT = Path("chatgpt_epl_run/profit")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    ("Over", 7.5): "O7.5",
    ("Over", 8.5): "O8.5",
    ("Under", 11.5): "U11.5",
    ("Under", 12.5): "U12.5",
}


def normalize_side(x: str) -> str:
    s = str(x).strip().lower()
    if "over" in s:
        return "Over"
    if "under" in s:
        return "Under"
    return str(x)


def canonical_team(s: str) -> str:
    s = str(s).lower().replace(" fc", "").replace("afc ", "").replace(" & ", " and ")
    for a,b in [("manchester united","man united"),("manchester city","man city"),("nottingham forest","nott'm forest"),("tottenham hotspur","tottenham"),("brighton and hove albion","brighton"),("wolverhampton wanderers","wolves"),("newcastle united","newcastle"),("west ham united","west ham")]:
        s=s.replace(a,b)
    return " ".join(s.split())


def closing_quotes(odds: pd.DataFrame, minutes_before: int = 5) -> pd.DataFrame:
    o=odds.copy()
    o["recorded_at"]=pd.to_datetime(o.recorded_at, utc=True, errors="coerce")
    o["commence_time"]=pd.to_datetime(o.commence_time, utc=True, errors="coerce")
    o["cutoff"]=o.commence_time-pd.Timedelta(minutes=minutes_before)
    o=o[o.recorded_at<=o.cutoff].copy()
    o["side_norm"]=o.side.map(normalize_side)
    o=o.sort_values("recorded_at").groupby(["fixture_id","bookmaker","line","side_norm"],as_index=False).tail(1)
    return o


def pair_devig(quotes: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (fid,book,line),g in quotes.groupby(["fixture_id","bookmaker","line"]):
        by={r.side_norm:r for _,r in g.iterrows()}
        if "Over" not in by or "Under" not in by:
            continue
        po,pu=devig_two_way(by["Over"].decimal_odds,by["Under"].decimal_odds)
        for side,pfair in [("Over",po),("Under",pu)]:
            r=by[side]
            rows.append({
                "fixture_id":fid,"bookmaker":book,"line":float(line),"side":side,
                "odds":float(r.decimal_odds),"market_fair_probability":pfair,
                "recorded_at":r.recorded_at,"commence_time":r.commence_time,
                "home_team_odds":r.home_team,"away_team_odds":r.away_team,
            })
    return pd.DataFrame(rows)


def join_predictions(quotes: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    p=pred.copy()
    p["Date"]=pd.to_datetime(p.Date)
    p["home_key"]=p.HomeTeam.map(canonical_team)
    p["away_key"]=p.AwayTeam.map(canonical_team)
    q=quotes.copy()
    q["home_key"]=q.home_team_odds.map(canonical_team)
    q["away_key"]=q.away_team_odds.map(canonical_team)
    q["date_key"]=pd.to_datetime(q.commence_time,utc=True).dt.tz_convert(None).dt.date
    p["date_key"]=p.Date.dt.date
    m=q.merge(p,on=["home_key","away_key","date_key"],how="inner")
    rows=[]
    for _,r in m.iterrows():
        market=TARGETS.get((r.side,float(r.line)))
        if not market:
            continue
        prob=float(r[f"p_{market}"])
        won=bool(r[f"y_{market}"])
        rows.append({
            "date":r.commence_time,"fixture_id":r.fixture_id,"bookmaker":r.bookmaker,
            "home_team":r.HomeTeam,"away_team":r.AwayTeam,"market":market,
            "line":float(r.line),"side":r.side,"probability":prob,"odds":float(r.odds),
            "market_fair_probability":float(r.market_fair_probability),
            "model_edge_probability":prob-float(r.market_fair_probability),
            "model_ev":expected_value(prob,float(r.odds)),"won":won,
        })
    return pd.DataFrame(rows)


def choose_best_price(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    return candidates.sort_values("odds").groupby(["date","home_team","away_team","market"],as_index=False).tail(1)


def main():
    odds=pd.read_csv(ODDS)
    pred=pd.read_csv(PRED)
    quotes=pair_devig(closing_quotes(odds, minutes_before=5))
    bets=choose_best_price(join_predictions(quotes,pred)).sort_values("date")
    bets.to_csv(OUT/"candidate_bets.csv",index=False)

    summaries=[]
    for min_ev in [0.00,0.02,0.05,0.08,0.10]:
        for scale in [0.25,0.5]:
            hist,metrics=backtest_singles(
                bets,starting_bankroll=10000,min_ev=min_ev,kelly_scale=scale,max_bet_fraction=0.02
            )
            metrics.update({"min_ev":min_ev,"kelly_scale":scale})
            summaries.append(metrics)
            hist.to_csv(OUT/f"history_ev{int(min_ev*100):02d}_kelly{int(scale*100):02d}.csv",index=False)
    pd.DataFrame(summaries).to_csv(OUT/"profitability_summary.csv",index=False)
    print(pd.DataFrame(summaries).round(4).to_string(index=False))

if __name__=="__main__":
    main()
