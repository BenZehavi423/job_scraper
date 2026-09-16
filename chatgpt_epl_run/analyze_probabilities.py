from __future__ import annotations

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

OUT = Path("chatgpt_epl_run/results")
PRED_FILE = OUT / "walk_forward_predictions.csv"
MODEL_COL = "lambda__attack_defense_ewm"
MARKETS = {
    "O7.5": ("over", 7.5),
    "O8.5": ("over", 8.5),
    "U11.5": ("under", 11.5),
    "U12.5": ("under", 12.5),
}


def brier(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p-y)**2))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1-1e-12); y = np.asarray(y, float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def make_market_probs(df):
    mu = df[MODEL_COL].to_numpy(float)
    total = df["total_corners"].to_numpy(float)
    for name, (side, line) in MARKETS.items():
        k = int(np.floor(line))
        if side == "over":
            df[f"p_{name}"] = poisson.sf(k, mu)
            df[f"y_{name}"] = (total > line).astype(int)
        else:
            df[f"p_{name}"] = poisson.cdf(k, mu)
            df[f"y_{name}"] = (total < line).astype(int)
        df[f"fair_odds_{name}"] = 1.0 / np.clip(df[f"p_{name}"], 1e-9, 1)
    return df


def calibration_table(p, y, bins=np.linspace(0,1,11)):
    tmp = pd.DataFrame({"p":p,"y":y})
    tmp["bin"] = pd.cut(tmp.p, bins=bins, include_lowest=True, duplicates="drop")
    out = tmp.groupby("bin", observed=True).agg(n=("y","size"), mean_pred=("p","mean"), actual_rate=("y","mean")).reset_index()
    out["gap"] = out.mean_pred - out.actual_rate
    return out


def friday_week_start(d):
    d = pd.Timestamp(d)
    days_since_friday = (d.weekday() - 4) % 7
    return (d - pd.Timedelta(days=days_since_friday)).normalize()


def joint_rows(df, market, legs):
    rows=[]
    pcol=f"p_{market}"; ycol=f"y_{market}"
    for week, g in df.groupby("week_start"):
        if len(g) < legs:
            continue
        records = g[["Date","HomeTeam","AwayTeam",pcol,ycol]].to_dict("records")
        for combo in combinations(records, legs):
            p_joint=float(np.prod([x[pcol] for x in combo]))
            y_joint=int(np.prod([x[ycol] for x in combo]))
            rows.append({
                "market": market,
                "legs": legs,
                "week_start": week,
                "p_joint": p_joint,
                "y_joint": y_joint,
                "matches": " | ".join([f"{x['HomeTeam']}-{x['AwayTeam']}" for x in combo]),
            })
    return rows


def main():
    df=pd.read_csv(PRED_FILE, parse_dates=["Date"])
    df=make_market_probs(df)
    df["week_start"]=df.Date.map(friday_week_start)

    summary=[]; cal=[]
    for market in MARKETS:
        p=df[f"p_{market}"].to_numpy(float); y=df[f"y_{market}"].to_numpy(float)
        summary.append({
            "market":market,"n":len(df),"mean_pred":p.mean(),"actual_rate":y.mean(),
            "calibration_gap":p.mean()-y.mean(),"brier":brier(p,y),"log_loss":logloss(p,y),
            "mean_fair_odds":np.mean(1/np.clip(p,1e-9,1)),
            "median_fair_odds":np.median(1/np.clip(p,1e-9,1)),
        })
        c=calibration_table(p,y); c.insert(0,"market",market); cal.append(c)
    pd.DataFrame(summary).to_csv(OUT/"target_market_probability_summary.csv",index=False)
    pd.concat(cal,ignore_index=True).to_csv(OUT/"target_market_calibration_bins.csv",index=False)

    joint=[]
    for market in MARKETS:
        for legs in (2,3):
            joint.extend(joint_rows(df,market,legs))
    j=pd.DataFrame(joint)
    j.to_csv(OUT/"joint_probability_observations.csv",index=False)
    js=[]
    for (market,legs),g in j.groupby(["market","legs"]):
        js.append({
            "market":market,"legs":legs,"n_combinations":len(g),
            "mean_pred_joint":g.p_joint.mean(),"actual_joint_rate":g.y_joint.mean(),
            "calibration_gap":g.p_joint.mean()-g.y_joint.mean(),
            "brier":brier(g.p_joint,g.y_joint),"log_loss":logloss(g.p_joint,g.y_joint),
        })
    pd.DataFrame(js).to_csv(OUT/"joint_probability_summary.csv",index=False)

    cols=["Date","season","HomeTeam","AwayTeam","total_corners",MODEL_COL]
    for m in MARKETS:
        cols += [f"p_{m}",f"fair_odds_{m}",f"y_{m}"]
    df[cols].to_csv(OUT/"target_market_predictions.csv",index=False)

    print("\n=== TARGET MARKET PROBABILITIES ===")
    print(pd.DataFrame(summary).round(5).to_string(index=False))
    print("\n=== JOINT PROBABILITY CHECK ===")
    print(pd.DataFrame(js).round(5).to_string(index=False))

if __name__=="__main__":
    main()
