from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

SEASONS = {
    "2022_23": "2223",
    "2023_24": "2324",
    "2024_25": "2425",
    "2025_26": "2526",
}
BASE = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/premier-league/season-{}.csv"
LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
OUT = Path("chatgpt_epl_run/results")
OUT.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    frames = []
    for season, code in SEASONS.items():
        df = pd.read_csv(BASE.format(code))
        required = {"Date", "HomeTeam", "AwayTeam", "HC", "AC"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{season}: missing {missing}")
        df = df[["Date", "HomeTeam", "AwayTeam", "HC", "AC"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["season"] = season
        df["HC"] = pd.to_numeric(df["HC"], errors="coerce")
        df["AC"] = pd.to_numeric(df["AC"], errors="coerce")
        df = df.dropna(subset=["HC", "AC"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True)
    out["match_id"] = np.arange(len(out))
    out["total_corners"] = out["HC"] + out["AC"]
    return out


def league_priors_no_same_day_leak(matches: pd.DataFrame) -> pd.DataFrame:
    """Expanding league means using only dates strictly earlier than current date."""
    df = matches.copy().sort_values(["Date", "match_id"], kind="stable")
    daily = df.groupby("Date", sort=True).agg(hc_sum=("HC", "sum"), ac_sum=("AC", "sum"), n=("HC", "size"))
    daily["prior_n"] = daily["n"].cumsum().shift(1).fillna(0)
    daily["prior_hc"] = daily["hc_sum"].cumsum().shift(1).fillna(0)
    daily["prior_ac"] = daily["ac_sum"].cumsum().shift(1).fillna(0)
    # Neutral, predeclared seed used only before any observed EPL match in our dataset.
    daily["league_home_corners_prior"] = np.where(daily["prior_n"] > 0, daily["prior_hc"] / daily["prior_n"], 5.0)
    daily["league_away_corners_prior"] = np.where(daily["prior_n"] > 0, daily["prior_ac"] / daily["prior_n"], 5.0)
    pri = daily[["league_home_corners_prior", "league_away_corners_prior"]].reset_index()
    df = df.merge(pri, on="Date", how="left")
    df["league_total_corners_prior"] = df["league_home_corners_prior"] + df["league_away_corners_prior"]
    return df


def to_team_match(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["match_id", "Date", "season", "HomeTeam", "AwayTeam", "HC", "AC"]].copy()
    home.columns = ["match_id", "Date", "season", "team", "opponent", "corners_for", "corners_against"]
    home["is_home"] = 1
    away = matches[["match_id", "Date", "season", "AwayTeam", "HomeTeam", "AC", "HC"]].copy()
    away.columns = ["match_id", "Date", "season", "team", "opponent", "corners_for", "corners_against"]
    away["is_home"] = 0
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "Date", "match_id"], kind="stable")


def build_features(matches: pd.DataFrame, ewm_halflife: float = 5.0, shrink_k: float = 6.0) -> pd.DataFrame:
    df = league_priors_no_same_day_leak(matches)
    tm = to_team_match(df)
    g = tm.groupby("team", group_keys=False, sort=False)
    for n in (5, 10):
        tm[f"corners_for_roll{n}"] = g["corners_for"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        tm[f"corners_against_roll{n}"] = g["corners_against"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    tm["corners_for_ewm"] = g["corners_for"].transform(lambda s: s.shift(1).ewm(halflife=ewm_halflife, adjust=False, min_periods=1).mean())
    tm["corners_against_ewm"] = g["corners_against"].transform(lambda s: s.shift(1).ewm(halflife=ewm_halflife, adjust=False, min_periods=1).mean())
    tm["prior_matches"] = g.cumcount()

    cols = ["match_id", "team", "prior_matches", "corners_for_roll5", "corners_against_roll5", "corners_for_roll10", "corners_against_roll10", "corners_for_ewm", "corners_against_ewm"]
    home = tm.loc[tm.is_home.eq(1), cols].copy().add_prefix("home_")
    away = tm.loc[tm.is_home.eq(0), cols].copy().add_prefix("away_")
    out = df.merge(home, left_on="match_id", right_on="home_match_id", how="left").merge(away, left_on="match_id", right_on="away_match_id", how="left")
    out = out.drop(columns=["home_match_id", "away_match_id"])

    # Shrink team estimates to point-in-time league environment; promoted/new teams start at league prior.
    for side in ("home", "away"):
        n = out[f"{side}_prior_matches"].fillna(0).astype(float)
        w = n / (n + shrink_k)
        scoring_prior = out["league_home_corners_prior"] if side == "home" else out["league_away_corners_prior"]
        conceding_prior = out["league_away_corners_prior"] if side == "home" else out["league_home_corners_prior"]
        for tag in ("roll5", "roll10", "ewm"):
            cf = f"{side}_corners_for_{tag}"
            ca = f"{side}_corners_against_{tag}"
            out[cf + "_shrunk"] = w * out[cf].fillna(scoring_prior) + (1-w) * scoring_prior
            out[ca + "_shrunk"] = w * out[ca].fillna(conceding_prior) + (1-w) * conceding_prior
    return out


def attack_defense_lambda(df: pd.DataFrame, tag: str) -> np.ndarray:
    hf = df[f"home_corners_for_{tag}_shrunk"].to_numpy(float)
    ha = df[f"home_corners_against_{tag}_shrunk"].to_numpy(float)
    af = df[f"away_corners_for_{tag}_shrunk"].to_numpy(float)
    aa = df[f"away_corners_against_{tag}_shrunk"].to_numpy(float)
    expected_home = 0.5 * (hf + aa)
    expected_away = 0.5 * (af + ha)
    return np.clip(expected_home + expected_away, 0.25, None)


def evaluate(y, mu):
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    summary = {
        "n": len(y),
        "mae_lambda": mean_absolute_error(y, mu),
        "poisson_deviance": mean_poisson_deviance(y, mu),
        "mean_actual": y.mean(),
        "mean_predicted": mu.mean(),
    }
    rows = []
    for line in LINES:
        p = poisson.sf(int(np.floor(line)), mu=mu)
        a = (y > line).astype(float)
        rows.append({"line": line, "actual_over_rate": a.mean(), "mean_predicted_over_probability": p.mean(), "calibration_gap": p.mean()-a.mean(), "brier": np.mean((p-a)**2)})
    ldf = pd.DataFrame(rows)
    summary["mean_brier_6_lines"] = ldf.brier.mean()
    summary["mean_abs_calibration_gap_6_lines"] = ldf.calibration_gap.abs().mean()
    return summary, ldf


FEATURES = {
    tag: [f"home_corners_for_{tag}_shrunk", f"home_corners_against_{tag}_shrunk", f"away_corners_for_{tag}_shrunk", f"away_corners_against_{tag}_shrunk"]
    for tag in ("roll5", "roll10", "ewm")
}


def main():
    matches = load_data()
    print(matches.groupby("season").size())
    assert matches.groupby("season").size().eq(380).all(), matches.groupby("season").size()
    features = build_features(matches)

    # Dataset/EDA audit
    eda = matches.groupby("season").agg(n=("total_corners", "size"), mean_total=("total_corners", "mean"), variance_total=("total_corners", "var"), mean_home=("HC", "mean"), mean_away=("AC", "mean")).reset_index()
    eda["var_mean_ratio"] = eda["variance_total"] / eda["mean_total"]
    eda.to_csv(OUT / "season_eda.csv", index=False)

    seasons = list(SEASONS)
    summary_rows, line_rows, pred_frames = [], [], []
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        train = features[features.season.isin(train_seasons)].copy()
        test = features[features.season.eq(test_season)].copy()
        y = test.total_corners.to_numpy(float)

        candidates = {}
        candidates["league_mean_expanding"] = test.league_total_corners_prior.to_numpy(float)
        candidates["train_mean_fixed"] = np.repeat(train.total_corners.mean(), len(test))
        for tag in ("roll5", "roll10", "ewm"):
            candidates[f"attack_defense_{tag}"] = attack_defense_lambda(test, tag)
            reg = PoissonRegressor(alpha=1.0, max_iter=2000)
            reg.fit(train[FEATURES[tag]].astype(float), train.total_corners.astype(float))
            candidates[f"poisson_reg_{tag}"] = np.clip(reg.predict(test[FEATURES[tag]].astype(float)), 0.25, None)

        pf = test[["Date", "season", "HomeTeam", "AwayTeam", "HC", "AC", "total_corners"]].copy()
        for name, mu in candidates.items():
            s, ldf = evaluate(y, mu)
            s.update({"model": name, "train_seasons": "+".join(train_seasons), "test_season": test_season})
            summary_rows.append(s)
            ldf.insert(0, "model", name)
            ldf.insert(1, "test_season", test_season)
            line_rows.append(ldf)
            pf[f"lambda__{name}"] = mu
        pred_frames.append(pf)

    summary = pd.DataFrame(summary_rows)
    lines = pd.concat(line_rows, ignore_index=True)
    preds = pd.concat(pred_frames, ignore_index=True)
    summary.to_csv(OUT / "walk_forward_model_summary.csv", index=False)
    lines.to_csv(OUT / "walk_forward_line_metrics.csv", index=False)
    preds.to_csv(OUT / "walk_forward_predictions.csv", index=False)

    # Pooled OOS metrics across all three test seasons (1140 matches), recomputed from predictions.
    pooled_rows, pooled_lines = [], []
    model_cols = [c for c in preds.columns if c.startswith("lambda__")]
    for col in model_cols:
        name = col.replace("lambda__", "")
        s, ldf = evaluate(preds.total_corners.to_numpy(float), preds[col].to_numpy(float))
        s["model"] = name
        pooled_rows.append(s)
        ldf.insert(0, "model", name)
        pooled_lines.append(ldf)
    pooled = pd.DataFrame(pooled_rows).sort_values(["mean_brier_6_lines", "poisson_deviance"])
    pooled_line = pd.concat(pooled_lines, ignore_index=True)
    pooled.to_csv(OUT / "pooled_oos_summary.csv", index=False)
    pooled_line.to_csv(OUT / "pooled_oos_line_metrics.csv", index=False)

    metadata = {
        "rows_total": int(len(matches)),
        "rows_by_season": {k: int(v) for k,v in matches.groupby("season").size().items()},
        "test_rows_total": int(len(preds)),
        "ewm_halflife": 5.0,
        "shrink_k": 6.0,
        "poisson_reg_alpha": 1.0,
        "lines": LINES,
        "same_day_league_leak_prevented": True,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\n=== SEASON EDA ===")
    print(eda.round(4).to_string(index=False))
    print("\n=== POOLED OOS (sorted by mean Brier) ===")
    print(pooled.round(5).to_string(index=False))
    print("\n=== BY FOLD ===")
    print(summary.sort_values(["test_season", "mean_brier_6_lines"]).round(5).to_string(index=False))

if __name__ == "__main__":
    main()
