from __future__ import annotations

import numpy as np
import pandas as pd


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / float(decimal_odds)


def devig_two_way(over_odds: float, under_odds: float) -> tuple[float, float]:
    io = 1.0 / float(over_odds)
    iu = 1.0 / float(under_odds)
    s = io + iu
    return io / s, iu / s


def expected_value(prob: float, decimal_odds: float) -> float:
    return float(prob) * float(decimal_odds) - 1.0


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    p = float(prob)
    o = float(decimal_odds)
    b = o - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    return max(0.0, (b * p - q) / b)


def stake_from_kelly(bankroll: float, prob: float, decimal_odds: float,
                     fraction: float = 0.25, max_bankroll_fraction: float = 0.02) -> float:
    raw = kelly_fraction(prob, decimal_odds) * float(fraction)
    capped = min(raw, float(max_bankroll_fraction))
    return max(0.0, float(bankroll) * capped)


def settle_single(stake: float, decimal_odds: float, won: bool) -> float:
    return stake * (decimal_odds - 1.0) if bool(won) else -stake


def backtest_singles(bets: pd.DataFrame, starting_bankroll: float = 10000.0,
                     min_ev: float = 0.0, kelly_scale: float = 0.25,
                     max_bet_fraction: float = 0.02) -> tuple[pd.DataFrame, dict]:
    """Expected columns: date, probability, odds, won. Optional identifiers preserved."""
    df = bets.sort_values("date").copy()
    bankroll = float(starting_bankroll)
    history = []
    turnover = 0.0
    peak = bankroll
    max_dd = 0.0

    for _, r in df.iterrows():
        p = float(r.probability)
        o = float(r.odds)
        ev = expected_value(p, o)
        stake = 0.0
        pnl = 0.0
        placed = ev >= min_ev
        if placed:
            stake = stake_from_kelly(bankroll, p, o, kelly_scale, max_bet_fraction)
            if stake > 0:
                pnl = settle_single(stake, o, bool(r.won))
                bankroll += pnl
                turnover += stake
        peak = max(peak, bankroll)
        dd = 1.0 - bankroll / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        out = r.to_dict()
        out.update({"model_ev": ev, "stake": stake, "pnl": pnl, "bankroll": bankroll, "placed": placed})
        history.append(out)

    h = pd.DataFrame(history)
    metrics = {
        "starting_bankroll": float(starting_bankroll),
        "final_bankroll": bankroll,
        "profit": bankroll - float(starting_bankroll),
        "turnover": turnover,
        "roi_on_turnover": (bankroll - float(starting_bankroll)) / turnover if turnover else np.nan,
        "return_on_bankroll": bankroll / float(starting_bankroll) - 1.0,
        "max_drawdown": max_dd,
        "bets_placed": int((h.stake > 0).sum()) if len(h) else 0,
    }
    return h, metrics


def accumulator_probability(probs) -> float:
    """For distinct matches under an independence assumption."""
    return float(np.prod(np.asarray(list(probs), dtype=float)))


def accumulator_odds(odds) -> float:
    return float(np.prod(np.asarray(list(odds), dtype=float)))
