from pathlib import Path

import numpy as np
import pandas as pd

from goog_dynamic_stock_test import load_goog
from goog_event_study import FEE_PER_SHARE, HORIZONS_MS, add_setup_features, future_index
from kalshi_probability_sizing import drawdown


OUT = Path("goog_rolling_fp_state")
OUT.mkdir(exist_ok=True)


def _day_edges(x, bins=5):
    vals = x[np.isfinite(x)]
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _bucket(edges, value):
    return int(np.searchsorted(edges, value, side="right") - 1)


def first_passage_label(g, i, horizon_ms):
    ts = g.ts.to_numpy()
    end = np.searchsorted(ts, ts[i] + horizon_ms, side="right")
    if end <= i + 1:
        return None

    ask0 = float(g.ask.iat[i])
    bid0 = float(g.bid.iat[i])
    long_hit = None
    short_hit = None
    long_pnl = None
    short_pnl = None

    for j in range(i + 1, end):
        lp = float(g.bid.iat[j]) - ask0 - 2 * FEE_PER_SHARE
        sp = bid0 - float(g.ask.iat[j]) - 2 * FEE_PER_SHARE
        if long_hit is None and lp > 0:
            long_hit = j
            long_pnl = lp
        if short_hit is None and sp > 0:
            short_hit = j
            short_pnl = sp
        if long_hit is not None and short_hit is not None:
            break

    fixed_j = future_index(ts, i, horizon_ms, len(g))
    if fixed_j is None:
        return None

    fixed_long_pnl = float(g.bid.iat[fixed_j]) - ask0 - 2 * FEE_PER_SHARE
    fixed_short_pnl = bid0 - float(g.ask.iat[fixed_j]) - 2 * FEE_PER_SHARE

    if long_hit is None and short_hit is None:
        y = 0
        hit = np.nan
        hit_pnl = 0.0
    elif short_hit is None or (long_hit is not None and long_hit <= short_hit):
        y = 1
        hit = long_hit
        hit_pnl = long_pnl
    else:
        y = -1
        hit = short_hit
        hit_pnl = short_pnl

    fixed_y = 0
    if fixed_long_pnl > 0 or fixed_short_pnl > 0:
        fixed_y = 1 if fixed_long_pnl >= fixed_short_pnl else -1

    return {
        "y": y,
        "hit": hit,
        "hit_pnl": hit_pnl,
        "fixed_y": fixed_y,
        "fixed_j": fixed_j,
        "fixed_long_pnl": fixed_long_pnl,
        "fixed_short_pnl": fixed_short_pnl,
    }


def build_anchor_events(df):
    rows = []
    for date, g0 in df.groupby("date", sort=False):
        g = g0.reset_index(drop=False).rename(columns={"index": "orig_index"})
        edges = _day_edges(g.imb1.to_numpy())
        setups = {
            "extreme_tight": (g.abs_imb >= 0.8) & (g.spread_rank <= 0.25),
            "stable5": (g.abs_imb >= 0.8) & (g.spread_rank <= 0.25) & (g.stable_bucket_5 == 1),
            "stable10": (g.abs_imb >= 0.8) & (g.spread_rank <= 0.25) & (g.stable_bucket_10 == 1),
        }
        eligible = np.arange(0, len(g), 5)
        for setup, mask in setups.items():
            idxs = [int(i) for i in eligible if bool(mask.iat[int(i)])]
            for horizon in HORIZONS_MS:
                for i in idxs:
                    lab = first_passage_label(g, i, horizon)
                    if lab is None:
                        continue
                    rows.append(
                        {
                            "date": date,
                            "orig_index": int(g.orig_index.iat[i]),
                            "local_index": i,
                            "setup": setup,
                            "horizon_ms": horizon,
                            "imb1": float(g.imb1.iat[i]),
                            "imb_sign": int(np.sign(g.imb1.iat[i])),
                            "imb_bucket": _bucket(edges, float(g.imb1.iat[i])),
                            "stable5": int(g.stable_bucket_5.iat[i]),
                            "stable10": int(g.stable_bucket_10.iat[i]),
                            "bid0": float(g.bid.iat[i]),
                            "ask0": float(g.ask.iat[i]),
                            "exit_bid": float(g.bid.iat[lab["fixed_j"]]),
                            "exit_ask": float(g.ask.iat[lab["fixed_j"]]),
                            "ts": int(g.ts.iat[i]),
                            **lab,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["date", "ts", "setup", "horizon_ms"]).reset_index(drop=True)


def pnl_for_side(r, side):
    if r.y == side and np.isfinite(r.hit):
        return 100 * float(r.hit_pnl)
    if side == 1:
        return 100 * (float(r.exit_bid) - float(r.ask0) - 2 * FEE_PER_SHARE)
    return 100 * (float(r.bid0) - float(r.exit_ask) - 2 * FEE_PER_SHARE)


def fixed_pnl_for_side(r, side):
    return 100 * (float(r.fixed_long_pnl) if side == 1 else float(r.fixed_short_pnl))


def key_for_mode(r, mode):
    if mode == "base":
        return ("all",)
    if mode == "sign":
        return (int(r.imb_sign),)
    if mode == "bucket":
        return (int(r.imb_bucket),)
    if mode == "sign_prev":
        return (int(r.imb_sign), int(r.prev_y))
    if mode == "bucket_prev":
        return (int(r.imb_bucket), int(r.prev_y))
    if mode == "bucket_sign_prev":
        return (int(r.imb_bucket), int(r.imb_sign), int(r.prev_y))
    raise ValueError(mode)


def probs_from_counts(counts, key):
    c = counts.get(key)
    if c is None or c.sum() < 5:
        c = counts.get(("all",), np.zeros(3))
    n = c.sum()
    # order: -1, 0, +1
    return (c + 0.5) / (n + 1.5)


def update_counts(counts, key, y):
    if key not in counts:
        counts[key] = np.zeros(3, dtype=float)
    idx = {-1: 0, 0: 1, 1: 2}[int(y)]
    counts[key][idx] += 1


def rolling_eval(events, warmup=30):
    rows = []
    trades = []
    modes = ["base", "sign", "bucket", "sign_prev", "bucket_prev", "bucket_sign_prev"]
    for (setup, horizon), ev0 in events.groupby(["setup", "horizon_ms"], sort=False):
        ev = ev0.sort_values(["date", "ts"]).reset_index(drop=True).copy()
        ev["prev_y"] = ev.y.shift(1).fillna(0).astype(int)
        if len(ev) <= warmup:
            continue
        for mode in modes:
            counts = {}
            for r in ev.iloc[:warmup].itertuples(index=False):
                update_counts(counts, ("all",), r.y)
                update_counts(counts, key_for_mode(r, mode), r.y)

            strat_pnl = []
            contra_pnl = []
            strat_fixed_pnl = []
            contra_fixed_pnl = []
            agree = []
            correct = []
            contra_correct = []
            fixed_fp_side_agree = []
            for r in ev.iloc[warmup:].itertuples(index=False):
                key = key_for_mode(r, mode)
                pminus, pzero, pplus = probs_from_counts(counts, key)
                side = 1 if pplus >= pminus else -1
                contra_side = -int(r.imb_sign) if int(r.imb_sign) != 0 else side
                sp = pnl_for_side(r, side)
                cp = pnl_for_side(r, contra_side)
                sfp = fixed_pnl_for_side(r, side)
                cfp = fixed_pnl_for_side(r, contra_side)
                strat_pnl.append(sp)
                contra_pnl.append(cp)
                strat_fixed_pnl.append(sfp)
                contra_fixed_pnl.append(cfp)
                agree.append(int(side == contra_side))
                correct.append(int(r.y == side))
                contra_correct.append(int(r.y == contra_side))
                fixed_fp_side_agree.append(int(r.fixed_y == r.y) if r.fixed_y != 0 and r.y != 0 else np.nan)
                trades.append(
                    {
                        "setup": setup,
                        "horizon_ms": horizon,
                        "mode": mode,
                        "date": r.date,
                        "ts": r.ts,
                        "imb1": r.imb1,
                        "y": r.y,
                        "fixed_y": r.fixed_y,
                        "pminus": pminus,
                        "pzero": pzero,
                        "pplus": pplus,
                        "stoch_side": side,
                        "contra_side": contra_side,
                        "stoch_pnl100": sp,
                        "contra_pnl100": cp,
                        "stoch_fixed_pnl100": sfp,
                        "contra_fixed_pnl100": cfp,
                    }
                )
                update_counts(counts, ("all",), r.y)
                update_counts(counts, key, r.y)

            strat_eq = np.r_[0, np.cumsum(strat_pnl)]
            contra_eq = np.r_[0, np.cumsum(contra_pnl)]
            rows.append(
                {
                    "setup": setup,
                    "horizon_ms": horizon,
                    "mode": mode,
                    "events": len(ev),
                    "warmup": warmup,
                    "trades": len(strat_pnl),
                    "stoch_pnl100": float(np.sum(strat_pnl)),
                    "contra_pnl100_same_anchors": float(np.sum(contra_pnl)),
                    "stoch_fixed_pnl100_same_anchors": float(np.sum(strat_fixed_pnl)),
                    "contra_fixed_pnl100_same_anchors": float(np.sum(contra_fixed_pnl)),
                    "stoch_mean100": float(np.mean(strat_pnl)),
                    "contra_mean100": float(np.mean(contra_pnl)),
                    "stoch_median100": float(np.median(strat_pnl)),
                    "contra_median100": float(np.median(contra_pnl)),
                    "stoch_maxdd100": drawdown(strat_eq),
                    "contra_maxdd100": drawdown(contra_eq),
                    "stoch_accuracy": float(np.mean(correct)),
                    "contra_accuracy": float(np.mean(contra_correct)),
                    "stoch_contra_side_agreement": float(np.mean(agree)),
                    "fixed_fp_side_agreement": float(pd.Series(fixed_fp_side_agree).dropna().mean())
                    if pd.Series(fixed_fp_side_agree).dropna().size
                    else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(trades)


def main():
    df = add_setup_features(load_goog())
    events = build_anchor_events(df)
    events.to_csv(OUT / "first_passage_anchor_events.csv", index=False)
    summary, trades = rolling_eval(events, warmup=30)
    summary = summary.sort_values(["setup", "horizon_ms", "stoch_pnl100"], ascending=[True, True, False])
    summary.to_csv(OUT / "rolling_fp_state_summary.csv", index=False)
    trades.to_csv(OUT / "rolling_fp_state_trades.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
