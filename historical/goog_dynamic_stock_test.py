from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_probability_sizing import drawdown
from kalshi_imbalance_hitting_time import edges_from_train, bucket, conditional_prob


OUT = Path("goog_dynamic_stock_test")
OUT.mkdir(exist_ok=True)


def load_goog():
    parts = []
    for p in sorted(Path("upload").glob("goog_2026-*.npz")):
        z = np.load(p)
        df = pd.DataFrame(
            {
                "date": p.stem.replace("goog_", ""),
                "ts": z["ts"],
                "bid": z["bb"].astype(float),
                "ask": z["ba"].astype(float),
                "bq": z["bq"].astype(float),
                "aq": z["aq"].astype(float),
            }
        )
        df = df[(df.ask > df.bid) & (df.bid > 0) & (df.ask > 0)]
        # Collapse duplicate top-of-book states. Keeps stream shape, removes quote spam repeats.
        changed = (
            (df.bid != df.bid.shift())
            | (df.ask != df.ask.shift())
            | (df.bq != df.bq.shift())
            | (df.aq != df.aq.shift())
        )
        df = df[changed].copy()
        df = df.iloc[::20].copy()
        parts.append(df)
    d = pd.concat(parts, ignore_index=True)
    d["mid"] = (d.bid + d.ask) / 2
    d["spread"] = d.ask - d.bid
    d["imb1"] = (d.bq - d.aq) / np.maximum(d.bq + d.aq, 1e-12)
    d["dt_ms"] = d.groupby("date").ts.diff().fillna(0)
    return d.reset_index(drop=True)


def build_events(df, per_share_fee=0.0035, min_fee=0.0, max_wait_updates=1000, anchor_stride=250):
    rows = []
    i = 0
    n = len(df)
    dates = df.date.to_numpy()
    while i < n - 1:
        bid0 = float(df.bid.iat[i])
        ask0 = float(df.ask.iat[i])
        if not np.isfinite(bid0 + ask0) or ask0 <= bid0:
            i += 1
            continue
        j = i + 1
        label = None
        end = min(n, i + max_wait_updates)
        while j < end and dates[j] == dates[i]:
            long_pnl = float(df.bid.iat[j]) - ask0 - 2 * per_share_fee
            short_pnl = bid0 - float(df.ask.iat[j]) - 2 * per_share_fee
            if long_pnl > 0 or short_pnl > 0:
                label = 1 if long_pnl >= short_pnl else 0
                break
            j += 1
        if label is None:
            i += anchor_stride
            continue
        rows.append(
            {
                "date": dates[i],
                "anchor": i,
                "exit": j,
                "y": label,
                "bid": bid0,
                "ask": ask0,
                "exit_bid": float(df.bid.iat[j]),
                "exit_ask": float(df.ask.iat[j]),
                "imb1": float(df.imb1.iat[i]),
                "spread": float(df.spread.iat[i]),
                "wait_updates": j - i,
                "wait_ms": int(df.ts.iat[j] - df.ts.iat[i]),
            }
        )
        i = max(j, i + anchor_stride)
    return pd.DataFrame(rows)


def add_state(ev, edges):
    ev = ev.copy()
    ev["prev_state"] = ev.y.shift(1)
    ev = ev.dropna().reset_index(drop=True)
    ev.prev_state = ev.prev_state.astype(int)
    ev["imb_bucket"] = [bucket(v, edges) for v in ev.imb1]
    return ev


def stable_mask(df, ev, edges, lookback):
    out = []
    for r in ev.itertuples(index=False):
        a = max(0, int(r.anchor) - lookback + 1)
        vals = df.iloc[a : int(r.anchor) + 1].imb1
        b0 = bucket(r.imb1, edges)
        out.append(all(bucket(v, edges) == b0 for v in vals))
    return np.asarray(out, dtype=bool)


def simulate(test, probs, threshold, per_share_fee, sizing, capital=1000.0):
    cash = 0.0
    equity = [cash]
    trades = []
    for r, p_up in zip(test.itertuples(index=False), probs):
        if not np.isfinite(p_up) or abs(p_up - 0.5) <= threshold:
            equity.append(cash)
            continue
        side = "LONG" if p_up >= 0.5 else "SHORT"
        if side == "LONG":
            entry, exitp = float(r.ask), float(r.exit_bid)
        else:
            entry, exitp = float(r.bid), float(r.exit_ask)
        if sizing == "fixed_1":
            q = 1
        elif sizing == "fixed_100":
            q = 100
        else:
            q = max(1, int((0.25 * capital) // entry))
        pnl = q * ((exitp - entry) if side == "LONG" else (entry - exitp)) - 2 * per_share_fee * q
        cash += pnl
        equity.append(cash)
        trades.append(
            {
                "side": side,
                "q": q,
                "p": p_up,
                "pnl": pnl,
                "correct": int((side == "LONG" and r.y == 1) or (side == "SHORT" and r.y == 0)),
                "wait_ms": r.wait_ms,
            }
        )
    t = pd.DataFrame(trades)
    return {
        "net_pnl": cash,
        "max_drawdown": drawdown(equity),
        "trades": len(t),
        "shares": int(t.q.sum()) if len(t) else 0,
        "accuracy": float(t.correct.mean()) if len(t) else np.nan,
        "avg_wait_ms": float(t.wait_ms.mean()) if len(t) else np.nan,
    }


def main():
    df = load_goog()
    rows = []
    audits = []
    fee_models = {
        "ibkr_tiered_0.0035": 0.0035,
    }
    for fee_name, fee in fee_models.items():
        ev = build_events(df, per_share_fee=fee)
        if len(ev) < 50:
            continue
        edges = edges_from_train(ev.imb1.iloc[: int(0.6 * len(ev))], 5)
        ev = add_state(ev, edges)
        n = len(ev)
        test_start = int(0.8 * n)
        test = ev.iloc[test_start:].copy()
        audits.append(
            {
                "fee_model": fee_name,
                "book_rows": len(df),
                "events": len(ev),
                "test_events": len(test),
                "median_spread": float(df.spread.median()),
                "median_wait_ms": float(test.wait_ms.median()),
                "test_up_rate": float(test.y.mean()),
            }
        )
        for mode in ["base", "prev", "imb", "prev_imb"]:
            for window in [50, 100, 250, 500]:
                probs = []
                for idx in range(test_start, n):
                    hist = ev.iloc[max(0, idx - window) : idx]
                    p, _, _ = conditional_prob(hist, ev.prev_state.iat[idx], ev.imb_bucket.iat[idx], mode)
                    probs.append(p)
                probs = np.asarray(probs)
                for filt in ["none", "stable3", "stable5", "stable10"]:
                    keep = np.ones(len(test), dtype=bool) if filt == "none" else stable_mask(df, test, edges, int(filt.replace("stable", "")))
                    p_use = np.where(keep, probs, np.nan)
                    for threshold in [0.0, 0.03, 0.05, 0.08, 0.12]:
                        for sizing in ["fixed_1", "fixed_100", "quarter_capital"]:
                            res = simulate(test, p_use, threshold, fee, sizing)
                            rows.append(
                                {
                                    "fee_model": fee_name,
                                    "mode": mode,
                                    "window": window,
                                    "filter": filt,
                                    "threshold": threshold,
                                    "sizing": sizing,
                                    "eligible_rate": float(keep.mean()),
                                    "events": len(ev),
                                    "test_events": len(test),
                                    **res,
                                }
                            )
    summary = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    summary.to_csv(OUT / "goog_dynamic_summary.csv", index=False)
    audit.to_csv(OUT / "goog_audit.csv", index=False)
    print("AUDIT")
    print(audit.to_string(index=False))
    print("\nBEST")
    best = summary[summary.trades > 0].sort_values(["fee_model", "net_pnl"], ascending=[True, False]).groupby("fee_model").head(10)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
