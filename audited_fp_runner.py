"""Explicit adapters and CLI for the audited replay.

No old deduplication, 20x subsampling, random splits or horizon grid search.
Run unit tests before running an asset experiment. Proxy clocks require opt-in.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audited_fp import Config, Fees, History, Tape, replay


def l1(prices, quantities):
    return np.stack([prices, quantities], axis=-1)[:, None, :]


def load_goog(path):
    with np.load(path) as z:
        return [Tape(z["ts"].astype(np.int64) * 1_000_000,
                     l1(z["bb"], z["bq"]), l1(z["ba"], z["aq"]),
                     np.arange(len(z["ts"])), "GOOG", Path(path).stem,
                     "book_updates", metadata={"source": str(path),
                     "limitation": "No trade prints or full depth; not a 50-trade test"})]


def load_nse(path):
    with np.load(path) as z:
        idx = z["tr_bookidx"]
        n = len(z["ts"])
        if idx.dtype.kind not in "iu" or np.any(idx < 0) or np.any(idx >= n) or np.any(np.diff(idx) < 0):
            raise ValueError("Invalid trade-to-book mapping")
        counts = np.cumsum(np.bincount(idx, minlength=n))
        return [Tape(z["ts"].astype(np.int64) * 1_000_000,
                     z["bids"], z["asks"], counts, str(z["symbol"]), Path(path).stem,
                     "snapshot_linked_trades", metadata={"source": str(path),
                     "limitation": "Trade counts known only at snapshots; intra-snapshot ordering unknown"})]


def load_kalshi(path, asset):
    d = pd.read_csv(path)
    required = {"t", "bid", "ask", "bq", "aq", "segment", "qty", "sign"}
    if not required.issubset(d.columns) or d[list(required)].isna().any().any():
        raise ValueError("Missing required trade-message fields")
    if d.empty:
        return []
    segments = d.segment.to_numpy()
    starts = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1]
    ends = np.r_[starts[1:], len(d)]
    if len(np.unique(segments[starts])) != len(starts):
        raise ValueError("Noncontiguous segment IDs need explicit session reconstruction")
    result = []
    for start, end in zip(starts, ends):
        g = d.iloc[start:end]
        # No quote-state deduplication: equal books can accompany distinct trades.
        ts = np.rint(g.t.to_numpy() * 1e9).astype(np.int64)
        result.append(Tape(ts, l1(g.bid, g.bq), l1(g.ask, g.aq), np.arange(len(g)),
                           asset, str(segments[start]), "retained_trade_messages", True,
                           {"source": str(path), "limitation":
                            "Original parser omits invalid-book trade messages; only trade-time L1 snapshots, not full update stream"}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["goog", "nse", "kalshi"], required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--asset", default="KALSHI")
    parser.add_argument("--fee-model", choices=["zero", "per_share", "kalshi", "nse_intraday"], required=True)
    parser.add_argument("--fee-per-share", type=float)
    parser.add_argument("--fee-rounding", choices=["0.01", "0.0001"], default="0.01")
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--allow-proxy-clock", action="store_true")
    parser.add_argument("--skip-late-diagnostics", action="store_true",
                        help="Skip optional post-horizon scans; primary labels and PnL are unchanged")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.fee_model == "per_share" and args.fee_per_share is None:
        parser.error("--fee-per-share must be supplied for that scenario")
    if args.kind == "kalshi" and args.fee_model not in {"kalshi", "zero"}:
        parser.error("Incompatible fee model")
    if args.kind != "kalshi" and args.fee_model == "kalshi":
        parser.error("Incompatible fee model")
    fees = Fees(args.fee_model, per_share=args.fee_per_share or 0, rounding=args.fee_rounding)
    tapes = load_kalshi(args.path, args.asset) if args.kind == "kalshi" else (
        load_goog(args.path) if args.kind == "goog" else load_nse(args.path))
    outputs = []
    for mode in ("imbalance", "imbalance_state"):
        cfg = Config(q=args.q, mode=mode, allow_proxy_clock=args.allow_proxy_clock,
                     diagnostic_after_50=not args.skip_late_diagnostics)
        history = History()
        for tape in tapes:
            if history.halted:
                outputs.append(dict(summary=dict(asset=tape.asset, segment=tape.segment,
                                    mode=mode, status="blocked_by_unresolved_prior_position")))
                continue
            result = replay(tape, cfg, fees, history=history)
            outputs.append(dict(config=asdict(cfg), fees=asdict(fees), metadata=tape.metadata, **result))
    manifest = dict(source=str(args.path), sha256=hashlib.sha256(args.path.read_bytes()).hexdigest(),
                    engine_sha256=hashlib.sha256(Path(__file__).with_name("audited_fp.py").read_bytes()).hexdigest(),
                    results=outputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps([x["summary"] for x in outputs], indent=2))


if __name__ == "__main__":
    main()
