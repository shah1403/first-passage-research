"""Causal empirical first-passage replay. No asset experiment runs on import.

Prices/fees are in the asset's native currency; timestamps are integer ns.
The target is a fee-covering *observable exit opportunity*, not a guarantee
that an order sent at that observation will fill profitably after latency.
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
import heapq
from typing import Optional

import numpy as np
import pandas as pd


EPS = 1e-9


@dataclass(frozen=True)
class Fees:
    kind: str
    per_share: float = 0.0
    multiplier: float = 1.0
    rounding: str = "0.01"
    provenance: str = "explicit research scenario, not verified account charges"

    def __post_init__(self):
        if self.kind not in {"zero", "per_share", "kalshi", "nse_intraday"}:
            raise ValueError("Unknown fee model")
        if (not np.isfinite(self.per_share) or not np.isfinite(self.multiplier)
                or self.per_share < 0 or self.multiplier < 0
                or not Decimal(self.rounding).is_finite() or Decimal(self.rounding) <= 0):
            raise ValueError("Invalid fee parameters")

    def leg(self, q, price, buy):
        if q <= 0 or not np.isfinite(price) or price <= 0:
            raise ValueError("Invalid fee input")
        if self.kind == "zero":
            return 0.0
        if self.kind == "per_share":
            return q * self.per_share
        if self.kind == "kalshi":
            if price >= 1 or int(q) != q:
                raise ValueError("Kalshi requires whole contracts and 0 < price < 1")
            p = Decimal(str(price))
            raw = Decimal(str(self.multiplier)) * Decimal("0.07") * Decimal(str(q)) * p * (1 - p)
            quantum = Decimal(self.rounding)
            return float((raw / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum)
        # Zerodha NSE cash intraday scenario; daily contract-note rounding and
        # broker square-off/other account-specific charges are not reconstructed.
        turnover = q * price
        brokerage = min(20.0, turnover * 0.0003)
        exchange = turnover * 0.0000307
        sebi = turnover * 0.000001
        ipft = turnover * 0.000000001
        gst = 0.18 * (brokerage + exchange + sebi + ipft)
        tax = turnover * (0.00003 if buy else 0.00025)
        return brokerage + exchange + sebi + ipft + gst + tax


@dataclass(frozen=True)
class Config:
    max_trades: int = 50
    latency_ns: int = 5_000_000
    q: int = 1
    warmup: int = 30
    mode: str = "imbalance_state"
    allow_proxy_clock: bool = False
    # Explicit legacy action policy, held fixed across the two signal variants.
    # A large p0 does NOT automatically imply abstention under this policy.
    action_policy: str = "direction_argmax"
    diagnostic_after_50: bool = True

    def __post_init__(self):
        if self.max_trades != 50:
            raise ValueError("Primary strategy is locked to 50 trades")
        if (self.latency_ns < 0 or int(self.latency_ns) != self.latency_ns
                or self.q < 1 or int(self.q) != self.q
                or self.warmup < 1 or int(self.warmup) != self.warmup):
            raise ValueError("Invalid latency, quantity or warmup")
        if self.mode not in {"imbalance", "imbalance_state"}:
            raise ValueError("Unknown signal variant")
        if self.action_policy not in {"direction_argmax", "ternary_argmax"}:
            raise ValueError("Unknown action policy")


@dataclass
class Tape:
    ts: np.ndarray
    bids: np.ndarray
    asks: np.ndarray
    trade_count: np.ndarray
    asset: str
    segment: str
    clock_kind: str
    binary: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.ts = np.asarray(self.ts)
        self.trade_count = np.asarray(self.trade_count)
        self.bids = np.asarray(self.bids, dtype=float)
        self.asks = np.asarray(self.asks, dtype=float)
        n = len(self.ts)
        if not n or self.ts.dtype.kind not in "iu" or self.trade_count.dtype.kind not in "iu":
            raise ValueError("Nonempty integer timestamps and trade counters required")
        if self.trade_count.shape != (n,) or np.any(np.diff(self.ts) < 0):
            raise ValueError("Unsorted timestamps or incompatible counter")
        if np.any(np.diff(self.trade_count) < 0) or self.trade_count[0] < 0:
            raise ValueError("Trade count must be monotone")
        if self.clock_kind == "trades" and np.any(np.diff(self.trade_count) > 1):
            raise ValueError("Exact trade clock cannot skip trade observations")
        for b in (self.bids, self.asks):
            if b.ndim != 3 or b.shape[0] != n or b.shape[2] != 2 or b.shape[1] < 1:
                raise ValueError("Book shape must be (rows, levels, price/quantity)")
        # Invalid rows stay in the clock. Never compress time by dropping them.
        self.valid = np.array([self._valid_book(i) for i in range(n)])

    def _valid_book(self, i):
        b, a = self.bids[i], self.asks[i]
        if not np.isfinite(b).all() or not np.isfinite(a).all():
            return False
        if np.any(b[:, 1] < 0) or np.any(a[:, 1] < 0):
            return False
        if b[0, 1] <= 0 or a[0, 1] <= 0:
            return False
        bp, ap = b[b[:, 1] > 0, 0], a[a[:, 1] > 0, 0]
        if not len(bp) or not len(ap) or np.any(bp <= 0) or np.any(ap <= 0):
            return False
        if np.any(np.diff(bp) >= 0) or np.any(np.diff(ap) <= 0) or bp[0] >= ap[0]:
            return False
        return not self.binary or (np.all(bp < 1) and np.all(ap < 1))

    def arrival(self, signal_i, latency_ns):
        # Snapshot approximation: execute at the first available observation at
        # or after arrival, never at a future-selected profitable observation.
        i = max(signal_i, int(np.searchsorted(self.ts, int(self.ts[signal_i]) + latency_ns)))
        return i if i < len(self.ts) else None

    def deadline(self, anchor_i, max_trades):
        i = int(np.searchsorted(self.trade_count, int(self.trade_count[anchor_i]) + max_trades))
        return i if i < len(self.ts) else None

    def fills(self, i, side, entry, q):
        if i is None or not self.valid[i]:
            return None
        book = self.asks[i] if (side == 1) == entry else self.bids[i]
        remaining, fills = float(q), []
        for price, size in book:
            take = min(remaining, size)
            if take > 0:
                if self.binary and side == -1:
                    price = float(Decimal("1") - Decimal(str(price)))
                fills.append((float(take), float(price)))
                remaining -= take
            if remaining <= EPS:
                return fills
        return None  # Explicit all-or-none visible-depth model, not a partial fill.


def fill_value(fills):
    q = sum(size for size, _ in fills)
    return sum(size * price for size, price in fills) / q


def fill_fee(fills, fees, buy):
    if fees.kind == "kalshi":
        return sum(fees.leg(q, p, buy) for q, p in fills)
    return fees.leg(sum(q for q, _ in fills), fill_value(fills), buy)


def accounting(tape, side, entry, out, fees):
    q = sum(size for size, _ in entry)
    direction = 1 if tape.binary else side
    gross = q * direction * (fill_value(out) - fill_value(entry))
    entry_buy = tape.binary or side == 1
    cost = fill_fee(entry, fees, entry_buy) + fill_fee(out, fees, not entry_buy)
    return float(gross), float(cost), float(gross - cost)


@dataclass(frozen=True)
class Label:
    y: Optional[int]
    resolved_i: Optional[int]
    status: str


def first_passage(tape, anchor_i, cfg, fees):
    if not tape.valid[anchor_i]:
        return Label(None, None, "invalid_anchor")
    entry_i = tape.arrival(anchor_i, cfg.latency_ns)
    end = tape.deadline(anchor_i, cfg.max_trades)
    if entry_i is None or (end is not None and entry_i >= end):
        return Label(None, None, "entry_unavailable")
    entries = {s: tape.fills(entry_i, s, True, cfg.q) for s in (-1, 1)}
    if any(x is None for x in entries.values()):
        return Label(None, None, "insufficient_entry_depth")
    stop = end if end is not None else len(tape.ts) - 1
    for j in range(entry_i + 1, stop + 1):
        if not tape.valid[j]:
            return Label(None, None, "observation_gap")
        hits = []
        for s in (-1, 1):
            out = tape.fills(j, s, False, cfg.q)
            if out is not None and accounting(tape, s, entries[s], out, fees)[2] > EPS:
                hits.append(s)
        if len(hits) > 1:
            return Label(None, j, "ambiguous_simultaneous_hit")
        if hits:
            return Label(hits[0], j, "hit")
    if end is None:
        return Label(None, None, "right_censored")
    if not tape.valid[entry_i:end + 1].all():
        return Label(None, None, "observation_gap")
    return Label(0, end, "timeout")


class History:
    """Expanding counts, updated only after the outcome's observation timestamp."""

    def __init__(self):
        self.counts = {}
        self.pending = []
        self.pending_counts = []
        self.total = 0
        self.latest_state = None
        self.latest_origin = -1
        self.next_origin = 0
        self.last_tape_end = None
        self.halted = False

    def schedule(self, label, tape, anchor_i, bucket, state):
        origin = self.next_origin
        self.next_origin += 1
        if label.y is not None:
            heapq.heappush(self.pending, (
                int(tape.ts[label.resolved_i]), origin, int(label.y), bucket, state
            ))
            deadline = tape.deadline(anchor_i, 50)
            if deadline is not None:
                heapq.heappush(self.pending_counts, (
                    int(tape.ts[deadline]), origin, int(label.y), bucket, state
                ))

    def advance(self, now):
        # Timestamps tied with the decision are excluded because their ordering
        # is not reliably supplied by every data source.
        while self.pending and self.pending[0][0] < now:
            _, origin, y, bucket, state = heapq.heappop(self.pending)
            if origin > self.latest_origin:
                self.latest_state, self.latest_origin = y, origin
        # Equal exposure windows avoid overrepresenting quick hits while
        # zero outcomes are still waiting for their full 50-trade follow-up.
        while self.pending_counts and self.pending_counts[0][0] < now:
            _, _, y, bucket, state = heapq.heappop(self.pending_counts)
            for key in [("all",), (bucket,), (bucket, state)]:
                self.counts.setdefault(key, np.zeros(3))[y + 1] += 1
            self.total += 1

    def probabilities(self, bucket, state, mode):
        # No silent fallback to unconditional counts. Sparse cells are exposed.
        key = (bucket,) if mode == "imbalance" else (bucket, state)
        c = self.counts.get(key, np.zeros(3))
        return (c + 0.5) / (c.sum() + 1.5), int(c.sum())


def features(tape):
    b, a = tape.bids[:, 0, :], tape.asks[:, 0, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = (b[:, 1] - a[:, 1]) / (b[:, 1] + a[:, 1])
    s = pd.Series(np.where(tape.valid, imbalance, np.nan))
    # Fixed signed bins have the same meaning across the expanding history.
    bucket = np.searchsorted([-0.8, -0.4, 0.4, 0.8], s.to_numpy(), side="right")
    spread = pd.Series(np.where(tape.valid, a[:, 0] - b[:, 0], np.nan))
    tight = spread <= spread.expanding(min_periods=20).quantile(0.25)
    eligible = tape.valid & (s.abs().to_numpy() >= 0.8) & tight.to_numpy()
    return bucket, eligible


def late_diagnostic(tape, anchor_i, entry_i, side, entry, cfg, fees):
    """Same chosen side and original entry; never feeds counts or booked PnL."""
    end = tape.deadline(anchor_i, cfg.max_trades)
    result = dict(late_hit_i=None, late_wait_trades=None, late_wait_ms=None,
                  diagnostic_censored=False, diagnostic_observation_gaps=False,
                  late_correct_first_passage=False, opposite_move_first=False)
    if end is None:
        result["diagnostic_censored"] = True
        return result
    opposite_entry = tape.fills(entry_i, -side, True, cfg.q)
    first_direction = None
    for j in range(entry_i + 1, len(tape.ts)):
        if not tape.valid[j]:
            result["diagnostic_observation_gaps"] = True
        out = tape.fills(j, side, False, cfg.q)
        other_out = tape.fills(j, -side, False, cfg.q)
        if (first_direction is None and opposite_entry is not None and other_out is not None
                and accounting(tape, -side, opposite_entry, other_out, fees)[2] > EPS):
            first_direction = -side
        if out is not None and accounting(tape, side, entry, out, fees)[2] > EPS:
            if first_direction is None:
                first_direction = side
            result["opposite_move_first"] = first_direction == -side
            if j <= end:
                result["diagnostic_status"] = "hit_within_50"
                return result
            result.update(late_hit_i=j,
                          late_wait_trades=int(tape.trade_count[j] - tape.trade_count[anchor_i]),
                          late_wait_ms=float((tape.ts[j] - tape.ts[anchor_i]) / 1e6),
                          late_correct_first_passage=(first_direction == side and opposite_entry is not None
                                                     and not result["diagnostic_observation_gaps"]),
                          diagnostic_status="late_opportunity_not_realized_profit")
            return result
    result.update(diagnostic_status="no_later_hit_observed", diagnostic_censored=True)
    return result


def execute(tape, anchor_i, side, cfg, fees):
    """One position, first observable trigger, then unconditional delayed order."""
    record = dict(anchor_i=anchor_i, side=side, status="no_entry", q=cfg.q)
    if side not in {-1, 1} or not tape.valid[anchor_i]:
        return record, []
    entry_i = tape.arrival(anchor_i, cfg.latency_ns)
    end = tape.deadline(anchor_i, cfg.max_trades)
    if entry_i is None or (end is not None and entry_i >= end):
        return record, []
    entry = tape.fills(entry_i, side, True, cfg.q)
    if entry is None:
        record["status"] = "entry_depth_rejected"
        return record, []
    record.update(entry_i=entry_i, entry=fill_value(entry), status="open_unresolved",
                  deadline_i=end, entry_fee=fill_fee(entry, fees, tape.binary or side == 1))
    marks = [(entry_i, -record["entry_fee"])]
    trigger = None
    trigger_reason = None
    stop = end if end is not None else len(tape.ts) - 1
    for j in range(entry_i + 1, stop + 1):
        out = tape.fills(j, side, False, cfg.q)
        if out is not None and accounting(tape, side, entry, out, fees)[2] > EPS:
            trigger, trigger_reason = j, "profit_opportunity"
            break
        if j == end:
            trigger, trigger_reason = j, "50_trade_timeout"
            break
    exit_i = None
    out = None
    if trigger is not None:
        record.update(exit_signal_i=trigger, exit_reason=trigger_reason)
        attempt = tape.arrival(trigger, cfg.latency_ns)
        # Rejected all-or-none exits retry only after the rejection is observed
        # and another latency interval has elapsed. No profit re-check.
        while attempt is not None:
            out = tape.fills(attempt, side, False, cfg.q)
            if out is not None:
                exit_i = attempt
                break
            record["exit_depth_rejections"] = record.get("exit_depth_rejections", 0) + 1
            nxt = tape.arrival(attempt, cfg.latency_ns)
            if nxt == attempt:
                nxt = attempt + 1 if attempt + 1 < len(tape.ts) else None
            attempt = nxt
    mark_end = exit_i if exit_i is not None else len(tape.ts) - 1
    for j in range(entry_i, mark_end + 1):
        mark = tape.fills(j, side, False, cfg.q)
        if mark is not None:
            marks.append((j, accounting(tape, side, entry, mark, fees)[2]))
        else:
            record["unmarkable_rows"] = record.get("unmarkable_rows", 0) + 1
    if exit_i is not None:
        gross, cost, net = accounting(tape, side, entry, out, fees)
        record.update(status="closed", exit_i=exit_i, exit=fill_value(out),
                      gross=gross, fees=cost, net=net,
                      wait_trades=int(tape.trade_count[exit_i] - tape.trade_count[anchor_i]),
                      signal_wait_trades=int(tape.trade_count[trigger] - tape.trade_count[anchor_i]),
                      wait_ms=float((tape.ts[exit_i] - tape.ts[anchor_i]) / 1e6))
    else:
        record["last_observed_liquidation_net"] = marks[-1][1]
    if cfg.diagnostic_after_50:
        record.update(late_diagnostic(tape, anchor_i, entry_i, side, entry, cfg, fees))
    return record, marks


def replay(tape, cfg, fees, anchors=None, buckets=None, history=None):
    if tape.clock_kind != "trades" and not cfg.allow_proxy_clock:
        raise ValueError(f"{tape.asset}: {tape.clock_kind}, not a verified trade clock; "
                         "explicit proxy opt-in is required")
    if tape.binary and fees.kind not in {"kalshi", "zero"}:
        raise ValueError("Binary contracts require an explicit compatible fee model")
    if buckets is None or anchors is None:
        natural_buckets, eligible = features(tape)
        if buckets is None:
            buckets = natural_buckets
        if anchors is None:
            anchors = np.flatnonzero(eligible & (np.arange(len(tape.ts)) % 5 == 0))
    anchors = np.asarray(anchors, dtype=int)
    if len(anchors) and (anchors[0] < 0 or anchors[-1] >= len(tape.ts) or np.any(np.diff(anchors) <= 0)):
        raise ValueError("Anchors must be strictly increasing unique row indices")
    if len(buckets) != len(tape.ts):
        raise ValueError("One bucket per observation required")
    history = history if history is not None else History()
    if history.halted:
        raise ValueError("An unresolved position in the prior segment blocks further replay")
    if history.last_tape_end is not None and tape.ts[0] < history.last_tape_end:
        raise ValueError("Overlapping segments cannot share an expanding history")
    history.last_tape_end = int(tape.ts[-1])
    predictions, trades, equity = [], [], [(0, 0.0)]
    blocked_until = -1
    blocked_ts = -1
    realized = 0.0
    unresolved = False
    for i in anchors:
        i = int(i)
        history.advance(int(tape.ts[i]))
        state = history.latest_state
        bucket = int(buckets[i])
        probabilities, cell_n = history.probabilities(bucket, state, cfg.mode)
        pm, p0, pp = probabilities
        side = 1 if pp > pm else -1 if pm > pp else 0
        if cfg.action_policy == "ternary_argmax" and p0 >= max(pm, pp):
            side = 0
        label = first_passage(tape, i, cfg, fees)
        predictions.append(dict(anchor_i=i, state=state, bucket=bucket,
                                pminus=pm, pzero=p0, pplus=pp, cell_n=cell_n,
                                history_n=history.total, side=side,
                                y=label.y, resolved_i=label.resolved_i,
                                label_status=label.status))
        # Precomputation of y is allowed; it is inaccessible to decisions until
        # advance() sees its resolution timestamp strictly in the past.
        history.schedule(label, tape, i, bucket, state)
        if history.total < cfg.warmup or not side or i <= blocked_until or int(tape.ts[i]) <= blocked_ts or unresolved:
            continue
        trade, marks = execute(tape, i, side, cfg, fees)
        trade.update(pminus=pm, pzero=p0, pplus=pp, state=state, cell_n=cell_n)
        trades.append(trade)
        equity.extend((j, realized + mark) for j, mark in marks)
        if trade["status"] == "closed":
            realized += trade["net"]
            blocked_until = trade["exit_i"]
            blocked_ts = int(tape.ts[blocked_until])
        elif trade["status"] == "open_unresolved":
            unresolved = True
            history.halted = True
    closed = [t for t in trades if t["status"] == "closed"]
    mtm = np.array([v for _, v in equity])
    realized_equity = np.r_[0., np.cumsum([t["net"] for t in closed])]
    mark_gaps = sum(t.get("unmarkable_rows", 0) for t in trades)
    summary = dict(asset=tape.asset, segment=tape.segment, mode=cfg.mode,
                   clock_kind=tape.clock_kind, fee_model=fees.kind, fee_provenance=fees.provenance,
                   max_wait_trades=50, latency_ms=cfg.latency_ns / 1e6, q=cfg.q,
                   closed_trades=len(closed), unresolved_positions=int(unresolved),
                   gross_closed=sum(t["gross"] for t in closed),
                   fees_closed=sum(t["fees"] for t in closed), net_closed=realized,
                   complete_net=None if unresolved else realized,
                   max_dd_closed=float(np.max(np.maximum.accumulate(realized_equity) - realized_equity)),
                   max_dd_observed_liquidation=float(np.max(np.maximum.accumulate(mtm) - mtm)),
                   unmarkable_rows=mark_gaps,
                   dd_is_lower_bound=bool(mark_gaps or unresolved),
                   late_same_side_opportunities=sum(t.get("late_hit_i") is not None for t in trades),
                   late_correct_first_passages=sum(t.get("late_correct_first_passage", False) for t in trades),
                   late_recoveries_after_opposite=sum(t.get("late_hit_i") is not None
                                                       and t.get("opposite_move_first", False) for t in trades),
                   labels_censored=sum(p["y"] is None for p in predictions))
    return dict(summary=summary, predictions=predictions, trades=trades, equity=equity)
