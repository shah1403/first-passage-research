# Backtester Audit: Fix Before Asset Testing

Status: the old active replay paths are disabled; a shared replacement engine
has regression tests. No new asset profitability results have been produced.
Passing these tests is not a claim that every possible bug is removed or that
the available datasets support exchange-level execution fidelity.

## Confirmed Defects

| Defect in the former tester | Replacement behavior |
| --- | --- |
| Immediately counting a future first-passage outcome, even in skipped/overlapping anchors | State publishes only after resolution; counts wait for the full 50-trade follow-up |
| Quick-hit outcomes enter the sample earlier than zeros | All outcomes share the same 50-trade count-maturity rule, avoiding outcome-dependent recency bias |
| Initializing with labels whose outcome extends beyond the evaluation boundary | Same chronological maturity queue during initialization and live rolling |
| Using `y.shift(1)` as the previous state although that outcome may still be unknown | Use the latest-origin anchor whose first-passage label has already resolved |
| Rechecking future delayed fill profitability and ignoring losing fills | Commit to the first observable exit trigger; accept the delayed fill even at a loss |
| Capping exit latency at the end of the horizon | Timeout submits at trade 50; fills can occur later because latency still applies |
| NSE labels ignoring fees and depth while execution used both | Labels and execution share fixed-size VWAP and fee functions |
| One-row latency substituted for 5 ms | Timestamp-based 5 ms arrival on both legs, including timeout |
| Deduplicated/downsampled quote rows called trades | Explicit clock provenance; unverified clocks rejected unless proxy mode is requested |
| Dataset/day boundaries treated as zero outcomes | Incomplete observation is censored, not counted as state 0 |
| Open positions disappearing if no exit is available | Retain unresolved positions, block further entries, withhold complete PnL |
| Drawdown measured only on closed-trade totals | Also record observed executable-liquidation equity; flag missing marks |
| Silent fallback from imbalance-conditioned counts to unconditional counts | Expose sparse cell size; smooth that conditional cell without silently dropping conditions |
| Selecting the best horizon/configuration after the test | Primary limit fixed to 50; compare the same two predetermined signal variants |

The original source is preserved in `backtester_before_audit.zip`. Old CSV
results have not been overwritten. They should not be treated as validated
performance of this replacement engine. Other historical experimental scripts
are outside this migration and should not be assumed audited.

## Locked Signal And State

Compare expanding empirical counts for:

1. `P(next first-passage state | current imbalance bucket)`.
2. `P(next first-passage state | current imbalance bucket, S_t)`.

`S_t` is the label of the latest-origin eligible anchor whose outcome is already
observable. It is not an unresolved previous anchor's future label, nor a claim
that this process satisfies the Markov property. Before a state is observable,
it is `None`, not a fabricated zero.

For each anchor, simulate the same quantity and delayed entry on both sides.
`+1` means the long/YES side first has an observed, depth-supported liquidation
value strictly above entry cost plus both fees before or at trade 50. `-1`
means the corresponding short/NO side reaches its threshold first. `0` means
neither threshold was observed by the complete 50-trade deadline. Missing
entry depth, observation gaps, and truncated windows are censored.

This is an **exit-opportunity label**, not a promise of profitable execution.
After such an opportunity triggers an exit order, the actual order is delayed
5 ms and can lose money. Realized PnL does not relabel that loss as a correct
profitable trade. Direction/label accuracy and profitable-fill rate must be
reported separately.

Counts use `(N_j + 0.5)/(N_total + 1.5)` within the selected cell. Warmup is 30
complete-window observations; counts then expand without a fixed model fit. Outcome
updates continue while a position is open. Shared history can carry across
chronological segments of the same asset; overlapping segments cannot be added
again as independent history. Unresolved positions block the next segment.

State observation and count maturity are separate: a quick first-passage hit
can update `S_t` once observable, but it joins the count sample only after the
same full 50-trade follow-up required for zero outcomes. A truncated window
with an observed hit can inform current state but does not enter that sample.

## Explicit Policy Choices

These are declared choices, not claimed bug fixes or optimized settings:

- Five fixed signed imbalance bins with edges `-0.8, -0.4, 0.4, 0.8`, avoiding
  changing the economic meaning of a bucket over time.
- Existing extreme-imbalance/tight-spread setup retained: absolute imbalance
  at least 0.8 and spread at most its expanding lower quartile, with 20 valid
  observations needed. Candidate stride is five observations, explicitly not
  five trades. No extra stable-imbalance filter is imposed on one asset only.
- Default action policy is the larger of up/down probabilities; exact ties
  abstain. A high `p0` alone does not veto a trade. A separately selectable
  ternary-argmax policy exists but has not been selected by results.
- Fixed user-supplied size, no probability sizing or simulated capital limit.
- All-or-none displayed-depth fills, no partial fills. Insufficient entry
  depth rejects entry. An insufficient-depth exit retries after another
  latency interval without a profit recheck.
- A single segment is a continuous observation session, not an asset pooling
  boundary. Session PnLs cannot simply have their individual drawdowns summed.

## Post-50 Diagnostics

For entered forecasts, retain the original entry and chosen side and inspect
the rest of that observed segment. Report the first later fee-covering
opportunity, its trade count and elapsed milliseconds. This never alters the
50-trade strategy PnL or its learning labels.

Distinguish a genuinely late first passage in the predicted direction from a
later recovery after the opposite direction reached its threshold first.
No later observed hit is right-censored at the end of the segment, not proof
the direction would never have occurred. Observation gaps are flagged.

## Execution And Data Limits Still Unresolved

The fill convention is the first observed book at or after order arrival.
Sparse snapshots do not reveal the actual book at 5 ms. L1 only supports an
all-or-none top-level size check, not an invented deeper fill or market impact.
There is no queue position, feed-to-decision delay, short-borrow availability,
portfolio margin, settlement, venue rejection, or endogenous market-impact
model. Do not call this a fully high-fidelity exchange simulator.

| Available input | What prevents an exact 50-trade/5 ms claim |
| --- | --- |
| GOOG NPZ files | L1 book updates, no trade identifiers/counts. Old loader also downsampled 20x; new loader does neither filtering nor downsampling |
| INFY/TATAMOTORS NPZ | L5 snapshots and `tr_bookidx` permit counts, but not precise within-snapshot trade times or intermediate execution books |
| Kalshi feature CSVs | Trade rows retained without further deduplication, but the original parser skipped invalid-book trades and has no intervening book updates |
| Kalshi raw ZST streams | Available for future full-stream reconstruction; not yet connected to the replacement runner |
| `orderbook.parquet` and `trades.parquet` | Successfully opened during this audit; 8,047 books and 111,160 trades across BUF/JAC NFL contracts. Alignment and sequence coverage still need validation |
| FI-2010 transformed text data | Native prices, instrument identities and timestamps have not been established for executable PnL |

`sports_v3`, `sports_v4`, and the pregame extracts are versions/windows of
underlying games, not automatically independent asset samples. An all-asset
runner must deduplicate overlapping source coverage before aggregating.

## Fees

The replacement has explicit fee scenarios shared by labels and fills.
It does not silently claim that a placeholder is an exact brokerage schedule.

- NSE cash intraday: Zerodha-schedule scenario includes brokerage (0.03%, capped
  at INR 20/order), STT on sells, stamp duty on buys, exchange, SEBI, IPFT and GST.
  Daily contract-note rounding/account-specific fees remain approximations.
  Source checked: https://zerodha.com/charges/ (2026-09-23).
- Kalshi: multiplier and fee quantum are configurable; Decimal rounding avoids
  floating-point over-rounding at fee boundaries. Current PDF wording describes
  centicent rounding while its illustrative table still shows cent-rounded
  amounts. The legacy cent-rounded fee is an explicit scenario, not verified
  per-contract historical account metadata. That metadata must be reconciled
  before claiming exact net PnL. Zero-fee/gross results remain separately useful.
  Source checked: https://kalshi.com/docs/kalshi-fee-schedule.pdf (2026-09-23).
- GOOG: per-share charges must be supplied explicitly and are labelled a
  research scenario; broker minimums, regulatory charges and borrow costs are
  not established by the files.

## Verification

Run `python verify_backtester.py` for the test suite and input inventory, without
running asset PnL experiments. See `verification.json` and `test_log.txt` for
actual test counts and source hashes.

Tests include future-suffix mutation and prefix replay invariance, mature-only
updates, overlapping anchors, timestamp ties, state-conditioned cells, cross-
segment history, 5 ms entry/exit latency, losing delayed fills, timeout latency,
depth rejection, unresolved positions, both directions and NO-contract pricing,
fees, truncation, the exact 50/51 boundary, post-50 recovery classification,
ledger reconciliation, and 100 random paths checked against an independent
scalar label/execution reference.

Use `python audited_fp_runner.py --help` for the explicit runner. Proxy datasets
are blocked by default. No asset profit table is asserted in this audit.

## Dataset provenance correction — 9 October 2026

The recovered TATAMOTORS and INFY datasets are **synthetic**, not historical NSE data. All stored arrays exactly match `nse_l2_sim.py` with its supplied seeds and parameters in the uploaded L2 toolkit. The generator deliberately embeds an imbalance/flow relationship and corrupt snapshots. All NSE results in this repository must be read as synthetic demonstrations. See the L2 repository’s `verification/synthetic_provenance.json` and audit script for the byte-level array comparison. This supersedes earlier “unknown provenance” wording.
