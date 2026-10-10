from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audited_fp import (
    Config, Fees, History, Label, Tape, accounting, execute, features,
    fill_value, first_passage, replay,
)
from audited_fp_runner import load_goog, load_kalshi, load_nse, main as runner_main


def tape(n=90, binary=False):
    bid, ask = (0.49, 0.50) if binary else (99., 100.)
    bids = np.tile([[[bid, 1000.]]], (n, 1, 1))
    asks = np.tile([[[ask, 1000.]]], (n, 1, 1))
    return Tape(np.arange(n, dtype=np.int64) * 1_000_000, bids, asks,
                np.arange(n), "TEST", "session", "trades", binary)


def with_prices(t, start, bid, ask, stop=None):
    b, a = t.bids.copy(), t.asks.copy()
    b[start:stop, 0, 0], a[start:stop, 0, 0] = bid, ask
    return replace(t, bids=b, asks=a)


class AccountingTests(unittest.TestCase):
    def test_kalshi_known_fees_and_float_boundary(self):
        f = Fees("kalshi")
        self.assertEqual(f.leg(1, .5, True), .02)
        self.assertEqual(f.leg(100, .5, True), 1.75)
        self.assertEqual(f.leg(100, .1, True), .63)
        self.assertEqual(Fees("kalshi", rounding="0.0001").leg(1, .5, True), .0175)

    def test_nse_manual_components(self):
        f = Fees("nse_intraday")
        turnover = 100_000
        base = 20 + 3.07 + .1 + .0001
        self.assertAlmostEqual(f.leg(100, 1000, True), base * 1.18 + 3)
        self.assertAlmostEqual(f.leg(100, 1000, False), base * 1.18 + 25)

    def test_stock_long_short_and_gross_minus_fees(self):
        t = tape()
        f = Fees("per_share", per_share=.1)
        self.assertEqual(accounting(t, 1, [(10, 100)], [(10, 102)], f), (20, 2, 18))
        self.assertEqual(accounting(t, -1, [(10, 100)], [(10, 98)], f), (20, 2, 18))

    def test_binary_no_is_purchased_not_stock_shorted(self):
        t = with_prices(tape(binary=True), 6, .39, .4)
        en = t.fills(5, -1, True, 1)
        ex = t.fills(6, -1, False, 1)
        self.assertAlmostEqual(fill_value(en), .51)
        self.assertAlmostEqual(fill_value(ex), .60)
        gross, fees, net = accounting(t, -1, en, ex, Fees("kalshi"))
        self.assertAlmostEqual(gross, .09)
        self.assertAlmostEqual(fees, .04)
        self.assertAlmostEqual(net, .05)

    def test_depth_vwap_and_rejection(self):
        t = tape()
        b = np.tile([[[99, 2], [98, 3]]], (90, 1, 1))
        a = np.tile([[[100, 2], [101, 3]]], (90, 1, 1))
        t = replace(t, bids=b, asks=a)
        self.assertEqual(fill_value(t.fills(5, 1, True, 5)), 100.6)
        self.assertIsNone(t.fills(5, 1, True, 6))

    def test_fee_and_quantity_parameters_validate(self):
        with self.assertRaises(ValueError):
            Config(q=0)
        with self.assertRaises(ValueError):
            Fees("unknown")
        with self.assertRaises(ValueError):
            Fees("per_share", per_share=-1)


class ExecutionTests(unittest.TestCase):
    def test_entry_uses_five_ms_not_one_row(self):
        tr, _ = execute(tape(), 0, 1, Config(), Fees("zero"))
        self.assertEqual(tr["entry_i"], 5)
        self.assertEqual(tr["exit_signal_i"], 50)
        self.assertEqual(tr["exit_i"], 55)

    def test_profit_trigger_must_accept_losing_delayed_fill(self):
        t = with_prices(tape(), 6, 101, 102, 7)
        t = with_prices(t, 11, 98, 99, 12)
        tr, _ = execute(t, 0, 1, Config(), Fees("zero"))
        self.assertEqual(tr["exit_signal_i"], 6)
        self.assertEqual(tr["exit_i"], 11)
        self.assertEqual(tr["net"], -2)

    def test_timeout_gets_latency_even_beyond_fifty(self):
        t = with_prices(tape(), 55, 95, 96)
        tr, _ = execute(t, 0, 1, Config(), Fees("zero"))
        self.assertEqual(tr["exit_reason"], "50_trade_timeout")
        self.assertEqual(tr["signal_wait_trades"], 50)
        self.assertEqual(tr["wait_trades"], 55)
        self.assertEqual(tr["net"], -5)

    def test_unfilled_entry_not_traded(self):
        t = tape()
        a = t.asks.copy()
        a[5, 0, 1] = 1
        t = replace(t, asks=a)
        tr, _ = execute(t, 0, 1, Config(q=2), Fees("zero"))
        self.assertEqual(tr["status"], "entry_depth_rejected")

    def test_unfilled_exit_retry_cannot_select_by_profit(self):
        t = with_prices(tape(), 6, 101, 102, 7)
        b = t.bids.copy()
        b[11, 0, 1] = .5
        t = replace(t, bids=b)
        tr, _ = execute(t, 0, 1, Config(), Fees("zero"))
        self.assertEqual(tr["exit_i"], 16)
        self.assertEqual(tr["exit_depth_rejections"], 1)
        self.assertEqual(tr["net"], -1)

    def test_missing_exit_preserves_open_trade(self):
        t = tape(53)
        tr, _ = execute(t, 0, 1, Config(), Fees("zero"))
        self.assertEqual(tr["status"], "open_unresolved")
        self.assertNotIn("net", tr)
        self.assertEqual(tr["exit_signal_i"], 50)

    def test_trades_not_quote_rows_drive_deadline(self):
        t = tape(130)
        t = replace(t, trade_count=np.arange(130) // 2)
        tr, _ = execute(t, 0, 1, Config(), Fees("zero"))
        self.assertEqual(tr["exit_signal_i"], 100)
        self.assertEqual(tr["signal_wait_trades"], 50)

    def test_late_diagnostic_cannot_change_pnl(self):
        t = with_prices(tape(), 70, 102, 103)
        tr, _ = execute(t, 0, 1, Config(), Fees("zero"))
        plain, _ = execute(t, 0, 1, Config(diagnostic_after_50=False), Fees("zero"))
        self.assertEqual(tr["late_hit_i"], 70)
        self.assertEqual(tr["late_wait_trades"], 70)
        self.assertEqual(tr["net"], plain["net"])
        self.assertEqual(tr["net"], -1)
        self.assertTrue(tr["late_correct_first_passage"])

    def test_late_recovery_is_not_correct_first_passage(self):
        t = with_prices(tape(), 20, 97, 98, 25)
        t = with_prices(t, 70, 102, 103)
        tr, _ = execute(t, 0, 1, Config(), Fees("zero"))
        self.assertEqual(tr["late_hit_i"], 70)
        self.assertTrue(tr["opposite_move_first"])
        self.assertFalse(tr["late_correct_first_passage"])

    def test_no_late_hit_is_censored_not_permanent_wrong_direction(self):
        tr, _ = execute(tape(), 0, 1, Config(), Fees("zero"))
        self.assertTrue(tr["diagnostic_censored"])

    def test_late_diagnostic_does_not_count_already_correct_prediction(self):
        t = with_prices(tape(), 10, 101, 102)
        tr, _ = execute(t, 0, 1, Config(), Fees("zero"))
        self.assertIsNone(tr["late_hit_i"])
        self.assertEqual(tr["diagnostic_status"], "hit_within_50")


class LabelTests(unittest.TestCase):
    def test_random_paths_against_independent_scalar_reference(self):
        rng = np.random.default_rng(731)
        cfg, fees = Config(diagnostic_after_50=False), Fees("per_share", per_share=.1)
        for trial in range(100):
            t = tape()
            prices = 100 + np.cumsum(rng.normal(0, .4, 90))
            b, a = t.bids.copy(), t.asks.copy()
            b[:, 0, 0], a[:, 0, 0] = prices - .1, prices + .1
            t = replace(t, bids=b, asks=a)
            expected_y, resolved = 0, 50
            for j in range(6, 51):
                if b[j, 0, 0] - a[5, 0, 0] > .2 + 1e-9:
                    expected_y, resolved = 1, j
                    break
                if b[5, 0, 0] - a[j, 0, 0] > .2 + 1e-9:
                    expected_y, resolved = -1, j
                    break
            label = first_passage(t, 0, cfg, fees)
            self.assertEqual((label.y, label.resolved_i), (expected_y, resolved))
            for side in [-1, 1]:
                entry = a[5, 0, 0] if side == 1 else b[5, 0, 0]
                trigger = 50
                for j in range(6, 51):
                    px = b[j, 0, 0] if side == 1 else a[j, 0, 0]
                    if side * (px - entry) - .2 > 1e-9:
                        trigger = j
                        break
                exit_px = b[trigger + 5, 0, 0] if side == 1 else a[trigger + 5, 0, 0]
                tr, _ = execute(t, 0, side, cfg, fees)
                self.assertEqual(tr["exit_i"], trigger + 5)
                self.assertAlmostEqual(tr["net"], side * (exit_px - entry) - .2)

    def test_same_delayed_entry_as_replay(self):
        t = with_prices(tape(), 5, 104, 105)
        label = first_passage(t, 0, Config(), Fees("zero"))
        self.assertEqual(label.y, 0)

    def test_fee_covering_label_not_merely_midprice_move(self):
        t = with_prices(tape(), 6, 100.1, 101.1)
        label = first_passage(t, 0, Config(), Fees("per_share", per_share=.1))
        self.assertEqual(label.y, 0)

    def test_positive_label_is_opportunity_not_fill_promise(self):
        t = with_prices(tape(), 6, 101, 102, 7)
        label = first_passage(t, 0, Config(), Fees("zero"))
        tr, _ = execute(t, 0, 1, Config(), Fees("zero"))
        self.assertEqual(label.y, 1)
        self.assertEqual(label.resolved_i, tr["exit_signal_i"])
        self.assertLess(tr["net"], 0)

    def test_end_of_file_is_not_zero_label(self):
        self.assertEqual(first_passage(tape(30), 0, Config(), Fees("zero")).status, "right_censored")

    def test_hit_exactly_fifty_is_included_and_fifty_one_is_not(self):
        t = with_prices(tape(), 50, 101, 102)
        self.assertEqual(first_passage(t, 0, Config(), Fees("zero")).y, 1)
        t = with_prices(tape(), 51, 101, 102)
        label = first_passage(t, 0, Config(), Fees("zero"))
        self.assertEqual(label.y, 0)
        self.assertEqual(label.resolved_i, 50)

    def test_invalid_book_censors_instead_of_hiding_earlier_passage(self):
        t = with_prices(tape(), 20, 101, 102)
        a = t.asks.copy()
        a[10, 0, 0] = np.nan
        t = replace(t, asks=a)
        self.assertEqual(first_passage(t, 0, Config(), Fees("zero")).status, "observation_gap")

    def test_insufficient_label_entry_depth_is_not_zero(self):
        label = first_passage(tape(), 0, Config(q=2000), Fees("zero"))
        self.assertIsNone(label.y)

    def test_first_direction_not_eventual_direction(self):
        t = with_prices(tape(), 10, 97, 98, 15)
        t = with_prices(t, 20, 101, 102)
        self.assertEqual(first_passage(t, 0, Config(), Fees("zero")).y, -1)


class CausalityTests(unittest.TestCase):
    def test_expanding_history_carries_across_nonoverlapping_segments(self):
        t = with_prices(tape(), 6, 101, 102)
        h = History()
        cfg = Config(mode="imbalance", warmup=1000)
        replay(t, cfg, Fees("zero"), [0], np.full(90, 4), history=h)
        later = replace(tape(), ts=tape().ts + 100_000_000, segment="next")
        r = replay(later, cfg, Fees("zero"), [0], np.full(90, 4), history=h)
        self.assertEqual(r["predictions"][0]["history_n"], 1)
        self.assertEqual(r["predictions"][0]["state"], 1)

    def test_overlapping_dataset_cannot_duplicate_history(self):
        h, t = History(), tape()
        replay(t, Config(), Fees("zero"), [], np.zeros(90), history=h)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            replay(t, Config(), Fees("zero"), [], np.zeros(90), history=h)

    def test_unresolved_position_blocks_next_segment(self):
        h = History()
        h.halted = True
        with self.assertRaisesRegex(ValueError, "unresolved"):
            replay(tape(), Config(), Fees("zero"), [], np.zeros(90), history=h)

    def test_no_early_publication_even_with_overlap_and_ties(self):
        h, t = History(), tape()
        h.schedule(Label(1, 50, "hit"), t, 0, 4, None)
        h.schedule(Label(-1, 20, "hit"), t, 10, 4, None)
        h.advance(20_000_000)
        self.assertEqual(h.total, 0)
        h.advance(21_000_000)
        self.assertEqual(h.total, 0)
        self.assertEqual(h.latest_state, -1)
        h.advance(51_000_000)
        self.assertEqual(h.total, 1)
        self.assertEqual(h.latest_state, -1)  # Most recent resolved anchor, not old anchor 0.
        h.advance(61_000_000)
        self.assertEqual(h.total, 2)

    def test_quick_hits_and_zeros_have_equal_count_maturity(self):
        t = tape()
        a, b = History(), History()
        a.schedule(Label(1, 6, "hit"), t, 0, 4, None)
        b.schedule(Label(0, 50, "timeout"), t, 0, 4, None)
        a.advance(7_000_000)
        b.advance(7_000_000)
        self.assertEqual(a.latest_state, 1)
        self.assertIsNone(b.latest_state)
        self.assertEqual((a.total, b.total), (0, 0))
        a.advance(51_000_000)
        b.advance(51_000_000)
        self.assertEqual((a.total, b.total), (1, 1))

    def test_truncated_hit_can_update_state_but_not_full_window_counts(self):
        t, h = tape(30), History()
        h.schedule(Label(1, 6, "hit"), t, 0, 4, None)
        h.advance(100_000_000)
        self.assertEqual(h.latest_state, 1)
        self.assertEqual(h.total, 0)

    def test_state_conditioning_learns_separate_cells(self):
        h, t = History(), tape()
        for i in range(5):
            h.schedule(Label(1, i + 10, "hit"), t, i, 4, 1)
        for i in range(5, 10):
            h.schedule(Label(-1, i + 10, "hit"), t, i, 4, -1)
        h.advance(70_000_000)
        p, n = h.probabilities(4, 1, "imbalance_state")
        m, _ = h.probabilities(4, -1, "imbalance_state")
        self.assertGreater(p[2], p[0])
        self.assertGreater(m[0], m[2])
        self.assertEqual(n, 5)

    def test_sparse_cell_does_not_silently_drop_condition(self):
        h, t = History(), tape()
        h.schedule(Label(1, 10, "hit"), t, 0, 4, None)
        h.advance(61_000_000)
        p, n = h.probabilities(0, 1, "imbalance_state")
        np.testing.assert_allclose(p, [1/3] * 3)
        self.assertEqual(n, 0)

    def test_feature_prefix_invariance(self):
        t = tape()
        t2 = with_prices(t, 40, 1, 10000)
        b1, e1 = features(t)
        b2, e2 = features(t2)
        np.testing.assert_array_equal(b1[:40], b2[:40])
        np.testing.assert_array_equal(e1[:40], e2[:40])

    def test_predictions_invariant_under_future_mutation(self):
        t = with_prices(tape(150), 15, 101, 102)
        t2 = with_prices(t, 60, 90, 91)
        cfg = Config(mode="imbalance", warmup=1, diagnostic_after_50=False)
        anchors, buckets = np.arange(0, 130, 5), np.full(150, 4)
        a = replay(t, cfg, Fees("zero"), anchors, buckets)["predictions"]
        b = replay(t2, cfg, Fees("zero"), anchors, buckets)["predictions"]
        fields = ["anchor_i", "state", "bucket", "pminus", "pzero", "pplus", "cell_n", "history_n", "side"]
        for pa, pb in zip(a, b):
            if pa["anchor_i"] < 60:
                self.assertEqual({k: pa[k] for k in fields}, {k: pb[k] for k in fields})

    def test_prefix_replay_matches_full_tape_predictions(self):
        t = with_prices(tape(150), 15, 101, 102)
        short = replace(t, ts=t.ts[:60], bids=t.bids[:60], asks=t.asks[:60], trade_count=t.trade_count[:60])
        cfg = Config(mode="imbalance", warmup=1, diagnostic_after_50=False)
        a = replay(t, cfg, Fees("zero"), np.arange(0, 60, 5), np.full(150, 4))["predictions"]
        b = replay(short, cfg, Fees("zero"), np.arange(0, 60, 5), np.full(60, 4))["predictions"]
        for pa, pb in zip(a, b):
            for key in ("state", "pminus", "pzero", "pplus", "history_n", "side"):
                self.assertEqual(pa[key], pb[key])

    def test_no_overlap_open_positions_and_no_closed_pnl_disguised_as_total(self):
        t = with_prices(tape(160), 6, 101, 102)
        b = t.bids.copy()
        b[110:, 0, 1] = 0
        t = replace(t, bids=b)
        r = replay(t, Config(mode="imbalance", warmup=1), Fees("zero"),
                   np.arange(0, 150, 10), np.full(160, 4))
        self.assertEqual(r["summary"]["unresolved_positions"], 1)
        self.assertIsNone(r["summary"]["complete_net"])
        self.assertEqual(sum(x["status"] == "open_unresolved" for x in r["trades"]), 1)

    def test_empty_anchors_no_error(self):
        r = replay(tape(), Config(), Fees("zero"), [], np.zeros(90))
        self.assertEqual(r["summary"]["complete_net"], 0)

    def test_trade_ledger_reconciles_and_positions_do_not_overlap(self):
        t = tape(500)
        for i in range(10, 500, 10):
            t = with_prices(t, i, 99 + i / 10, 100 + i / 10)
        r = replay(t, Config(mode="imbalance", warmup=1), Fees("per_share", per_share=.1),
                   np.arange(0, 440, 5), np.full(500, 4))
        closed = [tr for tr in r["trades"] if tr["status"] == "closed"]
        self.assertGreater(len(closed), 2)
        for tr in closed:
            self.assertAlmostEqual(tr["gross"] - tr["fees"], tr["net"])
        for a, b in zip(closed, closed[1:]):
            self.assertGreater(b["anchor_i"], a["exit_i"])
        self.assertAlmostEqual(sum(tr["net"] for tr in closed), r["summary"]["net_closed"])
        self.assertGreaterEqual(r["summary"]["max_dd_observed_liquidation"], r["summary"]["max_dd_closed"])


class AdapterTests(unittest.TestCase):
    def test_cli_synthetic_end_to_end_json_output(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "toy_goog.npz"
            out = Path(directory) / "result.json"
            n = 400
            np.savez(p, ts=np.arange(n), bb=99 + np.arange(n) * .1,
                     ba=100 + np.arange(n) * .1, bq=np.full(n, 1000), aq=np.ones(n))
            with redirect_stdout(io.StringIO()):
                runner_main(["--kind", "goog", "--path", str(p), "--fee-model", "zero",
                             "--q", "1", "--allow-proxy-clock", "--out", str(out)])
            result = json.loads(out.read_text())
            self.assertEqual(len(result["results"]), 2)
            self.assertEqual({r["config"]["mode"] for r in result["results"]},
                             {"imbalance", "imbalance_state"})
            self.assertTrue(all(r["config"]["max_trades"] == 50 for r in result["results"]))

    def test_proxy_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, "not a verified trade clock"):
            replay(replace(tape(), clock_kind="book_updates"), Config(), Fees("zero"))

    def test_timestamps_must_be_sorted_not_silently_reordered(self):
        ts = tape().ts.copy()
        ts[5] = 0
        with self.assertRaises(ValueError):
            replace(tape(), ts=ts)

    def test_negative_depth_rejected_without_deleting_row(self):
        t = tape()
        b = t.bids.copy()
        b[5, 0, 1] = -1
        t = replace(t, bids=b)
        self.assertFalse(t.valid[5])
        self.assertEqual(len(t.ts), 90)

    def test_skipped_trade_count_cannot_claim_exact_clock(self):
        with self.assertRaises(ValueError):
            replace(tape(), trade_count=np.arange(90) * 2)

    def test_goog_keeps_duplicates_and_every_row(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "goog.npz"
            np.savez(p, ts=np.arange(30), bb=np.full(30, 99), ba=np.full(30, 100),
                     bq=np.ones(30), aq=np.ones(30))
            t = load_goog(p)[0]
            self.assertEqual(len(t.ts), 30)
            self.assertEqual(t.clock_kind, "book_updates")

    def test_nse_keeps_book_trade_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "nse.npz"
            t = tape(4)
            t.asks[1, 0, 0] = 98  # Crossed row must not shift tr_bookidx.
            np.savez(p, ts=np.arange(4), bids=t.bids, asks=t.asks,
                     tr_bookidx=[0, 0, 1, 3], symbol="NSE")
            loaded = load_nse(p)[0]
            np.testing.assert_array_equal(loaded.trade_count, [2, 3, 3, 4])
            self.assertFalse(loaded.valid[1])
            self.assertEqual(len(loaded.ts), 4)

    def test_kalshi_keeps_same_book_trades_and_separates_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "kalshi.csv"
            pd.DataFrame(dict(t=[1, 1, 2, 3], bid=[.49]*4, ask=[.50]*4,
                              bq=[10]*4, aq=[10]*4, segment=[1, 1, 2, 2],
                              qty=[1]*4, sign=[1]*4)).to_csv(p, index=False)
            loaded = load_kalshi(p, "TEST")
            self.assertEqual(len(loaded), 2)
            self.assertEqual([len(t.ts) for t in loaded], [2, 2])
            self.assertTrue(all(t.clock_kind != "trades" for t in loaded))


if __name__ == "__main__":
    unittest.main()
