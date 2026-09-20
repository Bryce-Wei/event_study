"""Numerical checks for the new trading-session event-study extension.

Run from the repository root with ``python -m unittest discover -s tests -v``.
The examples use hand-calculated decimal returns, without network or data files.
"""

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from demo_data import make_demo_data
from event_analysis import analyze_events, summarize_car


class EventAnalysisTests(unittest.TestCase):
    def setUp(self):
        # Jan 6/7 is a weekend: session alignment is observable in this fixture.
        self.dates = pd.bdate_range("2024-01-01", periods=15)
        self.returns = pd.DataFrame(
            {"ret": np.zeros(15), "mkt": np.zeros(15)}, index=self.dates
        )
        self.returns.index.name = "Date"

    @staticmethod
    def event(event_id, date, event_type="upgrade", firm="Example Research"):
        return {
            "event_id": event_id,
            "firm": firm,
            "event_date": date,
            "event_type": event_type,
        }

    def two_event_fixture(self):
        returns = self.returns.copy()
        # AR is stock minus market, not the unadjusted stock return.
        returns["mkt"] = 0.005
        returns["ret"] = 0.005
        returns.loc[self.dates[2:5], "ret"] += [0.01, 0.02, -0.01]
        returns.loc[self.dates[8:11], "ret"] += [-0.01, 0.03, 0.02]
        events = pd.DataFrame(
            [self.event(11, self.dates[3]), self.event(22, self.dates[9])]
        )
        return returns, events

    def test_hand_calculated_ar_car_aar_and_caar(self):
        returns, events = self.two_event_fixture()
        result = analyze_events(returns, events, window=1)
        first = result.panel[result.panel.event_id == 11].sort_values("event_time")
        second = result.panel[result.panel.event_id == 22].sort_values("event_time")
        np.testing.assert_allclose(first.abnormal_return, [0.01, 0.02, -0.01])
        np.testing.assert_allclose(first.car, [0.01, 0.03, 0.02])
        np.testing.assert_allclose(second.car, [-0.01, 0.02, 0.04])
        summary = result.group_summary.sort_values("event_time")
        np.testing.assert_array_equal(summary.event_time, [-1, 0, 1])
        np.testing.assert_array_equal(summary.n, [2, 2, 2])
        np.testing.assert_allclose(summary.aar, [0, 0.025, 0.005], atol=1e-15)
        np.testing.assert_allclose(summary.caar, [0, 0.025, 0.03], atol=1e-15)
        np.testing.assert_allclose(
            result.event_summary.sort_values("event_id").car, [0.02, 0.04]
        )

    def test_caar_ci_uses_dispersion_of_cumulative_event_paths(self):
        returns, events = self.two_event_fixture()
        summary = analyze_events(returns, events, window=1).group_summary.sort_values(
            "event_time"
        )
        # Cumulative samples: [.01,-.01], [.03,.02], [.02,.04].
        # Their standard errors are .01, .005, .01 respectively.
        mean = np.array([0, 0.025, 0.03])
        se = np.array([0.01, 0.005, 0.01])
        half_width = stats.t.ppf(0.975, 1) * se
        np.testing.assert_allclose(summary.ci_lower, mean - half_width, atol=1e-14)
        np.testing.assert_allclose(summary.ci_upper, mean + half_width, atol=1e-14)
        np.testing.assert_allclose(summary.tstat, [0, 5, 3], atol=1e-14)
        np.testing.assert_allclose(
            summary.pvalue, 2 * stats.t.sf(np.abs([0, 5, 3]), df=1)
        )
        endpoint = summarize_car(
            analyze_events(returns, events, window=1).event_summary
        ).iloc[0]
        self.assertEqual(endpoint.n, 2)
        self.assertAlmostEqual(endpoint.mean_car, 0.03)
        self.assertAlmostEqual(endpoint.std_car, np.sqrt(0.0002))
        self.assertAlmostEqual(endpoint.ci_lower, mean[-1] - half_width[-1])
        self.assertAlmostEqual(endpoint.ci_upper, mean[-1] + half_width[-1])

    def test_weekend_aligns_to_next_available_session(self):
        events = pd.DataFrame([self.event(1, "2024-01-06")])
        result = analyze_events(self.returns, events, window=1)
        panel = result.panel.sort_values("event_time")
        np.testing.assert_array_equal(panel.event_time, [-1, 0, 1])
        self.assertEqual(
            list(pd.to_datetime(panel.ret_date)),
            list(pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])),
        )
        self.assertEqual(
            pd.Timestamp(result.event_summary.iloc[0].effective_event_date),
            pd.Timestamp("2024-01-08"),
        )
        self.assertEqual(result.event_summary.iloc[0].event_shift_days, 2)

    def test_complete_window_required_at_both_edges(self):
        events = pd.DataFrame(
            [
                self.event(1, self.dates[0]),
                self.event(2, self.dates[7]),
                self.event(3, self.dates[-1]),
            ]
        )
        result = analyze_events(self.returns, events, window=2)
        self.assertEqual(set(result.event_summary.event_id), {2})
        self.assertEqual(set(result.exclusions.event_id), {1, 3})
        self.assertEqual(len(result.panel), 5)
        self.assertEqual(set(result.group_summary.n), {1})

    def test_nonfinite_returns_exclude_whole_event_without_zero_fill(self):
        for column in ("ret", "mkt"):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(column=column, invalid=invalid):
                    returns, events = self.two_event_fixture()
                    returns.loc[self.dates[2], column] = invalid
                    result = analyze_events(returns, events, window=1)
                    self.assertEqual(set(result.event_summary.event_id), {22})
                    self.assertEqual(set(result.exclusions.event_id), {11})
                    self.assertEqual(set(result.group_summary.n), {1})
                    np.testing.assert_allclose(result.event_summary.car, [0.04])

    def test_invalid_returns_outside_windows_do_not_discard_valid_events(self):
        returns, events = self.two_event_fixture()
        returns.loc[self.dates[-1], "ret"] = np.nan
        result = analyze_events(returns, events, window=1)
        self.assertEqual(set(result.event_summary.event_id), {11, 22})
        self.assertEqual(len(result.exclusions), 0)

    def test_single_event_has_undefined_inference(self):
        returns, events = self.two_event_fixture()
        result = analyze_events(returns, events.iloc[:1], window=1)
        summary = result.group_summary
        self.assertTrue(summary[["tstat", "pvalue", "ci_lower", "ci_upper"]].isna().all().all())
        endpoint = summarize_car(result.event_summary)
        self.assertTrue(
            endpoint[["tstat", "pvalue", "std_car", "ci_lower", "ci_upper"]]
            .isna().all().all()
        )
        np.testing.assert_allclose(summary.sort_values("event_time").caar, [0.01, 0.03, 0.02])

    def test_duplicate_return_dates_raise(self):
        events = pd.DataFrame([self.event(1, self.dates[5])])
        duplicate_dates = pd.concat([self.returns, self.returns.iloc[[5]]])
        with self.assertRaises(ValueError):
            analyze_events(duplicate_dates, events, window=1)

    def test_identical_event_paths_do_not_report_spurious_significance(self):
        returns, events = self.two_event_fixture()
        returns.loc[self.dates[8:11], "ret"] = returns.loc[
            self.dates[2:5], "ret"
        ].to_numpy()
        result = analyze_events(returns, events, window=1)
        summary = result.group_summary
        self.assertTrue(summary[["tstat", "pvalue", "ci_lower", "ci_upper"]].isna().all().all())
        endpoint = summarize_car(result.event_summary)
        self.assertEqual(endpoint.iloc[0].std_car, 0.0)
        self.assertTrue(endpoint[["tstat", "pvalue", "ci_lower", "ci_upper"]].isna().all().all())

    def test_duplicate_event_ids_raise(self):
        events = pd.DataFrame(
            [self.event(1, self.dates[3]), self.event(1, self.dates[9])]
        )
        with self.assertRaises(ValueError):
            analyze_events(self.returns, events, window=1)

    def test_duplicate_firm_date_pairs_raise(self):
        events = pd.DataFrame(
            [self.event(1, self.dates[3]), self.event(2, self.dates[3])]
        )
        with self.assertRaises(ValueError):
            analyze_events(self.returns, events, window=1)

    def test_unsorted_return_input_yields_same_results(self):
        returns, events = self.two_event_fixture()
        expected = analyze_events(returns, events, window=1)
        shuffled = analyze_events(returns.iloc[::-1], events, window=1)
        pd.testing.assert_frame_equal(shuffled.panel, expected.panel)
        pd.testing.assert_frame_equal(shuffled.group_summary, expected.group_summary)

    def test_distant_event_is_not_silently_shifted_into_sample(self):
        events = pd.DataFrame([self.event(1, "2023-01-01")])
        result = analyze_events(self.returns, events, window=1)
        self.assertEqual(len(result.event_summary), 0)
        self.assertEqual(set(result.exclusions.event_id), {1})

    def test_demo_is_deterministic_and_has_complete_default_windows(self):
        returns, events = make_demo_data(seed=42)
        again_returns, again_events = make_demo_data(seed=42)
        pd.testing.assert_frame_equal(returns, again_returns)
        pd.testing.assert_frame_equal(events, again_events)
        other_returns, _ = make_demo_data(seed=43)
        self.assertFalse(returns.equals(other_returns))
        result = analyze_events(returns, events)
        self.assertEqual(len(events), 32)
        self.assertEqual(len(result.exclusions), 0)
        self.assertEqual(len(result.event_summary), 32)
        self.assertEqual(len(result.panel), 32 * 11)
        self.assertEqual(set(result.group_summary.n), {16})
        self.assertEqual(set(result.group_summary.event_time), set(range(-5, 6)))


if __name__ == "__main__":
    unittest.main()
