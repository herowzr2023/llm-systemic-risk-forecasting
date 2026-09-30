"""Synthetic contract tests; never fit a production estimator or run a neural model."""
from __future__ import annotations

import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from libs.model_comparison import lead_specific_comparison as comparison


ROOT = Path(__file__).resolve().parents[1]


class DirectComparisonTests(unittest.TestCase):
    def setUp(self):
        # The dates are synthetic sessions: the fixture calendar is the authority.
        self.calendar = pd.date_range("2020-01-01", periods=60, freq="B")
        self.data = pd.DataFrame({
            "date": self.calendar,
            "SR": np.arange(60, dtype=float),
            "feature": np.arange(60, dtype=float) * 2,
        })
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def prepare(self, data=None, lead=5):
        return comparison._prepare_samples(
            self.data if data is None else data, lead, 3, .7,
            "minmax", "date", "SR", self.calendar,
        )

    def archive(self, prepared, lead=5, aliases=False):
        test = slice(prepared.train_count, None)
        origins = prepared.first_target_indices[test] - 1
        targets = prepared.target_indices[test]
        return pd.DataFrame({
            "origin" if aliases else "origin_date": prepared.dates.iloc[origins].to_numpy(),
            "target_date" if aliases else "date": prepared.dates.iloc[targets].to_numpy(),
            "horizon" if aliases else "lead": lead,
            "true" if aliases else "actual_SR": prepared.raw_target[targets],
            "pred" if aliases else "predicted_SR": prepared.raw_target[targets] + .25,
            "forecast_mode": "direct_terminal_target" if aliases else "direct",
        })

    def load(self, frame, data=None, lead=5, provenance="direct_terminal_target"):
        path = self.directory / "archive.csv"
        frame.to_csv(path, index=False)
        return comparison.load_informer_lead(
            path, self.data if data is None else data, lead, "date", "SR",
            self.calendar, provenance,
        )

    def test_scalar_calendar_terminal_labels_and_training_embargo(self):
        prepared = self.prepare()
        origins = prepared.first_target_indices - 1
        self.assertEqual(prepared.fit_end, 42)
        self.assertEqual(prepared.targets.shape, (len(origins), 1))
        np.testing.assert_array_equal(prepared.target_indices, origins + 5)
        self.assertTrue((prepared.target_indices[:prepared.train_count] < 42).all())
        self.assertTrue((origins[prepared.train_count:] >= 41).all())
        self.assertFalse(np.isin(np.arange(37, 41), origins).any())
        expected = prepared.normalized.SR.to_numpy()[prepared.target_indices]
        np.testing.assert_allclose(prepared.targets[:, 0], expected)
        for sequence, origin in zip(prepared.sequences, origins):
            np.testing.assert_allclose(sequence[:, 0], prepared.normalized.SR.iloc[origin-2:origin+1])

    def test_future_extremes_do_not_fit_scalers(self):
        changed = self.data.copy()
        changed.loc[42:, ["SR", "feature"]] = 1e9
        before, after = self.prepare(), self.prepare(changed)
        np.testing.assert_array_equal(before.target_scaler.data_min_, after.target_scaler.data_min_)
        np.testing.assert_array_equal(before.target_scaler.data_max_, after.target_scaler.data_max_)
        np.testing.assert_array_equal(before.sequences[:before.train_count], after.sequences[:after.train_count])
        np.testing.assert_array_equal(before.targets[:before.train_count], after.targets[:after.train_count])
        self.assertGreater(after.normalized.feature.iloc[42], 1)

    def test_gap_uses_calendar_positions_and_rejects_noncontinuous_history(self):
        data = self.data.drop(index=44).reset_index(drop=True)
        prepared = self.prepare(data)
        positions = self.calendar.get_indexer(prepared.dates)
        origins = prepared.first_target_indices - 1
        np.testing.assert_array_equal(positions[prepared.target_indices] - positions[origins], 5)
        # Origin session 41 targets session 46, which is four retained rows later.
        sample = int(np.flatnonzero(positions[origins] == 41)[0])
        self.assertEqual(prepared.target_indices[sample] - origins[sample], 4)
        self.assertFalse(np.isin([45, 46], positions[origins]).any())
        archive = self.archive(prepared)
        self.load(archive, data)
        first = archive.index[archive.origin_date == self.calendar[41]][0]
        archive.loc[first, "date"] = data.date.iloc[origins[sample] + 5]
        archive.loc[first, "actual_SR"] = data.SR.iloc[origins[sample] + 5]
        with self.assertRaisesRegex(ValueError, "calendar"):
            self.load(archive, data)

    def test_one_session_mode_keeps_scalar_contract(self):
        prepared = self.prepare(lead=1)
        np.testing.assert_array_equal(prepared.target_indices, prepared.first_target_indices)
        actual = prepared.targets[prepared.train_count:]
        frame = comparison._prediction_frame(prepared, actual, 1, "GBR_pred", "date")
        np.testing.assert_allclose(frame.GBR_pred, prepared.raw_target[prepared.target_indices[prepared.train_count:]], rtol=1e-6)
        self.load(self.archive(prepared, lead=1), lead=1)

    def test_prediction_frame_rejects_joint_output(self):
        prepared = self.prepare()
        count = len(prepared.targets) - prepared.train_count
        with self.assertRaisesRegex(ValueError, "scalar shape"):
            comparison._prediction_frame(prepared, np.zeros((count, 5)), 5, "GBR_pred", "date")
        frame = comparison._prediction_frame(prepared, np.zeros(count), 5, "GBR_pred", "date")
        np.testing.assert_array_equal(frame.date, prepared.dates.iloc[prepared.target_indices[prepared.train_count:]])

    def test_direct_archive_accepts_both_supported_column_conventions(self):
        prepared = self.prepare()
        for aliases in [False, True]:
            with self.subTest(aliases=aliases):
                loaded = self.load(self.archive(prepared, aliases=aliases))
                self.assertEqual(len(loaded), len(prepared.targets) - prepared.train_count)
                self.assertEqual(loaded.columns.tolist(), ["origin_date", "date", "lead", "TRUE", "Informer_pred"])

    def test_legacy_archive_rejected_even_with_direct_config_provenance(self):
        archive = self.archive(self.prepare())
        for missing in ["forecast_mode", "origin_date", "lead"]:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.load(archive.drop(columns=missing))
        with self.assertRaises(ValueError):
            self.load(archive, provenance="legacy_joint")
        archive["forecast_mode"] = "joint"
        with self.assertRaises(ValueError):
            self.load(archive)

    def test_archive_conflicts_actuals_and_duplicates_rejected(self):
        archive = self.archive(self.prepare())
        for column, value in [("lead", 1), ("forecast_mode", "joint"),
                              ("target_mode", "joint"), ("provenance", "legacy"),
                              ("actual_SR", -100), ("predicted_SR", np.inf),
                              ("target_date", "2020-01-01"),
                              ("origin", "2020-01-01")]:
            altered = archive.copy()
            altered.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.load(altered)
        with self.assertRaises(ValueError):
            self.load(pd.concat([archive, archive.iloc[:1]], ignore_index=True))

    def test_gbr_adapter_passes_one_dimensional_labels_without_real_fit(self):
        prepared = self.prepare()
        calls = {}

        class CaptureEstimator:
            def __init__(self, **kwargs):
                calls["settings"] = kwargs

            def fit(self, x, y):
                calls["x"], calls["y"] = x.copy(), y.copy()

            def predict(self, x):
                calls["test"] = x.copy()
                return np.zeros(len(x))

        with patch.object(comparison, "GradientBoostingRegressor", CaptureEstimator):
            frame = comparison._gbr_predictions(prepared, 5, "date", {}, 2024)
        self.assertEqual(calls["y"].shape, (prepared.train_count,))
        np.testing.assert_array_equal(calls["y"], prepared.targets[:prepared.train_count, 0])
        self.assertEqual(calls["x"].shape, (prepared.train_count, 6))
        self.assertEqual(len(frame), len(prepared.targets) - prepared.train_count)
        self.assertEqual(calls["settings"], dict(n_estimators=100, learning_rate=.1, max_depth=3, random_state=2024))

    def test_neural_heads_static_output_dimension_without_training_or_forward(self):
        for function in [comparison._lstm_predictions, comparison._patchtst_predictions]:
            tree = ast.parse(inspect.getsource(function))
            linears = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "Linear"]
            # Last registered Linear in each adapter is its scalar terminal head.
            self.assertIsInstance(linears[-1].args[1], ast.Constant)
            self.assertEqual(linears[-1].args[1].value, 1)

    def test_simple_parameters_train_only_and_ets_assimilates_skipped_origins(self):
        prepared = self.prepare(self.data.drop(index=44).reset_index(drop=True))
        with patch.object(comparison, "_fit_ar1", return_value=(1., .5)) as ar_fit, \
             patch.object(comparison, "_fit_ets", return_value=(.5, 7.)) as ets_fit:
            frames = comparison._simple_predictions(prepared, {"Persistence", "AR(1)", "ETS"}, 5, "date")
        for call in [ar_fit, ets_fit]:
            np.testing.assert_array_equal(call.call_args.args[0], prepared.raw_target[:prepared.fit_end])
        level, seen = 7., prepared.fit_end - 1
        for row, first_target in enumerate(prepared.first_target_indices[prepared.train_count:]):
            for observed in range(seen+1, first_target):
                level = .5 * prepared.raw_target[observed] + .5 * level
            seen = first_target - 1
            self.assertEqual(frames["ETS"].ETS_pred.iloc[row], level)
            self.assertEqual(frames["Persistence"].Persistence_pred.iloc[row], prepared.raw_target[first_target-1])

    def test_active_runner_rejects_pretest_archives_before_model_fit(self):
        prepared = self.prepare()
        archive = self.archive(prepared)
        archive.loc[0, ["origin_date", "date", "actual_SR"]] = [self.calendar[10], self.calendar[15], 15.]
        archive_path = self.directory / "archive.csv"
        archive.to_csv(archive_path, index=False)
        data_path, calendar_path = self.directory / "data.csv", self.directory / "calendar.csv"
        self.data.to_csv(data_path, index=False)
        pd.DataFrame({"date": self.calendar}).to_csv(calendar_path, index=False)
        config = dict(date_col="date", target_col="SR", forecast_mode="direct_terminal_target",
                      baseline_provenance="direct_terminal_target", calendar_csv=str(calendar_path))
        with patch.object(comparison, "_set_seed"), \
             patch.object(comparison, "_gbr_predictions", side_effect=AssertionError("training forbidden")), \
             self.assertRaisesRegex(ValueError, "test cutoff"):
            comparison.run_lead_specific_comparison(archive_path, data_path, self.directory / "out.csv", 5, 3, ["GBR"], config)
        self.assertFalse((self.directory / "out.csv").exists())

    def test_active_export_has_direct_metadata_without_any_training(self):
        prepared = self.prepare()
        archive_path = self.directory / "archive.csv"
        self.archive(prepared).to_csv(archive_path, index=False)
        data_path, calendar_path = self.directory / "data.csv", self.directory / "calendar.csv"
        self.data.to_csv(data_path, index=False)
        pd.DataFrame({"date": self.calendar}).to_csv(calendar_path, index=False)
        config = dict(date_col="date", target_col="SR", forecast_mode="direct_terminal_target",
                      baseline_provenance="direct_terminal_target", calendar_csv=str(calendar_path))
        output = self.directory / "direct.csv"
        with patch.object(comparison, "_set_seed"):
            comparison.run_lead_specific_comparison(archive_path, data_path, output, 5, 3, [], config)
        result = pd.read_csv(output)
        self.assertTrue(result.forecast_mode.eq("direct").all())
        self.assertTrue(result.horizon.eq(5).all())
        self.assertTrue(result.date.eq(result.target_date).all())

    def test_unsupported_lead_and_invalid_dates_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare(lead=7)
        with self.assertRaises(ValueError):
            self.load(self.archive(self.prepare()), lead=7)
        for data in [pd.concat([self.data, self.data.iloc[:1]], ignore_index=True), self.data.iloc[::-1]]:
            with self.assertRaises(ValueError):
                self.prepare(data)

    def test_active_config_has_only_direct_archive_and_output_paths(self):
        config = json.loads((ROOT / "configs/07_model_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(config["forecast_mode"], "direct_terminal_target")
        self.assertEqual(config["baseline_provenance"], "direct_terminal_target")
        self.assertEqual(config["train_ratio"], .7)
        for job in config["jobs"]:
            lead = job["lead"]
            self.assertEqual(job["baseline_csv"], f"data/intermediate/informer_direct_{lead}d.csv")
            self.assertEqual(job["output_csv"], f"results/direct/complete_base_{lead}d_predictions.csv")


if __name__ == "__main__":
    unittest.main()
