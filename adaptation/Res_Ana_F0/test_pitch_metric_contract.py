"""Small deterministic checks for the H/O/R pitch-accuracy contract."""

import unittest
from pathlib import Path

import numpy as np

from pitch_analysis_shared import (
    F0Track,
    canonical_f0_cache_path,
    collect_pitch_pairs,
    evaluate_pitch,
    fold_scope_token,
    infer_monkey_id,
    select_pitch_metric,
)


class PitchMetricContractTest(unittest.TestCase):
    def setUp(self):
        time = np.arange(4, dtype=float)
        self.pairs = [{
            "key": (1, 1),
            "S": 1,
            "M": 1,
            "condition": "clean",
            "notes": [{
                "note_uid": "synthetic_note",
                "note_index": 0,
                "start_time_s": 0.0,
                "end_time_s": 4.0,
                "label_midi": 69,
                "label_hz": 440.0,
                "seen_group": "seen",
                "n_train_notes_same_pitch": 1,
                "train_pitch_probability": 0.1,
            }],
        }]
        self.tracks = {(1, 1): {
            # Original is voiced on frames 0–2; frame 2 has an octave error.
            "reference": F0Track(
                np.array([440.0, 440.0, 880.0, 440.0]),
                np.array([1.0, 1.0, 1.0, 0.0]),
                time,
                "synthetic",
            ),
            # Reconstructed is voiced on frames 0, 1, and 3.
            "reconstructed": F0Track(
                np.array([440.0, 880.0, 0.0, 440.0]),
                np.array([1.0, 1.0, 0.0, 1.0]),
                time,
                "synthetic",
            ),
        }}

    def test_canonical_cache_name_is_self_describing(self):
        result_root = Path(
            "/tmp/results_20260814_M160E_pixelLevel_100folds"
        )
        keys = [(1, 1), (1, 2)]
        path = canonical_f0_cache_path(
            result_root, "SWIPE", "SM(0.6)", keys
        )
        self.assertEqual(infer_monkey_id(result_root), "M160E")
        self.assertEqual(fold_scope_token(keys), "2folds-1e4f534a")
        self.assertEqual(path.parent.name, "f0_estimator_cache")
        self.assertEqual(
            path.name,
            "f0_tracks__monkey-M160E__condition-sm-0p6"
            "__roles-original-reconstructed__estimator-swipe"
            "__scope-2folds-1e4f534a.npz",
        )

    def test_frame_denominators_and_octave_folding(self):
        row = evaluate_pitch(
            self.pairs, self.tracks, "frame", thresholds=[0.5]
        ).iloc[0]
        self.assertEqual(row["n_human_voiced"], 4)
        self.assertEqual(row["n_original_voiced"], 3)
        self.assertAlmostEqual(row["rpa_orig"], 2 / 4)
        self.assertAlmostEqual(row["rpa_recon"], 2 / 4)
        # PDA is conditional on the two frames where O and R are both confident.
        self.assertAlmostEqual(row["pda"], 1 / 2)
        self.assertAlmostEqual(row["rca_orig"], 3 / 4)
        self.assertAlmostEqual(row["rca_recon"], 3 / 4)
        self.assertAlmostEqual(row["rca_pda"], 2 / 2)

    def test_selected_metric_does_not_change_other_rates(self):
        metrics = evaluate_pitch(
            self.pairs, self.tracks, "frame", thresholds=[0.5]
        )
        selected = select_pitch_metric(metrics, "rpa_recon").iloc[0]
        self.assertEqual(selected["pitch_accuracy_metric"], "rpa_recon")
        self.assertEqual(selected["n_accuracy_denominator"], 4)
        self.assertAlmostEqual(selected["pitch_accuracy"], 0.5)
        self.assertEqual(selected["n_candidate_voiced"], 3)
        self.assertAlmostEqual(selected["pda"], 1 / 2)

    def test_pda_selected_denominator_is_both_confident(self):
        metrics = evaluate_pitch(
            self.pairs, self.tracks, "frame", thresholds=[0.5]
        )
        selected = select_pitch_metric(metrics, "pda").iloc[0]
        self.assertEqual(selected["n_accuracy_denominator"], 2)
        self.assertEqual(selected["n_candidate_voiced"], 2)
        self.assertAlmostEqual(selected["candidate_voiced_recall"], 2 / 3)

    def test_pda_original_mask_uses_all_original_confident_frames(self):
        metrics = evaluate_pitch(
            self.pairs,
            self.tracks,
            "frame",
            thresholds=[0.5],
            pda_frame_mask="original",
        )
        selected = select_pitch_metric(metrics, "pda").iloc[0]
        # Original frames 0–2 enter the denominator. Frame 1 has an octave
        # error; frame 2 has no valid Reconstructed F0 and is counted wrong.
        self.assertEqual(selected["n_pda_evaluated"], 3)
        self.assertEqual(selected["n_accuracy_denominator"], 3)
        self.assertEqual(selected["n_pitch_correct"], 1)
        self.assertAlmostEqual(selected["pitch_accuracy"], 1 / 3)
        self.assertEqual(selected["pda_frame_mask"], "original")

        pairs = collect_pitch_pairs(
            self.pairs,
            self.tracks,
            "frame",
            threshold=0.5,
            metric="pda",
            pda_frame_mask="original",
        )
        # Continuous-error tables can only contain frames with a valid
        # Reconstructed F0, but they must not reapply its confidence threshold.
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs["pitch_correct"].tolist(), [True, False])
        self.assertTrue(pairs["pda_frame_mask"].eq("original").all())

    def test_pda_frame_mask_does_not_change_note_level(self):
        both = evaluate_pitch(
            self.pairs, self.tracks, "note", thresholds=[0.5],
            pda_frame_mask="both",
        ).iloc[0]
        original = evaluate_pitch(
            self.pairs, self.tracks, "note", thresholds=[0.5],
            pda_frame_mask="original",
        ).iloc[0]
        self.assertAlmostEqual(both["pda"], original["pda"])
        self.assertEqual(both["n_pda_evaluated"], original["n_pda_evaluated"])
        self.assertEqual(original["pda_frame_mask"], "not_applicable")

    def test_pda_frame_mask_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "pda_frame_mask"):
            evaluate_pitch(
                self.pairs, self.tracks, "frame", thresholds=[0.5],
                pda_frame_mask="reconstructed",
            )

    def test_pda_rejects_different_original_reconstructed_thresholds(self):
        with self.assertRaisesRegex(ValueError, "same confidence threshold"):
            evaluate_pitch(
                self.pairs,
                self.tracks,
                "frame",
                thresholds=[0.5],
                reconstructed_thresholds=[0.7],
            )

    def test_confidence_threshold_is_strictly_greater_than(self):
        row = evaluate_pitch(
            self.pairs, self.tracks, "frame", thresholds=[1.0]
        ).iloc[0]
        self.assertEqual(row["n_original_reconstructed_voiced"], 0)
        self.assertTrue(np.isnan(row["pda"]))

    def test_note_median_aggregation(self):
        rows = evaluate_pitch(
            self.pairs,
            self.tracks,
            "note",
            thresholds=[0.5, 1.0],
            note_aggregation="median",
            reconstructed_thresholds=[0.7, 0.9],
        )
        row = rows.iloc[0]
        self.assertAlmostEqual(row["rpa_orig"], 1.0)
        self.assertAlmostEqual(row["rpa_recon"], 1.0)
        # Note-level aggregation uses every valid F0 in the interval, including
        # Original frame 3 whose confidence is 0.0. Each track is aggregated
        # independently; both medians are therefore 440 Hz.
        self.assertAlmostEqual(row["original_representative_f0"], 440.0)
        self.assertAlmostEqual(row["reconstructed_representative_f0"], 440.0)
        self.assertEqual(row["n_original_reconstructed_voiced"], 1)
        self.assertAlmostEqual(row["pda"], 1.0)
        # Changing the confidence threshold must not change note-level results.
        for column in [
            "original_representative_f0",
            "reconstructed_representative_f0",
            "rpa_orig",
            "rpa_recon",
            "pda",
        ]:
            self.assertEqual(rows[column].nunique(dropna=False), 1)

        mean_rows = evaluate_pitch(
            self.pairs,
            self.tracks,
            "note",
            thresholds=[0.5, 1.0],
            note_aggregation="mean",
        )
        # The 440 Hz Original frame with confidence 0.0 is included: the mean
        # is (440 + 440 + 880 + 440) / 4 = 550 Hz at every threshold.
        np.testing.assert_allclose(
            mean_rows["original_representative_f0"], [550.0, 550.0]
        )
        np.testing.assert_allclose(
            mean_rows["reconstructed_representative_f0"],
            [1760.0 / 3.0, 1760.0 / 3.0],
        )

        pairs = collect_pitch_pairs(
            self.pairs,
            self.tracks,
            "note",
            threshold=1.0,
            note_aggregation="median",
            metric="pda",
        )
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs.iloc[0]["absolute_cents_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
