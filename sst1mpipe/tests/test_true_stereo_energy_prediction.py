import numpy as np
import pandas as pd

from sst1mpipe.reco.stereo_training import (
    build_true_stereo_feature_table,
    predict_true_stereo_energy,
)


class SumModel:
    def predict(self, x):
        return np.asarray(x).sum(axis=1)


def _sorted(df):
    return df.sort_values(["obs_id", "event_id"]).reset_index(drop=True)


def test_predict_true_stereo_energy_invariant_under_tel2_shuffle():
    features = ["f_a", "f_b"]
    feature_map = {
        "tel_001_features": features,
        "tel_002_features": ["tel_002_f_a", "tel_002_f_b"],
        "all_features": ["f_a", "f_b", "tel_002_f_a", "tel_002_f_b"],
    }

    tel1 = pd.DataFrame(
        {
            "obs_id": [1, 1, 2],
            "event_id": [10, 11, 20],
            "f_a": [1.0, 2.0, 3.0],
            "f_b": [10.0, 20.0, 30.0],
        }
    )

    tel2 = pd.DataFrame(
        {
            "obs_id": [1, 1, 2],
            "event_id": [10, 11, 20],
            "f_a": [100.0, 200.0, 300.0],
            "f_b": [1000.0, 2000.0, 3000.0],
        }
    )
    tel2_shuffled = tel2.sample(frac=1.0, random_state=7).reset_index(drop=True)

    pred_a = _sorted(
        predict_true_stereo_energy(tel1, tel2, model=SumModel(), feature_map=feature_map)
    )
    pred_b = _sorted(
        predict_true_stereo_energy(tel1, tel2_shuffled, model=SumModel(), feature_map=feature_map)
    )

    pd.testing.assert_frame_equal(pred_a, pred_b, check_exact=False, atol=1e-12)


def test_predict_true_stereo_energy_drops_missing_rows_explicitly():
    feature_map = {
        "tel_001_features": ["f_a", "f_b"],
        "tel_002_features": ["tel_002_f_a", "tel_002_f_b"],
        "all_features": ["f_a", "f_b", "tel_002_f_a", "tel_002_f_b"],
    }

    tel1 = pd.DataFrame(
        {
            "obs_id": [3, 3],
            "event_id": [100, 101],
            "f_a": [1.0, np.nan],
            "f_b": [5.0, 6.0],
        }
    )
    tel2 = pd.DataFrame(
        {
            "obs_id": [3, 3],
            "event_id": [100, 101],
            "f_a": [11.0, 12.0],
            "f_b": [15.0, 16.0],
        }
    )

    pred = predict_true_stereo_energy(tel1, tel2, model=SumModel(), feature_map=feature_map)

    assert len(pred) == 1
    assert int(pred.iloc[0]["event_id"]) == 100


def test_build_true_stereo_feature_table_gamma_proton_symmetry():
    features = ["f_a"]

    gamma_tel1 = pd.DataFrame(
        {
            "obs_id": [8, 8, 8],
            "event_id": [1, 2, 3],
            "f_a": [1.0, 2.0, 3.0],
        }
    )
    gamma_tel2 = pd.DataFrame(
        {
            "obs_id": [8, 8, 8],
            "event_id": [3, 1, 2],
            "f_a": [30.0, 10.0, 20.0],
        }
    )

    proton_tel1 = pd.DataFrame(
        {
            "obs_id": [9, 9, 9],
            "event_id": [4, 5, 6],
            "f_a": [4.0, 5.0, 6.0],
        }
    )
    proton_tel2 = pd.DataFrame(
        {
            "obs_id": [9, 9],
            "event_id": [4, 6],
            "f_a": [40.0, 60.0],
        }
    )

    gamma_combined = build_true_stereo_feature_table(gamma_tel1, gamma_tel2, base_features=features)
    proton_combined = build_true_stereo_feature_table(proton_tel1, proton_tel2, base_features=features)

    assert len(gamma_combined) == 3
    assert len(proton_combined) == 2
    assert set(zip(proton_combined["obs_id"], proton_combined["event_id"])) == {(9, 4), (9, 6)}
