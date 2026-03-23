import json
import os

import joblib
import numpy as np
import pandas as pd

from sst1mpipe.reco.reco import apply_true_stereo_energy_model


class SumModel:
    def predict(self, x):
        arr = np.asarray(x)
        return arr.sum(axis=1)


def _sort(df):
    return df.sort_values(["obs_id", "event_id"]).reset_index(drop=True)


def test_apply_true_stereo_energy_model_is_order_invariant(tmp_path):
    models_dir = tmp_path

    feature_map = {
        "tel_001_features": ["f_a", "f_b"],
        "tel_002_features": ["tel_002_f_a", "tel_002_f_b"],
        "all_features": ["f_a", "f_b", "tel_002_f_a", "tel_002_f_b"],
        "tel_role_policy": "sorted_by_tel_id",
        "expected_n_telescopes": 2,
    }

    joblib.dump(SumModel(), os.path.join(models_dir, "reg_energy_stereo.sav"))
    joblib.dump(feature_map, os.path.join(models_dir, "reg_energy_stereo_features.pkl"))

    params = pd.DataFrame(
        {
            "obs_id": [1, 1, 1, 1],
            "event_id": [10, 10, 11, 11],
            "tel_id": [1, 2, 1, 2],
            "f_a": [1.0, 100.0, 2.0, 200.0],
            "f_b": [10.0, 1000.0, 20.0, 2000.0],
        }
    )

    shuffled = params.sample(frac=1.0, random_state=11).reset_index(drop=True)

    a = _sort(apply_true_stereo_energy_model(params, models_dir=str(models_dir), config={"analysis": {}}))
    b = _sort(apply_true_stereo_energy_model(shuffled, models_dir=str(models_dir), config={"analysis": {}}))

    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_apply_true_stereo_energy_model_respects_requested_tel_ids(tmp_path):
    models_dir = tmp_path

    feature_map = {
        "tel_001_features": ["f_a"],
        "tel_002_features": ["tel_002_f_a"],
        "all_features": ["f_a", "tel_002_f_a"],
    }

    joblib.dump(SumModel(), os.path.join(models_dir, "reg_energy_stereo.sav"))
    joblib.dump(feature_map, os.path.join(models_dir, "reg_energy_stereo_features.pkl"))

    params = pd.DataFrame(
        {
            "obs_id": [1, 1],
            "event_id": [10, 10],
            "tel_id": [1, 2],
            "f_a": [1.0, 2.0],
        }
    )

    out = apply_true_stereo_energy_model(
        params,
        models_dir=str(models_dir),
        config={"analysis": {"true_stereo_energy_tel_ids": [21, 22]}},
    )
    assert out is None


def test_default_configs_have_explicit_true_stereo_switch():
    mc_cfg = json.load(open("sst1mpipe/data/sst1mpipe_mc_config.json", "r", encoding="utf-8"))
    data_cfg = json.load(open("sst1mpipe/data/sst1mpipe_data_config.json", "r", encoding="utf-8"))

    assert "use_true_stereo_energy" in mc_cfg["analysis"]
    assert "use_true_stereo_energy" in data_cfg["analysis"]
