import numpy as np
import pandas as pd
import pytest

from sst1mpipe.reco.stereo_training import build_true_stereo_training_table


def test_build_true_stereo_training_table_is_order_independent():
    base_features = ["f_a", "f_b"]

    tel1 = pd.DataFrame(
        {
            "obs_id": [1, 1, 1, 2],
            "event_id": [10, 11, 12, 20],
            "log_true_energy": [0.1, 0.2, 0.3, 0.4],
            "f_a": [10.0, 11.0, 12.0, 20.0],
            "f_b": [100.0, 110.0, np.nan, 200.0],
        }
    )

    # Same physical events, but deliberately shuffled in tel2.
    tel2 = pd.DataFrame(
        {
            "obs_id": [2, 1, 1, 1, 9],
            "event_id": [20, 12, 10, 11, 99],
            "log_true_energy": [0.4, 0.3, 0.1, 0.2, 9.9],
            "f_a": [200.0, 120.0, 100.0, 110.0, 999.0],
            "f_b": [2000.0, 1200.0, 1000.0, 1100.0, 9999.0],
        }
    )

    merged = build_true_stereo_training_table(
        data_tel1=tel1,
        data_tel2=tel2,
        base_features=base_features,
    )

    # event_id=12 is dropped because f_b is NaN in tel1; event_id=99 is not coincident.
    assert len(merged) == 3
    assert set(zip(merged["obs_id"], merged["event_id"])) == {(1, 10), (1, 11), (2, 20)}

    # Target is retained from tel1 after truth-consistency validation.
    expected_target = {(1, 10): 0.1, (1, 11): 0.2, (2, 20): 0.4}
    for _, row in merged.iterrows():
        key = (row["obs_id"], row["event_id"])
        assert row["log_true_energy"] == pytest.approx(expected_target[key])

    # Tel2 features must be aligned by event keys, not by row order.
    expected_tel2_fa = {(1, 10): 100.0, (1, 11): 110.0, (2, 20): 200.0}
    for _, row in merged.iterrows():
        key = (row["obs_id"], row["event_id"])
        assert row["tel_002_f_a"] == pytest.approx(expected_tel2_fa[key])


def test_build_true_stereo_training_table_truth_mismatch_drop_or_raise():
    base_features = ["f_a"]

    tel1 = pd.DataFrame(
        {
            "obs_id": [1, 1],
            "event_id": [10, 11],
            "log_true_energy": [0.1, 0.2],
            "f_a": [10.0, 11.0],
        }
    )
    tel2 = pd.DataFrame(
        {
            "obs_id": [1, 1],
            "event_id": [10, 11],
            "log_true_energy": [0.1000000001, 0.25],
            "f_a": [100.0, 110.0],
        }
    )

    merged = build_true_stereo_training_table(
        data_tel1=tel1,
        data_tel2=tel2,
        base_features=base_features,
        strict=False,
    )
    assert len(merged) == 1
    assert int(merged.iloc[0]["event_id"]) == 10

    with pytest.raises(ValueError):
        build_true_stereo_training_table(
            data_tel1=tel1,
            data_tel2=tel2,
            base_features=base_features,
            strict=True,
        )


def test_build_true_stereo_training_table_nan_in_middle_keeps_alignment():
    base_features = ["f_a", "f_b"]

    tel1 = pd.DataFrame(
        {
            "obs_id": [7, 7, 7],
            "event_id": [100, 101, 102],
            "log_true_energy": [1.0, 2.0, 3.0],
            "f_a": [10.0, 11.0, 12.0],
            "f_b": [100.0, np.nan, 120.0],
        }
    )
    # Deliberately shuffled to ensure masking happens after key-based merge.
    tel2 = pd.DataFrame(
        {
            "obs_id": [7, 7, 7],
            "event_id": [102, 100, 101],
            "log_true_energy": [3.0, 1.0, 2.0],
            "f_a": [212.0, 210.0, 211.0],
            "f_b": [1210.0, 1010.0, 1110.0],
        }
    )

    merged = build_true_stereo_training_table(
        data_tel1=tel1,
        data_tel2=tel2,
        base_features=base_features,
    )

    # event_id=101 is removed due to NaN in tel1.f_b; targets stay matched by key.
    assert set(zip(merged["obs_id"], merged["event_id"])) == {(7, 100), (7, 102)}

    expected_target = {(7, 100): 1.0, (7, 102): 3.0}
    expected_tel2_f_a = {(7, 100): 210.0, (7, 102): 212.0}
    for _, row in merged.iterrows():
        key = (row["obs_id"], row["event_id"])
        assert row["log_true_energy"] == pytest.approx(expected_target[key])
        assert row["tel_002_f_a"] == pytest.approx(expected_tel2_f_a[key])


def test_build_true_stereo_training_table_missing_coincidence_keeps_only_overlap():
    base_features = ["f_a"]

    tel1 = pd.DataFrame(
        {
            "obs_id": [5] * 10,
            "event_id": list(range(10)),
            "log_true_energy": np.linspace(0.1, 1.0, 10),
            "f_a": np.linspace(10.0, 19.0, 10),
        }
    )
    # Remove one event (10%) from tel2 to simulate missing coincidence.
    tel2 = tel1[tel1["event_id"] != 3].copy()
    tel2["f_a"] = tel2["f_a"] + 100.0

    merged = build_true_stereo_training_table(
        data_tel1=tel1,
        data_tel2=tel2,
        base_features=base_features,
    )

    assert len(merged) == 9
    assert (5, 3) not in set(zip(merged["obs_id"], merged["event_id"]))
    assert set(zip(merged["obs_id"], merged["event_id"])) == {
        (5, 0), (5, 1), (5, 2), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9)
    }
