import logging
from typing import List, Tuple

import numpy as np
import pandas as pd


def make_true_stereo_event_table(
    data_tel1: pd.DataFrame,
    data_tel2: pd.DataFrame,
    base_features: List[str],
    join_keys: Tuple[str, str] = ("obs_id", "event_id"),
    include_target: bool = False,
    target_col: str = "log_true_energy",
    strict: bool = False,
    truth_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a key-aligned TRUE STEREO event table using explicit coincidence merge.

    The returned table contains join keys, tel1 features, and tel2 features
    prefixed with ``tel_002_``. When ``include_target=True``, it also validates
    target consistency between telescopes and produces ``target_col``.
    """

    join_keys = list(join_keys)
    features_tel1 = list(base_features)
    features_tel2 = [f"tel_002_{feat}" for feat in base_features]

    required_tel1 = join_keys + features_tel1
    required_tel2 = join_keys + list(base_features)

    extra_required_tel1 = [target_col] if include_target else []
    extra_required_tel2 = [target_col] if include_target else []

    missing_tel1 = [col for col in required_tel1 + extra_required_tel1 if col not in data_tel1.columns]
    missing_tel2 = [col for col in required_tel2 + extra_required_tel2 if col not in data_tel2.columns]
    if missing_tel1 or missing_tel2:
        raise ValueError(
            "Missing required columns for TRUE STEREO feature table "
            f"(tel1 missing: {missing_tel1}, tel2 missing: {missing_tel2})."
        )

    tel1_subset = data_tel1[required_tel1 + extra_required_tel1].copy()
    tel2_subset = data_tel2[required_tel2 + extra_required_tel2].copy()
    tel2_subset = tel2_subset.rename(
        columns={
            **{feat: f"tel_002_{feat}" for feat in base_features},
            **({target_col: f"{target_col}_tel2"} if include_target else {}),
        }
    )
    if include_target:
        tel1_subset = tel1_subset.rename(columns={target_col: f"{target_col}_tel1"})

    merged = pd.merge(tel1_subset, tel2_subset, on=join_keys, how="inner")

    if not include_target:
        return merged

    if len(merged) == 0:
        merged[target_col] = []
        return merged

    delta = np.abs(merged[f"{target_col}_tel1"] - merged[f"{target_col}_tel2"])
    bad_truth = delta > truth_tolerance
    n_bad_truth = int(bad_truth.sum())
    if n_bad_truth:
        logging.warning(
            "TRUE STEREO builder: inconsistent truth rows = %d (tolerance=%g)",
            n_bad_truth,
            truth_tolerance,
        )
        if strict:
            raise ValueError(
                "Inconsistent true energy between telescopes: "
                f"{n_bad_truth} rows exceed tolerance {truth_tolerance}."
            )
        merged = merged.loc[~bad_truth].copy()

    logging.info(
        "TRUE STEREO builder: rows removed by inconsistent truth = %d",
        n_bad_truth,
    )

    merged[target_col] = merged[f"{target_col}_tel1"]

    required_for_training = features_tel1 + features_tel2 + [target_col]
    valid_mask = merged[required_for_training].notna().all(axis=1)
    missing_rows = int((~valid_mask).sum())
    if missing_rows:
        merged = merged.loc[valid_mask].copy()

    logging.info(
        "TRUE STEREO builder: rows removed by missing features/target = %d",
        missing_rows,
    )
    logging.info("TRUE STEREO builder: final training rows = %d", len(merged))

    return merged


def build_true_stereo_feature_table(
    data_tel1: pd.DataFrame,
    data_tel2: pd.DataFrame,
    base_features: List[str],
    join_keys: Tuple[str, str] = ("obs_id", "event_id"),
) -> pd.DataFrame:
    """Compatibility wrapper for key-aligned TRUE STEREO feature table."""

    return make_true_stereo_event_table(
        data_tel1=data_tel1,
        data_tel2=data_tel2,
        base_features=base_features,
        join_keys=join_keys,
        include_target=False,
    )


def build_true_stereo_training_table(
    data_tel1: pd.DataFrame,
    data_tel2: pd.DataFrame,
    base_features: List[str],
    join_keys: Tuple[str, str] = ("obs_id", "event_id"),
    target_col: str = "log_true_energy",
    strict: bool = False,
    truth_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a coincidence-based TRUE STEREO training table from per-telescope inputs.

    Events are paired by explicit event identity (join keys), not row order.
    Truth columns from both telescopes are compared and inconsistent rows are
    dropped (or raised in strict mode).
    """

    join_keys = list(join_keys)

    logging.info("TRUE STEREO builder: tel1 input rows = %d", len(data_tel1))
    logging.info("TRUE STEREO builder: tel2 input rows = %d", len(data_tel2))

    merged = make_true_stereo_event_table(
        data_tel1=data_tel1,
        data_tel2=data_tel2,
        base_features=base_features,
        join_keys=tuple(join_keys),
        include_target=True,
        target_col=target_col,
        strict=strict,
        truth_tolerance=truth_tolerance,
    )

    logging.info("TRUE STEREO builder: coincident rows after merge = %d", len(merged))

    return merged


def predict_true_stereo_energy_from_tables(
    data_tel1: pd.DataFrame,
    data_tel2: pd.DataFrame,
    model,
    feature_map: dict,
    join_keys: Tuple[str, str] = ("obs_id", "event_id"),
) -> pd.DataFrame:
    """
    Predict event-level TRUE STEREO energy on coincident telescope rows.

    Returns a DataFrame with join keys and ``log_reco_energy``.
    """

    join_keys = list(join_keys)
    features_tel1 = list(feature_map["tel_001_features"])
    features_tel2 = list(feature_map["tel_002_features"])
    all_features = list(feature_map["all_features"])

    combined = make_true_stereo_event_table(
        data_tel1=data_tel1,
        data_tel2=data_tel2,
        base_features=features_tel1,
        join_keys=tuple(join_keys),
    )

    n_coincident = len(combined)
    if n_coincident == 0:
        logging.warning("TRUE STEREO predictor: no coincident rows after merge.")
        return pd.DataFrame(columns=join_keys + ["log_reco_energy"])

    required_features = features_tel1 + features_tel2
    valid_mask = combined[required_features].notna().all(axis=1)
    missing_rows = int((~valid_mask).sum())
    if missing_rows:
        combined = combined.loc[valid_mask].copy()

    if len(combined) == 0:
        logging.warning(
            "TRUE STEREO predictor: all coincident rows removed by missing features (%d rows).",
            missing_rows,
        )
        return pd.DataFrame(columns=join_keys + ["log_reco_energy"])

    logging.info(
        "TRUE STEREO predictor: coincident rows=%d, removed_missing=%d, predicted=%d",
        n_coincident,
        missing_rows,
        len(combined),
    )

    combined["log_reco_energy"] = model.predict(combined[all_features])
    return combined[join_keys + ["log_reco_energy"]]


def predict_true_stereo_energy(
    data_tel1: pd.DataFrame,
    data_tel2: pd.DataFrame,
    model,
    feature_map: dict,
    join_keys: Tuple[str, str] = ("obs_id", "event_id"),
) -> pd.DataFrame:
    """Compatibility wrapper for TRUE STEREO event-level energy prediction."""

    return predict_true_stereo_energy_from_tables(
        data_tel1=data_tel1,
        data_tel2=data_tel2,
        model=model,
        feature_map=feature_map,
        join_keys=join_keys,
    )