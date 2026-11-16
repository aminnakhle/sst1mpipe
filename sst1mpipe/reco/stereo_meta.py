import logging
import numpy as np
import pandas as pd


_DEFAULT_JOIN_KEYS = ("obs_id", "event_id")


def _validate_required_columns(data, required_columns, label):
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise KeyError("Missing required columns in {}: {}".format(label, missing))


def _check_consistency(value_tel1, value_tel2, tolerance=1e-6):
    both_nan = value_tel1.isna() & value_tel2.isna()
    both_finite_close = np.isfinite(value_tel1) & np.isfinite(value_tel2) & (
        np.abs(value_tel1 - value_tel2) <= tolerance
    )
    return both_nan | both_finite_close


def build_stereo_energy_meta_event_table(
    data_tel1,
    data_tel2,
    feature_names,
    join_keys=("obs_id", "event_id"),
    include_target=False,
    target_col="log_true_energy",
    weights_col="camera_frame_hillas_intensity",
    strict=False,
    truth_tolerance=1e-6,
):
    """
    Build event-level stereo table from per-telescope tables.
    """

    join_keys = tuple(join_keys)
    base_required = {
        "log_reco_energy",
        weights_col,
        "leakage_intensity_width_2",
        "HillasReconstructor_tel_impact_distance",
        "HillasReconstructor_h_max",
    }

    for feature in feature_names:
        if feature.endswith("_tel1"):
            base_required.add(feature[:-5])
        elif feature.endswith("_tel2"):
            base_required.add(feature[:-5])

    columns_tel = list(join_keys) + sorted(base_required)
    if include_target:
        columns_tel = columns_tel + [target_col]

    _validate_required_columns(data_tel1, columns_tel, "tel1 table")
    _validate_required_columns(data_tel2, columns_tel, "tel2 table")

    tel1 = data_tel1[columns_tel].copy()
    tel2 = data_tel2[columns_tel].copy()

    rename_tel1 = {c: c + "_tel1" for c in tel1.columns if c not in join_keys}
    rename_tel2 = {c: c + "_tel2" for c in tel2.columns if c not in join_keys}
    tel1 = tel1.rename(columns=rename_tel1)
    tel2 = tel2.rename(columns=rename_tel2)

    event_table = pd.merge(tel1, tel2, on=list(join_keys), how="inner")
    if event_table.empty:
        logging.warning("No stereo coincident events after tel1/tel2 merge on %s", join_keys)
        logging.warning("Tel1 unique events: %d, Tel2 unique events: %d", len(tel1[list(join_keys)].drop_duplicates()), len(tel2[list(join_keys)].drop_duplicates()))
        return event_table

    hmax_tel1 = "HillasReconstructor_h_max_tel1"
    hmax_tel2 = "HillasReconstructor_h_max_tel2"
    if hmax_tel1 in event_table.columns and hmax_tel2 in event_table.columns:
        hmax_consistent = _check_consistency(
            event_table[hmax_tel1], event_table[hmax_tel2], tolerance=truth_tolerance
        )
        n_inconsistent_hmax = int((~hmax_consistent).sum())
        if n_inconsistent_hmax > 0:
            msg = (
                "Found {} events with inconsistent HillasReconstructor_h_max between telescopes"
                .format(n_inconsistent_hmax)
            )
            if strict:
                raise ValueError(msg)
            logging.warning("%s. Using tel1 values.", msg)
        event_table["HillasReconstructor_h_max"] = event_table[hmax_tel1]

    e1 = event_table["log_reco_energy_tel1"]
    e2 = event_table["log_reco_energy_tel2"]
    w1 = event_table[weights_col + "_tel1"]
    w2 = event_table[weights_col + "_tel2"]
    event_table["log_reco_energy_mean"] = (e1 + e2) / 2.0
    denom = w1 + w2
    event_table["log_reco_energy_weighted"] = np.where(
        denom > 0,
        (e1 * w1 + e2 * w2) / denom,
        np.nan,
    )
    event_table["log_reco_energy_absdiff"] = np.abs(e1 - e2)

    if include_target:
        t1 = target_col + "_tel1"
        t2 = target_col + "_tel2"
        target_consistent = _check_consistency(
            event_table[t1], event_table[t2], tolerance=truth_tolerance
        )
        n_inconsistent_target = int((~target_consistent).sum())
        if n_inconsistent_target > 0:
            msg = "Dropping {} events with inconsistent {} between telescopes".format(
                n_inconsistent_target, target_col
            )
            if strict:
                raise ValueError(msg)
            logging.warning(msg)
            event_table = event_table[target_consistent].copy()
        event_table[target_col] = (event_table[t1] + event_table[t2]) / 2.0

    required_for_output = list(join_keys) + list(feature_names)
    if include_target:
        required_for_output.append(target_col)

    _validate_required_columns(event_table, required_for_output, "stereo meta event table")

    before_drop = len(event_table)
    event_table = event_table.dropna(subset=required_for_output)
    n_dropped = before_drop - len(event_table)
    if n_dropped > 0:
        logging.warning(
            "Dropped %d stereo meta events with missing required values (%d remaining). "
            "Required columns: %s",
            n_dropped,
            len(event_table),
            required_for_output,
        )

    return event_table


def build_stereo_energy_meta_training_table_from_oof(
    data_tel1,
    data_tel2,
    feature_names,
    join_keys=("obs_id", "event_id"),
    target_col="log_true_energy",
    strict=False,
    truth_tolerance=1e-6,
):
    """
    Build event-level stage-2 training table from tel-level OOF predictions.
    """

    return build_stereo_energy_meta_event_table(
        data_tel1=data_tel1,
        data_tel2=data_tel2,
        feature_names=feature_names,
        join_keys=join_keys,
        include_target=True,
        target_col=target_col,
        strict=strict,
        truth_tolerance=truth_tolerance,
    )


def apply_stereo_energy_meta_model_from_tables(
    data_tel1,
    data_tel2,
    meta_model,
    feature_names,
    join_keys=("obs_id", "event_id"),
):
    """
    Apply stage-2 stereo energy meta-regressor from two telescope tables.
    """

    event_table = build_stereo_energy_meta_event_table(
        data_tel1=data_tel1,
        data_tel2=data_tel2,
        feature_names=feature_names,
        join_keys=join_keys,
        include_target=False,
    )
    if event_table.empty:
        return pd.DataFrame(columns=list(join_keys) + ["log_reco_energy"])

    prediction = meta_model.predict(event_table[feature_names])
    out = event_table[list(join_keys)].copy()
    out["log_reco_energy"] = prediction
    return out
