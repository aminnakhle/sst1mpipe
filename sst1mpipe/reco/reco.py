import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.table import QTable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
import joblib
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from sst1mpipe.utils import (
    camera_to_altaz, 
    disp_to_pos, 
    disp_vector, 
    mix_gamma_proton,
    remove_stereo,
    get_event_pos_in_camera,
    get_event_sample,
    get_telescopes,
    event_hillas_add_units,
    get_wr_timestamp,
    get_stereo_method,
    get_horizon_frame,
    get_finite,
    get_tel_string
)
from sst1mpipe.io import (
    load_more_dl1_tables_mono,
    load_dl1_sst1m,
    check_outdir,
    get_dl1_info,
    load_dl1_pedestals,
    write_dl1_pedestals
)
from sst1mpipe.analysis import add_reco_ra_dec
from sst1mpipe.reco.stereo_meta import (
    build_stereo_energy_meta_training_table_from_oof,
    apply_stereo_energy_meta_model_from_tables,
)

from ctaplot.ana import (
    angular_separation_altaz, 
    logbin_mean
)
import logging

from ctapipe.reco import ShowerProcessor
from ctapipe.io import HDF5EventSource
from ctapipe.io import DataWriter


def plot_feature_importance(
        model, features=None,
        outfile=None,
        telescope=None):
    """
    Save figures with feature importances for given RF model

    Parameters
    ----------
    model: sklearn.ensemble.RandomForestClassifier or sklearn.ensemble.RandomForestRegressor
    features: list of string
    outfile: str
        output file path
    telescope: str (tel_001/tel_002)

    Returns
    -------

    """

    plt.figure()
    ax = plt.gca()

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)

    ordered_features = [features[index] for index in indices]

    ax.set_title("Feature importances (impurity based), " + telescope)

    ax.barh(ordered_features,
            importances[indices],
            xerr=std[indices],
            align="center",
            )

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(ordered_features)
    ax.grid()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)


def reco_source_position_sky(
        cog_x, cog_y, disp_dx, disp_dy,
        focal_length, pointing_alt, pointing_az,
        config=None, telescope=None, times=None):

    src_x, src_y = disp_to_pos(disp_dx, disp_dy, cog_x, cog_y)
    return camera_to_altaz(src_x, src_y, focal_length, pointing_alt, pointing_az, config=config, telescope=telescope, times=times)


def train_rf_energy(
        data, config=None, plot=False,
        outdir=None, telescope=None, stereo=False):
    """
    Train RF Regressor for energy reconstruction

    Parameters
    ----------
    data: pandas.DataFrame
    config: dict
    plot: bool
    outdir: str
        output file path
    telescope: str (tel_001/tel_002)
    stereo: bool

    Returns
    -------
    Trained model, sklearn.ensemble.RandomForestRegressor

    """

    logging.info("Training Random Forest Regressor for Energy Reconstruction...")
    logging.info("Number of events for training: %d", len(data))

    reg, features = fit_rf_energy_regressor(data, config=config, stereo=stereo)

    logging.info("Energy reconstruction model trained!")

    if plot:
        plot_feature_importance(reg, features=features, outfile=outdir + '/reg_energy_features_' + str(telescope) + '.png', telescope=telescope)

    joblib.dump(reg, outdir + '/reg_energy_' + str(telescope) + '.sav', compress=3)
    del reg


def fit_rf_energy_regressor(data, config=None, stereo=False):
    """
    Fit RF regressor for energy reconstruction without saving to disk.

    Parameters
    ----------
    data: pandas.DataFrame
    config: dict
    stereo: bool

    Returns
    -------
    tuple
        (fitted RandomForestRegressor, ordered feature list)
    """

    energy_regression_args = config['random_forest_regressor_args']
    features = list(config['energy_regression_features'])

    if not stereo:
        features = remove_stereo(features)

    logging.info("Given features: %s", features)
    reg = RandomForestRegressor(**energy_regression_args)
    reg.fit(data[features], data['log_true_energy'])
    return reg, features


def _subset_on_event_keys(data, event_keys, join_keys=("obs_id", "event_id")):
    if event_keys.empty:
        return data.iloc[0:0].copy()
    return pd.merge(data, event_keys, on=list(join_keys), how='inner')


def _load_stereo_base_models(models_dir):
    """
    Load pre-trained stage-1 per-telescope energy models for meta-only mode.
    
    Parameters
    ----------
    models_dir: str
        Directory containing reg_energy_tel_001.sav and reg_energy_tel_002.sav
    
    Returns
    -------
    dict
        {'tel_001': model, 'tel_002': model} or raises KeyError if missing
    """
    models = {}
    for tel in ['tel_001', 'tel_002']:
        path = os.path.join(models_dir, f'reg_energy_{tel}.sav')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Base stage-1 energy model not found at {path}. '
                'For meta-only mode, provide pre-trained per-telescope models.'
            )
        models[tel] = joblib.load(path)
        logging.info('Loaded base stage-1 model: %s', path)
    return models


def _sample_coincident_stereo_events(
    data_tel1,
    data_tel2,
    join_keys=("obs_id", "event_id"),
    max_events=None,
):
    """
    Build deterministic coincident stereo event key table and optionally cap by count.
    
    Parameters
    ----------
    data_tel1, data_tel2: pd.DataFrame
    join_keys: tuple
    max_events: int or None
    
    Returns
    -------
    pd.DataFrame
        Deduplicated stereo event keys (obs_id, event_id), optionally capped
    """
    keys_tel1 = data_tel1[list(join_keys)].drop_duplicates()
    keys_tel2 = data_tel2[list(join_keys)].drop_duplicates()
    stereo_events = pd.merge(keys_tel1, keys_tel2, on=list(join_keys), how='inner')
    
    n_total = len(stereo_events)
    if max_events is not None and n_total > max_events:
        stereo_events = stereo_events.sample(n=max_events, random_state=42)
        logging.info(
            'Capped stereo events: %d total -> %d (via max_events)',
            n_total,
            max_events,
        )
    else:
        logging.info('Total coincident stereo events: %d', n_total)
    
    return stereo_events.reset_index(drop=True)


def train_rf_energy_stereo_meta(
        data_tel1,
        data_tel2,
        config=None,
        plot=False,
        outdir=None,
        telescope_pair=None,
        training_mode='full_oof',
        base_models_dir=None):
    """
    Train event-level stereo energy meta-regressor.
    
    Supports two modes:
    - 'full_oof': Train fold-local stage-1 models and generate OOF predictions (leakage-safe)
    - 'meta_only': Load pre-trained stage-1 models and train only stage-2 meta-regressor
    """

    meta_cfg = config.get('stereo_energy_meta', {})
    meta_enabled = meta_cfg.get('enabled', False) or (training_mode == 'meta_only')
    if not meta_enabled:
        logging.info('Stereo energy meta-regressor disabled in config.')
        return

    join_keys = tuple(meta_cfg.get('join_keys', ("obs_id", "event_id")))
    config_mode = meta_cfg.get('training_mode', 'full_oof')
    
    if training_mode not in ['full_oof', 'meta_only']:
        raise ValueError(f'Unknown training_mode: {training_mode}. Use "full_oof" or "meta_only".')
    if config_mode not in ['full_oof', 'meta_only']:
        raise ValueError(f'Unknown config training_mode: {config_mode}. Use "full_oof" or "meta_only".')
    
    effective_mode = training_mode  # CLI flag overrides config
    logging.info('Stereo energy meta-regressor training mode: %s', effective_mode)
    
    if effective_mode == 'meta_only':
        if base_models_dir is None:
            raise ValueError(
                'meta_only mode requires base_models_dir with pre-trained stage-1 models. '
                'Provide --base-models-dir /path/to/models/'
            )

    n_folds = int(meta_cfg.get('n_folds', 5))
    feature_names = meta_cfg.get('features', [
        'log_reco_energy_tel1',
        'log_reco_energy_tel2',
        'log_reco_energy_mean',
        'log_reco_energy_weighted',
        'log_reco_energy_absdiff',
        'camera_frame_hillas_intensity_tel1',
        'camera_frame_hillas_intensity_tel2',
        'leakage_intensity_width_2_tel1',
        'leakage_intensity_width_2_tel2',
        'HillasReconstructor_tel_impact_distance_tel1',
        'HillasReconstructor_tel_impact_distance_tel2',
        'HillasReconstructor_h_max',
    ])

    max_training_events = config.get('analysis', {}).get('n_training_events', None)
    
    stereo_events = _sample_coincident_stereo_events(
        data_tel1=data_tel1,
        data_tel2=data_tel2,
        join_keys=join_keys,
        max_events=max_training_events,
    )
    n_events = len(stereo_events)

    if n_events < 2:
        logging.warning('Not enough coincident stereo events to train energy meta-regressor: %d', n_events)
        return

    # Generate stage-1 predictions for event-level training table
    if effective_mode == 'full_oof':
        n_folds = max(2, min(n_folds, n_events))
        logging.info(
            'Stage-2 meta training (full_oof): %d stereo events, %d folds, leakage-safe OOF generation',
            n_events,
            n_folds,
        )
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_tel1 = []
        oof_tel2 = []

        for i_fold, (train_idx, valid_idx) in enumerate(kfold.split(stereo_events), start=1):
            fold_train_keys = stereo_events.iloc[train_idx]
            fold_valid_keys = stereo_events.iloc[valid_idx]

            train_tel1 = _subset_on_event_keys(data_tel1, fold_train_keys, join_keys=join_keys)
            train_tel2 = _subset_on_event_keys(data_tel2, fold_train_keys, join_keys=join_keys)
            valid_tel1 = _subset_on_event_keys(data_tel1, fold_valid_keys, join_keys=join_keys).copy()
            valid_tel2 = _subset_on_event_keys(data_tel2, fold_valid_keys, join_keys=join_keys).copy()

            reg_tel1, features_tel1 = fit_rf_energy_regressor(train_tel1, config=config, stereo=True)
            reg_tel2, features_tel2 = fit_rf_energy_regressor(train_tel2, config=config, stereo=True)

            valid_tel1['log_reco_energy'] = reg_tel1.predict(valid_tel1[features_tel1])
            valid_tel2['log_reco_energy'] = reg_tel2.predict(valid_tel2[features_tel2])

            oof_tel1.append(valid_tel1)
            oof_tel2.append(valid_tel2)

            logging.info(
                'Stereo energy meta OOF fold %d/%d: %d validation events',
                i_fold,
                n_folds,
                len(fold_valid_keys),
            )

        oof_tel1 = pd.concat(oof_tel1, ignore_index=True)
        oof_tel2 = pd.concat(oof_tel2, ignore_index=True)
        logging.info('Collected OOF predictions: tel1 rows=%d, tel2 rows=%d', len(oof_tel1), len(oof_tel2))
        
    else:  # meta_only mode
        logging.info(
            'Stage-2 meta training (meta_only): loading pre-trained stage-1 models from %s',
            base_models_dir,
        )
        base_models = _load_stereo_base_models(base_models_dir)
        
        # Use the SAME features the stage-1 models were trained with (in stereo context)
        energy_features = list(config.get('energy_regression_features', []))
        # Do NOT call remove_stereo() - models were trained with these exact features
        
        oof_tel1 = data_tel1.copy()
        oof_tel2 = data_tel2.copy()
        
        logging.info('Applying stage-1 models with features: %s', energy_features[:3])  # Log first 3 for brevity
        oof_tel1['log_reco_energy'] = base_models['tel_001'].predict(oof_tel1[energy_features])
        oof_tel2['log_reco_energy'] = base_models['tel_002'].predict(oof_tel2[energy_features])
        
        logging.info(
            'Applied base stage-1 models: tel1 rows=%d, tel2 rows=%d',
            len(oof_tel1),
            len(oof_tel2),
        )

    logging.info('Building stage-2 training table from tel %d rows and tel2 %d rows', len(oof_tel1), len(oof_tel2))
    logging.info('Expected meta features: %s', feature_names)
    logging.info('Available tel1 columns: %s', list(oof_tel1.columns)[:15])
    
    training_table = build_stereo_energy_meta_training_table_from_oof(
        data_tel1=oof_tel1,
        data_tel2=oof_tel2,
        feature_names=feature_names,
        join_keys=join_keys,
        target_col='log_true_energy',
    )
    logging.info('Built training table: %d rows', len(training_table))
    
    if training_table.empty:
        logging.warning('Stereo energy meta training table is empty. Skipping model training.')
        logging.warning('This usually means merge failed or all rows had missing feature values.')
        return

    logging.info('Number of stage-2 stereo meta training events: %d', len(training_table))

    meta_regression_args = config.get(
        'random_forest_stereo_meta_regressor_args',
        config['random_forest_regressor_args'],
    )
    meta_regressor = RandomForestRegressor(**meta_regression_args)
    meta_regressor.fit(training_table[feature_names], training_table['log_true_energy'])

    if plot:
        plot_feature_importance(
            meta_regressor,
            features=feature_names,
            outfile=os.path.join(outdir, 'reg_energy_stereo_meta_features.png'),
            telescope='stereo_meta',
        )

    logging.info('Number of stage-2 stereo meta training events (after sampling): %d', len(training_table))
    
    metadata = {
        'version': 'stereo_energy_meta_v1',
        'features': feature_names,
        'join_keys': join_keys,
        'target_col': 'log_true_energy',
        'weights_col': 'camera_frame_hillas_intensity',
        'telescope_pair': telescope_pair,
        'training_mode': effective_mode,
    }

    model_path = os.path.join(outdir, 'reg_energy_stereo_meta.sav')
    features_path = os.path.join(outdir, 'reg_energy_stereo_meta_features.pkl')
    joblib.dump(meta_regressor, model_path, compress=3)
    joblib.dump(metadata, features_path, compress=3)
    logging.info('Stereo energy meta-regressor (%s) stored: %s', effective_mode, model_path)
    logging.info('Stereo energy meta metadata stored: %s', features_path)


def apply_stereo_energy_meta_model(params, models_dir=None, config=None):
    """
    Apply event-level stereo energy meta-regressor to per-telescope stereo rows.
    """

    meta_cfg = config.get('stereo_energy_meta', {})
    if not meta_cfg.get('enabled', False):
        return None

    if models_dir is None:
        logging.warning('Stereo meta-regressor enabled but models_dir is None. Falling back to weighted average.')
        return None

    model_path = os.path.join(models_dir, 'reg_energy_stereo_meta.sav')
    features_path = os.path.join(models_dir, 'reg_energy_stereo_meta_features.pkl')

    logging.info(
        'Stereo energy meta-regressor enabled. Checking model files: model=%s, features=%s',
        model_path,
        features_path,
    )

    if (not os.path.exists(model_path)) or (not os.path.exists(features_path)):
        logging.warning(
            'Stereo energy meta model files missing (%s or %s). Falling back to weighted average.',
            model_path,
            features_path,
        )
        return None

    tel_ids = sorted([int(tel_id) for tel_id in params['tel_id'].dropna().unique()])
    if len(tel_ids) < 2:
        logging.warning('Stereo energy meta-regressor requires two telescopes, found %d', len(tel_ids))
        return None

    tel1_id, tel2_id = tel_ids[0], tel_ids[1]
    data_tel1 = params[params['tel_id'] == tel1_id].copy()
    data_tel2 = params[params['tel_id'] == tel2_id].copy()

    meta_model = joblib.load(model_path)
    metadata = joblib.load(features_path)
    feature_names = metadata.get('features', meta_cfg.get('features', []))
    join_keys = tuple(metadata.get('join_keys', ("obs_id", "event_id")))
    logging.info(
        'Loaded stereo energy meta-regressor from %s with %d features.',
        model_path,
        len(feature_names),
    )

    pred = apply_stereo_energy_meta_model_from_tables(
        data_tel1=data_tel1,
        data_tel2=data_tel2,
        meta_model=meta_model,
        feature_names=feature_names,
        join_keys=join_keys,
    )
    logging.info('Stereo energy meta predictions generated for %d events', len(pred))
    return pred


def train_disp_vector(
        data, config=None, plot=False, 
        outdir=None, telescope=None, stereo=False):
    """
    Train RF Regressor for DISP reconstruction (vector of disp_dx, disp_dy)

    Parameters
    ----------
    data: pandas.DataFrame
    config: dict
    plot: bool
    outdir: str
        output file path
    telescope: str (tel_001/tel_002)
    stereo: bool

    Returns
    -------
    Trained model, sklearn.ensemble.RandomForestRegressor

    """

    logging.info("Training Random Forest Regressor for DISP Reconstruction...")
    predict_features = ['disp_dx', 'disp_dy']
    disp_regression_args = config['random_forest_regressor_args']
    features = config['disp_regression_features']

    if not stereo:
        features = remove_stereo(features)

    logging.info("Given features: %s", features)
    logging.info("Number of events for training: %d", len(data))

    reg = RandomForestRegressor(**disp_regression_args)
    x = data[features]
    y = np.transpose([data[f] for f in predict_features])
    reg.fit(x, y)

    logging.info("DISP reconstruction model trained!")

    if plot:
        plot_feature_importance(reg, features=features, outfile=outdir + '/reg_disp_features_' + str(telescope) + '.png', telescope=telescope)

    joblib.dump(reg, outdir + '/reg_disp_' + str(telescope) + '.sav', compress=3)
    del reg

def train_disp_norm(
        data, config=None, plot=False, 
        outdir=None, telescope=None, stereo=False):
    """
    Train RF Regressor for DISP norm reconstruction

    Parameters
    ----------
    data: pandas.DataFrame
    config: dict
    plot: bool
    outdir: str
        output file path
    telescope: str (tel_001/tel_002)
    stereo: bool

    Returns
    -------
    Trained model, sklearn.ensemble.RandomForestRegressor

    """

    logging.info("Training Random Forest Regressor for DISP Reconstruction...")
    disp_regression_args = config['random_forest_regressor_args']
    features = config['disp_regression_features']

    if not stereo:
        features = remove_stereo(features)

    logging.info("Given features: %s", features)
    logging.info("Number of events for training: %d", len(data))

    reg = RandomForestRegressor(**disp_regression_args)
    reg.fit(data[features],
            data['disp_norm'])

    logging.info("DISP NORM reconstruction model trained!")

    if plot:
        plot_feature_importance(reg, features=features, outfile=outdir + '/reg_disp_norm_features_' + str(telescope) + '.png', telescope=telescope)

    joblib.dump(reg, outdir + '/reg_disp_norm_' + str(telescope) + '.sav', compress=3)
    del reg


def train_disp_sign(
        data, config=None, plot=False, 
        outdir=None, telescope=None, stereo=False):
    """
    Train RF Classifier for DISP sign reconstruction

    Parameters
    ----------
    data: pandas.DataFrame
    config: dict
    plot: bool
    outdir: str
        output file path
    telescope: str (tel_001/tel_002)
    stereo: bool

    Returns
    -------
    Trained model, sklearn.ensemble.RandomForestClassifier

    """

    logging.info("Training Random Forest Classifier for DISP sign...")
    classification_args = config['random_forest_classifier_args']
    features = config["disp_classification_features"]

    if not stereo:
        features = remove_stereo(features)

    logging.info("Given features: %s", features)
    logging.info("Number of events for training: %d", len(data))

    clf = RandomForestClassifier(**classification_args)

    clf.fit(data[features],
            data['disp_sign'])
    logging.info("DISP SIGN classifier trained!")

    if plot:
        plot_feature_importance(clf, features=features, outfile=outdir + '/class_disp_sign_features_' + str(telescope) + '.png', telescope=telescope)

    joblib.dump(clf, outdir + '/class_disp_sign_' + str(telescope) + '.sav', compress=3)
    del clf


def train_rf_separation(
        data, config=None, plot=False, 
        outdir=None, telescope=None, stereo=False):
    """
    Train RF Classifier for gamma/hadron separation

    Parameters
    ----------
    data: pandas.DataFrame
    config: dict
    plot: bool
    outdir: str
        output file path
    telescope: str (tel_001/tel_002)
    stereo: bool

    Returns
    -------
    Trained model, sklearn.ensemble.RandomForestClassifier

    """

    logging.info("Training Random Forest Classifier for Gamma/Hadron separation...")
    classification_args = config['random_forest_classifier_args']
    features = config["particle_classification_features"]

    if not stereo:
        features = remove_stereo(features)

    logging.info("Given features: %s", features)
    logging.info("Number of events for training: %d", len(data))

    clf = RandomForestClassifier(**classification_args)

    clf.fit(data[features],
            data['true_shower_primary_id'])  # 0 .. gamma, 101 .. proton
    logging.info("Gamma/hadron classifier trained!")

    if plot:
        plot_feature_importance(clf, features=features, outfile=outdir + '/class_gh_features_' + str(telescope) + '.png', telescope=telescope)

    joblib.dump(clf, outdir + '/class_gh_' + str(telescope) + '.sav', compress=3)
    del clf


def train_rf_misdirection(
    data, config=None,
    outdir=None, telescope=None,
    plot=None):

    regression_args = config['random_forest_regressor_args']
    if telescope == 'stereo':
        features = config['misdirection_regression_features_stereo']
    else:
        features = config['misdirection_regression_features_mono']

    reg = RandomForestRegressor(**regression_args)
    reg.fit(data[features], data['log_misdirection'])
    
    if plot:
        plot_feature_importance(reg, features=features, outfile=outdir + '/reg_mis_features_' + str(telescope) + '.png', telescope=tel)

    joblib.dump(reg, outdir + '/reg_misdirection_' + str(telescope) + '.sav', compress=3)
    del reg


def train_models(
        params_gamma, params_protons, config=None, 
        plot=None, outdir=None, telescope=None, 
    stereo=False,
    params_gamma_tel1=None,
    params_gamma_tel2=None,
    telescope_pair=None,
    train_stereo_meta_this_run=False,
    stereo_meta_training_mode='full_oof',
    stereo_meta_base_models_dir=None):
    """
    Run training of all RF models on MC gammas and protons

    Parameters
    ----------
    params_gamma: pandas.DataFrame
    params_protons: pandas.DataFrame
    config: dict
    plot: bool
    outdir: str
        output file path
    telescope: str (tel_001/tel_002)
    stereo: bool

    Returns
    -------

    """

    N_training_events = config["analysis"]["n_training_events"]
    gp_ratio = config["analysis"]["gamma_to_proton_training_ratio"]
    gamma_training_sample = get_event_sample(params_gamma, max_events=N_training_events)

    # Training models
    if (config['disp_method'] == 'disp_vector') & (not stereo):
        train_disp_vector(gamma_training_sample, config=config, plot=plot, outdir=outdir, telescope=telescope, stereo=stereo)
    elif (config['disp_method'] == 'disp_norm_sign') & (not stereo):
        train_disp_norm(gamma_training_sample, config=config, plot=plot, outdir=outdir, telescope=telescope, stereo=stereo)
        train_disp_sign(gamma_training_sample, config=config, plot=plot, outdir=outdir, telescope=telescope, stereo=stereo)
    elif stereo:
        train_disp_norm(gamma_training_sample, config=config, plot=plot, outdir=outdir, telescope=telescope, stereo=stereo)
        logging.info("STEREO reconstruction: ONLY DISP NORM REGRESSOR WILL BE TRAINED (NOT THE SIGN CLASSIFIER), regardless the method set in the config file.")
    else:
        raise Exception("ERROR: disp_method in the config file not recognized, use one of these: disp_vector, disp_norm_sign")

    train_rf_energy(gamma_training_sample, config=config, plot=plot, outdir=outdir, telescope=telescope, stereo=stereo)

    if params_protons is not None:

        # We want to use reco energy and disp_norm as features for g/h classifier (we skip disp_sign as it 
        # would be quite painful to implement again for stereo). To do this, we need to split training gammas into two subsets,
        # train energy and disp reconstructors on one of them, reco these two parameters in second of them, and then use the
        # second one as training sample for g/h separator
        if 'log_reco_energy' in config['particle_classification_features']:

            #N_gammas_train = int(len(gamma_training_sample)/2.)
            gamma_train_temp, gamma_test_temp = train_test_split(gamma_training_sample, test_size=0.33, random_state=42)

            # Training temporary models
            outdir_tmp = outdir + '/tmp/'
            check_outdir(outdir_tmp)
            logging.info('Training temporary models for energy and disp_norm reco for training of g/h separation.')
            train_disp_norm(gamma_train_temp, config=config, plot=plot, outdir=outdir_tmp, telescope=telescope, stereo=stereo)
            train_rf_energy(gamma_train_temp, config=config, plot=plot, outdir=outdir_tmp, telescope=telescope, stereo=stereo)

            # Reco energy and disp
            energy_regression_features = config["energy_regression_features"]
            disp_regression_features = config["disp_regression_features"]

            if not stereo:
                energy_regression_features = remove_stereo(energy_regression_features)
                disp_regression_features = remove_stereo(disp_regression_features)

            energy_re = joblib.load(os.path.join(outdir_tmp, 'reg_energy_' + str(telescope) + '.sav'))
            logging.info('Loading model: ' + outdir_tmp + '/reg_energy_' + str(telescope) + '.sav')
            logging.info('Temporary energy reconstruction for g/h separation training..')
            gamma_test_temp['log_reco_energy'] = energy_re.predict(gamma_test_temp[energy_regression_features])
            params_protons['log_reco_energy'] = energy_re.predict(params_protons[energy_regression_features])

            disp_re = joblib.load(os.path.join(outdir_tmp, 'reg_disp_norm_' + str(telescope) + '.sav'))
            logging.info('Loading model: ' + outdir_tmp + '/reg_disp_norm_' + str(telescope) + '.sav')
            logging.info('Temporary DISP norm reconstruction for g/h separation training..')
            gamma_test_temp['reco_disp_norm'] = disp_re.predict(gamma_test_temp[disp_regression_features])
            params_protons['reco_disp_norm'] = disp_re.predict(params_protons[disp_regression_features])

            gh_training_dataset = mix_gamma_proton(gamma_test_temp, params_protons, max_events=N_training_events, gp_training_ratio=gp_ratio)
            gh_training_dataset = get_finite(gh_training_dataset, config=config, stereo=stereo)

            train_rf_separation(gh_training_dataset, config=config, plot=plot, outdir=outdir, telescope=telescope, stereo=stereo)
            logging.info('Gamma/hadron separator trained using reco_energy and disp_norm as additional features.')

        else:
            gh_training_dataset = mix_gamma_proton(params_gamma, params_protons, max_events=N_training_events, gp_training_ratio=gp_ratio)
            train_rf_separation(gh_training_dataset, config=config, plot=plot, outdir=outdir, telescope=telescope, stereo=stereo)
    else:
        logging.warning('RF classifier cannot be trained, dus to missing input DL1 proton file!')

    meta_enabled_for_run = config.get('stereo_energy_meta', {}).get('enabled', False) or (stereo_meta_training_mode == 'meta_only')
    if stereo and meta_enabled_for_run:
        if train_stereo_meta_this_run:
            if (params_gamma_tel1 is not None) and (params_gamma_tel2 is not None):
                logging.info(
                    'Stereo meta training enabled for this run (primary telescope). '
                    'Mode: %s. Training stage-2 meta-regressor.',
                    stereo_meta_training_mode,
                )
                train_rf_energy_stereo_meta(
                    params_gamma_tel1,
                    params_gamma_tel2,
                    config=config,
                    plot=plot,
                    outdir=outdir,
                    telescope_pair=telescope_pair,
                    training_mode=stereo_meta_training_mode,
                    base_models_dir=stereo_meta_base_models_dir,
                )
            else:
                logging.warning(
                    'Stereo meta training requested but per-telescope gamma tables are missing. '
                    'Skipping stage-2 training.'
                )
        else:
            logging.info(
                'Stereo meta training skipped for this run to avoid duplicate stage-2 training. '
                '(Not primary telescope run).'
            )
        # If meta-only mode, still apply it even if we didn't train it
        if stereo_meta_training_mode != 'full_oof' and stereo_meta_base_models_dir is None:
            logging.warning(
                'Stereo meta training_mode set to %s but no models provided. '
                'Skipping meta training.',
                stereo_meta_training_mode,
            )

    logging.info('All Random Forest models trained and stored here: %s', outdir)


def reco_misdirection(dl2, models_dir=None, config=None, telescope=None):

    if telescope == 'stereo':

        # In data, we need to rename columns _tel21 -> tel1, tel22 -> tel2
        for tt,ttn in zip(['_tel21', '_tel22'], ['_tel1', '_tel2']):
            cols_to_rename = dl2.columns[dl2.columns.str.contains(tt)]
            rename_dict = {col: col.replace(tt, ttn) for col in cols_to_rename}
            dl2 = dl2.rename(columns=rename_dict)

        dl2['log_camera_frame_hillas_intensity_tel1'] = np.log10(dl2['camera_frame_hillas_intensity_tel1'])
        dl2['log_camera_frame_hillas_intensity_tel2'] = np.log10(dl2['camera_frame_hillas_intensity_tel2'])

    if telescope == 'stereo':
        features = config['misdirection_regression_features_stereo']
    else:
        features = config['misdirection_regression_features_mono']
    
    # Check finite
    mask = np.ones(len(dl2), dtype=bool)
    for key in features:
        try:
            mask &= np.isfinite(dl2[key])
        except:
            logging.warning('{} column not in data.'.format(key))
    dl2_finite = dl2[mask].copy()

    mis_mono_re = joblib.load(models_dir + '/reg_misdirection_'+telescope+'.sav')
    logging.info('Misdirection reconstruction..')
    dl2_finite['log_reco_misdirection'] = mis_mono_re.predict(dl2_finite[features])
    return dl2_finite


def get_evttype_edges(dl2, config=None, percentiles=[25, 50, 75]):
    
    ebins = np.logspace(config['analysis']["log_energy_min_tev"], config['analysis']["log_energy_max_tev"], config['analysis']["n_energy_bins"])
    energy_center = logbin_mean(ebins)

    edges = []
    for i in range(len(ebins)-1):
        mask = (dl2['reco_energy'] > ebins[i]) & (dl2['reco_energy'] < ebins[i+1])

        if sum(mask):
            evttype_edges = np.percentile(
                    dl2['log_reco_misdirection'][mask],
                    percentiles
                )
            evttype_edges = np.concatenate(
                ([-np.inf], evttype_edges, [np.inf])
            )
        else:
            evttype_edges = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        edges.append(evttype_edges)
    edges = np.array(edges)

    
    cut_table = QTable()
    cut_table["low"] = ebins[:-1]
    cut_table["high"] = ebins[1:]
    cut_table["center"] = energy_center
    for i in range(len(percentiles)+2):
        key = 'log_midirection_edge_' + str(i)
        cut_table[key] = np.asanyarray(np.nan, edges.dtype)
        cut_table[key].value[:] = edges[:,i]

    return cut_table


def classify_evt_types(dl2, edges=None):
    
    dl2['event_type'] = 0
    ebins_low = edges["low"]
    ebins_high = edges["high"]

    evttype_edges = [s for s in edges.keys() if 'log_midirection_' in s]

    for i in range(len(ebins_low)):
        mask_e = (dl2['reco_energy'] > ebins_low[i]) & (dl2['reco_energy'] < ebins_high[i])

        for j in range(len(evttype_edges)-1):
            mask_class = (dl2['log_reco_misdirection'] > edges[evttype_edges[j]][i]) & (dl2['log_reco_misdirection'] <= edges[evttype_edges[j+1]][i])
            dl2.loc[mask_e & mask_class, 'event_type'] = j+1

    return dl2


def apply_models(
        dl1, models_dir=None, config=None, 
        telescope=None, stereo=False, mc=True):
    """
    Applies all RF models from models directory to reconstruct 
    all events from DL1 to DL2. Works on both MC/data, mono/stereo.
    For stereo reconstruction, only DISP norm is reconstructed, 
    and both possible signs are stored in the output DataFrame for 
    each event (MARS-like stereo reconstruction). All quantities
    are reconstructed per-telescope even in stereo. Final averaging
    for stereo reconstruction is done in .reco.stereo_reconstruction()

    Parameters
    ----------
    dl1: pandas.DataFrame
    models_dir: str
        models dir path
    config: dict
    telescope: str (tel_001/tel_002)
    stereo: bool
    mc: bool

    Returns
    -------
    pandas.DataFrame

    """

    if not mc:
        if '1' in telescope:
            telescope_model = 'tel_001'
        else:
            telescope_model = 'tel_002'
    else:
        telescope_model = telescope

    dl2 = dl1.copy()

    energy_regression_features = config["energy_regression_features"]
    gh_clas_features = config["particle_classification_features"]
    disp_regression_features = config["disp_regression_features"]
    disp_classification_features = config["disp_classification_features"]

    if not stereo:
        energy_regression_features = remove_stereo(energy_regression_features)
        gh_clas_features = remove_stereo(gh_clas_features)
        disp_regression_features = remove_stereo(disp_regression_features)
        disp_classification_features = remove_stereo(disp_classification_features)

    try:
        energy_re = joblib.load(os.path.join(models_dir, 'reg_energy_' + str(telescope_model) + '.sav'))
        logging.info('Loading model: ' + models_dir + '/reg_energy_' + str(telescope_model) + '.sav')
        logging.info('Energy reconstruction..')
        dl2['log_reco_energy'] = energy_re.predict(dl1[energy_regression_features])
        dl1['log_reco_energy'] = dl2['log_reco_energy']
        dl2['reco_energy'] = 10**dl2['log_reco_energy']
    except FileNotFoundError:
        logging.warning('Energy reconstruction model was not found. Energy of all reconstructed events will be set to 0.')
        dl2['log_reco_energy'] = 0
        dl1['log_reco_energy'] = 0
        dl2['reco_energy'] = 0

    if (config['disp_method'] == 'disp_vector') and not stereo:

        logging.info('Mono DISP reconstruction using disp_vector..')

        try:
            logging.info('Loading model: ' + models_dir + '/reg_disp_' + str(telescope_model) + '.sav')
            disp_re = joblib.load(os.path.join(models_dir, 'reg_disp_' + str(telescope_model) + '.sav'))
            disp_vec = disp_re.predict(dl1[disp_regression_features])
        except FileNotFoundError:
            logging.warning('DISP model was not found. DISP vector of all reconstructed events will be set to [0, 0].')
            disp_vec = np.zeros((len(dl2), 2))


    elif (config['disp_method'] == 'disp_norm_sign') and not stereo:

        logging.info('Mono DISP reconstruction using disp_norm_sign..')
        try:
            logging.info('Loading model: ' + models_dir + '/reg_disp_norm_' + str(telescope_model) + '.sav')
            disp_re = joblib.load(os.path.join(models_dir, 'reg_disp_norm_' + str(telescope_model) + '.sav'))
            logging.info('Loading model: ' + models_dir + '/class_disp_sign_' + str(telescope_model) + '.sav')
            disp_clas = joblib.load(os.path.join(models_dir, 'class_disp_sign_' + str(telescope_model) + '.sav'))

            disp_norm = disp_re.predict(dl1[disp_regression_features])
            disp_sign = disp_clas.predict(dl1[disp_classification_features])

            dl2['reco_disp_norm'] = disp_norm
            dl1['reco_disp_norm'] = disp_norm
            dl2['reco_disp_sign'] = disp_sign

            # NOTE: Be carefull with the units! - must be revisited for proper unit conversion!
            disp_angle = dl2['camera_frame_hillas_psi'] * np.pi / 180

            disp_vec = disp_vector(disp_norm, disp_angle, disp_sign)
            
        except FileNotFoundError:
            logging.warning('DISP model was not found. DISP vector of all reconstructed events will be set to [0, 0].')
            disp_vec = np.zeros((len(dl2), 2))

    elif stereo:

        logging.info('Mono DISP reconstruction of disp norm..')
        disp_re = joblib.load(os.path.join(models_dir, 'reg_disp_norm_' + str(telescope_model) + '.sav'))
        disp_norm = disp_re.predict(dl1[disp_regression_features])
        dl2['reco_disp_norm'] = disp_norm
        dl1['reco_disp_norm'] = disp_norm

        # For MARS-like stereo reconstruction we need to consider both possible source positions
        dl2['reco_disp_sign_p'] = 1
        dl2['reco_disp_sign_m'] = -1

    else:
        raise Exception("ERROR: disp_method in the config file not recognized, use one of these: disp_vector, disp_norm_sign")

    # g/h needs to be reconstructed here, after reco_log_energy and reco_disp_norm columns are in dl1
    try:
        logging.info('Loading model: ' + models_dir + '/class_gh_' + str(telescope_model) + '.sav')
        gh_clas = joblib.load(os.path.join(models_dir, 'class_gh_' + str(telescope_model) + '.sav'))
        logging.info('Gamma/hadron classification..')
        dl2['gammaness'] = gh_clas.predict_proba(dl1[gh_clas_features])[:, 0]
    except FileNotFoundError:
        logging.warning('Gamma/hadron classification model was not found. Gammaness of all reconstructed events will be set to 0.')
        dl2['gammaness'] = 0

    if not stereo:
        
        dl2['reco_disp_dx'] = disp_vec[:, 0]
        dl2['reco_disp_dy'] = disp_vec[:, 1]
        dl2['reco_src_x'], dl2['reco_src_y'] = disp_to_pos(dl2.reco_disp_dx,
                                                            dl2.reco_disp_dy,
                                                            dl2.camera_frame_hillas_x,
                                                            dl2.camera_frame_hillas_y,
                                                            )

        # NOTE: Be carefull with the units! - must be revisited for proper unit conversion!
        if ~(dl2.true_alt_tel.values > np.pi).any():
            raise Exception("ERROR: true_alt_tel, true_az_tel are probably not in degrees! Check your DL1 file.")

        if mc:
            times=None
        else:
            times = Time(dl2.local_time, format='unix', scale='utc') 

        src_pos_reco = reco_source_position_sky(dl2.camera_frame_hillas_x.values * u.m,
                                                    dl2.camera_frame_hillas_y.values * u.m,
                                                    dl2.reco_disp_dx.values * u.m,
                                                    dl2.reco_disp_dy.values * u.m,
                                                    dl2.equivalent_focal_length.values * u.m,
                                                    dl2.true_alt_tel.values * u.deg,
                                                    dl2.true_az_tel.values * u.deg,
                                                    config=config,
                                                    telescope=telescope,
                                                    times=times)

        dl2['reco_alt'] = src_pos_reco.alt.deg
        azimuth_transformed = (src_pos_reco.az.to_value(u.deg) +  180) % 360 - 180
        if abs(dl2['true_az_tel'][0]) <= 90:
            dl2['reco_az'] = azimuth_transformed
            logging.info('Using transformed azimuth')
        else:
            dl2['reco_az'] = src_pos_reco.az.deg
            logging.info('Using non-transformed azimuth')

        radec = src_pos_reco.transform_to('icrs')
        dl2['reco_ra'] = radec.ra.deg
        dl2['reco_dec'] = radec.dec.deg

        logging.info('Reconstruction done.')

    return dl2


def stereo_reconstruction(
        params, config=None, 
    ismc=False, telescopes=None, models_dir=None):
    """
    Averages energy and gammaness stereoscopicly reconstructed for 
    each telescope to get final stereo quantities for each event.
    For direction reconstruction, alt/az for both signs from each 
    telescope are calculated. Then the combination leading to
    reconstructed coordinates with the smallest distance is
    selected and final direction is calculated as weighted average 
    of the two (MARS-like stereo reconstruction).

    Parameters
    ----------
    params: pandas.DataFrame
    config: dict
    ismc: bool
    telescopes: list of strings

    Returns
    -------
    pandas.DataFrame

    """

    logging.info('Stereo reconstruction..')
    weights = config["analysis"]["stereo_reco_weights"]
    logging.info('Stereo reconstruction event weights: %s', weights)

    energy_average, energy_var, energy_wvar = get_average_param(params, param='log_reco_energy', weights=weights)
    gammaness_average, gammaness_var, gammaness_wvar = get_average_param(params, param='gammaness', weights=weights)

    dl2 = get_stereo_dl2(params, ismc=ismc)

    dl2['log_reco_energy_avg'] = energy_average
    dl2['log_reco_energy'] = dl2['log_reco_energy_avg']

    meta_cfg = config.get('stereo_energy_meta', {})
    if meta_cfg.get('enabled', False):
        meta_pred = apply_stereo_energy_meta_model(params, models_dir=models_dir, config=config)
        if meta_pred is not None and len(meta_pred) > 0:
            pred_series = meta_pred.set_index(['obs_id', 'event_id'])['log_reco_energy']
            dl2['log_reco_energy_meta'] = pred_series.reindex(dl2.index)
            fallback_enabled = meta_cfg.get('fallback_to_weighted_average', True)
            n_total_events = int(len(dl2))
            n_meta_events = int(dl2['log_reco_energy_meta'].notna().sum())
            if fallback_enabled:
                n_fallback = int(dl2['log_reco_energy_meta'].isna().sum())
                if n_fallback > 0:
                    logging.warning('Stereo energy meta missing for %d events, using weighted average fallback.', n_fallback)
                dl2['log_reco_energy'] = dl2['log_reco_energy_meta'].fillna(dl2['log_reco_energy_avg'])
                logging.info(
                    'STEREO_META_USAGE_CONFIRMED: meta used for %d/%d events; weighted-average fallback for %d/%d events.',
                    n_meta_events,
                    n_total_events,
                    n_fallback,
                    n_total_events,
                )
            else:
                dl2['log_reco_energy'] = dl2['log_reco_energy_meta']
                logging.info(
                    'STEREO_META_USAGE_CONFIRMED: meta used for %d/%d events; fallback disabled.',
                    n_meta_events,
                    n_total_events,
                )
        else:
            logging.warning('Stereo energy meta-regressor unavailable or produced no predictions. Using weighted average.')
    else:
        logging.info('STEREO_META_USAGE_CONFIRMED: disabled in config, using weighted-average stereo energy for all events.')

    dl2['reco_energy'] = 10**dl2['log_reco_energy']
    dl2['var_reco_energy'] = (np.log(10) * dl2['reco_energy'])**2 * energy_var
    dl2['wvar_reco_energy'] = (np.log(10) * dl2['reco_energy'])**2 * energy_wvar
    dl2['gammaness'] = gammaness_average
    dl2['var_gammaness'] = gammaness_var
    dl2['wvar_gammaness'] = gammaness_wvar

    # We need to add these for event types classificantion
    tel_ids = sorted([int(tel_id) for tel_id in params['tel_id'].dropna().unique()])
    if len(tel_ids) < 2:
        raise ValueError('Stereo reconstruction requires two telescope IDs, found {}'.format(tel_ids))

    mask_tel1 = params['tel_id'] == tel_ids[0]
    mask_tel2 = params['tel_id'] == tel_ids[1]
    hillas_tel1 = params[mask_tel1][["obs_id", "event_id", "HillasReconstructor_tel_impact_distance"]]
    hillas_tel2 = params[mask_tel2][["obs_id", "event_id", "HillasReconstructor_tel_impact_distance"]]
    hillas_tel1 = hillas_tel1.rename(
        columns={"HillasReconstructor_tel_impact_distance":
        "HillasReconstructor_tel_impact_distance_tel1"}
        )
    hillas_tel2 = hillas_tel2.rename(
        columns={"HillasReconstructor_tel_impact_distance":
        "HillasReconstructor_tel_impact_distance_tel2"}
        )
    dl2 = dl2.merge(hillas_tel1, on=['obs_id', 'event_id'], how='left')
    dl2 = dl2.merge(hillas_tel2, on=['obs_id', 'event_id'], how='left')

    # Arrival direction reconstruction

    # NOTE: Be carefull with the units! - must be revisited for proper unit conversion!
    disp_angle = params['camera_frame_hillas_psi'] * np.pi / 180

    # Here we reconstruct both possible alt,az of the event for each telescope
    for sign in ['reco_disp_sign_p', 'reco_disp_sign_m']:

        disp_sign = params[sign]

        # initialization of two more columns
        params['reco_alt'+sign.replace('reco_disp','')] = 0
        params['reco_az'+sign.replace('reco_disp','')] = 0

        disp_vec = disp_vector(params['reco_disp_norm'], disp_angle, disp_sign)
        reco_disp_dx = disp_vec[:, 0]
        reco_disp_dy = disp_vec[:, 1]

        #for telescope in ['tel_001', 'tel_002']:
        # This should hopefuly work on data as well (where we have tel_021, tel_022)
        for telescope in telescopes:

            tel = int(telescope.split('_')[-1])
            mask = params['tel_id'] == tel

            src_pos_reco = reco_source_position_sky(
                params[mask].camera_frame_hillas_x.values * u.m,
                params[mask].camera_frame_hillas_y.values * u.m,
                reco_disp_dx[mask] * u.m,
                reco_disp_dy[mask] * u.m,
                params[mask].equivalent_focal_length.values * u.m,
                params[mask].true_alt_tel.values * u.deg,
                params[mask].true_az_tel.values * u.deg,
                config=config,
                telescope=telescope
                )
            
            azimuth_transformed = (src_pos_reco.az.to_value(u.deg) + 180) % 360 - 180            
            params.loc[mask, 'reco_alt'+sign.replace('reco_disp','')] = src_pos_reco.alt.deg
            if abs(params[mask].true_az_tel.values[0]) <= 90:
                params.loc[mask, 'reco_az'+sign.replace('reco_disp','')] = azimuth_transformed
                logging.info('Using transformed azimuth')
            else:
                params.loc[mask, 'reco_az'+sign.replace('reco_disp','')] = src_pos_reco.az.deg
                logging.info('Using non-transformed azimuth')

    # Now we have in params two azimuths and altitudes for each telescope, generated 
    # using either + or - sign togther with the reconstructd disp_norm. We now need 
    # to calculate all four possible angular distances (+,+), (-,-), (+,-), (-,+)
    # select the closest one, and calculate weighted average direction
    # NOTE: We can safely average directions in alt/az, no need to go for ra/dec, because the difference 
    # between the two frames is minor (of the order of 10^-3 deg).

    averaged_direction = get_averaged_direction(params, weights=weights, telescopes=telescopes)

    dl2 = pd.merge(dl2, averaged_direction, on=["obs_id", "event_id"])

    # Here we can use horizon frame of either of the telescopes
    if ismc:
        horizon_frame = get_horizon_frame(config=config, telescope='tel_001')
    else:
        horizon_frame = get_horizon_frame(config=config, telescope='tel_021', times=Time(dl2['local_time'], format='unix'))
    dl2 = add_reco_ra_dec(dl2, horizon_frame=horizon_frame)

    tel = []
    for tele in telescopes:
        tel.append(int(tele.split('_')[-1]))

    dl2.drop([
        'reco_alt_sign_p_tel'+str(tel[0]),
        'reco_alt_sign_m_tel'+str(tel[0]), 
        'reco_az_sign_p_tel'+str(tel[0]), 
        'reco_az_sign_m_tel'+str(tel[0]),
        'reco_alt_sign_p_tel'+str(tel[1]),
        'reco_alt_sign_m_tel'+str(tel[1]), 
        'reco_az_sign_p_tel'+str(tel[1]), 
        'reco_az_sign_m_tel'+str(tel[1])
        ],
        inplace=True, 
        axis=1
        )

    return dl2


def get_averaged_direction(params, weights=None, telescopes=None):
    """
    Calculate weighted average of the closest from all four possible
    alt/az combinations from both telescopes.

    Parameters
    ----------
    params: pandas.DataFrame
    weights: str
        Usualy one of the hillas features present in the input
        DL1 pandas.DataFrame
    telescopes: list of strings

    Returns
    -------
    pandas.DataFrame

    """

    tel = []
    for tele in telescopes:
        tel.append(int(tele.split('_')[-1]))

    params_reshaped = reshape_event_table_for_stereo(params, tels=tel)

    logging.info('Finding the closest combination of direction..')
    distance = pd.DataFrame()
    for combo in ['pp', 'mm', 'mp', 'pm']:
        distance[combo] = angular_distance(params_reshaped, combination=combo, tels=tel)

    params_reshaped['min_distance'] = distance.min(axis = 1)
    params_reshaped['best_sign_combination'] = distance.idxmin(axis = 1)

    params_reshaped['reco_alt'] = 0
    params_reshaped['reco_az'] = 0

    telescopes_averaged = params_reshaped.copy(deep=True)

    logging.info('Averaging reconstructed event directions..')
    for combo in ['pp', 'mm', 'mp', 'pm']:
        mask = telescopes_averaged['best_sign_combination'] == combo
        for coord in ['alt', 'az']:
            if weights is not None:
                telescopes_averaged.loc[mask, 'reco_'+coord] = (
                    params_reshaped.loc[mask]['reco_'+coord+'_sign_'+combo[0]+'_tel'+str(tel[0])] * 
                    params_reshaped.loc[mask][weights + '_tel'+str(tel[0])] + 
                    params_reshaped.loc[mask]['reco_'+coord+'_sign_'+combo[1]+'_tel'+str(tel[1])] * 
                    params_reshaped.loc[mask][weights + '_tel'+str(tel[1])] 
                    ) / (
                        params_reshaped.loc[mask][weights + '_tel'+str(tel[0])] + 
                        params_reshaped.loc[mask][weights + '_tel'+str(tel[1])]
                    )
            else:
                telescopes_averaged.loc[mask, 'reco_'+coord] = (
                    params_reshaped.loc[mask]['reco_'+coord+'_sign_'+combo[0]+'_tel'+str(tel[0])] + 
                    params_reshaped.loc[mask]['reco_'+coord+'_sign_'+combo[1]+'_tel'+str(tel[1])]
                    ) / 2

    return telescopes_averaged


def reshape_event_table_for_stereo(params, tels=None):

    data_tel1 = get_data_tel(params, tel=tels[0])
    data_tel2 = get_data_tel(params, tel=tels[1])
    params_reshaped = pd.merge(data_tel1, data_tel2, on=["obs_id", "event_id"])

    return params_reshaped


def get_data_tel(params, tel=1):

    mask_tel = params['tel_id'] == tel

    # We need those telescope dependent parameters (e.g. length or width)
    # for later use for event selection in performance evaluation
    data_tel = params[mask_tel][[
        "obs_id", 
        "event_id", 
        'reco_alt_sign_p', 
        'reco_alt_sign_m', 
        'reco_az_sign_p', 
        'reco_az_sign_m', 
        "camera_frame_hillas_intensity",
        "camera_frame_hillas_width",
        "camera_frame_hillas_length",
        "leakage_intensity_width_2",
        "camera_frame_hillas_wl", 
        "camera_frame_hillas_skewness", 
        "camera_frame_hillas_kurtosis", 
        "camera_frame_timing_slope"
        ]]
        
    data_tel = data_tel.rename(
                    columns={
                            "reco_alt_sign_p": "reco_alt_sign_p_tel"+str(tel),
                            "reco_alt_sign_m": "reco_alt_sign_m_tel"+str(tel),
                            "reco_az_sign_p": "reco_az_sign_p_tel"+str(tel),
                            "reco_az_sign_m": "reco_az_sign_m_tel"+str(tel),
                            "camera_frame_hillas_intensity": "camera_frame_hillas_intensity_tel"+str(tel),
                            "camera_frame_hillas_width": "camera_frame_hillas_width_tel"+str(tel),
                            "camera_frame_hillas_length": "camera_frame_hillas_length_tel"+str(tel),
                            "leakage_intensity_width_2": "leakage_intensity_width_2_tel"+str(tel),
                            "camera_frame_hillas_wl": "camera_frame_hillas_wl_tel"+str(tel),
                            "camera_frame_hillas_skewness": "camera_frame_hillas_skewness_tel"+str(tel),
                            "camera_frame_hillas_kurtosis": "camera_frame_hillas_kurtosis_tel"+str(tel),
                            "camera_frame_timing_slope": "camera_frame_timing_slope_tel"+str(tel)
                            }
                )

    return data_tel


def angular_distance(params, combination='pp', tels=None):
    """
    Calculates angular distance for all four combination of 
    reconstructed source positions for images of the two telescopes.

    Parameters
    ----------
    params: pandas.DataFrame
    combination: str
    tels: list of ints [1,2]

    Returns
    -------
    pandas.DataFrame
        distance in radians
    """

    distance = angular_separation_altaz(
                    params['reco_alt_sign_'+combination[0]+'_tel'+str(tels[0])].values * np.pi/180. * u.rad, 
                    params['reco_az_sign_'+combination[0]+'_tel'+str(tels[0])].values * np.pi/180. * u.rad, 
                    params['reco_alt_sign_'+combination[1]+'_tel'+str(tels[1])].values * np.pi/180. * u.rad, 
                    params['reco_az_sign_'+combination[1]+'_tel'+str(tels[1])].values * np.pi/180. * u.rad, 
                )
    return distance


def get_average_param(params, param=None, weights=None):
    """
    Get stereo quantities for both telescopes averaged

    Parameters
    ----------
    params: pandas.DataFrame
    param: str
        Quantity to be averaged, e.g. gammaness
    weights: str

    Returns
    -------
    average:    
        pandas.DataFrame
    variance:
        Note that variance is calculated always from unweighted quantities
        pandas.DataFrame
    """

    logging.info('Averaging reconstructed %s', param)

    if weights is not None:
        weights = params[weights]

        df = pd.DataFrame(
            data={
                "obs_id": params["obs_id"],
                "event_id": params["event_id"],
                "weight": weights,
                "param": params[param],
                "weighted_param": params[param] * weights,
            }
        )

        df_sum = df.groupby(["obs_id", "event_id"]).sum()
        average = df_sum["weighted_param"] / df_sum["weight"]
        variance = df.groupby(["obs_id", "event_id"]).var()["param"]

        # weighted variance
        # Merge the mean back to the original df for subtraction
        df = df.join(average.rename("mean_weighted"), on=["obs_id", "event_id"])
        # Calculate (x - mean)^2 * weight
        df["squared_diff"] = df["weight"] * (df["param"] - df["mean_weighted"])**2
        # Sum over groups and divide by sum of weights to get variance
        sum_w_squared_diff = df.groupby(["obs_id", "event_id"])["squared_diff"].sum()
        weighted_variance = sum_w_squared_diff / df_sum["weight"]

    else:
        df = pd.DataFrame(
            data={
                "obs_id": params["obs_id"],
                "event_id": params["event_id"],
                "param": params[param],
            }
        )

        df_sum = df.groupby(["obs_id", "event_id"]).sum()
        average = df_sum["param"] / df.groupby(["obs_id", "event_id"]).size()
        variance = df.groupby(["obs_id", "event_id"]).var()["param"]
        weighted_variance = variance.copy()

    return average, variance, weighted_variance


def get_stereo_dl2(params, ismc=False):
    """
    Select columns and create DL2 stereo table

    Parameters
    ----------
    params: pandas.DataFrame
    ismc: bool

    Returns
    -------
    pandas.DataFrame

    """

    features = [
        "obs_id", 
        "event_id",
        "true_az_tel", 
        "true_alt_tel", 
        "HillasReconstructor_core_x",
        "HillasReconstructor_core_y",
        "HillasReconstructor_h_max"
        ]

    # For MC we store some extra true parameters in DL2 for performance evaluation
    if ismc:
        features = features + [
            "true_az", 
            "true_alt",
            "true_energy",
            "log_true_energy",
            "true_core_x", 
            "true_core_y", 
            "true_h_first_int", 
            "true_x_max", 
            "true_shower_primary_id", 
            "true_camera_x",
            "true_camera_y",
        ]
    else:
        # For data we need to add also some more parameters
        features = features + [
            "local_time",
            "date",
            "equivalent_focal_length"
            ]

    if 'min_true_energy_cut' in params.keys():
        features.append('min_true_energy_cut')

    # Here we ask for mean(), but these quantities should be the same for both telescopes
    # NOTE: ..Maybe except true_camera_x, y? This should be checked!
    dl2 = params[features].groupby(["obs_id", "event_id"]).mean()
    return dl2


def find_coincidence_offset(
        tel1_file=None, tel2_files=None, outdir=None, 
        config=None, save_figures=False):
    """
    Finds time offset between both telescope timestamps
    which results in the most coincident events (for 
    given time window). This has to be used if the data does 
    not contain White Rabbit timestamps.

    Parameters
    ----------
    tel1_file: str
        tel1 file path
    tel2_files: list of strings
    outdir: str
        Output directory path
    config: dict
    save_figures: bool

    Returns
    -------
    float

    """

    dl1_data_t1 = load_dl1_sst1m(
        tel1_file, 
        tel=get_telescopes(tel1_file)[0], 
        config=config, 
        table='pandas', 
        check_finite=True, 
        quality_cuts=True
        )
    dl1_data_t2 = load_more_dl1_tables_mono(
        tel2_files, 
        config=config, 
        check_finite=True, 
        quality_cuts=True, 
        time_min=min(dl1_data_t1['local_time']), 
        time_max=max(dl1_data_t1['local_time'])
        )

    timestamp_tel1 = dl1_data_t1['local_time'].to_numpy()
    timestamp_tel2 = dl1_data_t2['local_time'].to_numpy()

    # Preselection, removing clearly not coincident events to speedup the code
    mask = (timestamp_tel2 > min(timestamp_tel1) - 1.) & (timestamp_tel2 < max(timestamp_tel1) + 1.)
    timestamp_tel2_masked = timestamp_tel2[mask]
    logging.info(f'Only {len(timestamp_tel2_masked)} TEL2 events out of {len(timestamp_tel2)} kept for the offset search.')

    n_coincidences = []

    time_offsets = np.arange(
        start=config["stereo"]["SlidingWindow"]["offset_search"]["time_start"],
        stop=config["stereo"]["SlidingWindow"]["offset_search"]["time_stop"],
        step=config["stereo"]["SlidingWindow"]["offset_search"]["time_step"],
    )

    window_half_width = config["stereo"]["SlidingWindow"]["window_half_width"]

    for offset in time_offsets:

        lolim = timestamp_tel1 + offset - window_half_width 
        uplim = timestamp_tel1 + offset + window_half_width

        cond_lolim = timestamp_tel2_masked >= lolim[:, np.newaxis]
        cond_uplim = timestamp_tel2_masked <= uplim[:, np.newaxis]

        mask = np.logical_and(cond_lolim, cond_uplim)

        n_coincidence = np.count_nonzero(mask)

        logging.info(f"time offset: {offset} --> {n_coincidence} events")

        n_coincidences.append([offset, n_coincidence])

    n_coincidences = np.array(n_coincidences)

    optimal_offset = n_coincidences[np.argmax(n_coincidences[:, 1]), 0]
    logging.info(f"Optimal offset found: {optimal_offset} seconds. Giving {max(n_coincidences[:, 1])} coincident events.")

    if save_figures:
        plt.figure()
        plt.plot(n_coincidences[:, 0], n_coincidences[:, 1])
        plt.grid()
        plt.xlabel('delta T [s]')
        plt.ylabel('N coincidences')
        plt.savefig(os.path.join(outdir, tel1_file.split('/')[-1].rstrip(".h5") + "_coincidences.png"))

    return optimal_offset


def make_dl1_stereo(
        dl1_file_tel1=None,
        input_dir_tel2=None,
        dl1_data_tel2=None,
        file_pattern='',
        output_path='./test.h5',
        config=None,
        time_offset=0):
    """
    Finds coincidet events in all passed tel2 DL1 files for 
    given tel1 DL1 file. Resulting DL1 file contains coincident
    events only in two tabs (
    /dl1/event/telescope/parameters/{tel_021/tel_022}). 
    Coincident events have the same event_id. There is also 
    geometrical stereo reconstruction applied in this step 
    providing extra columns to be used as features for 
    stereo reconstruction (tel_impact_distance, h_max). 
    This does not have to be run on MC, where the 
    coincident events are found on sim_telarray level.

    Parameters
    ----------
    dl1_file_tel1: str
        tel1 file path
    input_dir_tel2: str
    dl1_data_tel2: pandas.DataFrame
    file_pattern: str
    output_path: str
    config: dict
    time_offset: float

    Returns
    -------

    """

    logging.info('Making stereo DL1 file...')
    telescope_list = []
    telescope_list.append(int(get_telescopes(dl1_file_tel1)[0].split('_')[-1]))
    telescope_list.append(dl1_data_tel2['tel_id'][0])

    dl1_info = get_dl1_info(dl1_file_tel1)

    good_events = 0
    multiple_coincidences = 0
    mono_events = 0

    source = HDF5EventSource(dl1_file_tel1)
    current_t2_file = "" 
    t2_event_id = 0

    stereo_method = get_stereo_method(config)

    if stereo_method == 'SWATEventIDs' and ~dl1_info['swat_event_ids_used']:
        logging.warning('SWATEventIDs stereo method requested, but there are no SWAT event IDs in the DL1 file. WhiteRabbitClosest method used instead.')
        stereo_method = 'WhiteRabbitClosest'

    shower_processor  = ShowerProcessor(subarray=source.subarray, config=config)
    
    if stereo_method == "SlidingWindow":
        window = config["stereo"]["SlidingWindow"]["window_half_width"]
    elif stereo_method == "WhiteRabbitClosest":
        window = config["stereo"]["WhiteRabbitClosest"]["max_time_diff_ns"]
    elif stereo_method == 'SWATEventIDs':
        window = 0

    source_t2 = None
    evt_t2    = None

    if stereo_method == "WhiteRabbitClosest":
        dl1_data_tel1 = load_dl1_sst1m(dl1_file_tel1, tel='tel_021', table='pandas')
        t_t1_all = get_wr_timestamp(dl1_data_tel1)
        t_t2 = get_wr_timestamp(dl1_data_tel2)
    
    with DataWriter(source, 
                    output_path=output_path, 
                    overwrite        = True, 
                    write_showers    = True,
                    write_parameters = True,
                    write_images     = True,
                    ) as writer:

        
        for jj, evt in enumerate(source):
        
            logging.info('TEL1 event: ' + str(evt.index.event_id))

            if stereo_method == "SlidingWindow":
                t_t1 = evt.trigger.time.unix
                mask_coincidence = (dl1_data_tel2['local_time'] > t_t1 + time_offset - window) & (dl1_data_tel2['local_time'] < t_t1 + time_offset + window)
                tel2_event = dl1_data_tel2[mask_coincidence]

            elif stereo_method == "SWATEventIDs":
                ev_id1 = evt.index.event_id
                mask_coincidence = dl1_data_tel2['event_id'] == ev_id1
                tel2_event = dl1_data_tel2[mask_coincidence]

            elif stereo_method == "WhiteRabbitClosest":

                # Read ns timestamps from dl1 tab in tel1 file, which are not provided by HDF5EventSource()
                event_tel1_mask = dl1_data_tel1['event_id'] == evt.index.event_id

                t_t1 = t_t1_all[event_tel1_mask]
                diff = np.abs(t_t1 - t_t2)
                idx = np.argmin(diff)
                if diff[idx] < window:
                    mask_concidence = diff == diff[idx]
                    tel2_event = dl1_data_tel2[mask_concidence]
                else: tel2_event = np.array([])

            tel_1 = evt.trigger.tels_with_trigger[0]
            tel1_idx = telescope_list.index(tel_1)
            tel_2 = telescope_list[tel1_idx-1]

            if tel2_event.shape[0]==1:
                
                if stereo_method == "SlidingWindow":
                    logging.info('Coincident event found! Time difference [s]: ' + str(abs(tel2_event['local_time'].iloc[0] - t_t1 - time_offset)))
                elif stereo_method == "WhiteRabbitClosest":
                    logging.info('Coincident event found! Time difference [ns]: ' + str(diff[idx]))
                elif stereo_method == "SWATEventIDs":
                    logging.info('Coincident event with the same SWAT Event ID found!')
                logging.info('Desired TEL ' + str(tel_2) + ' event_id: ' + str(tel2_event['event_id'].iloc[0]))

                # Looking only for coincident events to tel1 events which survived cleaning (all hillas params are calculated)
                # NOTE: Original 'hilas_reco.check_parameters(parameters=evt.dl1.tel[tel_1].parameters).all()' does not work for me
                tel1_dl1_valid = (evt.dl1.tel[tel_1].is_valid) & (evt.dl1.tel[tel_1].parameters.hillas.intensity > 0)
                tel2_dl1_valid = tel2_event['camera_frame_hillas_intensity'].iloc[0] > 0

                if tel1_dl1_valid & tel2_dl1_valid:

                    # Date and obs_id extracted from R0 filenames are stored in DL1 (DANGEROUS).
                    # obs_id should contain {DATE}{file_no}
                    datestr  = str(tel2_event['date'].iloc[0])
                    filestr  = str(tel2_event['obs_id'].iloc[0]).replace(datestr,'')

                    dl1_t2_filename = glob.glob(input_dir_tel2+'/SST1M2*{}*{}*{}*.h5'.format(datestr, filestr, file_pattern))[0].split('/')[-1]
                    logging.info('Coincident events from TEL ' + str(tel_2) + ' was found in file: ' + dl1_t2_filename)
                    dl1_t2_file = os.path.join(input_dir_tel2, dl1_t2_filename)
                    
                    evt.trigger.tels_with_trigger = np.append(evt.trigger.tels_with_trigger, tel_2)

                    # Handling a first event from the file
                    if dl1_t2_file != current_t2_file:
                        current_t2_file = dl1_t2_file
                        if source_t2 is not None:
                            source_t2.close()
                        source_t2 = iter(HDF5EventSource(current_t2_file))
                        evt_t2 = next(source_t2)

                    if tel2_event['event_id'].iloc[0] != t2_event_id:
                        t2_event_id = tel2_event['event_id'].iloc[0]
                        
                        try:
                            while evt_t2.index.event_id != t2_event_id :
                                evt_t2 = next(source_t2)
                        except:
                            pass
                    else:
                        logging.warning('This event from TEL ' + str(tel_2) + 'was already used! Skipping')
                        multiple_coincidences += 1
                        continue

                    evt.dl2.tel[tel_2] = evt_t2.dl2.tel[tel_2]
                    evt.dl1.tel[tel_2] = evt_t2.dl1.tel[tel_2]

                    evt.trigger.tel[tel_2].time = evt_t2.trigger.tel[tel_2].time
                    evt.pointing.tel[tel_2].azimuth  = evt_t2.pointing.tel[tel_2].azimuth
                    evt.pointing.tel[tel_2].altitude = evt_t2.pointing.tel[tel_2].altitude

                    # add units to hillas parameters (important for stereo reconstruction with shower_processor)
                    evt = event_hillas_add_units(evt)

                    shower_processor(evt)

                    writer(evt)
                    good_events += 1

                else:
                    logging.info('Coincident event in one of the two telescopes did not survived cleaning.')
            elif tel2_event.shape[0] == 0:
                logging.info('No matching event found in TEL ' + str(tel_2) + ' data. Window: ' + str(window))
                mono_events += 1
            else:
                logging.warning('More potentialy coincident events found in TEL ' + str(tel_2) + ' data. Window: ' + str(window))
                multiple_coincidences += 1

    logging.info(f"Good coincident events found: {good_events}")
    logging.info(f"Multiple coincidences: {multiple_coincidences}")
    logging.info(f"TEL {tel_1} mono events: {mono_events}")
    source.close()

    # Propagate tel1 pedestals
    # NOTE: We propagate only tel1 pedestals
    # Both could be stored, but I am afraid that /pedestal/tel_XX
    # table is not compatible with ctapipe readers
    logging.info('Propagating pedestals from mono tel1 DL1 to stereo DL1..')
    try:
        pedestals = load_dl1_pedestals(dl1_file_tel1)
        write_dl1_pedestals(output_path, pedestal_table=pedestals)
    except:
        logging.warning('No pedestals found in tel1 DL1 file!')
