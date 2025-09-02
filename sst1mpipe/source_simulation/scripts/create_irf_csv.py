#!/usr/bin/env python3
"""
SST-1M DL2 to CSV Converter

This script converts SST-1M DL2 files to CSV format with proper background IRF calculation
using proton data and energy-dependent cut optimization.

Features:
- Uses proton simulations for background IRF (scientifically correct)
- Implements cut optimization to maximize significance per energy bin
- Generates IRF files for multiple cut efficiencies (0.4, 0.7, 0.9)
- Matches LST CSV format and structure

Usage:
    python create_irf_csv.py [--cut-efficiency 0.7] [--output-dir ../]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
from scipy.stats import skewnorm
from sklearn.metrics import mean_squared_error
from pathlib import Path
import logging
import json
import hashlib
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_file_checksum(filepath):
    """Calculate SHA256 checksum of a file for reproducibility tracking"""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.warning(f"Could not calculate checksum for {filepath}: {e}")
        return None

# CSV Schema Version
CSV_SCHEMA_VERSION = "1.0"

# Standard LST CSV IRF Schema
LST_IRF_SCHEMA = {
    'gamma': [
        'ZD_deg',           # Zenith angle in degrees
        'Etrue_min_TeV',    # True energy bin minimum (TeV)
        'Etrue_max_TeV',    # True energy bin maximum (TeV)
        'Aeff_m2',          # Effective area (m²)
        'emig_mu',          # Energy migration mean (log10(Ereco/Etrue))
        'emig_sigma',       # Energy migration sigma
        'emig_alpha',       # Energy migration skewness
        'emig_model',       # Energy migration model type
        'fit_mse',          # Fit quality metric
        'n_events'          # Number of events in bin
    ],
    'background': [
        'ZD_deg',           # Zenith angle in degrees
        'Ereco_min_TeV',    # Reconstructed energy bin minimum (TeV)
        'Ereco_max_TeV',    # Reconstructed energy bin maximum (TeV)
        'BckgRate_per_second', # Background rate (events/second)
        'Theta_cut_deg',    # Theta cut (degrees)
        'Gammaness_cut',    # Gammaness cut threshold
        'n_events'          # Number of events in bin
    ]
}

# DL2 Schema mapping - handles different DL2 producers
DL2_SCHEMA_MAP = {
    # Common variations for energy columns
    'true_energy': ['true_energy', 'mc_energy', 'energy_true', 'E_true'],
    'reco_energy': ['reco_energy', 'energy_reco', 'E_reco', 'reconstructed_energy'],
    
    # Common variations for classification
    'gammaness': ['gammaness', 'gh_score', 'gamma_score', 'classifier_score'],
    
    # Common variations for direction
    'reco_alt': ['reco_alt', 'alt_reco', 'reconstructed_altitude', 'altitude_reco'],
    'reco_az': ['reco_az', 'az_reco', 'reconstructed_azimuth', 'azimuth_reco'],
    'true_alt': ['true_alt', 'alt_true', 'mc_alt', 'altitude_true'],
    'true_az': ['true_az', 'az_true', 'mc_az', 'azimuth_true'],
    
    # Common variations for event weights
    'weight': ['weight', 'mc_weight', 'generator_weight', 'event_weight', 'w'],
}

# Default column names (fallback)
DEFAULT_COLUMNS = {
    'true_energy': 'true_energy',
    'reco_energy': 'reco_energy', 
    'gammaness': 'gammaness',
    'reco_alt': 'reco_alt',
    'reco_az': 'reco_az',
    'true_alt': 'true_alt',
    'true_az': 'true_az',
    'weight': 'weight',
}


def map_dl2_columns(available_columns, custom_weights_col=None):
    """
    Map available DL2 columns to standard names using schema mapping.
    
    Args:
        available_columns: List of column names in the DL2 file
        custom_weights_col: Custom weight column name specified by user
        
    Returns:
        dict: Mapping from standard names to actual column names
    """
    column_mapping = {}
    missing_columns = []
    
    # Handle custom weights column first
    if custom_weights_col:
        if custom_weights_col in available_columns:
            column_mapping['weight'] = custom_weights_col
            logger.info(f"Using custom weight column: {custom_weights_col}")
        else:
            logger.warning(f"Custom weight column '{custom_weights_col}' not found in DL2 file")
    
    for standard_name, possible_names in DL2_SCHEMA_MAP.items():
        # Skip weight mapping if custom column was specified
        if standard_name == 'weight' and custom_weights_col:
            continue
            
        found = False
        for possible_name in possible_names:
            if possible_name in available_columns:
                column_mapping[standard_name] = possible_name
                found = True
                logger.debug(f"Mapped {standard_name} -> {possible_name}")
                break
        
        if not found:
            # Use default name if available
            if DEFAULT_COLUMNS[standard_name] in available_columns:
                column_mapping[standard_name] = DEFAULT_COLUMNS[standard_name]
                logger.debug(f"Mapped {standard_name} -> {DEFAULT_COLUMNS[standard_name]} (default)")
            else:
                missing_columns.append(standard_name)
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {available_columns}")
        logger.info("This may cause issues with theta2 calculation or other features")
    
    return column_mapping


def detect_energy_units(data, column_mapping, force_unit=None):
    """
    Detect energy units in the data using heuristics and metadata.
    
    Args:
        data: DL2 data dictionary
        column_mapping: Column mapping dictionary
        force_unit: Force specific unit ('TeV' or 'GeV'), bypassing detection
        
    Returns:
        str: Detected or forced unit ('TeV' or 'GeV')
    """
    # If unit is forced, use it directly
    if force_unit:
        if force_unit.upper() not in ['TEV', 'GEV']:
            raise ValueError(f"Invalid force_unit '{force_unit}'. Must be 'TeV' or 'GeV'")
        logger.info(f"Using forced energy unit: {force_unit}")
        return force_unit.upper()
    
    # Try to get energy column
    energy_col = None
    for col_name in ['true_energy', 'reco_energy']:
        if col_name in column_mapping:
            energy_col = column_mapping[col_name]
            break
    
    if energy_col is None or energy_col not in data.dtype.names:
        logger.warning("Cannot detect energy units - no energy column found")
        return 'TeV'  # Default assumption
    
    energy_data = data[energy_col]
    
    # Remove any infinite or NaN values for analysis
    valid_energy = energy_data[np.isfinite(energy_data)]
    if len(valid_energy) == 0:
        logger.warning("No valid energy values found for unit detection")
        return 'TeV'  # Default assumption
    
    # Calculate statistics on valid data
    min_energy = np.min(valid_energy)
    max_energy = np.max(valid_energy)
    median_energy = np.median(valid_energy)
    mean_energy = np.mean(valid_energy)
    
    logger.info(f"Energy statistics: min={min_energy:.3f}, max={max_energy:.1f}, median={median_energy:.3f}, mean={mean_energy:.3f}")
    
    # Improved heuristics with multiple criteria
    tev_score = 0
    gev_score = 0
    
    # Criterion 1: Median energy range
    if median_energy < 1.0:
        tev_score += 2  # Strong TeV indicator
    elif median_energy > 1000:
        gev_score += 2  # Strong GeV indicator
    elif 1.0 <= median_energy <= 100:
        # Ambiguous range - check other criteria
        pass
    
    # Criterion 2: Maximum energy
    if max_energy < 1000:
        tev_score += 1
    elif max_energy > 10000:
        gev_score += 1
    
    # Criterion 3: Energy distribution percentiles
    p25 = np.percentile(valid_energy, 25)
    p75 = np.percentile(valid_energy, 75)
    
    if p75 < 10:
        tev_score += 1
    elif p25 > 100:
        gev_score += 1
    
    # Criterion 4: Check for typical CTA/SST-1M energy ranges
    # SST-1M typically operates in 0.1-100 TeV range
    if min_energy >= 0.01 and max_energy <= 1000:
        tev_score += 1
    elif min_energy >= 10 and max_energy <= 1000000:
        gev_score += 1
    
    # Make decision based on scores
    if tev_score > gev_score:
        detected_unit = 'TeV'
        confidence = "high" if tev_score >= 3 else "medium"
    elif gev_score > tev_score:
        detected_unit = 'GeV'
        confidence = "high" if gev_score >= 3 else "medium"
    else:
        # Tie or low confidence - use default with warning
        detected_unit = 'TeV'
        confidence = "low"
        logger.warning(f"Energy unit detection inconclusive (TeV score: {tev_score}, GeV score: {gev_score})")
        logger.warning("Using TeV as default. Use --energy-unit to force correct unit if needed.")
    
    logger.info(f"Detected energy unit: {detected_unit} (confidence: {confidence}, TeV score: {tev_score}, GeV score: {gev_score})")
    
    # Additional validation warning for edge cases
    if detected_unit == 'TeV' and max_energy > 1000:
        logger.warning(f"Detected TeV but max energy {max_energy:.1f} seems high for TeV. Consider using --energy-unit GeV")
    elif detected_unit == 'GeV' and max_energy < 100:
        logger.warning(f"Detected GeV but max energy {max_energy:.1f} seems low for GeV. Consider using --energy-unit TeV")
    
    return detected_unit


def convert_energy_units(data, column_mapping, from_unit, to_unit='TeV'):
    """
    Convert energy units in the data.
    
    Args:
        data: DL2 data dictionary
        column_mapping: Column mapping dictionary
        from_unit: Source unit ('TeV' or 'GeV')
        to_unit: Target unit ('TeV' or 'GeV')
    """
    if from_unit == to_unit:
        return
    
    conversion_factor = 1.0
    if from_unit == 'GeV' and to_unit == 'TeV':
        conversion_factor = 1e-3  # GeV to TeV
    elif from_unit == 'TeV' and to_unit == 'GeV':
        conversion_factor = 1e3   # TeV to GeV
    
    logger.info(f"Converting energy units: {from_unit} → {to_unit} (factor: {conversion_factor})")
    
    # Convert all energy columns
    energy_columns = ['true_energy', 'reco_energy']
    for col_name in energy_columns:
        if col_name in column_mapping:
            actual_col = column_mapping[col_name]
            if actual_col in data.dtype.names:
                data[actual_col] = data[actual_col] * conversion_factor
                logger.debug(f"Converted {actual_col}: {from_unit} → {to_unit}")


def validate_dl2_schema(column_mapping):
    """
    Validate that essential columns are available.
    
    Args:
        column_mapping: Dictionary mapping standard names to actual column names
        
    Returns:
        bool: True if schema is valid, False otherwise
    """
    essential_columns = ['true_energy', 'reco_energy', 'gammaness']
    missing_essential = [col for col in essential_columns if col not in column_mapping]
    
    if missing_essential:
        logger.error(f"Missing essential columns: {missing_essential}")
        logger.error("Cannot proceed without these columns")
        return False
    
    optional_columns = ['reco_alt', 'reco_az', 'true_alt', 'true_az', 'weight']
    missing_optional = [col for col in optional_columns if col not in column_mapping]
    
    if missing_optional:
        logger.warning(f"Missing optional columns: {missing_optional}")
        if 'weight' in missing_optional:
            logger.warning("Event weights not found - using unweighted analysis (may be inaccurate)")
        if any(col in missing_optional for col in ['reco_alt', 'reco_az', 'true_alt', 'true_az']):
            logger.warning("Theta2 calculation will be disabled")
    
    return True


def load_dl2_data(filepath, custom_weights_col=None, force_energy_unit=None):
    """Load DL2 data from HDF5 file with flexible schema mapping"""
    logger.info(f"Loading {filepath.name}...")
    
    with h5py.File(filepath, 'r') as f:
        # Try different possible data paths
        possible_paths = [
            'dl2/event/telescope/parameters/stereo',
            'dl2/event/telescope/parameters/tel_001',
            'dl2/event/telescope/parameters/tel_002', 
            'dl2/event/telescope/parameters/tel_003',
            'dl2/event/telescope/parameters/tel_004',
            'dl2/event/telescope/parameters/tel_005',
            'dl2/event/telescope/parameters/tel_006',
            'dl2/event/telescope/parameters/tel_007',
            'dl2/event/telescope/parameters/tel_008',
            'dl2/event/telescope/parameters/tel_009',
            'dl2/event/telescope/parameters/tel_010',
        ]
        
        data_path = None
        for path in possible_paths:
            if path in f:
                data_path = path
                break
        
        if data_path is None:
            logger.error(f"Could not find DL2 data in any of the expected paths:")
            for path in possible_paths:
                logger.error(f"  - {path}")
            logger.error("Available paths in file:")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    logger.error(f"  - {name}")
            f.visititems(print_structure)
            raise ValueError(f"Could not find DL2 data in {filepath.name}")
        
        logger.info(f"Using data path: {data_path}")
        
        # Load the data
        dl2_data = f[data_path][:]
        logger.info(f"Loaded DL2 data: shape {dl2_data.shape}")
        
        # Keep as structured array for consistent access patterns
        logger.info(f"Available columns: {list(dl2_data.dtype.names)}")
        
        # Map columns to standard names
        column_mapping = map_dl2_columns(list(dl2_data.dtype.names), custom_weights_col)
        
        # Validate schema
        if not validate_dl2_schema(column_mapping):
            raise ValueError(f"Invalid DL2 schema in {filepath.name}")
        
        # Detect and convert energy units (with optional force override)
        detected_unit = detect_energy_units(dl2_data, column_mapping, force_energy_unit)
        if detected_unit != 'TeV':
            logger.warning(f"Energy units detected/forced as {detected_unit}, converting to TeV")
            convert_energy_units(dl2_data, column_mapping, detected_unit, 'TeV')
            logger.info("Energy units converted to TeV for consistent processing")
        else:
            if force_energy_unit:
                logger.info(f"Energy units forced as {force_energy_unit} and confirmed as TeV")
            else:
                logger.info("Energy units detected and confirmed as TeV")
        
        # Create a new structured array with column mapping stored as attributes
        result = dl2_data.copy()
        result._column_mapping = column_mapping
        
        logger.info(f"Column mapping: {column_mapping}")
        return result


def extract_mc_parameters(filepath, cli_overrides=None):
    """Extract MC simulation parameters from DL2 file headers with CLI overrides and validation"""
    # Initialize with None values - no dangerous defaults!
    mc_params = {
        'livetime': None,
        'thrown_area': None,
        'solid_angle': None,
        'spectral_index': None,
        'energy_min': None,
        'energy_max': None,
        'energy_unit': 'TeV',  # Default energy unit
    }
    
    # Apply CLI overrides first (highest priority)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                mc_params[key] = value
                logger.info(f"Using CLI override for {key}: {value}")
    
    # Try to extract from file if not overridden by CLI
    try:
        with h5py.File(filepath, 'r') as f:
            # Try to find MC parameters in various possible locations
            possible_paths = [
                'simulation/run_config',
                'simulation/run_parameters', 
                'dl1/event/telescope/parameters/tel_001',
                'dl2/event/telescope/parameters/tel_001'
            ]
            
            for path in possible_paths:
                if path in f:
                    group = f[path]
                    if 'attrs' in dir(group):
                        attrs = group.attrs
                        for key in attrs.keys():
                            if 'livetime' in key.lower() and mc_params['livetime'] is None:
                                mc_params['livetime'] = float(attrs[key])
                            elif 'area' in key.lower() and mc_params['thrown_area'] is None:
                                mc_params['thrown_area'] = float(attrs[key])
                            elif 'solid_angle' in key.lower() and mc_params['solid_angle'] is None:
                                mc_params['solid_angle'] = float(attrs[key])
                            elif 'spectral_index' in key.lower() and mc_params['spectral_index'] is None:
                                mc_params['spectral_index'] = float(attrs[key])
                            elif 'energy_min' in key.lower() and mc_params['energy_min'] is None:
                                mc_params['energy_min'] = float(attrs[key])
                            elif 'energy_max' in key.lower() and mc_params['energy_max'] is None:
                                mc_params['energy_max'] = float(attrs[key])
                            elif 'energy_unit' in key.lower() or 'unit' in key.lower():
                                mc_params['energy_unit'] = str(attrs[key])
        
        # Convert MC energy parameters to TeV if needed
        mc_energy_unit = mc_params.get('energy_unit', 'TeV')
        if mc_energy_unit != 'TeV':
            logger.info(f"Converting MC energy parameters from {mc_energy_unit} to TeV")
            if mc_energy_unit == 'GeV':
                if mc_params['energy_min'] is not None:
                    mc_params['energy_min'] *= 1e-3  # GeV to TeV
                if mc_params['energy_max'] is not None:
                    mc_params['energy_max'] *= 1e-3  # GeV to TeV
            mc_params['energy_unit'] = 'TeV'
        
    except Exception as e:
        logger.warning(f"Could not extract MC parameters from {filepath.name}: {e}")
    
    # Validate that all critical parameters are available
    critical_params = ['livetime', 'thrown_area', 'solid_angle', 'spectral_index']
    missing_critical = [param for param in critical_params if mc_params[param] is None]
    
    if missing_critical:
        error_msg = (f"CRITICAL: Missing MC parameters: {missing_critical}. "
                    f"Cannot proceed with normalization without these values. "
                    f"Use CLI arguments (--mc-livetime, --mc-thrown-area, etc.) to provide them.")
        logger.error(error_msg)
        logger.error("Available CLI arguments:")
        logger.error("  --mc-livetime SECONDS        # MC livetime in seconds")
        logger.error("  --mc-thrown-area M2          # MC thrown area in m²")
        logger.error("  --mc-solid-angle STERADIANS  # MC solid angle in steradians")
        logger.error("  --mc-spectral-index INDEX    # MC spectral index")
        logger.error("  --mc-energy-min TEV          # MC minimum energy in TeV")
        logger.error("  --mc-energy-max TEV          # MC maximum energy in TeV")
        raise ValueError(error_msg)
    
    # Validate parameter ranges
    if mc_params['livetime'] <= 0:
        raise ValueError(f"Invalid MC livetime: {mc_params['livetime']} seconds (must be > 0)")
    if mc_params['thrown_area'] <= 0:
        raise ValueError(f"Invalid MC thrown area: {mc_params['thrown_area']} m² (must be > 0)")
    if mc_params['solid_angle'] <= 0:
        raise ValueError(f"Invalid MC solid angle: {mc_params['solid_angle']} sr (must be > 0)")
    
    # Warn about potentially suspicious values
    if mc_params['livetime'] == 3600.0:
        logger.warning("⚠️  MC livetime is exactly 1 hour (3600s) - verify this is correct")
    if mc_params['thrown_area'] == 1e6:
        logger.warning("⚠️  MC thrown area is exactly 1 km² (1e6 m²) - verify this is correct")
    if abs(mc_params['solid_angle'] - 2 * np.pi) < 0.01:
        logger.warning("⚠️  MC solid angle is exactly 2π sr - verify this is correct")
    
    logger.info(f"Final MC parameters: {mc_params}")
    return mc_params


def apply_cuts(data, gammaness_cut=0.5, theta2_cut_deg2=0.1):
    """Apply standard cuts to the data
    
    Args:
        theta2_cut_deg2: Theta squared cut in deg² (internal variable)
    """
    # Get column mapping from data
    column_mapping = getattr(data, '_column_mapping', {})
    
    # Use mapped column names - fail fast if mapping failed
    mc_energy_col = column_mapping.get('true_energy')
    reco_energy_col = column_mapping.get('reco_energy')
    gammaness_col = column_mapping.get('gammaness')
    
    # Validate essential columns exist
    missing_essential = []
    if not mc_energy_col or mc_energy_col not in data.dtype.names:
        missing_essential.append(f"true_energy (mapped to {mc_energy_col})")
    if not gammaness_col or gammaness_col not in data.dtype.names:
        missing_essential.append(f"gammaness (mapped to {gammaness_col})")
    
    if missing_essential:
        raise ValueError(f"Missing essential columns in data: {missing_essential}")
    
    # Handle reco_energy consistently - use true_energy as fallback if missing
    if not reco_energy_col or reco_energy_col not in data.dtype.names:
        logger.warning(f"reco_energy column not found, using true_energy as fallback")
        reco_energy_col = mc_energy_col
    
    # Calculate theta2 from reconstructed and true directions using great-circle distance
    reco_alt_col = column_mapping.get('reco_alt')
    reco_az_col = column_mapping.get('reco_az')
    true_alt_col = column_mapping.get('true_alt')
    true_az_col = column_mapping.get('true_az')
    
    # Validate that all direction columns are available
    missing_direction_cols = []
    if not reco_alt_col or reco_alt_col not in data.dtype.names:
        missing_direction_cols.append(f"reco_alt (mapped to {reco_alt_col})")
    if not reco_az_col or reco_az_col not in data.dtype.names:
        missing_direction_cols.append(f"reco_az (mapped to {reco_az_col})")
    if not true_alt_col or true_alt_col not in data.dtype.names:
        missing_direction_cols.append(f"true_alt (mapped to {true_alt_col})")
    if not true_az_col or true_az_col not in data.dtype.names:
        missing_direction_cols.append(f"true_az (mapped to {true_az_col})")
    
    if missing_direction_cols:
        raise ValueError(f"Cannot calculate theta² without direction columns: {missing_direction_cols}")
    
    # Convert to radians
    reco_alt_rad = np.radians(data[reco_alt_col])
    reco_az_rad = np.radians(data[reco_az_col])
    true_alt_rad = np.radians(data[true_alt_col])
    true_az_rad = np.radians(data[true_az_col])
    
    # Calculate great-circle angular distance using haversine formula
    # Convert alt/az to unit vectors in Cartesian coordinates
    reco_x = np.cos(reco_alt_rad) * np.cos(reco_az_rad)
    reco_y = np.cos(reco_alt_rad) * np.sin(reco_az_rad)
    reco_z = np.sin(reco_alt_rad)
    
    true_x = np.cos(true_alt_rad) * np.cos(true_az_rad)
    true_y = np.cos(true_alt_rad) * np.sin(true_az_rad)
    true_z = np.sin(true_alt_rad)
    
    # Dot product of unit vectors
    dot_product = reco_x * true_x + reco_y * true_y + reco_z * true_z
    
    # Clamp to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Angular distance in radians
    theta_rad = np.arccos(dot_product)
    
    # Convert to degrees squared
    theta2_deg = np.degrees(theta_rad)**2  # Note: variable name is theta2_deg but contains θ² values
    
    logger.info(f"Using columns: MC energy={mc_energy_col}, reco energy={reco_energy_col}, gammaness={gammaness_col}")
    
    mask = np.ones(len(data[mc_energy_col]), dtype=bool)
    
    # Apply gammaness cut
    if gammaness_col in data:
        mask &= data[gammaness_col] > gammaness_cut
        logger.info(f"Applied gammaness cut: {gammaness_col} > {gammaness_cut}")
    
    # Apply theta2 cut (internal variable is θ² in deg²)
    mask &= theta2_deg < theta2_cut_deg2
    logger.info(f"Applied theta2 cut: theta2 < {theta2_cut_deg2} deg²")
    
    # Apply energy cuts (reasonable range in TeV - units should be converted by load_dl2_data)
    energy_min_tev = 0.05  # TeV
    energy_max_tev = 100.0  # TeV
    mask &= (data[mc_energy_col] > energy_min_tev) & (data[mc_energy_col] < energy_max_tev)
    
    # Apply reco energy cuts if reco_energy is available (not using fallback)
    if reco_energy_col != mc_energy_col:  # Only if we have actual reco_energy
        mask &= (data[reco_energy_col] > energy_min_tev) & (data[reco_energy_col] < energy_max_tev)
    
    logger.info(f"Applied cuts: {np.sum(mask)}/{len(mask)} events passed")
    return mask, mc_energy_col, reco_energy_col, gammaness_col, theta2_deg


def create_energy_bins(n_bins=20, min_energy=0.05, max_energy=100):
    """Create fixed logarithmic energy bins that are consistent across all runs"""
    log_min = np.log10(min_energy)
    log_max = np.log10(max_energy)
    log_bins = np.logspace(log_min, log_max, n_bins + 1)
    return log_bins


def get_global_energy_bins(n_bins=20, min_energy=0.1, max_energy=100.0):
    """Get the global energy bin definition used consistently across all processing"""
    # These should match the energy range of your data
    # Adjust these values based on your actual data range
    return create_energy_bins(n_bins=n_bins, min_energy=min_energy, max_energy=max_energy)


def calculate_background_rate(proton_events, energy_mask, gammaness_mask, theta2_mask, theta2_cut_deg2, zd_deg, mc_params, column_mapping):
    """
    Calculate physically correct background rate with proper MC normalization using event weights.
    
    Background rate = Σ(w_i * weight_i) / (MC_livetime * solid_angle_acceptance)
    
    where:
    - w_i: event weight from DL2 file (generator weight)
    - weight_i: additional physics weight including:
      - Proton spectral reweighting to physical CR spectrum
      - MC thrown area normalization
    - solid_angle_acceptance: π * θ² (where θ² is the theta cut in deg²)
    
    Returns:
        Background rate in events/s (not events/s/deg²)
    """
    
    # Get events passing all cuts
    passing_events = energy_mask & gammaness_mask & theta2_mask
    
    if np.sum(passing_events) == 0:
        return 0.0
    
    # Extract MC parameters
    mc_livetime = mc_params['livetime']
    mc_thrown_area = mc_params['thrown_area']
    mc_solid_angle = mc_params['solid_angle']
    mc_spectral_index = mc_params['spectral_index']
    
    # Physical cosmic ray spectrum parameters (Gaisser model)
    # dN/dE ∝ E^(-2.7) for protons
    cr_spectral_index = 2.7
    
    # Get energies and event weights of passing events
    reco_energy_col = column_mapping.get('reco_energy')
    if not reco_energy_col or reco_energy_col not in proton_events.dtype.names:
        raise ValueError(f"reco_energy column not found in proton data for background rate calculation")
    passing_energies = proton_events[reco_energy_col][passing_events]
    
    # Get event weights from DL2 file
    weight_col = column_mapping.get('weight')
    if weight_col and weight_col in proton_events.dtype.names:
        event_weights = proton_events[weight_col][passing_events]
        logger.debug(f"Using DL2 event weights: min={np.min(event_weights):.6f}, max={np.max(event_weights):.6f}, mean={np.mean(event_weights):.6f}")
    else:
        # Fallback to unweighted (all weights = 1)
        event_weights = np.ones(len(passing_energies))
        logger.debug("No DL2 event weights found - using unweighted analysis")
    
    # Calculate total weighted background rate
    total_weight = 0.0
    for energy, event_weight in zip(passing_energies, event_weights):
        # 1. Spectral reweighting: weight ∝ E^(-2.7) / MC_spectrum
        # If MC was thrown with flat spectrum (index=0), weight ∝ E^(-2.7)
        # If MC was thrown with different index, weight ∝ E^(-2.7) / E^(-MC_index)
        if mc_spectral_index == 0.0:  # Flat spectrum
            spectral_weight = energy**(-cr_spectral_index)
        else:
            spectral_weight = energy**(-cr_spectral_index) / energy**(-mc_spectral_index)
        
        # 2. MC normalization weight
        # Account for MC thrown area and solid angle
        mc_weight = 1.0 / (mc_thrown_area * mc_solid_angle)
        
        # Total weight per event = DL2_weight * physics_weight
        # Note: We do NOT multiply by solid angle here - that's handled in the final normalization
        physics_weight = spectral_weight * mc_weight
        total_event_weight = event_weight * physics_weight
        total_weight += total_event_weight
    
    # Background rate in events per second
    # The solid angle acceptance is handled in the final normalization
    # theta2_cut_deg2 is θ² in deg² (internal variable)
    solid_angle_acceptance_deg2 = np.pi * theta2_cut_deg2  # deg²
    bkg_rate = total_weight / (mc_livetime * solid_angle_acceptance_deg2)
    
    logger.debug(f"Background rate calculation: {len(passing_energies)} events, "
                f"total_weight={total_weight:.6f}, "
                f"solid_angle={solid_angle_acceptance_deg2:.6f} deg², "
                f"rate={bkg_rate:.6f} events/s")
    
    return bkg_rate


def calculate_effective_area(gamma_events, bin_mask, final_mask, mc_params, etrue_min, etrue_max, column_mapping):
    """
    Calculate proper effective area with MC normalization using event weights.
    
    Aeff = (Σw_passing / Σw_thrown_in_bin) * A_thrown
    
    where:
    - Σw_passing: sum of weights for events passing cuts in this energy bin
    - Σw_thrown_in_bin: sum of weights for all events in this energy bin
    - A_thrown: MC thrown area
    
    Note: With standard generator weights per thrown phase space, no energy correction
    factor is needed. The weights already account for the phase space density.
    """
    
    # Get events in this energy bin (before cuts)
    bin_events = gamma_events[bin_mask]
    n_thrown = len(bin_events)
    
    if n_thrown == 0:
        return 0.0
    
    # Get event weights
    weight_col = column_mapping.get('weight')
    if weight_col and weight_col in gamma_events.dtype.names:
        # Use actual event weights
        weights_thrown = gamma_events[weight_col][bin_mask]
        weights_passing = gamma_events[weight_col][final_mask]
        sum_weights_thrown = np.sum(weights_thrown)
        sum_weights_passing = np.sum(weights_passing)
        logger.debug(f"Using event weights: sum_thrown={sum_weights_thrown:.2f}, sum_passing={sum_weights_passing:.2f}")
    else:
        # Fallback to unweighted (all weights = 1)
        sum_weights_thrown = n_thrown
        sum_weights_passing = np.sum(final_mask)
        logger.debug("No event weights found - using unweighted analysis")
    
    # MC parameters
    mc_thrown_area = mc_params['thrown_area']  # m²
    
    # Calculate effective area using weighted efficiency
    # Aeff = (Σw_passing / Σw_thrown_in_bin) * A_thrown
    weighted_efficiency = sum_weights_passing / sum_weights_thrown if sum_weights_thrown > 0 else 0.0
    
    aeff = weighted_efficiency * mc_thrown_area
    
    logger.debug(f"Effective area calculation: "
                f"n_thrown={n_thrown}, n_passing={np.sum(final_mask)}, "
                f"weighted_efficiency={weighted_efficiency:.4f}, "
                f"A_thrown={mc_thrown_area:.0f} m², "
                f"Aeff={aeff:.2f} m²")
    
    return aeff


def fit_energy_migration(gamma_events, bin_mask, final_mask, etrue_min, etrue_max, column_mapping):
    """
    Fit energy migration model using skew-normal distribution.
    
    Fits log10(Ereco/Etrue) distribution with skewnorm parameters:
    - mu: location parameter
    - sigma: scale parameter  
    - alpha: shape parameter (skewness)
    
    Returns fitted parameters and MSE.
    """
    
    # Get column mapping - fail fast if mapping failed
    true_energy_col = column_mapping.get('true_energy')
    reco_energy_col = column_mapping.get('reco_energy')
    
    # Validate columns exist
    if not true_energy_col or true_energy_col not in gamma_events.dtype.names:
        logger.warning(f"true_energy column not found for energy migration fit")
        return {
            'mu': 0.0,
            'sigma': 0.1,
            'alpha': 0.0
        }, 0.0
    
    if not reco_energy_col or reco_energy_col not in gamma_events.dtype.names:
        logger.warning(f"reco_energy column not found for energy migration fit")
        return {
            'mu': 0.0,
            'sigma': 0.1,
            'alpha': 0.0
        }, 0.0
    
    # Get events passing cuts in this energy bin
    passing_events = gamma_events[final_mask]
    
    if len(passing_events) < 10:
        # Not enough events for fitting, return default values
        return {
            'mu': 0.0,
            'sigma': 0.1,
            'alpha': 0.0
        }, 0.0
    
    # Get true and reconstructed energies
    etrue = passing_events[true_energy_col]
    ereco = passing_events[reco_energy_col]
    
    # Calculate log10(Ereco/Etrue) - this is what we fit
    log_energy_ratio = np.log10(ereco / etrue)
    
    # Remove any infinite or NaN values
    valid_mask = np.isfinite(log_energy_ratio)
    if np.sum(valid_mask) < 5:
        return {
            'mu': 0.0,
            'sigma': 0.1,
            'alpha': 0.0
        }, 0.0
    
    log_energy_ratio = log_energy_ratio[valid_mask]
    
    try:
        # Fit skew-normal distribution using scipy.stats.skewnorm.fit
        # This uses maximum likelihood estimation
        
        # Initial parameter guesses
        mu_init = np.mean(log_energy_ratio)
        sigma_init = np.std(log_energy_ratio)
        alpha_init = 0.0  # Start with symmetric distribution
        
        # Estimate initial skewness from data
        data_skewness = np.mean(((log_energy_ratio - mu_init) / sigma_init)**3)
        if np.abs(data_skewness) > 0.1:
            alpha_init = np.sign(data_skewness) * min(10.0, np.abs(data_skewness) * 5)
        
        # Fit using maximum likelihood estimation
        params = skewnorm.fit(log_energy_ratio, 
                             loc=mu_init, 
                             scale=sigma_init, 
                             a=alpha_init)
        
        alpha_fit, mu_fit, sigma_fit = params
        
        # Calculate MSE between fitted and observed distribution
        # Create histogram of observed data
        hist, bin_edges = np.histogram(log_energy_ratio, bins=20, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate fitted PDF values using the fitted parameters
        fitted_pdf = skewnorm.pdf(bin_centers, alpha_fit, loc=mu_fit, scale=sigma_fit)
        
        # Calculate MSE between observed and fitted distributions
        mse = mean_squared_error(hist, fitted_pdf)
        
        logger.debug(f"Energy migration fit: "
                    f"mu={mu_fit:.4f}, sigma={sigma_fit:.4f}, alpha={alpha_fit:.4f}, "
                    f"MSE={mse:.6f}, n_events={len(log_energy_ratio)}")
        
        return {
            'mu': mu_fit,
            'sigma': sigma_fit,
            'alpha': alpha_fit
        }, mse
        
    except Exception as e:
        logger.warning(f"Energy migration fitting failed: {e}")
        # Return default values if fitting fails
        return {
            'mu': 0.0,
            'sigma': 0.1,
            'alpha': 0.0
        }, 1.0


def optimize_cuts_per_energy_bin(gamma_data, proton_data, energy_bin, reco_energy_col, gammaness_col, target_efficiency, gamma_grid, theta_grid, efficiency_tolerance=0.02):
    """Optimize cuts for a specific energy bin using grid search to achieve target gamma efficiency while maximizing S/√B
    
    IMPORTANT: This function assumes gamma_data contains on-axis point source events.
    For proper S/√B calculation, the signal should be from a point source, not diffuse.
    
    Args:
        gamma_data: Point source gamma events (preferred) or diffuse gamma events
        proton_data: Proton background events
        energy_bin: Energy bin (emin, emax) in TeV
        reco_energy_col: Column name for reconstructed energy
        gammaness_col: Column name for gammaness/classification
        target_efficiency: Target gamma efficiency (0.4, 0.7, 0.9)
        gamma_grid: List of gammaness cut values to test
        theta_grid: List of theta² cut values (in deg²) to test
        efficiency_tolerance: Tolerance for efficiency matching (default: 0.02)
    
    Returns:
        gammaness_threshold: Gammaness cut value
        theta2_threshold_deg2: Theta squared threshold in deg² (internal variable)
    """
    
    # Get events in this energy bin
    gamma_mask = (gamma_data[reco_energy_col] >= energy_bin[0]) & (gamma_data[reco_energy_col] < energy_bin[1])
    proton_mask = (proton_data[reco_energy_col] >= energy_bin[0]) & (proton_data[reco_energy_col] < energy_bin[1])
    
    if np.sum(gamma_mask) < 10 or np.sum(proton_mask) < 10:
        return 0.5, 0.1  # Default cuts if not enough events
    
    # Get gamma and proton events in this energy bin
    gamma_events = gamma_data[gamma_mask]
    proton_events = proton_data[proton_mask]
    
    gamma_gammaness = gamma_events[gammaness_col]
    gamma_theta2 = gamma_events['theta2_deg']
    proton_gammaness = proton_events[gammaness_col]
    proton_theta2 = proton_events['theta2_deg']
    
    # Grid search over all combinations
    best_significance = -1
    best_gammaness = 0.5
    best_theta2 = 0.1
    best_efficiency = 0.0
    valid_cuts = []
    
    for gammaness_cut in gamma_grid:
        for theta2_cut_deg2 in theta_grid:
            # Apply cuts to gamma events
            gamma_passing = (gamma_gammaness >= gammaness_cut) & (gamma_theta2 <= theta2_cut_deg2)
            gamma_efficiency = np.sum(gamma_passing) / len(gamma_events)
            
            # Check if efficiency is within tolerance
            if abs(gamma_efficiency - target_efficiency) <= efficiency_tolerance:
                # Apply same cuts to proton events
                proton_passing = (proton_gammaness >= gammaness_cut) & (proton_theta2 <= theta2_cut_deg2)
            
            # Calculate significance
            signal = np.sum(gamma_passing)
            background = np.sum(proton_passing)
            
            if background > 0:
                significance = signal / np.sqrt(background)
            else:
                significance = signal if signal > 0 else 0
            
                valid_cuts.append({
                    'gammaness': gammaness_cut,
                    'theta2': theta2_cut_deg2,
                    'efficiency': gamma_efficiency,
                    'significance': significance,
                    'signal': signal,
                    'background': background
                })
                
                # Track best significance
            if significance > best_significance:
                best_significance = significance
                best_gammaness = gammaness_cut
                    best_theta2 = theta2_cut_deg2
                    best_efficiency = gamma_efficiency
    
    # If no cuts meet efficiency tolerance, find the closest one
    if not valid_cuts:
        logger.warning(f"No cuts found within efficiency tolerance {efficiency_tolerance} for target {target_efficiency}")
        
        # Find cut with closest efficiency
        min_efficiency_diff = float('inf')
        for gammaness_cut in gamma_grid:
            for theta2_cut_deg2 in theta_grid:
                gamma_passing = (gamma_gammaness >= gammaness_cut) & (gamma_theta2 <= theta2_cut_deg2)
                gamma_efficiency = np.sum(gamma_passing) / len(gamma_events)
                efficiency_diff = abs(gamma_efficiency - target_efficiency)
                
                if efficiency_diff < min_efficiency_diff:
                    min_efficiency_diff = efficiency_diff
                    best_gammaness = gammaness_cut
                    best_theta2 = theta2_cut_deg2
                    best_efficiency = gamma_efficiency
        
        # Calculate significance for the closest cut
        gamma_passing = (gamma_gammaness >= best_gammaness) & (gamma_theta2 <= best_theta2)
        proton_passing = (proton_gammaness >= best_gammaness) & (proton_theta2 <= best_theta2)
        signal = np.sum(gamma_passing)
        background = np.sum(proton_passing)
        best_significance = signal / np.sqrt(background) if background > 0 else (signal if signal > 0 else 0)
    
    logger.info(f"Energy bin {energy_bin[0]:.3f}-{energy_bin[1]:.3f} TeV: "
                f"target efficiency={target_efficiency:.2f}, actual={best_efficiency:.2f}, "
                f"gammaness_cut={best_gammaness:.3f}, theta2_cut_deg2={best_theta2:.3f}, "
                f"significance={best_significance:.2f}, valid_cuts={len(valid_cuts)}")
    
    return best_gammaness, best_theta2


def process_gamma_irf(data, zd_deg, target_efficiency, mc_params, gamma_grid, theta_grid):
    """Process gamma data to create IRF parameters using efficiency-based cuts"""
    
    # Get column mapping from data
    column_mapping = getattr(data, '_column_mapping', {})
    
    # Use mapped column names - fail fast if mapping failed
    mc_energy_col = column_mapping.get('true_energy')
    reco_energy_col = column_mapping.get('reco_energy')
    gammaness_col = column_mapping.get('gammaness')
    
    # Validate essential columns exist
    missing_essential = []
    if not mc_energy_col or mc_energy_col not in data.dtype.names:
        missing_essential.append(f"true_energy (mapped to {mc_energy_col})")
    if not gammaness_col or gammaness_col not in data.dtype.names:
        missing_essential.append(f"gammaness (mapped to {gammaness_col})")
    
    if missing_essential:
        raise ValueError(f"Missing essential columns in gamma data: {missing_essential}")
    
    # Handle reco_energy consistently - use true_energy as fallback if missing
    if not reco_energy_col or reco_energy_col not in data.dtype.names:
        logger.warning(f"reco_energy column not found in gamma data, using true_energy as fallback")
        reco_energy_col = mc_energy_col
    
    # Calculate theta2 using great-circle distance
    reco_alt_col = column_mapping.get('reco_alt')
    reco_az_col = column_mapping.get('reco_az')
    true_alt_col = column_mapping.get('true_alt')
    true_az_col = column_mapping.get('true_az')
    
    # Validate that all direction columns are available
    missing_direction_cols = []
    if not reco_alt_col or reco_alt_col not in data.dtype.names:
        missing_direction_cols.append(f"reco_alt (mapped to {reco_alt_col})")
    if not reco_az_col or reco_az_col not in data.dtype.names:
        missing_direction_cols.append(f"reco_az (mapped to {reco_az_col})")
    if not true_alt_col or true_alt_col not in data.dtype.names:
        missing_direction_cols.append(f"true_alt (mapped to {true_alt_col})")
    if not true_az_col or true_az_col not in data.dtype.names:
        missing_direction_cols.append(f"true_az (mapped to {true_az_col})")
    
    if missing_direction_cols:
        raise ValueError(f"Cannot calculate theta² without direction columns: {missing_direction_cols}")
    
    # Convert to radians
    reco_alt_rad = np.radians(data[reco_alt_col])
    reco_az_rad = np.radians(data[reco_az_col])
    true_alt_rad = np.radians(data[true_alt_col])
    true_az_rad = np.radians(data[true_az_col])
    
    # Calculate great-circle angular distance
    reco_x = np.cos(reco_alt_rad) * np.cos(reco_az_rad)
    reco_y = np.cos(reco_alt_rad) * np.sin(reco_az_rad)
    reco_z = np.sin(reco_alt_rad)
    
    true_x = np.cos(true_alt_rad) * np.cos(true_az_rad)
    true_y = np.cos(true_alt_rad) * np.sin(true_az_rad)
    true_z = np.sin(true_alt_rad)
    
    dot_product = reco_x * true_x + reco_y * true_y + reco_z * true_z
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta_rad = np.arccos(dot_product)
    theta2_deg = np.degrees(theta_rad)**2
    
    # Apply basic energy cuts (TeV - units should be converted by load_dl2_data)
    energy_min_tev = 0.05  # TeV
    energy_max_tev = 100.0  # TeV
    energy_mask = (data[mc_energy_col] > energy_min_tev) & (data[mc_energy_col] < energy_max_tev)
    if reco_energy_col != mc_energy_col:  # Only if we have actual reco_energy
        energy_mask &= (data[reco_energy_col] > energy_min_tev) & (data[reco_energy_col] < energy_max_tev)
    
    # Get energy data with basic cuts applied
    etrue = data[mc_energy_col][energy_mask]
    ereco = data[reco_energy_col][energy_mask]  # reco_energy_col is now guaranteed to exist (fallback to mc_energy_col)
    gammaness = data[gammaness_col][energy_mask]
    theta2 = theta2_deg[energy_mask]
    
    # Use global energy bins for consistency across all zenith angles
    etrue_bins = get_global_energy_bins()
    
    results = []
    
    for i in range(len(etrue_bins) - 1):
        etrue_min = etrue_bins[i]
        etrue_max = etrue_bins[i + 1]
        
        # Get events in this energy bin
        bin_mask = (etrue >= etrue_min) & (etrue < etrue_max)
        if np.sum(bin_mask) < 10:
            continue
        
        # Apply efficiency-based cuts to this energy bin
        bin_gammaness = gammaness[bin_mask]
        bin_theta2 = theta2[bin_mask]
        
        # Use CLI grids to find optimal cuts for this energy bin that achieve target efficiency
        # For gamma IRF, we optimize for efficiency rather than S/√B since we don't have background
        
        best_efficiency_diff = float('inf')
        best_gammaness = None
        best_theta2_deg2 = None
        best_final_mask = None
        
        # Grid search over CLI-specified ranges
        for gammaness_cut in gamma_grid:
            for theta2_cut_deg2 in theta_grid:
                # Apply cuts to events in this energy bin
                test_mask = (bin_gammaness >= gammaness_cut) & (bin_theta2 <= theta2_cut_deg2)
                efficiency = np.sum(test_mask) / len(bin_gammaness) if len(bin_gammaness) > 0 else 0
                
                # Find cuts that best match target efficiency
                efficiency_diff = abs(efficiency - target_efficiency)
                if efficiency_diff < best_efficiency_diff and np.sum(test_mask) >= 5:
                    best_efficiency_diff = efficiency_diff
                    best_gammaness = gammaness_cut
                    best_theta2_deg2 = theta2_cut_deg2
                    best_final_mask = test_mask
        
        if best_final_mask is not None:
            final_mask = best_final_mask
            logger.debug(f"Energy bin {etrue_min:.3f}-{etrue_max:.3f} TeV: "
                        f"target efficiency={target_efficiency:.2f}, actual={np.sum(final_mask)/len(bin_gammaness):.2f}, "
                        f"gammaness_cut={best_gammaness:.3f}, theta2_cut_deg2={best_theta2_deg2:.3f}")
        else:
            logger.warning(f"No valid cuts found in grid for energy bin {etrue_min:.3f}-{etrue_max:.3f} TeV")
            logger.warning("Falling back to percentile-based cuts")
            
            # Fallback to percentile-based approach
            gammaness_threshold = np.percentile(bin_gammaness, (1 - target_efficiency) * 100)
            
            # Apply gammaness cut and find theta2 threshold
            gammaness_passing = bin_gammaness >= gammaness_threshold
            if np.sum(gammaness_passing) < 5:
            continue
        
            theta2_passing = bin_theta2[gammaness_passing]
            theta2_threshold_deg2 = np.percentile(theta2_passing, (1 - target_efficiency) * 100)
            theta2_threshold_deg2 = min(theta2_threshold_deg2, 0.1)  # Cap at 0.1 deg²
            
            # Count events passing both cuts
            final_mask = (bin_gammaness >= gammaness_threshold) & (bin_theta2 <= theta2_threshold_deg2)
        
        n_events = np.sum(final_mask)
        
        if n_events < 5:
            continue
        
        # Calculate proper effective area with MC normalization
        aeff = calculate_effective_area(
            gamma_events, bin_mask, final_mask, mc_params, etrue_min, etrue_max, column_mapping
        )
        
        # Fit energy migration model
        emig_params, fit_mse = fit_energy_migration(
            gamma_events, bin_mask, final_mask, etrue_min, etrue_max, column_mapping
        )
        
        results.append({
            'ZD_deg': zd_deg,
            'Etrue_min_TeV': etrue_min,
            'Etrue_max_TeV': etrue_max,
            'Aeff_m2': aeff,
            'emig_mu': emig_params['mu'],
            'emig_sigma': emig_params['sigma'],
            'emig_alpha': emig_params['alpha'],
            'emig_model': 'skewnorm',
            'fit_mse': fit_mse,
            'n_events': n_events
        })
    
    return pd.DataFrame(results)


def process_background_irf_with_protons(gamma_data, proton_data, zd_deg, target_efficiency, mc_params, gamma_grid, theta_grid):
    """Process proton data to create proper background rate parameters with cut optimization"""
    
    # Add theta2 calculation to both datasets using great-circle distance
    for data in [gamma_data, proton_data]:
        column_mapping = data.get('_column_mapping', {})
        reco_alt_col = column_mapping.get('reco_alt')
        reco_az_col = column_mapping.get('reco_az')
        true_alt_col = column_mapping.get('true_alt')
        true_az_col = column_mapping.get('true_az')
        
        if all(col in data.dtype.names for col in [reco_alt_col, reco_az_col, true_alt_col, true_az_col] if col):
            # Convert to radians
            reco_alt_rad = np.radians(data[reco_alt_col])
            reco_az_rad = np.radians(data[reco_az_col])
            true_alt_rad = np.radians(data[true_alt_col])
            true_az_rad = np.radians(data[true_az_col])
            
            # Calculate great-circle angular distance using haversine formula
            # Convert alt/az to unit vectors in Cartesian coordinates
            reco_x = np.cos(reco_alt_rad) * np.cos(reco_az_rad)
            reco_y = np.cos(reco_alt_rad) * np.sin(reco_az_rad)
            reco_z = np.sin(reco_alt_rad)
            
            true_x = np.cos(true_alt_rad) * np.cos(true_az_rad)
            true_y = np.cos(true_alt_rad) * np.sin(true_az_rad)
            true_z = np.sin(true_alt_rad)
            
            # Dot product of unit vectors
            dot_product = reco_x * true_x + reco_y * true_y + reco_z * true_z
            
            # Clamp to avoid numerical errors
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            # Angular distance in radians
            theta_rad = np.arccos(dot_product)
            
            # Convert to degrees squared
            theta2_deg = np.degrees(theta_rad)**2  # Note: variable name is theta2_deg but contains θ² values
        else:
            # Use mapped column name for fallback - ensure it exists
            true_energy_col = column_mapping.get('true_energy')
            if not true_energy_col or true_energy_col not in data.dtype.names:
                raise ValueError(f"Cannot calculate theta2: missing direction columns and true_energy column '{true_energy_col}' not found in data")
            theta2_deg = np.zeros(len(data[true_energy_col]))
        
        # Create new structured array with theta2_deg column added
        n_events = len(data)
        new_dtype = list(data.dtype.descr) + [('theta2_deg', 'f8')]
        new_data = np.zeros(n_events, dtype=new_dtype)
        
        # Copy existing data
        for field in data.dtype.names:
            new_data[field] = data[field]
        
        # Add theta2_deg column
        new_data['theta2_deg'] = theta2_deg
        
        # Copy column mapping
        new_data._column_mapping = column_mapping
        
        # Replace data with new structured array
        data = new_data
    
    # Use global energy bins for consistency across all zenith angles
    etrue_bins = get_global_energy_bins()
    
    results = []
    
    for i in range(len(etrue_bins) - 1):
        etrue_min = etrue_bins[i]
        etrue_max = etrue_bins[i + 1]
        
        # Get events in this energy bin
        bin_mask = (data[mc_energy_col] >= etrue_min) & (data[mc_energy_col] < etrue_max)
        
        if np.sum(bin_mask) < 10:
            continue
        
        # Apply efficiency-based cuts to this energy bin
        bin_gammaness = data[gammaness_col][bin_mask]
        bin_theta2 = data['theta2_deg'][bin_mask]
        
        # Find thresholds that give target efficiency
        gammaness_threshold = np.percentile(bin_gammaness, (1 - target_efficiency) * 100)
        
        # Apply gammaness cut and find theta2 threshold
        gammaness_passing = bin_gammaness >= gammaness_threshold
        if np.sum(gammaness_passing) < 5:
            continue
        
        theta2_passing = bin_theta2[gammaness_passing]
        theta2_threshold_deg2 = np.percentile(theta2_passing, (1 - target_efficiency) * 100)
        theta2_threshold_deg2 = min(theta2_threshold_deg2, 0.1)  # Cap at 0.1 deg²
        
        # Count events passing both cuts
        final_mask = (bin_gammaness >= gammaness_threshold) & (bin_theta2 <= theta2_threshold_deg2)
        n_events = np.sum(final_mask)
        
        if n_events < 5:
            continue
        
        # Calculate proper effective area with MC normalization
        aeff = calculate_effective_area(
            data, bin_mask, final_mask, mc_params, etrue_min, etrue_max, column_mapping
        )
        
        # Fit energy migration model
        emig_params, fit_mse = fit_energy_migration(
            data, bin_mask, final_mask, etrue_min, etrue_max, column_mapping
        )
        
        results.append({
            'ZD_deg': zd_deg,
            'Etrue_min_TeV': etrue_min,
            'Etrue_max_TeV': etrue_max,
            'Aeff_m2': aeff,
            'emig_mu': emig_params['mu'],
            'emig_sigma': emig_params['sigma'],
            'emig_alpha': emig_params['alpha'],
            'emig_model': 'skewnorm',
            'fit_mse': fit_mse,
            'n_events': n_events
        })
    
    return pd.DataFrame(results)


def process_background_irf_with_protons(gamma_data, proton_data, zd_deg, target_efficiency, mc_params, gamma_grid, theta_grid):
    """Process proton data to create background IRF parameters with cut optimization"""
    
    # Get column mapping from proton data
    column_mapping = getattr(proton_data, '_column_mapping', {})
    reco_energy_col = column_mapping.get('reco_energy')
    gammaness_col = column_mapping.get('gammaness')
    
    # Validate essential columns exist
    missing_essential = []
    if not gammaness_col or gammaness_col not in proton_data.dtype.names:
        missing_essential.append(f"gammaness (mapped to {gammaness_col})")
    
    if missing_essential:
        raise ValueError(f"Missing essential columns in proton data: {missing_essential}")
    
    # Handle reco_energy consistently - use true_energy as fallback if missing
    if not reco_energy_col or reco_energy_col not in proton_data.dtype.names:
        logger.warning(f"reco_energy column not found in proton data, using true_energy as fallback")
        true_energy_col = column_mapping.get('true_energy')
        if not true_energy_col or true_energy_col not in proton_data.dtype.names:
            raise ValueError(f"Neither reco_energy nor true_energy found in proton data")
        reco_energy_col = true_energy_col
    
    # Add theta2_deg column to proton_data if not present
    if 'theta2_deg' not in proton_data.dtype.names:
        # Calculate theta2 from reconstructed and true directions using great-circle distance
        reco_alt_col = column_mapping.get('reco_alt')
        reco_az_col = column_mapping.get('reco_az')
        true_alt_col = column_mapping.get('true_alt')
        true_az_col = column_mapping.get('true_az')
        
        # Validate that all direction columns are available
        missing_direction_cols = []
        if not reco_alt_col or reco_alt_col not in proton_data.dtype.names:
            missing_direction_cols.append(f"reco_alt (mapped to {reco_alt_col})")
        if not reco_az_col or reco_az_col not in proton_data.dtype.names:
            missing_direction_cols.append(f"reco_az (mapped to {reco_az_col})")
        if not true_alt_col or true_alt_col not in proton_data.dtype.names:
            missing_direction_cols.append(f"true_alt (mapped to {true_alt_col})")
        if not true_az_col or true_az_col not in proton_data.dtype.names:
            missing_direction_cols.append(f"true_az (mapped to {true_az_col})")
        
        if missing_direction_cols:
            raise ValueError(f"Cannot calculate theta² without direction columns: {missing_direction_cols}")
        
        # Convert to radians
        reco_alt_rad = np.radians(proton_data[reco_alt_col])
        reco_az_rad = np.radians(proton_data[reco_az_col])
        true_alt_rad = np.radians(proton_data[true_alt_col])
        true_az_rad = np.radians(proton_data[true_az_col])
        
        # Calculate great-circle angular distance
        reco_x = np.cos(reco_alt_rad) * np.cos(reco_az_rad)
        reco_y = np.cos(reco_alt_rad) * np.sin(reco_az_rad)
        reco_z = np.sin(reco_alt_rad)
        
        true_x = np.cos(true_alt_rad) * np.cos(true_az_rad)
        true_y = np.cos(true_alt_rad) * np.sin(true_az_rad)
        true_z = np.sin(true_alt_rad)
        
        # Dot product of unit vectors
        dot_product = reco_x * true_x + reco_y * true_y + reco_z * true_z
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Angular distance in radians
        theta_rad = np.arccos(dot_product)
        
        # Convert to degrees squared
        theta2_deg = np.degrees(theta_rad)**2  # Note: variable name is theta2_deg but contains θ² values
        
        # Create new structured array with theta2_deg column added
        n_events = len(proton_data)
        new_dtype = list(proton_data.dtype.descr) + [('theta2_deg', 'f8')]
        new_proton_data = np.zeros(n_events, dtype=new_dtype)
        
        # Copy existing data
        for field in proton_data.dtype.names:
            new_proton_data[field] = proton_data[field]
        
        # Add theta2_deg column
        new_proton_data['theta2_deg'] = theta2_deg
        
        # Copy column mapping
        new_proton_data._column_mapping = column_mapping
        
        # Replace proton_data with new structured array
        proton_data = new_proton_data
    
    # Use global energy bins for consistency across all zenith angles
    ereco_bins = get_global_energy_bins()
    
    results = []
    
    for i in range(len(ereco_bins) - 1):
        ereco_min = ereco_bins[i]
        ereco_max = ereco_bins[i + 1]
        
        # Optimize cuts for this energy bin to achieve target efficiency
        # Uses gamma_data (point source preferred) for signal estimation in S/√B calculation
        optimal_gammaness, optimal_theta2_deg2 = optimize_cuts_per_energy_bin(
            gamma_data, proton_data, (ereco_min, ereco_max), 
            reco_energy_col, gammaness_col, target_efficiency, gamma_grid, theta_grid
        )
        
        # Apply optimized cuts to proton events in this energy bin
        energy_mask = (proton_data[reco_energy_col] >= ereco_min) & (proton_data[reco_energy_col] < ereco_max)
        gammaness_mask = proton_data[gammaness_col] > optimal_gammaness
        theta2_mask = proton_data['theta2_deg'] < optimal_theta2_deg2  # theta2_deg is θ² in deg²
        
        # Count background events passing cuts
        background_events = energy_mask & gammaness_mask & theta2_mask
        n_background = np.sum(background_events)
        
        if n_background < 5:
            continue
        
        # Calculate proper background rate with MC normalization
        bkg_rate = calculate_background_rate(
            proton_events, energy_mask, gammaness_mask, theta2_mask, 
            optimal_theta2_deg2, zd_deg, mc_params, column_mapping
        )
        
        results.append({
            'ZD_deg': zd_deg,
            'Ereco_min_TeV': ereco_min,
            'Ereco_max_TeV': ereco_max,
            'BckgRate_per_second': bkg_rate,
            'Theta_cut_deg': np.sqrt(optimal_theta2_deg2),  # Convert θ² (deg²) to θ (deg) for CSV
            'Gammaness_cut': optimal_gammaness,
            'n_events': n_background
        })
    
    return pd.DataFrame(results)


def save_irf_csv(df, output_path, irf_type, schema_version=CSV_SCHEMA_VERSION, include_comments=True, strict_validation=True):
    """
    Save IRF DataFrame to CSV with proper schema versioning and validation.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the CSV file
        irf_type: Type of IRF ('gamma' or 'background')
        schema_version: Schema version string
        include_comments: If True, add comment header lines. If False, save clean CSV.
        strict_validation: If True, fail on missing critical columns. If False, only warn.
    """
    if df.empty:
        logger.warning(f"No data to save for {irf_type} IRF")
        return
    
    # Validate schema with fail-fast for critical columns
    expected_columns = LST_IRF_SCHEMA[irf_type]
    missing_columns = set(expected_columns) - set(df.columns)
    extra_columns = set(df.columns) - set(expected_columns)
    
    # Define critical columns that must be present
    if irf_type == 'gamma':
        critical_columns = {'ZD_deg', 'Etrue_min_TeV', 'Etrue_max_TeV', 'Aeff_m2', 'n_events'}
        optional_columns = {'emig_mu', 'emig_sigma', 'emig_alpha', 'emig_model', 'fit_mse'}
    elif irf_type == 'background':
        critical_columns = {'ZD_deg', 'Ereco_min_TeV', 'Ereco_max_TeV', 'BckgRate_per_second', 'n_events'}
        optional_columns = {'Theta_cut_deg', 'Gammaness_cut'}
    else:
        raise ValueError(f"Unknown IRF type: {irf_type}")
    
    # Check for missing critical columns
    missing_critical = missing_columns & critical_columns
    missing_optional = missing_columns & optional_columns
    
    if missing_critical:
        error_msg = (f"CRITICAL: Missing required columns in {irf_type} IRF: {missing_critical}. "
                    f"Cannot generate valid CSV without these columns.")
        if strict_validation:
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.warning(f"VALIDATION DISABLED: {error_msg}")
            logger.warning("Proceeding with incomplete data due to --skip-schema-validation")
    
    if missing_optional:
        logger.warning(f"Missing optional columns in {irf_type} IRF: {missing_optional}")
    
    if extra_columns:
        logger.warning(f"Extra columns in {irf_type} IRF (will be ignored): {extra_columns}")
    
    # Validate data quality for critical columns
    for col in critical_columns:
        if col in df.columns:
            if df[col].isnull().any():
                error_msg = f"CRITICAL: Column '{col}' contains null/NaN values in {irf_type} IRF"
                if strict_validation:
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.warning(f"VALIDATION DISABLED: {error_msg}")
            
            # Additional validation for specific columns
            if col in ['Aeff_m2', 'BckgRate_per_second'] and (df[col] < 0).any():
                error_msg = f"CRITICAL: Column '{col}' contains negative values in {irf_type} IRF"
                if strict_validation:
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.warning(f"VALIDATION DISABLED: {error_msg}")
            
            if col == 'n_events' and (df[col] <= 0).any():
                error_msg = f"CRITICAL: Column '{col}' contains zero or negative event counts in {irf_type} IRF"
                if strict_validation:
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.warning(f"VALIDATION DISABLED: {error_msg}")
    
    # Reorder columns to match schema (only include existing columns)
    ordered_columns = [col for col in expected_columns if col in df.columns]
    df_ordered = df[ordered_columns]
    
    logger.info(f"Schema validation passed for {irf_type} IRF: {len(ordered_columns)}/{len(expected_columns)} columns present")
    
    # Write CSV file
    if include_comments:
        # Create header with schema information
        header_lines = [
            f"# SST-1M IRF CSV Schema Version: {schema_version}",
            f"# IRF Type: {irf_type}",
            f"# Generated: {pd.Timestamp.now().isoformat()}",
            f"# Columns: {', '.join(ordered_columns)}",
            ""
        ]
        
        # Write header and data
        with open(output_path, 'w') as f:
            for line in header_lines:
                f.write(line + '\n')
            df_ordered.to_csv(f, index=False)
    else:
        # Write clean CSV without comments for downstream compatibility
        df_ordered.to_csv(output_path, index=False)
    
    csv_format = "with comments" if include_comments else "clean (LST-compatible)"
    logger.info(f"Saved {irf_type} IRF to {output_path} with schema version {schema_version} ({csv_format})")
    logger.info(f"Shape: {df_ordered.shape}, Columns: {list(df_ordered.columns)}")


def create_run_metadata(args, processed_files, energy_bins, output_dir):
    """
    Create comprehensive run metadata for reproducibility.
    
    Args:
        args: Parsed command line arguments
        processed_files: Dict with file information and event counts
        energy_bins: Energy bin edges used
        output_dir: Output directory path
        
    Returns:
        dict: Complete run metadata
    """
    metadata = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "script_version": "1.0",
            "csv_schema_version": CSV_SCHEMA_VERSION,
            "command_line": " ".join(sys.argv),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "parameters": {
            "cut_efficiency": args.cut_efficiency,
            "energy_bins": args.bins,
            "energy_min_tev": args.emin,
            "energy_max_tev": args.emax,
            "theta_grid": args.theta_grid,
            "gamma_grid": args.gamma_grid,
                         "use_point_source": args.use_point,
             "use_diffuse_source": args.use_diffuse,
             "custom_weights_column": args.weights_col,
             "forced_energy_unit": args.energy_unit,
             "include_csv_comments": args.include_csv_comments,
             "skip_schema_validation": args.skip_schema_validation,
             "mc_livetime": args.mc_livetime,
             "mc_thrown_area": args.mc_thrown_area,
             "mc_solid_angle": args.mc_solid_angle,
             "mc_spectral_index": args.mc_spectral_index,
             "mc_energy_min": args.mc_energy_min,
             "mc_energy_max": args.mc_energy_max,
             "dl2_directory": str(args.dl2_dir),
             "output_directory": str(args.output_dir),
        },
        "energy_binning": {
            "n_bins": len(energy_bins) - 1,
            "bin_edges_tev": energy_bins.tolist(),
            "bin_centers_tev": ((energy_bins[:-1] + energy_bins[1:]) / 2).tolist(),
            "bin_widths_tev": (energy_bins[1:] - energy_bins[:-1]).tolist(),
        },
        "input_files": processed_files,
        "output_files": {
            "gamma_irf": None,  # Will be filled after CSV creation
            "background_irf": None,  # Will be filled after CSV creation
            "energy_bin_definition": None,  # Will be filled after creation
        }
    }
    
    return metadata


def save_run_metadata(metadata, output_dir):
    """Save run metadata to JSON sidecar file"""
    metadata_file = output_dir / "run_metadata.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return obj
    
    # Recursively convert numpy types
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_numpy(obj)
    
    metadata_serializable = recursive_convert(metadata)
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_serializable, f, indent=2, sort_keys=True)
    
    logger.info(f"Saved run metadata to {metadata_file}")
    return metadata_file


def main():
    parser = argparse.ArgumentParser(description='Convert SST-1M DL2 files to CSV IRF format with proper background')
    parser.add_argument('--cut-efficiency', type=float, default=0.7,
                       help='Cut efficiency (0.4, 0.7, or 0.9)')
    parser.add_argument('--output-dir', type=str, default='../',
                       help='Output directory for CSV files')
    parser.add_argument('--dl2-dir', type=str, default='../dl2_gamma',
                       help='Directory containing DL2 files')
    
    # Energy binning parameters
    parser.add_argument('--bins', type=int, default=20,
                       help='Number of energy bins (default: 20)')
    parser.add_argument('--emin', type=float, default=0.05,
                       help='Minimum energy in TeV (default: 0.05)')
    parser.add_argument('--emax', type=float, default=100.0,
                       help='Maximum energy in TeV (default: 100.0)')
    
    # Cut optimization parameters
    parser.add_argument('--theta-grid', type=str, default="0.01,0.02,0.05,0.1,0.2,0.3",
                       help='Comma-separated theta² cut grid in deg² (default: "0.01,0.02,0.05,0.1,0.2,0.3")')
    parser.add_argument('--gamma-grid', type=str, default="0.1:0.9:0.1",
                       help='Gammaness cut grid as start:end:step (default: "0.1:0.9:0.1")')
    
    # Source selection for cut tuning
    parser.add_argument('--use-point', action='store_true',
                       help='Force use of point source gamma data for cut optimization (preferred)')
    parser.add_argument('--use-diffuse', action='store_true',
                       help='Force use of diffuse source gamma data for cut optimization (suboptimal)')
    
        # Data handling parameters
    parser.add_argument('--weights-col', type=str, default=None,
                        help='Specify generator weight column name (auto-detected if not provided)')
    parser.add_argument('--energy-unit', type=str, choices=['TeV', 'GeV'], default=None,
                        help='Force energy unit (bypasses auto-detection). Use if detection fails or is incorrect.')
    
    # CSV output format
    parser.add_argument('--include-csv-comments', action='store_true', default=False,
                        help='Include comment header lines in CSV files (default: clean CSV for LST compatibility)')
    
    # Schema validation control
    parser.add_argument('--skip-schema-validation', action='store_true', default=False,
                        help='Skip strict schema validation (allows missing critical columns, for debugging only)')
    
    # MC parameter overrides (to avoid dangerous defaults)
    parser.add_argument('--mc-livetime', type=float, default=None,
                        help='MC livetime in seconds (overrides file extraction, prevents dangerous defaults)')
    parser.add_argument('--mc-thrown-area', type=float, default=None,
                        help='MC thrown area in m² (overrides file extraction, prevents dangerous defaults)')
    parser.add_argument('--mc-solid-angle', type=float, default=None,
                        help='MC solid angle in steradians (overrides file extraction, prevents dangerous defaults)')
    parser.add_argument('--mc-spectral-index', type=float, default=None,
                        help='MC spectral index (overrides file extraction, prevents dangerous defaults)')
    parser.add_argument('--mc-energy-min', type=float, default=None,
                        help='MC minimum energy in TeV (overrides file extraction, prevents dangerous defaults)')
    parser.add_argument('--mc-energy-max', type=float, default=None,
                        help='MC maximum energy in TeV (overrides file extraction, prevents dangerous defaults)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_point and args.use_diffuse:
        parser.error("Cannot specify both --use-point and --use-diffuse")
    
    # Parse grid parameters
    try:
        theta_grid = [float(x.strip()) for x in args.theta_grid.split(',')]
        if not all(x > 0 for x in theta_grid):
            parser.error("All theta² grid values must be positive")
    except ValueError:
        parser.error("Invalid theta-grid format. Use comma-separated values like '0.01,0.02,0.05'")
    
    try:
        gamma_parts = args.gamma_grid.split(':')
        if len(gamma_parts) != 3:
            parser.error("Invalid gamma-grid format. Use start:end:step like '0.1:0.9:0.1'")
        gamma_start, gamma_end, gamma_step = map(float, gamma_parts)
        if not (0 <= gamma_start < gamma_end <= 1 and gamma_step > 0):
            parser.error("Invalid gamma-grid values. Must be 0 <= start < end <= 1 and step > 0")
    except ValueError:
        parser.error("Invalid gamma-grid format. Use start:end:step like '0.1:0.9:0.1'")
    
    # Validate energy parameters
    if not (0 < args.emin < args.emax):
        parser.error("Energy parameters must satisfy 0 < emin < emax")
    
    if args.bins < 1:
        parser.error("Number of bins must be at least 1")
    
    # Validate energy unit parameter
    if args.energy_unit and args.energy_unit.upper() not in ['TEV', 'GEV']:
        parser.error(f"Invalid energy unit '{args.energy_unit}'. Must be 'TeV' or 'GeV'")
    
    # Set up paths
    base_path = Path(args.dl2_dir)
    output_path = Path(args.output_dir)
    
    # Create properly named output directory for IRF tables with schema versioning
    irf_output_dir = output_path / f"sst1m_irf_tables_efficiency_{args.cut_efficiency:.1f}_schema_{CSV_SCHEMA_VERSION.replace('.', '_')}"
    irf_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {irf_output_dir.absolute()}")
    logger.info(f"CSV Schema Version: {CSV_SCHEMA_VERSION}")
    
    # Log CLI parameters
    logger.info(f"\nCLI Parameters:")
    logger.info(f"  Cut efficiency: {args.cut_efficiency}")
    logger.info(f"  Energy bins: {args.bins}")
    logger.info(f"  Energy range: {args.emin} - {args.emax} TeV")
    logger.info(f"  Theta² grid: {theta_grid} deg²")
    logger.info(f"  Gammaness grid: {gamma_start}:{gamma_end}:{gamma_step}")
    if args.use_point:
        logger.info(f"  Source selection: Point source only (--use-point)")
    elif args.use_diffuse:
        logger.info(f"  Source selection: Diffuse source only (--use-diffuse)")
    else:
        logger.info(f"  Source selection: Auto (prefer point source)")
    if args.weights_col:
        logger.info(f"  Custom weights column: {args.weights_col}")
    else:
        logger.info(f"  Weights column: Auto-detect")
    if args.energy_unit:
        logger.info(f"  Energy unit: Forced to {args.energy_unit}")
    else:
        logger.info(f"  Energy unit: Auto-detect")
    
    # Define global energy bins and log them
    global_energy_bins = get_global_energy_bins(n_bins=args.bins, min_energy=args.emin, max_energy=args.emax)
    logger.info(f"Global energy bin definition:")
    logger.info(f"  Number of bins: {len(global_energy_bins) - 1} (requested: {args.bins})")
    logger.info(f"  Energy range: {global_energy_bins[0]:.3f} - {global_energy_bins[-1]:.1f} TeV (requested: {args.emin} - {args.emax})")
    logger.info(f"  Bin edges: {global_energy_bins}")
    logger.info(f"  This ensures consistent binning across all zenith angles")
    
    # Save bin definition to file for reference
    bin_info_file = irf_output_dir / "energy_bin_definition.txt"
    with open(bin_info_file, 'w') as f:
        f.write("Global Energy Bin Definition\n")
        f.write("============================\n\n")
        f.write(f"Number of bins: {len(global_energy_bins) - 1}\n")
        f.write(f"Energy range: {global_energy_bins[0]:.3f} - {global_energy_bins[-1]:.1f} TeV\n")
        f.write(f"Bin type: Logarithmic\n\n")
        f.write("Bin edges (TeV):\n")
        for i, edge in enumerate(global_energy_bins):
            f.write(f"  Bin {i}: {edge:.6f}\n")
        f.write("\nBin centers (TeV):\n")
        for i in range(len(global_energy_bins) - 1):
            center = (global_energy_bins[i] + global_energy_bins[i + 1]) / 2
            f.write(f"  Bin {i}: {center:.6f}\n")
    logger.info(f"Saved bin definition to {bin_info_file}")
    
    # Define zenith angles and corresponding files
    zenith_angles = [20, 30, 40, 60]  
    
    # Gamma file patterns - prefer point source for cut optimization
    gamma_file_patterns = {
        "point": "gamma_point_50_300E3GeV_{zd}_{zd}deg_testing_dl1_dl2.h5",
        "diffuse": "gamma_200_300E3GeV_{zd}_{zd}deg_testing_dl1_dl2.h5"
    }
    
    # Proton file pattern for background estimation
    proton_file_pattern = "proton_400_500E3GeV_{zd}_{zd}deg_testing_dl1_dl2.h5"
    
    logger.info("Gamma file priority for cut optimization:")
    logger.info("  1. Point source (preferred for S/√B calculation)")
    logger.info("  2. Diffuse source (fallback only)")
    logger.info("  Note: Cut optimization requires on-axis point source for proper S/√B")
    
    logger.info("Available files:")
    for zd in zenith_angles:
        # Check gamma files
        for sim_type, pattern in gamma_file_patterns.items():
            filename = pattern.format(zd=zd)
            filepath = base_path / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024*1024)
                logger.info(f"  ✓ {filename} ({size_mb:.1f} MB)")
            else:
                logger.warning(f"  ✗ {filename} (NOT FOUND)")
        
        # Check proton files
        proton_filename = proton_file_pattern.format(zd=zd)
        proton_filepath = base_path / proton_filename
        if proton_filepath.exists():
            size_mb = proton_filepath.stat().st_size / (1024*1024)
            logger.info(f"  ✓ {proton_filename} ({size_mb:.1f} MB)")
        else:
            logger.warning(f"  ✗ {proton_filename} (NOT FOUND)")
    
    # Process all files and create CSV outputs
    all_gamma_results = []
    all_background_results = []
    processed_zeniths = {}  # Track which source types were used
    
    # Track processed files for reproducibility
    processed_files = {
        "gamma_files": {},
        "proton_files": {},
        "event_counts": {},
        "file_checksums": {}
    }
    
    for zd in zenith_angles:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing zenith angle {zd}°")
        logger.info(f"{'='*50}")
        
        # Load gamma data - respect CLI flags for source selection
        gamma_data = None
        gamma_source_type = None
        
        # Determine source type priority based on CLI flags
        if args.use_point:
            source_priority = ["point"]
            logger.info("  Using point source only (--use-point specified)")
        elif args.use_diffuse:
            source_priority = ["diffuse"]
            logger.info("  Using diffuse source only (--use-diffuse specified)")
        else:
            # Default: prefer point source for cut optimization
            source_priority = ["point", "diffuse"]
            logger.info("  Preferring point source for cut optimization (optimal for S/√B)")
        
        # Try sources in priority order
        for sim_type in source_priority:
            if sim_type not in gamma_file_patterns:
                continue
                
            filename = gamma_file_patterns[sim_type].format(zd=zd)
            filepath = base_path / filename
            
            if filepath.exists():
                try:
                    # Calculate file checksum for reproducibility
                    file_checksum = calculate_file_checksum(filepath)
                    file_size = filepath.stat().st_size
                    
                                         gamma_data = load_dl2_data(filepath, args.weights_col, args.energy_unit)
                    gamma_source_type = sim_type
                    
                    # Get initial event count
                    column_mapping = getattr(gamma_data, '_column_mapping', {})
                    true_energy_col = column_mapping.get('true_energy')
                    if true_energy_col and true_energy_col in gamma_data.dtype.names:
                        initial_count = len(gamma_data[true_energy_col])
                    else:
                        # Try alternative energy column names
                        for col in ['mc_energy', 'energy_true', 'E_true']:
                            if col in gamma_data.dtype.names:
                                initial_count = len(gamma_data[col])
                                break
                        else:
                            initial_count = len(gamma_data)
                    
                    logger.info(f"Loaded {sim_type} gamma simulation data for zenith {zd}°")
                    logger.info(f"  File: {filename}")
                    logger.info(f"  Size: {file_size / (1024*1024):.1f} MB")
                    logger.info(f"  Checksum: {file_checksum[:16]}..." if file_checksum else "  Checksum: Failed")
                    logger.info(f"  Initial events: {initial_count:,}")
                    
                    # Store file information for metadata
                    processed_files["gamma_files"][f"zd_{zd}"] = {
                        "filename": filename,
                        "filepath": str(filepath),
                        "source_type": sim_type,
                        "file_size_bytes": file_size,
                        "checksum_sha256": file_checksum,
                        "initial_event_count": initial_count,
                        "column_mapping": gamma_data.get('_column_mapping', {})
                    }
                    
                    if sim_type == "point":
                        logger.info("  ✓ Using point source for cut optimization (optimal for S/√B)")
                    else:
                        logger.warning("  ⚠ Using diffuse source for cut optimization (suboptimal for S/√B)")
                        logger.warning("    Point source is preferred for proper signal estimation")
                    
                    break
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    continue
        
        if gamma_data is None:
            logger.warning(f"No gamma data found for zenith angle {zd}°")
            continue
        
        # Add theta2_deg column to gamma_data if not present
        if 'theta2_deg' not in gamma_data.dtype.names:
            # Get column mapping from gamma data
            gamma_column_mapping = getattr(gamma_data, '_column_mapping', {})
            
            # Calculate theta2 from reconstructed and true directions using great-circle distance
            reco_alt_col = gamma_column_mapping.get('reco_alt')
            reco_az_col = gamma_column_mapping.get('reco_az')
            true_alt_col = gamma_column_mapping.get('true_alt')
            true_az_col = gamma_column_mapping.get('true_az')
            
            # Validate that all direction columns are available
            missing_direction_cols = []
            if not reco_alt_col or reco_alt_col not in gamma_data.dtype.names:
                missing_direction_cols.append(f"reco_alt (mapped to {reco_alt_col})")
            if not reco_az_col or reco_az_col not in gamma_data.dtype.names:
                missing_direction_cols.append(f"reco_az (mapped to {reco_az_col})")
            if not true_alt_col or true_alt_col not in gamma_data.dtype.names:
                missing_direction_cols.append(f"true_alt (mapped to {true_alt_col})")
            if not true_az_col or true_az_col not in gamma_data.dtype.names:
                missing_direction_cols.append(f"true_az (mapped to {true_az_col})")
            
            if missing_direction_cols:
                raise ValueError(f"Cannot calculate theta² without direction columns: {missing_direction_cols}")
            
            # Convert to radians
            reco_alt_rad = np.radians(gamma_data[reco_alt_col])
            reco_az_rad = np.radians(gamma_data[reco_az_col])
            true_alt_rad = np.radians(gamma_data[true_alt_col])
            true_az_rad = np.radians(gamma_data[true_az_col])
            
            # Calculate great-circle angular distance
            reco_x = np.cos(reco_alt_rad) * np.cos(reco_az_rad)
            reco_y = np.cos(reco_alt_rad) * np.sin(reco_az_rad)
            reco_z = np.sin(reco_alt_rad)
            
            true_x = np.cos(true_alt_rad) * np.cos(true_az_rad)
            true_y = np.cos(true_alt_rad) * np.sin(true_az_rad)
            true_z = np.sin(true_alt_rad)
            
            # Dot product of unit vectors
            dot_product = reco_x * true_x + reco_y * true_y + reco_z * true_z
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            # Angular distance in radians
            theta_rad = np.arccos(dot_product)
            
            # Convert to degrees squared
            theta2_deg = np.degrees(theta_rad)**2  # Note: variable name is theta2_deg but contains θ² values
            
            # Create new structured array with theta2_deg column added
            n_events = len(gamma_data)
            new_dtype = list(gamma_data.dtype.descr) + [('theta2_deg', 'f8')]
            new_gamma_data = np.zeros(n_events, dtype=new_dtype)
            
            # Copy existing data
            for field in gamma_data.dtype.names:
                new_gamma_data[field] = gamma_data[field]
            
            # Add theta2_deg column
            new_gamma_data['theta2_deg'] = theta2_deg
            
            # Copy column mapping
            new_gamma_data._column_mapping = gamma_column_mapping
            
            # Replace gamma_data with new structured array
            gamma_data = new_gamma_data
        
        # Track which source type was used
        processed_zeniths[zd] = gamma_source_type
        
        # Load proton data
        proton_filename = proton_file_pattern.format(zd=zd)
        proton_filepath = base_path / proton_filename
        
        if not proton_filepath.exists():
            logger.warning(f"No proton data found for zenith angle {zd}°")
            continue
        
        try:
            # Calculate file checksum for reproducibility
            proton_checksum = calculate_file_checksum(proton_filepath)
            proton_size = proton_filepath.stat().st_size
            
                         proton_data = load_dl2_data(proton_filepath, args.weights_col, args.energy_unit)
            
            # Get initial event count
            column_mapping = getattr(proton_data, '_column_mapping', {})
            true_energy_col = column_mapping.get('true_energy')
            if true_energy_col and true_energy_col in proton_data.dtype.names:
                proton_initial_count = len(proton_data[true_energy_col])
            else:
                # Try alternative energy column names
                for col in ['mc_energy', 'energy_true', 'E_true']:
                    if col in proton_data.dtype.names:
                        proton_initial_count = len(proton_data[col])
                        break
                else:
                    proton_initial_count = len(proton_data)
            
            logger.info(f"Loaded proton simulation data")
            logger.info(f"  File: {proton_filename}")
            logger.info(f"  Size: {proton_size / (1024*1024):.1f} MB")
            logger.info(f"  Checksum: {proton_checksum[:16]}..." if proton_checksum else "  Checksum: Failed")
            logger.info(f"  Initial events: {proton_initial_count:,}")
            
            # Store proton file information for metadata
            processed_files["proton_files"][f"zd_{zd}"] = {
                "filename": proton_filename,
                "filepath": str(proton_filepath),
                "file_size_bytes": proton_size,
                "checksum_sha256": proton_checksum,
                "initial_event_count": proton_initial_count,
                "column_mapping": proton_data.get('_column_mapping', {})
            }
            
            # Extract MC parameters from proton file with CLI overrides
            cli_mc_overrides = {
                'livetime': args.mc_livetime,
                'thrown_area': args.mc_thrown_area,
                'solid_angle': args.mc_solid_angle,
                'spectral_index': args.mc_spectral_index,
                'energy_min': args.mc_energy_min,
                'energy_max': args.mc_energy_max,
            }
            mc_params = extract_mc_parameters(proton_filepath, cli_mc_overrides)
            
        except Exception as e:
            logger.error(f"Error loading {proton_filename}: {e}")
            continue
        
        # Process gamma IRF
        gamma_df = process_gamma_irf(gamma_data, zd, args.cut_efficiency, mc_params, gamma_grid, theta_grid)
        if not gamma_df.empty:
            all_gamma_results.append(gamma_df)
            total_gamma_events = gamma_df['n_events'].sum()
            logger.info(f"Generated {len(gamma_df)} gamma IRF entries")
            logger.info(f"  Total gamma events processed: {total_gamma_events:,}")
            
            # Store event count information
            processed_files["event_counts"][f"gamma_zd_{zd}"] = {
                "irf_entries": len(gamma_df),
                "total_events": int(total_gamma_events),
                "energy_bins_used": len(gamma_df)
            }
        
        # Process background IRF with protons
        background_df = process_background_irf_with_protons(gamma_data, proton_data, zd, args.cut_efficiency, mc_params, gamma_grid, theta_grid)
        if not background_df.empty:
            all_background_results.append(background_df)
            total_background_events = background_df['n_events'].sum()
            logger.info(f"Generated {len(background_df)} background IRF entries")
            logger.info(f"  Total background events processed: {total_background_events:,}")
            
            # Store event count information
            processed_files["event_counts"][f"background_zd_{zd}"] = {
                "irf_entries": len(background_df),
                "total_events": int(total_background_events),
                "energy_bins_used": len(background_df)
            }
    
    # Combine all results and save with schema validation
    if all_gamma_results:
        gamma_final = pd.concat(all_gamma_results, ignore_index=True)
        gamma_final = gamma_final.sort_values(['ZD_deg', 'Etrue_min_TeV'])
        
        # Save gamma IRF CSV with schema validation
        gamma_output = irf_output_dir / f"SST1M_gamma_irf_gheffi_{args.cut_efficiency:.2f}_theffi_{args.cut_efficiency:.2f}_schema_{CSV_SCHEMA_VERSION.replace('.', '_')}.csv"
        save_irf_csv(gamma_final, gamma_output, 'gamma', CSV_SCHEMA_VERSION, args.include_csv_comments, not args.skip_schema_validation)
    
    if all_background_results:
        background_final = pd.concat(all_background_results, ignore_index=True)
        background_final = background_final.sort_values(['ZD_deg', 'Ereco_min_TeV'])
        
        # Save background IRF CSV with schema validation
        background_output = irf_output_dir / f"SST1M_backg_irf_gheffi_{args.cut_efficiency:.2f}_theffi_{args.cut_efficiency:.2f}_schema_{CSV_SCHEMA_VERSION.replace('.', '_')}.csv"
        save_irf_csv(background_final, background_output, 'background', CSV_SCHEMA_VERSION, args.include_csv_comments, not args.skip_schema_validation)
        
        # Display first few rows
        logger.info("\nFirst few background IRF entries:")
        logger.info(background_final.head())
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SST-1M Improved DL2 to CSV Conversion Summary")
    logger.info("="*50)
    
    # Report gamma source types used
    logger.info(f"\nGamma source types used for cut optimization:")
    for zd in zenith_angles:
        if zd in processed_zeniths:
            source_type = processed_zeniths[zd]
            if source_type == "point":
                logger.info(f"  Zenith {zd}°: ✓ Point source (optimal for S/√B)")
            else:
                logger.info(f"  Zenith {zd}°: ⚠ {source_type.title()} source (suboptimal for S/√B)")
        else:
            logger.info(f"  Zenith {zd}°: ✗ No data processed")
    
    if all_background_results:
        logger.info(f"\nBackground IRF Data (using proton data):")
        logger.info(f"  Total entries: {len(background_final)}")
        logger.info(f"  Zenith angles: {sorted(background_final['ZD_deg'].unique())}")
        logger.info(f"  Energy range: {background_final['Ereco_min_TeV'].min():.3f} - {background_final['Ereco_max_TeV'].max():.1f} TeV")
        logger.info(f"  Theta cuts range: {background_final['Theta_cut_deg'].min():.3f} - {background_final['Theta_cut_deg'].max():.3f} deg")
        logger.info(f"  Gammaness cuts range: {background_final['Gammaness_cut'].min():.2f} - {background_final['Gammaness_cut'].max():.2f}")
        logger.info(f"  Background rate range: {background_final['BckgRate_per_second'].min():.6f} - {background_final['BckgRate_per_second'].max():.3f} events/s")
    
    # Create and save run metadata for reproducibility
    logger.info(f"\n{'='*50}")
    logger.info("Creating run metadata for reproducibility")
    logger.info(f"{'='*50}")
    
    # Update output file information in metadata
    if all_gamma_results:
        gamma_output = irf_output_dir / f"SST1M_gamma_irf_gheffi_{args.cut_efficiency:.2f}_theffi_{args.cut_efficiency:.2f}_schema_{CSV_SCHEMA_VERSION.replace('.', '_')}.csv"
        processed_files["output_files"]["gamma_irf"] = {
            "filename": gamma_output.name,
            "filepath": str(gamma_output),
            "entries": len(gamma_final),
            "total_events": int(gamma_final['n_events'].sum())
        }
    
    if all_background_results:
        background_output = irf_output_dir / f"SST1M_backg_irf_gheffi_{args.cut_efficiency:.2f}_theffi_{args.cut_efficiency:.2f}_schema_{CSV_SCHEMA_VERSION.replace('.', '_')}.csv"
        processed_files["output_files"]["background_irf"] = {
            "filename": background_output.name,
            "filepath": str(background_output),
            "entries": len(background_final),
            "total_events": int(background_final['n_events'].sum())
        }
    
    # Add energy bin definition file
    bin_info_file = irf_output_dir / "energy_bin_definition.txt"
    processed_files["output_files"]["energy_bin_definition"] = {
        "filename": bin_info_file.name,
        "filepath": str(bin_info_file),
        "n_bins": len(global_energy_bins) - 1
    }
    
    # Create and save metadata
    run_metadata = create_run_metadata(args, processed_files, global_energy_bins, irf_output_dir)
    metadata_file = save_run_metadata(run_metadata, irf_output_dir)
    
    logger.info(f"\nOutput files created in: {irf_output_dir.absolute()}")
    logger.info(f"CSV Schema Version: {CSV_SCHEMA_VERSION}")
    csv_format = "with comment headers" if args.include_csv_comments else "clean format (LST-compatible)"
    logger.info(f"CSV Format: {csv_format}")
    validation_mode = "disabled (debugging mode)" if args.skip_schema_validation else "strict (production mode)"
    logger.info(f"Schema Validation: {validation_mode}")
    if args.skip_schema_validation:
        logger.warning("⚠️  WARNING: Schema validation was disabled! Generated CSV files may be incomplete or invalid.")
        logger.warning("⚠️  Only use --skip-schema-validation for debugging purposes.")
    
    # Show MC parameter status
    mc_overrides = [
        ('livetime', args.mc_livetime),
        ('thrown_area', args.mc_thrown_area),
        ('solid_angle', args.mc_solid_angle),
        ('spectral_index', args.mc_spectral_index),
        ('energy_min', args.mc_energy_min),
        ('energy_max', args.mc_energy_max)
    ]
    mc_overrides_used = [name for name, value in mc_overrides if value is not None]
    
    if mc_overrides_used:
        logger.info(f"MC Parameter Overrides: {', '.join(mc_overrides_used)}")
    else:
        logger.info("MC Parameter Overrides: none (extracted from files)")
        logger.info("  Use --mc-* arguments if extraction fails or values are incorrect")
    
    logger.info(f"Run metadata saved to: {metadata_file}")
    logger.info("\nNext steps:")
    logger.info("1. Verify the optimized cuts make sense")
    logger.info("2. Compare background rates with expected values")
    logger.info("3. Test the source simulation with these improved IRFs")
    logger.info("4. Use run_metadata.json to reproduce this analysis")
    logger.info("\nEnergy unit detection:")
    if args.energy_unit:
        logger.info(f"  ✓ Energy unit was forced to {args.energy_unit}")
    else:
        logger.info("  ⚠ Energy unit was auto-detected")
        logger.info("  If cuts/IRFs seem wrong, try --energy-unit TeV or --energy-unit GeV")
        logger.info("  Check the energy statistics logged above for validation")


if __name__ == "__main__":
    main()
