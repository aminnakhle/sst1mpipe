This is a source simulator for the sst1mpipe, the pipeline itself has a readme. So please read that before going further in this.

In the folder notebooks are the following python notebooks:
 - LST1_observation_simulator.ipynb : this acts as a basic guidline to what our source simulator should look like.
    In the case of LST observation simulator they start with loading aproximate image response functions, these need to be created in our case for our telescopes also
 - from_dl2_to_csv.ipynb : this notebook takes our testing dl2 files from dl2_gamma and creates the approximate instrument response functions csv. 
    The format of the csv is based on the LST format, available in lst_data. The cuts needs to be the same for both background and gammas.

## DL2 to CSV Conversion

The DL2 to CSV conversion has been implemented and tested successfully. The conversion process:

1. **Input**: SST-1M DL2 HDF5 files from `dl2_gamma/` directory
2. **Output**: CSV files in LST format for use with the source simulator
3. **Script**: `scripts/create_irf_csv.py`

### Generated Files

- `SST1M_gamma_irf_gheffi_0.70_theffi_0.70.csv` - Gamma IRF data with effective area and energy migration parameters
- `SST1M_backg_irf_gheffi_0.70_theffi_0.70.csv` - Background IRF data with rates and cuts

### Data Coverage

- **Zenith angles**: 20°, 30°, 40°, 60° (50° files not available)
- **Energy range**: ~0.5 - 100 TeV
- **Cut efficiency**: 0.7 (configurable)
- **Models**: Skewnorm and Moyal distributions for energy migration

### Usage

```bash
cd scripts
python create_irf_csv.py --cut-efficiency 0.7 --dl2-dir ../dl2_gamma --output-dir ../
```

### Next Steps

1. Create additional cut efficiency versions (0.4, 0.9)
2. Build the SST-1M observation simulator using these CSV files
3. Test with known sources (e.g., Crab Nebula)