.. _sst1m_analysis_workflow:

SST-1M analysis workflow
========================

.. _automatic_processing:

Automatic data processing
-------------------------

Raw data from both telescopes are processed authomaticaly up to DL3 level 
every morning after an observing night on **Calculus** with ``Daily analysis``
pipeline:

``/mnt/nfs22_auger/jurysek/sst1mpipe/sst1mpipe/scripts/daily_analysis.py``

Its products (all data levels up to DL3) can be found in the following directory:

``/data/work/analysis/Daily_analysis/daily_data/``

Those are sorted by date, source observed, and the telescope used (cs1, cs2 and stereo for coincident events).
All data level directories also contain configuration file and log file for each processed run, so 
user can easily check if everything gone OK. Random Forest models used in in authomatic analysis 
can be found in

``/data/work/analysis/MC/prod_january_2023/``

for both mono and stereo regime of observation, together with expected performances. IRFs for DL3
production can be found in

``/data/work/analysis/IRFs/``

DL3 products of the authomatic analysis can be further analyzed using standard tools 
such as `gammapy <https://github.com/gammapy>`_. 

.. note::

    As the software development is progressing very fast, this processing is usually 
    done with slightly obsolete version of sst1mpipe and with configuration not fully 
    tunned for atmospheric conditions during particular nights. To obtain *paper level* 
    analysis, one should proceed with the steps described in :ref:`detailed_processing`.


.. _detailed_processing:

Detailed data processing
------------------------

All data files are stored on telescope servers cs1/cs2. They are also mirrored on Calculus
(``/net/cs1/data/raw/``, ``/net/cs2/data/raw/``) and FZU farm (``/mnt/sst1m/sst1m_backups/``), 
where can be processed and analyzed. .

To get some feeling how the event reconstruction and data analysis works, one can take 
a look at materials from the last hands-on analysis session in Prague, December 2023:

* `Introductory talk <https://indico.cern.ch/event/1337334/contributions/5692346/attachments/2775295/4836434/data_analysis_basics.pdf>`_
* Jupyter notebooks on Calculus: ``/data/work/analysis/sst1m_analysis_handson_2023/``

Configuration files
~~~~~~~~~~~~~~~~~~~

There are two slightly different configuration files. One to be used for MC processing
(needed only if one wants to get own RF models / IRFs / performance curves), and one 
for real data processing. These contain configuration for all reconstruction/analysis steps 
from R0 up to DL3 (or performance evaluation in case of MC).

Default config file for MC:

.. toggle:: 

    .. include:: ../sst1mpipe/data/sst1mpipe_mc_config.json
       :code: json

Default config file for data:

.. toggle:: 

    .. include:: ../sst1mpipe/data/sst1mpipe_data_config.json
       :code: json


R0 to DL1
~~~~~~~~~

To calibrate raw data (R0) or MC (R1) and process them to DL1, one may run script ``sst1mpipe_r0_dl1``. Inputs are a single raw 
.fits.fz data file (containing single telescope data) or .simtel.gz output file of sim_telarray (may contain more telescopes).
Output is HDF5 file with a table of DL1 parameters, one outputfile per observing run and per telescope (DL1 files from individual 
telescopes are merged for stereoscopic reconstruction in the next step :ref:`dl1_dl1_stereo`).

It applies dc to p.e. calibration on raw R0 waveforms, integrate them, clean the images (get rid of the noisy pixels which do not 
contain any Cherenkov photons), and parametrize the shower images with Hillas ellipses. All this can be done by running 
See ``--help`` for possible inputs. Some of them, which might not be obvious:

* ``--px-charges`` - the script stores also distribution of all integrated charges in individual pixels for all events merged. This is good for further MC/data tunning and to get some impression on the level of NSB in the data.

* ``--precise-timestamps`` - stores also White Rabbit timestamps in the DL1 output with the precision needed for matching matching of coincident events. Keep it on for all data taken after 25th September 2023.

* ``--pointing-ra/dec`` and ``--force-pointing`` - allows to specify the telescope pointing direction. In all data taken from begining of September 2023 it can be ignored (i.e. do not use it for any new data), because the pointing coordinates are being written automaticaly in the fits file header during the datataking and the script understands where to look for it.

* ``—-reclean`` - experimental method of data re-cleaning based on pixel charge variation. For now it needs distribution of pixel charges stored in the first pass of the script. I.e. to apply re-cleaning, one has to run the script for the second time having this switch activa.

**Relevant parts of the config file** applied in this analysis step:

* ``telescope_calibration`` - calibration files based on dark run analysis. Should be taken relatively close to the data of observation

* ``window_transmittance`` - files with for camera window transmittance correction (measured in the lab and can be kept default)

* ``CameraCalibrator`` - Pulse integration settings

* ``ImageProcessor`` - Settings of image cleaning method, tailcuts and NSB bins with different tailcuts

* ``ShowerProcessor`` - Shower geometry reconstruction. Only applied if event source contains data from more telescopes, i.e. it's only relevant for MC processing in this analysis step.


Random Forest training
~~~~~~~~~~~~~~~~~~~~~~

.. note::

    In most cases **analyser does not need to train dedicated Random Forest models and this step can be safely skipped** using pre-trained RFs 
    referenced in :ref:`dl1_dl2`. Training of dedicated RFs is, hovewer, necessary in some performance studies of if one wants to use different 
    configuration for ``sst1mpipe_r0_dl1`` than MC was processed with.

Random Forests can be trained on **DL1 MC diffuse gammas and diffuse proton** files using ``sst1mpipe_mc_train_rfs`` script (see 
``--help`` for possible inputs). Before running ``sst1mpipe_mc_train_rfs`` it is useful to merge many small DL1 MC files resulting 
from paralelized MC simulations into a single file per particle with ``sst1mpipe_merge_hdf5`` script to reach satisfactory 
statistics for RF training. Outputs are trained models in ``scikit.learn`` format (.sav). There is RF classifier for gamma/hadron
separation, RF regressor for energy reconstruction, and either RF regressor (``disp_vector``) or RF regressor+classifier (``disp_norm_sign``) 
for arrival direction reconstruction depenting on the method selected.

RF are trained for each telescope, even in case of stereo reconstruction. In stereo, we only use extra stereo features, 
which are reconstructed geometricaly, such as ``HillasReconstructor_h_max`` and ``HillasReconstructor_tel_impact_distance``. 
Then, in :ref:`dl1_dl2`, reconstruction is done for each telescope independently, and final reconstructed quantities are 
obtained as weighted average of the values for each telescope.

**Relevant parts of the config file** applied in this analysis step:

* Setup of the forests and training procedure ``random_forest_regressor_args``, ``random_forest_classifier_args``

* Random Forest features used for the reconstruction - ``energy_regression_features``, ``disp_regression_features``, ``disp_classification_features``, ``particle_classification_features``. The very featuresused for RF training have to be used later in :ref:`dl1_dl2` reconstruction 

* ``n_training_events`` - Total number of events used for individual RF training. I.e. if ``n_training_events=200000``, 200k diffuse gammas are used for energy regressor and DISP regressor and classifier, and 100k diffuse gammas + 100k diffuse protons is used for particle classifier (if ``gamma_to_proton_training_ratio=1``).

* ``gamma_to_proton_training_ratio`` - Ratio of gammas and protons in training sample for particle classifier.


.. _dl1_dl1_stereo:

DL1 to DL1 stereo
~~~~~~~~~~~~~~~~~

For stereo reconstruction, we need to find coincident events in tel2 DL1 data to each tel1 DL1 event. As of now, for each tel1 DL1 file, 
we search in the events taken with Tel2 to find the closest one, resulting in a new DL1 file containing events from both 
telescopes, matched by their event_id (only coincident events are stored in resulting DL1 files). 

This is performed by script ``sst1mpipe_data_dl1_dl1_stereo`` (see ``--help`` for possible inputs.). Input is a single DL1 file from tel1 
and a directory with all relevant DL1 files for tel2. Coincidence finder is driven by **the config file** field ``stereo``. Possible 
options are:

* ``SlidingWindow`` - For analysis of the data without precise White Rabbit timestamps (i.e. taken until 25th September 2023) one needs to use this method, which first finds the time time offset between the two tables providing maximum number of coindicent events and then selects the closest ones.

* ``WhiteRabbitClosest`` - Works on data with precise WR timestamps in the DL1 table, i.e. all data taken after 25th September 2023. It only finds the closest tel2 event to each tel1 event (precision of WR is high enough to avoid random coincidences for usual trigger rates of the telescopes).

* ``SWATEventIDs`` - After 30th January 2024 the coincident events are tagged by SWAT, providing them with the same ``arrayEvtNum``, resulting in the same ``event_id`` in the DL1 files. The DL1 mono events can be then matched just based on the ``event_id``.

.. note::

    ``sst1mpipe_data_dl1_dl1_stereo`` is **not intended to be run on MC**, as in MC DL1 we already have coincident events matched by their ``event_id`` (mono events are in the DL1 
    tables as well, so those can be used for both mono and stereo analysis).


.. _dl1_dl2:

DL1 to DL2
~~~~~~~~~~

This step uses pre-trained Random Forests to reconstruct shower parameters (gammaness, direction and energy) using Hillas parameters stored in 
DL1 files as features. One can run ``sst1mpipe_data_dl1_dl2`` on either mono DL1 files (outputs of ``sst1mpipe_r0_dl1``) for each telescope separately (using RFs for mono reconstruction), 
or on stereo DL1 containing coincident events only (outputs of ``sst1mpipe_data_dl1_dl1_stereo``). The script can handle both types of DL1, 
but stereo reconstruction has to be requested using ``-—stereo`` switch. RFs trained on MC can be found on **Calculus** for both mono and stereo 
reconstruction and different zenith angles:

* ``/data/work/analysis/MC/prod_january_2023/$SST1MPIPE_VER/models_mono_psf_vaod0.2/``

* ``/data/work/analysis/MC/prod_january_2023/$SST1MPIPE_VER/models_stereo_psf_vaod0.2/``

.. note::

    One should always use RF models trained with the same sst1mpipe version that is used for the analysis. 

**Relevant parts of the config file** applied in this analysis step:

* Random Forest features used for the reconstruction - ``energy_regression_features``, ``disp_regression_features``, ``disp_classification_features``, ``particle_classification_features``. These should be the very same features which were used for RF training (check cfg files stored in the directories together with the models)

* ``disp_method`` - Direction reconstruction method used. For now we only use ``disp_norm_sign`` which requires RF regressor to reconstruct source distance from a shower Center of Gravity, and RF classifier to determine on which side along the main axis of the Hillas ellipse the source lies.

* ``stereo_reco_weights`` - Parameter used as a weight for averaging of stereo reconstructed parameters.


DL2 MC to IRFs
~~~~~~~~~~~~~~

.. note::

    In most cases **analyser does not need to produce own Instrument Response Functions and this step can be safely skipped** using IRFs referenced in 
    :ref:`automatic_processing`. IRF production, however, is necessary in performance studies, or is one use custom RFs to produce DL2, or applies
    custom selection cuts in DL2 to DL3 step.

To make IRFs from MC DL2 files, one can run ``sst1mpipe_mc_make_irfs``, which currently produce only full enclosure IRFs, so it has to be provided with 
diffuse protons and gammas. The script applies event selection cuts defined in the config file (``event_selection``), including cut on gammaness. 
The gammaness cut can be either global (one number independent on energy) or energy dependent (expected performance improvement as gammaness 
distribution depends on energy). Global gammaness cut can be set in the config file (``global_gammaness_cut`` field), while energy dependent
cuts, must be provided as path a to the HDF5 table, where the cuts for individual energy bins are stored, using parameter 
``--gammaness-cut-dir``. These can be generated with ``sst1mpipe_mc_performance`` (see :ref:`performance`). Pre-calculated energy dependent gammaness 
cuts are stored on Calculus for mono/stereo and different zenith angles:

``/data/work/analysis/MC/prod_january_2023/$SST1MPIPE_VER/performance/*_performance_*``

Again, one should make sure that the event selection applied to produce the cuts is the same as in the cfg file for IRF. The IRF maker creates 
some directory structure inside the ``--output-dir``, automaticaly recognizing proper bin in zenith, azimuth, NSB level and gammaness cut applied. 
This directory structure should remain untouched for :ref:`dl2_dl3` to work properly.

Output IRF files are fully compatible with gammapy and may be read and explored with the use of gammapy funkcionalities:

.. code-block:: console

    from gammapy.irf import (
        EffectiveAreaTable2D,
        PSF3D,
        EnergyDispersion2D
        Background2D
    )
    irf_filename = 'SST1M_tel_021_Zen30deg_gcut0.75_irfs.fits'
    aeff = EffectiveAreaTable2D.read(irf_filename, hdu="EFFECTIVE AREA")
    edisp = EnergyDispersion2D.read(irf_filename, hdu="ENERGY DISPERSION")
    psf = PSF3D.read(irf_filename, hdu='POINT SPREAD FUNCTION')
    bg_2d = Background2D.read(irf_filename, hdu='BACKGROUND')


.. _dl2_dl3:

DL2 to DL3
~~~~~~~~~~

``sst1mpipe_data_dl2_dl3`` is a tool to create DL3 data files from DL2 data files. It is supposed to be provided with a directory with input DL2 files 
(typicaly a directory with DL2 files for one source observed in one night, but can be run on larger sample as well). It merges the DL2 ``HDF5`` per-run 
files into per-wobble DL3 ``fits`` files. It also finds proper IRF based on zenith, azimuth and NSB level of each input DL2 file. It also creates 
per-night index files needded for further analysis in gammapy. The script applies event selection cuts defined in the config file (``event_selection``), 
including cut on gammaness, so **one should make sure that these are the same as used on MC to produce IRFs**.

Output DL3 files are fully compatible with gammapy and may be further analyzed using gammapy tools. See e.g. 

* `Tutorial on 1D spectral analysis <https://docs.gammapy.org/1.2/tutorials/analysis-1d/spectral_analysis.html>`_

* `Tutorial on 2D ring background map <https://docs.gammapy.org/1.2/tutorials/analysis-2d/ring_background.html>`_

A typical use case is to run joint gammapy analysis on data from several nights. In such case one has to use ``create_hdu_indexes`` script to create 
HDU index files indexing all DL3s to be used in the final analysis.


.. _performance:

RF performance and sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TBD