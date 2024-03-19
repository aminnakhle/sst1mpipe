.. _sst1m_analysis_workflow:

SST-1M analysis workflow
========================

Authomatic data processing
--------------------------

Raw data from both telescopes are processed authomaticaly up to DL3 level 
every morning after an observing night on **Calculus**. This processing is 
performed with the following pipeline:

``/mnt/nfs22_auger/jurysek/sst1mpipe/sst1mpipe/scripts/daily_analysis.py``

It's products (all data levels up to DL3) can be found in the following directory:

``/data/work/analysis/Daily_analysis/daily_data/``

Sorted by date, source observed, and the telescope used (cs1, cs2 and stereo for coincident events).
All data level directories also contain configuration file and log file for each processed run, so 
user can easily check if everything gone OK. Random Forest models used in in authomatic analysis can be found in

``/data/work/analysis/MC/prod_january_2023/``

for both mono and stereo regime of observation, together with expected performances. IRFs for DL3
production can be found in

``/data/work/analysis/IRFs/``

DL3 products of the authomatic analysis can be further analyzed using using standard tools 
such as `gammapy <https://github.com/gammapy>`_. 

.. note::

    As the software developemnt is progressing very fast, this processing is usually 
    done with slightly obsolete version of sst1mpipe and with configuration not fully 
    tunned for atmospheric conditions during particular nights. To obtain *paper level* 
    analysis, one should proceed with the steps described in :ref: `detailed_processing`.


.. _detailed_processing:

Detailed data processing
------------------------