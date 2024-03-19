.. _sst1m_analysis_workflow:

SST-1M analysis workflow
========================

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

