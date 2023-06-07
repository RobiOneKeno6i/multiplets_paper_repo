# seismic multiplets scan routine

The repository contains:

* **mathematica_code.nb**; the originally written *mathematica code notebook*. This is the software complete with all features and an interactive GUI. The notebook runs in a licensed Wolfram Mathematica environment (https://www.wolfram.com/mathematica) or in the Wolfram Player Pro (https://www.wolfram.com/player-pro/), the commercial paid version with I/O enabled.

* **Wolfram_CDF_example.cdf.zip** a Wolfram CDF file (<https://www.wolfram.com/cdf/>). This file runs freely under the CDF player and behaves exactly as the mathematica interactive window code. The only difference is that data I/O is not enabled and catalog is completely contained and hardcoded in the CDF. It is provided as the GUI functionality demo.

* the python porting of the catalog scanning routine core algorithm, added in two formats: 
   * **python\_code.py**. This plain python code needs some classes that must be installed in the local machine. It can easily be done with standard python package managers like *pip* or conda <https://docs.conda.io/en/latest/>. File **conda\_env\_export.yml**     contains some (not strictly necessary) info about the environment under which the software actually runs.
   * the Jupyter/Colab *ipython version* (**multiplets_core_colab.ipynb**) is the same software but ready to run under Jupyter (<https://jupyter.org/>), a web based IDE much more similar to the mathematica model. A great aspect of this paradigm is the possibility to let the program run in a web environment like Google Colab (<https://colab.google>), without the need to have a python interpreter locally installed.
 
 The code has been checked to behave exactly as the mathematica code. No GUI programming is anyway present: I/O is actually based on disk files read-write. Any user interface can be easily added by experienced Python programmers. 

* an *example catalog file* for the python code testing (**Simulated_100k_005-02-2_geo.txt.zip**). It is in the format needed by the software importer. 
This file is a table of 6 columns, each row being a seismic event. The six columns in order from left to right must contain these event info:
1) Time in decimal years, 2) latitude in decimal degrees 3) longitude in decimal degrees, 4) depth in km (positive numbers, higher number is deeper), 5) magnitude, 6) free event ID left for user reference.
The last ID item is not used by the software but this version of the importer is expecting some number to be present.
* other not strictly necessary files (python_mp.txt, python_orig.txt, python_vs_out_new.txt) have been added for more information: they are generated by the python code itself on the given example catalog when parameters are set to (see also yaml files):
  * removed = "GK"
  * gkrad = "sum"; 
  * magthresh=5.5
  * dmplus=.4
  * dmminus=.6

They are present for reference only: in particular 
* **python_mp.txt** contains the mp vector extracted in the python porting when the magnitude prefiltering is applied to the catalog
* **python_orig.txt** contains the internally complete converted catalog table where coordinates are referred to a local cartesian coordinate system (the exact origin is not important since filtering is dependent upon *relative* coordinates). 
  The last column of the converted table is the row ID of the complete catalog and is used by the internal routines
* **python_vs_out_new.txt** is the complete result of the cluster search. It contains catalog file and searching parameters info, counts of found multiplets together with detailed lists of the event IDs clustered together. 

On a modern Mac arm M1 processor the rather big synthetic catalog example file contains 337254 events from which a subset of 95807 events is prefiltered. Search time on this subset, for the parameters reported above, gives out 2123 clusters found in about 2 minutes and half of processing time.

The python code uses some classes that must be installed in the local environment, the file conda_env_export.yml is the Conda export of the environment under which the software actually runs. Even if it is not straightforward to reconstruct the python environment from this file using Conda, it can anyway give some useful info.

In both python executables, assignable variables are actually hard-coded in the program code. For as regarding the I/O file names, they have no assigned path, therefore the python interpreter expects to find them in the same directory were the program is. Before running the program, user should thus check the path to files, the threshold magnitudes and the flags. The Gardner & Knopoff tables are defined as lists in the code too.

The jupyter version should follow the same prescriptions of the python IDE if it is run locally. However, in the web IDE version of Google Colab, user could need extra disk space and should connect the web IDE to the corresponding (same user) Google Drive space. In the code there is a cell written to the purpose, and the procedure is  well documented in the Google Colab documentation.

As a last add-on, two files generated from the reproducible class (the code to do this has been left commented) generated yaml file has been added to give informations on the actual working environment under which the file runs (see also the Conda environment file described above).

The code is open source, not maintained and hopefully "citation-ware" i.e. it can be used completely free, and a citation to the author is more than welcome.
