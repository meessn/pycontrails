This is a fork of pycontrails (see below) and contains the framework implemented in https://resolver.tudelft.nl/uuid:0e27a8d7-7ddf-470d-9da5-cc19a2abe612
# Important 
Requires GSP (Gas Turbine Simulation Programme) and its API (GSP12). The API is a 32-bit DLL and should therefore be run with a 
32-bit Python interpreter (Python 39-32). See main_emissions.py where a subprocess runs part of the code (interaction with the API) with a 32-bit interpreter. The rest of the code is executed using a 64 bit interpreter (Python 310 64).
See env_pythons folder for used packages. 
# Structure
`main.py` uses functions from `main_emissions.py` and `main_climate.py` to compute the emissions and climate impact of the considered flights. `main_ams_temp.py` and `main_new_flights.py` are examples to compute the emissions and climate impact of extra flights not considered in the thesis report. Fuel flow and emissions prediction can be based on pycontrails (Poll-Schumann model + FFM2 + T4/T2 method) or on the master thesis 'mees' (P3T3 NOx, Adjusted MEEM, GSP (see thesis)).

Also a computational time improvements cr_appr can be used. Next to that ERA5 pressure or model level data can be used (see master thesis chapter 4.7)

`flight_trajectories.py` and `flight_trajectories_extra.py` show examples of how to use OpenSky Network retrieved flight data to prepare them for the simulations in this framework. It also fills gaps in the trajectory.

`main_results_read_out.py` correctly calculates the total emissions and climate impact per flight. IMPORTANT: here WTW EI are used for SAF blends and the aCCF calculations are correct! Therefore use `results_read_out.py` for climate impact results!

The GSP API can fail to retrieve output, then use code such as in `main_check_results_gsp.py` to look for flights where this happened and to recompute the flights. 

`main_results_figures` folder contains the results per flight and simulated time. Important are the emissions, climate impact and (CoCiP) contrails (mean contrail properties and full properties in parquet file) csv files. Column name explanation can be found in `main_results_figures/column_names_explanation` folder for emissions and climate csv files, CoCiP column names explained in pycontrails documentation. 

`results_xxxxx.py` files are result generation used for the thesis report. 

Some folders have been added that contain additional code for midterm results, water_injection optimization etc.

GSP engine models are required. Contact github owner. 

# pycontrails

> Python library for modeling aviation climate impacts

|               |                                                                   |
|---------------|-------------------------------------------------------------------|
| **Version**   | [![PyPI version](https://img.shields.io/pypi/v/pycontrails.svg)](https://pypi.python.org/pypi/pycontrails)  [![conda-forge version](https://anaconda.org/conda-forge/pycontrails/badges/version.svg)](https://anaconda.org/conda-forge/pycontrails) [![Supported python versions](https://img.shields.io/pypi/pyversions/pycontrails.svg)](https://pypi.python.org/pypi/pycontrails) |
| **Citation**  | [![DOI](https://zenodo.org/badge/617248930.svg)](https://zenodo.org/badge/latestdoi/617248930) |
| **Tests**     | [![Unit test](https://github.com/contrailcirrus/pycontrails/actions/workflows/test.yaml/badge.svg)](https://github.com/contrailcirrus/pycontrails/actions/workflows/test.yaml) [![Docs](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yaml/badge.svg?event=push)](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yaml) [![Release](https://github.com/contrailcirrus/pycontrails/actions/workflows/release.yaml/badge.svg)](https://github.com/contrailcirrus/pycontrails/actions/workflows/release.yaml) [![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/contrailcirrus/pycontrails/badge)](https://securityscorecards.dev/viewer?uri=github.com/contrailcirrus/pycontrails)|
| **License**   | [![Apache License 2.0](https://img.shields.io/pypi/l/pycontrails.svg)](https://github.com/contrailcirrus/pycontrails/blob/main/LICENSE) |
| **Community** | [![Github Discussions](https://img.shields.io/github/discussions/contrailcirrus/pycontrails)](https://github.com/contrailcirrus/pycontrails/discussions) [![Github Issues](https://img.shields.io/github/issues/contrailcirrus/pycontrails)](https://github.com/contrailcirrus/pycontrails/issues) [![Github PRs](https://img.shields.io/github/issues-pr/contrailcirrus/pycontrails)](https://github.com/contrailcirrus/pycontrails/pulls) |

**pycontrails** is an open source project and Python package for modeling aircraft contrails and other
aviation related climate impacts.

`pycontrails` defines common [data structures](https://py.contrails.org/api.html#data) and [interfaces](https://py.contrails.org/api.html#datalib) to efficiently build and run [models](https://py.contrails.org/api.html#models) of aircraft performance, emissions, and radiative forcing.

## Documentation

Documentation and examples available at [py.contrails.org](https://py.contrails.org/).

<!-- Try out an [interactive Colab Notebook](). -->

## Install

### Install with pip

You can install pycontrails from PyPI with `pip` (Python 3.10 or later required):

```bash
$ pip install pycontrails

# install with all optional dependencies
$ pip install "pycontrails[complete]"
```

Install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/contrailcirrus/pycontrails.git
```

### Install with conda

You can install pycontrails from the [conda-forge](https://conda-forge.org/) channel with `conda` (or other `conda`-like package managers such as `mamba`):

```bash
conda install -c conda-forge pycontrails
```

The conda-forge package includes all optional runtime dependencies.

See more installation options in the [install documentation](https://py.contrails.org/install).

## Get Involved

- Ask questions, discuss models, and present ideas in [GitHub Discussions](https://github.com/contrailcirrus/pycontrails/discussions).
- Report bugs or suggest changes in [GitHub Issues](https://github.com/contrailcirrus/pycontrails/issues).
- Review the [contributing guidelines](https://py.contrails.org/contributing.html) and contribute improvements as [Pull Requests](https://github.com/contrailcirrus/pycontrails/pulls).

## License

[Apache License 2.0](https://github.com/contrailcirrus/pycontrails/blob/main/LICENSE)

Additional attributions in [NOTICE](https://github.com/contrailcirrus/pycontrails/blob/main/NOTICE).
