TDMDpy
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/tdmdpy/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/tdmdpy/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/TDMDpy/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/TDMDpy/branch/master)


TDMDpy: A python package to calculate thermodynamic quantities from MD simulations.

### Feautres
* Orthobaric densities
* Radial distribution functinos
* Self diffusion coefficients
* Enthalpy of vaporization
* Vibrational density of states

### Advanced Features
* Single-point energy and force predictions with ML potentials
* Normal mode analysis for single water molecule in gas-phase, using [kALDo](https://github.com/nanotheorygroup/kaldo)
* Create systems of hydrogen-disordered ice structures, using [genice2](https://github.com/vitroid/GenIce)

### Disclaimers
* Examples highlighting usage of TDMDpy will be acessible in a spearate [repo](https://github.com/nanotheorygroup/water_ice_nep) as relevant publication is avialable. 
* TDMDpy is currently developed for linux machine only, and might be made to compatiable with Windows later. 

### Installation

```console
pip install git+https://github.com/ZKC19940412/tdmdpy
```

### Copyright

Copyright (c) 2024, Zekun Chen, Bohan Li


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
