#!/bin/sh

# Pip install pyblind11 as the simlink seems to break
pip install pybind11

# Obtain PyNEP from remote
git clone https://github.com/ZKC19940412/PyNEP.git

# Install from source
cd PyNEP
pip install .
