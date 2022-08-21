#!/bin/sh

# Obtain kALDo from remote
git clone https://github.com/nanotheorygroup/kaldo.git

# Install from source
cd kaldo
pip install .
