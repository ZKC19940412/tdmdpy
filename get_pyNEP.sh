#!/bin/sh

# Obtain PyNEP from remote
git clone https://github.com/ZKC19940412/PyNEP.git

# Install from source
cd PyNEP
python -m setup.py
