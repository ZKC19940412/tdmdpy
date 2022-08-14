"""A python package for postprocessing MD simulations from LAMMPS."""

# Add imports here
from .atom_manipulate import *
from .thermodynamic_properties import *
from .mlps import *
from .utility import *
from .scorer import *
from .ffscore import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
