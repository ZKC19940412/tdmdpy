from ase.build import molecule
from ase.optimize import BFGS
from kaldo.phonons import Phonons
from kaldo.forceconstants import ForceConstants
import numpy as np
import os
from pynep.calculate import NEP
import scipy.constants as spc
import shutil

# Remote precomputed file
if os.path.exists('fd'):
    shutil.rmtree('fd')

# os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Representinfg single molecule at gamma point
nrep = 1
k_points = 1

# Generate initial structure
atoms = molecule('H2O')
nep_calculator = NEP('nep.txt')
atoms.calc = nep_calculator

# Set up an arbtilary large box
atoms.set_pbc([True, True, True])
atoms.cell = np.array([100.1,100.1,100.1])

# Optimize single water molecule structutre
BFGS(atoms).run(fmax=0.001)

# Replicate the unit cell 'nrep'=3 times
supercell = np.array([nrep, nrep, nrep])

# Create a finite difference object
forceconstants_config  = {'atoms':atoms,'supercell': supercell,'folder':'fd'}
forceconstants = ForceConstants(**forceconstants_config)

# Compute 2nd and 3rd IFCs with the defined calculators
forceconstants.second.calculate(calculator = nep_calculator, delta_shift=1e-3)
forceconstants.third.calculate(calculator = nep_calculator, delta_shift=1e-3)

# Define the k-point mesh using 'kpts' parameter
is_classic = False
phonons_config = {'kpts': [k_points, k_points, k_points],
                  'is_classic': is_classic,
                  'temperature': 300,  # 'temperature'=300K
                  'folder': 'ALD',
                  'storage': 'memory'}

phonons = Phonons(forceconstants=forceconstants, **phonons_config)
frequencies = spc.deka * spc.giga * spc.value('hertz-inverse meter relationship') * phonons.frequency.flatten(order='C').copy()

# Reference literature:
# https://physicsopenlab.org/2022/01/08/water-molecule-vibrations-with-raman-spectroscopy/
print('\n')
print('Number of imaginary frequencies: %d' % (len(frequencies[frequencies < 0])))
n = 44
print(n * "=")
print(f"{'              Frequency Summary       ':{n}s}")
print(n * "=")
print("ML modelled (cm^-1)  | Exp Ref (cm^-1)")
print('     ' + str(int(np.round(frequencies[6]))) + '                ' + str(1595))
print('     ' + str(int(np.round(frequencies[7]))) + '                ' + str(3657))
print('     ' + str(int(np.round(frequencies[8]))) + '                ' + str(3756))
print(n*"=")
