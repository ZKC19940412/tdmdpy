from ase.calculators.lammpslib import LAMMPSlib
from ase.io import read
from pynep.calculate import NEP
from tdmdpy.mlps import single_point_energy_force_prediction
import sys

if __name__ == "__main__":

    # Uptake user define data path
    data_path = './liquid_water_aimd_300K.xyz'

    # Derive number of configurations
    number_of_configurations = len(read(data_path, index = ':'))

    # Define NEP calculator
    nep_calculator = NEP('nep.txt')

    print('ML Potential : NEP')
    single_point_energy_force_prediction(number_of_configurations, nep_calculator,
                                         data_path, atoms_per_molecule=3)
