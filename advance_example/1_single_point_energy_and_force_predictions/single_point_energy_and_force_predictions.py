from ase.calculators.lammpslib import LAMMPSlib
from ase.io import read
from pynep.calculate import NEP
from tdmdpy.mlps import single_point_energy_force_prediction
import sys

if __name__ == "__main__":

    # Uptake user define data path
    data_path = sys.argv[1]

    # Derive number of configurations
    number_of_configurations = len(read(data_path, index = ':'))

    # Define NEP calculator
    nep_calculator = NEP('nep.txt')

    # Define SNAP calculator under LAMMPSlib
    lammps_inputs = {
       'lmpcmds': [
           'pair_style snap',
           'pair_coeff * * H2O_pot.snapcoeff H2O_pot.snapparam H O'],
       'keep_alive': True}
    snap_calculator = LAMMPSlib(**lammps_inputs)
    print('ML Potential : NEP')
    single_point_energy_force_prediction(number_of_configurations, nep_calculator,
                                         data_path, atoms_per_molecule=3)
    