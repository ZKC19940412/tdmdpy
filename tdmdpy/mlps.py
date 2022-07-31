from ase.calculators.tip4p import TIP4P
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.io import read
from ase import units
import numpy as np
from sklearn.metrics import mean_squared_error


def extract_E_F_from_single_frame(truth_xyz, calculator_object, index):
    truth_ase_object = read(truth_xyz, index=index)
    truth_ase_object.calc = calculator_object

    number_of_atom = len(truth_ase_object.get_masses())
    per_atom_energy_truth = truth_ase_object.info['energy'] / number_of_atom
    per_atom_energy_prediction = truth_ase_object.get_total_energy() / number_of_atom

    force_truth_vector = truth_ase_object.arrays['forces'].flatten(order='C')
    force_predict_vector = truth_ase_object.get_forces().flatten(order='C')
    return per_atom_energy_truth, per_atom_energy_prediction, force_truth_vector, force_predict_vector


def single_point_energy_force_prediction(number_of_configurations, calculator,
                                         data_path,
                                         is_save_energy_and_force_file=False,
                                         atoms_per_molecule=3):

    # Preset lists for data intake
    per_atom_E_list = []
    per_atom_E_predictions_list = []
    F_list = []
    F_predictions_list = []

    for i in range(number_of_configurations):
        index = str(i)
        per_atom_E, per_atom_E_prediction, F, F_prediction = extract_E_F_from_single_frame(
            data_path, calculator, index)
        per_atom_E_list.append(per_atom_E)
        per_atom_E_predictions_list.append(per_atom_E_prediction)
        F_list.append(F)
        F_predictions_list.append(F_prediction)

    # Cast energy lists into arrays and concatenate force lists
    per_atom_E_vector = np.array(per_atom_E_list)
    per_atom_E_predictions_vector = np.array(per_atom_E_predictions_list)
    F_full_vector = np.concatenate(F_list)
    F_predictions_full_vector = np.concatenate(F_predictions_list)

    if is_save_energy_and_force_file:
        np.save('energy.npy', np.vstack([per_atom_E_vector, per_atom_E_predictions_vector]).T)
        np.save('force.npy', np.vstack([F_full_vector, F_predictions_full_vector]).T)

    print('Energy RMSE (meV/molecule): %.3f ' % (
            atoms_per_molecule * 1e3 * mean_squared_error(per_atom_E_vector, per_atom_E_predictions_vector) ** 0.5))
    print('Force RMSE (meV/A): %.3f ' % (
            1e3 * mean_squared_error(F_full_vector, F_predictions_full_vector) ** 0.5))
