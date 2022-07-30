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


def md_simulator(atoms, ensemble_str, thermostat_str,
                 tag=None, calculator=None, rc_val=4.5,
                 T_initial=None, T_final=None, T_tau=None,
                 P_final=None, P_tau=None,
                 time_step_val=1, total_step=1000):
    """Perform MD simulation with machine learning (ML) potentials.

           input:
           atoms: (ASE atom object) ASE atom object containing the whole trajectory
           ensemble_str: (str) Name of the ensemble
           thermostat_str: (str) Name of thermostats
           tag: (str) Tag for output log
           calculator: (ASE calculator object) ML potential
           rc_val: (float) cutoff distance used in TIP4P potential
           T_initial: (float) Initial temperature
           T_final: (float) Target temperature
           T_tau (float): Temperature coupling constant
           P_initial: (float) Initial pressure
           P_final: (float) Target pressure
           P_tau (float): pressure coupling constant
           time_step_val (float): time step in fs
           total_step （int）: Total number of steps

           output:
           md_simulation (ASE MD object)

    """

    # Set TIP4P as default calculator
    if calculator is None:
        calculator = TIP4P(rc=rc_val)

    # Allocate calculator object into atoms
    atoms.calc = calculator

    # Set the momenta corresponding to initial temperature
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_initial)

    # Initialize MD simulation based on ensemble_str
    if ensemble_str == 'NVE':
        from ase.md.verlet import VelocityVerlet

        md_simulation = VelocityVerlet(atoms, time_step=time_step_val * units.fs, logfile=tag + '.log')

    elif ensemble_str == 'NVT':
        if thermostat_str == 'langevin':
            from ase.md.langevin import Langevin
            md_simulation = Langevin(atoms, timestep=time_step_val * units.fs,
                                     temperature=T_final * units.kB,
                                     friction=0.001, logfile=tag + '.log')

        if thermostat_str == 'berendsen':
            from ase.md.nvtberendsen import NVTBerendsen
            md_simulation = NVTBerendsen(atoms, timestep=time_step_val * units.fs,
                                         temperature=T_final * units.kB,
                                         taut=T_tau * units.fs, logfile=tag + '.log')

    elif ensemble_str == 'NPT':
        if thermostat_str == 'berendsen':
            from ase.md.nptberendsen import NPTBerendsen
            md_simulation = NPTBerendsen(atoms, timestep=time_step_val * units.fs,
                                         temperature=T_final * units.kB,
                                         pressure_au=P_final,
                                         taut=T_tau * units.fs,
                                         taup=P_tau * units.fs,
                                         logfile=tag + '.log')

    trajectory = Trajectory(tag + '.traj', 'w', atoms)
    md_simulation.attach(trajectory.write, interval=1)
    md_simulation.run(total_step)

    return md_simulation


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
    print('Force RMSE (meV/Å): %.3f ' % (
            1e3 * mean_squared_error(F_full_vector, F_predictions_full_vector) ** 0.5))
