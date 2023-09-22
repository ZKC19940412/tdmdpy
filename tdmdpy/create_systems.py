from .atom_manipulate import make_atom_object
from ase import Atoms
from ase.constraints import FixBondLengths
from ase.calculators.tip4p import TIP4P, rOH, angleHOH
from ase.md import Langevin
import ase.units as units
from ase.io.trajectory import Trajectory
from ase.io import read, write
import numpy as np
import os
import scipy.constants as spc
import subprocess


def generate_water_box(target_density=0.994,
                       number_of_molecules=64,
                       is_pre_equilibrate=True,
                       equilibrate_temperature=300,
                       equilibrate_time_in_fs=4000):
    """Generate water box

                 input:
                 target_density: (float) target density of systems
                 number_of_molecules: (int) number of molecules
                 is_pre_equilibrate: (boolean) whether to pre-equilibrate system
                 equilibrate_temperature: (float) temperature for equilibrate
                 equilibrate_time_in_fs: (float) total time for equilibrate

                 output:
                 systems : (atom object) water box system
    """

    # Set up water box at 20 deg C density
    x = angleHOH * np.pi / 180 / 2
    positions = [[0, 0, 0],
                 [0, rOH * np.cos(x), rOH * np.sin(x)],
                 [0, rOH * np.cos(x), -rOH * np.sin(x)]]
    atoms = Atoms('OH2', positions=positions)

    molar_mass_water = 18.01528
    NA = spc.value('Avogadro constant')
    box_length = ((molar_mass_water / NA) / (target_density / 1e24))**(1 / 3.)
    atoms.set_cell((box_length, box_length, box_length))
    atoms.center()

    number_or_replica = int(np.ceil(number_of_molecules ** (1 / 3.)))
    atoms = atoms.repeat(number_or_replica)
    atoms.set_pbc(True)

    # RATTLE-type constraints on O-H1, O-H2, H1-H2.
    atoms.constraints = FixBondLengths([(3 * i + j, 3 * i + (j + 1) % 3)
                                        for i in range(3 ** 3)
                                        for j in [0, 1, 2]])
    atoms.arrays['mass'] = atoms.get_masses()
    systems = atoms
    if is_pre_equilibrate:
        tag = 'tip4p_equil'
        atoms.calc = TIP4P(rc=10.0)
        md = Langevin(atoms, 1 * units.fs,
                      temperature=equilibrate_temperature * units.kB,
                      friction=0.01, logfile=tag + '.log')

        trajectory = Trajectory(tag + '.traj', 'w', atoms)
        print('Equilibration ongoing ...')
        md.attach(trajectory.write, interval=1)
        md.run(equilibrate_time_in_fs)
        equilibrated_atoms = read(tag + '.traj')
        equilibrated_atoms.arrays['mass'] = equilibrated_atoms.get_masses()
        write('model.xyz', equilibrated_atoms)
        os.remove(tag + '.traj')
        systems = equilibrated_atoms
    else:
        write('model.xyz', atoms)

    return systems


def generate_ice_structures(
                            type_str='ih',
                            number_of_total_replica=27,
                            is_output_lmp=False):
    """Generate hydrogen disordered ice structure, powered by genice2

                    input:
                    target_density: (float) target density of systems
                    type_str: (str) type of  hydrogen disordered ice
                    number_of_molecules: (int) number of molecules
                    is_output_lmp: (boolean) whether to output lmp file

                    output:
                    systems : (atom object) hydrogen disordered ice system
    """

    # Derive number of replica
    number_of_replica = int(number_of_total_replica ** (1 / 3.))

    # Construct genice command
    genice_command = 'genice2 --rep ' + str(number_of_replica) + ' ' + str(
        number_of_replica) + ' ' + str(
        number_of_replica) + ' ' + type_str + ' --format exyz > tmp.xyz'

    # Run this command via subprocess
    subprocess.call(genice_command, shell=True)

    # Load in chemical symbols and coordinate from generated structure
    # "-4" is used to get rid of the extract line generated fom genice2
    configuration_chemical_symbols = np.loadtxt('tmp.xyz', skiprows=4,
                                                usecols=0, dtype=str)[:-4]
    configuration_coordinate = np.loadtxt('tmp.xyz',
                                          skiprows=4, usecols=(1, 2, 3))[:-4]
    
    # Compute box length based on density and number of atoms
    number_of_atoms = len(configuration_coordinate)
    
    # Derive cell dimension
    cell = np.zeros([3])
    for i in range(3):
        cell[i] = np.loadtxt('tmp.xyz', skiprows=number_of_atoms + 5 + i, 
                                    usecols=i + 1)[0]
    
    # Get rid of tmp file
    os.remove('tmp.xyz')

    # Make a configuration
    system = make_atom_object(configuration_chemical_symbols,
                              configuration_coordinate,
                              cell)
    system.arrays['mass'] = system.get_masses()
    if is_output_lmp:
        write('model.lmp', system)
    else:
        write('model.xyz', system)
    return system
