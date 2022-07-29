# Import statements
from ase import Atoms
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.constraints import FixBondLengths
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units
import numpy as np

if __name__ == "__main__":

    # Set up water box
    x = angleHOH * np.pi / 180 / 2
    pos = [[0, 0, 0],
        [0, rOH * np.cos(x), rOH * np.sin(x)],
        [0, rOH * np.cos(x), -rOH * np.sin(x)]]
    atoms = Atoms('OH2', positions=pos)

    # Denote constants
    molar_mass_H2O = 18.01528  # g/mol
    avogadro_constant = 6.022140857e23  # mol^-1
    density_target = 0.9982  # g/cm^3

    # Back solve box length such that density will stay at 0.9982 g/cm^3 , assumed cubic box
    box_length = ((molar_mass_H2O / avogadro_constant) / (density_target / 1e24))**(1 / 3.)
    atoms.set_cell((box_length, box_length, box_length))
    atoms.center()

    # Replicate atoms 3 times in each direction, a total of 27 atoms
    nrep = 3
    atoms = atoms.repeat((nrep, nrep, nrep))
    atoms.set_pbc(True)

    # RATTLE-type constraints on O-H1, O-H2, H1-H2.
    atoms.constraints = FixBondLengths([(3 * i + j, 3 * i + (j + 1) % 3)
                                    for i in range(3**3)
                                    for j in [0, 1, 2]])

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    tag = 'tip3p_27_molcues_equilibrate'
    atoms.calc = TIP3P(rc=4.5)
    md = Langevin(atoms, 1 * units.fs, temperature_K=300 * units.kB,
                  friction=0.01, logfile=tag + '.log')

    trajectory = Trajectory(tag + '.traj', 'w', atoms)
    md.attach(trajectory.write, interval=1)

    # Run MD simulation for 2 ps
    md.run(2000)

    print('Initial water structure equilibrated with TIP3P is obtained!')
