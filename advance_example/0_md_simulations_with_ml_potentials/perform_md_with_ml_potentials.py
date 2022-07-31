from ase.io import read
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from pynep.calculate import NEP
import time

if __name__ == "__main__":

    # Read in liquid water structure with 64 molecules
    atoms = read('coord_liquid_water_64_molecules.pdb')

    # Define NEP calculator
    nep_calculator = NEP('nep.txt')

    # Allocate calculator into atom object
    atoms.calc = nep_calculator

    # Set the momenta corresponding to T=300K
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # We want to run MD with constant energy using the Velocity Verlet algorithm.
    dyn = VelocityVerlet(atoms, 1 * units.fs, logfile='nve_with_nep.log')
    traj = Trajectory('nve_with_nep' + '.traj', 'w', atoms)
    dyn.attach(traj.write, interval=1)
    tic = time.perf_counter()
    dyn.run(10000)
    toc = time.perf_counter()
    print(f"NVE simulation for 10 ps with NEP potential takes {toc - tic:0.4f} seconds real time.")
