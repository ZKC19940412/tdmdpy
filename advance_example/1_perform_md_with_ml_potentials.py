from ase.calculators.lammpslib import LAMMPSlib
from ase.io import Trajectory
from pynep.calculate import NEP
from tdmdpy.mlps import md_simulator
import time

if __name__ == "__main__":

    # Read in trajectory from TIP3P simulation
    #  last frame is the equilibrated one
    traj = Trajectory('tip3p_27_molcues_equilibrate.traj')
    atoms = traj[-1]

    # Instruct whether to run tip4p reference simulation
    is_run_tip4p_simulation = False
    if is_run_tip4p_simulation:
        tic = time.perf_counter()
        nvt_simulation_tip4p = md_simulator(atoms, 'NVT', 'berendsen',
                                            tag='tip4p_27_molecules_nvt_production',
                                            calculator=None, T_initial=300,
                                            T_final=300, T_tau=100,
                                            time_step_val=1, total_step=1000)
        toc = time.perf_counter()
        print(f"NVT simulation for 1 ps using TIP4P potential takes {toc - tic:0.4f} seconds real time.")

    # Define NEP calculator
    nep_calculator = NEP('nep.txt')
     
    # Conduct an NVT simulation with berendsen thermostat
    tic = time.perf_counter()
    nvt_simulation_nep = md_simulator(atoms, 'NVT', 'berendsen',
                                       tag='nep_27_molcues_nvt_production',
                                       calculator=nep_calculator,
                                       T_initial=300, T_final=300, T_tau=100,
                                       time_step_val=1, total_step=10000)
    toc = time.perf_counter()
    print(f"NVT simulation for 10 ps using NEP potential takes {toc - tic:0.4f} seconds real time.")
