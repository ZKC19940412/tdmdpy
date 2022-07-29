from ase.calculators.tip4p import TIP4P
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase import units


def md_simulator(atoms, ensemble_str, thermostat_str,
                 tag=None, calculator=None, rc_val=4.5,
                 T_initial=None, T_final=None, T_tau=None,
                 P_final=None, P_tau=None,
                 time_step_val=1, total_step=1000):

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
