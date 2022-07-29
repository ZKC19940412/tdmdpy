from ase.io import read, write
import numpy as np
from pylab import *
from tdmdpy import compute_all_rdfs
from tdmdpy import decompose_dump_xyz
from tdmdpy import get_quantity_averages
from tdmdpy import load_with_cell


def set_fig_properties(ax_list):
    tl = 6
    tw = 2
    tlm = 4

    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)


if __name__ == "__main__":
    
    #  Set up Figure Styles
    aw = 2
    fs = 16
    font = {'size': fs}
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', linewidth=aw)

    # Denote sample rate and time step in ps
    sample_rate = 1000
    time_step_in_ps = 1e-3  # equal to 1 fs

    # Load in temperature data from thermo.out
    data = np.loadtxt('thermo.out')

    # Extract temperature and create time series accordingly
    temperature = data[:, 0]
    time = sample_rate * time_step_in_ps * np.arange(0, len(temperature), 1)

    # Extract box dimension from thermo.out
    # length scale goes from angstrom to nm
    unit_cell_length_matrix = data[:, -3:] / 10.0
    unit_cell_angle_matrix = 90 * ones_like(unit_cell_length_matrix)

    # Print average temperature
    print('Average temperature in K:  %.3f' % get_quantity_averages(temperature))

    # Decompose dump.xyz
    decompose_dump_xyz('dump.xyz')

    # Inject Reference PDB file into the trajectory'
    pos_trajectory = load_with_cell('pos.xyz', unit_cell_length_matrix, unit_cell_angle_matrix,
                                    top='coord_liquid_water_1566_atoms.pdb')
    vel_trajectory = load_with_cell('vel.xyz', unit_cell_length_matrix, unit_cell_angle_matrix,
                                    top='coord_liquid_water_1566_atoms.pdb')

    # Compute RDFS
    rdfs = compute_all_rdfs(pos_trajectory, n_bins=400, is_save_rdf=False)

    # Save RDFS for each pair, multiple by 10 to go from nm to angstrom for length unit.
    np.savetxt('nep_goo_T_300K.out', np.vstack([rdfs['O-O'][0][:] * 10, rdfs['O-O'][1][:]]).T)
    np.savetxt('nep_goh_T_300K.out', np.vstack([rdfs['O-H'][0][:] * 10, rdfs['O-H'][1][:]]).T)
    np.savetxt('nep_ghh_T_300K.out', np.vstack([rdfs['H-H'][0][:] * 10, rdfs['H-H'][1][:]]).T)

    figure()
    set_fig_properties([gca()])
    plot(time, temperature)
    xlabel('Time (ps)')
    ylabel('Temperature (K)')
    savefig('Temperature_profile_NVT_NEP.png', dpi=300)
    show()
