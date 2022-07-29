from ase.io import read, write
import mdtraj as mdt
import numpy as np
from pylab import *
from tdmdpy import compute_hydrodynamic_radius
from tdmdpy import decompose_dump_xyz
from tdmdpy import get_quantity_averages
from tdmdpy import load_with_cell
from tdmdpy import process_diffusion_coefficients
from tdmdpy import score_property


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
    print(compute_hydrodynamic_radius(pos_trajectory))
    #exit()
    # Compute mass density
    mass_density = mdt.density(pos_trajectory)
    print('Average density in g/cm^3:  %.3f' % get_quantity_averages(mass_density / 1000.0))
    score_property(get_quantity_averages(mass_density / 1000.0), 0.997, 0.5, property_str='density at 298K, 0 bar')

    # Process diffusion coefficients from sdc.out
    D, D_x, D_y, D_z, correlation_time = process_diffusion_coefficients('sdc.out', dimension_factor=3, is_verbose=True)
    score_property(np.log(D * 1e-5), -10.68, 0.5, property_str='diffusion coefficient')
    print('lnD: %.2f' % np.log(D * 1e-5))

    figure()
    set_fig_properties([gca()])
    plot(time, temperature)
    xlabel('Time (ps)')
    ylabel('Temperature (K)')
    savefig('Temperature_profile_NPT.png', dpi=300)
    show()

    figure()
    set_fig_properties([gca()])
    plot(time, mass_density / 1000.0)
    xlabel('Time (ps)')
    ylabel(r'$\rho (g/cm^{3}$')
    savefig('Density_profile_NPT_NEP.png', dpi=300)
    show()

    figure()
    set_fig_properties([gca()])
    plot(correlation_time, D_x, label=r'$D_{x}$')
    plot(correlation_time, D_y, label=r'$D_{y}$')
    plot(correlation_time, D_z, label=r'$D_{z}$')
    xlabel('Correlation time (ps)')
    ylabel(r'$D (\AA^{2}/ps)$')
    legend(loc='best')
    savefig('Diffusion_coefficients_NPT_NEP.png', dpi=300)
    show()
