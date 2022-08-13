from ase.io import read, write
import mdtraj as mdt
import numpy as np
from pylab import *
from tdmdpy.atom_manipulate import decompose_dump_xyz
from tdmdpy.atom_manipulate import load_with_cell
from tdmdpy.thermodynamic_properties import get_quantity_averages
from tdmdpy.thermodynamic_properties import process_diffusion_coefficients
from tdmdpy.thermodynamic_properties import score_property


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
    sample_rate = 100
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

    # Compute mass density
    mass_density = mdt.density(pos_trajectory)
    print('Average density in g/cm^3:  %.3f' % get_quantity_averages(mass_density / 1000.0))
    score_property(get_quantity_averages(mass_density / 1000.0), 0.997, 0.5, property_str='density at STP')
    print('Block-averaged density in g/cm^3 : %.3f' % get_quantity_averages(mass_density / 1000.0, mode='block').mean())
    print('Stand deviation of density in g/cm^3: %.3f' % get_quantity_averages(
        mass_density / 1000.0, mode='block').std())

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
