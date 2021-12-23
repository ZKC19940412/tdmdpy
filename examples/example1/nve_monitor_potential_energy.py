from tdmdpy import grep_from_md_output
from tdmdpy import get_quantity_averages
from pylab import *

aw = 2
fs = 25
font = {'size': fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes', linewidth=aw)


def set_fig_properties(ax_list):
    tl = 6
    tw = 2
    tlm = 4

    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both',
                       direction='in', right=True, top=True)


# -----------------------------------------------------------------------------#

# Define basic constants during the runs
# ensemble_log_name : (str) name of the log files
# sample_rate: (int) frequency to compute thermo
# time_step: (float) time step used in MD simulation
# sample rate, time step and total number of steps should be accessible
# from lammps.inp file
nve_ensemble_log_name = 'log_ff_nve.lammps'
sampling_rate = 100
time_step = 1e-3
total_number_of_steps = 1000000

# Define the pattern string for the parser 
# usually the strings before first line of 
# data should do.
pattern = '  Time '

# Grep data from md output
# row_index_to_skip = 1 is to skip 
# the grabbed string before the data
data = grep_from_md_output(nve_ensemble_log_name, time_step,
                           total_number_of_steps,
                           pattern, row_index_to_skip=1)

# Extract useful quantities from data
time_in_ps = data[:, 0]
potential_energy_in_eV = data[:, 2]

average_E_potent_in_eV_all_mode = get_quantity_averages(potential_energy_in_eV,
                                                        mode='all')

average_E_potent_in_eV_diff_mode = get_quantity_averages(potential_energy_in_eV,
                                                         mode='diff')

# Print average potential energy
print('Average potential energy from all mode: %.3f eV '
      % average_E_potent_in_eV_all_mode)
print('Average potential energy from diff mode: %.3f eV '
      % average_E_potent_in_eV_diff_mode)
print('\n')

# Make plots to trace the convergence
figure(figsize=(12, 10))
subplot()
set_fig_properties([gca()])
plot(time_in_ps, potential_energy_in_eV, lw=2, label=r'$E_{potential}$')
plot(time_in_ps,
     average_E_potent_in_eV_all_mode * np.ones_like(potential_energy_in_eV),
     'r--', lw=3, label=r'$E_{potential, average,all}$')
plot(time_in_ps,
     average_E_potent_in_eV_diff_mode * np.ones_like(potential_energy_in_eV),
     'k--', lw=3, label=r'$E_{potential, average,diff}$')
xlabel('Time (ps)')
ylabel(r'E (eV)')
legend(loc=1)
show()
