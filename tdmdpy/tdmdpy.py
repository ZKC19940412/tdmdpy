from ase.io import write
import numpy as np
import subprocess

def get_quantity_averages(quantities, mode='all'):
    """get averages of quantity derived from MD simulation
       input:
       quantities: (ndarray) quantity derived from MD simulation
       mode: (str) different modes to take averages

       output:
       average: (float) average of quantities
    """

    if mode not in ['all', 'diff']:
        print('Only "all" or "diff" are valid mode strings!')
        exit()

    if mode == 'all':
        return np.mean(quantities)

    if mode == 'diff':
        # Find where to perform average by using derivatives
        rate_of_change_quantities = np.diff(quantities)
        max_change = np.abs(rate_of_change_quantities).max()

        # 10 % of max change as bar is used as the standard
        index = np.where(np.abs(rate_of_change_quantities) <= 0.01 * max_change)[0][0]

    return np.mean(quantities[index:])

def grep_from_md_output(md_output_file_name, time_step_in_ps, total_number_of_steps, patten_str, row_index_to_skip=0):
    """grep several observables from snap md output

       input:
       md_output_file_name: (str) file name of the md output
       time_step_in_ps: (float) time step in ps
       total_number_of_steps: (int) total number of time steps

       output:
       data: (numpy array) data parsed from the md output

    """

    # create key variable for the command line
    # + 1 for the header
    total_number_of_lines_to_grep = int(time_step_in_ps * total_number_of_steps + 1)

    # Generate a cmd pattern
    cmd = "grep -A " + str(total_number_of_lines_to_grep) + "pattern " + md_output_file_name + ">" + "tmp.out"
    cmd_full = cmd.replace("pattern", patten_str)
    subprocess.call(cmd_full, shell=True)

    try:
        data = np.loadtxt('tmp.out', skiprows=row_index_to_skip)
    except (ValueError, FileNotFoundError, IndexError):
        print('Data can not be properly loaded! Check file manually for more details.')

    return data

def map_steps_with_simulation_time(time_step=5e-4, total_number_of_steps=20000):
    """map between steps (0 ~ about 200000) and simulation time (0 ~ 100 ps)
       input:
       time_step: (float) time step in fs

       total_number_of_steps: (int) total number of steps

       output:
       time_span: (ndarray) time span from 0 ~ 100 ps, with 5e-4 ps

    """
    # Derive total time
    total_time = time_step * total_number_of_steps

    # Generate time span
    time_span = np.linspace(0, total_time, total_number_of_steps)

    return time_span

def merge_all_xyz_into_one(xyz_folder):
    """merge all xyz files in one folder to a single extended xyz file
    """
    xyz_files = sorted(os.listdir(xyz_folder))
    for xyz_file in xyz_files:
        xyz = read(xyz_folder + '/' + xyz_file)
        write('out.xyz', xyz, append=True)
