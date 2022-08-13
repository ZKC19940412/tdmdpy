from ase import Atoms
from ase.io import read, write
from .atom_manipulate import decompose_dump_xyz, get_unique_atom_types, load_with_cell
import mdtraj as mdt
import numpy as np
import subprocess


def compute_all_rdfs(trj, n_bins=150, is_save_rdf=True, **kwargs):
    """Computes the RDFs between all pairs of atom types in a trajectory.
        https://www.mdtraj.org/1.9.8.dev0/api/generated/mdtraj.compute_rdf.html#mdtraj.compute_rdf

       input:
       trj: (mdtraj object) A trajectory from MD simulation
       n_bins: (int) Number of bins used for the RDF

       output:
       rdfs_all: (dictionary) RDFs for all paris

    """

    top = trj.topology

    rdfs_all = {}

    # Determine the unique atom types
    atom_types = get_unique_atom_types(trj.topology)

    for i1, t1 in enumerate(atom_types):

        # select indices of the first atom type
        idx_t1 = top.select('name ' + t1)

        # unique atom type pairs only
        for i2 in range(i1, len(atom_types)):

            t2 = atom_types[i2]

            # select indices of the second atom type
            idx_t2 = top.select('name ' + t2)

            # prepare all pairs of indices
            pairs = trj.topology.select_pairs(idx_t1, idx_t2)

            # single atom with itself -> no RDF
            if len(pairs) == 0:
                continue

            # OM: not sure this should be done here
            min_dimension = trj[0].unitcell_lengths.min() / 2
            # print(min_dimension)

            r, g_r = mdt.compute_rdf(
                trj, pairs,
                (0, min_dimension), n_bins=n_bins, **kwargs
            )

            rdfs_all[t1 + '-' + t2] = r, g_r
            if is_save_rdf:
                np.savetxt('g' + str(t1) + '-' + str(t2) + '.dat', np.vstack([r, g_r]).T)

    return rdfs_all


def get_quantity_averages(quantities, mode='diff'):
    """Get averages of quantity derived from MD simulation

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

        # 10 % of max change is used as the standard
        index = np.where(np.abs(
            rate_of_change_quantities) <= 0.01 * max_change)[0][0]

    return np.mean(quantities[index:])


def get_block_average_quantities(data_arr, n_block=10):
    """Compute blocked average quantities.
           input:
           data: (ndarray) Array that contains the data
           n_block: (int) Number of blocks


           output:
           block_average: (ndarray) Block-averaged quantities
    """

    # Denote array for block average quantities
    block_average_quantities = np.zeros([n_block])

    # Denote steps, interval
    steps = len(data_arr)
    k = 0
    s = 0
    interval = int(np.modf(steps / n_block)[1])
    while k < n_block:
        block_average_quantities[k] = np.mean(
            data_arr[s:s + interval])
        k += 1
        s += interval
    return block_average_quantities


def grep_from_md_output(md_output_file_name, time_step_in_ps, total_number_of_steps, patten_str, row_index_to_skip=0):
    """Grep several observables from snap md output

       input:
       md_output_file_name: (str) file name of the md output
       time_step_in_ps: (float) time step in ps
       total_number_of_steps: (int) total number of time steps

       output:
       data: (numpy array) data parsed from the md output

    """

    # Create key variable for the command line
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


def load_nth_frames(total_xyz_name, reference_chemical_symbols, frame_index=-1, frame_output_format='pdb'):
    """Load nth frame from the xyz output dumped from lammps simulation

       input:
       total_xyz_name: (str) name of the lammps dumped xyz file
       frame_index: (int) index to signify which frame to take from md
       reference_chemical_symbols: (nd str array) nd string array
       frame_output_format: (str) format of the output frames

       output:
       coordinate of nth frame in specific format

    """

    # Load in the whole configuration and extract specific frames
    configurations = read(total_xyz_name, index=':', format='lammps-dump-text')
    selected_frames = configurations[frame_index]

    # Fix misread issue, workable even it does not happen
    if type(frame_index) != int:
        for i in range(len(selected_frames)):
            selected_frames[i].set_chemical_symbols(reference_chemical_symbols)
            selected_frames[i].set_pbc([True, True, True])

    else:
        selected_frames.set_chemical_symbols(reference_chemical_symbols)
        selected_frames.set_pbc([True, True, True])
    if frame_output_format == '.lmp':
        write('extracted_frames' + frame_output_format, selected_frames, format='lammps-data')
    else:
        write('extracted_frames' + frame_output_format, selected_frames)


def process_diffusion_coefficients(sdc_out_str, dimension_factor=3, is_verbose=True):
    """Process self-diffusion coefficients computed from GPUMD

       input:
       sdc_out_str: (str) name of the sdc file
       dimension factor: (int) dimension factor, 3 for all 3 dimensions
       is_verbose: (bool) whether to print out fitted diffusion coefficients

       output:
       D: (float) Diffusion coefficients in 10^-5 cm^2/s
    """
    # Load in data from sdc.out
    data = np.loadtxt(sdc_out_str)

    # Extract correlation time and self-diffusion coefficients along each direction
    correlation_time = data[:, 0]
    D_x = data[:, 4]
    D_y = data[:, 5]
    D_z = data[:, 6]

    # Obtain average along each direction
    # 10.0 makes it goes from A^2/ps to 10^-5 cm^2/s
    D = 10.0 * (get_quantity_averages(D_x) + get_quantity_averages(D_y) + get_quantity_averages(D_z)) / dimension_factor

    # Display results in original form:
    # 10.0 makes it goes from A^2/ps to 10^-5 cm^2/s
    if is_verbose:
        print('Diffusion coefficients in 10^-5 cm^2/s: %.1f' % D)
    return D, D_x, D_y, D_z, correlation_time


def score_property(prediction, ground_truth, tolerance, property_str):
    """Score static property of water using the score function
       from Carlos Vega et al.
       DOI: 10.1039/c1cp22168j

       input:
       prediction: (float) predicted property from snap md
       ground_truth: (float) experimental properties
       tolerance: (float) tolerance factor in %
       property_str: (str) name of predicted property

       output:
       final score: (int) final score of the property,

    """
    base_score = np.round(10 - np.abs(100 * (
            prediction - ground_truth) / (ground_truth * tolerance)))
    final_score = np.max([base_score, 0])
    print('Predicted ' + property_str + ' earned a score of %d' % final_score)

    return final_score
