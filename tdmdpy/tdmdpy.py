from ase import Atoms
from ase.io import read, write
import mdtraj as mdt
import numpy as np
from scipy.stats import linregress
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


def decompose_dump_xyz(dump_xyz_str, pos_xyz_str='pos.xyz',
                       frc_xyz_str='frc.xyz',
                       vel_xyz_str='vel.xyz'):
    """Decompose information from dump.xyz, generated by GPUMD.
       https://gpumd.zheyongfan.org/index.php/The_dump_exyz_keyword

       input:
       dump_xyz_str: (str) Name of the dump xyz file
       pos_xyz_str: (str) Name of the pos xyz file
       pos_xyz_str: (str) Name of the force/frc xyz file
       vel_xyz_str: (str) Name of the  velocity/vel xyz file

    """
    # Load in static information
    configurations = read(dump_xyz_str, index=':')
    chemical_symbols = configurations[0].get_chemical_symbols()

    # Preset list for sudo atom objects for positions, forces and velocities
    pos_atom_objects = []
    frc_atom_objects = []
    vel_atom_objects = []

    # Decompose position, force and velocity information into lists individually
    for configuration in configurations:
        pos_atom_objects.append(make_atom_object(chemical_symbols, configuration.positions, configuration.cell))
        frc_atom_objects.append(make_atom_object(chemical_symbols, configuration.arrays['forces'], configuration.cell))
        vel_atom_objects.append(make_atom_object(chemical_symbols, configuration.arrays['vel'], configuration.cell))

    # Write individual information into xyz file
    write(pos_xyz_str, pos_atom_objects)
    write(frc_xyz_str, frc_atom_objects)
    write(vel_xyz_str, vel_atom_objects)

def get_unique_atom_types(topology):
    """Obtain unique atom types from a reference topology.

       input:
       topology: (pdb) Reference topology, often in a format of pdb

       output:
       atom_types: (list) List of unique atom type strings
    """
    atom_types = list(set(atom.name for atom in topology.atoms))

    return atom_types


# def get_diffusion_coefficients(MSD,
#                                sample_rate,
#                                time_step,
#                                msd_type_str='xyz',
#                                is_fft=True,
#                                start_index=None,
#                                skip_index=None,
#                                end_index=None,
#                                is_return_fit_para=False,
#                                is_verbose=False):
#     """get mean square displacement (MSD) and self-diffusion coefficients (D)
#          from MD simulation: https://docs.mdanalysis.org/stable/documentation_pages/analysis/msd.html
#
#          input:
#          MSD: (MSD MDAnalysis object) Object to config MSD calculations
#          time_step: (float) time step, usually in ps
#          start_index (int): starting index to compute MSD
#          skip_index (int): frames to skip while computing MSD
#          end_index (int): ending index to compute MSD
#          is_return_fit_paraï¼š(bool) whether to return linear fitting parameter to obtin diffusion coefficients
#          is_verbose: (bool) whether to print out fitted diffusion coefficients
#
#          output:
#          D (float): diffusion coefficients in 10^-5 cm/s
#          lagtimes (ndarray): Time-axis for MSD
#          msd (ndarray): MSD
#     """
#
#     # Perform calculation of MSD from the input object
#     MSD.run(start_index, end_index, skip_index)
#
#     # Declare total time step
#     time_step_total = time_step * sample_rate
#
#     # Exact msd and lag times from MSD object
#     msd = MSD.results.timeseries
#     nframes = MSD.n_frames
#     lagtimes = np.arange(nframes) * time_step_total
#
#     # Define linear model and extract D from slope
#     linear_model = linregress(lagtimes, msd)
#     slope = linear_model.slope
#
#     # dim_fac is 3 as we computed a 3D msd with 'xyz'
#     D = slope * 1 / (2 * MSD.dim_fac) / time_step_total
#     if is_verbose:
#         print('Diffusion coefficients in 10^-5 cm/s: %.1f' % D)
#
#     if is_return_fit_para:
#         return D, linear_model, lagtimes, msd
#     else:
#         return D, lagtimes, msd


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


def load_with_cell(filename, reference_topo, start=None, stop=None, step=None, **kwargs):
    """Load a trajectory and inject cell dimensions from a topology PDB file if not present.
    All arguments and keyword arguments are passed on to `mdtraj.load`. The `top`
    keyword argument is used load a PDB file and get cell information from it.

    input:
    file_name (str):

    """

    # load the "topology frame" to get cell dimensions
    #top = kwargs.get("top")
    #if top is not None and isinstance(top, str):
        # load topology frame - just the first one from the file, in case there are more frames
    #    frame_top = mdt.load_frame(top, 0)
    #    unitcell_lengths = frame_top.unitcell_lengths
    #    unitcell_angles = frame_top.unitcell_angles
    #    if (unitcell_lengths is None) or (unitcell_angles is None):
    #        raise ValueError("Frame providing topology is missing cell information.")
    #else:
    #    raise ValueError("Provide a PDB with cell dimensions.")

    # load the trajectory itself
    #trj = mdt.load(filename, **kwargs)
    #trj = trj[start:stop:step]
    #trj_raw_xyz_in_angstrom = trj.xyz.copy()
    # inject the cell information
    #len_trj = len(trj)
    #trj.unitcell_lengths = unitcell_lengths.repeat(len_trj, axis=0)
    #trj.unitcell_angles = unitcell_angles.repeat(len_trj, axis=0)
    
    # Derive length of trajectory
    len_trj = len(md.load(filename, **kwargs)[start:stop:step])
    
    # Load trj frame by frame so as to add proper unit cell
    for i in range(len_trj):
        frame_reference_topo = mdt.load_frame(reference_topo, i)
        trj_tmp = md.load_frame(filename, i)
        trj_tmp.unitcell_lengths = frame_reference_topo.unitcell_lengths
        trj_tmp.unitcell_angles = frame_reference_topo.unitcell_angles
        trj = mdt.join(trj_tmp)
    return trj


def load_nth_frames(total_xyz_name, reference_chemical_symbols, frame_index=-1, frame_output_format='pdb'):
    """Load nth frame from the xyz output dumped from lammps simulation

       input:
       total_xyz_name: (str) name of the lammps dumped xyz file
       frame_index: (int) index to siginfy which frame to take from md
       reference_chemical_symbols: (nd str array) nd string array
       frame_ouput_format: (str) format of the output frames

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


def make_atom_object(atomic_string, coordinate,cell):
    """Create simple atom object based on chemical symbol and "position-like" vectors

           input:
           atomic_string: (nd str array) nd string array
           coordinate: (nd float array) "position-like" vector

           output:
           atoms (ASE Atoms object): atom object

    """
    # Build ase atom object from starch
    tmp_str = 'H' * len(atomic_string)

    # Ghost molecule!
    atoms = Atoms(tmp_str)

    # Inheritance properties from input
    atoms.set_chemical_symbols(atomic_string)
    atoms.set_positions(coordinate)
    atoms.set_pbc([True, True, True])
    atoms.cell = cell
    return atoms


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
