from ase.io import read, write
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import numpy as np
from scipy.stats import linregress
import subprocess


def get_diffusion_coefficients(MSD,
                               sample_rate,
                               time_step,
                               msd_type_str='xyz',
                               is_fft=True,
                               start_index=None,
                               skip_index=None,
                               end_index=None,
                               is_return_fit_para=False,
                               is_verbose=False):
    """get mean square displacement (MSD) and self-diffusion coefficients (D)
         from MD simulation: https://docs.mdanalysis.org/stable/documentation_pages/analysis/msd.html

         input:
         MSD: (MSD MDAnalysis object) Object to config MSD calculations
         time_step: (float) time step, usually in ps
         start_index (int): starting index to compute MSD
         skip_index (int): frames to skip while computing MSD
         end_index (int): ending index to compute MSD
         is_return_fit_paraï¼š(bool) whether to return linear fitting parameter to obtin diffusion coefficients
         is_verbose: (bool) whether to print out fitted diffusion coefficients

         output:
         D (float): diffusion coefficients in 10^-5 cm/s
         lagtimes (ndarray): Time-axis for MSD
         msd (ndarray): MSD
    """

    # Perform calculation of MSD from the input object
    MSD.run(start_index, end_index, skip_index)

    # Declare total time step
    time_step_total = time_step * sample_rate

    # Exact msd and lag times from MSD object
    msd = MSD.results.timeseries
    nframes = MSD.n_frames
    lagtimes = np.arange(nframes) * time_step_total

    # Define linear model and extract D from slope
    linear_model = linregress(lagtimes, msd)
    slope = linear_model.slope

    # dim_fac is 3 as we computed a 3D msd with 'xyz'
    D = slope * 1 / (2 * MSD.dim_fac) / time_step_total
    if is_verbose:
        print('Diffusion coefficients in 10^-5 cm/s: %.1f' % D)

    if is_return_fit_para:
        return D, linear_model, lagtimes, msd
    else:
        return D, lagtimes, msd


def get_quantity_averages(quantities, mode='diff'):
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

        # 10 % of max change is used as the standard
        index = np.where(np.abs(
            rate_of_change_quantities) <= 0.01 * max_change)[0][0]

    return np.mean(quantities[index:])


def get_radial_distribution_function(typology_file, dcd_traj=None,
                                     chemical_symbol_atom1='H',
                                     chemical_symbol_atom2='O',
                                     nbin=400,
                                     cut_off=12,
                                     start_index=None,
                                     skip_index=None,
                                     end_index=None,
                                     is_save_rdf=True):
    """get radial distribution function (RDF) from MD simulation
       https://docs.mdanalysis.org/1.0.0//documentation_pages/analysis/rdf.html

       input:
       typology_file: (xyz or pdb) initial configuration of the system
       dcd_traj: (dcd) trajectory in dcd format
       nbin (int): number of bins for RDFs
       cut_off (float): cut off radius for RFS
       start_index (int): starting index to compute RDFS
       skip_index (int):  frames to skip while computing RDFS
       end_index (int):   ending index to compute RDFS
       is_save_rdf (bool): whether to save resultant rdf into .dat file

       output:
       g(RDF objects): RDF objects

       plotting usage:
       plot(g.results.bins, g.results.rdf)
    """

    # Create universe object
    if not dcd_traj:
      u = mda.Universe(typology_file, dcd_traj)
    else:
      u = mda.Universe(typology_file)

    # Get atom groups
    atom_group1 = u.select_atoms('name ' + chemical_symbol_atom1)
    atom_group2 = u.select_atoms('name ' + chemical_symbol_atom2)

    # Define individual rdf objects and perform calculations
    g11 = InterRDF(atom_group1, atom_group1, nbin, range=(0, cut_off))
    g11.run(start_index, end_index, skip_index)

    g22 = InterRDF(atom_group2, atom_group2, nbin, range=(0, cut_off))
    g22.run(start_index, end_index, skip_index)

    g12 = InterRDF(atom_group1, atom_group2, nbin, range=(0, cut_off))
    g12.run(start_index, end_index, skip_index)

    if is_save_rdf:
        np.savetxt('g11.dat', np.vstack([g11.results.bins, g11.results.rdf]).T)
        np.savetxt('g22.dat', np.vstack([g22.results.bins, g22.results.rdf]).T)
        np.savetxt('g12.dat', np.vstack([g12.results.bins, g12.results.rdf]).T)
    return g11, g22, g12


def grep_from_md_output(md_output_file_name, time_step_in_ps, total_number_of_steps, patten_str, row_index_to_skip=0):
    """grep several observables from snap md output

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
    """load nth frame from the xyz output dumped from lammps simulation

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
        write('extracted_frames' + frame_output_format, selected_frames, format = 'lammps-data')
    else:
        write('extracted_frames' + frame_output_format, selected_frames)


def score_property(prediction, ground_truth, tolerance, property_str):
    """score static property of water using the score function
       from Carlos Vega et al.
       DOI: 10.1039/c1cp22168j

       input:
       prediction: (float) predicted property from snap md
       ground_truth: (float) experimental properties
       tolerance: (float) tolerance factor in %
       property_str (str) name of predicted property

       output:
       final score (int) final score of the property,

    """
    base_score = np.round(10 - np.abs(100 * (
            prediction - ground_truth) / (ground_truth * tolerance)))
    final_score = np.max([base_score, 0])
    print('Predicted ' + property_str + ' earned a score of %d' % final_score)

    return final_score
