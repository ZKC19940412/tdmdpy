from ase import Atoms
from ase.io import read, write
import mdtraj as mdt

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
    write('ini_pos.pdb', pos_atom_objects[0])
    write('ini_vel.pdb', vel_atom_objects[0])


def get_unique_atom_types(topology):
    """Obtain unique atom types from a reference topology.

       input:
       topology: (pdb) Reference topology, often in a format of pdb

       output:
       atom_types: (list) List of unique atom type strings
    """
    atom_types = list(set(atom.name for atom in topology.atoms))

    return atom_types


def load_with_cell(filename, unitcell_length_matrix, unitcell_angle_matrix,
                   start=None, stop=None, step=None, **kwargs):
    """Load a trajectory and inject cell dimensions from a topology PDB file if not present.
    All arguments and keyword arguments are passed on to `mdtraj.load`. The `top`
    keyword argument is used load a PDB file and get cell information from it.

    input:
    file_name (str):

    """
    # load the "topology frame" to get cell dimensions
    top = kwargs.get("top")
    if top is not None and isinstance(top, str):
        # load topology frame - just the first one from the file, in case there are more frames
        frame_top = mdt.load_frame(top, 0)
        unitcell_lengths = frame_top.unitcell_lengths
        unitcell_angles = frame_top.unitcell_angles
        if (unitcell_lengths is None) or (unitcell_angles is None):
            raise ValueError("Frame providing topology is missing cell information.")
    else:
        raise ValueError("Provide a PDB with cell dimensions.")

    # load the trajectory itself
    trj = mdt.load(filename, **kwargs)
    trj = trj[start:stop:step]

    # inject the cell information
    trj.unitcell_lengths = unitcell_length_matrix
    trj.unitcell_angles = unitcell_angle_matrix

    return trj


def load_nth_frames(total_xyz_name, reference_chemical_symbols,
                    frame_index=-1, frame_output_format='pdb'):
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


def make_atom_object(atomic_string, coordinate, cell):
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


def wrap_coordinates(pos_xyz_str='pos.xyz'):
    """Wrap coordinates back to periodic box

               input:
               pos_xyz_str (str): name of the pos.xyz file

               output:
               single_atom_object (ASE Atoms object): atom object with wrapped coordinate

        """

    # Read in pos.xyz
    single_atom_object = read(pos_xyz_str)

    # Derive cell from atom object and along each direction
    cell = single_atom_object.cell
    cell_x = cell[0][0]
    cell_y = cell[1][1]
    cell_z = cell[2][2]

    # Obtain a copy of positions
    positions = single_atom_object.positions.copy()

    # Process positions based on wrapping rule
    for i in range(positions.shape[0]):
        while positions[i, 0] < 0:
            positions[i, 0] += cell_x
        while positions[i, 0] >= cell_x:
            positions[i, 0] -= cell_x

        while positions[i, 1] < 0:
            positions[i, 1] += cell_y
        while positions[i, 1] >= cell_y:
            positions[i, 1] -= cell_y

        while positions[i, 2] < 0:
            positions[i, 2] += cell_z
        while positions[i, 2] >= cell_z:
            positions[i, 2] -= cell_z

    # Set wrapped positions back to the object
    single_atom_object.positions = positions

    return single_atom_object
