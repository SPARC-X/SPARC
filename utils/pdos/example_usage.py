
from calculate_pdos import PDOSCalculator

# Example system: Al FCC
upf_fname_list = [
    './examples/Al_FCC/Al.upf',  # Optional UPF file
    # If no UPF files provided, program will use built-in generator
]

output_fname = './examples/Al_FCC/Al.out'
static_fname = './examples/Al_FCC/Al.static'
eigen_fname  = './examples/Al_FCC/Al.eigen'
psi_fname    = './examples/Al_FCC/Al.psi'
out_dirname  = './examples/Al_FCC/PDOS_output'

# Initialize PDOSCalculator
pdos_calculator = PDOSCalculator(
    upf_fname_list = upf_fname_list,  # Optional: can be None for automatic generation
    output_fname = output_fname,
    eigen_fname = eigen_fname,
    static_fname = static_fname,
    psi_fname = psi_fname,
    out_dirname = out_dirname,
    r_cut_max = 15.0,  # Optional: cutoff radius in Bohr
    atomic_wave_function_tol = 1e-5,  # Optional: tolerance for atomic wave functions
    orthogonalize_atomic_orbitals = False,  # Optional: whether to orthogonalize orbitals
    is_relaxation = False,  # Optional: whether from relaxation calculation
    k_point_parallelization = True,  # Optional: enable k-point parallelization
)

# Run the PDOS calculation
E, PDOS, DOS = \
    pdos_calculator.run(
        mu_PDOS = 0.2721140795,  # Gaussian broadening width in eV
        N_PDOS = 1000,  # Number of energy points
        min_E_plot = 5.0,  # Minimum energy in eV (optional, auto-determined if None)
        max_E_plot = 65.0,  # Maximum energy in eV (optional, auto-determined if None)
        sum_over_m_index = False,  # Whether to sum over magnetic quantum number m
        print_projection_data = False,  # Whether to save projection data
    )

# Some downstream analysis
print("E.shape    = ", E.shape)    # (N_PDOS,)
print("DOS.shape  = ", DOS.shape)  # (N_PDOS,)
print("PDOS.shape = ", PDOS.shape) # (N_PDOS, number_of_orbitals)


# For selective atom calculation
E_single_atom, PDOS_single_atom, DOS_single_atom = \
    pdos_calculator.run_single_atom(
        atom_type = "Al",  # or atomic number 13
        atom_index_for_specified_atom_type = 0,  # 0-indexed
        mu_PDOS = 0.2721140795,
        N_PDOS = 1000,
        min_E_plot = 5.0,
        max_E_plot = 65.0,
        sum_over_m_index = False,
    )

# Some downstream analysis
print("E_single_atom.shape    = ", E_single_atom.shape)    # (N_PDOS,)
print("DOS_single_atom.shape  = ", DOS_single_atom.shape)  # (N_PDOS,)
print("PDOS_single_atom.shape = ", PDOS_single_atom.shape) # (N_PDOS, number_of_orbitals_for_single_atom)