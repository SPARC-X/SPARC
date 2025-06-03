/**
 * @file    isddft.h
 * @brief   This file contains the structure definitions for SPARC
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Alfredo Metere (GPU Support), Lawrence Livermore National Laboratory, <metere1@llnl.gov>, <alfredo.metere@xsilico.com>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#ifndef ISDDFT_H
#define ISDDFT_H


#include <mpi.h>
#include <complex.h>
#include <stdio.h>
#include "dssq.h"
// #include "exactExchange.h"
#include "mlff_types.h"

// max length of pseudopotential path
#define L_PSD 4096
// max length of input strings
#define L_STRING 512
#define L_ATMTYPE 8
#define L_QMASS  60
// max length of kpoint set
#define L_kpoint 512

// TO PRINT IN COLOR
#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"



/* @brief   Physical constants.
 * @ref     NIST (https://physics.nist.gov/cuu/Constants/index.html)
 */
// Bohr radius in Angstrom (1e-10 m)
#define CONST_BOHR 0.529177210903

// convert atomic mass unit to atomic unit (electonic mass)  
// 1 au = 9.10938356e-31 Kg; 1 amu =  1.660539040e-27 Kg;
#define CONST_AMU2AU 1822.888485332371

// convert femtosecond to atomic time unit  
// 1atu = 2.418884326509e-17 s;
#define CONST_FS2ATU 41.34137333649300

// Boltzmann constant in Ha/K
#define CONST_KB 3.1668115634556e-6

// Hartree energy in eV
#define CONST_EH 27.211386245988

// Ha/Bohr^3 in GPa (converted by Ha->N*m, Bohr->m)
#define CONST_HA_BOHR3_GPA 29421.01569650548

// amu/Bohr^3 in g/cc
#define CONST_AMU_BOHR3_GCC 11.2058730627683


typedef struct _KRON_LAP {
    int Nx;
    int Ny;
    int Nz;
    int Nd;
    double *Vx;
    double *Vy;
    double *Vz;
    double _Complex *Vx_kpt;
    double _Complex *Vy_kpt;
    double _Complex *Vz_kpt;
    double _Complex *VyH_kpt;
    double _Complex *VzH_kpt;
    double *eig;
    double *inv_eig;
    int isGammaPoint;
} KRON_LAP;


typedef struct _MPEXP_OBJ {
    double *r_pos_r;    
    double *r_pos_r_aug[6];    
    double **Ylm;
    double **Ylm_aug[6];    
    int nd_cor[6];    
    int nd_phi[6];
    int nd_phi_max;
} MPEXP_OBJ;

typedef struct _D2D_OBJ {
    int n_target; // number of target processes to communicate with
    int *target_ranks; // target ranks in union communicator
    //int *target_coords; // target coords in target communicator
} D2D_OBJ; 

typedef struct _D2DEXT_OBJ {
    int layers[6];
    int *x_counts;
    int *y_counts;
    int *z_counts;
    int *counts;
    int *displs; 
    int n;
} D2Dext_OBJ; 


/**
 * @brief   This structure type is for storing pseudopotential variables.
 */
typedef struct _PSD_OBJ {
    double *rVloc;     // stores local part of pseudopotential times radius
    double *UdV;        // KB nonlocal SC projectors
    double *rhoIsoAtom;       // stores isolated atom electron density
    double *RadialGrid;   // stores the radial grid
    double *SplinerVlocD;  // stores derivative of rVloc from Spline
    double *SplineFitUdV; // derivative of UdV from Spline
    double *SplineFitIsoAtomDen;
    double *SplineRhocD; // derivative of rho_c_table for spline
    double *rc;     // component pseudopotential cutoff
    double *Gamma;  // KB SC energy for each channel
    double *rho_c_table;  // model core charge for nonlinear core correction
    double fchrg;   // fchrg value for nonlinear core correction
    double Vloc_0;  // stores Vloc(r = 0)
    int is_r_uniform; // flag to check if RadialGrid is uniform
    int pspxc;      // ixc, 2 - LDA_PZ, 7 - LDA_PW, 11 - GGA_PBE
    int lmax;       // maximum pseudopotential component
    int size;       // size of the arrays storing the pseudopotentials   
    int *ppl;       // number of nonlocal projectors per l
    int pspsoc;     // Flag for whether the psp has spin-orbit coupling (SOC) for each type of atom
    int *ppl_soc;   // number of nonlocal projectors per l for SOC
    double *Gamma_soc;  // KB SO energy for each channel
    double *UdV_soc;    // KB nonlocal SO projectors
    double *SplineFitUdV_soc;   // derivative of UdV_soc from Spline
} PSD_OBJ;



/**
* @brief This structure type is for storing atom solution variables and use in hubbard DFT (Dudarev Scheme).
*/
typedef struct _HUB_DUDAREV_OBJ {
    double U[4];            // U values for each l = 0, 1, 2, 3
    double *orbitals;       // orbitals
    double *RadialGrid;     // stores the radial grid
    double *SplineFitOrb;   // derivative of orbitals from Spline
    int size;               // size of the arrays storing the orbitals
    int nspinor;            // number of spin orbitals
    int max_l;              // maximum value of azimuthal quantum number
    int angnum;             // storing the size of rho_mn matrix
    int *ppl;               // To store the number of orbitals for each "l" azimuthal quantum number
    int val_len;            // total number of valence orbitals
    int *occ;               // occupation of each orbital
    double *Uval;           // store Hubbard U values for each orbital, could be used to store any scalar multiplier to the projectors/orbitals
    double rc[3];           // This is a cutoff set to cutoff orbitals beyond this point for cells in x, y, and z direction
    double rc2;
    // double ***rho_mn_init;    // Storing initial rho_mn[spinor][iat][angnum x angnum]
} HUB_DUDAREV_OBJ;



/**
 * @brief   This structure type is designed for storing info. of atoms
 *          that have influence on the distributed domain owned by current
 *          process. Each struct contains atoms of one type.
 */
typedef struct _ATOM_INFLUENCE_OBJ {
    // overlapping atom info.
    int n_atom;     // number of atoms of this type
    double *coords; // coordinates of atoms, dimension = n_atom x 3
    double *atom_spin; // spin on the atom
    int *atom_index; // original atom index the image atom corresponds to
    // overlapping domain info.
    // corners of overlapping region
    int *xs; int *xe;
    int *ys; int *ye;
    int *zs; int *ze; // each of dimension = n_atom x 1
} ATOM_INFLUENCE_OBJ;



/**
 * @brief   This structure type is designed for storing info. of atoms
 *          that have nonlocal influence on the distributed domain owned 
 *          by current process. Each struct contains atoms of one type.
 */
typedef struct _ATOM_NLOC_INFLUENCE_OBJ {
    // overlapping atom info.
    int n_atom;     // number of atoms of this type
    double *coords; // coordinates of atoms, dimension = n_atom x 3
    int *atom_index; // original atom index the image atom corresponds to
    // overlapping domain info.
    // corners of overlapping region
    int *xs; int *xe;
    int *ys; int *ye;
    int *zs; int *ze; // each of dimension = n_atom x 1
        
    int *ndc; // number of grids in the spherical rc-domain
    int **grid_pos; // local positions of the grids in the spherical rc-domain
    
} ATOM_NLOC_INFLUENCE_OBJ;



/**
 * @brief   This structure type is designed for storing info. of atoms
 *          that have local orbital influence on the distributed domain owned 
 *          by current process. Each struct contains atoms of one type.
 */
 typedef struct _ATOM_LOC_INFLUENCE_OBJ {
    // overlapping atom info.
    int n_atom;     // number of atoms of this type
    double *coords; // coordinates of atoms, dimension = n_atom x 3
    int *atom_index; // original atom index the image atom corresponds to
    // overlapping domain info.
    // corners of overlapping region
    int *xs; int *xe;
    int *ys; int *ye;
    int *zs; int *ze; // each of dimension = n_atom x 1
        
    int *ndc; // number of grids in the spherical rc-domain
    int **grid_pos; // local positions of the grids in the spherical rc-domain
    
} ATOM_LOC_INFLUENCE_OBJ;



/**
 * @brief   This structure type is designed for storing nonlocal projectors,
 *          each struct contains nonlocal projectors of atoms of one type.
 */
typedef struct _NLOC_PROJ_OBJ {
    int nproj;                  // number of projectors per atom
    double **Chi;               // projector real
    double _Complex **Chi_c;     // projector complex
    // variables for spin-orbit coupling
    int nprojso;                // number of SO projectors per atom
    double _Complex **Chiso;     // SO projector complex
    int nprojso_ext;            // number of SO projectors (columns) of Chiso matrix in SOC (after extraction)
    double _Complex **Chisowt0;  // Chi matrix withou m = 0
    double _Complex **Chisowtl;  // Chi matrix without m = l
    double _Complex **Chisowtnl; // Chi matrix without m = -l
    // variables for cyclix code
    double **Chi_cyclix;
    double _Complex **Chi_c_cyclix;
    double _Complex **Chiso_cyclix;
    double _Complex **Chisowt0_cyclix;
    double _Complex **Chisowtl_cyclix;
    double _Complex **Chisowtnl_cyclix;
} NLOC_PROJ_OBJ;


/**
 * @brief   This structure type is designed for storing local orbitals,
 *          each struct contains nonlocal projectors of atoms of one type.
 */
 typedef struct _LOC_PROJ_OBJ {
    int nproj;                  // number of projectors per atom
    double **Orb;               // projector real
    double _Complex **Orb_c;     // projector complex
} LOC_PROJ_OBJ;



typedef struct _SPARC_OBJ{
    char SPARCROOT[L_STRING]; // SPARC root directory
    
    /* File names */
    char filename[L_STRING]; 
    char filename_out[L_STRING]; 
    char OutFilename[L_STRING];
    char StaticFilename[L_STRING];
    char AtomFilename[L_STRING];
    char EigenFilename[L_STRING];
    char MDFilename[L_STRING];
    char RelaxFilename[L_STRING];  
    char restart_Filename[L_STRING];
    char restartC_Filename[L_STRING];
    char restartP_Filename[L_STRING];  
    char DensTCubFilename[L_STRING];
    char DensDCubFilename[L_STRING];
    char DensUCubFilename[L_STRING];
    char OrbitalsFilename[L_STRING];
    char KinEnDensTCubFilename[L_STRING];
    char KinEnDensUCubFilename[L_STRING];
    char KinEnDensDCubFilename[L_STRING];
    char XcEnDensCubFilename[L_STRING];
    char ExxEnDensTCubFilename[L_STRING];
    char ExxEnDensUCubFilename[L_STRING];
    char ExxEnDensDCubFilename[L_STRING];
    
    /* Parallelizing parameters */
    int num_node;       // number of processor nodes
    int num_cpu_per_node; // number of cpu per node
    int num_acc_per_node; // number of accelerators per node (e.g., GPUs)
    int npspin;         // number of spin communicators
    int npkpt;          // number of processes for paral. over k-points
    int npband;         // number of processes for paral. over bands
    int npNdx;          // number of processes for paral. over domain in x-dir
    int npNdy;          // number of processes for paral. over domain in y-dir
    int npNdz;          // number of processes for paral. over domain in z-dir
    int npNd;           // total number of processes for paral. over domain
    int npNdx_phi;      // number of processes for calculating phi in paral. over domain in x-dir
    int npNdy_phi;      // number of processes for calculating phi in paral. over domain in y-dir
    int npNdz_phi;      // number of processes for calculating phi in paral. over domain in z-dir 
    int npNdx_kptcomm;  // number of processes in x-dir for creating Cartesian topology in kptcomm 
    int npNdy_kptcomm;  // number of processes in y-dir for creating Cartesian topology in kptcomm 
    int npNdz_kptcomm;  // number of processes in z-dir for creating Cartesian topology in kptcomm
    int useDefaultParalFlag; // Flag for using default parallelization
    int FixRandSeed;    // flag to fix the random number seeds so that all random numbers generated in parallel 
                        // under MPI are the same as those generated in sequential execution
                        
    MPI_Comm spincomm;  // communicator for spin calculation (LOCAL)
    MPI_Comm spin_bridge_comm; // bridging communicator that connects all processes in spincomm that have the same rank (LOCAL)
    MPI_Comm kptcomm;   // communicator for k-point calculations (LOCAL)
    MPI_Comm kptcomm_topo; // Cartesian topology set up on top of a kptcomm (LOCAL)
    MPI_Comm kptcomm_topo_excl; // processors excluded from the Cart topo within a kptcomm (LOCAL)
    MPI_Comm kptcomm_inter; // inter-communicator connecting the Cart topology and the rest in a kptcomm (LOCAL)
    MPI_Comm kpt_bridge_comm; // bridging communicator that connects all processes in kptcomm that have the same rank (LOCAL)
    MPI_Comm bandcomm;  // communicator for band calculations (LOCAL)
    MPI_Comm dmcomm;    // communicator for domain decomposition (LOCAL)
    MPI_Comm blacscomm; // communicator for using blacs to do calculation (LOCAL)
    int ictxt_blacs;    // fortran handle for the context corresponding to blacscomm (for ScaLAPACK) (LOCAL)
    int ictxt_blacs_topo;  // fortran handle for the Cartesian topology within the ictxt_blacs context (for ScaLAPACK) (LOCAL)
    int nprow_ictxt_blacs_topo; // number of rows in the ScaLAPACK context ictxt_blacs_topo (LOCAL)
    int npcol_ictxt_blacs_topo; // number of cols in the ScaLAPACK context ictxt_blacs_topo (LOCAL)
    int desc_orbitals[9];  // ScaLAPACK descriptor for storage of the orbitals on each blacscomm
    int desc_orb_BLCYC[9]; // descriptor for BLOCK CYCLIC distribution of the orbitals on each blacscomm
    int desc_Hp_BLCYC[9];  // descriptor for BLOCK CYCLIC distribution of the projected Hamiltonian on each ictxt_blacs_topo
    int desc_Mp_BLCYC[9];  // descriptor for BLOCK CYCLIC distribution of the overlap matrix on each ictxt_blacs_topo
    int desc_Q_BLCYC[9];   // descriptor for BLOCK CYCLIC distribution of the eigenvectors on each ictxt_blacs_topo
    MPI_Comm dmcomm_phi;// communicator for calculating phi using domain decomposition (LOCAL)
    MPI_Comm comm_dist_graph_phi; // for nonorthogonal phi calc 
    MPI_Comm comm_dist_graph_psi; // for nonorthogonal psi calc
    MPI_Comm kptcomm_topo_dist_graph; // for nonorthogonal in lanczos
    int spincomm_index; // index of current spincomm (LOCAL)
    int Nspin_spincomm;  // number of spin assigned to current spincomm (LOCAL)
    int Nspinor_spincomm;  // number of spinor assigned to current spincomm (LOCAL)
    int kptcomm_index;  // index of current kptcomm (LOCAL)
    int bandcomm_index; // index of current bandcomm (LOCAL)
    int Nkpts_kptcomm;  // number of k-points assigned to current kptcomm (LOCAL)
    int Nband_bandcomm; // number of bands assigned to current bandcomm (LOCAL)
    int band_start_indx;// start index of bands assigned to current bandcomm (LOCAL)
    int band_end_indx;  // end index of bands assigned to current bandcomm (LOCAL)
    
    /* spin options */
    int spin_typ;       // flag to choose 0 - spin unpolarized, 1 - collinear spin, 2 - non-collinear spin
    int Nspin;          // number of spin in a calculation. 1 - spin unpolarized and 2 - spin polarized
    double *mag;        // magnetization
    double *mag_at;     // initial magnetization
    double netM[4];     // net magnetization
    int spin_start_indx; // start index (global) of spin in the spin communicator
    int spinor_start_indx; // start index (global) of spinor in the spin communicator
    int spin_end_indx;  // end index (global) of spin in the spin communicator
    int spinor_end_indx;  // end index (global) of spinor in the spin communicator
    
    /* spin orbit coupling options */
    int Nspinor;        // Number of spinor in wavefunction
    int SOC_Flag;       // Flag for spin-orbit coupling (SOC) calculation
    int Nspinor_eig;    // Number of spinor in eigenvalue problem
    int Nspdentd;       // Number of columns of electron density, total and diagonal term
    int Nspdend;        // Number of columns of electron density, diagonal terms only
    int Nspden;         // Number of columns of spin density
    int Nmag;           // Number of columns of magnetization

    /* Options for MD & Relaxation */
    int MDFlag;
    int RelaxFlag;
    char MDMeth[32];
    char RelaxMeth[32];
    
    int RestartFlag;
    double TWtime;
    
    /* Domain description */
    int cell_typ; // Flag for cell shape (orthogonal/nonorthogonal/helical)
    int Flag_latvec_scale; // Flag indicating wether LATVEC_SCALE is specified
    int numIntervals_x; // number of intervals in x direction
    int numIntervals_y; // number of intervals in y direction
    int numIntervals_z; // number of intervals in z direction  
    int Nx; // number of nodes in x direction
    int Ny; // number of nodes in y direction
    int Nz; // number of nodes in z direction  
    int Nd; // total number of grid nodes
    
    double range_x;
    double range_y;
    double range_z;
    double latvec_scale_x; // scaling factor for the 1st latvec
    double latvec_scale_y; // scaling factor for the 2nd latvec
    double latvec_scale_z; // scaling factor for the 3rd latvec
    double LatVec[9];
    double delta_x;     // mesh size in x-direction
    double delta_y;     // mesh size in y-direction
    double delta_z;     // mesh size in z-direction
    double dV;         
    
    /* discretization */
    // the following two variables are approx. provided by the user,
    // the actual mesh spacing are adjusted and stored in delta_x, 
    // delta_y and delta_z
    double mesh_spacing; // finite-difference mesh spacing
    double ecut;         // ecut used in plane-wave codes, equivalent to mesh-spacing in real-space codes


    // Non cartesian cell 
    double Jacbdet;
    double LatUVec[9];
    double metricT[9];
    double gradT[9];
    double lapcT[9];
    
    /* Distributed domain info. */
    // in phi domain: dmcomm_phi
    int DMVertices[6];  // local domain vertices for calculating phi (LOCAL)
    int Nx_d;           // gridsize of distributed domain in x-dir (LOCAL)
    int Ny_d;           // gridsize of distributed domain in y-dir (LOCAL)
    int Nz_d;           // gridsize of distributed domain in z-dir (LOCAL)
    int Nd_d;           // total number of grids of distributed domain (LOCAL)
    // in kptcomm_topo
    int DMVertices_kptcomm[6]; // local domain vertices for Lanczos (LOCAL)
    int Nx_d_kptcomm;          // gridsize of distributed domain in x-dir in each k-point topology (LOCAL)
    int Ny_d_kptcomm;          // gridsize of distributed domain in y-dir in each k-point topology (LOCAL)
    int Nz_d_kptcomm;          // gridsize of distributed domain in z-dir in each k-point topology (LOCAL)
    int Nd_d_kptcomm;          // total number of grids of distributed domain in each k-point topology (LOCAL)
    // in dmcomm
    int DMVertices_dmcomm[6]; // local domain vertices in dmcomm for storing psi (LOCAL)
    int Nx_d_dmcomm;          // gridsize of distributed domain in x-dir in each dmcomm process (LOCAL)
    int Ny_d_dmcomm;          // gridsize of distributed domain in y-dir in each dmcomm process (LOCAL)
    int Nz_d_dmcomm;          // gridsize of distributed domain in z-dir in each dmcomm process (LOCAL)
    int Nd_d_dmcomm;          // total number of grids of distributed domain in each dmcomm process (LOCAL)
    
    ATOM_INFLUENCE_OBJ *Atom_Influence_local; // atom info. for atoms that have local influence on the distributed domain (LOCAL)
    int isRbOut[3];           // flag for if Rb region is outside domain for Dirichlet BC
    
    /* nonlocal */
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc; // atom info. for atoms that have nonlocal influence on the distributed domain (LOCAL)
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc_kptcomm; // atom info. for atoms that have nonlocal influence on the distributed domain in kptcomm_topo (LOCAL)
    NLOC_PROJ_OBJ *nlocProj;  // nonlocal projectors in psi-domain (LOCAL)
    NLOC_PROJ_OBJ *nlocProj_kptcomm;  // nonlocal projectors in kptcomm_topo (LOCAL)
    NLOC_PROJ_OBJ *nlocProjso;  // nonlocal SO projectors in psi-domain (LOCAL)
    NLOC_PROJ_OBJ *nlocProjso_kptcomm;  // nonlocal SO projectors in kptcomm_topo (LOCAL)

    //int *IP_len;              // nonlocal inner product length corresponding to each atom, size: n_atom x 1
    int *IP_displ;              // start index for storing nonlocal inner product, size: (n_atom + 1) x 1
    int *IP_displ_SOC;          // start index for storing nonlocal inner product, size: (n_atom + 1) x 1
    
    /* Finite difference */
    int order;          // order of central difference
    double *FDweights_D1; // finite difference weights for first derivatives
    double *FDweights_D2; // second difference weights for second derivatives
    double *D2_stencil_coeffs_x; // D2 FD weights / dx2
    double *D2_stencil_coeffs_y; // D2 FD weights / dy2
    double *D2_stencil_coeffs_z; // D2 FD weights / dz2
    double *D2_stencil_coeffs_xy; // D2 FD weights / dxdy
    double *D2_stencil_coeffs_yz; // D2 FD weights / dydz
    double *D2_stencil_coeffs_xz; // D2 FD weights / dzdx
    double *D1_stencil_coeffs_x; // D1 FD weights / dx
    double *D1_stencil_coeffs_y; // D1 FD weights / dy
    double *D1_stencil_coeffs_z; // D1 FD weights / dz
    double *D1_stencil_coeffs_xy;
    double *D1_stencil_coeffs_yx;
    double *D1_stencil_coeffs_xz;
    double *D1_stencil_coeffs_zx;
    double *D1_stencil_coeffs_yz;
    double *D1_stencil_coeffs_zy;
    
    MPEXP_OBJ *MpExp;
    double MaxEigVal_mhalfLap; // max eigenval of -0.5*Lap with periodic boundary conditions
    
    // NOT USED YET
    double coeffs_x;    // stores the weights of the finite-difference laplacian
    double coeffs_y;    // stores the weights of the finite-difference laplacian
    double coeffs_z;    // stores the weights of the finite-difference laplacian
    double stencil_coeffs_x; // stores the weights of the finite-difference laplacian
    double stencil_coeffs_y; // stores the weights of the finite-difference laplacian
    double stencil_coeffs_z; // stores the weights of the finite-difference laplacian
    double coeffs_grad_x; // stores the weights of the finite difference gradient
    double coeffs_grad_y; // stores the weights of the finite difference gradient
    double coeffs_grad_z; // stores the weights of the finite difference gradient
    
    /* Poisson solver */
    int POISSON_SOLVER;  // AAR or CG
    /* Iterations: tolerances and max_iters */
    int MAXIT_SCF;      // max number of SCF iterations
    int MINIT_SCF;      // min number of SCF iterations
    int MAXIT_POISSON;  // max number of iterations for Poisson solver
    int Relax_Niter;    // max number of relaxation iterations in the current run
    int accuracy_level; // accuracy level, 1 - 'low', 2 - 'medium', 3 - 'high', 4 - 'extreme'
    double target_force_accuracy;  // target force accuracy
    double target_energy_accuracy; // target energy accuracy
    double TOL_SCF;     // SCF tolerance
    double TOL_RELAX;   // Relaxation tolerance
    double TOL_POISSON; // Poisson tolerance
    double TOL_LANCZOS; // Lanczos tolerance
    double TOL_PSEUDOCHARGE;    // tolerance for calculating 
                                // pseudocharge density radius
    double TOL_PRECOND;  // tolerance for Kerker preconditioner
    
    /* Preconditioner */
    int precond_fitpow;    // fit power of the rational function
    int precondcoeff_n;    // number of coefficient terms in the rational fit of the preconditioner 
    double precond_kerker_kTF;
    double precond_kerker_thresh;
    double precond_kerker_kTF_mag; // for preconditioning the magnetization
    double precond_kerker_thresh_mag; // for preconditioning the magnetization
    double precond_resta_q0;
    double precond_resta_Rs;
    double precondcoeff_k; // constant term in the rational fit of the preconditioner
    double _Complex *precondcoeff_a; // coeff in the numerator of the rational fit of the preconditioner
    double _Complex *precondcoeff_lambda_sqr; // coeff in the denominator of the rational fit of the preconditioner

    int RelaxCount;     // current relaxation step
    int StressCount;    // current stress count used in full relaxation
    
    double REFERENCE_CUTOFF;
    int REFERENCE_CUTOFF_FAC;
    double *CUTOFF_x;       // pseudocharge cutoff radius in x-direction
    double *CUTOFF_y;       // pseudocharge cutoff radius in y-direction
    double *CUTOFF_z;       // pseudocharge cutoff radius in z-direction
    
    double *Lanczos_x0;                       // initial guess vector for Lanczos
    double _Complex *Lanczos_x0_complex;       // initial guess vector (complex) for Lanczos

    double *Veff_loc_dmcomm;            // effective local potential distributed in psi-domain (LOCAL)
    double *Veff_loc_dmcomm_phi;        // effective local potential distributed in phi-domain (LOCAL)
    double *Veff_dia_loc_dmcomm_phi;    // effective local potential distributed in phi-domain (LOCAL), diagonal term
    double *Veff_loc_dmcomm_phi_in;     // input effective local potential at each SCF distributed in phi-domain (LOCAL)
    double *Veff_loc_kptcomm_topo;      // effective local potential distributed in each kptcomm_topo (LOCAL), only used if npkpt > 1    

    double *mixing_hist_xk;      // previous effective local potential distributed in phi-domain (LOCAL)
    double *mixing_hist_xkm1;    // residual of mixed Veff_loc (between new mixed and previous mixed) in phi-domain (LOCAL)
    double *mixing_hist_fk;      // residual of Veff_loc (between new unmixed and previous mixed) in phi-domain (LOCAL)
    double *mixing_hist_fkm1;    // previous residual of Veff_loc in phi-domain (LOCAL)
    double *mixing_hist_Xk;      // residual matrix of Veff_loc, for mixing (LOCAL)
    double *mixing_hist_Fk;      // residual matrix of the residual of Veff_loc (LOCAL)
    double *mixing_hist_Pfk;     // the preconditioned residual distributed in phi-domain (LOCAL)
    double *mix_Gamma;           // the calculated Gamma for Anderson Mixing
    
    double *psdChrgDens;          // pseudocharge density, "b" (LOCAL)
    double *psdChrgDens_ref;      // reference pseudocharge density, "b_ref" (LOCAL)
    double *Vc;                   // difference between reference pseudopotential V_ref and pseudopotential V, Vc = V_ref - V (LOCAL)
    double *electronDens;         // electron density, "rho" (LOCAL)
    double *electronDens_at;      // electron density guess by summing atomic charge densities (LOCAL)
    double *electronDens_core;    // model core electron density for Non-Linear Core Correction (NLCC)
    double *electronDens_in;      // initial electron density, "rho" (LOCAL) at each SCF distributed in phi-domain (LOCAL)
    double *electronDens_pbe; // electron density for pbe, only in hybrid
    double *elecstPotential;      // electrostatic potential, "phi" (LOCAL)
    double *XCPotential;          // exchange-correlation potential, "Vxc" (LOCAL)
    double *XCPotential_nc;       // exchange-correlation potential, "Vxc" (LOCAL), noncollinear 
    double *e_xc;                 // exchange-correlation energy per particle (LOCAL)
    double *Dxcdgrho;             // derivative of exchange-correlation enrgy per particle wrt to norm of the gradient
    double xc_rhotol;             // minimum value of rho below which it is made equal to xc_rhotol
    double xc_magtol;             // minimum value of mag below which it is made equal to xc_magtol
    double xc_sigmatol;           // minimum value of sgima below which it is made equal to xc_sigmatol
    int ixc[4];                   // decomposition of xc
    int xcoption[2];              // option flag in xc 
    int isgradient;            // flag to determine if xc includes gradient of rho term 
    double *occ;                  // occupations corresponding to k-points owned by local process (LOCAL)
    double *occ_sorted;           // occupations corresponding to sorted lambda
    double occfac;                // occupation's scaling factor
    double *lambda;               // eigenvalues of the Hamiltonian
    double *lambda_sorted;        // eigenvalues of the Hamiltonian in the sorted fashion
    double *Xorb;                 // Kohn-Sham orbitals (LOCAL)
    double *Yorb;                 // Kohn-Sham orbitals (LOCAL)
    double *Xorb_BLCYC;           // block-cyclically distributed orbitals (LOCAL)
    double *Yorb_BLCYC;           // block-cyclically distributed orbitals (LOCAL)
    double _Complex *Xorb_kpt;                 // Kohn-Sham orbitals (LOCAL)
    double _Complex *Yorb_kpt;                 // Kohn-Sham orbitals (LOCAL)
    double _Complex *Xorb_BLCYC_kpt;           // block-cyclically distributed orbitals (LOCAL)
    double _Complex *Yorb_BLCYC_kpt;           // block-cyclically distributed orbitals (LOCAL)
    int    nr_orb_BLCYC;          // number of rows of the local distributed orbitals owned by the process (LOCAL)
    int    nc_orb_BLCYC;          // number of cols of the local distributed orbitals owned by the process (LOCAL)
    int    nr_Hp_BLCYC;           // number of rows of the local distributed projected Hamiltonian owned by the process (LOCAL)
    int    nc_Hp_BLCYC;           // number of cols of the local distributed projected Hamiltonian owned by the process (LOCAL)
    int    nr_Mp_BLCYC;           // number of rows of the local distributed projected Hamiltonian owned by the process (LOCAL)
    int    nc_Mp_BLCYC;           // number of cols of the local distributed projected Hamiltonian owned by the process (LOCAL)
    int    nr_Q_BLCYC;            // number of rows of the local distributed subspace eigenvectors owned by the process (LOCAL)
    int    nc_Q_BLCYC;            // number of cols of the local distributed subspace eigenvectors owned by the process (LOCAL)
    double *Hp;                   // projected Hamiltonian matrix: Hp = Psi' * H * Psi (LOCAL)
    double *Mp;                   // projected mass matrix: Mp = Psi' * Psi (LOCAL)
    double *Q;                    // eigenvectors of the generalized eigenproblem: Hp * Q_i  = lambda_i * Mp * Q_i
    double _Complex *Hp_kpt;                   // projected Hamiltonian matrix: Hp = Psi' * H * Psi (LOCAL)
    double _Complex *Mp_kpt;                   // projected mass matrix: Mp = Psi' * Psi (LOCAL)
    double _Complex *Q_kpt;                    // eigenvectors of the generalized eigenproblem: Hp * Q_i  = lambda_i * Mp * Q_i
    double *Hp_s;                 // whole projected Hamiltonian Hp redistributed for solving eigenproblem (GLOBAL)   
    double *Mp_s;                 // whole projected mass matrix Mp redistributed for solving eigenproblem (GLOBAL)
    #ifdef ACCEL
    int useACCEL;                 // SPARCX_ACCEL_NOTE Flag needed to trigger GPU Acceleration
    int useHIP;                   // Flag to hook in HIP
    int useACCELGT;                  // Glag to indicate whether exact exchange code is accelerated  
    #endif
    int useLAPACK;                // flag for using LAPACK_dsygv to solve subspace eigenproblem
    int eig_serial_maxns;// maximum Nstates for using LAPACK to solve the subspace eigenproblem by default,
                        // for Nstates greater than this value, ScaLAPACK will be used instead, unless 
                        // useLAPACK is turned off.
    int eig_paral_blksz; // block size for distributing the subspace eigenproblem
    double eig_paral_orfac; // specifies which eigenvectors should be reorthogonalized when the "expert" parallel
                            // eigensolver p?syevx or p?sygvx is used.
    int eig_paral_maxnp;           // max number of processes for eigenvalue solver
    int eig_paral_subdims[2];      // dimensions of subgrid of eigensolver

    /* tool variable*/
    MPI_Request req_veff_loc;     // when transmitting Veff_loc, we use nonblocking collectives, 
                                  // this is for checking if transmition is completed later
    D2D_OBJ d2d_dmcomm_phi;       // D2D structure containing target ranks for D2D transfer (obtained by processes in phi domain)
    D2D_OBJ d2d_dmcomm;           // D2D structure containing target ranks for D2D transfer (obtained by processes in psi domain)
    D2D_OBJ d2d_dmcomm_lanczos;   // D2D structure containing target ranks for D2D transfer (obtained by processes in psi domain for Lanczos)
    D2D_OBJ d2d_kptcomm_topo;     // D2D structure containing target ranks for D2D transfer (obtained by processes in kptcomm_topo domain)
    int is_phi_eq_kpt_topo; // flag indicating if dmcomm_phi have the same group of processes as kptcomm_topo
    D2Dext_OBJ *d2dext_dmcomm;
    D2Dext_OBJ *d2dext_dmcomm_ext;
    MPI_Comm dmcomm_d2dext;

    /* Mixing */
    int MixingVariable; // mixing options: 0 - density mixing (default), 1 - potential mixing
    int MixingPrecond;  // Preconditioner: 0 - none, 1 - Kerker (default)
    int MixingPrecondMag;  // Preconditioner for magnetization: 0 - none, 1 - Kerker (default)
    int MixingHistory;       // number of history vectors to keep
    int PulayFrequency;      // Pulay frequency in periodic pulay method
    int PulayRestartFlag;    // Pulay restart flag
    double MixingParameter;  // mixing parameter, often denoted as beta
    double MixingParameterSimple;  // mixing parameter for simple mixing step, often denoted as omega
    double MixingParameterMag;  // mixing parameter for the magnetization density, denoted as beta_mag
    double MixingParameterSimpleMag;  // mixing parameter for the magnetization density in simple mixing step, often denoted as omega_mag

    /* k-points */
    int Nkpts;          // number of k-points
    int Kx;             // number of kpoints in x direction
    int Ky;             // number of kpoints in y direction
    int Kz;             // number of kpoints in z direction
    int Nkpts_sym;      // number of k-points after symmetry reduction
    int NkptsGroup;     // number of k-point groups for parallelization
    int kptParalFlag;   // k-point parallelization flag
    int kctr;           // counter for k points  
    int kpt_start_indx; // start k point index in local kptcomm (LOCAL)
    int kpt_end_indx;   // end k point index in local kptcomm (LOCAL)
    int isGammaPoint;   // 1 - only gamma-point calculation, 0 - not gamma-point calculation
    double kptshift[3]; // k point shift in each direction
    double *lambdakpt;
    double *kptWts;
    double *k1;
    double *k2;
    double *k3;
    double *kptWts_loc;
    double *k1_loc;
    double *k2_loc;
    double *k3_loc;

    /* system description */
    int BC;             // boundary conditions
    int BCx;            // boundary condition in x dir
    int BCy;            // boundary condition in y dir
    int BCz;            // boundary condition in z dir
    int Nstates;        // number of states
    int Ntypes;         // number of atome types
    int Nelectron;      // total number of electrons, read from ion file
    double Nelectron_up; // Total number of alpha electrons   
    double Nelectron_dn; // Total number of beta electrons   
    int NetCharge;      // net charge of the system
    double PosCharge;   // positive charge, found by integrating the pseudocharge density
    double NegCharge;   // negative charge, defined as -1*integarl of electron density
    int elec_T_type;    // electronic temperature (smearing) type
    double Beta;        // electronic smearing (1/k_B*T) [1/Ha]
    double elec_T;      // electronic temperature in [Kelvin] or [Hartree] 
    double rho;         // electron density
    char XC[32];        // exchange correlation name
    /////////////////////////////////////////////////////////
    int is_default_psd;     // flag for reading pseudopotential
    int NLCC_flag;          // flag for Non-Linear Core Correction (NLCC)
    int n_atom;             // total number of atoms
    int *localPsd;          // respective local component of pseudopotential for 
                            // each type of atom. 0 for s, 1 for p, 2 for d, 3 for f                  
    double *Mass;           // atomic mass (for MD)
    double TotalMass;       // Total atomic mass
    char *atomType;         // atom type name for every type of atom    
    int *Zatom;             // atom number
    int *Znucl;             // valence charge of each type
    int *nAtomv;            // number of atoms of each type
    char *psdName;          // pseudopotential file name for each type of atom
    double *atom_pos;       // atom positions
    int *IsFrac;            // array to store the coordinate type (cart or frac) of each atom type
    int *IsSpin;            // Flag to store whether spin of particular type of atom is provided or not
    int *mvAtmConstraint;   // atom relax constraint, 1/0 -- move/don't move atom in this DOF
    double *atom_spin;      // stores the net spin on each atom
    PSD_OBJ *psd;           // struct array storing pseudopotential info.
    ///////////////////////////////////////////////////////////

    /* atom solver */
    int *atom_solve_flag; // To indicate if a radial atom solve is required for the particular atom type
    int atom_solve_count; // total number of atom types for which atom_solve_count is desired
    int atomSolvSpinFlag;

    /* hubbard DFT */
    HUB_DUDAREV_OBJ *AtmU;      // struct array storing atom solution info
    double E_Hub_HF;
    int is_hubbard;             // flag indicating a hubbard DFT calculation
    int *IP_displ_U;            // Storing displacements for inner product (per atom)
    int *rho_mn_displ;          // storing displacements for occupation matrix (per atom)
    double ***rho_mn;           // storing occupation matrix for atoms having U corrections
    double ***rho_mn_at;        // for occupation matrix extrapolation
    double ***drho_mn;          // for occupation matrix extrapolation
    double ***drho_mn_0dt;      // for occupation matrix extrapolation
    double ***drho_mn_1dt;      // for occupation matrix extrapolation
    double ***drho_mn_2dt;      // for occupation matrix extrapolation
    double *atom_pos_nm_occM;   // for occupation matrix extrapolation
    double *atom_pos_0dt_occM;  // for occupation matrix extrapolation
    double *atom_pos_1dt_occM;  // for occupation matrix extrapolation
    double *atom_pos_2dt_occM;  // for occupation matrix extrapolation
    double **rho_mn_X;          // Residual matrix for occ mat
    double **rho_mn_F;          // Residual matrix of the residual for occ mat
    double **rho_mn_xkm1;       // residual of mixed occ mat (new mixed and previous mixed)
    double **rho_mn_fkm1;       // previous residual of mixed occ mat
    double **rho_mn_xk;         // previous occ mat
    double **rho_mn_fk;         // previous residual occ mat
    LOC_PROJ_OBJ *locProj;      // local orbitals in psi-domain (LOCAL)
    LOC_PROJ_OBJ *locProj_kptcomm;      // local orbitals in kptcomm_topo (LOCAL)
    ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc_orb;     // atom info. for atoms that have local orbital influence on distributed domain (LOCAL)
    ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc_orb_kptcomm;     // atom info. for atoms that have local orbital influence on distributed domain in kptcomm_topo (LOCAL)
    double stress_hub[6];                   // Hubbard contribution to stress
    double pres_hub;                       // Hubbard contribution to pressure
 
    /* Chebyshev filtering */
    int ChebDegree;        // degree of Chebyshev polynomial
    int CheFSI_Optmz;      // flag for optimizing Chebyshev filtering polynomial degrees
    int rhoTrigger;        // triger for starting to update electron density during scf iterations
    int chefsibound_flag;  // flag for estimating upper bounds of Chebyshev Filtering in every SCF iter
    double *eigmin;        // Stores minimum eigenvalue of Hamiltonian/Laplacian
    double *eigmax;        // Stores maximum eigenvalue of Hamiltonian/Laplacian
    int npl_min;
    int npl_max;
    int Nchefsi;           // Number of ChefSi for each scf step

    /* MLFF */
    MLFF_Obj *mlff_str;
    double condK_min;
    double factor_multiply_sigma_tol;
    int if_sparsify_before_train;

    int begin_train_steps;
    int if_atom_data_available;
    // int if_merge_mlff;
    // int n_merge_mlff;
    // char fname_merge_prefix[120];
    int N_max_SOAP;
    int L_max_SOAP;
    int n_str_max_mlff;
    int n_train_max_mlff;
    int mlff_flag;
    int last_train_iter;
    int kernel_typ_MLFF;
    int descriptor_typ_MLFF;
    int N_rgrid_MLFF;
    int print_mlff_flag;
    int mlff_internal_energy_flag;
    int mlff_pressure_train_flag;
    // gmp
    double radial_min;
    double radial_max;
    //
    double rcut_SOAP;
    double sigma_atom_SOAP;
    double beta_2_SOAP;
    double beta_3_SOAP;
    double xi_3_SOAP;
    double F_tol_SOAP;
    double F_rel_scale;
    char mlff_data_folder[L_STRING];
    double stress_rel_scale[6];
    int MLFF_DFT_fq;
    
    /* Energies */
    double Esc;            // self + correction energy, Esc = Eself + Ec
    double Efermi;         // fermi energy
    double Exc;            // exchange-correlation energy
    double Exc_corr;       // correction in exchange-correlation energy
    double Eband;          // band structure energy
    double Entropy;        // entropy
    double Etot;           // total free energy
    // double Eatom;          // free energy per atom
    double Escc;           // Self-consistency correction energy
    
    /* Forces */
    double *forces;        // atomic forces
    double *AtomMag;       // atom magnetization
    
    /* Stress and Pressure */
    int Calc_stress;       // Flag for calculating stress
    int Calc_pres;         // Flag for calculating pressure
    double stress[6];      // Full stress tensor
    double stress_k[6];    // Kinetic contribution to stress tensor
    double stress_xc[6];   // Exchange-correlation contr. to ST
    double stress_el[6];   // Electrostatics contr. to ST
    double stress_nl[6];   // Nonlocal psp. contr. to ST
    double stress_i[6];    // Ionic contr. to ST
    double stress_exx[6];  // Exact Exchange contr. to ST
    double pres;           // Full pressure
    double pres_xc;        // Exchange correlation contr. to pres
    double pres_el;        // Electrostatics contr. to pres
    double pres_nl;        // Nonlocal psp. contr. to pres
    double pres_i;         // Ionic contr. to pres
    double pres_exx;       // Exact Exchange contr. to pres

    /* MD/relax options */
    double *ion_vel;       // Ionic velocity 
    double *ion_accel;     // Ionic acceleration
    double ion_T;          // Ionic temperature
    double PE, KE, TE, TE_ext;     // potential, kinetic and total energies respectively
    double kB;             // Boltzmann constant
    double MD_dt;          // MD time step [femtosecond]
    double mean_elec_T;    // Average of electronic temperature
    double mean_ion_T;     // Average of ionic temperature
    double mean_TE;        // Average of total energy
    double mean_KE;        // Average of kinetic energy of ions
    double mean_PE;        // Average of electronic energy
    double mean_U;         // Average of internal energy
    double mean_Entropy;   // Average of entropic energy
    double mean_TE_ext;    // Average of extended system energy
    double std_elec_T;     // Standard deviation of electronic temperature
    double std_ion_T;      // Standard deviation of ionic temperature
    double std_TE;         // Standard deviation of total energy
    double std_KE;         // Standard deviation of kinetic energy of ions
    double std_PE;         // Standard deviation of electronic energy
    double std_U;          // Standard deviation of internal energy
    double std_Entropy;    // Standard deviation of entropic energy
    double std_TE_ext;     // Standard deviation of extended system energy
    int MD_maxStep;        // Maximum number of MD steps to be performed
    int MD_Nstep;        // Number of MD steps to be performed in the current run
    int dof;               // Degree of freedom 
    int MDCount;           // Flag accounting for step count of MD
    int restartCount;      // Flag accounting for step count of restarted MD
    int StopCount;         // Flag to count the MD steps written in .aimd file
    int ion_vel_dstr;      // Flag to choose the initial velocity distribution
    int ion_vel_dstr_rand; // Flag to randomize the initial velocity distribution
    int ion_elec_eqT;      // Flag to choose whether ionic and electronic temperature will be same throughout MD
    double thermos_T;      // Temperature of the thermostat at the current time step
    double thermos_Ti;     // Initial temperature of the thermostat
    double thermos_Tf;     // Final temperature of the thermostat 
    double v2nose;         // Next 4 are Nose Hoover thermostat parameters: energy parameter
    double snose;          // diatance parameter
    double xi_nose;        // velocity parameter
    double qmass;          // mass parameter
    double amu2au;         // conversion factor for atomic mass unit -> atomic unit of mass
    double fs2atu;         // conversion factor for femto second -> atomic unit of time (Jiffy) 
    // NPT
    int NPTscaleVecs[3];    // which lattice vector can be rescaled?
    int NPTconstraintFlag; // confinement on side length of cell. none: no length confinement (default); 1: a:b keeps unchanged; 2: a:c keeps unchanged; 
    // 3: a:c keeps unchanged; 4: a:b:c keeps unchanged, isotropic expansion. It is only available for NPT_NP.
    int NPTisotropicFlag;   // whether it is an isotropic cell expansion; a:b:c keeps similar during NPT. 
    // For NPT_NH, if all 3 lattive vectors are scalable, it will be an isotropic expansion;
    // For NPT_NP, if all 3 lattive vectors are scalable, AND NPTconstraintFlag is 4, it will be an isotropic expansion.
    double prtarget;       // Target pressure of barostatic system, used in both NPT_NH and NPT_NP
    double scale;          // length ratio of the size of cell in NPT, used in both NPT_NH and NPT_NP
    double volumeCell;  // volume of the cell, used in both NPT_NH and NPT_NP
    double initialLatVecLength[3]; // used for outputting LATVEC_SCALE
    // NPT-NH
    int NPT_NHnnos;              // amount of thermostat variables in the Nose-Hoover chain, it should be smaller than 100
    double NPT_NHqmass[L_QMASS];    // qmass in NPT, the largest number of thermostat variable is 60 //ORIGINAL VARIABLES! DON'T CHANGE IT!
    double NPT_NHbmass;          // bmass in NPT
    double glogs[L_QMASS];       // Accelerations of virtual thermal variables
    double vlogs[L_QMASS];       // Velocities of virtual thermal variables
    double vlogv;          // Velocity of virtual baro variables
    double xlogs[L_QMASS];       // Positions of virtual thermal variables
    double Hamiltonian_NPT_NH; // Hamiltonian of the NPT-NH system
    // NPT-NP
    int maxTimeIter;     // largest allowed amount of iteration
    double NPT_NP_qmass; // qmass used in NPT_NP
    double NPT_NP_bmass; // bmass used in NPT_NP
    double range_x_velo; // velocity of x sidelength
    double range_y_velo; // velocity of y sidelength
    double range_z_velo; // velocity of z sidelength
    double G_NPT_NP[3]; // G tensor, for barostat control
    double Pm_NPT_NP[3]; // Pm tensor
    double Kbaro; // kinetic energy of barostat variables
    double Ubaro; // potential energy of barostat variables
    double S_NPT_NP; // S, for thermostat control
    double Sv_NPT_NP; // velocity of S
    double Kther; // kinetic energy of thermostat variable
    double Uther; // potential energy of thermostat variable
    double Hamiltonian_NPT_NP; // Hamiltonian of the NPT-NP system
    double init_Hamil_NPT_NP; // initial Hamiltonian of the system
    // Relaxation
    double Relax_fac;      // Relaxation factor
    int elecgs_Count;      // To count the number of times electronic ground state is calculated
    double *d;             // Search direction in case of NLCG 
    double NLCG_sigma;          // parameter used in NLCG   
    int L_history;         // maximum number of relaxation steps in LBFGS stored
    double L_finit_stp;    // finite step for line optiization
    double L_maxmov;       // maximum allowed step size for translation
    int L_autoscale;       // automatically determine icurv
    int L_lineopt;         // perform force based line minimizer
    double L_icurv;        // Initial inverse curvature, used to construct the inverse Hessian matrix
    double *deltaX;        // LBFGS variable
    double *deltaG;        // LBFGS variable
    double *iys;           // LBFGS variable
    double *fold;          // LBFGS variable
    double *atom_disp;     // LBFGS variable
    int isFD;              // LBFGS variable
    int isReset;           // LBFGS variable
    int step;              // LBFGS variable
    double FIRE_dt;        // FIRE variable
    double FIRE_mass;      // FIRE variable
    double FIRE_maxmov;    // FIRE variable
    double FIRE_alpha;     // FIRE variable
    double *FIRE_vel;      // FIRE variable
    double FIRE_dtNow;     // FIRE variable
    int FIRE_resetIter;    // FIRE variable
    
    // cell relaxation (volume only)
    int cellrelax_ndim;
    int cellrelax_dims[3];
    double max_dilatation;
    double TOL_RELAX_CELL;

    /* EigenValue problem*/
    int StandardEigenFlag;

    // DFT-D3 correction
    double d3Energy[4]; // total d3 energy, e6, e8, e63
    double *d3Grads;   // atomic forces caused by d3 (-grad)
    double d3Stress[9]; // stress of cell caused by d3

    int d3Flag;
    double d3Rthr;        // cutoff radius for calculating d3 energy correction
    double d3Cn_thr;      // cutoff radius for calculating CN parameter of every atom

    int *atomicNumbers;  // atomic numbers of atoms
    double *atomScaledR2R4;
    double *atomScaledRcov;
    double *atomCN;
    int *atomMaxci;
    int nImageCN[3]; // largest amount of image cells on 3 directions needed to consider for computing CN
    int nImageEG[3]; // largest amount of image cells on 3 directions needed to consider for computing d3 energy
    double *****c6ab; // pointer of C6 sample results
    double lattice[9];
    int BCtype[3];
    int periodicBCFlag;
    double d3Rs6;
    double d3S18;
    FILE *d3Output;
    double d3Sigma[9]; // stress of cell caused by d3

    // vdW-DF1 and vdW-DF2
    int vdWDFnrpoints;
    int vdWDFnqs;
    double *vdWDFecLinear;
    double *vdWDFVcLinear;
    double vdWDFdr;
    double vdWDFdk;
    double vdWDFZab;
    double detLattice;
    double *vdWDFqmesh;
    double **vdWDFkernelPhi;
    double **vdWDFd2Phidk2;
    double **vdWDFd2Splineydx2;
    double **Drho;
    double *gradRhoLen;
    double *vdWDFq0;
    double *vdWDFdq0drho;
    double *vdWDFdq0dgradrho;
    double **vdWDFps;
    double **vdWDFdpdq0s;
    MPI_Comm zAxisComm;
    int newzAxisDims[3];
    int zAxisVertices[6];
    D2D_OBJ gatherThetasSender;
    D2D_OBJ gatherThetasRecvr;
    D2D_OBJ scatterThetasSender;
    D2D_OBJ scatterThetasRecvr;
    double _Complex **vdWDFthetaFTs;
    double reciLattice[9]; // inverse of lattice multiply a coefficient
    int **timeReciLattice; // reciprocal lattice grid point i = timeReciLattice[0][i]*pSPARC->reciLattice0(0 1 2)+timeReciLattice[1][i]*pSPARC->reciLattice1(3 4 5)+timeReciLattice[2][i]*pSPARC->reciLattice2(6 7 8)
    double **vdWDFreciLatticeGrid; // 000 100 200 ... 010 110 210 ... 001 101 201 ... 011 111 211 ...
    double *vdWDFreciLength;
    double **vdWDFkernelReciPoints;
    double _Complex **vdWDFuFTs;
    double **vdWDFu;
    double vdWDFenergy;
    double *vdWDFpotential;

    int countPotentialCalculate; // for helping output variables in 1st step, to be deleted in the future
    
    /* metaGGA functionals (SCAN) */
    double *KineticTauPhiDomain;
    double *vxcMGGA1; // d(n\epsilon)/dn, in dmcomm_phi
    double *vxcMGGA2; // d(n\epsilon)/d|grad n|, in dmcomm_phi
    double *vxcMGGA3; // d(n\epsilon)/d(\tau), in dmcomm_psi
    double *vxcMGGA3_loc_dmcomm; // d(n\epsilon)/d(\tau), in dmcomm (saving \psi)
    double *vxcMGGA3_loc_kptcomm; // d(n\epsilon)/d(\tau), in kptcomm (handling different kpts)

    /* Exact Exchange */
    int usefock;                    // Flag for if using Hartree-Fock operator 
    double TOL_FOCK;                // Exact exchange potential option
    double TOL_SCF_INIT;            // Exact exchange potential option
    int MAXIT_FOCK;                 // Maximum number of iterations for Hartree-Fock outer loop
    int MINIT_FOCK;                 // Minimum number of iterations for Hartree-Fock outer loop
    double exx_frac;                // hybrid mixing coefficient
    double hyb_range_fock;          // hybrid short range for fock operator 
    double hyb_range_pbe;           // hybrid short range for exchange correlation 
    int EXXMeth_Flag;               // Method to solve Poisson's equation, in Real space or Fourier space
    double Eexx;                    // Exact Exchange energy
    double *psi_outer;              // outer orbitals to construct Hartree-Fock operator 
    double *occ_outer;              // outer occupations to construct Hartree-Fock operator 
    double *psi_outer_kptcomm_topo; // outer orbitals in kptcomm for Lanczos 
    double _Complex *psi_outer_kpt;  // outer orbitals to construct Hartree-Fock operator 
    double _Complex *psi_outer_kptcomm_topo_kpt; // outer orbitals in kptcomm for Lanczos 
    double *Xi;                     // ACE operator
    double *Xi_kptcomm_topo;        // ACE operator in kptcomm_topo
    double _Complex *Xi_kpt;                 // ACE operator
    double _Complex *Xi_kptcomm_topo_kpt;    // ACE operator in kptcomm_topo
    // k-points variables for hybrid calculation
    int Nkpts_hf;                   // number of k-points for hybrid calculation
    int Kx_hf;                      // number of k-points for hybrid calculation in x direction
    int Ky_hf;                      // number of k-points for hybrid calculation in y direction
    int Kz_hf;                      // number of k-points for hybrid calculation in z direction
    int Nkpts_hf_red;               // actual number of k-points for hybrid calculation after downsampling
    double *k1_hf;                  // list of x-coordinates of k-points for hybrid calculation
    double *k2_hf;                  // list of y-coordinates of k-points for hybrid calculation
    double *k3_hf;                  // list of z-coordinates of k-points for hybrid calculation
    double kptWts_hf;               // k-point weights for hybrid calculation
    int *kpthf_ind;                 // mapping from global k-point list to k-point list for hyrbid 
    int *kpthf_ind_red;             // index w.r.t. Nkpts_hf_red
    int *kpthf_pn;                  // indication of whether it's a +k or -k
    double *k1_shift;               // list of x-coordinates of unique Bloch vector shifts (k-q)
    double *k2_shift;               // list of y-coordinates of unique Bloch vector shifts (k-q)
    double *k3_shift;               // list of z-coordinates of unique Bloch vector shifts (k-q)
    int Nkpts_shift;                // number of unique Bloch vector shifts (k-q)
    int *Kptshift_map;              // mapping from given 2 kpoints to a Block vector shift
    int Nkpts_hf_kptcomm;           // number of k-points in kptcomm for hybrid calculation
    int *kpthf_flag_kptcomm;        // flags for whether the k-point used or not for hybrid calculation
    int *Nkpts_hf_list;             // Number of k-point orbitals gathered from each k-point processes
    int kpthf_start_indx;           // starting index for k-point for hybrid calculation
    int *kpthf_start_indx_list;     // starting index for k-point for hybrid calculation
    int *kpts_hf_red_list;          // list of reduced k-point for hybrid calculation 
    double _Complex *neg_phase;      // exp(-i*r*k_shift)
    double _Complex *pos_phase;      // exp(i*r*k_shift)
    int (*kpthfred2kpthf)[3];       // mapping from kpthf_red to kpthf
    // tool variables for hybrid calculation
    double ACEtime;                 // Time for creating ace operator
    double Exxtime;                 // Time for applying Vexx operator
    double Exxtime_comm;
    double Exxtime_solver;
    double Exxtime_memload;
    double Exxtime_cyc;
    double Exxtime_rhs;
    double Exxtime_move;
    double *pois_const;             // Constants for FFT solver in Poisson's equation
    double *pois_const_stress;      // Constants for FFT solver in Poisson's equation in stress
    double *pois_const_stress2;     // Constants for FFT solver in Poisson's equation in stress
    double *pois_const_press;       // Constants for FFT solver in Poisson's equation in press
    int ExxAcc;                     // Flag for enable accelerating method for either low-T or high-T
    int Nstates_occ;                // Number of occupied states 
    int Nstates_occ_list[2];        // List of number of occupied states 
    int ExxMemBatch;               // Option for speed or memory efficiency when using ACE operator
    int EeeAceValState;            // Number of extra unoccupied states in constructing ACE operator
    int EXXDownsampling[3];         // Downsampling info
    double const_aux;               // constant for auxlliary function
    int ExxDivFlag;                // Method for integrable singularity 
    int flag_kpttopo_dm;            // flag of whether the dmcomm and kpttopo are the same
    int flag_kpttopo_dm_type;       // flag for receving or sending the correct occupations
    MPI_Comm kpttopo_dmcomm_inter;  // the extra communicator for occupations transferring 
    // variabels for band parallelization with ACE
    int Nband_bandcomm_M;           // number of bands of M assigned to current bandcomm (LOCAL)
    KRON_LAP** kron_lap_exx;        // structure for kronecker product laplacian
    MPEXP_OBJ *MpExp_exx;           // structure for multipole expansion
    int ExxMethod;                  // method for solving poissons equation, kronecker product way or FFT
    double fock_err;                // Error in outer loop

    /* SQ methods */
    int sqAmbientFlag;                     // Flag of SQ method
    // int SQ_highT_gauss_mem;               // Memory option for gauss quadrature 
    int SQ_highT_hybrid_gauss_mem;        // Memory option for gauss quadrature for SQ
    int SQ_npl_g;                   // Degree of polynomial (should be a multiple of 4) for Gauss Quadrature
    int SQ_correction;              // Flag for culculating "charge overlap correction".
    double SQ_rcut;                 // Truncation or localization radius    
    double SQ_tol_occ;              // Tolerance for occupation corresponding to maximum eigenvalue
    int npNdx_SQ;           // number of processes for paral. over domain in x-dir
    int npNdy_SQ;           // number of processes for paral. over domain in y-dir
    int npNdz_SQ;           // number of processes for paral. over domain in z-dir
    D2D_OBJ d2d_s2p_sq;             // D2D object for communication from sqcomm to phicomm, sq end
    D2D_OBJ d2d_s2p_phi;            // D2D object for communication from sqcomm to phicomm, phi end
    ATOM_NLOC_INFLUENCE_OBJ **Atom_Influence_nloc_SQ;   // atom info. for atoms that have nonlocal influence on the distributed domain (LOCAL)
    NLOC_PROJ_OBJ **nlocProj_SQ;    // nonlocal projectors in psi-domain (LOCAL)
    SQ_OBJ *pSQ;                    // SQ object
    int sqHighTFlag;

    /* OFDFT */
    int OFDFTFlag;
    double OFDFT_tol;
    double OFDFT_Cf;
    double OFDFT_lambda;
    double *OFDFT_u;
    double OFDFT_Ek;
    double OFDFT_Eele;

    /* cyclix */
    int CyclixFlag;
    double twist;  // Twist of the cyclinder in radian/Bohr
    double twistpercell; // Total twist per unit cell
    double xin; // Coordinate of the 0th node of the cell in x-direction
    double RotM_cyclix[9]; // Rotational matrix to rotate the coordinates in cyclic-helical system
    double xmin_at; // Minimum of all atomic radial coordinates
    double xmax_at; // Maximum of all atomic radial coordinates
    double xvac; //  Vacuum in the radial direction
    double xout; // Outer radius of the disc
    double *Intgwt_kpttopo; // To store the integration weights for nodes in kptcomm_topo communicator processors
    double *Intgwt_psi; // To store the integration weights for nodes in psi domain communicator processors
    double *Intgwt_phi; // To store the integration weights for nodes in phi domain communicator processors

    // Variables needed for LAPACKE eigensolver for nonsymmetric matrices
    double *lambda_temp1;
    double *lambda_temp2;
    double *lambda_temp3;
    double *vl;
    double *vr;
    double _Complex *lambda_temp1_kpt;
    double _Complex *lambda_temp2_kpt;
    double _Complex *vl_kpt;
    double _Complex *vr_kpt;

    // Extrapolation options
    double *delectronDens;
    double *delectronDens_0dt;
    double *delectronDens_1dt;
    double *delectronDens_2dt;
    double *atom_pos_nm;
    double *atom_pos_0dt;
    double *atom_pos_1dt;
    double *atom_pos_2dt;
    
    /* print options */
    int Verbosity;
    int PrintForceFlag;
    int PrintAtomPosFlag;
    int PrintAtomVelFlag;
    int PrintEigenFlag;
    int PrintElecDensFlag;
    int PrintMDout;
    int PrintRelaxout;
    int Printrestart;
    int Printrestart_fq;
    int suffixNum;  // the number appended to the output filename, only used if it's greater than 0    
    int PrintPsiFlag[7];
    int PrintEnergyDensFlag;
    
    /* Energy density */
    double *KineticRho;         // Kinetic energy density
    double *ExxRho;             // Exact exchange energy density
    double *ExcRho;             // Exchange correlation energy density

    /* Timing */
    double time_start;
    
    /* memory */
    double memory_usage;
    
    // Domain parallelization (decomposition) data layout for calculating projected Hamiltonian, 
    // generalized eigen problem, and subspace rotation
    void *DP_CheFSI;     // Pointer to a DP_CheFSI_s data structure for those three procedures w/o Kpt
    void *DP_CheFSI_kpt; // Pointer to a DP_CheFSI_kpt_s data structure for those three procedures w/ Kpt    
    /* Band structure plot*/
    int n_kpt_line;
    double kredx[L_kpoint],kredy[L_kpoint],kredz[L_kpoint];
    double k1_inpt_kpt[L_kpoint],k2_inpt_kpt[L_kpoint],k3_inpt_kpt[L_kpoint];
    int BandStructFlag;
    int kpt_per_line;
    char InDensTCubFilename[L_STRING];
    char InDensUCubFilename[L_STRING];
    char InDensDCubFilename[L_STRING]; 
    int densfilecount;
    int readInitDens; // flag for reading inital density
   /* Socket interface
      Please keep this section as the last block in SPARC_OBJ definition,
      add new features before this block.
    */
  
#ifdef USE_SOCKET
    int SocketFlag;             // boolean value to indicate whether to switch on socket support at all
    int SocketSCFCount;   	// scf count in socket mode
    char socket_host[L_STRING]; // socket file descriptor
    int socket_port;            // socket host
    int socket_inet;            // boolean value to indicate whether to use inet socket
    int socket_fd;             // socket file descriptor; This should be initialized to -1
    int socket_max_niter;           // max number of iterations. Default is 10000
#endif
}SPARC_OBJ;




/**
 * @brief   This structure type is for reading inputs and broadcasting. 
 *           
 *          This structure contains part of the members of SPARC_OBJ, 
 *          which are read from input files. These members are seperated
 *          from the whole structure in order to avoid broadcasting 
 *          uninitialized variables. Also one can reorder the members in  
 *          order to improve the speed of broadcasting the structure.
 *
 */
typedef struct _SPARC_INPUT_OBJ{ 

    /* MLFF */
    
    int if_sparsify_before_train;
    int begin_train_steps;
    int if_atom_data_available;
    int N_max_SOAP;
    int L_max_SOAP;
    int n_str_max_mlff;
    int n_train_max_mlff;
    int mlff_flag;
    int kernel_typ_MLFF;
    int descriptor_typ_MLFF;
    int print_mlff_flag;
    int mlff_internal_energy_flag;
    int mlff_pressure_train_flag;
    int N_rgrid_MLFF;
    // gmp
    double radial_min;
    double radial_max;
    //
    double condK_min;
    double factor_multiply_sigma_tol;
    double rcut_SOAP;
    double sigma_atom_SOAP;
    double beta_3_SOAP;
    double xi_3_SOAP;
    double F_tol_SOAP;
    double F_rel_scale;
    double stress_rel_scale[6];
    char mlff_data_folder[L_STRING];
    int MLFF_DFT_fq;
    
    /* Parallelizing parameters */
    int num_node;       // number of processor nodes
    int num_cpu_per_node; // number of cpu per node
    int num_acc_per_node; // number of accelerators per node (e.g., GPUs)
    int npspin;         // number of spin communicators
    int npkpt;          // number of processes for paral. over k-points
    int npband;         // number of processes for paral. over bands
    int npNdx;          // number of processes for paral. over domain in x-dir
    int npNdy;          // number of processes for paral. over domain in y-dir
    int npNdz;          // number of processes for paral. over domain in z-dir
    int npNdx_phi;      // number of processes for calculating phi in paral. over domain in x-dir
    int npNdy_phi;      // number of processes for calculating phi in paral. over domain in y-dir
    int npNdz_phi;      // number of processes for calculating phi in paral. over domain in z-dir    
    int eig_serial_maxns;   // maximum Nstates for using LAPACK to solve the subspace eigenproblem by default,
                        // for Nstates greater than this value, ScaLAPACK will be used instead, unless 
                        // useLAPACK is turned off.
    int eig_paral_blksz; // block size for distributing the subspace eigenproblem
    
    int MDFlag;
    int RelaxFlag;
    int RestartFlag;
    int Flag_latvec_scale; // Flag indicating wether LATVEC_SCALE is specified
    int numIntervals_x; // number of intervals in x direction
    int numIntervals_y; // number of intervals in y direction
    int numIntervals_z; // number of intervals in z direction  
    
    /* spin options */
    int spin_typ;       // flag to choose between spin unpolarized and spin polarized calculation
    
    
    /* system description */
    int BC;             // boundary conditions
    int BCx;            // boundary condition in x dir
    int BCy;            // boundary condition in y dir
    int BCz;            // boundary condition in z dir
    int Nstates;        // number of states
    int Ntypes;         // number of atome types
    int NetCharge;      // net charge of the system
    
    int order;          // order of central difference    

    /* Chebyshev filtering */
    int ChebDegree;     // degree of Chebyshev polynomial   
    int CheFSI_Optmz;   // flag for optimizing Chebyshev filtering polynomial degrees
    int chefsibound_flag; // flag for calculating bounds for Chebyshev filtering
    int rhoTrigger;      // triger for starting to update electron density during scf iterations
    int Nchefsi;         // Number of ChefSi for each scf step
    
    /* Iterations */
    int FixRandSeed;    // flag to fix the random number seeds so that all random numbers generated in parallel 
                        // under MPI are the same as those generated in sequential execution
                        // 0 - off (default), 1 - on
    int accuracy_level; // accuracy level, 1 - 'low', 2 - 'medium', 3 - 'high', 4 - 'extreme'
    int MAXIT_SCF;      // max number of SCF iterations
    int MINIT_SCF;      // min number of SCF iterations
    int MAXIT_POISSON;  // max number of iterations for Poisson solver
    int Relax_Niter;    // max number of relaxation iterations in the current run
    int MixingVariable; // mixing options: 0 - density mixing (default), 1 - potential mixing
    int MixingPrecond;  // Preconditioner: 0 - none, 1 - kerker (default)
    int MixingPrecondMag;  // Preconditioner for magnetization: 0 - none, 1 - kerker (default)
    int MixingHistory;
    int PulayFrequency;
    int PulayRestartFlag;

    /* fit power for the real-space preconditioner */
    int precond_fitpow;    // fit power of the rational function

    /* k-points */
    int Nkpts;          // number of k-points
    int Kx;             // number of kpoints in x direction
    int Ky;             // number of kpoints in y direction
    int Kz;             // number of kpoints in z direction
    int NkptsGroup;     // number of k-point groups for parallelization
    int kctr;           // counter for k points  
    
    /* print options */
    int Verbosity;
    int PrintForceFlag;
    int PrintAtomPosFlag;
    int PrintAtomVelFlag;
    int PrintEigenFlag;
    int PrintElecDensFlag;
    int PrintMDout;
    int PrintRelaxout;
    int Printrestart;
    int Printrestart_fq;
    int PrintPsiFlag[7];
    int PrintEnergyDensFlag;
    
    /* Smearing */
    int elec_T_type;    // electronic temperature (smearing) type, 0 - fermi-dirac, 1 - gaussian
    
    /* MD options */
    int MD_Nstep;     // Number of MD steps to run in the current simulation
    int ion_elec_eqT; // Flag to choose whether ionic and electronic temperature will be same throughout MD    
    int ion_vel_dstr;   // initial distribution of ionic velocities(1-uniform,2-Maxwell Boltzmann,3-input)
    int ion_vel_dstr_rand; // Flag to randomize the initial velocity distribution
    
    /* Relaxation options */
    int L_history;
    int L_autoscale;
    int L_lineopt;
    
    /* Stress options*/
    int Calc_stress;
    int Calc_pres;

    /* Linear solver*/
    int Poisson_solver;

    /* Eigenvalue Problem */
    int StandardEigenFlag;

    /* Domain description */
    double range_x;
    double range_y;
    double range_z;
    double latvec_scale_x; // scaling factor for the 1st latvec
    double latvec_scale_y; // scaling factor for the 2nd latvec
    double latvec_scale_z; // scaling factor for the 3rd latvec
    double LatVec[9];
    
    /* discretization */
    double mesh_spacing; // finite-difference mesh spacing
    double ecut;         // ecut used in plane-wave codes, equivalent to mesh-spacing in real-space codes

    /* k-points */
    double kptshift[3]; // k point shift in each direction
    
    /* Iterations: tolerances and max_iters */
    double target_energy_accuracy; // target energy accuracy
    double target_force_accuracy;  // target force accuracy
    double TOL_SCF;     // SCF tolerance
    double TOL_RELAX;   // Relaxation tolerance
    double TOL_POISSON; // Poisson tolerance
    double TOL_LANCZOS; // Lanczos tolerance
    double TOL_PSEUDOCHARGE;    // tolerance for calculating 
                                // pseudocharge density radius
    double TOL_PRECOND;  // tolerance for real-space preconditioner in SCF

    // preconditioner for SCF
    double precond_kerker_kTF;
    double precond_kerker_thresh;
    double precond_kerker_kTF_mag;
    double precond_kerker_thresh_mag;
    double precond_resta_q0;
    double precond_resta_Rs;

    double REFERENCE_CUTOFF;
    int REFERENCE_CUTOFF_FAC;
    
    /* System description */
    double Beta;        // electronic smearing (1/k_B*T) [1/Ha]
    double elec_T;      // electronic temperature in Kelvin (only one of elec_T and Beta should be specified)
     
    /* Mixing */
    double MixingParameter;
    double MixingParameterSimple;
    double MixingParameterMag;  // mixing parameter for the magnetization density, denoted as beta_mag
    double MixingParameterSimpleMag;  // mixing parameter for the magnetization density in simple mixing step, often denoted as omega_mag
    
    /* MD options */
    double MD_dt;        // MD time step
    double ion_T;        // Ionic temperature in Kelvin = initial temp of thermostat in NVT_NH
    double thermos_Tf;   // Final temperature of the thermostat
    double qmass;        // mass parameter of Nose Hoover thermostat

    int NPT_NHnnos;            // number of thermostat variables in NPT_NH
    int NPTscaleVecs[3];       // which lattice vector can be rescaled?
    int NPTconstraintFlag; // confinement on side length of cell. none: no length confinement (default); 1: a:b keeps unchanged; 2: a:c keeps unchanged; 
    // 3: a:c keeps unchanged; 4: a:b:c keeps unchanged, isotropic expansion. It is only available for NPT_NP.
    double NPT_NHqmass[L_QMASS];// qmass used in NPT_NH
    double NPT_NHbmass;        // Bmass used in NPT_NH
    double prtarget;     // Target pressure of barostatic system, UNIT on input file is GPa
    double NPT_NP_qmass; // qmass used in NPT_NP
    double NPT_NP_bmass; // Bmass used in NPT_NP
    
    /* Walltime */
    double TWtime;
    
    /* Relax options */
    double NLCG_sigma;
    double L_finit_stp;
    double L_maxmov;
    double L_icurv;
    double FIRE_dt;
    double FIRE_mass;
    double FIRE_maxmov;
    
    /* Cell relaxation */
    double max_dilatation;
    double TOL_RELAX_CELL;

    /* eigensolver */
    double eig_paral_orfac; // specifies which eigenvectors should be reorthogonalized when the "expert" parallel
                            // eigensolver p?syevx or p?sygvx is used.
    int eig_paral_maxnp;           // max number of processes for eigenvalue solver

    /* Method options */
    char MDMeth[32];
    char RelaxMeth[32];
    
    char XC[32];        // exchage-correlation approx. name

    /* DFT-D3 options */
    int d3Flag;
    double d3Rthr;        // cutoff radius for calculating d3 energy correction
    double d3Cn_thr;      // cutoff radius for calculating CN parameter of every atom

    /* Exact Exchange */
    double TOL_FOCK;        // Exact exchange potential option
    double TOL_SCF_INIT;    // Exact exchange potential option
    int MAXIT_FOCK;         // Maximum number of iterations for Hartree-Fock outer loop
    int MINIT_FOCK;         // Minimum number of iterations for Hartree-Fock outer loop    
    int ExxAcc;             // Flag for enable accelerating method for either low-T or high-T
    int ExxMemBatch;       // Option for speed or memory efficiency when using ACE operator
    int EeeAceValState;    // Number of extra unoccupied states in constructing ACE operator
    int EXXDownsampling[3]; // Downsampling info 
    int ExxDivFlag;        // Method for integrable singularity 
    double hyb_range_fock;  // hybrid short range for fock operator 
    double hyb_range_pbe;   // hybrid short range for exchange correlation 
    double exx_frac;        // hybrid mixing coefficient    
    int ExxMethod;          // method for solving poissons equation, kronecker product way or FFT

    /* DFT+U */
    int is_hubbard;         // DFT+U flag

    /* SQ methods */
    int sqAmbientFlag;             // Flag of SQ method
    // int SQ_highT_gauss_mem;       // Memory option for gauss quadrature 
    int SQ_highT_hybrid_gauss_mem;// Memory option for gauss quadrature for SQ
    int SQ_npl_g;           // Degree of polynomial (should be a multiple of 4) for Gauss Quadrature
    double SQ_rcut;         // Truncation or localization radius
    double SQ_tol_occ;      // Tolerance for occupation corresponding to maximum eigenvalue
    int npNdx_SQ;           // number of processes for paral. over domain in x-dir
    int npNdy_SQ;           // number of processes for paral. over domain in y-dir
    int npNdz_SQ;           // number of processes for paral. over domain in z-dir
    int sqHighTFlag;
    
    /* OFDFT */
    int OFDFTFlag;
    double OFDFT_tol;
    double OFDFT_lambda;

    /* cyclix */
    double twist;
    
    /* File names */
    char filename[L_STRING]; 
    char filename_out[L_STRING];
    
    char SPARCROOT[L_STRING]; // SPARC root directory


    /* Band structure plot*/
    int n_kpt_line;
    double kredx[L_kpoint],kredy[L_kpoint],kredz[L_kpoint];
    int BandStructFlag;
    int kpt_per_line;
    char InDensTCubFilename[L_STRING];
    char InDensUCubFilename[L_STRING];
    char InDensDCubFilename[L_STRING]; 
    int densfilecount;
    int readInitDens; // flag for reading inital density    

    /* Socket interface
       Please keep the socket interface as the last block in the SPARC_INPUT_OBJ
       definition. All new features should be added before this block
     */
#ifdef USE_SOCKET
    int SocketFlag;             // boolean value to indicate whether to switch on socket support at all
    char socket_host[L_STRING]; // socket file descriptor
    int socket_port;            // socket port
    int socket_inet;            // boolean value to indicate whether to use inet socket
    int socket_max_niter;           // max number of iterations
#endif
}SPARC_INPUT_OBJ;


/**
 * @brief   This structure type is for reading atom info. and broadcasting. 
 *           
 *          This structure contains part of the members of SPARC_OBJ, which
 *          are atom informations such as mass, atom type and pseudopotential
 *          informations. Some of these members are dynamically allocated
 *          arrays, which have to be treated differently. 
 *
 */
//typedef struct _SPARC_ATOM_OBJ{ 
//    int Ntypes;             // number of atom types
//    int is_default_psd;     // flag for reading pseudopotential
//    char filename[L_STRING];     // filename parsed by commandline

//    int n_atom;             // total number of atoms
//    int *localPsd;          // respective local component of pseudopotential for 
//                            // each type of atom. 0 for s, 1 for p, 2 for d, 3 for f
//                         
//    double *Mass;           // atomic mass (for MD)

//    char *atomType;         // atom type name for every type of atom    
//    int *Znucl;             // valence charge of each type
//    int *nAtomv;            // number of atoms of each type
//    char *psdName;          // pseudopotential file name for each type of atom
//    
//    double *atom_pos;       // atom positions
//    int *mvAtmConstraint;   // atom relax constraint, 1/0 -- move/don't move atom in this DOF
//    
//    PSD_OBJ *psd;           // struct array storing pseudopotential info.


//}SPARC_ATOM_OBJ;

#endif // ISDDFT_H







