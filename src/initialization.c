/**
 * @file    initialization.c
 * @brief   This file contains the functions for initialization.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 *          Alfredo Metere (GPU support), Lawrence Livermore National Laboratory <metere1@llnl.gov>, <alfredo.metere@xsilico.com>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <assert.h>
// this is for checking existence of files
# include <unistd.h>
#include "initialization.h"
#include "readfiles.h"
#include "nlocVecRoutines.h"
#include "electrostatics.h"
#include "tools.h"
#include "eigenSolver.h" // Mesh2ChebDegree, init_GTM_CheFSI()
#include "eigenSolverKpt.h"  // init_GTM_CheFSI_kpt()
#include "parallelization.h"
#include "isddft.h"
#include "d3initialization.h"
#include "vdWDFinitialization.h"
#include "vdWDFgenerateKernelSpline.h"
#include "MGGAinitialization.h"
#include "exactExchangeInitialization.h"
#include "spinOrbitCoupling.h"
#include "sqInitialization.h"
#include "sqParallelization.h"

#define TEMP_TOL 1e-12

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

#define N_MEMBR 167



/**
 * @brief   Prints usage of SPARC through command line.
 */
void print_usage() {
    printf("\n");
    printf("USAGE:\n"); 
    printf("    mpirun -np <nproc> {SPARCROOT}/lib/sparc -name <filename>\n");
    printf("\n");
    printf("    {SPARCROOT} is the location of the SPARC folder\n");
    printf("\n");
    printf("REQUIRED ARGUMENT:\n");
    printf("    -name <filename>\n");
    printf("           The filename shared by .inpt file and .ion\n");
    printf("           file (without extension)\n");
    printf("\n");
    printf("OPTIONS: \n");
    printf("    -h, --help\n");
    printf("           Display help (from command line).\n");
    printf("    -n <number of Nodes>\n");
    printf("    -c <number of CPUs per node>\n");
    printf("    -a <number of Accelerators (e.g., GPUs) per node>\n");
    printf("\n");
    printf("EXAMPLE:\n");
    printf("\n");
    printf("    mpirun -np 8 {SPARCROOT}/lib/sparc -name test\n");
    printf("\n");
    printf("    The example command runs sparc with 8 cores, with input file named\n");
    printf("    test.inpt, and ion file named test.ion.\n");
    printf("\n");
    printf("NOTE: \n");
    printf("    This is a short description of the usage of SPARC. For a detailed \n");
    printf("    discription, refer to the manual online at\n");
    printf("\n");
    printf("        https://github.com/SPARC-X/SPARC/tree/master/doc \n");
    printf("\n");
}



/**
 * @brief   Performs necessary initialization.
 */
void Initialize(SPARC_OBJ *pSPARC, int argc, char *argv[]) {
#ifdef DEBUG
    double t1,t2;
#endif
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Request req;

    // these two structs are for reading info. and broadcasting
    SPARC_INPUT_OBJ SPARC_Input;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    /* Create new MPI struct datatype SPARC_INPUT_MPI (for broadcasting) */
    MPI_Datatype SPARC_INPUT_MPI;
    SPARC_Input_MPI_create(&SPARC_INPUT_MPI);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nCreating SPARC_INPUT_MPI datatype took %.3f ms\n",(t2-t1)*1000);
#endif

    if (rank == 0) {
#ifdef DEBUG
        printf("Initializing ...\n");
        t1 = MPI_Wtime();
#endif
        // check input arguments and read filename
        check_inputs(&SPARC_Input, argc, argv); 

#ifdef DEBUG
        t2 = MPI_Wtime();
        printf("\nChecking inputs parsed by commandline took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
#endif
        set_defaults(&SPARC_Input, pSPARC); // set default values

#ifdef DEBUG
        t2 = MPI_Wtime();
        printf("\nSet default values took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
#endif
        read_input(&SPARC_Input, pSPARC); // read input file

#ifdef DEBUG
        t2 = MPI_Wtime();
        printf("\nReading input file took %.3f ms\n",(t2-t1)*1000);
#endif

        // broadcast the parameters read from the input files
        MPI_Bcast(&SPARC_Input, 1, SPARC_INPUT_MPI, 0, MPI_COMM_WORLD);

#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        read_ion(&SPARC_Input, pSPARC); // read ion file

#ifdef DEBUG
        t2 = MPI_Wtime();
        printf("\nReading ion file took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
#endif

        // broadcast Ntypes read from ion file
        MPI_Ibcast(&pSPARC->Ntypes, 1, MPI_INT, 0, MPI_COMM_WORLD, &req);

        // disable default pseudopotential path and name
        if (pSPARC->is_default_psd) {
            printf("\n"
                   "Default path and names of pseudopotentials are currently disabled,\n"
                   "Please specify pseudopotential filename!\n");
            exit(EXIT_FAILURE);
        }

        //read_pseudopotential_TM(&SPARC_Input, pSPARC); // read TM format pseudopotential file
        read_pseudopotential_PSP(&SPARC_Input, pSPARC); // read psp format pseudopotential file

#ifdef DEBUG
        t2 = MPI_Wtime();
        printf("\nReading pseudopotential file took %.3f ms\n",(t2-t1)*1000);
#endif

    } else {

#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        MPI_Bcast(&SPARC_Input, 1, SPARC_INPUT_MPI, 0, MPI_COMM_WORLD);
#ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Broadcasting the input parameters took %.3f ms\n",(t2-t1)*1000);
#endif
        // broadcast Ntypes read from ion file
        MPI_Ibcast(&pSPARC->Ntypes, 1, MPI_INT, 0, MPI_COMM_WORLD, &req);
    }
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    MPI_Type_free(&SPARC_INPUT_MPI); // free the new MPI datatype

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nFreeing SPARC_INPUT_MPI datatype took %.3f ms\n",(t2-t1)*1000);
    t1 = MPI_Wtime();
#endif

    // Ntypes is no longer read from .inpt file
    // pSPARC->Ntypes = SPARC_Input.Ntypes;
    // make sure Ntypes is broadcasted
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    // broadcast SPARC members regarding Atom info. using MPI_Pack & MPI_Unpack
    bcast_SPARC_Atom(pSPARC);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Broadcasting Atom info. using MPI_Pack & MPI_Unpack in SPARC took %.3f ms\n",(t2-t1)*1000);

    t1 = MPI_Wtime();
#endif

    // copy the data read from input files into struct SPARC
    SPARC_copy_input(pSPARC,&SPARC_Input);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nrank = %d, Copying data from SPARC_Input into SPARC & set up subcomm took %.3f ms\n",rank,(t2-t1)*1000);
#endif

    // set up sub-communicators
    if (pSPARC->SQFlag == 1) {
        Setup_Comms_SQ(pSPARC);
    } else {
        Setup_Comms(pSPARC);

        #ifdef USE_DP_SUBEIG
        #if !defined(USE_MKL) && !defined(USE_SCALAPACK)
        if (pSPARC->useLAPACK == 0)
        {
            #ifdef DEBUG
            if (rank == 0) printf("[WARNING] ScaLAPACK is not compiled and Nstates > MAX_NS, subspace eigen-problem will be solved in sequential.\n");
            #endif
            pSPARC->useLAPACK = 1;
        }
        #endif

        // SPARCX_ACCEL_NOTE Need to add this. Make sure it is always within "#ifdef USE_DP_SUBEIG" branch
        // --- BEGIN. Alfredo Metere
        #ifdef ACCEL // Activating flag for using hardware acceleration at compile time.
        pSPARC->useACCEL = 1;
    //	#else
    //	pSPARC->useACCEL = 0;
        

        if (rank == 0) 
        {	
            char *hwaccel[2] = { "DISABLED", "ENABLED" };
            printf ("[INFO] Hardware acceleration is %s\n", hwaccel[pSPARC->useACCEL]);
        }
        #endif // ACCEL
        // --- END. Alfredo Metere

        pSPARC->DP_CheFSI     = NULL;
        pSPARC->DP_CheFSI_kpt = NULL;
        if (pSPARC->isGammaPoint) init_DP_CheFSI(pSPARC);
        else init_DP_CheFSI_kpt(pSPARC);
        #endif

        // calculate maximum number of processors for eigenvalue solver
        if (pSPARC->useLAPACK == 0) {
            if (pSPARC->eig_paral_maxnp < 0) {
                char RorC, SorG;
                RorC = (pSPARC->isGammaPoint) ? 'R' : 'C';
                SorG = 'G'; // standard or generalized eigenproblem
                pSPARC->eig_paral_maxnp = parallel_eigensolver_max_processor(pSPARC->Nstates, RorC, SorG);
            }
                
            int gridsizes[2] = {pSPARC->Nstates,pSPARC->Nstates}, ierr = 1, size_blacscomm = 0;
            if (pSPARC->blacscomm != MPI_COMM_NULL)
                MPI_Comm_size(pSPARC->blacscomm, &size_blacscomm);
            SPARC_Dims_create(min(size_blacscomm,pSPARC->eig_paral_maxnp), 2, gridsizes, 1, pSPARC->eig_paral_subdims, &ierr);
            if (ierr) pSPARC->eig_paral_subdims[0] = pSPARC->eig_paral_subdims[1] = 1;

        #ifdef DEBUG
            if (rank == 0) printf("\nMaximun number of processors for eigenvalue solver is %d\n", pSPARC->eig_paral_maxnp);
            if (rank == 0) printf("The dimension of subgrid for eigen sovler is (%d x %d).\n", 
                                    pSPARC->eig_paral_subdims[0], pSPARC->eig_paral_subdims[1]);
        #endif
        }
    }
    
    // estimate memory usage here

    // Allocate memory space for Exx methods
    if (pSPARC->usefock == 1) {
        init_exx(pSPARC);
    }

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    // calculate spline derivatives for interpolation
    Calculate_SplineDerivRadFun(pSPARC);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nCalculate_SplineDerivRadFun took %.3f ms\n",(t2-t1)*1000);
#endif
    
    if (pSPARC->SQFlag == 1) {
        init_SQ(pSPARC);
    } else {
        // calculate indices for storing nonlocal inner product
        CalculateNonlocalInnerProductIndex(pSPARC);
        if (pSPARC->SOC_Flag == 1)
            CalculateNonlocalInnerProductIndexSOC(pSPARC);
    }

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // calculate pseudocharge density cutoff ("rb")
    Calculate_PseudochargeCutoff(pSPARC);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nCalculating rb for all atom types took %.3f ms\n",(t2-t1)*1000);
#endif

    // initialize DFT-D3
    if (pSPARC->d3Flag == 1) {
        if ((strcmpi(pSPARC->XC, "GGA_PBE") != 0) && (strcmpi(pSPARC->XC, "GGA_PBEsol") != 0) && (strcmpi(pSPARC->XC, "GGA_RPBE") != 0) && (strcmpi(pSPARC->XC, "PBE0") != 0) && (strcmpi(pSPARC->XC, "HSE") != 0)) {
            if (rank == 0) 
                printf(RED "ERROR: Cannot find D3 coefficients for this functional. DFT-D3 correction calculation canceled!\n" RESET);
            exit(EXIT_FAILURE);
        }
        else {
            set_D3_coefficients(pSPARC); // this function is moved from electronicGroundState.c
        }
    }

    // initialize vdW-DF
    if (pSPARC->vdWDFFlag != 0) {
        if ((pSPARC->vdWDFKernelGenFlag) && (rank == 0)) {
            vdWDF_generate_kernel(pSPARC->filename); // if there is no file of kernel function and d2Spline, then generate one
        }
        MPI_Barrier(MPI_COMM_WORLD);
        vdWDF_initial_read_kernel(pSPARC); // read kernel function and 2nd derivative of spline functions
        // printf("rank %d, d2 of kernel function vdWDFd2Phidk2[2][4]=%.9e\n", rank, pSPARC->vdWDFd2Phidk2[2][4]); // to verify it
    }

    // initialize metaGGA
    if(pSPARC->mGGAflag == 1) {
        initialize_MGGA(pSPARC);
    }

    // estimate memory usage
    pSPARC->memory_usage = estimate_memory(pSPARC);

    // write initialized parameters into output file
    if (rank == 0) {
        write_output_init(pSPARC);
    }        
}



/**
 * @brief   Check input arguments and read filename.
 */
void check_inputs(SPARC_INPUT_OBJ *pSPARC_Input, int argc, char *argv[]) {
#ifdef DEBUG
    printf("Checking input arguments parsed by command line.\n");
#endif
    int i;
    char *pch, name_flag = 'N';
    
    // extract SPARCROOT from the executable's name
    char *libbase = argv[0];
    pch = strrchr(libbase,'/'); // find last occurrence of '/'
    // printf (RED "Last occurence of '/' found at %ld \n" RESET,pch-libbase+1);
    if (pch == NULL) {
        // in fact, unless SAPRCROOT/lib is added to $(PATH), ./sparc is needed
        // strcpy(pSPARC_Input->SPARCROOT,".."); // in case '/' is not found
        snprintf(pSPARC_Input->SPARCROOT, L_STRING, "..");
    } else {
        memcpy(pSPARC_Input->SPARCROOT, libbase, pch-libbase);
        pSPARC_Input->SPARCROOT[pch-libbase] = '\0';
        if ( strcmp(pSPARC_Input->SPARCROOT + (int)(pch-libbase) - 4, "/lib") == 0) {
            pSPARC_Input->SPARCROOT[pch-libbase-4] = '\0'; // directly truncate string
        } else {
            strcat(pSPARC_Input->SPARCROOT,"/..");
        }
    }

    // save input filename
    memset(pSPARC_Input->filename, '\0', sizeof(pSPARC_Input->filename));
    pSPARC_Input->num_node = 0;
    pSPARC_Input->num_cpu_per_node = 0;
    pSPARC_Input->num_acc_per_node = 0;
    for (i = 1; i < argc-1; i++) {
        if (strcmp(argv[i],"--help") == 0 || strcmp(argv[i],"-h") == 0) {
            print_usage();
            exit(EXIT_FAILURE);
        }
        if (strcmp(argv[i],"-name") == 0) {
            name_flag = 'Y';
            simplifyPath(argv[i+1], pSPARC_Input->filename, L_STRING);
            // break;
        }
        if (strcmp(argv[i],"-n") == 0) {
            pSPARC_Input->num_node = atoi(argv[i+1]);
        }
        if (strcmp(argv[i],"-c") == 0) {
            pSPARC_Input->num_cpu_per_node = atoi(argv[i+1]);
        }
        if (strcmp(argv[i],"-a") == 0) {
            pSPARC_Input->num_acc_per_node = atoi(argv[i+1]);
        }
    }
    
    if (name_flag != 'Y') {
        print_usage();
        exit(EXIT_FAILURE);
    }
}



/**
 * @brief   Set default values for SPARC members.
 */
void set_defaults(SPARC_INPUT_OBJ *pSPARC_Input, SPARC_OBJ *pSPARC) {
    /* init strings to null */
    memset(pSPARC_Input->filename_out, '\0', sizeof(pSPARC_Input->filename_out));
    memset(pSPARC_Input->SPARCROOT,    '\0', sizeof(pSPARC_Input->SPARCROOT));
    memset(pSPARC_Input->MDMeth,       '\0', sizeof(pSPARC_Input->MDMeth));
    memset(pSPARC_Input->RelaxMeth,    '\0', sizeof(pSPARC_Input->RelaxMeth));
    memset(pSPARC_Input->XC,           '\0', sizeof(pSPARC_Input->XC));

    /* default file names (unless otherwise supplied) */
    snprintf(pSPARC_Input->filename_out, L_STRING, "%s", pSPARC_Input->filename);
    snprintf(pSPARC_Input->SPARCROOT, L_STRING, "%s", "UNDEFINED");

    /* Parallelizing parameters */
    pSPARC_Input->npspin = 0;         // number of processes for paral. over k-points
    pSPARC_Input->npkpt = 0;          // number of processes for paral. over k-points
    pSPARC_Input->npband = 0;         // number of processes for paral. over bands
    pSPARC_Input->npNdx = 0;          // number of processes for paral. over domain in x-dir
    pSPARC_Input->npNdy = 0;          // number of processes for paral. over domain in y-dir
    pSPARC_Input->npNdz = 0;          // number of processes for paral. over domain in z-dir
    pSPARC_Input->npNdx_phi = 0;      // number of processes for calculating phi in paral. over domain in x-dir
    pSPARC_Input->npNdy_phi = 0;      // number of processes for calculating phi in paral. over domain in y-dir
    pSPARC_Input->npNdz_phi = 0;      // number of processes for calculating phi in paral. over domain in z-dir
    pSPARC_Input->eig_serial_maxns = 1500; // maximum Nstates for solving the subspace eigenproblem in serial by default,
                                      // for Nstates greater than this value, a parallel methods will be used instead, unless 
                                      // ScaLAPACK is not compiled or useLAPACK is turned off.
    pSPARC_Input->eig_paral_blksz = 128; // block size for distributing the subspace eigenproblem
    pSPARC_Input->eig_paral_orfac = 0.0; // no reorthogonalization when using p?syevx or p?sygvx
    pSPARC_Input->eig_paral_maxnp = -1;  // using default value from linear fitting model

    /* default spin_typ */
    pSPARC_Input->spin_typ = 0;       // Default is spin unpolarized calculation

    /* default MD and Relaxation options */
    pSPARC_Input->MDFlag = 0;                 // default: no MD
    strncpy(pSPARC_Input->MDMeth,"NVT_NH",sizeof(pSPARC_Input->MDMeth));       // default MD method: NVT
    pSPARC_Input->RelaxFlag = 0;              // default: no relaxation
    strncpy(pSPARC_Input->RelaxMeth,"LBFGS",sizeof(pSPARC_Input->RelaxMeth));   // default relax method: LBFGS

    pSPARC_Input->RestartFlag = 0;            // default: no retart
    /* default finite difference scheme info. */
    pSPARC_Input->order = 12;                 // default FD order: 12th

    /* default poisson solver */
    pSPARC_Input->Poisson_solver = 0;          // default AAR solver
    /* Iterations: tolerances and max_iters */
    pSPARC_Input->FixRandSeed = 0;            // default flag for fixing random numbers for MPI paralllelization
    pSPARC_Input->accuracy_level = -1;        // default accuracy level (2 - 'medium', 1e-3 in energy and forces)
    pSPARC_Input->scf_err_type = 0;           // default scf error definition: relative error in rho or veff
    pSPARC_Input->MAXIT_SCF = 100;            // default maximum number of SCF iterations
    pSPARC_Input->MINIT_SCF = 3;              // default minimum number of SCF iterations
    pSPARC_Input->MAXIT_POISSON = 3000;       // default maximum number of iterations for Poisson solver
    pSPARC_Input->Relax_Niter = 300;          // default maximum number of relaxation iterations, for RelaxFlag = 1 only
    pSPARC_Input->target_energy_accuracy = -1.0; // default target energy accuracy
    pSPARC_Input->target_force_accuracy = -1.0;  // default target force accuracy
    pSPARC_Input->TOL_SCF = -1.0;             // default SCF tolerance
    pSPARC_Input->TOL_RELAX = 5e-4;           // default Relaxation tolerance
    pSPARC_Input->TOL_POISSON = -1.0;         // default Poisson solve tolerance (will be set up later)
    pSPARC_Input->TOL_LANCZOS = 1e-2;         // default Lanczos tolerance
    pSPARC_Input->TOL_PSEUDOCHARGE = -1.0;    // default tolerance for calculating pseudocharge density radius (will be set up later)
    pSPARC_Input->TOL_PRECOND = -1.0;         // default Kerker tolerance will be set up later, depending on the mesh size
    pSPARC_Input->precond_kerker_kTF = 1.0;    // Thomas-Fermi screening length in the Kerker preconditioner
    pSPARC_Input->precond_kerker_thresh = 0.1; // Threshold for the truncated Kerker preconditioner
    pSPARC_Input->precond_kerker_kTF_mag = 1.0;    // Thomas-Fermi screening length in the Kerker preconditioner
    pSPARC_Input->precond_kerker_thresh_mag = 0.1; // Threshold for the truncated Kerker preconditioner
    pSPARC_Input->precond_resta_q0 = 1.36;
    pSPARC_Input->precond_resta_Rs = 2.76;

    pSPARC_Input->REFERENCE_CUTOFF = 0.5;     // default reference cutoff radius for nonlocal pseudopotential

    /* default mixing */
    pSPARC_Input->MixingVariable = -1;        // default mixing variabl (will be set to 'density' mixing)
    pSPARC_Input->MixingPrecond = -1;         // default mixing preconditioner (will be set later)
    pSPARC_Input->MixingPrecondMag = -1;      // default mixing preconditioner for magnetization density/potential (will be set later)
    pSPARC_Input->MixingParameter = 0.3;      // default mixing parameter
    pSPARC_Input->MixingParameterSimple = -1.0;     // default mixing parameter for simple mixing step (will be set up later)
    pSPARC_Input->MixingParameterMag = -1.0;   // default mixing parameter for magnetization density/potential
    pSPARC_Input->MixingParameterSimpleMag = -1.0; // default mixing parameter for magnetization density/potential in simple mixing step (will be set up later)
    pSPARC_Input->MixingHistory = 7;          // default mixing history
    pSPARC_Input->PulayFrequency = 1;         // default Pulay frequency
    pSPARC_Input->PulayRestartFlag = 0;       // default Pulay restart flag
    pSPARC_Input->precond_fitpow = 2;         // default fit power for the real-space preconditioner

    /* default k-points info. */
    pSPARC_Input->Kx = 1;
    pSPARC_Input->Ky = 1;
    pSPARC_Input->Kz = 1;
    pSPARC_Input->Nkpts = 1;
    pSPARC_Input->NkptsGroup = 1; // unused
    pSPARC_Input->kctr = 1;       // unused
    pSPARC_Input->kptshift[0] = 0.0;
    pSPARC_Input->kptshift[1] = 0.0;
    pSPARC_Input->kptshift[2] = 0.0;

    // default lattice vector array
    for (int i = 0; i < 9; i++){
        if(i % 4 == 0)
            pSPARC_Input->LatVec[i] = 1.0;
        else
            pSPARC_Input->LatVec[i] = 0.0;
    }

    /* discretization */
    pSPARC_Input->mesh_spacing = -1.0;
    pSPARC_Input->ecut = -1.0;
    pSPARC_Input->numIntervals_x = -1;
    pSPARC_Input->numIntervals_y = -1;
    pSPARC_Input->numIntervals_z = -1;

    /* default system info. */
    pSPARC_Input->range_x = -1.0;
    pSPARC_Input->range_y = -1.0;
    pSPARC_Input->range_z = -1.0;
    pSPARC_Input->Flag_latvec_scale = 0;
    pSPARC_Input->latvec_scale_x = -1.0;
    pSPARC_Input->latvec_scale_y = -1.0;
    pSPARC_Input->latvec_scale_z = -1.0;
    pSPARC_Input->BC = -1;                    // default BC will be set up after reading input
	pSPARC_Input->BCx = -1;
	pSPARC_Input->BCy = -1;
	pSPARC_Input->BCz = -1;
    // pSPARC_Input->Beta = 1000;                // electronic smearing (1/(k_B*T)) [1/Ha]
    // pSPARC_Input->elec_T = 315.7751307269723; // default electronic temperature in Kelvin
    pSPARC_Input->Beta = -1.0;                // electronic smearing (1/(k_B*T)) [1/Ha], will be specified later
    pSPARC_Input->elec_T = -1.0;              // default electronic temperature in Kelvin, will be specified later
    pSPARC_Input->Ntypes = -1;                // default number of atom types
    pSPARC_Input->Nstates = -1;               // default number of states
    pSPARC_Input->NetCharge = 0;              // default net charge: 0
    //strncpy(pSPARC_Input->XC, "LDA",sizeof(pSPARC_Input->XC));          // default exchange-correlation approx: LDA
    strncpy(pSPARC_Input->XC, "UNDEFINED",sizeof(pSPARC_Input->XC));      // default: UNDEFINED

    /* default Chebyshev filter */
    pSPARC_Input->ChebDegree = -1;            // default chebyshev polynomial degree (will be automatically found based on spectral width)
    pSPARC_Input->CheFSI_Optmz = 0;           // default is off
    pSPARC_Input->chefsibound_flag = 0;       // default is to find bound using Lanczos on H in the first SCF of each MD/Relax only
    pSPARC_Input->rhoTrigger = 4;              // default step to start updating electron density, later will be subtracted by 1

    /* default smearing */
    pSPARC_Input->elec_T_type = 1;            // default smearing type: 1 - gaussian smearing (the other option is 0 - fermi-dirac)

    /* Default MD parameters */
    pSPARC_Input->MD_dt = 1.0;                // default MD time step: 1.0 femtosecond
    pSPARC_Input->MD_Nstep = 10000000;        // default MD maximum steps
    pSPARC_Input->ion_T = -1.0;               // default ionic temperature in Kelvin
    pSPARC_Input->thermos_Tf = -1.0;          // Final temperature of the thermostat
    pSPARC_Input->ion_elec_eqT = 0;           // default ionic and electronic temp will be different throughout MD
    pSPARC_Input->ion_vel_dstr = 2;           // default initial velocity distribution is Maxwell-Boltzmann
    pSPARC_Input->ion_vel_dstr_rand = 0;      // default initial velocity are fixed (different runs give the same answer)
    pSPARC_Input->qmass = 40.0 * CONST_FS2ATU; // default values of thermostat parameter mass (a.u.)
    pSPARC_Input->TWtime = 1000000000;        // default value of walltime in min
    pSPARC_Input->NPTscaleVecs[0] = 1; 
    pSPARC_Input->NPTscaleVecs[1] = 1; 
    pSPARC_Input->NPTscaleVecs[2] = 1;        // default lattice vectors to be rescaled in NPT
    pSPARC_Input->NPT_NHnnos = 0;                   // default amount of thermo variable for NPT_NH. If MDMeth is this but nnos is 0, program will stop
    for (int subscript_NPTNH_qmass = 0; subscript_NPTNH_qmass < L_QMASS; subscript_NPTNH_qmass++){
        pSPARC_Input->NPT_NHqmass[subscript_NPTNH_qmass] = 0.0;
    }                                         // default mass of thermo variables for NPT_NH. If MDMeth is this but one of qmass is 0, program will stop
    pSPARC_Input->NPT_NHbmass = 0.0;          // default mass of baro variable for NPT_NH. If MDMeth is this but bmass is 0, program will stop
    pSPARC_Input->prtarget = 0.0;             // default target pressure for NPT_NH.

    pSPARC_Input->NPT_NP_qmass = 0.0;         // default mass of thermo variables for NPT_NP. If MDMeth is this but qmass is 0, program will stop
    pSPARC_Input->NPT_NP_bmass = 0.0;         // default mass of thermo variables for NPT_NP. If MDMeth is this but bmass is 0, program will stop

    /* Default Relax parameters */
    pSPARC_Input->NLCG_sigma = 0.5;
    pSPARC_Input->L_history = 20;
    pSPARC_Input->L_finit_stp = 5e-3;
    pSPARC_Input->L_maxmov = 0.2;
    pSPARC_Input->L_autoscale = 1;
    pSPARC_Input->L_lineopt = 1;
    pSPARC_Input->L_icurv = 1.0;
    pSPARC_Input->FIRE_dt = 1.0;
    pSPARC_Input->FIRE_mass = 1.0;
    pSPARC_Input->FIRE_maxmov = 0.2;

    /* Default cell relaxation parameters*/
    pSPARC_Input->max_dilatation = 1.06;      // maximum lattice dilatation
    pSPARC_Input->TOL_RELAX_CELL = 1e-2;      // in GPa (all periodic)

    /* Default DFT-D3 correction */
    pSPARC_Input->d3Flag = 0;
    pSPARC_Input->d3Rthr = 1600.0;
    pSPARC_Input->d3Cn_thr = 625.0;

    /* Default vdW-DF option */
    pSPARC_Input->vdWDFKernelGenFlag = 1;

    /* Default stress flags*/
    pSPARC_Input->Calc_stress = 0;
    pSPARC_Input->Calc_pres = 0;

    /* print options */
    pSPARC_Input->Verbosity = 1;              // Flag for specifying the amount of output in .out file
    pSPARC_Input->PrintForceFlag = 1;         // flag for printing forces
    pSPARC_Input->PrintAtomPosFlag = 1;       // flag for printing atomic positions
    pSPARC_Input->PrintAtomVelFlag = 1;       // flag for printing atom velocities in case of MD/relax
    pSPARC_Input->PrintElecDensFlag = 0;      // flag for printing final electron density
    pSPARC_Input->PrintEigenFlag = 0;         // Flag for printing final eigenvalues and occupations
    pSPARC_Input->PrintMDout = 1;             // Flag for printing MD output in a .aimd file
    pSPARC_Input->PrintRelaxout = 1;          // Flag for printing relax output in a .relax file
    pSPARC_Input->Printrestart = 1;           // Flag for printing output needed for restarting a simulation
    pSPARC_Input->Printrestart_fq = 1;        // Steps after which the output is written in the restart file
    pSPARC_Input->PrintPsiFlag[0] = 0;        // Flag for printing Kohn-Sham orbitals
    for (int i = 1; i < 7; i++) 
        pSPARC_Input->PrintPsiFlag[i] = -1;   // defualt spin, kpt, band start and end index for printing psi
    pSPARC_Input->PrintEnergyDensFlag = 0;    // flag for printing kinetic energy density
    
    /* Default pSPARC members */
    pSPARC->is_default_psd = 0;               // default pseudopotential path is disabled

    /* Default Exact exchange potential */    
    pSPARC_Input->TOL_FOCK = -1.0;            // default tolerance for Fock operator 
    pSPARC_Input->TOL_SCF_INIT = -1.0;        // default tolerance for first PBE SCF
    pSPARC_Input->MAXIT_FOCK = 20;            // default maximum number of iterations for Hartree-Fock outer loop
    pSPARC_Input->MINIT_FOCK = 2;             // default minimum number of iterations for Hartree-Fock outer loop
    pSPARC_Input->EXXMeth_Flag = 0;           // default method to solve Poisson's equation of Exact Exchange in Fourier space
    pSPARC_Input->ACEFlag = 1;                // default setting for not using ACE operator
    pSPARC_Input->EXXMem_batch = 20;           // default setting for using high memory option
    pSPARC_Input->EXXACEVal_state = 3;        // default setting for using high memory option
    pSPARC_Input->EXXDownsampling[0] = 1;     // default setting for downsampling, using full k-points
    pSPARC_Input->EXXDownsampling[1] = 1;     // default setting for downsampling, using full k-points
    pSPARC_Input->EXXDownsampling[2] = 1;     // default setting for downsampling, using full k-points
    pSPARC_Input->EXXDiv_Flag = -1;           // default setting for singularity in exact exchange, default spherical trucation    
    pSPARC_Input->hyb_range_fock = 0.1587;    // default using VASP's HSE03 value
    pSPARC_Input->hyb_range_pbe = 0.1587;     // default using VASP's HSE03 value
    pSPARC_Input->exx_frac = -1;              // default exx_frac

    /* Default parameter for spin-orbit coupling */
    pSPARC->Nspinor = 1;
    pSPARC->SOC_Flag = 0;

    /* Default SQ method option */
    pSPARC_Input->SQFlag = 0;
    pSPARC_Input->SQ_typ_dm = 2;                // default using Gauss quadrature for density matrix
    pSPARC_Input->SQ_gauss_mem = 0;             // default not saving Lanczos vectors and eigenvectors 
    pSPARC_Input->SQ_npl_c = -1;
    pSPARC_Input->SQ_npl_g = -1;
    pSPARC_Input->SQ_EigshiftFlag = 0;
    pSPARC_Input->SQ_rcut = -1;
    pSPARC_Input->SQ_fac_g2c = 2.0;
    pSPARC_Input->SQ_tol_occ = 1e-6;
    pSPARC_Input->SQ_eigshift = 5;
    pSPARC_Input->npNdx_SQ = 0;
    pSPARC_Input->npNdy_SQ = 0;
    pSPARC_Input->npNdz_SQ = 0;
}



/**
 * @brief Broadcast Atom info. in SPARC using MPI_Pack & MPI_Unpack.
 */
void bcast_SPARC_Atom(SPARC_OBJ *pSPARC) {
    int i, l, rank, position, l_buff, Ntypes, n_atom, nproj, lmax_sum, size_sum, nprojsize_sum, nproj_sum;
    int *tempbuff, *is_r_uniformv, *pspxcv, *pspsocv, *lmaxv, *sizev, *pplv, *pplv_soc, *ppl_sdispl, *ppl_soc_sdispl;
    ppl_soc_sdispl = NULL, pplv_soc = NULL;
    char *buff;

#ifdef DEBUG
    double t1, t2;
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Ntypes = pSPARC->Ntypes;
    /* preparation for broadcasting the structure */
    lmaxv = (int *)malloc( Ntypes * sizeof(int) );
    sizev = (int *)malloc( Ntypes * sizeof(int) );
    is_r_uniformv = (int *)malloc( Ntypes * sizeof(int) );
    pspxcv = (int *)malloc( Ntypes * sizeof(int) );
    pspsocv = (int *)malloc( Ntypes * sizeof(int) );
    ppl_sdispl = (int *)malloc( (Ntypes+1) * sizeof(int) );
    tempbuff = (int *)malloc( (5*Ntypes+3) * sizeof(int) );
    assert(lmaxv != NULL && sizev != NULL && is_r_uniformv != NULL 
        && pspxcv!= NULL && pspsocv!= NULL && ppl_sdispl != NULL 
        && tempbuff != NULL);

    // send n_atom, lmax, size
    if (rank == 0) {
        // pack the info. into temp. buffer
        tempbuff[0] = pSPARC->n_atom;
        tempbuff[1] = pSPARC->Nspinor;
        tempbuff[2] = pSPARC->SOC_Flag;
        for (i = 0; i < Ntypes; i++) {
            lmaxv[i] = pSPARC->psd[i].lmax;
            sizev[i] = pSPARC->psd[i].size;
            pspxcv[i] = pSPARC->psd[i].pspxc;
            is_r_uniformv[i] = pSPARC->psd[i].is_r_uniform;
            pspsocv[i] = pSPARC->psd[i].pspsoc;
            //pplv[i] = pSPARC->psd[i].ppl;
            tempbuff[i+3] = lmaxv[i];
            tempbuff[i+Ntypes+3] = sizev[i];
            tempbuff[i+2*Ntypes+3] = pspxcv[i];
            tempbuff[i+3*Ntypes+3] = is_r_uniformv[i];
            tempbuff[i+4*Ntypes+3] = pspsocv[i];

            //tempbuff[i+2*Ntypes+1] = pplv[i];
        }
        MPI_Bcast( tempbuff, 5*Ntypes+3, MPI_INT, 0, MPI_COMM_WORLD);

        // pack psd[i].ppl[l] and bcast
        ppl_sdispl[0] = 0;
        for (i = 0; i < Ntypes; i++) {
            ppl_sdispl[i+1] = ppl_sdispl[i] + lmaxv[i] + 1;
        }
        pplv = (int *)malloc( ppl_sdispl[Ntypes] * sizeof(int) );
        for (i = 0; i < Ntypes; i++) {
            for (l = 0; l <= lmaxv[i]; l++) {
                pplv[ppl_sdispl[i]+l] = pSPARC->psd[i].ppl[l];
            }
        }
        MPI_Bcast( pplv, ppl_sdispl[Ntypes], MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        // allocate psd array for receiver process
        pSPARC->psd = (PSD_OBJ *)malloc(pSPARC->Ntypes * sizeof(PSD_OBJ));
        assert(pSPARC->psd != NULL);
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        MPI_Bcast( tempbuff, 5*Ntypes+3, MPI_INT, 0, MPI_COMM_WORLD);

#ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Bcast pre-info. took %.3f ms\n", (t2-t1)*1000);
#endif
        // unpack info.
        pSPARC->n_atom = tempbuff[0];
        pSPARC->Nspinor = tempbuff[1];
        pSPARC->SOC_Flag = tempbuff[2];
        for (i = 0; i < Ntypes; i++) {
            lmaxv[i] = tempbuff[i+3];
            sizev[i] = tempbuff[i+Ntypes+3];
            pspxcv[i] = tempbuff[i+2*Ntypes+3];
            is_r_uniformv[i] = tempbuff[i+3*Ntypes+3];
            pspsocv[i] = tempbuff[i+4*Ntypes+3];
            //pplv[i] = tempbuff[i+2*Ntypes+1];
            pSPARC->psd[i].lmax = lmaxv[i];
            pSPARC->psd[i].size = sizev[i];
            pSPARC->psd[i].pspxc = pspxcv[i];
            pSPARC->psd[i].is_r_uniform = is_r_uniformv[i];
            pSPARC->psd[i].pspsoc = pspsocv[i];
            //pSPARC->psd[i].ppl = pplv[i];
        }

        // bcast psd[i].ppl[l] and unpack
        ppl_sdispl[0] = 0;
        for (i = 0; i < Ntypes; i++) {
            ppl_sdispl[i+1] = ppl_sdispl[i] + lmaxv[i] + 1;
        }

        pplv = (int *)malloc( ppl_sdispl[Ntypes] * sizeof(int) );
        MPI_Bcast( pplv, ppl_sdispl[Ntypes], MPI_INT, 0, MPI_COMM_WORLD);

        for (i = 0; i < Ntypes; i++) {
            pSPARC->psd[i].ppl = (int *)malloc((lmaxv[i] + 1) * sizeof(int));
            for (l = 0; l <= lmaxv[i]; l++) {
                pSPARC->psd[i].ppl[l] = pplv[ppl_sdispl[i]+l];
            }
        }
    }
    n_atom = pSPARC->n_atom;

    if (pSPARC->SOC_Flag == 1) {
        ppl_soc_sdispl = (int *)malloc( (Ntypes+1) * sizeof(int) );
        // pack psd[i].ppl[l] and bcast
        ppl_soc_sdispl[0] = 0;
        for (i = 0; i < Ntypes; i++) {
            ppl_soc_sdispl[i+1] = ppl_soc_sdispl[i] + pspsocv[i] * lmaxv[i]; // l from 1 to lmax
        }
        pplv_soc = (int *)malloc( ppl_soc_sdispl[Ntypes] * sizeof(int) );
        
        if (rank == 0) {
            for (i = 0; i < Ntypes; i++) {
                if (!pspsocv[i]) continue;
                for (l = 1; l <= lmaxv[i]; l++) {
                    pplv_soc[ppl_soc_sdispl[i]+l-1] = pSPARC->psd[i].ppl_soc[l-1];
                }
            }
            MPI_Bcast( pplv_soc, ppl_soc_sdispl[Ntypes], MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Bcast( pplv_soc, ppl_soc_sdispl[Ntypes], MPI_INT, 0, MPI_COMM_WORLD);
            for (i = 0; i < Ntypes; i++) {
                pSPARC->psd[i].ppl_soc = (int *)malloc(lmaxv[i] * sizeof(int));
                for (l = 1; l <= lmaxv[i]; l++) {
                    pSPARC->psd[i].ppl_soc[l-1] = pplv_soc[ppl_soc_sdispl[i]+l-1];
                }
            }
        }
    }

    // allocate memory for buff, the extra 16*(Ntypes+3*n_atom) byte is spare memory
    lmax_sum = 0;
    size_sum = 0;
    nproj_sum = 0;
    nprojsize_sum = 0;
    for (i = 0; i < Ntypes; i++) {
        lmax_sum += lmaxv[i]+1;
        size_sum += sizev[i];
        // sizelmax_sum += ((lmaxv[i]+1) * sizev[i]);
        // ppllmax_sum += pplv[i] * (lmaxv[i]+1);
        nproj = 0;
        for (l = 0; l <= lmaxv[i]; l++) {
             nproj += pSPARC->psd[i].ppl[l];
        }
        nproj_sum += nproj;
        nprojsize_sum += nproj * sizev[i];
    }

    int nprojso, nprojso_sum, nprojsosize_sum;
    nprojso_sum = nprojsosize_sum = 0;
    if (pSPARC->SOC_Flag == 1) {
        for (i = 0; i < Ntypes; i++) {
            if (!pSPARC->psd[i].pspsoc) continue;
            nprojso = 0;
            for (l = 1; l <= lmaxv[i]; l++) {
                nprojso += pSPARC->psd[i].ppl_soc[l-1];
            }
            nprojso_sum += nprojso;
            nprojsosize_sum += nprojso * sizev[i];
        }
    }

    l_buff = (3*Ntypes + 3*n_atom + 4*size_sum + lmax_sum + nproj_sum + nprojsize_sum + nprojsosize_sum + nprojsosize_sum + n_atom) * sizeof(double)
             + (6*Ntypes + 3*n_atom) * sizeof(int)
             + Ntypes * (L_PSD + L_ATMTYPE) * sizeof(char)
             + 0*(Ntypes+3*n_atom) *16; // last term is spare memory in case
    buff = (char *)malloc( l_buff*sizeof(char) );
    assert(buff != NULL);

    if (rank == 0) {
        // pack the variables
        position = 0;
        MPI_Pack(pSPARC->localPsd, Ntypes, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->Zatom, Ntypes, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->Znucl, Ntypes, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->nAtomv, Ntypes, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->Mass, Ntypes, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->atomType, Ntypes * L_ATMTYPE, MPI_CHAR, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->psdName, Ntypes * L_PSD, MPI_CHAR, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->atom_pos, 3*n_atom, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->IsFrac, pSPARC->Ntypes, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->mvAtmConstraint, 3*n_atom, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->IsSpin, pSPARC->Ntypes, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->atom_spin, n_atom, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
        for (i = 0; i < Ntypes; i++) {
            nproj = 0;
            for (l = 0; l <= lmaxv[i]; l++) {
                nproj += pSPARC->psd[i].ppl[l];
            }
            MPI_Pack(pSPARC->psd[i].RadialGrid, sizev[i], MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->psd[i].UdV, nproj*sizev[i], MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(&pSPARC->psd[i].Vloc_0, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(&pSPARC->psd[i].fchrg, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->psd[i].rVloc, sizev[i], MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->psd[i].rhoIsoAtom, sizev[i], MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->psd[i].rc, lmaxv[i]+1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->psd[i].Gamma, nproj, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->psd[i].rho_c_table, sizev[i], MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
               
            if (pSPARC->psd[i].pspsoc) {
                nprojso = 0;
                for (l = 1; l <= lmaxv[i]; l++) {
                    nprojso += pSPARC->psd[i].ppl_soc[l-1];
                }
                MPI_Pack(pSPARC->psd[i].Gamma_soc, nprojso, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
                MPI_Pack(pSPARC->psd[i].UdV_soc, nprojso*sizev[i], MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            }
        }
        // broadcast the packed buffer
        MPI_Bcast(buff, l_buff, MPI_PACKED, 0, MPI_COMM_WORLD);
    } else {
        /* allocate memory for receiver processes */
        pSPARC->localPsd = (int *)malloc( Ntypes * sizeof(int) );
        pSPARC->Zatom = (int *)malloc( Ntypes * sizeof(int) );
        pSPARC->Znucl = (int *)malloc( Ntypes * sizeof(int) );
        pSPARC->nAtomv = (int *)malloc( Ntypes * sizeof(int) );
        pSPARC->IsFrac = (int *)malloc( Ntypes * sizeof(int) );
        pSPARC->IsSpin = (int *)malloc( Ntypes * sizeof(int) );
        pSPARC->Mass = (double *)malloc( Ntypes * sizeof(double) );
        pSPARC->atomType = (char *)malloc( Ntypes * L_ATMTYPE * sizeof(char) );
        pSPARC->psdName = (char *)malloc( Ntypes * L_PSD * sizeof(char) );

        if (pSPARC->localPsd == NULL || pSPARC->Zatom == NULL ||
            pSPARC->Znucl == NULL || pSPARC->nAtomv == NULL || 
            pSPARC->Mass == NULL || pSPARC->atomType == NULL || 
            pSPARC->psdName == NULL) {
            printf("\nmemory cannot be allocated3\n");
            exit(EXIT_FAILURE);
        }

        // values to receive
        pSPARC->atom_pos = (double *)malloc(3*n_atom*sizeof(double));
        pSPARC->mvAtmConstraint = (int *)malloc(3*n_atom*sizeof(int));
        pSPARC->atom_spin = (double *)malloc(n_atom*sizeof(double));
        if (pSPARC->atom_pos == NULL || pSPARC->mvAtmConstraint == NULL || pSPARC->atom_spin == NULL) {
            printf("\nmemory cannot be allocated4\n");
            exit(EXIT_FAILURE);
        }

        for (i = 0; i < Ntypes; i++) {
            nproj = 0;
            for (l = 0; l <= lmaxv[i]; l++) {
                nproj += pSPARC->psd[i].ppl[l];
            }
            pSPARC->psd[i].RadialGrid = (double *)malloc(sizev[i] * sizeof(double));
            pSPARC->psd[i].UdV = (double *)malloc(nproj * sizev[i] * sizeof(double));
            pSPARC->psd[i].rVloc = (double *)malloc(sizev[i] * sizeof(double));
            pSPARC->psd[i].rhoIsoAtom = (double *)malloc(sizev[i] * sizeof(double));
            pSPARC->psd[i].rc = (double *)malloc((lmaxv[i]+1) * sizeof(double));
            pSPARC->psd[i].Gamma = (double *)malloc(nproj * sizeof(double));
            pSPARC->psd[i].rho_c_table = (double *)malloc(sizev[i] * sizeof(double));
            // check if memory is allocated successfully!
            if (pSPARC->psd[i].RadialGrid == NULL || pSPARC->psd[i].UdV == NULL ||
                pSPARC->psd[i].rVloc == NULL || pSPARC->psd[i].rhoIsoAtom == NULL ||
                pSPARC->psd[i].rc == NULL || pSPARC->psd[i].Gamma == NULL ||
                pSPARC->psd[i].rho_c_table == NULL)
            {
                printf("\nmemory cannot be allocated5\n");
                exit(EXIT_FAILURE);
            }
            if (pSPARC->psd[i].pspsoc) {
                nprojso = 0;
                for (l = 1; l <= lmaxv[i]; l++) {
                    nprojso += pSPARC->psd[i].ppl_soc[l-1];
                }
                pSPARC->psd[i].Gamma_soc = (double *)malloc(nprojso * sizeof(double));
                pSPARC->psd[i].UdV_soc = (double *)malloc(nprojso * sizev[i] * sizeof(double));
            }
        }
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        // broadcast the packed buffer
        MPI_Bcast(buff, l_buff, MPI_PACKED, 0, MPI_COMM_WORLD);
#ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("MPI_Bcast packed buff of length %d took %.3f ms\n", l_buff,(t2-t1)*1000);
#endif
        // unpack the variables
        position = 0;
        MPI_Unpack(buff, l_buff, &position, pSPARC->localPsd, Ntypes, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->Zatom, Ntypes, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->Znucl, Ntypes, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->nAtomv, Ntypes, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->Mass, Ntypes, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->atomType, Ntypes*L_ATMTYPE, MPI_CHAR, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->psdName, Ntypes*L_PSD, MPI_CHAR, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->atom_pos, 3*n_atom, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->IsFrac, Ntypes, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->mvAtmConstraint, 3*n_atom, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->IsSpin, Ntypes, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->atom_spin, n_atom, MPI_DOUBLE, MPI_COMM_WORLD);
        for (i = 0; i < Ntypes; i++) {
            nproj = 0;
            for (l = 0; l <= lmaxv[i]; l++) {
                nproj += pSPARC->psd[i].ppl[l];
            }
            MPI_Unpack(buff, l_buff, &position, pSPARC->psd[i].RadialGrid,  sizev[i], MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->psd[i].UdV,  nproj*sizev[i], MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, &pSPARC->psd[i].Vloc_0,  1, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, &pSPARC->psd[i].fchrg, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->psd[i].rVloc,  sizev[i], MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->psd[i].rhoIsoAtom,  sizev[i], MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->psd[i].rc,  lmaxv[i]+1, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->psd[i].Gamma,  nproj, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->psd[i].rho_c_table, sizev[i], MPI_DOUBLE, MPI_COMM_WORLD);
            if (pSPARC->psd[i].pspsoc) {
                nprojso = 0;
                for (l = 1; l <= lmaxv[i]; l++) {
                    nprojso += pSPARC->psd[i].ppl_soc[l-1];
                }
                MPI_Unpack(buff, l_buff, &position, pSPARC->psd[i].Gamma_soc,  nprojso, MPI_DOUBLE, MPI_COMM_WORLD);
                MPI_Unpack(buff, l_buff, &position, pSPARC->psd[i].UdV_soc, nprojso*sizev[i], MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
    }

    // deallocate memory
    free(tempbuff);
    free(ppl_sdispl);
    free(is_r_uniformv);
    free(pspxcv);
    free(pspsocv);
    free(pplv);
    free(lmaxv);
    free(sizev);
    free(buff);
    if (pSPARC->SOC_Flag == 1) {
        free(ppl_soc_sdispl);
        free(pplv_soc);
    }
}



/**
 * @brief   Copy the data read from input files into struct SPARC.
 */
void SPARC_copy_input(SPARC_OBJ *pSPARC, SPARC_INPUT_OBJ *pSPARC_Input) {
    int rank, nproc, i, p, FDn, Ntypes;
#ifdef DEBUG
    double t1, t2;
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    /* copy input values from input struct */
    // int type values
    pSPARC->num_node = pSPARC_Input->num_node;
    pSPARC->num_cpu_per_node = pSPARC_Input->num_cpu_per_node;
    pSPARC->num_acc_per_node = pSPARC_Input->num_acc_per_node;
    pSPARC->npspin = pSPARC_Input->npspin;
    pSPARC->npkpt = pSPARC_Input->npkpt;
    pSPARC->npband = pSPARC_Input->npband;
    pSPARC->npNdx = pSPARC_Input->npNdx;
    pSPARC->npNdy = pSPARC_Input->npNdy;
    pSPARC->npNdz = pSPARC_Input->npNdz;
    pSPARC->npNdx_phi = pSPARC_Input->npNdx_phi;
    pSPARC->npNdy_phi = pSPARC_Input->npNdy_phi;
    pSPARC->npNdz_phi = pSPARC_Input->npNdz_phi;
    pSPARC->eig_serial_maxns = pSPARC_Input->eig_serial_maxns;
    pSPARC->eig_paral_blksz = pSPARC_Input->eig_paral_blksz;
    pSPARC->spin_typ = pSPARC_Input->spin_typ;
    pSPARC->MDFlag = pSPARC_Input->MDFlag;
    pSPARC->RelaxFlag = pSPARC_Input->RelaxFlag;
    pSPARC->RestartFlag = pSPARC_Input->RestartFlag;
    pSPARC->Flag_latvec_scale = pSPARC_Input->Flag_latvec_scale;
    pSPARC->numIntervals_x = pSPARC_Input->numIntervals_x;
    pSPARC->numIntervals_y = pSPARC_Input->numIntervals_y;
    pSPARC->numIntervals_z = pSPARC_Input->numIntervals_z;
    pSPARC->BC = pSPARC_Input->BC;
    pSPARC->BCx = pSPARC_Input->BCx;
    pSPARC->BCy = pSPARC_Input->BCy;
    pSPARC->BCz = pSPARC_Input->BCz;
    pSPARC->Nstates = pSPARC_Input->Nstates;
    //pSPARC->Ntypes = pSPARC_Input->Ntypes;
    pSPARC->NetCharge = pSPARC_Input->NetCharge;
    pSPARC->order = pSPARC_Input->order;
    pSPARC->ChebDegree = pSPARC_Input->ChebDegree;
    pSPARC->CheFSI_Optmz = pSPARC_Input->CheFSI_Optmz;
    pSPARC->chefsibound_flag = pSPARC_Input->chefsibound_flag;
    pSPARC->rhoTrigger = pSPARC_Input->rhoTrigger; //pSPARC->rhoTrigger--;
    pSPARC->FixRandSeed = pSPARC_Input->FixRandSeed;
    pSPARC->accuracy_level = pSPARC_Input->accuracy_level;
    pSPARC->scf_err_type = pSPARC_Input->scf_err_type;
    pSPARC->MAXIT_SCF = pSPARC_Input->MAXIT_SCF;
    pSPARC->MINIT_SCF = pSPARC_Input->MINIT_SCF;
    pSPARC->MAXIT_POISSON = pSPARC_Input->MAXIT_POISSON;
    pSPARC->Relax_Niter = pSPARC_Input->Relax_Niter;
    pSPARC->MixingVariable = pSPARC_Input->MixingVariable;
    pSPARC->MixingPrecond = pSPARC_Input->MixingPrecond;
    pSPARC->MixingPrecondMag = pSPARC_Input->MixingPrecondMag;
    pSPARC->MixingHistory = pSPARC_Input->MixingHistory;
    pSPARC->PulayFrequency = pSPARC_Input->PulayFrequency;
    pSPARC->PulayRestartFlag = pSPARC_Input->PulayRestartFlag;
    pSPARC->precond_fitpow = pSPARC_Input->precond_fitpow;
    pSPARC->Nkpts = pSPARC_Input->Nkpts;
    pSPARC->Kx = pSPARC_Input->Kx;
    pSPARC->Ky = pSPARC_Input->Ky;
    pSPARC->Kz = pSPARC_Input->Kz;
    pSPARC->NkptsGroup = pSPARC_Input->NkptsGroup;
    pSPARC->kctr = pSPARC_Input->kctr;
    pSPARC->Verbosity = pSPARC_Input->Verbosity;
    pSPARC->PrintForceFlag = pSPARC_Input->PrintForceFlag;
    pSPARC->PrintAtomPosFlag = pSPARC_Input->PrintAtomPosFlag;
    pSPARC->PrintAtomVelFlag = pSPARC_Input->PrintAtomVelFlag;
    pSPARC->PrintEigenFlag = pSPARC_Input->PrintEigenFlag;
    pSPARC->PrintElecDensFlag = pSPARC_Input->PrintElecDensFlag;
    pSPARC->PrintMDout = pSPARC_Input->PrintMDout;
    pSPARC->PrintRelaxout = pSPARC_Input->PrintRelaxout;
    pSPARC->Printrestart = pSPARC_Input->Printrestart;
    pSPARC->Printrestart_fq = pSPARC_Input->Printrestart_fq;
    pSPARC->elec_T_type = pSPARC_Input->elec_T_type;
    pSPARC->MD_Nstep = pSPARC_Input->MD_Nstep;
    pSPARC->NPTscaleVecs[0] = pSPARC_Input->NPTscaleVecs[0];
    pSPARC->NPTscaleVecs[1] = pSPARC_Input->NPTscaleVecs[1];
    pSPARC->NPTscaleVecs[2] = pSPARC_Input->NPTscaleVecs[2];
    pSPARC->NPT_NHnnos = pSPARC_Input->NPT_NHnnos;
    pSPARC->ion_elec_eqT = pSPARC_Input->ion_elec_eqT;
    pSPARC->ion_vel_dstr = pSPARC_Input->ion_vel_dstr;
    pSPARC->ion_vel_dstr_rand = pSPARC_Input->ion_vel_dstr_rand;
    pSPARC->L_history = pSPARC_Input->L_history;
    pSPARC->L_autoscale = pSPARC_Input->L_autoscale;
    pSPARC->L_lineopt = pSPARC_Input->L_lineopt;
    pSPARC->Calc_stress = pSPARC_Input->Calc_stress;
    pSPARC->Calc_pres = pSPARC_Input->Calc_pres;
    pSPARC->d3Flag = pSPARC_Input->d3Flag;
    pSPARC->vdWDFKernelGenFlag = pSPARC_Input->vdWDFKernelGenFlag;
    pSPARC->MAXIT_FOCK = pSPARC_Input->MAXIT_FOCK;
    pSPARC->EXXMeth_Flag = pSPARC_Input->EXXMeth_Flag;
    pSPARC->ACEFlag = pSPARC_Input->ACEFlag;
    pSPARC->EXXMem_batch = pSPARC_Input->EXXMem_batch;
    pSPARC->EXXACEVal_state = pSPARC_Input->EXXACEVal_state;
    pSPARC->EXXDiv_Flag = pSPARC_Input->EXXDiv_Flag;
    pSPARC->EXXDownsampling[0] = pSPARC_Input->EXXDownsampling[0];
    pSPARC->EXXDownsampling[1] = pSPARC_Input->EXXDownsampling[1];
    pSPARC->EXXDownsampling[2] = pSPARC_Input->EXXDownsampling[2];
    pSPARC->MINIT_FOCK = pSPARC_Input->MINIT_FOCK;
    pSPARC->SQFlag = pSPARC_Input->SQFlag;
    pSPARC->SQ_typ_dm = pSPARC_Input->SQ_typ_dm;
    pSPARC->SQ_gauss_mem = pSPARC_Input->SQ_gauss_mem;
    pSPARC->SQ_npl_c = pSPARC_Input->SQ_npl_c;
    pSPARC->SQ_npl_g = pSPARC_Input->SQ_npl_g;
    pSPARC->SQ_EigshiftFlag = pSPARC_Input->SQ_EigshiftFlag;
    pSPARC->npNdx_SQ = pSPARC_Input->npNdx_SQ;
    pSPARC->npNdy_SQ = pSPARC_Input->npNdy_SQ;
    pSPARC->npNdz_SQ = pSPARC_Input->npNdz_SQ;
    for (i = 0; i < 7; i++)
        pSPARC->PrintPsiFlag[i] = pSPARC_Input->PrintPsiFlag[i];
    pSPARC->PrintEnergyDensFlag = pSPARC_Input->PrintEnergyDensFlag;
    
    // double type values
    pSPARC->range_x = pSPARC_Input->range_x;
    pSPARC->range_y = pSPARC_Input->range_y;
    pSPARC->range_z = pSPARC_Input->range_z;
    if (pSPARC->range_x <= 0.0 || pSPARC->range_y <= 0.0 || pSPARC->range_z <= 0.0) {
        if (!rank) printf("\nERROR: Please specify valid CELL dimensions!\n");
        exit(EXIT_FAILURE);
    }
    pSPARC->latvec_scale_x = pSPARC_Input->latvec_scale_x;
    pSPARC->latvec_scale_y = pSPARC_Input->latvec_scale_y;
    pSPARC->latvec_scale_z = pSPARC_Input->latvec_scale_z;
    // allocate the memory for lattice vector array
    for(i = 0; i < 9; i++)
        pSPARC->LatVec[i] = pSPARC_Input->LatVec[i];
    pSPARC->mesh_spacing = pSPARC_Input->mesh_spacing;
    pSPARC->ecut = pSPARC_Input->ecut;
    pSPARC->kptshift[0] = pSPARC_Input->kptshift[0];
    pSPARC->kptshift[1] = pSPARC_Input->kptshift[1];
    pSPARC->kptshift[2] = pSPARC_Input->kptshift[2];
    pSPARC->target_energy_accuracy = pSPARC_Input->target_energy_accuracy;
    pSPARC->target_force_accuracy = pSPARC_Input->target_force_accuracy;
    pSPARC->TOL_SCF = pSPARC_Input->TOL_SCF;
    pSPARC->TOL_RELAX = pSPARC_Input->TOL_RELAX;
    pSPARC->TOL_POISSON = pSPARC_Input->TOL_POISSON;
    pSPARC->TOL_LANCZOS = pSPARC_Input->TOL_LANCZOS;
    pSPARC->TOL_PSEUDOCHARGE = pSPARC_Input->TOL_PSEUDOCHARGE;
    pSPARC->TOL_PRECOND = pSPARC_Input->TOL_PRECOND;
    pSPARC->POISSON_SOLVER = pSPARC_Input->Poisson_solver;
    pSPARC->precond_kerker_kTF = pSPARC_Input->precond_kerker_kTF;
    pSPARC->precond_kerker_thresh = pSPARC_Input->precond_kerker_thresh;
    pSPARC->precond_kerker_kTF_mag = pSPARC_Input->precond_kerker_kTF_mag;
    pSPARC->precond_kerker_thresh_mag = pSPARC_Input->precond_kerker_thresh_mag;
    pSPARC->precond_resta_q0 = pSPARC_Input->precond_resta_q0;
    pSPARC->precond_resta_Rs = pSPARC_Input->precond_resta_Rs;
    pSPARC->REFERENCE_CUTOFF = pSPARC_Input->REFERENCE_CUTOFF;
    pSPARC->Beta = pSPARC_Input->Beta;
    pSPARC->elec_T = pSPARC_Input->elec_T;
    pSPARC->MixingParameter = pSPARC_Input->MixingParameter;
    pSPARC->MixingParameterSimple = pSPARC_Input->MixingParameterSimple;
    pSPARC->MixingParameterMag = pSPARC_Input->MixingParameterMag;
    pSPARC->MixingParameterSimpleMag = pSPARC_Input->MixingParameterSimpleMag;
    pSPARC->MD_dt = pSPARC_Input->MD_dt;
    pSPARC->ion_T = pSPARC_Input->ion_T;
    pSPARC->thermos_Tf = pSPARC_Input->thermos_Tf;
    pSPARC->qmass = pSPARC_Input->qmass;
    pSPARC->TWtime = pSPARC_Input->TWtime;// buffer time
    for (i = 0; i < pSPARC->NPT_NHnnos; i++) {
        pSPARC->NPT_NHqmass[i] = pSPARC_Input->NPT_NHqmass[i];
    }
    pSPARC->NPT_NHbmass = pSPARC_Input->NPT_NHbmass;
    pSPARC->prtarget = pSPARC_Input->prtarget;
    pSPARC->NPT_NP_bmass = pSPARC_Input->NPT_NP_bmass;
    pSPARC->NPT_NP_qmass = pSPARC_Input->NPT_NP_qmass;
    pSPARC->NLCG_sigma = pSPARC_Input->NLCG_sigma;
    pSPARC->L_finit_stp = pSPARC_Input->L_finit_stp;
    pSPARC->L_maxmov = pSPARC_Input->L_maxmov;
    pSPARC->L_icurv = pSPARC_Input->L_icurv;
    pSPARC->FIRE_dt = pSPARC_Input->FIRE_dt;
    pSPARC->FIRE_mass = pSPARC_Input->FIRE_mass;
    pSPARC->FIRE_maxmov = pSPARC_Input->FIRE_maxmov;
    pSPARC->max_dilatation = pSPARC_Input->max_dilatation;
    pSPARC->TOL_RELAX_CELL = pSPARC_Input->TOL_RELAX_CELL;
    pSPARC->eig_paral_orfac = pSPARC_Input->eig_paral_orfac;
    pSPARC->eig_paral_maxnp = pSPARC_Input->eig_paral_maxnp;
    pSPARC->d3Rthr = pSPARC_Input->d3Rthr;
    pSPARC->d3Cn_thr = pSPARC_Input->d3Cn_thr;
    pSPARC->TOL_FOCK = pSPARC_Input->TOL_FOCK;
    pSPARC->TOL_SCF_INIT = pSPARC_Input->TOL_SCF_INIT;
    pSPARC->hyb_range_fock = pSPARC_Input->hyb_range_fock;
    pSPARC->hyb_range_pbe = pSPARC_Input->hyb_range_pbe;
    pSPARC->exx_frac = pSPARC_Input->exx_frac;
    pSPARC->SQ_rcut = pSPARC_Input->SQ_rcut;
    pSPARC->SQ_fac_g2c = pSPARC_Input->SQ_fac_g2c;
    pSPARC->SQ_tol_occ = pSPARC_Input->SQ_tol_occ;
    pSPARC->SQ_eigshift = pSPARC_Input->SQ_eigshift;

    // char type values
    strncpy(pSPARC->MDMeth , pSPARC_Input->MDMeth,sizeof(pSPARC->MDMeth));
    strncpy(pSPARC->RelaxMeth , pSPARC_Input->RelaxMeth,sizeof(pSPARC->RelaxMeth));
    strncpy(pSPARC->XC, pSPARC_Input->XC,sizeof(pSPARC->XC));
    if (strcmp(pSPARC->XC,"UNDEFINED") == 0) {
        if (!rank) printf("\nERROR: Please specify XC type!\n");
        exit(EXIT_FAILURE);
    }
    strncpy(pSPARC->filename , pSPARC_Input->filename,sizeof(pSPARC->filename));
    strncpy(pSPARC->filename_out , pSPARC_Input->filename_out,sizeof(pSPARC->filename_out));
    strncpy(pSPARC->SPARCROOT , pSPARC_Input->SPARCROOT,sizeof(pSPARC->SPARCROOT));

    // check XC compatibility with pseudopotential
    pSPARC->usefock = 0;                    // default: no fock operator 
    int ixc = 0; // input XC
    if (strcmpi(pSPARC->XC, "LDA_PZ") == 0) {
        ixc = 2;
    } else if (strcmpi(pSPARC->XC, "LDA_PW") == 0) {
        ixc = 7;
    } else if (strcmpi(pSPARC->XC, "GGA_PBE") == 0) {
        ixc = 11;
    } else if (strcmpi(pSPARC->XC, "GGA_RPBE") == 0) {
        ixc = 15;
    } else if (strcmpi(pSPARC->XC, "GGA_PBEsol") == 0) {
        ixc = 116;
    } else if (strcmpi(pSPARC->XC, "HF") == 0) {
        ixc = 40;
        pSPARC->usefock = 1;
        if (pSPARC->exx_frac < 0) pSPARC->exx_frac = 1;
        if (fabs(1-pSPARC->exx_frac) > TEMP_TOL) {
            if (!rank) printf("ERROR: HF functional could be only defined with 1.0 EXX_FRAC.\n");
        }
    } else if (strcmpi(pSPARC->XC, "PBE0") == 0) {
        ixc = 41;
        pSPARC->usefock = 1;
        if (pSPARC->exx_frac < 0) pSPARC->exx_frac = 0.25;
        if (fabs(0.25-pSPARC->exx_frac) > TEMP_TOL) {
            if (!rank) printf("Note: You are using PBE0 with %.5g exact exchange.\n", pSPARC->exx_frac);
        }
    } else if (strcmpi(pSPARC->XC, "HSE") == 0) {
        ixc = 427;
        pSPARC->usefock = 1;
        if (pSPARC->exx_frac < 0) pSPARC->exx_frac = 0.25;
        if (fabs(0.25-pSPARC->exx_frac) > TEMP_TOL) {
            if (!rank) printf("Note: You are using HSE with %.5g exact exchange.\n", pSPARC->exx_frac);
        }
    } else if (strcmpi(pSPARC->XC, "SCAN") == 0) {
        ixc = -263267;
    } else if (strcmpi(pSPARC->XC, "vdWDF1") == 0) {
        ixc = -102; // this is the index of Zhang-Yang revPBE exchange in Libxc
    } else if (strcmpi(pSPARC->XC, "vdWDF2") == 0) {
        ixc = -108; // this is the index of PW86 exchange in Libxc
    }
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->usefock && pSPARC->psd[ityp].pspxc != ixc) {
            if (!rank) printf(YEL "\nWARNING: Pseudopotential file for atom type %s has pspxc = %d,\n"
                    "not equal to input ixc = %d (%s). Be careful with the result.\n" RESET, 
                    &pSPARC->atomType[ityp*L_ATMTYPE], pSPARC->psd[ityp].pspxc, ixc, pSPARC->XC);
        }
        if (pSPARC->usefock && pSPARC->psd[ityp].pspxc != 11) {
            if (!rank) printf(YEL "\nWARNING: Pseudopotential file for atom type %s has pspxc = %d,\n"
                    "while hybrid calculation needs a PBE pseudopotential. Be careful with the result.\n" RESET, 
                    &pSPARC->atomType[ityp*L_ATMTYPE], pSPARC->psd[ityp].pspxc);
        }
    }    

    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        // pSPARC->hyb_range_fock = 0.106;     // QE's value
        // pSPARC->hyb_range_pbe = 0.106;      // QE's value
        // pSPARC->hyb_range_fock = 0.106066017177982;     // ABINIT's value
        // pSPARC->hyb_range_pbe = 0.188988157484231;      // ABINIT's value
        if (!rank) {
            printf("Careful: You are using HSE with range-separation parameter omega_HF = %.6f (1/Bohr) and omega_PBE = %.6f (1/Bohr)\n", pSPARC->hyb_range_fock, pSPARC->hyb_range_pbe);
            printf("If you want to change it, please use EXX_RANGE_FOCK and EXX_RANGE_PBE input options.\n");
        }
    } else {
        pSPARC->hyb_range_fock = -1;
        pSPARC->hyb_range_pbe = -1;
    }

    // check MDMeth availability
    if ((strcmpi(pSPARC->MDMeth,"NVT_NH") && strcmpi(pSPARC->MDMeth,"NVE")
        && strcmpi(pSPARC->MDMeth,"NVK_G") && strcmpi(pSPARC->MDMeth,"NPT_NH") && strcmpi(pSPARC->MDMeth,"NPT_NP")) != 0) {
            if (!rank){
				printf("\nCannot recognize MDMeth = \"%s\"\n",pSPARC->MDMeth);
                printf("MDMeth (MD Method) must be one of the following:\n\tNVT_NH\t NVE\t NVK_G\t NPT_NH\t NPT_NP\n");
            }
            exit(EXIT_FAILURE);
    }

    /* process the data read from input files */
    Ntypes = pSPARC->Ntypes;

    // calculate total number of electrons
    pSPARC->Nelectron = 0;
    for (i = 0; i < Ntypes; i++) {
        pSPARC->Nelectron += pSPARC->Znucl[i] * pSPARC->nAtomv[i];
    }
    pSPARC->Nelectron -= pSPARC->NetCharge;

    // check if NLCC is present
    int NLCC_flag = 0;
    for (int ityp = 0; ityp < Ntypes; ityp++) {
        if (pSPARC->psd[ityp].fchrg > TEMP_TOL) {
            NLCC_flag = 1;
            break;
        }
    }
    pSPARC->NLCC_flag = NLCC_flag;

    // check if exchange-correlation functional is metaGGA
    pSPARC->mGGAflag = 0;
    if (strcmpi(pSPARC->XC, "SCAN") == 0) { // it can be expand, such as adding r2SCAN 
        pSPARC->mGGAflag = 1;
    }
    // check if exchange-correlation functional is vdW-DF1 or vdW-DF2
    pSPARC->vdWDFFlag = 0;
    if (strcmpi(pSPARC->XC,"vdWDF1") == 0) {
        pSPARC->vdWDFFlag = 1;
    }
    if (strcmpi(pSPARC->XC,"vdWDF2") == 0) {
        pSPARC->vdWDFFlag = 2;
    }

    // initialize energy values
    pSPARC->Esc = 0.0;
    pSPARC->Efermi = 0.0;
    pSPARC->Exc = 0.0;
    pSPARC->Eband = 0.0;
    pSPARC->Entropy = 0.0;
    pSPARC->Etot = 0.0;

    // estimate Nstates if not provided
    if (pSPARC->Nstates == -1) {
        // estimate Nstates using the linear function y = 1.2 * x + 5
        pSPARC->Nstates = (int) ((pSPARC->Nelectron / 2) * 1.2 + 5) * pSPARC->Nspinor;
    }

    if (pSPARC->Nspinor == 1) {
        if (pSPARC->Nstates < (pSPARC->Nelectron / 2) && !pSPARC->SQFlag) {
            if (!rank) printf("\nERROR: number of states is less than Nelectron/2!\n");
            exit(EXIT_FAILURE);
        }
    } else if (pSPARC->Nspinor == 2) {
        if (pSPARC->Nstates < pSPARC->Nelectron && !pSPARC->SQFlag) {
            if (!rank) printf("\nERROR: number of states is less than Nelectron!\n");
            exit(EXIT_FAILURE);
        }
    }

    // filenames
    if (rank == 0) {
        snprintf(pSPARC->OutFilename,       L_STRING, "%s.out"  ,     pSPARC->filename_out);
        snprintf(pSPARC->StaticFilename,    L_STRING, "%s.static",     pSPARC->filename_out);
        snprintf(pSPARC->AtomFilename,      L_STRING, "%s.atom",      pSPARC->filename_out);
        snprintf(pSPARC->EigenFilename,     L_STRING, "%s.eigen",     pSPARC->filename_out);
        snprintf(pSPARC->MDFilename,        L_STRING, "%s.aimd",      pSPARC->filename_out);
        snprintf(pSPARC->RelaxFilename,     L_STRING, "%s.geopt",     pSPARC->filename_out);
        snprintf(pSPARC->restart_Filename,  L_STRING, "%s.restart",   pSPARC->filename_out);
        snprintf(pSPARC->restartC_Filename, L_STRING, "%s.restart-0", pSPARC->filename_out);
        snprintf(pSPARC->restartP_Filename, L_STRING, "%s.restart-1", pSPARC->filename_out);
        snprintf(pSPARC->DensTCubFilename,  L_STRING, "%s.dens",   pSPARC->filename_out);
        snprintf(pSPARC->DensUCubFilename,  L_STRING, "%s.densUp",    pSPARC->filename_out);
        snprintf(pSPARC->DensDCubFilename,  L_STRING, "%s.densDwn",   pSPARC->filename_out);
        snprintf(pSPARC->OrbitalsFilename,  L_STRING, "%s.psi",       pSPARC->filename_out);
        snprintf(pSPARC->KinEnDensTCubFilename,  L_STRING, "%s.kedens",       pSPARC->filename_out);
        snprintf(pSPARC->KinEnDensUCubFilename,  L_STRING, "%s.kedensUp",     pSPARC->filename_out);
        snprintf(pSPARC->KinEnDensDCubFilename,  L_STRING, "%s.kedensDwn",    pSPARC->filename_out);
        snprintf(pSPARC->XcEnDensCubFilename,    L_STRING, "%s.xcedens",      pSPARC->filename_out);
        snprintf(pSPARC->ExxEnDensTCubFilename,  L_STRING, "%s.exxedens",     pSPARC->filename_out);
        snprintf(pSPARC->ExxEnDensUCubFilename,  L_STRING, "%s.exxedensUp",   pSPARC->filename_out);
        snprintf(pSPARC->ExxEnDensDCubFilename,  L_STRING, "%s.exxedensDwn",  pSPARC->filename_out);

        // check if the name for out file exits
        char temp_outfname[L_STRING];
        snprintf(temp_outfname, L_STRING, "%s", pSPARC->OutFilename);
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        int MAX_OUTPUT = 100; // max number of output files allowed before we overwrite existing output files
        i = 0;
        while ( (access( temp_outfname, F_OK ) != -1) && i <= MAX_OUTPUT ) {
            i++;
            snprintf(temp_outfname, L_STRING, "%s_%02d", pSPARC->OutFilename, i);
        }
        pSPARC->suffixNum = i; // note that this is only known to rank 0!

#ifdef DEBUG
        t2 = MPI_Wtime();
        printf("\nChecking existence of (%d) out file(s) took %.3f ms\n", i, (t2-t1)*1000);
#endif
        if (i >= (int)(MAX_OUTPUT*0.75) && i <= MAX_OUTPUT) {
            printf("\n#WARNING: There's a limit on total number of output files allowed!\n"
                     "         After the limit is reached, SPARC will start using  the\n"
                     "         name provided exactly (without attached index) and old\n"
                     "         results will be overwritten! Please move some existing\n"
                     "         results to some other directories if you want to keep \n"
                     "         storing results in different files!\n"
                     "         Current usage: %.1f %%\n\n", i/(double) MAX_OUTPUT * 100);
        }

        if (i > MAX_OUTPUT) {
            printf("\n#WARNING: The limit of total number of output files is reached! \n"
                     "         Current output name (without suffix): %s\n\n", pSPARC->filename_out);
        } else if (i > 0) {
            char tempchar[L_STRING];
            snprintf(tempchar, L_STRING, "%s", pSPARC->OutFilename);
            snprintf(pSPARC->OutFilename,   L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->StaticFilename);
            snprintf(pSPARC->StaticFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->AtomFilename);
            snprintf(pSPARC->AtomFilename,  L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->EigenFilename);
            snprintf(pSPARC->EigenFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->MDFilename);
            snprintf(pSPARC->MDFilename,    L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->RelaxFilename);
            snprintf(pSPARC->RelaxFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->DensTCubFilename);
            snprintf(pSPARC->DensTCubFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->DensUCubFilename);
            snprintf(pSPARC->DensUCubFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->DensDCubFilename);
            snprintf(pSPARC->DensDCubFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->OrbitalsFilename);
            snprintf(pSPARC->OrbitalsFilename, L_STRING, "%s_%02d", tempchar, i);
            // energy density files 
            snprintf(tempchar, L_STRING, "%s", pSPARC->KinEnDensTCubFilename);
            snprintf(pSPARC->KinEnDensTCubFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->KinEnDensUCubFilename);
            snprintf(pSPARC->KinEnDensUCubFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->KinEnDensDCubFilename);
            snprintf(pSPARC->KinEnDensDCubFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->XcEnDensCubFilename);
            snprintf(pSPARC->XcEnDensCubFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->ExxEnDensTCubFilename);
            snprintf(pSPARC->ExxEnDensTCubFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->ExxEnDensUCubFilename);
            snprintf(pSPARC->ExxEnDensUCubFilename, L_STRING, "%s_%02d", tempchar, i);
            snprintf(tempchar, L_STRING, "%s", pSPARC->ExxEnDensDCubFilename);
            snprintf(pSPARC->ExxEnDensDCubFilename, L_STRING, "%s_%02d", tempchar, i);
        }
    }
    // Not only rank 0 printing orbitals
    MPI_Bcast(pSPARC->OrbitalsFilename, L_STRING, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Initialize MD/relax variables
    pSPARC->RelaxCount = 0; // initialize current relaxation step
    pSPARC->StressCount = 0; // initialize current stress relaxation step (used in full relaxation)
    pSPARC->elecgs_Count = 0; // count the number of times forces are calculated(useful in relaxation)
    pSPARC->MDCount = 0; // initialize current MD step
    pSPARC->StopCount = 0;
    pSPARC->restartCount = 0;
    pSPARC->amu2au = CONST_AMU2AU; // 1 au = 9.10938356e-31 Kg; 1 amu =  1.660539040e-27 Kg;
    pSPARC->fs2atu = CONST_FS2ATU; //1atu = 2.418884326509e-17 s;
    pSPARC->kB = CONST_KB; // Boltzmann constant in Ha/K

    if(pSPARC->RelaxFlag == 2 || pSPARC->RelaxFlag == 3){   
        pSPARC->Printrestart = 0;   
    }

    // Value of xc_rhotol used in xc functional
    pSPARC->xc_rhotol = 1e-14;

    if (pSPARC->Beta < 0) {
        if (pSPARC->elec_T_type == 1) { // gaussian
            // The electronic temperature corresponding to 0.2 eV is 2320.904422 K
            pSPARC->Beta = CONST_EH / 0.2; // smearing = 0.2 eV = 0.00734986450 Ha, Beta := 1 / smearing
        } else { // fermi-dirac
            // The electronic temperature corresponding to 0.1 eV is 1160.452211 K
            pSPARC->Beta = CONST_EH / 0.1; // smearing = 0.1 eV = 0.00367493225 Ha, Beta := 1 / smearing
        }
        pSPARC->elec_T = 1./(pSPARC->kB * pSPARC->Beta);
    }

    // Check the cell typ
    double mult;
    int j;
    pSPARC->cell_typ = 0; // orthogonal cell by default
    for(i = 0; i < 2; i++){
        for(j = i+1; j < 3; j++){
            mult = fabs(pSPARC->LatVec[3*i] * pSPARC->LatVec[3*j] + pSPARC->LatVec[3*i+1] * pSPARC->LatVec[3*j+1] + pSPARC->LatVec[3*i+2] * pSPARC->LatVec[3*j+2]);
            if(mult > TEMP_TOL){
                pSPARC->cell_typ = 1;
                i = j = 3;
            }
        }
    }

    // determine boundary conditions in each direction
    // BCx = 0 -> periodic, BCx = 1 -> dirichlet
    if (pSPARC->BC > 0) {
	    if (pSPARC->BC == 1) {
	        // dirichlet boundary
	        pSPARC->BCx = 1; pSPARC->BCy = 1; pSPARC->BCz = 1;
	    } else if (pSPARC->BC == 2 || pSPARC->BC == 0) {
	        // periodic in all three directions
	        pSPARC->BCx = 0; pSPARC->BCy = 0; pSPARC->BCz = 0;
	    } else if (pSPARC->BC == 3) {
	        // periodic in x and y directions
	        pSPARC->BCx = 0; pSPARC->BCy = 0; pSPARC->BCz = 1;
	    } else if (pSPARC->BC == 4) {
	        // periodic in x direction
	        pSPARC->BCx = 1; pSPARC->BCy = 1; pSPARC->BCz = 0;
	    } else if (pSPARC->BC > 7) {
	        exit(EXIT_FAILURE);
	    }
	} else if (pSPARC->BCx >= 0 && pSPARC->BCy >= 0 && pSPARC->BCz >= 0) {
		int n_Dirichlet = pSPARC->BCx + pSPARC->BCy + pSPARC->BCz;
		switch (n_Dirichlet) {
			case 0: pSPARC->BC = 2; break;
			case 1: pSPARC->BC = 3; break;
			case 2: pSPARC->BC = 4; break;
			case 3: pSPARC->BC = 1; break;
			default: printf("Error in BC values\n"); break;
		}
	} else {
		// if user does not provide any BC, set default to periodic in all directions
		pSPARC->BC = 2;
		pSPARC->BCx = pSPARC->BCy = pSPARC->BCz = 0;
	}

    FDn = pSPARC->order / 2;    // half the FD order
    // calculate number of finite-difference intervals in case it's provided indirectly
    if (pSPARC->ecut > 0) {
        double h_ecut = Ecut2h(pSPARC->ecut, FDn);
        pSPARC->numIntervals_x = max(round(pSPARC->range_x/h_ecut),FDn);
        pSPARC->numIntervals_y = max(round(pSPARC->range_y/h_ecut),FDn);
        pSPARC->numIntervals_z = max(round(pSPARC->range_z/h_ecut),FDn);
    } else if (pSPARC->mesh_spacing > 0) {
        pSPARC->numIntervals_x = max(round(pSPARC->range_x/pSPARC->mesh_spacing),FDn);
        pSPARC->numIntervals_y = max(round(pSPARC->range_y/pSPARC->mesh_spacing),FDn);
        pSPARC->numIntervals_z = max(round(pSPARC->range_z/pSPARC->mesh_spacing),FDn);
    }

    // calculate number of nodes in each direction
    pSPARC->Nx = pSPARC->numIntervals_x + pSPARC->BCx;
    pSPARC->Ny = pSPARC->numIntervals_y + pSPARC->BCy;
    pSPARC->Nz = pSPARC->numIntervals_z + pSPARC->BCz;
    pSPARC->Nd = pSPARC->Nx * pSPARC->Ny * pSPARC->Nz;

    // mesh size
    pSPARC->delta_x = pSPARC->range_x/(pSPARC->numIntervals_x);
    pSPARC->delta_y = pSPARC->range_y/(pSPARC->numIntervals_y);
    pSPARC->delta_z = pSPARC->range_z/(pSPARC->numIntervals_z);
    pSPARC->dV = pSPARC->delta_x * pSPARC->delta_y * pSPARC->delta_z; // will be multiplied by the Jacobian later

    pSPARC->Jacbdet = 1.0;
    // Compute transformation matrices needed in non cartesian coordinate system and perform atomic coordinate transformation from cartesian to cell coordinates
    if (pSPARC->cell_typ < 20) {
        Cart2nonCart_transformMat(pSPARC);
    }

    // Convert cartesian coordinates of atom positions into cell coordinates
    int atm, count = 0;
    if(pSPARC->cell_typ != 0){
        //Cart2nonCart_transformMat(pSPARC);
        for(i = 0; i < pSPARC->Ntypes; i++){
            if(pSPARC->IsFrac[i] == 0){
                for(atm = 0; atm < pSPARC->nAtomv[i]; atm++) {
                    Cart2nonCart_coord(pSPARC, &pSPARC->atom_pos[3*count], &pSPARC->atom_pos[3*count+1], &pSPARC->atom_pos[3*count+2]);
                    count++;
                }
            } else{
                count += pSPARC->nAtomv[i];
            }
        }
    }

    pSPARC->xin = 0.0; // starting coordinate (global) of the cell in the x-direction

    // Provide number of spin
    if(pSPARC->spin_typ == 0)
        pSPARC->Nspin = 1;
    else
        pSPARC->Nspin = 2;

    // Provide default spin if not already provided
    if(pSPARC->spin_typ != 0) { // spin polarized calculation
        srand(1); // TODO: provide this as a user input
        count = 0;
        for(i = 0; i < pSPARC->Ntypes; i++){
            if(pSPARC->IsSpin[i] == 0){
                for (atm = 0; atm < pSPARC->nAtomv[i]; atm++) {
                    pSPARC->atom_spin[count] = -pSPARC->Znucl[i] + 2 * pSPARC->Znucl[i] * ((double) rand() / RAND_MAX);
                    count++;
                }
            } else{
                count += pSPARC->nAtomv[i];
            }
        }
    }

    // Update the volume of the volume element bounded by finite difference grid
    pSPARC->dV *= pSPARC->Jacbdet;

    // map positions back into the domain if necessary
    if (pSPARC->BCx == 0) {
        double x;
        for (i = 0; i < pSPARC->n_atom; i++) {
            x = *(pSPARC->atom_pos+3*i);
            if (x < 0 || x > pSPARC->range_x)
            {
                x = fmod(x,pSPARC->range_x);
                *(pSPARC->atom_pos+3*i) = x + (x<0.0)*pSPARC->range_x;
            }
        }
    } else if (rank == nproc-1) {
        // for Dirichlet BC, exit if atoms goes out of the domain
        double x;
        for (i = 0; i < pSPARC->n_atom; i++) {
            x = *(pSPARC->atom_pos+3*i);
            if (x < pSPARC->xin || x > pSPARC->range_x + pSPARC->xin)
            {
                printf("\nERROR: position of atom # %d is out of the domain!\n",i+1);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (pSPARC->BCy == 0) {
        double y;
        for (i = 0; i < pSPARC->n_atom; i++) {
            y = *(pSPARC->atom_pos+1+3*i);
            if (y < 0 || y > pSPARC->range_y)
            {
                y = fmod(y,pSPARC->range_y);
                *(pSPARC->atom_pos+1+3*i) = y + (y<0.0)*pSPARC->range_y;
            }
        }
    } else if (rank == nproc-1) {
        // for Dirichlet BC, exit if atoms goes out of the domain
        double y;
        for (i = 0; i < pSPARC->n_atom; i++) {
            y = *(pSPARC->atom_pos+1+3*i);
            if (y < 0 || y > pSPARC->range_y)
            {
                printf("\nERROR: position of atom # %d is out of the domain!\n",i+1);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (pSPARC->BCz == 0) {
        double z;
        for (i = 0; i < pSPARC->n_atom; i++) {
            z = *(pSPARC->atom_pos+2+3*i);
            if (z < 0 || z > pSPARC->range_z)
            {
                z = fmod(z,pSPARC->range_z);
                *(pSPARC->atom_pos+2+3*i) = z + (z<0.0)*pSPARC->range_z;
            }
        }
    } else if (rank == nproc-1){
        // for Dirichlet BC, exit if atoms goes out of the domain
        double z;
        for (i = 0; i < pSPARC->n_atom; i++) {
            z = *(pSPARC->atom_pos+2+3*i);
            if (z < 0 || z > pSPARC->range_z)
            {
                printf("\nERROR: position of atom # %d is out of the domain!\n",i+1);
                exit(EXIT_FAILURE);
            }
        }
    }

#ifdef DEBUG
    if (!rank) {
        printf("\nRange:\n%12.6f\t%12.6f\t%12.6f\n",pSPARC->range_x,pSPARC->range_y,pSPARC->range_z);
        printf("\nCOORD AFTER MAPPING:\n");
        for (i = 0; i < 3 * pSPARC->n_atom; i++) {
            printf("%12.6f\t",pSPARC->atom_pos[i]);
            if (i%3==2 && i>0) printf("\n");
        }
        printf("\n");
    }
#endif
    /* find finite difference weights for first & second derivatives */
    pSPARC->FDweights_D1 = (double *)malloc((FDn + 1) * sizeof(double));
    pSPARC->FDweights_D2 = (double *)malloc((FDn + 1) * sizeof(double));
    pSPARC->D1_stencil_coeffs_x = (double *)malloc((FDn + 1) * sizeof(double));
    pSPARC->D1_stencil_coeffs_y = (double *)malloc((FDn + 1) * sizeof(double));
    pSPARC->D1_stencil_coeffs_z = (double *)malloc((FDn + 1) * sizeof(double));
    pSPARC->D2_stencil_coeffs_x = (double *)malloc((FDn + 1) * sizeof(double));
    pSPARC->D2_stencil_coeffs_y = (double *)malloc((FDn + 1) * sizeof(double));
    pSPARC->D2_stencil_coeffs_z = (double *)malloc((FDn + 1) * sizeof(double));
    if (pSPARC->FDweights_D1 == NULL || pSPARC->FDweights_D1 == NULL ||
        pSPARC->D1_stencil_coeffs_x == NULL || pSPARC->D1_stencil_coeffs_y == NULL ||
        pSPARC->D1_stencil_coeffs_z == NULL || pSPARC->D2_stencil_coeffs_x == NULL ||
        pSPARC->D2_stencil_coeffs_y == NULL || pSPARC->D2_stencil_coeffs_z == NULL) {
        printf("\nmemory cannot be allocated6\n");
        exit(EXIT_FAILURE);
    }

    // 1st derivative weights
    pSPARC->FDweights_D1[0] = 0;
    for (p = 1; p < FDn + 1; p++) {
        pSPARC->FDweights_D1[p] = (2*(p%2)-1) * fract(FDn,p) / p;
    }
    // 2nd derivative weights
    pSPARC->FDweights_D2[0] = 0;
    for (p = 1; p < FDn + 1; p++) {
        pSPARC->FDweights_D2[0] -= (2.0/(p*p));
        pSPARC->FDweights_D2[p] = (2*(p%2)-1) * 2 * fract(FDn,p) / (p*p);
    }

    // 1st derivative weights including mesh
    double dx_inv, dy_inv, dz_inv;
    dx_inv = 1.0 / pSPARC->delta_x;
    dy_inv = 1.0 / pSPARC->delta_y;
    dz_inv = 1.0 / pSPARC->delta_z;
    for (p = 1; p < FDn + 1; p++) {
        pSPARC->D1_stencil_coeffs_x[p] = pSPARC->FDweights_D1[p] * dx_inv;
        pSPARC->D1_stencil_coeffs_y[p] = pSPARC->FDweights_D1[p] * dy_inv;
        pSPARC->D1_stencil_coeffs_z[p] = pSPARC->FDweights_D1[p] * dz_inv;
    }

    // 2nd derivative weights including mesh
    double dx2_inv, dy2_inv, dz2_inv;
    dx2_inv = 1.0 / (pSPARC->delta_x * pSPARC->delta_x);
    dy2_inv = 1.0 / (pSPARC->delta_y * pSPARC->delta_y);
    dz2_inv = 1.0 / (pSPARC->delta_z * pSPARC->delta_z);

    // Stencil coefficients for mixed derivatives
    if(pSPARC->cell_typ == 0){
        for (p = 0; p < FDn + 1; p++) {
            pSPARC->D2_stencil_coeffs_x[p] = pSPARC->FDweights_D2[p] * dx2_inv;
            pSPARC->D2_stencil_coeffs_y[p] = pSPARC->FDweights_D2[p] * dy2_inv;
            pSPARC->D2_stencil_coeffs_z[p] = pSPARC->FDweights_D2[p] * dz2_inv;
        }
    } else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
        pSPARC->D2_stencil_coeffs_xy = (double *)malloc((FDn + 1) * sizeof(double));
        pSPARC->D2_stencil_coeffs_yz = (double *)malloc((FDn + 1) * sizeof(double));
        pSPARC->D2_stencil_coeffs_xz = (double *)malloc((FDn + 1) * sizeof(double));
        pSPARC->D1_stencil_coeffs_xy = (double *)malloc((FDn + 1) * sizeof(double));
        pSPARC->D1_stencil_coeffs_yx = (double *)malloc((FDn + 1) * sizeof(double));
        pSPARC->D1_stencil_coeffs_xz = (double *)malloc((FDn + 1) * sizeof(double));
        pSPARC->D1_stencil_coeffs_zx = (double *)malloc((FDn + 1) * sizeof(double));
        pSPARC->D1_stencil_coeffs_yz = (double *)malloc((FDn + 1) * sizeof(double));
        pSPARC->D1_stencil_coeffs_zy = (double *)malloc((FDn + 1) * sizeof(double));
        if ( pSPARC->D2_stencil_coeffs_xy == NULL || pSPARC->D2_stencil_coeffs_yz == NULL || pSPARC->D2_stencil_coeffs_xz == NULL
            || pSPARC->D1_stencil_coeffs_xy == NULL || pSPARC->D1_stencil_coeffs_yx == NULL || pSPARC->D1_stencil_coeffs_xz == NULL
            || pSPARC->D1_stencil_coeffs_zx == NULL || pSPARC->D1_stencil_coeffs_yz == NULL || pSPARC->D1_stencil_coeffs_zy == NULL) {
            printf("\nmemory cannot be allocated\n");
            exit(EXIT_FAILURE);
        }
        for (p = 0; p < FDn + 1; p++) {
            pSPARC->D2_stencil_coeffs_x[p] = pSPARC->lapcT[0] * pSPARC->FDweights_D2[p] * dx2_inv;
            pSPARC->D2_stencil_coeffs_y[p] = pSPARC->lapcT[4] * pSPARC->FDweights_D2[p] * dy2_inv;
            pSPARC->D2_stencil_coeffs_z[p] = pSPARC->lapcT[8] * pSPARC->FDweights_D2[p] * dz2_inv;
            pSPARC->D2_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_12 d/dx(df/dy)
            pSPARC->D2_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_13 d/dx(df/dz)
            pSPARC->D2_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // 2*T_23 d/dy(df/dz)
            pSPARC->D1_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dy_inv; // d/dx(2*T_12 df/dy) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
            pSPARC->D1_stencil_coeffs_yx[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // d/dy(2*T_12 df/dx) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
            pSPARC->D1_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dz_inv; // d/dx(2*T_13 df/dz) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
            pSPARC->D1_stencil_coeffs_zx[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // d/dz(2*T_13 df/dx) used in d/dz(2*T_13 df/dz + 2*T_23 df/dy)
            pSPARC->D1_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dz_inv; // d/dy(2*T_23 df/dz) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
            pSPARC->D1_stencil_coeffs_zy[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // d/dz(2*T_23 df/dy) used in d/dz(2*T_12 df/dx + 2*T_23 df/dy)
        }
    }

#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
    pSPARC->MaxEigVal_mhalfLap = 0.0; // initialize to 0 to avoid accessing uninitialized variable
    if(pSPARC->cell_typ == 0) {
        // maximum eigenvalue of -0.5 * Lap (only accurate with periodic boundary conditions)
        pSPARC->MaxEigVal_mhalfLap = pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_y[0]
                                     + pSPARC->D2_stencil_coeffs_z[0];
        double scal_x, scal_y, scal_z;
        scal_x = (pSPARC->Nx - pSPARC->Nx % 2) / (double) pSPARC->Nx;
        scal_y = (pSPARC->Ny - pSPARC->Ny % 2) / (double) pSPARC->Ny;
        scal_z = (pSPARC->Nz - pSPARC->Nz % 2) / (double) pSPARC->Nz;
        for (p = 1; p < FDn + 1; p++) {
            pSPARC->MaxEigVal_mhalfLap += 2.0 * (pSPARC->D2_stencil_coeffs_x[p] * cos(M_PI*p*scal_x) 
                                               + pSPARC->D2_stencil_coeffs_y[p] * cos(M_PI*p*scal_y) 
                                               + pSPARC->D2_stencil_coeffs_z[p] * cos(M_PI*p*scal_z));
        }
        pSPARC->MaxEigVal_mhalfLap *= -0.5;
    } else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
        // WARNING: for non-orthogonal cells, this is only a rough estimate, which gives a lowerbound of the accurate answer
        // maximum eigenvalue of -0.5 * Lap (non-orthogonal lattice)
        pSPARC->MaxEigVal_mhalfLap = pSPARC->D2_stencil_coeffs_x[0]  // note lapcT[0] is already multiplied in D2_stencil_coeffs_x above
                                   + pSPARC->D2_stencil_coeffs_y[0]
                                   + pSPARC->D2_stencil_coeffs_z[0];
        double scal_x, scal_y, scal_z;
        scal_x = (pSPARC->Nx - pSPARC->Nx % 2) / (double) pSPARC->Nx;
        scal_y = (pSPARC->Ny - pSPARC->Ny % 2) / (double) pSPARC->Ny;
        scal_z = (pSPARC->Nz - pSPARC->Nz % 2) / (double) pSPARC->Nz;
        for (p = 1; p < FDn + 1; p++) {
            pSPARC->MaxEigVal_mhalfLap += 2.0 * pSPARC->D2_stencil_coeffs_x[p] * cos(M_PI*p*scal_x) 
                                        + 2.0 * pSPARC->D2_stencil_coeffs_y[p] * cos(M_PI*p*scal_y) 
                                        + 2.0 * pSPARC->D2_stencil_coeffs_z[p] * cos(M_PI*p*scal_z);
        }
        // for mixed terms (actually it's a better approx. if we neglect these terms!)
        double sx, sy, sz;
        sx = sy = sz = 0.0;
        for (p = 1; p < FDn + 1; p++) {
            sx += 2.0 * pSPARC->D1_stencil_coeffs_x[p] * sin(M_PI*p*scal_x); // very close to 0 (exactly 0 for even Nx)
            sy += 2.0 * pSPARC->D1_stencil_coeffs_y[p] * sin(M_PI*p*scal_y); // very close to 0 (exactly 0 for even Ny)
            sz += 2.0 * pSPARC->D1_stencil_coeffs_z[p] * sin(M_PI*p*scal_z); // very close to 0 (exactly 0 for even Nz)
        }
        sx = sy = sz = 0.0; // forcing the mixed terms to be zero here!
        pSPARC->MaxEigVal_mhalfLap += 2.0 * pSPARC->lapcT[1] * sx * sy; // x,y
        pSPARC->MaxEigVal_mhalfLap += 2.0 * pSPARC->lapcT[2] * sx * sz; // x,z
        pSPARC->MaxEigVal_mhalfLap += 2.0 * pSPARC->lapcT[5] * sy * sz; // y,z
        pSPARC->MaxEigVal_mhalfLap *= -0.5;
    }
#ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank) printf("Max eigenvalue of -0.5*Lap is %.13f, time taken: %.3f ms\n",pSPARC->MaxEigVal_mhalfLap, (t2-t1)*1e3);
#endif

    // find Chebyshev polynomial degree based on max eigenvalue (spectral width)
    if (pSPARC->ChebDegree < 0) {
        double h_eff = 0.0;
        if (pSPARC->cell_typ == 0) {
            if (fabs(pSPARC->delta_x - pSPARC->delta_y) < 1E-12 &&
                fabs(pSPARC->delta_y - pSPARC->delta_z) < 1E-12) {
                h_eff = pSPARC->delta_x;
            } else {
                // find effective mesh s.t. it has same spectral width
                h_eff = sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
            }
        } else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
            // use max eigenvalue of -1/2*Lap to estimate the effective mesh size for orthogonal Laplacian
            const double lambda_ref = 6.8761754299116333; // max eigval for 1D orthogonal -1.0*Lap for h_eff = 1.0
            h_eff = sqrt(3.0*lambda_ref/(2.0*pSPARC->MaxEigVal_mhalfLap));
        }
        pSPARC->ChebDegree = Mesh2ChebDegree(h_eff);
#ifdef DEBUG
        if (!rank && h_eff < 0.1) {
            printf("#WARNING: for mesh less than 0.1, the default Chebyshev polynomial degree might not be enought!\n");
        }
        if (!rank) printf("h_eff = %.2f, npl = %d\n", h_eff,pSPARC->ChebDegree);
#endif
    }
#ifdef DEBUG
    else
    {
        if (!rank) printf("Chebyshev polynomial degree (provided by user): npl = %d\n",pSPARC->ChebDegree);
    }
#endif

    // set default simple (linear) mixing parameter to be the same as for pulay mixing
    if (pSPARC->MixingParameterSimple < 0.0) {
        pSPARC->MixingParameterSimple = pSPARC->MixingParameter;
    }

    // set default mixing parameter for magnetization density to the same as mixing
    // parameter for total density/potential
    if (pSPARC->MixingParameterMag < 0.0) {
        pSPARC->MixingParameterMag = pSPARC->MixingParameter;
    }

    // set default simple (linear) mixing parameter for magnetization density to be the
    // same as for pulay mixing
    if (pSPARC->MixingParameterSimpleMag < 0.0) {
        pSPARC->MixingParameterSimpleMag = pSPARC->MixingParameterMag;
    }

    if (pSPARC->MixingVariable < 0) { // mixing variable not provided
        pSPARC->MixingVariable = 0; // set default to 'density'
    }

    if (pSPARC->MixingPrecond < 0) { // mixing preconditioner not provided
        pSPARC->MixingPrecond = 1;  // set default to 'Kerker' preconditioner
    }

    if (pSPARC->MixingPrecondMag < 0) { // mixing preconditioner for magnetization not provided
        pSPARC->MixingPrecondMag = 0;  // set default to 'none'
    }

    // set up real space preconditioner coefficients
    if (pSPARC->MixingPrecond == 2) { // coeff for Resta preconditioner
        pSPARC->precondcoeff_n = 1;
        pSPARC->precondcoeff_a = (double complex *)malloc(pSPARC->precondcoeff_n * sizeof(double complex));
        pSPARC->precondcoeff_lambda_sqr = (double complex *)malloc(pSPARC->precondcoeff_n * sizeof(double complex));
        pSPARC->precondcoeff_a[0]          = 0.820368124615603 - 0.330521220406859 * I;
        pSPARC->precondcoeff_lambda_sqr[0] = 1.539184309857566 + 1.454446707757472 * I;
        pSPARC->precondcoeff_k             = 0.179473117041025;

        // check env for what system we're running to set the coeffs,
        // if not provided, the default will be used
        // TODO: remove the following after check!
        char system[L_STRING] = "default";
        char *precond_system = getenv("PRECOND_SYSTEM");
        if (precond_system != NULL)
            snprintf(system, L_STRING, "%s", precond_system);

        if (strcmpi(system,"MoS2") == 0) {
            if (pSPARC->precond_fitpow == 2) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for MoS2 system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif

                pSPARC->precondcoeff_n = 1;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = 0.897326075519806 - 0.837703986538753 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = 0.328766315380339 + 0.183508748834006 * I;
                pSPARC->precondcoeff_k             = 0.102576229227011;
            } else if (pSPARC->precond_fitpow == 4) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for MoS2 system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif
                pSPARC->precondcoeff_n = 3;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = 0.797410061005122 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[1]          = -0.000029265620523 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[2]          = 0.103239343777798 - 0.003381206211038 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = 0.601186842883198 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[1] = -0.256060441488722 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[2] = -0.104178068950438 + 0.493948292725977 * I;
                pSPARC->precondcoeff_k             = 0.099398800263940;
            }
        } else if (strcmpi(system,"Si") == 0) {
            if (pSPARC->precond_fitpow == 2) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for Si system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif
                pSPARC->precondcoeff_n = 1;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = 0.914678024418436 - 1.055347015597097 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = 0.238671535971552 + 0.106323808659314 * I;
                pSPARC->precondcoeff_k             = 0.085289070702772;
            } else if (pSPARC->precond_fitpow == 4) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for Si system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif
                pSPARC->precondcoeff_n = 3;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = -0.000124974499632 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[1]          = 0.822613437367865 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[2]          = 0.094666235811611 - 0.004627781592542 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = -1.072175758908308 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[1] = 0.420975552998538 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[2] = -0.054999300909744 + 0.349588273989346 * I;
                pSPARC->precondcoeff_k             = 0.082856817316465;
            }
        } else if (strcmpi(system,"C") == 0) {
            if (pSPARC->precond_fitpow == 2) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for C system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif
                pSPARC->precondcoeff_n = 1;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = 0.8206 - 0.3427 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = 0.4284 + 0.4019 * I;
                pSPARC->precondcoeff_k             = 0.1793;
            } else if (pSPARC->precond_fitpow == 4) {
                #ifdef DEBUG
                if (!rank) printf(RED "WARNING: coeffs for C system with fitpow %d are not set!\n" RESET, pSPARC->precond_fitpow);
                #endif
                // pSPARC->precondcoeff_n = 3;
                // // reallocate memory
                // pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                // pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                // pSPARC->precondcoeff_a[0]          = -0.000124974499632 + 0.000000000000000 * I;
                // pSPARC->precondcoeff_a[1]          = 0.822613437367865 + 0.000000000000000 * I;
                // pSPARC->precondcoeff_a[2]          = 0.094666235811611 - 0.004627781592542 * I;
                // pSPARC->precondcoeff_lambda_sqr[0] = -1.072175758908308 + 0.000000000000000 * I;
                // pSPARC->precondcoeff_lambda_sqr[1] = 0.420975552998538 + 0.000000000000000 * I;
                // pSPARC->precondcoeff_lambda_sqr[2] = -0.054999300909744 + 0.349588273989346 * I;
                // pSPARC->precondcoeff_k             = 0.082856817316465;
            }
        }
    } else if (pSPARC->MixingPrecond == 3) { // coeff for truncated Kerker preconditioner
        pSPARC->precondcoeff_n = 1;
        pSPARC->precondcoeff_a = (double complex *)malloc(pSPARC->precondcoeff_n * sizeof(double complex));
        pSPARC->precondcoeff_lambda_sqr = (double complex *)malloc(pSPARC->precondcoeff_n * sizeof(double complex));
        pSPARC->precondcoeff_a[0]          = 0.740197283447608 - 2.187940485005530 * I;
        pSPARC->precondcoeff_lambda_sqr[0] = 0.515764278984552 + 0.261718938132583 * I;
        pSPARC->precondcoeff_k             = 0.259680843800232;

        // check env for what system we're running to set the coeffs,
        // it not provided, the default will be used
        // TODO: remove the following after check!
        char system[L_STRING] = "default";
        char *precond_system = getenv("PRECOND_SYSTEM");
        if (precond_system != NULL)
            snprintf(system, L_STRING, "%s", precond_system);

        if (strcmpi(system,"MoS2") == 0) {
            if (pSPARC->precond_fitpow == 2) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for MoS2 system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif
                pSPARC->precondcoeff_n = 2;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = 1.069131757115932 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[1]          = -0.171827850593795 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = 0.261519729188790 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[1] = 0.024058288033320 + 0.000000000000000 * I;
                pSPARC->precondcoeff_k             = 0.102669136088733;
            } else if (pSPARC->precond_fitpow == 4) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for MoS2 system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif
                pSPARC->precondcoeff_n = 3;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = 0.000011385765477 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[1]          = 0.994255001880647 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[2]          = -0.093994967542657 - 0.006240439304379 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = -0.580143676837624 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[1] = 0.281390031341584 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[2] = -0.005192385910338 + 0.009670637051448 * I;
                pSPARC->precondcoeff_k             = 0.099729735832187;
            }
        } else if (strcmpi(system,"Si") == 0) {
            if (pSPARC->precond_fitpow == 2) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for Si system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif
                pSPARC->precondcoeff_n = 2;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = 1.045423322787217 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[1]          = -0.130145326907590 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = 0.267115428215830 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[1] = 0.019530203373891 + 0.000000000000000 * I;
                pSPARC->precondcoeff_k             = 0.084702403406033;
            } else if (pSPARC->precond_fitpow == 4) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for Si system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif
                pSPARC->precondcoeff_n = 3;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = -0.000450002447564 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[1]          = 0.991616958994114 + 0.000000000000000 * I;
                pSPARC->precondcoeff_a[2]          = -0.074468796694241 - 0.014060128507695 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = 3.578501584073372 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[1] = 0.283063390321347 + 0.000000000000000 * I;
                pSPARC->precondcoeff_lambda_sqr[2] = -0.004905277505535 + 0.011599970024290 * I;
                pSPARC->precondcoeff_k             = 0.083301273707655;
            }
        } else if (strcmpi(system,"C") == 0) {
            if (pSPARC->precond_fitpow == 2) {
                #ifdef DEBUG
                if (!rank) printf(RED "Using coeffs for C system with fitpow %d\n" RESET, pSPARC->precond_fitpow);
                #endif
                pSPARC->precondcoeff_n = 2;
                // reallocate memory
                pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                pSPARC->precondcoeff_a[0]          = 1.2926 + 0.0000 * I;
                pSPARC->precondcoeff_a[1]          = -0.4780 + 0.0000 * I;
                pSPARC->precondcoeff_lambda_sqr[0] = 0.2310 + 0.0000 * I;
                pSPARC->precondcoeff_lambda_sqr[1] = 0.0552 + 0.0000 * I;
                pSPARC->precondcoeff_k             = 0.1854;
            } else if (pSPARC->precond_fitpow == 4) {
                #ifdef DEBUG
                if (!rank) printf(RED "WARNING: coeffs for C system with fitpow %d are not set!\n" RESET, pSPARC->precond_fitpow);
                #endif
                // pSPARC->precondcoeff_n = 3;
                // // reallocate memory
                // pSPARC->precondcoeff_a = realloc(pSPARC->precondcoeff_a, pSPARC->precondcoeff_n * sizeof(double complex));
                // pSPARC->precondcoeff_lambda_sqr = realloc(pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_n * sizeof(double complex));
                // pSPARC->precondcoeff_a[0]          = -0.000124974499632 + 0.000000000000000 * I;
                // pSPARC->precondcoeff_a[1]          = 0.822613437367865 + 0.000000000000000 * I;
                // pSPARC->precondcoeff_a[2]          = 0.094666235811611 - 0.004627781592542 * I;
                // pSPARC->precondcoeff_lambda_sqr[0] = -1.072175758908308 + 0.000000000000000 * I;
                // pSPARC->precondcoeff_lambda_sqr[1] = 0.420975552998538 + 0.000000000000000 * I;
                // pSPARC->precondcoeff_lambda_sqr[2] = -0.054999300909744 + 0.349588273989346 * I;
                // pSPARC->precondcoeff_k             = 0.082856817316465;
            }
        }
    }

    // scf error type, 0 - default, 1 - QE (conv_thr)
    if (pSPARC->scf_err_type != 0 && pSPARC->scf_err_type != 1) {
        if (!rank) printf("Cannot recognize SCF error type!\n");
        exit(EXIT_FAILURE);
    }

    // for evaluating QE scf error, we need to perform some extra calculations
    // e.g., an extra Poisson solve, this timer keeps track of the extra time spent
    if (pSPARC->scf_err_type == 1) {
        pSPARC->t_qe_extra = 0.0;
    }

    // default SCF tolerance based on accuracy_level
    // we use a model curve to correlate scf tolerance and energy and force accuracy
    //     log10(y) = a * log10(x) + b' + c * log10(Nelectron/n_atom)
    // (1) if y is energy accuracy (Ha/atom), a = 2.02, b' = -0.25440, c = 1.0000
    // (2) if y is force accuracy  (Ha/Bohr), a = 1.03, b' = -0.33145, c = 1.1761
    // if scf tol is not set, we'll use accuracy_level to find scf tol
    if (pSPARC->TOL_SCF < 0.0) {
        const double log10_neatom = log10(pSPARC->Nelectron / (double) pSPARC->n_atom);
        double target_force_accuracy = -1.0;
        double target_energy_accuracy = -1.0;

        // accuracy_levels      : 0 - minimal | 1 - low  | 2 - medium | 3 - high | 4 - extreme
        // target force accuracy: 0 - 1e-1    | 1 - 1e-2 | 2 - 1e-3   | 3 - 1e-4 | 4 - 1e-5
        if (pSPARC->accuracy_level >= 0) {
            target_force_accuracy = pow(10.0, -(pSPARC->accuracy_level + 1.0));
        } else if (pSPARC->target_force_accuracy > 0.0) {
            target_force_accuracy = pSPARC->target_force_accuracy;
        } else if (pSPARC->target_energy_accuracy > 0.0) {
            target_energy_accuracy = pSPARC->target_energy_accuracy;
        }

        // if none of the accuracy levels are specified, set force_accuracy to 1e-3
        if (target_force_accuracy < 0  && target_energy_accuracy < 0) {
            target_force_accuracy = 1e-3;
        }

        // calculate SCF TOL based on specified target accuracy
        if (target_force_accuracy > 0.0) { // find scf tol based on target force accuracy
            const double a = 1.03;
            const double b = -0.33145 + 1.1761 * log10_neatom;
            double log10_target = log10(target_force_accuracy);
            pSPARC->TOL_SCF = pow(10.0, (log10_target - b)/a);
        } else if (target_energy_accuracy > 0.0) { // find scf tol based on target energy accuracy
            const double a = 2.02;
            const double b = -0.25440 + 1.0000 * log10_neatom;
            double log10_target = log10(target_energy_accuracy);
            pSPARC->TOL_SCF = pow(10.0, (log10_target - b)/a);
        }
    }

    // default Kerker tolerance
    if (pSPARC->TOL_PRECOND < 0.0) { // kerker tol not provided by user
        double h_eff = 0.0;
        if (fabs(pSPARC->delta_x - pSPARC->delta_y) < 1E-12 &&
            fabs(pSPARC->delta_y - pSPARC->delta_z) < 1E-12) {
            h_eff = pSPARC->delta_x;
        } else {
            // find effective mesh s.t. it has same spectral width
            h_eff = sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
        }
        pSPARC->TOL_PRECOND = (h_eff * h_eff) * 1e-3;
    }

    // default poisson tolerance
    // In most cases this is enough. The error in energy/atom due to poisson
    // error is usually 1~2 orders of magnitude smaller than the poisson tol.
    // And the error in forces due to poisson error is of 1 order of magnitude
    // larger than poisson tol.
    // Another issue about using moderate poisson tolerance is SCF convergence.
    // we found that usually keeping poisson tol as 1 order of magnitude lower
    // than scf_tol, the SCF convergence won't be affected. However, if SCF
    // stagers near convergence, consider reducing poisson tolerance further
    // (e.g. scf_tol*0.01).
    if (pSPARC->TOL_POISSON < 0.0) { // if poisson tol not provided by user
        pSPARC->TOL_POISSON = pSPARC->TOL_SCF * 0.01;
    }

    // default rb tolerance
    // Note that rb tolerance is the absolute tolerance. 
    // The error in energy/atom due to rb error is usually 1~2 orders of
    // magnitude smaller than rb tol. While the error in force due to rb
    // error is roughly the same order as rb tol.
    if (pSPARC->TOL_PSEUDOCHARGE < 0.0) { // if rb tol not provided by user
        pSPARC->TOL_PSEUDOCHARGE = pSPARC->TOL_SCF * 0.01;
    }

    // The following will override the user-provided FixRandSeed
    // check env for FixRandSeed variable: 0 - off, 1 - on
    int FixRandSeed = pSPARC->FixRandSeed;
    char *env_var = getenv("FIX_RAND");
    if (env_var != NULL)
        FixRandSeed = atoi(env_var);
    pSPARC->FixRandSeed = FixRandSeed;

    // allocate memory for pseudocharge cutoff radius
    pSPARC->CUTOFF_x = (double *)malloc(Ntypes * sizeof(double));
    pSPARC->CUTOFF_y = (double *)malloc(Ntypes * sizeof(double));
    pSPARC->CUTOFF_z = (double *)malloc(Ntypes * sizeof(double));

    if (pSPARC->CUTOFF_x == NULL || pSPARC->CUTOFF_y == NULL ||
        pSPARC->CUTOFF_z == NULL) {
        printf("\nmemory cannot be allocated7\n");
        exit(EXIT_FAILURE);
    }

    // number of k-points after symmetry reduction (currently only
    // takes into account the time reversal symmetry)
    // WARNING: Time-reversal symmetry only if there is no magnetic field applied
    // this won't work with kpt shift
    // pSPARC->Nkpts_sym = ceil(pSPARC->Kx*pSPARC->Ky*pSPARC->Kz/2.0); 
    pSPARC->Nkpts_sym = pSPARC->Nkpts; // will be updated after actual symmetry reduction

    // at this point symmetry reduction is not done yet
    pSPARC->kptWts = (double *)malloc(pSPARC->Nkpts_sym * sizeof(double));
    pSPARC->k1 = (double *)malloc(pSPARC->Nkpts_sym * sizeof(double));
    pSPARC->k2 = (double *)malloc(pSPARC->Nkpts_sym * sizeof(double));
    pSPARC->k3 = (double *)malloc(pSPARC->Nkpts_sym * sizeof(double));
    // calculate the k point and weights (shift in kpt may apply)
    Calculate_kpoints(pSPARC);

    // flag to indicate if it is a gamma-point calculation
    pSPARC->isGammaPoint = (int)(pSPARC->Nkpts_sym == 1 
        && fabs(pSPARC->k1[0]) < TEMP_TOL 
        && fabs(pSPARC->k2[0]) < TEMP_TOL 
        && fabs(pSPARC->k3[0]) < TEMP_TOL
        && pSPARC->SOC_Flag == 0);

    if (pSPARC->vdWDFFlag != 0){
        if ((pSPARC->BCx)||(pSPARC->BCy)||(pSPARC->BCz)) {
            if (rank == 0)
                printf(RED "ERROR: vdW-DF does not support Dirichlet boundary condition!\n" RESET);
            exit(EXIT_FAILURE); 
        }
    }

    #if !defined(USE_MKL) && !defined(USE_FFTW)
    if (pSPARC->vdWDFFlag != 0){
        if (rank == 0)
            printf(RED "ERROR: To use vdW-DF, please turn on MKL or FFTW in makefile!\n"
            "Or you can stop using vdW-DF by setting other exchange-correlation functionals.\n" RESET);
        exit(EXIT_FAILURE); 
    }
    if (pSPARC->usefock == 1){
        if (rank == 0)
            printf(RED "ERROR: To use hybrid functionals like PBE0 or HF, please turn on MKL or FFTW in makefile!\n" RESET);
        exit(EXIT_FAILURE); 
    }
    #endif // #if !defined(USE_MKL) && !defined(USE_FFTW)

    if (pSPARC->mGGAflag == 1) {
        // if (pSPARC->spin_typ != 0) {
        //     if (rank == 0) 
        //         printf(RED "ERROR: currently SCAN does not support spin polarization!\n" RESET);
        //     exit(EXIT_FAILURE); 
        // }
        if (pSPARC->SOC_Flag || pSPARC->usefock || pSPARC->SQFlag) {
            if (!rank) 
                printf(RED "ERROR: Spin-orbit coupling, hybrid and SQ are not supported in this version of SCAN implementation.\n" RESET);
            exit(EXIT_FAILURE);
        }
    }

#if !defined(USE_MKL) && !defined(USE_SCALAPACK)
    if (pSPARC->usefock == 1 && pSPARC->ACEFlag == 1){
        if (rank == 0)
            printf(RED "ERROR: To use hybrid functional with ACE method, please turn on MKL or SCALAPACK in makefile!\n"RESET);
        exit(EXIT_FAILURE);
    }
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    if (pSPARC->usefock == 1) {
        if (pSPARC->SOC_Flag || pSPARC->SQFlag) {
            if (!rank) 
                printf(RED "ERROR: Spin-orbit coupling and SQ are not supported in this version of hybrid implementation.\n" RESET);
            exit(EXIT_FAILURE);
        }
        if (pSPARC->EXXDiv_Flag < 0) {
            if (pSPARC->BC > 2) {
                #ifdef DEBUG
                if (!rank) 
                    printf(RED "For Wire and Slab with hybrid funcitonal, deafults to use auxiliary functioin method.\n" RESET);  
                #endif
                pSPARC->EXXDiv_Flag = 1;
            } else {
                if (strcmpi(pSPARC->XC,"HSE") == 0) {
                    #ifdef DEBUG
                    if (!rank) 
                        printf(RED "For Bulk and Cluster with HSE hybrid funcitonal, deafults to use ERFC method.\n" RESET);  
                    #endif
                    pSPARC->EXXDiv_Flag = 2;
                } else {
                    #ifdef DEBUG
                    if (!rank) 
                        printf(RED "For Bulk and Cluster with hybrid funcitonal, deafults to use spherical truncation method.\n" RESET);  
                    #endif
                    pSPARC->EXXDiv_Flag = 0;
                }
            }
        } else {
            if (strcmpi(pSPARC->XC,"HSE") != 0 && pSPARC->EXXDiv_Flag == 2) {
                printf(RED "ERROR: ERFC method could only be used with HSE functional.\n" RESET);
                exit(EXIT_FAILURE);
            }
        }
        
        if (pSPARC->TOL_FOCK < 0.0) {
            // TODO: This model is based on the results of 5 tests. Do more tests to improve the robustness. 
            // default FOCK outer loop tolerance based on accuracy_level
            // we use a model curve to correlate fock tolerance and energy and force accuracy
            //     log10(y) = a * log10(x) + b, a = 1, b = 0.3
            // when related to TOL_SCF, this could be simplified as 0.2 * TOL_SCF for the same force accuracy.
            pSPARC->TOL_FOCK = 0.2*pSPARC->TOL_SCF;
        }
    
        // If initial PBE SCF tolerance is not defined, use default 10*pSPARC->TOL_FOCK
        if (pSPARC->TOL_SCF_INIT < 0.0) {
            pSPARC->TOL_SCF_INIT = max(10*pSPARC->TOL_FOCK,1e-3);
        }
        pSPARC->MAXIT_FOCK = max(1,pSPARC->MAXIT_FOCK);
        pSPARC->MINIT_FOCK = max(1,pSPARC->MINIT_FOCK);
        
        if (pSPARC->EXXMem_batch < 0) {
            // use default EXXMem_batch if it's negative
            pSPARC->EXXMem_batch = 0;
        }

        // If using ACE operator, only do domain parallelization
        if (pSPARC->ACEFlag == 1) {
            if (pSPARC->EXXACEVal_state < 0) {
                // Use default EXXACEVal_state if it's negative
                pSPARC->EXXACEVal_state = 3;
            }
        } else {
            pSPARC->EXXACEVal_state = 0;
        }
    } else {
        pSPARC->ACEFlag = 0;
        pSPARC->EXXMem_batch = 0;
        pSPARC->EXXACEVal_state = 0;
    }

    // constraints on SOC 
    if (pSPARC->SOC_Flag == 1) {
        if (pSPARC->usefock || pSPARC->mGGAflag || pSPARC->SQFlag) {
            if (rank == 0) 
                printf(RED "ERROR: Hybrid functional, SCAN and SQ are not supported in this version of spin-orbit coupling implementation.\n" RESET);
            exit(EXIT_FAILURE);
        }
        if (pSPARC->spin_typ == 1) {
            if (rank == 0) 
                printf(RED "ERROR: Spin-polarized calculation is not supported in this version of spin-orbit coupling implementation.\n" RESET);
            exit(EXIT_FAILURE);
        }
    }

    // constraints on SQ
    if (pSPARC->SQFlag == 1) {
        if (pSPARC->BCx || pSPARC->BCy || pSPARC->BCz) {
            if (!rank) 
                printf(RED "ERROR: SQ method only supports periodic boundary conditions.\n" RESET);
            exit(EXIT_FAILURE);
        }
        if (pSPARC->cell_typ != 0) {
            if (!rank) 
                printf(RED "ERROR: SQ method only supports orthogonal systems in this version.\n" RESET);
            exit(EXIT_FAILURE);
        }
        if (pSPARC->isGammaPoint != 1 || pSPARC->spin_typ == 1) {
            if (rank == 0)
                printf(RED "ERROR: Polarized calculation and Kpoint options are not supported in this version of SQ implementation.\n" RESET);
            exit(EXIT_FAILURE);
        }
        if (pSPARC_Input->Nstates != -1) {
            if (rank == 0)
                printf(RED "ERROR: NSTATES is not vaild in SQ method.\n" RESET);
            exit(EXIT_FAILURE);
        }
        // Only Gauss Quadrature has been implemented.
        // TODO: Add Clenshaw Curtis method and SQ_typ input option. 
        pSPARC->SQ_typ = 2;
        if (pSPARC->SOC_Flag || pSPARC->usefock || pSPARC->mGGAflag) {
            if (!rank) 
                printf(RED "ERROR: Hybrid functional, spin-orbit coupling, and SCAN are not supported in this version of SQ implementation." RESET);
            exit(EXIT_FAILURE);
        }
        if (pSPARC->SQ_rcut < 0) {
            if (!rank)
                printf(RED "ERROR: SQ_RCUT must be provided when SQ method is turned on.\n" RESET);
            exit(EXIT_FAILURE);
        }
        // if (pSPARC->SQ_typ == 1) {
            // npl_c is set to nearest factor of 4 to facilitate Cleanshaw Curtis for Energy
            // pSPARC->SQ_npl_c = (int) ceil(pSPARC->SQ_fac_g2c * pSPARC->SQ_npl_g / 4.0) * 4;
        // }
        if (pSPARC->SQ_typ == 2) {
            if (pSPARC->SQ_npl_g <= 0) {
                if (!rank)
                    printf(RED "ERROR: SQ_NPL_G must be provided a positive integer when Gauss Quadrature method is turned on in SQ method.\n" RESET);
                exit(EXIT_FAILURE);
            }

            if (pSPARC->SQ_npl_c <= 0) {
                if (pSPARC->SQ_fac_g2c <= 0) {
                    if (!rank)
                        printf(RED "ERROR: SQ_FAC_G2C must be positive when Gauss Quadrature method is turned on \n"
                                "and npl_c is not provided correctly in SQ method.\n" RESET);
                    exit(EXIT_FAILURE);
                }
                // If npl_c is not provided, then npl_c = fac_g2c * npl_g. 
                pSPARC->SQ_npl_c = (int) ceil(pSPARC->SQ_fac_g2c * pSPARC->SQ_npl_g);
            } else {
                // if npl_c is provided correctly, then SQ_FAC_G2C is set to negative and not printed 
                pSPARC->SQ_fac_g2c = -1;
            }
        }
        if (pSPARC->PrintEigenFlag > 0) {
            if (!rank)
                printf(RED "ERROR: PRINT_EIGEN is not valid in SQ method.\n" RESET);
                exit(EXIT_FAILURE);
        }
        pSPARC->SQ_correction = 0;          // The correction term in energy and forces hasn't been implemented in this version.
        if (pSPARC->SQ_typ_dm == 1) {
            pSPARC->SQ_gauss_mem = 0;       // Ensure the high memory option is only valid for Gauss Qaudrature
        }
    }

    if (pSPARC->PrintPsiFlag[0] == 1 && pSPARC->PrintPsiFlag[1] < 0) {
        pSPARC->PrintPsiFlag[1] = 0; pSPARC->PrintPsiFlag[2] = pSPARC->Nspin-1;     // spin start/end index
        pSPARC->PrintPsiFlag[3] = 0; pSPARC->PrintPsiFlag[4] = pSPARC->Nkpts-1;     // k-point start/end index
        pSPARC->PrintPsiFlag[5] = 0; pSPARC->PrintPsiFlag[6] = pSPARC->Nstates-1;   // band start/end index
    }
}


/**
 * @brief   Estimate the memory required for the simulation.
 */
double estimate_memory(const SPARC_OBJ *pSPARC) {
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (pSPARC->SQFlag == 1) {
        SQ_OBJ *pSQ = pSPARC->pSQ;
        // TODO: add accurate memory estimation
        double mem_PR, mem_Rcut, mem_phi, mem_chi;
        mem_PR = (double) sizeof(double) * pSQ->Nd_PR;
        mem_Rcut = 0;
        mem_phi = (double) sizeof(double) * pSPARC->Nd_d * (6+pSPARC->SQ_npl_g*2+pSPARC->SQ_npl_c+1);
        mem_chi = (double) sizeof(double) * pSPARC->Nd_d * pSQ->Nd_loc * 0.3; 
        if (pSPARC->SQ_typ_dm == 2) { 
            if (pSPARC->SQ_gauss_mem == 1) {                    // save vectors for all FD nodes
                mem_Rcut += (double) sizeof(double) * pSPARC->Nd_d * pSQ->Nd_loc * pSPARC->SQ_npl_g;
                mem_phi += (double) sizeof(double) * pSPARC->Nd_d * pSPARC->SQ_npl_g * pSPARC->SQ_npl_g;
            } else {
                mem_Rcut += (double) sizeof(double) * pSQ->Nd_loc * pSPARC->SQ_npl_g;
            }
        } else if (pSPARC->SQ_typ_dm == 1) {
            mem_Rcut += (double) sizeof(double) * pSPARC->Nd_d * pSQ->Nd_loc;
        }

        double memory_usage = 0.0;
        memory_usage = mem_PR + mem_Rcut + mem_phi + mem_chi;

        #ifdef DEBUG
        if (rank == 0) {
            char mem_str[32];
            printf("----------------------\n");
            printf("Estimated memory usage\n");
            formatBytes(memory_usage, 32, mem_str);
            printf("Total: %s\n", mem_str);
            formatBytes(mem_PR, 32, mem_str);
            printf("Vectors in P.R. domain : %s\n", mem_str);
            formatBytes(mem_Rcut, 32, mem_str);
            printf("Vectors in Rcut domain : %s\n", mem_str);
            formatBytes(mem_phi, 32, mem_str);
            printf("Vectors in phi domain : %s\n", mem_str);
            formatBytes(mem_chi, 32, mem_str);
            printf("All saved nonlocal projectors: %s\n", mem_str);
            printf("----------------------------------------------\n");
            formatBytes(memory_usage/nproc,32,mem_str);
            printf("Estimated memory usage per processor: %s\n",mem_str);
        }
        #endif
        return memory_usage;
    }
    int Nd = pSPARC->Nd * pSPARC->Nspinor;
    int Ns = pSPARC->Nstates;
    int Nspin = pSPARC->Nspin;
    int Nkpts_sym = pSPARC->Nkpts_sym;
    int m = pSPARC->MixingHistory;

    int type_size;
    if (pSPARC->isGammaPoint) {
        type_size = sizeof(double);
    } else {
        type_size = sizeof(double complex);
    }

    // orbitals (dominant)
    int ncpy_orbitals = 6; // copies required during chebyshev filtering
    double memory_orbitals = (double) ncpy_orbitals * Nd * Ns * Nspin * Nkpts_sym * type_size;

    // vectors: rho, phi, Veff, mixing history vectors, etc.
    int ncpy_vectors = 6 + 4 * Nspin + 2 * m * Nspin + 3 * (2*Nspin-1) + 1;
    double memory_vectors = (double) ncpy_vectors * Nd * sizeof(double);

    // total memory
    double memory_usage = memory_orbitals + memory_vectors;

    // memory for Exact Exchange part
    double memory_exx = 0.0;
    if (pSPARC->usefock > 0) {
        memory_exx = estimate_memory_exx(pSPARC);
        memory_usage += memory_exx;
    }    

    #ifdef DEBUG
    if (rank == 0) {
        char mem_str[32];
        printf("----------------------\n");
        printf("Estimated memory usage\n");
        formatBytes(memory_usage, 32, mem_str);
        printf("Total: %s\n", mem_str);
        formatBytes(memory_orbitals, 32, mem_str);
        printf("orbitals             : %s\n", mem_str);
        formatBytes(memory_vectors, 32, mem_str);
        printf("global sized vectors : %s\n", mem_str);
        if (pSPARC->usefock > 0) {
            formatBytes(memory_exx, 32, mem_str);
            printf("global exact exchange memory : %s\n", mem_str);
        }
        printf("----------------------------------------------\n");
        formatBytes(memory_usage/nproc,32,mem_str);
        printf("Estimated memory usage per processor: %s\n",mem_str);
    }
    #endif
    return memory_usage;
}



/**
 * @brief   Find equivalent mesh size to a given Ecut.
 *
 * @param Ecut  Energy cutoff used in plane-wave codes, in Hartree.
 * @param FDn   Finite difference order divided by 2.
 */
double Ecut2h(double Ecut, int FDn) {
    double epsilon = 0.1;
    double *w2;
    w2 = (double *)malloc((FDn+1) * sizeof(double));

    // 2nd derivative weights
    w2[0] = 0;
    for (int p = 1; p < FDn + 1; p++) {
        w2[0] -= (2.0/(p*p));
        w2[p] = (2*(p%2)-1) * 2 * fract(FDn,p) / (p*p);
    }

    // k grid within interval (0,pi]
    int N = 1000;
    double dk = M_PI / (double)N;
    double *kk = (double *)malloc(N * sizeof(double));
    double *y_fd = (double *)malloc(N * sizeof(double));
    double *k2   = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        kk[i] = (i + 1) * dk;
        k2[i] = kk[i] * kk[i];
    }

    // y_fd(k) = -w[0] + sum_{p=1}^{FDn} -2*w[p]*cos(k*p) (assuming h = 1)
    for (int i = 0; i < N; i++) {
        y_fd[i] = -w2[0];
        for (int p = 1; p < FDn + 1; p++) {
            y_fd[i] -= 2 * w2[p] * cos(kk[i] * p); 
        }
    }

    // find out at which point |k^2 - y_fd| > epsilon
    int ind = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(y_fd[i] - k2[i]) > epsilon) {
            ind = i;
            break;
        }
    }

    // k_cutoff
    double k_cutoff = kk[ind];

    double h = k_cutoff / sqrt(2.0*Ecut);

    free(kk);
    free(y_fd);
    free(k2);
    free(w2);

    return h;
}




/**
 * @brief   Calculate the weight of a given k-point.
 */
double kpointWeight(double kx,double ky,double kz) {
    /*
    * Appropriate weights for k-point sampling
    * we find the weight of the x-direction k point. If the kx has 0, then the weight is 1.0
    * else the weight is 2.0
    */
    double w;
    if(fabs(kx)>0) {
        w=2.0;
    } else {
        w=1.0;
    }
    return w;
}


/**
 * @brief   Calculate k-points and the associated weights.
 */
void Calculate_kpoints(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Initialize the weights of k points
    // by going over all the k points from the Monkhorst Pack grid
    int k,nk1,nk2,nk3,k_hf,k_hf_rd;
    double k1,k2,k3;
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    k = k_hf = k_hf_rd = 0;
    
    if (pSPARC->usefock == 1) {
        // constrains on EXX_DOWNSAMPLING
        if (pSPARC->EXXDownsampling[0] < 0 || pSPARC->EXXDownsampling[1] < 0 || pSPARC->EXXDownsampling[2] < 0) {
            if (rank == 0)
                printf(RED "ERROR: EXX_DOWNSAMPLING must be non-negative.\n" RESET);
            exit(EXIT_FAILURE);
        }

        if ((pSPARC->EXXDownsampling[0] > 0 && pSPARC->Kx % pSPARC->EXXDownsampling[0]) ||
            (pSPARC->EXXDownsampling[1] > 0 && pSPARC->Ky % pSPARC->EXXDownsampling[1]) ||
            (pSPARC->EXXDownsampling[2] > 0 && pSPARC->Kz % pSPARC->EXXDownsampling[2])) {
            if (rank == 0)
                printf(RED "ERROR: Number of kpoints must be divisible by EXX_DOWNSAMPLING in all directions if EXX_DOWNSAMPLING is positive.\n" RESET);
            exit(EXIT_FAILURE);
        }

        if (pSPARC->EXXDownsampling[0] == 0) {
            pSPARC->Nkpts_hf = 1;
            pSPARC->Kx_hf = 1;
        } else {
            pSPARC->Kx_hf = pSPARC->Kx / pSPARC->EXXDownsampling[0];
            pSPARC->Nkpts_hf = pSPARC->Kx / pSPARC->EXXDownsampling[0];
        }

        if (pSPARC->EXXDownsampling[1] == 0) {
            pSPARC->Nkpts_hf *= 1;
            pSPARC->Ky_hf = 1;
        } else {
            pSPARC->Ky_hf = pSPARC->Ky / pSPARC->EXXDownsampling[1];
            pSPARC->Nkpts_hf *= (pSPARC->Ky / pSPARC->EXXDownsampling[1]);
        }

        if (pSPARC->EXXDownsampling[2] == 0) {
            pSPARC->Nkpts_hf *= 1;
            pSPARC->Kz_hf = 1;
        } else {
            pSPARC->Kz_hf = pSPARC->Kz / pSPARC->EXXDownsampling[2];
            pSPARC->Nkpts_hf *= (pSPARC->Kz / pSPARC->EXXDownsampling[2]);
        }
        pSPARC->k1_hf = (double *) calloc(sizeof(double), pSPARC->Nkpts_hf);
        pSPARC->k2_hf = (double *) calloc(sizeof(double), pSPARC->Nkpts_hf);
        pSPARC->k3_hf = (double *) calloc(sizeof(double), pSPARC->Nkpts_hf);
        pSPARC->kpthf_ind = (int *) calloc(sizeof(int), pSPARC->Nkpts_hf);
        pSPARC->kpthf_ind_red = (int *) calloc(sizeof(int), pSPARC->Nkpts_hf);
        pSPARC->kpthf_pn  = (int *) calloc(sizeof(int), pSPARC->Nkpts_hf);
        pSPARC->kptWts_hf = 1.0 / pSPARC->Nkpts_hf;
    }

    double sumx = 2.0 * M_PI / Lx;
    double sumy = 2.0 * M_PI / Ly;
    double sumz = 2.0 * M_PI / Lz;

    int nk, flag;
    int flag_0x, flag_0y, flag_0z;                      // flag for finding 0 k-point in 3 directions when exx_downsampling is 0
    int flag_cx, flag_cy, flag_cz;                      // flag for correct slot in x,y,z direction
    flag_0x = flag_0y = flag_0z = 0;

    // calculate M-P grid similar to that in ABINIT
    int nk1_s = -floor((pSPARC->Kx - 1)/2);
    int nk1_e = nk1_s + pSPARC->Kx;
    int nk2_s = -floor((pSPARC->Ky - 1)/2);
    int nk2_e = nk2_s + pSPARC->Ky;
    int nk3_s = -floor((pSPARC->Kz - 1)/2);
    int nk3_e = nk3_s + pSPARC->Kz;
    
    for (nk1 = nk1_s; nk1 < nk1_e; nk1++) {
        for (nk2 = nk2_s; nk2 < nk2_e; nk2++) {
            for (nk3 = nk3_s; nk3 < nk3_e; nk3++) {
                double k1_red, k2_red, k3_red;
                // calculate Monkhorst-Pack k points (reduced) using Monkhorst pack grid
                k1_red = nk1 * 1.0/pSPARC->Kx;
                k2_red = nk2 * 1.0/pSPARC->Ky;
                k3_red = nk3 * 1.0/pSPARC->Kz;
                k1_red = fmod(k1_red + pSPARC->kptshift[0] / pSPARC->Kx + 0.5 - TEMP_TOL, 1.0) - 0.5 + TEMP_TOL;
                k2_red = fmod(k2_red + pSPARC->kptshift[1] / pSPARC->Ky + 0.5 - TEMP_TOL, 1.0) - 0.5 + TEMP_TOL;
                k3_red = fmod(k3_red + pSPARC->kptshift[2] / pSPARC->Kz + 0.5 - TEMP_TOL, 1.0) - 0.5 + TEMP_TOL;
#ifdef DEBUG
                if (!rank) printf(BLU "[k1_red,k2_red,k3_red] = %8.4f %8.4f %8.4f\n" RESET, k1_red, k2_red, k3_red);
#endif
                k1 = k1_red * 2.0 * M_PI / Lx;
                k2 = k2_red * 2.0 * M_PI / Ly;
                k3 = k3_red * 2.0 * M_PI / Lz;

                flag = 1;
                for (nk = 0; nk < k; nk++) {
                    if (   (fabs(k1 + pSPARC->k1[nk]) < TEMP_TOL || fabs(k1 + pSPARC->k1[nk] - sumx) < TEMP_TOL) 
                        && (fabs(k2 + pSPARC->k2[nk]) < TEMP_TOL || fabs(k2 + pSPARC->k2[nk] - sumy) < TEMP_TOL)
                        && (fabs(k3 + pSPARC->k3[nk]) < TEMP_TOL || fabs(k3 + pSPARC->k3[nk] - sumz) < TEMP_TOL) ) {
                        flag = 0;
                        break;
                    }
                }

                if (flag) {
                    pSPARC->k1[k] = k1;
                    pSPARC->k2[k] = k2;
                    pSPARC->k3[k] = k3;
                    pSPARC->kptWts[k]= 1.0;
                    k++;
                } else {
                    pSPARC->kptWts[nk] = 2.0;
                }

                if (pSPARC->usefock == 1) {
                    if (pSPARC->EXXDownsampling[0] == 0) {
                        flag_cx = (fabs(k1) < TEMP_TOL);
                        if (flag_cx) flag_0x = 1;
                    } else {
                        flag_cx = !((nk1 - nk1_s + 1) % pSPARC->EXXDownsampling[0]);
                    }

                    if (pSPARC->EXXDownsampling[1] == 0) {
                        flag_cy = (fabs(k2) < TEMP_TOL);
                        if (flag_cy) flag_0y = 1;
                    } else {
                        flag_cy = !((nk2 - nk2_s + 1) % pSPARC->EXXDownsampling[1]);
                    }

                    if (pSPARC->EXXDownsampling[2] == 0) {
                        flag_cz = (fabs(k3) < TEMP_TOL);
                        if (flag_cz) flag_0z = 1;
                    } else {
                        flag_cz = !((nk3 - nk3_s + 1) % pSPARC->EXXDownsampling[2]);
                    }

                    if (flag_cx && flag_cy && flag_cz) {
                        pSPARC->k1_hf[k_hf] = k1;
                        pSPARC->k2_hf[k_hf] = k2;
                        pSPARC->k3_hf[k_hf] = k3;
                        k_hf ++;
                    }
                }
            }
        }
    }

    pSPARC->Nkpts_sym = k; // update number of k points after symmetry reduction

    if (pSPARC->usefock == 1) {
        if (!pSPARC->EXXDownsampling[0] && !flag_0x) {
            if (rank == 0)
                printf(RED "ERROR: Gamma point is not one of the k-vectors. Please use positive EXX_DOWNSAMPLING or change k-point grid in the first direction.\n" RESET);
            exit(EXIT_FAILURE);
        }
        if (!pSPARC->EXXDownsampling[1] && !flag_0y) {
            if (rank == 0)
                printf(RED "ERROR: Gamma point is not one of the k-vectors. Please use positive EXX_DOWNSAMPLING or change k-point grid in the second direction.\n" RESET);
            exit(EXIT_FAILURE);
        }
        if (!pSPARC->EXXDownsampling[2] && !flag_0z) {
            if (rank == 0)
                printf(RED "ERROR: Gamma point is not one of the k-vectors. Please use positive EXX_DOWNSAMPLING or change k-point grid in the third direction.\n" RESET);
            exit(EXIT_FAILURE);
        }

        for (k_hf = 0; k_hf < pSPARC->Nkpts_hf; k_hf ++) {
            for (k = 0; k < pSPARC->Nkpts_sym; k ++) {
                if (   (fabs(pSPARC->k1[k] - pSPARC->k1_hf[k_hf]) < TEMP_TOL) 
                    && (fabs(pSPARC->k2[k] - pSPARC->k2_hf[k_hf]) < TEMP_TOL) 
                    && (fabs(pSPARC->k3[k] - pSPARC->k3_hf[k_hf]) < TEMP_TOL) ) {
                    pSPARC->kpthf_ind[k_hf] = k;        // index w.r.t. Nkpts_sym
                    pSPARC->kpthf_pn[k_hf] = 1;         // 1 -> k, 0 -> -k
                    break;
                }
            }
            if (pSPARC->kpthf_pn[k_hf] == 1) continue;
            for (k = 0; k < pSPARC->Nkpts_sym; k ++) {
                if (   ((fabs(pSPARC->k1[k] + pSPARC->k1_hf[k_hf]) < TEMP_TOL) || (fabs(pSPARC->k1[k] + pSPARC->k1_hf[k_hf] - sumx) < TEMP_TOL) )
                    && ((fabs(pSPARC->k2[k] + pSPARC->k2_hf[k_hf]) < TEMP_TOL) || (fabs(pSPARC->k2[k] + pSPARC->k2_hf[k_hf] - sumy) < TEMP_TOL) )
                    && ((fabs(pSPARC->k3[k] + pSPARC->k3_hf[k_hf]) < TEMP_TOL) || (fabs(pSPARC->k3[k] + pSPARC->k3_hf[k_hf] - sumz) < TEMP_TOL) )) {
                    pSPARC->kpthf_ind[k_hf] = k;        // index w.r.t. Nkpts_sym
                    pSPARC->kpthf_pn[k_hf] = 0;         // 1 -> k, 0 -> -k
                }
            }
        }

        // find list of k-points after reduce for HF
        pSPARC->Nkpts_hf_red = 1;
        pSPARC->kpts_hf_red_list = (int *) calloc(sizeof(int), pSPARC->Nkpts_sym);
        pSPARC->kpts_hf_red_list[0] = pSPARC->kpthf_ind[0];
        for (k_hf = 1; k_hf < pSPARC->Nkpts_hf; k_hf ++) {
            flag = 0;
            for (k = 0; k < k_hf; k ++) {
                if (pSPARC->kpthf_ind[k] == pSPARC->kpthf_ind[k_hf]) {
                    flag = 1;
                    break;
                }
            }
            if (flag) continue;
            pSPARC->kpts_hf_red_list[pSPARC->Nkpts_hf_red++] = pSPARC->kpthf_ind[k_hf];
        }

        pSPARC->kpthfred2kpthf = (int (*)[3]) calloc(sizeof(int[3]), pSPARC->Nkpts_hf_red);
        for (k = 0; k < pSPARC->Nkpts_hf_red; k++) pSPARC->kpthfred2kpthf[k][0] = 0;
        // update the reduced index to be w.r.t. Nkpts_hf_red
        // find inverse mapping from Nkpts_hf_red to Nkpts_hf
        for (k_hf = 0; k_hf < pSPARC->Nkpts_hf; k_hf ++) {
            for (k = 0; k < pSPARC->Nkpts_hf_red; k++) {
                if (pSPARC->kpthf_ind[k_hf] == pSPARC->kpts_hf_red_list[k]) {
                    pSPARC->kpthf_ind_red[k_hf] = k;        // index w.r.t. Nkpts_hf_red
                    pSPARC->kpthfred2kpthf[k][0] ++;
                    int indx = pSPARC->kpthfred2kpthf[k][0];
                    pSPARC->kpthfred2kpthf[k][indx] = k_hf;
                    break;
                }
            }
        }
    }

#ifdef DEBUG
    if (!rank) printf("After symmetry reduction, Nkpts_sym = %d\n", pSPARC->Nkpts_sym);
    for (nk = 0; nk < pSPARC->Nkpts_sym; nk++) {
        double tpiblx = 2 * M_PI / Lx;
        double tpibly = 2 * M_PI / Ly;
        double tpiblz = 2 * M_PI / Lz;
        if (!rank) printf("k1[%2d]: %8.4f, k2[%2d]: %8.4f, k3[%2d]: %8.4f, kptwt[%2d]: %.3f \n",
            nk,pSPARC->k1[nk]/tpiblx,nk,pSPARC->k2[nk]/tpibly,nk,pSPARC->k3[nk]/tpiblz,nk,pSPARC->kptWts[nk]);
    }

    if (pSPARC->usefock == 1) {
        if (!rank) printf("K-points for Hartree-Fock operator after downsampling, Nkpts_hf %d\n", pSPARC->Nkpts_hf);
        for (nk = 0; nk < pSPARC->Nkpts_hf; nk++) {
            double tpiblx = 2 * M_PI / Lx;
            double tpibly = 2 * M_PI / Ly;
            double tpiblz = 2 * M_PI / Lz;
            if (!rank) printf("k1_hf[%2d]: %8.4f, k2_hf[%2d]: %8.4f, k3_hf[%2d]: %8.4f, kptwt[%2d]: %.3f, kpthf_ind[%2d]: %2d, kpthf_ind_red[%2d]: %2d, kpthf_pn[%2d]: %2d \n",
                nk,pSPARC->k1_hf[nk]/tpiblx,nk,pSPARC->k2_hf[nk]/tpibly,nk,pSPARC->k3_hf[nk]/tpiblz,
                nk,pSPARC->kptWts_hf,nk,pSPARC->kpthf_ind[nk],nk,pSPARC->kpthf_ind_red[nk],nk,pSPARC->kpthf_pn[nk]);
        }
        if (!rank) printf("K-points for Hartree-Fock operator after downsampling mapping into kpts_system, Nkpts_hf_red %d\n", pSPARC->Nkpts_hf_red);
        for (nk = 0; nk < pSPARC->Nkpts_hf_red; nk++) {
            if (!rank) printf("kpts_hf_red_list[%d]: %d\n", nk, pSPARC->kpts_hf_red_list[nk]);
        }
    }
#endif
}



void Calculate_local_kpoints(SPARC_OBJ *pSPARC) {
    // Initialize the weights of k points
    // by going over all the k points from the Monkhorst Pack grid
    int k, kstart, kend;
    kstart = pSPARC->kpt_start_indx;
    kend = pSPARC->kpt_end_indx;
    int nk;
    k = 0;
    for(nk = kstart; nk <= kend; nk++){
        pSPARC->k1_loc[k] = pSPARC->k1[nk];
        pSPARC->k2_loc[k] = pSPARC->k2[nk];
        pSPARC->k3_loc[k] = pSPARC->k3[nk];
        pSPARC->kptWts_loc[k] = pSPARC->kptWts[nk];
        k++;
    }
}


/**
 * @brief   Call Spline to calculate derivatives of the tabulated functions and
 *          store them for later use (during interpolation).
 */
void Calculate_SplineDerivRadFun(SPARC_OBJ *pSPARC) {
    int ityp, l, lcount, lcount2, np, ppl_sum, psd_len;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int lloc = pSPARC->localPsd[ityp];
        psd_len = pSPARC->psd[ityp].size;
        pSPARC->psd[ityp].SplinerVlocD = (double *)malloc(sizeof(double)*psd_len);
        pSPARC->psd[ityp].SplineFitIsoAtomDen = (double *)malloc(sizeof(double)*psd_len);
        pSPARC->psd[ityp].SplineRhocD = (double *)malloc(sizeof(double)*psd_len);
        assert(pSPARC->psd[ityp].SplinerVlocD != NULL);
        assert(pSPARC->psd[ityp].SplineFitIsoAtomDen != NULL);
        assert(pSPARC->psd[ityp].SplineRhocD != NULL);
        getYD_gen(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc,pSPARC->psd[ityp].SplinerVlocD,psd_len);
        getYD_gen(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rhoIsoAtom,pSPARC->psd[ityp].SplineFitIsoAtomDen,psd_len);
        getYD_gen(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rho_c_table,pSPARC->psd[ityp].SplineRhocD,psd_len);
        // note we neglect lloc
        ppl_sum = 0;
        for (l = 0; l <= pSPARC->psd[ityp].lmax; l++) {
            //if (l == pSPARC->localPsd[ityp]) continue; // this fails under -O3, -O2 optimization
            if (l == lloc) continue;
            ppl_sum += pSPARC->psd[ityp].ppl[l];
        }
        pSPARC->psd[ityp].SplineFitUdV = (double *)malloc(sizeof(double)*psd_len * ppl_sum);
        if(pSPARC->psd[ityp].SplineFitUdV == NULL) {
            printf("Memory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
        for (l = lcount = lcount2 = 0; l <= pSPARC->psd[ityp].lmax; l++) {
            if (l == lloc) {
                lcount2 += pSPARC->psd[ityp].ppl[l];
                continue;
            }
            for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                // note that UdV is of size (psd_len, lmax+1), while SplineFitUdV has size (psd_len, lmax)
                getYD_gen(pSPARC->psd[ityp].RadialGrid, pSPARC->psd[ityp].UdV+lcount2*psd_len, pSPARC->psd[ityp].SplineFitUdV+lcount*psd_len, psd_len);
                lcount++; lcount2++;
            }
        }
        if (pSPARC->psd[ityp].pspsoc) {
            ppl_sum = 0;
            for (l = 1; l <= pSPARC->psd[ityp].lmax; l++) {
                //if (l == pSPARC->localPsd[ityp]) continue; // this fails under -O3, -O2 optimization
                if (l == lloc) continue;
                ppl_sum += pSPARC->psd[ityp].ppl_soc[l-1];
            }
            pSPARC->psd[ityp].SplineFitUdV_soc = (double *)malloc(sizeof(double)*psd_len * ppl_sum);
            assert(pSPARC->psd[ityp].SplineFitUdV_soc != NULL);
            lcount = lcount2 = 0;
            for (l = 1; l <= pSPARC->psd[ityp].lmax; l++) {
                if (l == lloc) {
                    lcount2 += pSPARC->psd[ityp].ppl_soc[l-1];
                    continue;
                }
                for (np = 0; np < pSPARC->psd[ityp].ppl_soc[l-1]; np++) {
                    // note that UdV is of size (psd_len, lmax+1), while SplineFitUdV has size (psd_len, lmax)
                    getYD_gen(pSPARC->psd[ityp].RadialGrid, pSPARC->psd[ityp].UdV_soc+lcount2*psd_len, pSPARC->psd[ityp].SplineFitUdV_soc+lcount*psd_len, psd_len);
                    lcount++; lcount2++;
                }
            }
        }
    }
}


/**
 * @ brief: function to calculate cell type, the det(Jacobian) for integration 
 *          and transformation matrices for distance, gradient, and laplacian 
 *          in a non cartesian coordinate system.
 **/
void Cart2nonCart_transformMat(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int i, j, k;

    // Construct LatUVec;
    double mag;
    for(i = 0; i < 3; i++){
        mag = sqrt(pow(pSPARC->LatVec[3 * i], 2.0) 
                 + pow(pSPARC->LatVec[3 * i + 1], 2.0) 
                 + pow(pSPARC->LatVec[3 * i + 2], 2.0));
        pSPARC->LatUVec[3 * i] = pSPARC->LatVec[3 * i]/mag;
        pSPARC->LatUVec[3 * i + 1] = pSPARC->LatVec[3 * i + 1]/mag;
        pSPARC->LatUVec[3 * i + 2] = pSPARC->LatVec[3 * i + 2]/mag;
    }

    // determinant of 3x3 Jacobian
    pSPARC->Jacbdet = 0.0;
    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
            for(k = 0; k < 3; k++){
                if(i != j && j != k && k != i)
                    pSPARC->Jacbdet += ((i - j) * (j - k) * (k - i)/2) * pSPARC->LatUVec[3 * i] * pSPARC->LatUVec[3 * j + 1] * pSPARC->LatUVec[3 * k + 2];
            }
        }
    }

    if(pSPARC->Jacbdet <= 0){
        if(rank == 0)
            printf("ERROR: Volume(det(jacobian)) %lf is <= 0\n", pSPARC->Jacbdet);
        exit(EXIT_FAILURE);
    }

    // transformation matrix for distance
    for(i = 0; i < 9; i++)
        pSPARC->metricT[i] = 0.0;

    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
            for(k = 0; k < 3; k++){
                pSPARC->metricT[3*i + j] += pSPARC->LatUVec[3*i + k] * pSPARC->LatUVec[3*j + k];
            }
        }
    }

    pSPARC->metricT[1] = 2 * pSPARC->metricT[1];
    pSPARC->metricT[2] = 2 * pSPARC->metricT[2];
    pSPARC->metricT[5] = 2 * pSPARC->metricT[5];

    // transformation matrix for gradient
    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
           pSPARC->gradT[3*j + i] = (pSPARC->LatUVec[3 * ((j+1) % 3) + (i+1) % 3] * pSPARC->LatUVec[3 * ((j+2) % 3) + (i+2) % 3] - pSPARC->LatUVec[3 * ((j+1) % 3) + (i+2) % 3] * pSPARC->LatUVec[3 * ((j+2) % 3) + (i+1) % 3])/pSPARC->Jacbdet;
        }
    }

    // transformation matrix for laplacian
    for(i = 0; i < 9; i++)
        pSPARC->lapcT[i] = 0.0;

    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
            for(k = 0; k < 3; k++){
                pSPARC->lapcT[3*i + j] += pSPARC->gradT[3*i + k] * pSPARC->gradT[3*j + k];
            }
        }
    }

    /* Different cell types for laplacian */
    if(fabs(pSPARC->lapcT[1]) > TEMP_TOL && fabs(pSPARC->lapcT[2]) < TEMP_TOL && fabs(pSPARC->lapcT[5]) < TEMP_TOL)
        pSPARC->cell_typ = 11;
    else if(fabs(pSPARC->lapcT[1]) < TEMP_TOL && fabs(pSPARC->lapcT[2]) > TEMP_TOL && fabs(pSPARC->lapcT[5]) < TEMP_TOL)
        pSPARC->cell_typ = 12;
    else if(fabs(pSPARC->lapcT[1]) < TEMP_TOL && fabs(pSPARC->lapcT[2]) < TEMP_TOL && fabs(pSPARC->lapcT[5]) > TEMP_TOL)
        pSPARC->cell_typ = 13;
    else if(fabs(pSPARC->lapcT[1]) > TEMP_TOL && fabs(pSPARC->lapcT[2]) > TEMP_TOL && fabs(pSPARC->lapcT[5]) < TEMP_TOL)
        pSPARC->cell_typ = 14;
    else if(fabs(pSPARC->lapcT[1]) < TEMP_TOL && fabs(pSPARC->lapcT[2]) > TEMP_TOL && fabs(pSPARC->lapcT[5]) > TEMP_TOL)
        pSPARC->cell_typ = 15;
    else if(fabs(pSPARC->lapcT[1]) > TEMP_TOL && fabs(pSPARC->lapcT[2]) < TEMP_TOL && fabs(pSPARC->lapcT[5]) > TEMP_TOL)
        pSPARC->cell_typ = 16;
    else if(fabs(pSPARC->lapcT[1]) > TEMP_TOL && fabs(pSPARC->lapcT[2]) > TEMP_TOL && fabs(pSPARC->lapcT[5]) > TEMP_TOL)
        pSPARC->cell_typ = 17;
#ifdef DEBUG
    if(!rank)
        printf("\n\nCELL_TYP: %d\n\n",pSPARC->cell_typ);
#endif
    /* transform the coefficiens of lapacian*/
    // int p, FDn = pSPARC->order/2;
    // double dx_inv, dy_inv, dz_inv, dx2_inv, dy2_inv, dz2_inv;
    // dx_inv = 1.0 / (pSPARC->delta_x);
    // dy_inv = 1.0 / (pSPARC->delta_y);
    // dz_inv = 1.0 / (pSPARC->delta_z);
    // dx2_inv = 1.0 / (pSPARC->delta_x * pSPARC->delta_x);
    // dy2_inv = 1.0 / (pSPARC->delta_y * pSPARC->delta_y);
    // dz2_inv = 1.0 / (pSPARC->delta_z * pSPARC->delta_z);
    // for (p = 0; p < FDn + 1; p++) {
    //     pSPARC->D2_stencil_coeffs_x[p] = pSPARC->lapcT[0] * pSPARC->FDweights_D2[p] * dx2_inv;
    //     pSPARC->D2_stencil_coeffs_y[p] = pSPARC->lapcT[4] * pSPARC->FDweights_D2[p] * dy2_inv;
    //     pSPARC->D2_stencil_coeffs_z[p] = pSPARC->lapcT[8] * pSPARC->FDweights_D2[p] * dz2_inv;
    //     pSPARC->D2_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_12 d/dx(df/dy)
    //     pSPARC->D2_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_13 d/dx(df/dz)
    //     pSPARC->D2_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // 2*T_23 d/dy(df/dz)
    //     pSPARC->D1_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dy_inv; // d/dx(2*T_12 df/dy) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
    //     pSPARC->D1_stencil_coeffs_yx[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // d/dy(2*T_12 df/dx) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
    //     pSPARC->D1_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dz_inv; // d/dx(2*T_13 df/dz) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
    //     pSPARC->D1_stencil_coeffs_zx[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // d/dz(2*T_13 df/dx) used in d/dz(2*T_13 df/dz + 2*T_23 df/dy)
    //     pSPARC->D1_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dz_inv; // d/dy(2*T_23 df/dz) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
    //     pSPARC->D1_stencil_coeffs_zy[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // d/dz(2*T_23 df/dy) used in d/dz(2*T_12 df/dx + 2*T_23 df/dy)
    // }
    // TODO: Find maximum eigenvalue of Hamiltionian (= max. eigvalue of -0.5 lap) for non orthogonal periodic systems
}


/**
 * @ brief: function to convert non cartesian to cartesian coordinates
 */
void nonCart2Cart_coord(const SPARC_OBJ *pSPARC, double *x, double *y, double *z) {
    double x1, x2, x3;
    x1 = pSPARC->LatUVec[0] * (*x) + pSPARC->LatUVec[3] * (*y) + pSPARC->LatUVec[6] * (*z);
    x2 = pSPARC->LatUVec[1] * (*x) + pSPARC->LatUVec[4] * (*y) + pSPARC->LatUVec[7] * (*z);
    x3 = pSPARC->LatUVec[2] * (*x) + pSPARC->LatUVec[5] * (*y) + pSPARC->LatUVec[8] * (*z);
    *x = x1; *y = x2; *z = x3;
}

/**
 * @brief: function to convert cartesian to non cartesian coordinates
 */
void Cart2nonCart_coord(const SPARC_OBJ *pSPARC, double *x, double *y, double *z) {
    double x1, x2, x3;
    x1 = pSPARC->gradT[0] * (*x) + pSPARC->gradT[1] * (*y) + pSPARC->gradT[2] * (*z);
    x2 = pSPARC->gradT[3] * (*x) + pSPARC->gradT[4] * (*y) + pSPARC->gradT[5] * (*z);
    x3 = pSPARC->gradT[6] * (*x) + pSPARC->gradT[7] * (*y) + pSPARC->gradT[8] * (*z);
    *x = x1; *y = x2; *z = x3;
}


/**
 * @brief: function to convert gradients along lattice directions to cartesian gradients
 */
void nonCart2Cart_grad(SPARC_OBJ *pSPARC, double *x, double *y, double *z) {
    double x1, x2, x3;
    x1 = pSPARC->gradT[0] * (*x) + pSPARC->gradT[3] * (*y) + pSPARC->gradT[6] * (*z);
    x2 = pSPARC->gradT[1] * (*x) + pSPARC->gradT[4] * (*y) + pSPARC->gradT[7] * (*z);
    x3 = pSPARC->gradT[2] * (*x) + pSPARC->gradT[5] * (*y) + pSPARC->gradT[8] * (*z);
    *x = x1; *y = x2; *z = x3;
}


/**
 * @brief: function to calculate the distance btween two points
 */
void CalculateDistance(SPARC_OBJ *pSPARC, double x, double y, double z, double xref, double yref, double zref, double *d) {
    if(pSPARC->cell_typ == 0) {
        *d = sqrt(pow((x-xref),2.0) + pow((y-yref),2.0) + pow((z-zref),2.0));
    } else if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
        double xx = x - xref; double yy = y - yref; double zz = z - zref;
        *d = sqrt(pSPARC->metricT[0] * (xx*xx) + pSPARC->metricT[1] * (xx*yy) + pSPARC->metricT[2] * (xx*zz)
                + pSPARC->metricT[4] * (yy*yy) + pSPARC->metricT[5] * (yy*zz) + pSPARC->metricT[8] * (zz*zz) );
    }
}


/**
 * @brief   Write the initialized parameters into the output file.
 */
void write_output_init(SPARC_OBJ *pSPARC) {
    int i, j, nproc, count;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // time_t current_time = time(NULL);
    // char *c_time_str = ctime(&current_time);
    time_t current_time;
    time(&current_time);
    char *c_time_str = ctime(&current_time);
    // ctime includes a newline char '\n', remove manually
    if (c_time_str[strlen(c_time_str)-1] == '\n') 
        c_time_str[strlen(c_time_str)-1] = '\0';

    FILE *output_fp = fopen(pSPARC->OutFilename,"w");
    if (output_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
        exit(EXIT_FAILURE);
    }

    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"*                       SPARC (version Nov 07, 2022)                      *\n");
    fprintf(output_fp,"*   Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech   *\n");
    fprintf(output_fp,"*           Distributed under GNU General Public License 3 (GPL)          *\n");
    fprintf(output_fp,"*                   Start time: %s                  *\n",c_time_str);
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                           Input parameters                                \n");
    fprintf(output_fp,"***************************************************************************\n");
    if (pSPARC->Flag_latvec_scale == 0) {
        fprintf(output_fp,"CELL: %.15g %.15g %.15g \n",pSPARC->range_x,pSPARC->range_y,pSPARC->range_z);
    } else {
        fprintf(output_fp,"LATVEC_SCALE: %.15g %.15g %.15g \n",pSPARC->latvec_scale_x,pSPARC->latvec_scale_y,pSPARC->latvec_scale_z);
    }
    fprintf(output_fp,"LATVEC:\n");
    fprintf(output_fp,"%.15g %.15g %.15g \n",pSPARC->LatVec[0],pSPARC->LatVec[1],pSPARC->LatVec[2]);
    fprintf(output_fp,"%.15g %.15g %.15g \n",pSPARC->LatVec[3],pSPARC->LatVec[4],pSPARC->LatVec[5]);
    fprintf(output_fp,"%.15g %.15g %.15g \n",pSPARC->LatVec[6],pSPARC->LatVec[7],pSPARC->LatVec[8]);
    fprintf(output_fp,"FD_GRID: %d %d %d\n",pSPARC->numIntervals_x,pSPARC->numIntervals_y,pSPARC->numIntervals_z);
    fprintf(output_fp,"FD_ORDER: %d\n",pSPARC->order);
    fprintf(output_fp,"BC:");
    fprintf(output_fp," %s", pSPARC->BCx == 0 ? "P" : "D");
    fprintf(output_fp," %s", pSPARC->BCy == 0 ? "P" : "D");
    fprintf(output_fp," %s", pSPARC->BCz == 0 ? "P" : "D");
    fprintf(output_fp,"\n");
    if (pSPARC->BC>1 && !pSPARC->SQFlag) {
        fprintf(output_fp,"KPOINT_GRID: %d %d %d\n",pSPARC->Kx,pSPARC->Ky,pSPARC->Kz);
        fprintf(output_fp,"KPOINT_SHIFT: %.15g %.15g %.15g\n",pSPARC->kptshift[0],pSPARC->kptshift[1],pSPARC->kptshift[2]);
    }
    fprintf(output_fp,"SPIN_TYP: %d\n",pSPARC->spin_typ);
    if (pSPARC->elec_T_type == 0)
        fprintf(output_fp,"ELEC_TEMP_TYPE: Fermi-Dirac\n");
    else if (pSPARC->elec_T_type == 1)
        fprintf(output_fp,"ELEC_TEMP_TYPE: Gaussian\n");

    if (pSPARC->MDFlag == 1) {
        fprintf(output_fp,"ELEC_TEMP: %.15g\n",pSPARC->elec_T);
    } else {
        fprintf(output_fp,"SMEARING: %.10g\n",1./pSPARC->Beta);
    }
    fprintf(output_fp,"EXCHANGE_CORRELATION: %s\n",pSPARC->XC);
    if (strcmpi(pSPARC->XC, "HSE") == 0) {
        fprintf(output_fp,"EXX_RANGE_FOCK: %.6f\n", pSPARC->hyb_range_fock);
        fprintf(output_fp,"EXX_RANGE_PBE: %.6f\n", pSPARC->hyb_range_pbe);
    }
    if (pSPARC->SQFlag == 1) {
        fprintf(output_fp,"SQ_FLAG: %d\n", pSPARC->SQFlag);
        if (pSPARC->SQ_typ == 2)
            fprintf(output_fp,"SQ_TYPE: GAUSS\n");
        if (pSPARC->SQ_typ_dm == 1) {
            fprintf(output_fp,"SQ_TYPE_DM: CC\n");
            fprintf(output_fp,"SQ_NPL_C: %d\n", pSPARC->SQ_npl_c);
        } else if (pSPARC->SQ_typ_dm == 2) {
            fprintf(output_fp,"SQ_TYPE_DM: GAUSS\n");
            if (pSPARC->SQ_gauss_mem == 1) {
                fprintf(output_fp,"SQ_GAUSS_MEM: HIGH\n");
            } else {
                fprintf(output_fp,"SQ_GAUSS_MEM: LOW\n");
            }
        }    
        fprintf(output_fp,"SQ_NPL_G: %d\n", pSPARC->SQ_npl_g);
        fprintf(output_fp,"SQ_RCUT: %.10g\n", pSPARC->SQ_rcut);
        if (pSPARC->SQ_fac_g2c > 0)
            fprintf(output_fp,"SQ_FAC_G2C: %.10g\n", pSPARC->SQ_fac_g2c);
        fprintf(output_fp,"SQ_EIGSHIFT_FLAG: %d\n", pSPARC->SQ_EigshiftFlag);
        if (pSPARC->SQ_EigshiftFlag == 1)
            fprintf(output_fp,"SQ_EIGSHIFT: %.10g\n", pSPARC->SQ_eigshift);
        fprintf(output_fp,"SQ_TOL_OCC: %.2E\n", pSPARC->SQ_tol_occ);    
    } else {
        fprintf(output_fp,"NSTATES: %d\n",pSPARC->Nstates);
        // this should depend on temperature and preconditoner used
        if (pSPARC->Nstates < (int)(1.2*(pSPARC->Nelectron/2)+5)*pSPARC->Nspinor ) { // with kerker a factor of 1.1 might be needed
            printf("#WARNING: Number of bands may be insufficient for efficient SCF convergence.\n");
        }
        fprintf(output_fp,"CHEB_DEGREE: %d\n",pSPARC->ChebDegree);
        if (pSPARC->CheFSI_Optmz == 1) {
            fprintf(output_fp,"CHEFSI_OPTMZ: %d\n",pSPARC->CheFSI_Optmz);
        }
        fprintf(output_fp,"CHEFSI_BOUND_FLAG: %d\n",pSPARC->chefsibound_flag);
    }
    
    if (pSPARC->RelaxFlag >= 1) {
        fprintf(output_fp,"RELAX_FLAG: %d\n",pSPARC->RelaxFlag);
    }

    if (pSPARC->RelaxFlag == 1 || pSPARC->RelaxFlag == 3) {
        fprintf(output_fp,"RELAX_METHOD: %s\n",pSPARC->RelaxMeth);
        fprintf(output_fp,"RELAX_NITER: %d\n",pSPARC->Relax_Niter);
        if(strcmpi(pSPARC->RelaxMeth,"LBFGS") == 0){
            fprintf(output_fp,"L_HISTORY: %d\n",pSPARC->L_history);
            fprintf(output_fp,"L_FINIT_STP: %.15g\n",pSPARC->L_finit_stp);
            fprintf(output_fp,"L_MAXMOV: %.15g\n",pSPARC->L_maxmov);
            fprintf(output_fp,"L_AUTOSCALE: %d\n",pSPARC->L_autoscale);
            fprintf(output_fp,"L_LINEOPT: %d\n",pSPARC->L_lineopt);
            fprintf(output_fp,"L_ICURV: %.15g\n",pSPARC->L_icurv);
        } else if(strcmpi(pSPARC->RelaxMeth,"NLCG") == 0){
            fprintf(output_fp,"NLCG_sigma: %.15g\n",pSPARC->NLCG_sigma);
        } else if(strcmpi(pSPARC->RelaxMeth,"FIRE") == 0){
            fprintf(output_fp,"FIRE_dt: %.15g\n",pSPARC->FIRE_dt);
            fprintf(output_fp,"FIRE_mass: %.15g\n",pSPARC->FIRE_mass);
            fprintf(output_fp,"FIRE_maxmov: %.15g\n",pSPARC->FIRE_maxmov);
        }
    }

    fprintf(output_fp,"CALC_STRESS: %d\n",pSPARC->Calc_stress);
    if(pSPARC->Calc_stress == 0)
        fprintf(output_fp,"CALC_PRES: %d\n",pSPARC->Calc_pres);
    if (pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1)
        fprintf(output_fp,"TWTIME: %G\n",pSPARC->TWtime);
    if (pSPARC->MDFlag == 1) {
        fprintf(output_fp,"MD_FLAG: %d\n",pSPARC->MDFlag);
        fprintf(output_fp,"MD_METHOD: %s\n",pSPARC->MDMeth);
        fprintf(output_fp,"MD_TIMESTEP: %.15g\n",pSPARC->MD_dt);
        fprintf(output_fp,"MD_NSTEP: %d\n",pSPARC->MD_Nstep);
        // fprintf(output_fp,"ION_ELEC_EQT: %d\n",pSPARC->ion_elec_eqT);
        fprintf(output_fp,"ION_VEL_DSTR: %d\n",pSPARC->ion_vel_dstr);
        fprintf(output_fp,"ION_VEL_DSTR_RAND: %d\n",pSPARC->ion_vel_dstr_rand);
        fprintf(output_fp,"ION_TEMP: %.15g\n",pSPARC->ion_T);
        if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0) {
            fprintf(output_fp,"ION_TEMP_END: %.15g\n",pSPARC->thermos_Tf);
            fprintf(output_fp,"QMASS: %.15g\n",pSPARC->qmass);
        }
        if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0) {
            //fprintf(output_fp,"AMOUNT_THERMO_VARIABLE: %d\n",pSPARC->NPT_NHnnos);
            fprintf(output_fp,"NPT_SCALE_VECS:");
            if (pSPARC->NPTscaleVecs[0] == 1) fprintf(output_fp," 1");
            if (pSPARC->NPTscaleVecs[1] == 1) fprintf(output_fp," 2");
            if (pSPARC->NPTscaleVecs[2] == 1) fprintf(output_fp," 3");
            fprintf(output_fp,"\n");
            fprintf(output_fp,"NPT_NH_QMASS:");
            fprintf(output_fp," %d",pSPARC->NPT_NHnnos);
            for (i = 0; i < pSPARC->NPT_NHnnos; i++){
                if (i%5 == 0){
                    fprintf(output_fp,"\n");
                }
                fprintf(output_fp," %.15g",pSPARC->NPT_NHqmass[i]);
            }
            fprintf(output_fp,"\n");
            fprintf(output_fp,"NPT_NH_BMASS: %.15g\n",pSPARC->NPT_NHbmass);
            fprintf(output_fp,"TARGET_PRESSURE: %.15g GPa\n",pSPARC->prtarget);
        }
        if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) {
            //fprintf(output_fp,"AMOUNT_THERMO_VARIABLE: %d\n",pSPARC->NPT_NHnnos);
            fprintf(output_fp,"NPT_NP_QMASS: %.15g\n",pSPARC->NPT_NP_qmass);
            fprintf(output_fp,"NPT_NP_BMASS: %.15g\n",pSPARC->NPT_NP_bmass);
            fprintf(output_fp,"TARGET_PRESSURE: %.15g GPa\n",pSPARC->prtarget);
        }
    }

    if (pSPARC->RestartFlag == 1) {
        fprintf(output_fp,"RESTART_FLAG: %d\n",pSPARC->RestartFlag);
    }
    if (pSPARC->NetCharge != 0) {
        fprintf(output_fp,"NET_CHARGE: %d\n",pSPARC->NetCharge);
    }
    fprintf(output_fp,"MAXIT_SCF: %d\n",pSPARC->MAXIT_SCF);
    fprintf(output_fp,"MINIT_SCF: %d\n",pSPARC->MINIT_SCF);
    fprintf(output_fp,"MAXIT_POISSON: %d\n",pSPARC->MAXIT_POISSON);
    if (pSPARC->scf_err_type == 0) {
        fprintf(output_fp,"TOL_SCF: %.2E\n",pSPARC->TOL_SCF);
    } else if (pSPARC->scf_err_type == 1) {
        fprintf(output_fp,"TOL_SCF_QE: %.2E\n",pSPARC->TOL_SCF);
        if (pSPARC->Nspin > 1) {
            fprintf(output_fp,"#WARNING: TOL_SCF_QE is not appropriatly set up for spin-polarized systems\n");
        }
        if (pSPARC->MixingVariable == 1) {
            fprintf(output_fp,"#WARNING: TOL_SCF_QE is not appropriatly set up for potential mixing\n");
        }
    }
    if (pSPARC->POISSON_SOLVER == 0){
        fprintf(output_fp,"POISSON_SOLVER: AAR\n");
    }else{
        fprintf(output_fp,"POISSON_SOLVER: CG\n");
    }
    fprintf(output_fp,"TOL_POISSON: %.2E\n",pSPARC->TOL_POISSON);
    fprintf(output_fp,"TOL_LANCZOS: %.2E\n",pSPARC->TOL_LANCZOS);
    fprintf(output_fp,"TOL_PSEUDOCHARGE: %.2E\n",pSPARC->TOL_PSEUDOCHARGE);
    if (pSPARC->MixingVariable == 0) {
        fprintf(output_fp,"MIXING_VARIABLE: density\n");
    } else if (pSPARC->MixingVariable == 1) {
        fprintf(output_fp,"MIXING_VARIABLE: potential\n");
    }

    if (pSPARC->MixingPrecond == 0) {
        fprintf(output_fp,"MIXING_PRECOND: none\n");
    } else if (pSPARC->MixingPrecond == 1) {
        fprintf(output_fp,"MIXING_PRECOND: kerker\n");
    } else if (pSPARC->MixingPrecond == 2) {
        fprintf(output_fp,"MIXING_PRECOND: resta\n");
    } else if (pSPARC->MixingPrecond == 3) {
        fprintf(output_fp,"MIXING_PRECOND: truncated_kerker\n");
    }
    
    if (pSPARC->Nspin > 1) {
        if (pSPARC->MixingPrecondMag == 0) {
            fprintf(output_fp,"MIXING_PRECOND_MAG: none\n");
        } else if (pSPARC->MixingPrecondMag == 1) {
            fprintf(output_fp,"MIXING_PRECOND_MAG: kerker\n");
        } else if (pSPARC->MixingPrecondMag == 2) {
            fprintf(output_fp,"MIXING_PRECOND_MAG: resta\n");
        } else if (pSPARC->MixingPrecondMag == 3) {
            fprintf(output_fp,"MIXING_PRECOND_MAG: truncated_kerker\n");
        }
    }

    // for large periodic systems, give warning if preconditioner is not chosen
    if (pSPARC->BC == 2) {
        double Lx, Ly, Lz, L_diag;
        Lx = pSPARC->range_x;
        Ly = pSPARC->range_y;
        Lz = pSPARC->range_z;
        L_diag = sqrt(Lx * Lx + Ly * Ly + Lz * Lz);
        if (L_diag > 20.0 && pSPARC->MixingPrecond == 0) {
            fprintf(output_fp,
                   "#WARNING: the preconditioner for SCF has been turned off, this \n"
                   "#might lead to slow SCF convergence. To specify SCF preconditioner, \n"
                   "#use 'MIXING_PRECOND' in the .inpt file\n");
        }
    }

    if (pSPARC->MixingPrecond != 0) {
        fprintf(output_fp,"TOL_PRECOND: %.2E\n",pSPARC->TOL_PRECOND);
    }

    if (pSPARC->MixingPrecond == 1) { // kerker
        fprintf(output_fp,"PRECOND_KERKER_KTF: %.10G\n",pSPARC->precond_kerker_kTF);
        fprintf(output_fp,"PRECOND_KERKER_THRESH: %.10G\n",pSPARC->precond_kerker_thresh);
    } else if (pSPARC->MixingPrecond == 2) { // resta
        fprintf(output_fp,"PRECOND_RESTA_Q0: %.3f\n",pSPARC->precond_resta_q0);
        fprintf(output_fp,"PRECOND_RESTA_RS: %.3f\n",pSPARC->precond_resta_Rs);
        fprintf(output_fp,"PRECOND_FITPOW: %d\n",pSPARC->precond_fitpow);
    } else if (pSPARC->MixingPrecond == 3) { // truncated kerker
        fprintf(output_fp,"PRECOND_KERKER_KTF: %.10G\n",pSPARC->precond_kerker_kTF);
        fprintf(output_fp,"PRECOND_KERKER_THRESH: %.10G\n",pSPARC->precond_kerker_thresh);
        fprintf(output_fp,"PRECOND_FITPOW: %d\n",pSPARC->precond_fitpow);
    }

    if (pSPARC->Nspin > 1) {
        if (pSPARC->MixingPrecondMag == 1) {
            fprintf(output_fp,"PRECOND_KERKER_KTF_MAG: %.10G\n",pSPARC->precond_kerker_kTF_mag);
            fprintf(output_fp,"PRECOND_KERKER_THRESH_MAG: %.10G\n",pSPARC->precond_kerker_thresh_mag);
        }
    }

    fprintf(output_fp,"MIXING_PARAMETER: %.10G\n",pSPARC->MixingParameter);
    if (pSPARC->PulayFrequency > 1) {
        fprintf(output_fp,"MIXING_PARAMETER_SIMPLE: %.10G\n",pSPARC->MixingParameterSimple);
    }
    if (pSPARC->Nspin > 1) {
        fprintf(output_fp,"MIXING_PARAMETER_MAG: %.10G\n",pSPARC->MixingParameterMag);
        if (pSPARC->PulayFrequency > 1) {
            fprintf(output_fp,"MIXING_PARAMETER_SIMPLE_MAG: %.10G\n",pSPARC->MixingParameterSimpleMag);
        }
    }
    fprintf(output_fp,"MIXING_HISTORY: %d\n",pSPARC->MixingHistory);
    fprintf(output_fp,"PULAY_FREQUENCY: %d\n",pSPARC->PulayFrequency);
    fprintf(output_fp,"PULAY_RESTART: %d\n",pSPARC->PulayRestartFlag);
    fprintf(output_fp,"REFERENCE_CUTOFF: %.10g\n",pSPARC->REFERENCE_CUTOFF);
    fprintf(output_fp,"RHO_TRIGGER: %d\n",pSPARC->rhoTrigger);
    fprintf(output_fp,"FIX_RAND: %d\n",pSPARC->FixRandSeed);
    fprintf(output_fp,"VERBOSITY: %d\n",pSPARC->Verbosity);
    fprintf(output_fp,"PRINT_FORCES: %d\n",pSPARC->PrintForceFlag);
    fprintf(output_fp,"PRINT_ATOMS: %d\n",pSPARC->PrintAtomPosFlag);
    fprintf(output_fp,"PRINT_EIGEN: %d\n",pSPARC->PrintEigenFlag);
    fprintf(output_fp,"PRINT_DENSITY: %d\n",pSPARC->PrintElecDensFlag);
    if(pSPARC->MDFlag == 1)
        fprintf(output_fp,"PRINT_MDOUT: %d\n",pSPARC->PrintMDout);
    if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag >= 1){
        fprintf(output_fp,"PRINT_VELS: %d\n",pSPARC->PrintAtomVelFlag);
        fprintf(output_fp,"PRINT_RESTART: %d\n",pSPARC->Printrestart);
        if(pSPARC->Printrestart == 1)
            fprintf(output_fp,"PRINT_RESTART_FQ: %d\n",pSPARC->Printrestart_fq);
    }
    if (pSPARC->PrintPsiFlag[0] == 1) {
        fprintf(output_fp,"PRINT_ORBITAL: %d %d %d %d %d %d %d\n",
            pSPARC->PrintPsiFlag[0],pSPARC->PrintPsiFlag[1],pSPARC->PrintPsiFlag[2],pSPARC->PrintPsiFlag[3],
            pSPARC->PrintPsiFlag[4],pSPARC->PrintPsiFlag[5],pSPARC->PrintPsiFlag[6]);
    }
    fprintf(output_fp,"PRINT_ENERGY_DENSITY: %d\n",pSPARC->PrintEnergyDensFlag);

    if (pSPARC->RelaxFlag == 1) {
        fprintf(output_fp,"TOL_RELAX: %.2E\n",pSPARC->TOL_RELAX);
        fprintf(output_fp,"PRINT_RELAXOUT: %d\n",pSPARC->PrintRelaxout);
    } else if (pSPARC->RelaxFlag == 2) {
        fprintf(output_fp,"TOL_RELAX_CELL: %.2E\n",pSPARC->TOL_RELAX_CELL);
        fprintf(output_fp,"RELAX_MAXDILAT: %.2E\n",pSPARC->max_dilatation);
        fprintf(output_fp,"PRINT_RELAXOUT: %d\n",pSPARC->PrintRelaxout);
    } else if(pSPARC->RelaxFlag == 3) { 
        fprintf(output_fp,"TOL_RELAX: %.2E\n",pSPARC->TOL_RELAX);   
        fprintf(output_fp,"TOL_RELAX_CELL: %.2E\n",pSPARC->TOL_RELAX_CELL); 
        fprintf(output_fp,"RELAX_MAXDILAT: %.2E\n",pSPARC->max_dilatation); 
        fprintf(output_fp,"PRINT_RELAXOUT: %d\n",pSPARC->PrintRelaxout);
    }
    if (pSPARC->usefock == 1){
        fprintf(output_fp,"EXX_FRAC: %.5g\n",pSPARC->exx_frac);
        fprintf(output_fp,"TOL_FOCK: %.2E\n",pSPARC->TOL_FOCK);
        fprintf(output_fp,"TOL_SCF_INIT: %.2E\n",pSPARC->TOL_SCF_INIT);
        fprintf(output_fp,"MAXIT_FOCK: %d\n",pSPARC->MAXIT_FOCK);
        fprintf(output_fp,"MINIT_FOCK: %d\n",pSPARC->MINIT_FOCK);
        if (pSPARC->EXXMeth_Flag == 0)
            fprintf(output_fp,"EXX_METHOD: FOURIER_SPACE\n");
        else
            fprintf(output_fp,"EXX_METHOD: REAL_SPACE\n");
        if (pSPARC->EXXDiv_Flag == 0)
            fprintf(output_fp,"EXX_DIVERGENCE: SPHERICAL\n");
        else if (pSPARC->EXXDiv_Flag == 1)
            fprintf(output_fp,"EXX_DIVERGENCE: AUXILIARY\n");
        else if (pSPARC->EXXDiv_Flag == 2)
            fprintf(output_fp,"EXX_DIVERGENCE: ERFC\n");
        fprintf(output_fp,"EXX_MEM: %d\n", pSPARC->EXXMem_batch);
        fprintf(output_fp,"ACE_FLAG: %d\n",pSPARC->ACEFlag);
        if (pSPARC->ACEFlag == 1) {
            fprintf(output_fp,"EXX_ACE_VALENCE_STATES: %d\n", pSPARC->EXXACEVal_state);
        }
        fprintf(output_fp,"EXX_DOWNSAMPLING: %d %d %d\n",pSPARC->EXXDownsampling[0]
                            ,pSPARC->EXXDownsampling[1],pSPARC->EXXDownsampling[2]);
    }
    if (pSPARC->d3Flag == 1) {
        fprintf(output_fp,"D3_FLAG: %d\n",pSPARC->d3Flag);
        fprintf(output_fp,"D3_RTHR: %.15G\n",pSPARC->d3Rthr);   
        fprintf(output_fp,"D3_CN_THR: %.15G\n",pSPARC->d3Cn_thr);
    }
    if (pSPARC->vdWDFFlag != 0) {
        fprintf(output_fp,"VDWDF_GEN_KERNEL: %d\n",pSPARC->vdWDFKernelGenFlag);
    }
    fprintf(output_fp,"OUTPUT_FILE: %s\n",pSPARC->filename_out);
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                                Cell                                       \n");
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"Lattice vectors (Bohr):\n");
    fprintf(output_fp,"%.15f %.15f %.15f \n",pSPARC->LatUVec[0]*pSPARC->range_x,pSPARC->LatUVec[1]*pSPARC->range_x,pSPARC->LatUVec[2]*pSPARC->range_x);
    fprintf(output_fp,"%.15f %.15f %.15f \n",pSPARC->LatUVec[3]*pSPARC->range_y,pSPARC->LatUVec[4]*pSPARC->range_y,pSPARC->LatUVec[5]*pSPARC->range_y);
    fprintf(output_fp,"%.15f %.15f %.15f \n",pSPARC->LatUVec[6]*pSPARC->range_z,pSPARC->LatUVec[7]*pSPARC->range_z,pSPARC->LatUVec[8]*pSPARC->range_z);
    fprintf(output_fp,"Volume: %-.10E (Bohr^3)\n", pSPARC->range_x * pSPARC->range_y * pSPARC->range_z * pSPARC->Jacbdet);
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                           Parallelization                                 \n");
    fprintf(output_fp,"***************************************************************************\n");
    if (pSPARC->SQFlag == 1) {
        fprintf(output_fp,"NP_DOMAIN_SQ_PARAL: %d %d %d\n",pSPARC->npNdx_SQ,pSPARC->npNdy_SQ,pSPARC->npNdz_SQ);
        fprintf(output_fp,"NP_DOMAIN_PHI_PARAL: %d %d %d\n",pSPARC->npNdx_phi,pSPARC->npNdy_phi,pSPARC->npNdz_phi);
    } else {
        if (pSPARC->num_node || pSPARC->num_cpu_per_node || pSPARC->num_acc_per_node) {
            fprintf(output_fp,"# Command line arguments: ");
            if (pSPARC->num_node) fprintf(output_fp,"-n %d ", pSPARC->num_node);
            if (pSPARC->num_cpu_per_node) fprintf(output_fp,"-c %d ", pSPARC->num_cpu_per_node);
            if (pSPARC->num_acc_per_node) fprintf(output_fp,"-a %d ", pSPARC->num_acc_per_node);
            fprintf(output_fp,"\n");
        }
        fprintf(output_fp,"NP_SPIN_PARAL: %d\n",pSPARC->npspin);
        fprintf(output_fp,"NP_KPOINT_PARAL: %d\n",pSPARC->npkpt);
        fprintf(output_fp,"NP_BAND_PARAL: %d\n",pSPARC->npband);
        fprintf(output_fp,"NP_DOMAIN_PARAL: %d %d %d\n",pSPARC->npNdx,pSPARC->npNdy,pSPARC->npNdz);
        fprintf(output_fp,"NP_DOMAIN_PHI_PARAL: %d %d %d\n",pSPARC->npNdx_phi,pSPARC->npNdy_phi,pSPARC->npNdz_phi);
        fprintf(output_fp,"EIG_SERIAL_MAXNS: %d\n",pSPARC->eig_serial_maxns);
        if (pSPARC->useLAPACK == 0) {
            fprintf(output_fp,"EIG_PARAL_BLKSZ: %d\n",pSPARC->eig_paral_blksz);
            fprintf(output_fp,"EIG_PARAL_ORFAC: %.1e\n",pSPARC->eig_paral_orfac);
            fprintf(output_fp,"EIG_PARAL_MAXNP: %d\n",pSPARC->eig_paral_maxnp);
        }
    }
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                             Initialization                                \n");
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"Number of processors               :  %d\n",nproc);

    if ( (fabs(pSPARC->delta_x-pSPARC->delta_y) <=1e-12) && (fabs(pSPARC->delta_x-pSPARC->delta_z) <=1e-12)
        && (fabs(pSPARC->delta_y-pSPARC->delta_z) <=1e-12) ) {
        fprintf(output_fp,"Mesh spacing                       :  %.6g (Bohr)\n",pSPARC->delta_x);
    } else {
        fprintf(output_fp,"Mesh spacing in x-direction        :  %.6g (Bohr)\n",pSPARC->delta_x);
        fprintf(output_fp,"Mesh spacing in y-direction        :  %.6g (Bohr)\n",pSPARC->delta_y);
        fprintf(output_fp,"Mesh spacing in z-direction        :  %.6g (Bohr)\n",pSPARC->delta_z);
    }

    if (pSPARC->BC==2 || pSPARC->BC==3 || pSPARC->BC==4) {
        fprintf(output_fp,"Number of symmetry adapted k-points:  %d\n",pSPARC->Nkpts_sym);
    }

    fprintf(output_fp,"Output printed to                  :  %s\n",pSPARC->OutFilename);

    //if (pSPARC->PrintAtomPosFlag==1) {
    //    fprintf(output_fp,"Atom positions printed to          :  %s\n",pSPARC->AtomFilename);
    //}

    //if (pSPARC->PrintForceFlag==1) {
    //    fprintf(output_fp,"Forces printed to                  :  %s\n",pSPARC->StaticFilename);
    //}

    if (pSPARC->PrintEigenFlag==1) {
        fprintf(output_fp,"Final eigenvalues printed to       :  %s\n",pSPARC->EigenFilename);
    }

    if (pSPARC->MDFlag == 1 && pSPARC->PrintMDout == 1) {
        fprintf(output_fp,"MD output printed to               :  %s\n",pSPARC->MDFilename);
    }

    if (pSPARC->RelaxFlag == 1 && pSPARC->PrintRelaxout == 1) {
        fprintf(output_fp,"Relax output printed to            :  %s\n",pSPARC->RelaxFilename);
    }

    fprintf(output_fp,"Total number of atom types         :  %d\n",pSPARC->Ntypes);
    fprintf(output_fp,"Total number of atoms              :  %d\n",pSPARC->n_atom);
    fprintf(output_fp,"Total number of electrons          :  %d\n",pSPARC->Nelectron);

    // count = 0;
    for (i = 0; i < pSPARC->Ntypes; i++) {
        fprintf(output_fp,"Atom type %-2d (valence electrons)   :  %s %d\n",i+1,&pSPARC->atomType[L_ATMTYPE*i], pSPARC->Znucl[i]);
        fprintf(output_fp,"Pseudopotential                    :  %s\n",pSPARC->psdName + i*L_PSD);
        // fprintf(output_fp,"lloc                               :  %d\n",pSPARC->localPsd[i]);
        if (pSPARC->MDFlag == 1) {
            fprintf(output_fp,"Atomic mass                        :  %.15g\n",pSPARC->Mass[i]);
        }
        fprintf(output_fp,"Pseudocharge radii of atom type %-2d :  %.2f %.2f %.2f (x, y, z dir)\n",i+1,pSPARC->CUTOFF_x[i],pSPARC->CUTOFF_y[i],pSPARC->CUTOFF_z[i]);
        fprintf(output_fp,"Number of atoms of type %-2d         :  %d\n",i+1,pSPARC->nAtomv[i]);
        // if (pSPARC->PrintAtomPosFlag == 1 && pSPARC->MDFlag == 0 && pSPARC->RelaxFlag == 0) {
        //     fprintf(output_fp,"Fractional coordinates of atoms of type %-2d    :\n",i+1);
        //     for (j = 0; j < pSPARC->nAtomv[i]; j++) {
        //         fprintf(output_fp,"%18.10f %18.10f %18.10f\n",pSPARC->atom_pos[3*count]/pSPARC->range_x,pSPARC->atom_pos[3*count+1]/pSPARC->range_y,pSPARC->atom_pos[3*count+2]/pSPARC->range_z);
        //         count++;
        //     }
        // }
    }

    char mem_str[32];
    formatBytes(pSPARC->memory_usage,32,mem_str);
    fprintf(output_fp,"Estimated total memory usage       :  %s\n",mem_str);
    formatBytes(pSPARC->memory_usage/nproc,32,mem_str);
    fprintf(output_fp,"Estimated memory per processor     :  %s\n",mem_str);

    fclose(output_fp);

    // write .static file
    if ((pSPARC->PrintAtomPosFlag == 1 || pSPARC->PrintForceFlag == 1) && pSPARC->MDFlag == 0 && pSPARC->RelaxFlag == 0) {
        FILE *static_fp = fopen(pSPARC->StaticFilename,"w");
        if (static_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->StaticFilename);
            exit(EXIT_FAILURE);
        }

        // print atoms
        if (pSPARC->PrintAtomPosFlag == 1) {
            fprintf(static_fp,"***************************************************************************\n");
            fprintf(static_fp,"                            Atom positions                                 \n");
            fprintf(static_fp,"***************************************************************************\n");
            count = 0;
            for (i = 0; i < pSPARC->Ntypes; i++) {
                fprintf(static_fp,"Fractional coordinates of %s:\n",&pSPARC->atomType[L_ATMTYPE*i]);
                for (j = 0; j < pSPARC->nAtomv[i]; j++) {
                    fprintf(static_fp,"%18.10f %18.10f %18.10f\n",
                            pSPARC->atom_pos[3*count]/pSPARC->range_x,
                            pSPARC->atom_pos[3*count+1]/pSPARC->range_y,
                            pSPARC->atom_pos[3*count+2]/pSPARC->range_z);
                    count++;
                }
            }
        }
        fclose(static_fp);
    }
}



/**
 * @brief Create MPI struct type SPARC_INPUT_MPI for broadcasting.
*/
void SPARC_Input_MPI_create(MPI_Datatype *pSPARC_INPUT_MPI) {
    SPARC_INPUT_OBJ sparc_input_tmp;

    MPI_Datatype SPARC_types[N_MEMBR] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, 
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, 
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, 
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
                                         MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR,
                                         MPI_CHAR};
    int blens[N_MEMBR] = {1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 3, 1, 1, 
                          1, 1, 1, 1, 1, 
                          3, 1, 1, 1, 1, 
                          1, 1, 1, 1, 1, 
                          1, 7, 1, 1, /* int */ 
                          1, 1, 1, 1, 1, 
                          1, 9, 1, 1, 3,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 
                          1, 1, 1, 1, 1,
                          1, 1, 1, L_QMASS, 1,
                          1, 1, 1, 1, 1, 
                          1, 1, 1, 1, 1,
                          1, 1, /* double */
                          32, 32, 32, L_STRING, L_STRING, /* char */
                          L_STRING};

    // calculating offsets in an architecture independent manner
    MPI_Aint addr[N_MEMBR],disps[N_MEMBR], base;
    int i = 0;
    MPI_Get_address(&sparc_input_tmp, &base);
    // int type
    MPI_Get_address(&sparc_input_tmp.num_node, addr + i++);
    MPI_Get_address(&sparc_input_tmp.num_cpu_per_node, addr + i++);
    MPI_Get_address(&sparc_input_tmp.num_acc_per_node, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npspin, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npkpt, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npband, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npNdx, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npNdy, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npNdz, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npNdx_phi, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npNdy_phi, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npNdz_phi, addr + i++);
    MPI_Get_address(&sparc_input_tmp.eig_serial_maxns, addr + i++);
    MPI_Get_address(&sparc_input_tmp.eig_paral_blksz, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MDFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.spin_typ, addr + i++);
    MPI_Get_address(&sparc_input_tmp.RelaxFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.RestartFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Flag_latvec_scale, addr + i++);
    MPI_Get_address(&sparc_input_tmp.numIntervals_x, addr + i++);
    MPI_Get_address(&sparc_input_tmp.numIntervals_y, addr + i++);
    MPI_Get_address(&sparc_input_tmp.numIntervals_z, addr + i++);
    MPI_Get_address(&sparc_input_tmp.BC, addr + i++);
    MPI_Get_address(&sparc_input_tmp.BCx, addr + i++);
    MPI_Get_address(&sparc_input_tmp.BCy, addr + i++);
    MPI_Get_address(&sparc_input_tmp.BCz, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Nstates, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Ntypes, addr + i++);
    MPI_Get_address(&sparc_input_tmp.NetCharge, addr + i++);
    MPI_Get_address(&sparc_input_tmp.order, addr + i++);
    MPI_Get_address(&sparc_input_tmp.ChebDegree, addr + i++);
    MPI_Get_address(&sparc_input_tmp.CheFSI_Optmz, addr + i++);
    MPI_Get_address(&sparc_input_tmp.chefsibound_flag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.rhoTrigger, addr + i++);
    MPI_Get_address(&sparc_input_tmp.FixRandSeed, addr + i++);
    MPI_Get_address(&sparc_input_tmp.accuracy_level, addr + i++);
    MPI_Get_address(&sparc_input_tmp.scf_err_type, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MAXIT_SCF, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MINIT_SCF, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MAXIT_POISSON, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Relax_Niter, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MixingVariable, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MixingPrecond, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MixingPrecondMag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MixingHistory, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PulayFrequency, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PulayRestartFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.precond_fitpow, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Nkpts, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Kx, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Ky, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Kz, addr + i++);
    MPI_Get_address(&sparc_input_tmp.NkptsGroup, addr + i++);
    MPI_Get_address(&sparc_input_tmp.kctr, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Verbosity, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PrintForceFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PrintAtomPosFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PrintAtomVelFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PrintEigenFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PrintElecDensFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PrintMDout, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PrintRelaxout, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Printrestart, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Printrestart_fq, addr + i++);
    MPI_Get_address(&sparc_input_tmp.elec_T_type, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MD_Nstep, addr + i++);
    MPI_Get_address(&sparc_input_tmp.ion_elec_eqT, addr + i++);
    MPI_Get_address(&sparc_input_tmp.ion_vel_dstr, addr + i++);
    MPI_Get_address(&sparc_input_tmp.ion_vel_dstr_rand, addr + i++);
    MPI_Get_address(&sparc_input_tmp.L_history, addr + i++);
    MPI_Get_address(&sparc_input_tmp.L_autoscale, addr + i++);
    MPI_Get_address(&sparc_input_tmp.L_lineopt, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Calc_stress, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Calc_pres, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Poisson_solver, addr + i++);
    MPI_Get_address(&sparc_input_tmp.d3Flag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.NPT_NHnnos, addr + i++);
    MPI_Get_address(&sparc_input_tmp.NPTscaleVecs, addr + i++);
    MPI_Get_address(&sparc_input_tmp.vdWDFKernelGenFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MAXIT_FOCK, addr + i++);
    MPI_Get_address(&sparc_input_tmp.EXXMeth_Flag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.ACEFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.EXXMem_batch, addr + i++);
    MPI_Get_address(&sparc_input_tmp.EXXACEVal_state, addr + i++);
    MPI_Get_address(&sparc_input_tmp.EXXDiv_Flag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.EXXDownsampling, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MINIT_FOCK, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQ_typ_dm, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQ_gauss_mem, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQ_npl_c, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQ_npl_g, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQ_EigshiftFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npNdx_SQ, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npNdy_SQ, addr + i++);
    MPI_Get_address(&sparc_input_tmp.npNdz_SQ, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PrintPsiFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.PrintEnergyDensFlag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.eig_paral_maxnp, addr + i++);
    // double type
    MPI_Get_address(&sparc_input_tmp.range_x, addr + i++);
    MPI_Get_address(&sparc_input_tmp.range_y, addr + i++);
    MPI_Get_address(&sparc_input_tmp.range_z, addr + i++);
    MPI_Get_address(&sparc_input_tmp.latvec_scale_x, addr + i++);
    MPI_Get_address(&sparc_input_tmp.latvec_scale_y, addr + i++);
    MPI_Get_address(&sparc_input_tmp.latvec_scale_z, addr + i++);
    MPI_Get_address(&sparc_input_tmp.LatVec, addr + i++);
    MPI_Get_address(&sparc_input_tmp.mesh_spacing, addr + i++);
    MPI_Get_address(&sparc_input_tmp.ecut, addr + i++);
    MPI_Get_address(&sparc_input_tmp.kptshift, addr + i++);
    MPI_Get_address(&sparc_input_tmp.target_energy_accuracy, addr + i++);
    MPI_Get_address(&sparc_input_tmp.target_force_accuracy, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TOL_SCF, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TOL_RELAX, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TOL_POISSON, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TOL_LANCZOS, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TOL_PSEUDOCHARGE, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TOL_PRECOND, addr + i++);
    MPI_Get_address(&sparc_input_tmp.precond_kerker_kTF, addr + i++);
    MPI_Get_address(&sparc_input_tmp.precond_kerker_thresh, addr + i++);
    MPI_Get_address(&sparc_input_tmp.precond_kerker_kTF_mag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.precond_kerker_thresh_mag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.precond_resta_q0, addr + i++);
    MPI_Get_address(&sparc_input_tmp.precond_resta_Rs, addr + i++);
    MPI_Get_address(&sparc_input_tmp.REFERENCE_CUTOFF, addr + i++);
    MPI_Get_address(&sparc_input_tmp.Beta, addr + i++);
    MPI_Get_address(&sparc_input_tmp.elec_T, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MixingParameter, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MixingParameterSimple, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MixingParameterMag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MixingParameterSimpleMag, addr + i++);
    MPI_Get_address(&sparc_input_tmp.MD_dt, addr + i++);
    MPI_Get_address(&sparc_input_tmp.ion_T, addr + i++);
    MPI_Get_address(&sparc_input_tmp.thermos_Tf, addr + i++);
    MPI_Get_address(&sparc_input_tmp.qmass, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TWtime, addr + i++);
    MPI_Get_address(&sparc_input_tmp.NLCG_sigma, addr + i++);
    MPI_Get_address(&sparc_input_tmp.L_finit_stp, addr + i++);
    MPI_Get_address(&sparc_input_tmp.L_maxmov, addr + i++);
    MPI_Get_address(&sparc_input_tmp.L_icurv, addr + i++);
    MPI_Get_address(&sparc_input_tmp.FIRE_dt, addr + i++);
    MPI_Get_address(&sparc_input_tmp.FIRE_mass, addr + i++);
    MPI_Get_address(&sparc_input_tmp.FIRE_maxmov, addr + i++);
    MPI_Get_address(&sparc_input_tmp.max_dilatation, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TOL_RELAX_CELL, addr + i++);
    MPI_Get_address(&sparc_input_tmp.eig_paral_orfac, addr + i++);
    MPI_Get_address(&sparc_input_tmp.d3Rthr, addr + i++);
    MPI_Get_address(&sparc_input_tmp.d3Cn_thr, addr + i++);
    MPI_Get_address(&sparc_input_tmp.NPT_NHqmass, addr + i++);
    MPI_Get_address(&sparc_input_tmp.NPT_NHbmass, addr + i++);
    MPI_Get_address(&sparc_input_tmp.prtarget, addr + i++);
    MPI_Get_address(&sparc_input_tmp.NPT_NP_qmass, addr + i++);
    MPI_Get_address(&sparc_input_tmp.NPT_NP_bmass, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TOL_FOCK, addr + i++);
    MPI_Get_address(&sparc_input_tmp.TOL_SCF_INIT, addr + i++);
    MPI_Get_address(&sparc_input_tmp.hyb_range_fock, addr + i++);
    MPI_Get_address(&sparc_input_tmp.hyb_range_pbe, addr + i++);
    MPI_Get_address(&sparc_input_tmp.exx_frac, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQ_rcut, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQ_fac_g2c, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQ_tol_occ, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SQ_eigshift, addr + i++);
    // char type
    MPI_Get_address(&sparc_input_tmp.MDMeth, addr + i++);
    MPI_Get_address(&sparc_input_tmp.RelaxMeth, addr + i++);
    MPI_Get_address(&sparc_input_tmp.XC, addr + i++);
    MPI_Get_address(&sparc_input_tmp.filename, addr + i++);
    MPI_Get_address(&sparc_input_tmp.filename_out, addr + i++);
    MPI_Get_address(&sparc_input_tmp.SPARCROOT, addr + i++);

    for (i = 0; i < N_MEMBR; i++) {
        disps[i] = addr[i] - base;
    }

    MPI_Type_create_struct(N_MEMBR, blens, disps, SPARC_types, pSPARC_INPUT_MPI);
    MPI_Type_commit(pSPARC_INPUT_MPI);

    //MPI_Aint extend = sizeof(sparc_input_tmp);
    //MPI_Datatype SPARC_INPUT_MPI_tmp;
    //MPI_Type_create_struct(N_MEMBR, blens, disps, SPARC_types, &SPARC_INPUT_MPI_tmp);
    //MPI_Type_create_resized(SPARC_INPUT_MPI_tmp, 0, extend, pSPARC_INPUT_MPI);
    //MPI_Type_commit(pSPARC_INPUT_MPI);
}


/**
 * @brief   Computing nearest neighbohr distance
 */
double compute_nearest_neighbor_dist(SPARC_OBJ *pSPARC, char CorN) {
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    int atm1, atm2;
    double nn, dist = 0.0;
    nn = 100000000000;
    if (CorN == 'N') {           // Non-Cartesian coordinates
        for (atm1 = 0; atm1 < pSPARC->n_atom-1; atm1++) {
            for (atm2 = atm1+1; atm2 < pSPARC->n_atom; atm2++) {
                CalculateDistance(pSPARC, pSPARC->atom_pos[3*atm1], pSPARC->atom_pos[3*atm1+1], pSPARC->atom_pos[3*atm1+2],
                                    pSPARC->atom_pos[3*atm2], pSPARC->atom_pos[3*atm2+1], pSPARC->atom_pos[3*atm2+2], &dist);
                if (dist < nn) nn = dist;
            }
        }
    } else if (CorN == 'C') {                            // Cartesian coordinates
        for (atm1 = 0; atm1 < pSPARC->n_atom-1; atm1++) {
            for (atm2 = atm1+1; atm2 < pSPARC->n_atom; atm2++) {
                dist = fabs(sqrt(pow(pSPARC->atom_pos[3*atm1] - pSPARC->atom_pos[3*atm2],2.0) 
                               + pow(pSPARC->atom_pos[3*atm1+1] - pSPARC->atom_pos[3*atm2+1],2.0) 
                               + pow(pSPARC->atom_pos[3*atm1+2] - pSPARC->atom_pos[3*atm2+2],2.0) ));
                if (dist < nn) nn = dist;
            }
        }
    } else {
        printf("ERROR: please use 'N' for non-cartesian coordinates and 'C' for cartesian coordinates in compute_nearest_neighbor_dist function.");
        exit(-1);
    }
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    printf("\nComputing nearest neighbor distance (%.3f Bohr) takes %.3f ms\n", nn, (t2-t1)*1000);
#endif
    return nn;
}


/**
 * @brief   Simple linear model for selecting maximum number of 
 *          processors for eigenvalue solver. 
 */
int parallel_eigensolver_max_processor(int N, char RorC, char SorG)
{
    if (SorG == 'S') {
        return (int)(0.026918*N+1);
    } else if (SorG == 'G') {
        if (RorC == 'R')
            return (int) (0.036215*N+1);
        else if (RorC == 'C')
            return (int) (0.038695*N+1);
    } 
    return -1;
}

