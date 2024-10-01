/***
 * @file    exactExchangeInitialization.c
 * @brief   This file contains the functions for Exact Exchange.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <complex.h>

#include "exactExchangeInitialization.h"
#include "tools.h"
#include "electrostatics.h"
#include "parallelization.h"
#include "sqInitialization.h"
#include "kroneckerLaplacian.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


#define TEMP_TOL (1e-12)


/**
 * @brief   Initialization of all variables for exact exchange.
 */
void init_exx(SPARC_OBJ *pSPARC) {
    int rank, DMnd, DMndsp, len_full, len_full_kpt, Ns_full, blacs_size, kpt_bridge_size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(pSPARC->blacscomm, &blacs_size);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);

    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    
    find_local_kpthf(pSPARC);                                                                           // determine local number kpts for HF
    Ns_full = pSPARC->Nstates * pSPARC->Nkpts_sym * pSPARC->Nspin_spincomm;                           // total length across all kpts.

    if (pSPARC->ExxAcc == 0) {
        len_full = DMndsp * pSPARC->Nstates * pSPARC->Nkpts_hf_red;                  // total length across all kpts.
        len_full_kpt = pSPARC->Nd_d_kptcomm * pSPARC->Nspinor_spincomm * pSPARC->Nstates * pSPARC->Nkpts_hf_red;
        if (pSPARC->isGammaPoint == 1) {
            // Storage of psi_outer in dmcomm
            pSPARC->psi_outer = (double *)calloc( len_full , sizeof(double) ); 
            // Storage of psi_outer in kptcomm_topo
            pSPARC->psi_outer_kptcomm_topo = (double *)calloc(len_full_kpt, sizeof(double));
            assert (pSPARC->psi_outer != NULL && pSPARC->psi_outer_kptcomm_topo != NULL);

            pSPARC->occ_outer = (double *)calloc(Ns_full, sizeof(double));
            assert(pSPARC->occ_outer != NULL);
        } else {
            // Storage of psi_outer_kpt in dmcomm
            pSPARC->psi_outer_kpt = (double _Complex *)calloc( len_full , sizeof(double _Complex) ); 
            // Storage of psi_outer in kptcomm_topo
            pSPARC->psi_outer_kptcomm_topo_kpt = (double _Complex *)calloc(len_full_kpt, sizeof(double _Complex));
            assert (pSPARC->psi_outer_kpt != NULL && pSPARC->psi_outer_kptcomm_topo_kpt != NULL);
            
            pSPARC->occ_outer = (double *)calloc(Ns_full, sizeof(double));
            assert(pSPARC->occ_outer != NULL);
        }
    } else {        
        if (pSPARC->isGammaPoint == 1) {
            pSPARC->Nstates_occ = 0;
        } else {
            pSPARC->Nstates_occ = 0;
            pSPARC->occ_outer = (double *)calloc(Ns_full, sizeof(double));
            assert(pSPARC->occ_outer != NULL);
        }
    }
    
    find_k_shift(pSPARC);
    kshift_phasefactor(pSPARC);

    if (pSPARC->ExxDivFlag == 1) {
        auxiliary_constant(pSPARC);
    }

    // compute the constant coefficients for solving Poisson's equation using FFT
    // For even conjugate space (FFT), only half of the coefficients are needed
    if (pSPARC->dmcomm != MPI_COMM_NULL || pSPARC->kptcomm_topo != MPI_COMM_NULL) {
        allocate_singularity_removal_const(pSPARC);

        if (pSPARC->ExxMethod == 0) {
            compute_pois_fft_const(pSPARC);
        } else {
            // initialization for real space solver
            pSPARC->kron_lap_exx = (KRON_LAP **) malloc(sizeof(KRON_LAP*) * pSPARC->Nkpts_shift);
            for (int i = 0; i < pSPARC->Nkpts_shift; i++) {
                double k1, k2, k3;
                if (i < pSPARC->Nkpts_shift - 1) { // The last shift is all zeros
                    k1 = pSPARC->k1_shift[i];
                    k2 = pSPARC->k2_shift[i];
                    k3 = pSPARC->k3_shift[i];
                } else {
                    k1 = k2 = k3 = 0;
                }
                pSPARC->kron_lap_exx[i] = (KRON_LAP *) malloc(sizeof(KRON_LAP));
                if (pSPARC->BC == 1) {
                    init_kron_Lap(pSPARC, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, 
                        pSPARC->BCx, pSPARC->BCy, pSPARC->BCz, k1, k2, k3, pSPARC->isGammaPoint, pSPARC->kron_lap_exx[i]);
                } else {
                    init_kron_Lap(pSPARC, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, 
                        0, 0, 0, k1, k2, k3, pSPARC->isGammaPoint, pSPARC->kron_lap_exx[i]);
                }
            }
            if (pSPARC->BC == 1) {
                // dirichlet BC needs multipole expansion
                int DMVertices[6] = {0, pSPARC->Nx-1, 0, pSPARC->Ny-1, 0, pSPARC->Nz-1};
                pSPARC->MpExp_exx = (MPEXP_OBJ *) malloc(sizeof(MPEXP_OBJ));
                init_multipole_expansion(pSPARC, pSPARC->MpExp_exx, 
                    pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, DMVertices, MPI_COMM_SELF);
            } else {
                // periodic BC needs singularity removal
                compute_pois_kron_cons(pSPARC);
            }
        }
    }
    
    // create kpttopo_dmcomm_inter
    pSPARC->flag_kpttopo_dm = 0;
    create_kpttopo_dmcomm_inter(pSPARC);

    // initialize Eexx
    pSPARC->Eexx = 0;
}


/**
 * @brief   Allocate singularity removal constants
 */
void allocate_singularity_removal_const(SPARC_OBJ *pSPARC)
{
    int Nx = pSPARC->Nx;
    int Nxh = pSPARC->Nx / 2 + 1;
    int Ny = pSPARC->Ny;
    int Nz = pSPARC->Nz;
    int len = ((pSPARC->isGammaPoint == 1) && (pSPARC->ExxMethod == 0)) ? (Nxh*Ny*Nz) : (Nx*Ny*Nz);
    len *= pSPARC->Nkpts_shift;

    // nothing required here
    if (pSPARC->ExxMethod == 1 && pSPARC->BC == 1) return;

    pSPARC->pois_const = (double *)malloc(sizeof(double) * len);
    assert(pSPARC->pois_const != NULL);
    if (pSPARC->Calc_stress == 1) {
        pSPARC->pois_const_stress = (double *)malloc(sizeof(double) * len);
        assert(pSPARC->pois_const_stress != NULL);
        if (pSPARC->ExxDivFlag == 0) {
            pSPARC->pois_const_stress2 = (double *)malloc(sizeof(double) * len);
            assert(pSPARC->pois_const_stress2 != NULL);
        }
    } else if (pSPARC->Calc_pres == 1) {
        if (pSPARC->ExxDivFlag != 0) {
            pSPARC->pois_const_press = (double *)malloc(sizeof(double) * len);
            assert(pSPARC->pois_const_press != NULL);
        }
    }
}


/**
 * @brief   singularity removal calculation
 * 
 *          Spherical Cutoff - Method by James Spencer and Ali Alavi 
 *          DOI: 10.1103/PhysRevB.77.193110
 *          Auxiliary function - Method by Gygi and Baldereschi
 *          DOI: 10.1103/PhysRevB.34.4405
 */
void singularity_remooval_const(SPARC_OBJ *pSPARC, double G2, double *sr_const, 
                    double *sr_const_stress, double *sr_const_stress2, double *sr_const_press)
{
    double L1 = pSPARC->delta_x * pSPARC->Nx;
    double L2 = pSPARC->delta_y * pSPARC->Ny;
    double L3 = pSPARC->delta_z * pSPARC->Nz;
    double V =  L1 * L2 * L3 * pSPARC->Jacbdet * pSPARC->Nkpts_hf;       // Nk_hf * unit cell volume
    double R_c = pow(3*V/(4*M_PI),(1.0/3));
    double omega = pSPARC->hyb_range_fock;
    double omega2 = omega * omega;

    // spherical truncation
    if (pSPARC->ExxDivFlag == 0) {        
        double G4 = G2*G2;
        double x = R_c*sqrt(G2);
        if (fabs(G2) > 1e-4) {
            *sr_const = 4*M_PI*(1-cos(x))/G2;
            if (pSPARC->Calc_stress == 1) {
                *sr_const_stress = 4*M_PI*( 1-cos(x)- x/2*sin(x) )/G4;
                *sr_const_stress2 = 4*M_PI*( x/2*sin(x) )/G2/3;
            }
        } else {
            *sr_const = 2*M_PI*(R_c * R_c);
            if (pSPARC->Calc_stress == 1) {
                *sr_const_stress = 4*M_PI*pow(R_c,4)/24;
                *sr_const_stress2 = 4*M_PI*( R_c*R_c/2 )/3;
            }
        }
    // auxiliary function
    } else if (pSPARC->ExxDivFlag == 1) {        
        double G4 = G2*G2;
        double x = -0.25/omega2*G2;
        if (fabs(G2) > 1e-4) {
            if (omega > 0) {
                *sr_const = 4*M_PI/G2 * (1 - exp(x));
                if (pSPARC->Calc_stress == 1) {
                    *sr_const_stress = 4*M_PI*(1 - exp(x)*(1-x))/G4 /4;
                } else if (pSPARC->Calc_pres == 1) {
                    *sr_const_press = 4*M_PI*(1 - exp(x)*(1-x))/G2 /4;
                }
            } else {
                *sr_const = 4*M_PI/G2;
                if (pSPARC->Calc_stress == 1) {                        
                    *sr_const_stress = 4*M_PI/G4 /4;
                } else if (pSPARC->Calc_pres == 1) {
                    *sr_const_press = 4*M_PI/G2 /4;
                }
            }
        } else {
            if (omega > 0) {
                *sr_const = 4*M_PI*(pSPARC->const_aux + 0.25/omega2);
                if (pSPARC->Calc_stress == 1) {
                    *sr_const_stress = 0;
                } else if (pSPARC->Calc_pres == 1) {
                    *sr_const_press = 0;
                }
            } else {
                *sr_const = 4*M_PI*pSPARC->const_aux;
                if (pSPARC->Calc_stress == 1) {
                    *sr_const_stress = 0;
                } else if (pSPARC->Calc_pres == 1) {
                    *sr_const_press = 0;
                }
            }
        }
    // ERFC short ranged screened 
    } else if (pSPARC->ExxDivFlag == 2) {        
        double G4 = G2*G2;
        double x = -0.25/omega2*G2;
        if (fabs(G2) > 1e-4) {
            *sr_const = 4*M_PI*(1-exp(x))/G2;;
            if (pSPARC->Calc_stress == 1) {
                *sr_const_stress = 4*M_PI*( 1-exp(x)*(1-x) )/G4;
            } else if (pSPARC->Calc_pres == 1) {
                *sr_const_press = 4*M_PI*( 1-exp(x)*(1-x) )/G2;
            }
        } else {
            *sr_const = M_PI/omega2;
            if (pSPARC->Calc_stress == 1) {
                *sr_const_stress = 0;
            } else if (pSPARC->Calc_pres == 1) {
                *sr_const_press = 0;
            }
        }
    }
}


/**
 * @brief   Compute eigenvalues of Laplacian applied with singularity removal methods
 */
void compute_pois_kron_cons(SPARC_OBJ *pSPARC) {    
    int Nd = pSPARC->Nd;    

    for (int l = 0; l < pSPARC->Nkpts_shift; l++) {
        KRON_LAP* kron_lap = pSPARC->kron_lap_exx[l];
        for (int i = 0; i < Nd; i++) {
            double G2 = -kron_lap->eig[i];
            singularity_remooval_const(pSPARC, G2, pSPARC->pois_const + i + l*Nd, 
                        pSPARC->pois_const_stress + i + l*Nd, pSPARC->pois_const_stress2 + i + l*Nd, pSPARC->pois_const_press + i + l*Nd);            
        }
    }
}


/**
 * @brief   Compute constant coefficients for solving Poisson's equation using FFT
 */
void compute_pois_fft_const(SPARC_OBJ *pSPARC) {
    int Nx_len = pSPARC->isGammaPoint ? (pSPARC->Nx/2+1) : pSPARC->Nx;
    int Nx = pSPARC->Nx;
    int Ny = pSPARC->Ny;
    int Nz = pSPARC->Nz;
    int Nd = pSPARC->Nd;
    // When BC is Dirichelet, one more FD node is incldued.
    // Real length of each side has to be added by 1 mesh length.
    double L1 = pSPARC->delta_x * Nx;
    double L2 = pSPARC->delta_y * Ny;
    double L3 = pSPARC->delta_z * Nz;

    double G[3], g[3];

    for (int l = 0; l < pSPARC->Nkpts_shift; l++) {
        int count = 0;
        for (int k = 0; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx_len; i++) {
                    // G = [(k1-1)*2*pi/L1, (k2-1)*2*pi/L2, (k3-1)*2*pi/L3];
                    g[0] = g[1] = g[2] = 0.0;
                    G[0] = (i < Nx/2+1) ? (i*2*M_PI/L1) : ((i-Nx)*2*M_PI/L1);
                    G[1] = (j < Ny/2+1) ? (j*2*M_PI/L2) : ((j-Ny)*2*M_PI/L2);
                    G[2] = (k < Nz/2+1) ? (k*2*M_PI/L3) : ((k-Nz)*2*M_PI/L3);
                    if (l < pSPARC->Nkpts_shift - 1) { // The last shift is all zeros
                        G[0] += pSPARC->k1_shift[l];
                        G[1] += pSPARC->k2_shift[l];
                        G[2] += pSPARC->k3_shift[l];
                    }
                    matrixTimesVec_3d(pSPARC->lapcT, G, g);
                    double G2 = G[0] * g[0] + G[1] * g[1] + G[2] * g[2];
                    singularity_remooval_const(pSPARC, G2, pSPARC->pois_const + count + l*Nd, 
                        pSPARC->pois_const_stress + count + l*Nd, pSPARC->pois_const_stress2 + count + l*Nd, pSPARC->pois_const_press + count + l*Nd);
                    count ++;
                }
            }
        }
    }
}


/**
 * @brief   Compute constant coefficients for solving Poisson's equation using FFT
 * 
 *          Spherical Cutoff - Method by James Spencer and Ali Alavi 
 *          DOI: 10.1103/PhysRevB.77.193110
 * 
 *          Note: using Finite difference approximation of G2
 *          Note: it's not used in the code. Only use exact G2. 
 */
void compute_pois_fft_const_FD(SPARC_OBJ *pSPARC) 
{
#define alpha(i,j,k) alpha[(k)*Nxy+(j)*Nxh+(i)]
    int FDn, i, j, k, p, Nx, Nxh, Ny, Nz, Ndc, Nxy;
    double *w2_x, *w2_y, *w2_z, V, R_c, *alpha;

    Nx = pSPARC->Nx;
    Nxh = pSPARC->Nx / 2 + 1;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;
    Ndc = Nz * Ny * Nxh;
    Nxy = Ny * Nxh;
    FDn = pSPARC->order / 2;
    // scaled finite difference coefficients
    w2_x = pSPARC->D2_stencil_coeffs_x;
    w2_y = pSPARC->D2_stencil_coeffs_y;
    w2_z = pSPARC->D2_stencil_coeffs_z;

    pSPARC->pois_const = (double *)malloc(sizeof(double) * Ndc);
    assert(pSPARC->pois_const != NULL);
    alpha = pSPARC->pois_const;
    /********************************************************************/

    for (k = 0; k < Nz; k++) {
        for (j = 0; j < Ny; j++) {
            for (i = 0; i < Nxh; i++) {
                alpha(i,j,k) = w2_x[0] + w2_y[0] + w2_z[0];
                for (p = 1; p < FDn + 1; p++){
                    alpha(i,j,k) += 2*(w2_x[p]*cos(2*M_PI/Nx*p*i) 
                        + w2_y[p]*cos(2*M_PI/Ny*p*j) + w2_z[p]*cos(2*M_PI/Nz*p*k));
                }
            }
        }
    }
    alpha[0] = -1;                                                  // For taking sqrt, will change in the end 
    V = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;        // Volume of the domain
    R_c = pow(3*V/(4*M_PI),(1.0/3));

    // For singularity issue, use (1-cos(R_c*sqrt(G^2)))/G^2
    for (k = 0; k < Nz; k++) {
        for (j = 0; j < Ny; j++) {
            for (i = 0; i < Nxh; i++) {
                alpha(i,j,k) = -4*M_PI*(1-cos(R_c*sqrt(-alpha(i,j,k))))/alpha(i,j,k);
            }
        }
    }
    alpha[0] = 2*M_PI*(R_c * R_c);
#undef alpha
}


/**
 * @brief   Create communicators between k-point topology and dmcomm.
 * 
 * Notes:   Occupations are only correct within all dmcomm processes.
 *          When not using ACE method, this communicator might be required. 
 */
void create_kpttopo_dmcomm_inter(SPARC_OBJ *pSPARC) 
{
    int i, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (pSPARC->kptcomm_topo != MPI_COMM_NULL && pSPARC->ExxAcc == 0) {
        int nproc_kptcomm;
        int nproc_kptcomm_topo = pSPARC->npNdx_kptcomm * pSPARC->npNdy_kptcomm * pSPARC->npNdz_kptcomm;
        MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);

        int size_bandcomm = nproc_kptcomm / pSPARC->npband;
        int npNd = pSPARC->npNdx * pSPARC->npNdy * pSPARC->npNdz;
        int *ndm_list = (int *) calloc(sizeof(int), nproc_kptcomm_topo);
        assert(ndm_list != NULL);

    #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
    #endif
        int count_ndm = 0;
        for (i = 0; i < nproc_kptcomm_topo; i++) {
            if (i >= pSPARC->npband * size_bandcomm || i % size_bandcomm >= npNd) 
                ndm_list[count_ndm++] = i;
        }
        if (count_ndm == 0) {
            pSPARC->kpttopo_dmcomm_inter = MPI_COMM_NULL;
            pSPARC->flag_kpttopo_dm = 0;
        } else {
            pSPARC->flag_kpttopo_dm = 1;
            MPI_Group kpttopo_group, kpttopo_group_excl, kpttopo_group_incl;
            MPI_Comm kpttopo_excl, kpttopo_incl;
            MPI_Comm_group(pSPARC->kptcomm_topo, &kpttopo_group);
            MPI_Group_incl(kpttopo_group, count_ndm, ndm_list, &kpttopo_group_excl);
            MPI_Group_excl(kpttopo_group, count_ndm, ndm_list, &kpttopo_group_incl);

            MPI_Comm_create_group(pSPARC->kptcomm_topo, kpttopo_group_excl, 110, &kpttopo_excl);
            MPI_Comm_create_group(pSPARC->kptcomm_topo, kpttopo_group_incl, 110, &kpttopo_incl);
            // yong excl zhiyao yige list
            if (kpttopo_excl != MPI_COMM_NULL) {
                MPI_Intercomm_create(kpttopo_excl, 0, pSPARC->kptcomm_topo, 0, 111, &pSPARC->kpttopo_dmcomm_inter);
                pSPARC->flag_kpttopo_dm_type = 2;               // - recv
            } else {
                MPI_Intercomm_create(kpttopo_incl, 0, pSPARC->kptcomm_topo, ndm_list[0], 111, &pSPARC->kpttopo_dmcomm_inter);
                pSPARC->flag_kpttopo_dm_type = 1;               // - send
            }

            if (kpttopo_excl != MPI_COMM_NULL)
                MPI_Comm_free(&kpttopo_excl);
            if (kpttopo_incl != MPI_COMM_NULL)
                MPI_Comm_free(&kpttopo_incl);
            MPI_Group_free(&kpttopo_group);
            MPI_Group_free(&kpttopo_group_excl);
            MPI_Group_free(&kpttopo_group_incl);
        }
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank && count_ndm > 0) {
            printf("\nThere are %d processes need to get correct occupations in each kptcomm_topo.\n",count_ndm);
            printf("\n--set up kpttopo_dmcomm_inter took %.3f ms\n",(t2-t1)*1000);
        }
    #endif  

        free(ndm_list);
    } else {
        pSPARC->kpttopo_dmcomm_inter = MPI_COMM_NULL;
        pSPARC->flag_kpttopo_dm = 0;
    }
}

/**
 * @brief   Compute the coefficient for G = 0 term in auxiliary function method
 * 
 *          Note: Using QE's method. Not the original one by Gygi 
 */
void auxiliary_constant(SPARC_OBJ *pSPARC) 
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int Nx, Ny, Nz, i, j, k, ihf, jhf, khf, Kx_hf, Ky_hf, Kz_hf;
    double L1, L2, L3, V, alpha, scaled_intf, sumfGq;
    double tpiblx, tpibly, tpiblz, q[3], g[3], G[3], modGq, omega, omega2;

    Nx = pSPARC->Nx;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;

    Kx_hf = pSPARC->Kx_hf;
    Ky_hf = pSPARC->Ky_hf;
    Kz_hf = pSPARC->Kz_hf;

    // When BC is Dirichelet, one more FD node is incldued.
    // Real length of each side has to be added by 1 mesh length.
    L1 = pSPARC->delta_x * Nx;
    L2 = pSPARC->delta_y * Ny;
    L3 = pSPARC->delta_z * Nz;

    tpiblx = 2 * M_PI / L1;
    tpibly = 2 * M_PI / L2;
    tpiblz = 2 * M_PI / L3;
    
    double ecut = ecut_estimate(pSPARC->delta_x, pSPARC->delta_y, pSPARC->delta_z);
    alpha = 10.0/(2.0*ecut);

#ifdef DEBUG
    if(!rank)
        printf("Ecut estimation is %.2f Ha (%.2f Ry) and alpha within auxiliary function is %.6f\n", ecut, 2*ecut, alpha);
#endif
    
    V =  L1 * L2 * L3 * pSPARC->Jacbdet;    // volume of unit cell
    scaled_intf = V*pSPARC->Nkpts_hf/(4*M_PI*sqrt(M_PI*alpha));
    omega = pSPARC->hyb_range_fock;
    omega2 = omega * omega;

    sumfGq = 0;
    for (khf = 0; khf < Kz_hf; khf++) {
        for (jhf = 0; jhf < Ky_hf; jhf++) {
            for (ihf = 0; ihf < Kx_hf; ihf++) {
                q[0] = ihf * tpiblx / Kx_hf;
                q[1] = jhf * tpibly / Ky_hf;
                q[2] = khf * tpiblz / Kz_hf;
                
                for (k = 0; k < Nz; k++) {
                    for (j = 0; j < Ny; j++) {
                        for (i = 0; i < Nx; i++) {
                            // G = [(k1-1)*2*pi/L1, (k2-1)*2*pi/L2, (k3-1)*2*pi/L3];
                            g[0] = g[1] = g[2] = 0.0;
                            G[0] = (i < Nx/2+1) ? (i*tpiblx) : ((i-Nx)*tpiblx);
                            G[1] = (j < Ny/2+1) ? (j*tpibly) : ((j-Ny)*tpibly);
                            G[2] = (k < Nz/2+1) ? (k*tpiblz) : ((k-Nz)*tpiblz);

                            G[0] += q[0];
                            G[1] += q[1];
                            G[2] += q[2];
                            matrixTimesVec_3d(pSPARC->lapcT, G, g);
                            modGq = G[0] * g[0] + G[1] * g[1] + G[2] * g[2];
                            
                            if (modGq > 1e-8) {
                                if (omega < 0)
                                    sumfGq += exp(-alpha*modGq)/modGq;
                                else
                                    sumfGq += exp(-alpha*modGq)/modGq*(1-exp(-modGq/4.0/omega2));
                            }
                        }
                    }
                }
            }
        }
    }
    

    if (omega < 0) {
        pSPARC->const_aux = scaled_intf + alpha - sumfGq;    
    } else {
        sumfGq += 0.25/omega2;
        int nqq, iq;
        double dq, q_, qq, aa;

        nqq = 100000;
        dq = 5.0/sqrt(alpha)/nqq;
        aa = 0;
        for (iq = 0; iq < nqq; iq++) {
            q_ = dq*(iq+0.5);
            qq = q_*q_;
            aa = aa - exp( -alpha * qq) * exp(-0.25*qq/omega2)*dq;
        }
        aa = 2.0*aa/M_PI + 1.0/sqrt(M_PI*alpha);
        scaled_intf = V*pSPARC->Nkpts_hf/(4*M_PI)*aa;
        pSPARC->const_aux = scaled_intf - sumfGq;
    }
        
#ifdef DEBUG
    if(!rank)
        printf("The constant for zero G (auxiliary function) is %.6f\n", pSPARC->const_aux);
#endif
}


/**
 * @brief   Estimation of Ecut by (pi/h)^2/2
 */
double ecut_estimate(double hx, double hy, double hz)
{
    double dx2_inv, dy2_inv, dz2_inv, h_eff, ecut;

    dx2_inv = 1.0/(hx * hx);
    dy2_inv = 1.0/(hy * hy);
    dz2_inv = 1.0/(hz * hz);
    h_eff = sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
    
    // curve fitting of Ecut2h function
    ecut = exp(-2*log(h_eff)+0.848379709041268);
    return ecut;
}


/**
 * @brief   Find out the unique Bloch vector shifts (k-q)
 */
void find_k_shift(SPARC_OBJ *pSPARC) 
{
#define Kptshift_map(i,j) Kptshift_map[i+j*pSPARC->Nkpts_sym]
    int rank, k_ind, q_ind, i, count, flag;
    double *k1_shift, *k2_shift, *k3_shift;
    double k1_shift_temp, k2_shift_temp, k3_shift_temp;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    k1_shift = (double *) calloc(sizeof(double), pSPARC->Nkpts_sym*pSPARC->Nkpts_hf);
    k2_shift = (double *) calloc(sizeof(double), pSPARC->Nkpts_sym*pSPARC->Nkpts_hf);
    k3_shift = (double *) calloc(sizeof(double), pSPARC->Nkpts_sym*pSPARC->Nkpts_hf);
    pSPARC->Kptshift_map = (int*) calloc(sizeof(int), pSPARC->Nkpts_sym*pSPARC->Nkpts_hf);
    assert(k1_shift != NULL && k2_shift != NULL && k3_shift != NULL && pSPARC->Kptshift_map != NULL);

    int *Kptshift_map = pSPARC->Kptshift_map;
    // k_shift[0] = {0,0,0}. remove later.
    k1_shift[0] = k2_shift[0] = k3_shift[0] = 0.0;
    count = 1;
    for (q_ind = 0; q_ind < pSPARC->Nkpts_hf; q_ind ++) {
        for (k_ind = 0; k_ind < pSPARC->Nkpts_sym; k_ind ++) {
            
            k1_shift_temp = pSPARC->k1[k_ind] - pSPARC->k1_hf[q_ind];
            k2_shift_temp = pSPARC->k2[k_ind] - pSPARC->k2_hf[q_ind];
            k3_shift_temp = pSPARC->k3[k_ind] - pSPARC->k3_hf[q_ind];
            if (fabs(k1_shift_temp) < TEMP_TOL && fabs(k2_shift_temp) < TEMP_TOL 
                && fabs(k3_shift_temp) < TEMP_TOL) {
                Kptshift_map(k_ind, q_ind) = 0;
                continue;
            }

            // flag - 0, new shift;
            flag = 0;
            for (i = 0; i < count; i++) {
                if (fabs(k1_shift[i] - k1_shift_temp) < TEMP_TOL && 
                    fabs(k2_shift[i] - k2_shift_temp) < TEMP_TOL &&
                    fabs(k3_shift[i] - k3_shift_temp) < TEMP_TOL) {
                    flag = i;
                    break;
                }
            }
            if (!flag) {
                k1_shift[count] = k1_shift_temp;
                k2_shift[count] = k2_shift_temp;
                k3_shift[count] = k3_shift_temp;
                Kptshift_map(k_ind, q_ind) = count++;
            } else {
                Kptshift_map(k_ind, q_ind) = flag;
            }
        }
    }
    pSPARC->Nkpts_shift = count;
    // No need to store [0,0,0] shift, which is always there. 
    pSPARC->k1_shift = (double *) calloc(sizeof(double), count - 1);
    pSPARC->k2_shift = (double *) calloc(sizeof(double), count - 1);
    pSPARC->k3_shift = (double *) calloc(sizeof(double), count - 1);
    assert(pSPARC->k1_shift !=NULL && pSPARC->k2_shift != NULL && pSPARC->k3_shift != NULL);

    for (i = 0; i < pSPARC->Nkpts_shift - 1; i++) {
        pSPARC->k1_shift[i] = k1_shift[i+1];
        pSPARC->k2_shift[i] = k2_shift[i+1];
        pSPARC->k3_shift[i] = k3_shift[i+1];
    }

    free(k1_shift);
    free(k2_shift);
    free(k3_shift);
#ifdef DEBUG
    if (!rank) printf("Unique Block vector shift (k-q), Nkpts_shift = %d\n", pSPARC->Nkpts_shift);
    for (int nk = 0; nk < min(pSPARC->Nkpts_shift - 1, 10); nk++) {
        double tpiblx = 2 * M_PI / pSPARC->range_x;
        double tpibly = 2 * M_PI / pSPARC->range_y;
        double tpiblz = 2 * M_PI / pSPARC->range_z;
        if (!rank) printf("k1_shift[%2d]: %8.4f, k2_shift[%2d]: %8.4f, k3_shift[%2d]: %8.4f\n",
            nk,pSPARC->k1_shift[nk]/tpiblx,nk,pSPARC->k2_shift[nk]/tpibly,nk,pSPARC->k3_shift[nk]/tpiblz);
    }
    if (pSPARC->Nkpts_shift - 1 <= 10) {
        if (!rank) printf("k1_shift[%2d]: %8.4f, k2_shift[%2d]: %8.4f, k3_shift[%2d]: %8.4f\n",
            pSPARC->Nkpts_shift - 1,0.0,pSPARC->Nkpts_shift - 1,0.0,pSPARC->Nkpts_shift - 1,0.0);
    }
#endif
#undef Kptshift_map
}


/**
 * @brief   Find out the positive shift exp(i*(k-q)*r) and negative shift exp(-i*(k-q)*r)
 *          for each unique Bloch shifts
 */
void kshift_phasefactor(SPARC_OBJ *pSPARC) {
    if (pSPARC->Nkpts_shift == 1) return;
#define neg_phase(i,j,k,l) neg_phase[(l)*Nd+(k)*Nxy+(j)*Nx+(i)]
#define pos_phase(i,j,k,l) pos_phase[(l)*Nd+(k)*Nxy+(j)*Nx+(i)]

    int i, j, k, l, Nx, Ny, Nxy, Nz, Nd, Nkpts_shift;
    double L1, L2, L3, rx, ry, rz, dot_prod;
    double _Complex *neg_phase, *pos_phase;

    L1 = pSPARC->range_x;
    L2 = pSPARC->range_y;
    L3 = pSPARC->range_z;
    Nx = pSPARC->Nx;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;
    Nxy = Nx * Ny;
    Nd = pSPARC->Nd;
    Nkpts_shift = pSPARC->Nkpts_shift;

    pSPARC->neg_phase = (double _Complex *) calloc(sizeof(double _Complex), Nd * (Nkpts_shift - 1));
    pSPARC->pos_phase = (double _Complex *) calloc(sizeof(double _Complex), Nd * (Nkpts_shift - 1));
    assert(pSPARC->neg_phase != NULL && pSPARC->pos_phase != NULL);
    neg_phase = pSPARC->neg_phase;
    pos_phase = pSPARC->pos_phase;

    for (l = 0; l < Nkpts_shift - 1; l ++) {
        for (k = 0; k < Nz; k ++) {
            for (j = 0; j < Ny; j ++) {
                for (i = 0; i < Nx; i ++) {
                    rx = i * L1 / Nx;
                    ry = j * L2 / Ny;
                    rz = k * L3 / Nz;
                    dot_prod = rx * pSPARC->k1_shift[l] + ry * pSPARC->k2_shift[l] + rz * pSPARC->k3_shift[l];
                    // neg_phase(:,i) = exp(-1i*r*k_shift');
                    neg_phase(i,j,k,l) = cos(dot_prod) - I * sin(dot_prod);
                    // pos_phase(:,i) = exp(1i*r*k_shift');
                    pos_phase(i,j,k,l) = cos(dot_prod) + I * sin(dot_prod);
                }
            }
        }
    }
#undef neg_phase
#undef pos_phase
}


/**
 * @brief   Find out the k-point for hartree-fock exact exchange in local process
 */
void find_local_kpthf(SPARC_OBJ *pSPARC) 
{
    int k, nk_hf, count, kpt_bridge_size, kpt_bridge_rank, rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->kpt_bridge_comm, &kpt_bridge_rank);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);

    pSPARC->kpthf_flag_kptcomm = (int *) calloc(pSPARC->Nkpts_kptcomm, sizeof(int));
    count = 0;
    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
        for (nk_hf = 0; nk_hf < pSPARC->Nkpts_hf_red; nk_hf ++) {
            if (pSPARC->kpts_hf_red_list[nk_hf] == (pSPARC->kpt_start_indx + k)){
                pSPARC->kpthf_flag_kptcomm[k] = 1;
                count ++;
                break;
            }
        }
    }
    pSPARC->Nkpts_hf_kptcomm = count;
    pSPARC->Nkpts_hf_list = (int *) calloc(sizeof(int), kpt_bridge_size);
    MPI_Allgather(&count, 1, MPI_INT, pSPARC->Nkpts_hf_list, 1, MPI_INT, pSPARC->kpt_bridge_comm);

    pSPARC->kpthf_start_indx_list = (int *) calloc(sizeof(int), kpt_bridge_size);
    pSPARC->kpthf_start_indx = pSPARC->kpthf_start_indx_list[0] = 0;                // shift in terms of Nkpts_hf_red
    for (k = 0; k < kpt_bridge_size; k++) {
        if (k < kpt_bridge_rank)
            pSPARC->kpthf_start_indx += pSPARC->Nkpts_hf_list[k];
        if (k > 0)
            pSPARC->kpthf_start_indx_list[k] += pSPARC->kpthf_start_indx_list[k-1] + pSPARC->Nkpts_hf_list[k-1];
    }    

#ifdef DEBUG
    if (!rank) printf("Number of k-point orbitals gathered from each k-point processes:\n");
    for (k = 0; k < kpt_bridge_size; k++) {
        if (!rank) printf("Nkpts_hf_list [%d]: %d\n",k,pSPARC->Nkpts_hf_list[k]);
    }
#endif
}


/**
 * @brief   Estimate memory requirement for hybrid calculation
 */
double estimate_memory_exx(const SPARC_OBJ *pSPARC)
{
    double memory_exx = 0.0;
    int Nd = pSPARC->Nd * pSPARC->Nspinor;
    int Ns = pSPARC->Nstates;
    int Nspin = pSPARC->Nspin;
    int Nkpts_sym = pSPARC->Nkpts_sym;
    int npband = pSPARC->npband;
    int npkpt = pSPARC->npkpt;
    int Nkpts_hf_red = pSPARC->Nkpts_hf_red;
    int Nkpts_hf = pSPARC->Nkpts_hf;
    int Nband = pSPARC->Nband_bandcomm;
    int type_size = (pSPARC->isGammaPoint == 1) ? sizeof(double) : sizeof(double _Complex);
    int dmcomm_size = pSPARC->npNd;

    if (pSPARC->ExxAcc == 0) {
        // storage of psi outer
        int len_full_tot = Nd * Ns * Nkpts_hf_red * Nspin;
        memory_exx += (double) len_full_tot * type_size * 2; 
        // memory for constructing Vx
        if (pSPARC->isGammaPoint == 1) {
            // int index
            memory_exx += (double) Ns * Ns * sizeof(int) * 2;
            // for poissons equations
            int ncol = (pSPARC->ExxMemBatch == 0) ? (Ns * Ns) : pSPARC->ExxMemBatch;
            memory_exx += (double) ncol * Nd * (2 + 2*(dmcomm_size > 1)) * type_size;
        } else {
            // int index
            memory_exx += (double) Ns * Ns * sizeof(int) * Nkpts_hf * 3;
            // for poissons equations
            int ncol = (pSPARC->ExxMemBatch == 0) ? (Ns * Ns * Nkpts_hf) : pSPARC->ExxMemBatch;
            memory_exx += (double) ncol * Nd * (2 + 2*(dmcomm_size > 1)) * type_size;
        }
    } else {
        int len_full_tot = Nd * Ns * Nkpts_hf_red * Nspin;
        if (pSPARC->isGammaPoint == 1) {
            // storage of ACE operator in dmcomm and kptcomm_topo
            memory_exx += (double) len_full_tot * type_size * (npband + 1) * npkpt;
            // memory for constructing ACE
            int reps = (npband == 1) ? 0 : ((npband - 2) / 2 + 1); // ceil((nproc_blacscomm-1)/2)
            if (reps > 0) memory_exx += (double) Nd * Ns * 2 * type_size;
            // int index
            memory_exx += (double) Nband * Ns * 2 * sizeof(int);
            // for poissons equations
            int ncol = (pSPARC->ExxMemBatch == 0) ? (Nband * Nband) : pSPARC->ExxMemBatch;
            memory_exx += (double) ncol * Nd * (2 + 2*(dmcomm_size > 1)) * type_size;
        } else {
            // storage of ACE operator in dmcomm and kptcomm_topo
            memory_exx += (double) len_full_tot * type_size * (npband + 1) * npkpt;
            // memory for constructing ACE
            int reps_kpt = pSPARC->npkpt - 1;
            int reps_band = pSPARC->npband - 1;
            memory_exx += (double) Nd * Ns * Nkpts_hf_red * type_size * (1+(reps_kpt > 0));            
            if (reps_band > 0) memory_exx += (double) Nd * Ns * 2 * type_size;
            // int index
            memory_exx += (double) Ns * Nband * Nkpts_sym * 3 * sizeof(int);
            // for poissons equations
            int ncol = (pSPARC->ExxMemBatch == 0) ? (Nband * Nband * Nkpts_sym) : pSPARC->ExxMemBatch;            
            memory_exx += (double) ncol * Nd * (2 + 2*(dmcomm_size > 1)) * type_size;

        }
    }
    return memory_exx;
}
