/***
 * @file    sqExactExchange.c
 * @brief   This file contains the functions for SQ method.
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
#include <time.h>
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#include "sqHighTExactExchange.h"
#include "isddft.h"
#include "sqDensity.h"
#include "sqHighTDensity.h"
#include "sq.h"
#include "electrostatics.h"
#include "tools.h"
#include "parallelization.h"
#include "initialization.h"
#include "sqProperties.h"
#include "stress.h"
#include "kroneckerLaplacian.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)


/**
 * @brief   claculate exact exchange energy using SQ 
 */
void exact_exchange_energy_SQ(SPARC_OBJ *pSPARC)
{
#ifdef DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    if (pSPARC->ExxAcc == 1) {
        evaluate_exact_exchange_energy_Acc_SQ(pSPARC);
    } else {
        evaluate_exact_exchange_energy_SQ(pSPARC);
    }
#ifdef DEBUG
    t2 = MPI_Wtime();
if(!rank) 
    printf("\nEvaluating Exact exchange energy took: %.3f ms\nExact exchange energy %.6f.\n", (t2-t1)*1e3, pSPARC->Eexx);
#endif
}

/**
 * @brief   claculate exact exchange energy in SQ by solving poisson's equations
 */
void evaluate_exact_exchange_energy_SQ(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int DMnd = pSQ->DMnd_SQ;
    int *nloc = pSQ->nloc;
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nz_loc = pSQ->Nz_loc;
    int Nd_loc = pSQ->Nd_loc;
    int NxNy_loc = Nx_loc*Ny_loc;
    int center = nloc[0] + nloc[1]*Nx_loc + nloc[2]*NxNy_loc;
    
    double Eex = 0;
    int DMVertices[6] = {0, Nx_loc-1, 0, Ny_loc-1, 0, Nz_loc-1}; 
    double *rhs = (double *)malloc( Nd_loc * sizeof(double) );
    double *d_cor = (double *)malloc( Nd_loc * sizeof(double) );
    double *sol = (double *)malloc( Nd_loc * sizeof(double) );
    for (int nd = 0; nd < DMnd; nd++) {
        for (int i = 0; i < Nd_loc; i++) rhs[i] = pSQ->Dn[nd][i]*pSQ->Dn[nd][i];
        poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
        apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
        for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
        LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                 rhs, pSQ->kron_lap->inv_eig, sol);
        Eex += sol[center];
    }

    Eex *= (-pSPARC->dV);
    MPI_Allreduce(MPI_IN_PLACE, &Eex, 1, MPI_DOUBLE, MPI_SUM, pSQ->dmcomm_SQ);    
    pSPARC->Eexx = Eex * pSPARC->exx_frac;

    free(rhs);
    free(d_cor);
    free(sol);
}


/**
 * @brief   claculate exact exchange energy in SQ by using the basis
 */
void evaluate_exact_exchange_energy_Acc_SQ(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int DMnd = pSQ->DMnd_SQ;
    int *nloc = pSQ->nloc;    
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;    
    int Nd_loc = pSQ->Nd_loc;
    int NxNy_loc = Nx_loc*Ny_loc;
    int center = nloc[0] + nloc[1]*Nx_loc + nloc[2]*NxNy_loc;

    double Eex = 0;
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int indx = 0; indx < DMnd; indx++) {
            for (int i = 0; i < Nd_loc; i++) {
                Eex -= pSQ->Dn[indx][i] * pSQ->basis[center+i*pSQ->Nd_loc] * pSQ->Dn[indx][i] * pSQ->erfcR[center+i*pSQ->Nd_loc];
            }
        }
    } else {
        for (int indx = 0; indx < DMnd; indx++) {
            for (int i = 0; i < Nd_loc; i++) {
                Eex -= pSQ->Dn[indx][i] * pSQ->basis[center+i*pSQ->Nd_loc] * pSQ->Dn[indx][i];
            }
        }
    }
    
    Eex *= pSPARC->dV;
    MPI_Allreduce(MPI_IN_PLACE, &Eex, 1, MPI_DOUBLE, MPI_SUM, pSQ->dmcomm_SQ);
    pSPARC->Eexx = Eex * pSPARC->exx_frac;
}


/**
 * @brief   claculate exact exchange potential using SQ 
 */
void exact_exchange_potential_SQ(SPARC_OBJ *pSPARC, const double *x, const int indx, double *Hx)
{
    double t1, t2;
    t1 = MPI_Wtime();
    if (pSPARC->ExxAcc == 1) {
        evaluate_exact_exchange_potential_ACC_SQ(pSPARC, x, indx, Hx);
    } else {
        evaluate_exact_exchange_potential_SQ(pSPARC, x, indx, Hx);
    }
    t2 = MPI_Wtime();
    pSPARC->Exxtime += (t2 - t1);
}


/**
 * @brief   claculate exact exchange potential in SQ by using the basis
 */
void evaluate_exact_exchange_potential_ACC_SQ(SPARC_OBJ *pSPARC, const double *x, const int indx, double *Hx)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    double *exxPot = (pSPARC->SQ_highT_hybrid_gauss_mem == 1) ? pSQ->exxPot[indx] : pSQ->exxPot[0];
    cblas_dgemv(CblasColMajor, CblasNoTrans, pSQ->Nd_loc, pSQ->Nd_loc, pSPARC->exx_frac, 
                    exxPot, pSQ->Nd_loc, x, 1, 1.0, Hx, 1);
}


/**
 * @brief   claculate exact exchange potential in SQ by solving poisson's equations
 */
void evaluate_exact_exchange_potential_SQ(SPARC_OBJ *pSPARC, const double *x, const int indx, double *Hx)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int DMnx = pSQ->DMnx_SQ;
    int DMny = pSQ->DMny_SQ;
    int *nloc = pSQ->nloc;
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nz_loc = pSQ->Nz_loc;
    int Nd_loc = pSQ->Nd_loc;
    
    int DMVertices[6] = {0, Nx_loc-1, 0, Ny_loc-1, 0, Nz_loc-1}; 
    double *rhs = (double *)malloc( Nd_loc * sizeof(double) );
    double *d_cor = (double *)malloc( Nd_loc * sizeof(double) );
    double *sol = (double *)malloc( Nd_loc * sizeof(double) );

    // node indx (i_dm, j_dm, k_dm) in local dmcomm_SQ domain
    int k_dm = indx / (DMnx * DMny);
    int j_dm = (indx - k_dm * (DMnx * DMny)) / DMnx;
    int i_dm = indx % DMnx;
    k_dm += nloc[2];
    j_dm += nloc[1];
    i_dm += nloc[0];

    int count = 0;
    for (int k_loc = -nloc[2]; k_loc <= nloc[2]; k_loc++) {
        for (int j_loc = -nloc[1]; j_loc <= nloc[1]; j_loc++) {
            for (int i_loc = -nloc[0]; i_loc <= nloc[0]; i_loc++) {
                int indx_ = (i_loc+i_dm) + (j_loc+j_dm)*pSQ->DMnx_PR + (k_loc+k_dm)*pSQ->DMnx_PR*pSQ->DMny_PR;
                assert(indx_>=0 && indx_<pSQ->DMnd_PR);

                poisson_RHS_hybrid_eccentric(pSPARC, pSQ->Dn_exx[indx_], i_loc, j_loc, k_loc, count, Nx_loc, Ny_loc, Nz_loc, x, rhs);
                apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
                for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
                LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                            rhs, pSQ->kron_lap->inv_eig, sol);
                Hx[count] -= pSPARC->exx_frac * sol[count];
                count++;
            }
        }
    }

    free(rhs);
    free(d_cor);
    free(sol);
}


/**
 * @brief   calculate poisson's right hand side for hybrid functional
 */
void poisson_RHS_hybrid(SPARC_OBJ *pSPARC, double *rhs, int Nd)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        int *nloc = pSQ->nloc;
        int Nx_loc = pSQ->Nx_loc;
        int Ny_loc = pSQ->Ny_loc;
        int NxNy_loc = Nx_loc*Ny_loc;
        int center = nloc[0] + nloc[1]*Nx_loc + nloc[2]*NxNy_loc;

        for (int i = 0; i < Nd; i++) {
            rhs[i] = (-4*M_PI) * rhs[i] * pSQ->erfcR[i+center*Nd];
        }
    } else {
        for (int i = 0; i < Nd; i++) {
            rhs[i] = (-4*M_PI) * rhs[i];
        }
    }
}


/**
 * @brief   calculate poisson's right hand side for hybrid functional
 * 
 * Note:    this is used when rhs and density matrix are eccentric
 */
void poisson_RHS_hybrid_eccentric(SPARC_OBJ *pSPARC, 
                    const double *Dn, const int cDx, const int cDy, const int cDz, const int indx, 
                    const int Nx, const int Ny, const int Nz, const double *x, double *rhs)
{
#define rhs(i,j,k) rhs[(i)+(j)*Nx+(k)*NxNy]
#define x(i,j,k) x[(i)+(j)*Nx+(k)*NxNy]
#define erfcR(i,j,k,l) erfcR[(i)+(j)*Nx+(k)*NxNy+(l)*Nd]

    SQ_OBJ *pSQ = pSPARC->pSQ; 
    restrict_Dn(Dn, cDx, cDy, cDz, Nx, Ny, Nz, rhs);

    int NxNy = Nx * Ny;
    int x_o_spos = max(0,cDx);
    int x_o_epos = min(Nx+cDx,Nx)-1;
    int y_o_spos = max(0,cDy);
    int y_o_epos = min(Ny+cDy,Ny)-1;
    int z_o_spos = max(0,cDz);
    int z_o_epos = min(Nz+cDz,Nz)-1;
    
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        int Nd = NxNy * Nz;        
        double *erfcR = pSQ->erfcR;
        for (int k = z_o_spos; k <= z_o_epos; k++) {
            for (int j = y_o_spos; j <= y_o_epos; j++) {
                for (int i = x_o_spos; i <= x_o_epos; i++) {
                    rhs(i,j,k) *= (-4*M_PI*x(i,j,k)*erfcR(i,j,k,indx));
                }
            }
        }
    } else {
        for (int k = z_o_spos; k <= z_o_epos; k++) {
            for (int j = y_o_spos; j <= y_o_epos; j++) {
                for (int i = x_o_spos; i <= x_o_epos; i++) {
                    rhs(i,j,k) *= (-4*M_PI*x(i,j,k));
                }
            }
        }
    }
    
#undef rhs
#undef x
#undef erfcR
}


/**
 * @brief   collect column of density matrix
 */
void collect_col_of_Density_matrix(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    
    int *nloc = pSQ->nloc;
    int DMnx = pSQ->DMnx_SQ;
    int DMny = pSQ->DMny_SQ;
    int DMnz = pSQ->DMnz_SQ;

    double *send = (double *) malloc(pSQ->DMnd_SQ * sizeof(double));
    double *recv = (double *) malloc(pSQ->DMnd_PR * sizeof(double));
    assert(send != NULL && recv != NULL);

    // when Nd_loc becomes large, the data to be communicated becomes extremely large. 
    // MPI would fail because of limited cache. Thus we do it Nd_loc times. 
    for (int ind = 0; ind < pSQ->Nd_loc; ind++) {
        for (int i = 0; i < pSQ->DMnd_SQ; i++) {
            send[i] = pSQ->Dn[i][ind];
        }
        D2Dext(pSQ->d2dext_dmcomm_sq, pSQ->d2dext_dmcomm_sq_ext, DMnx, DMny, DMnz, 
            nloc[0], nloc[1], nloc[2], send, recv, pSQ->dmcomm_d2dext_sq, sizeof(double));
        for (int i = 0; i < pSQ->DMnd_PR; i++) {
            pSQ->Dn_exx[i][ind] = recv[i];
        }
    }

    free(send);
    free(recv);
}


/**
 * @brief   Calculate the basis, i.e. the solution of poissons equation with e_n as rhs
 */
void calculate_basis(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int rank, size;
    MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);
    MPI_Comm_size(pSQ->dmcomm_SQ, &size);
    
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nz_loc = pSQ->Nz_loc;
    int Nd_loc = pSQ->Nd_loc;
    int DMVertices[6] = {0, Nx_loc-1, 0, Ny_loc-1, 0, Nz_loc-1};

    int *rcounts = (int *) malloc(sizeof(int) * size);
    int *rdispls = (int *) malloc(sizeof(int) * size);
    assert(rcounts != NULL && rdispls != NULL);
    rdispls[0] = 0;
    for (int i = 0; i < size; i++) {
        rcounts[i] = Nd_loc / size + ((i < Nd_loc % size) ? 1 : 0);
        if (i < size - 1) {
            rdispls[i+1] = rdispls[i] + rcounts[i];
        }
    }

    double *rhs = (double *) malloc(sizeof(double) * Nd_loc);
    double *d_cor = (double *) calloc(sizeof(double), Nd_loc);
    assert(rhs != NULL && d_cor != NULL);

    for (int nd = rdispls[rank]; nd < rdispls[rank]+rcounts[rank]; nd++) {
        memset(rhs, 0, sizeof(double)*Nd_loc);
        rhs[nd] = 1*(-4*M_PI);
        apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
        for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
        LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                    rhs, pSQ->kron_lap->inv_eig, pSQ->basis+nd*Nd_loc);
    }

    for (int i = 0; i < size; i++) {
        rcounts[i] *= Nd_loc;
        rdispls[i] *= Nd_loc;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 1, MPI_DOUBLE, pSQ->basis, rcounts, rdispls, MPI_DOUBLE, pSQ->dmcomm_SQ);

    free(rhs);
    free(d_cor);
}


/**
 * @brief   Restrict column of density matrix onto current local domain
 */
void restrict_Dn(const double *Dn, const int cDx, const int cDy, const int cDz, 
                const int Nx, const int Ny, const int Nz, double *Dn_res)
{
    int NxNy = Nx * Ny;
    int x_o_spos = max(0,cDx);
    int x_o_epos = min(Nx+cDx,Nx)-1;
    int y_o_spos = max(0,cDy);
    int y_o_epos = min(Ny+cDy,Ny)-1;
    int z_o_spos = max(0,cDz);
    int z_o_epos = min(Nz+cDz,Nz)-1;
    int x_i_spos = max(0,-cDx);
    int y_i_spos = max(0,-cDy);
    int z_i_spos = max(0,-cDz);

    restrict_to_subgrid(Dn, Dn_res, Nx, Nx, NxNy, NxNy, 
        x_o_spos, x_o_epos, y_o_spos, y_o_epos, z_o_spos, z_o_epos,
        x_i_spos, y_i_spos, z_i_spos, sizeof(double));
}


/**
 * @brief   Compute explicit exact exchange potential matrix 
 */
void compute_exx_potential_SQ(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    for (int indx = 0; indx < pSQ->DMnd_SQ; indx++) {
        compute_exx_potential_node_SQ(pSPARC, indx, pSQ->exxPot[indx]);
    }
}


/**
 * @brief   Compute explicit exact exchange potential matrix for each node
 */
void compute_exx_potential_node_SQ(SPARC_OBJ *pSPARC, int indx, double *exxPot)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int *nloc = pSQ->nloc;
    int DMnx = pSQ->DMnx_SQ;
    int DMny = pSQ->DMny_SQ;
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nz_loc = pSQ->Nz_loc;
    int Nd_loc = pSQ->Nd_loc;
    int DMnd_PR = pSQ->DMnd_PR;

    double *rho = (double *) malloc(sizeof(double)*Nd_loc*Nd_loc);
    assert(rho != NULL);
    
    // local indx w.r.t. domain_SQ
    int k_dm = indx / (DMnx * DMny);
    int j_dm = (indx - k_dm * (DMnx * DMny)) / DMnx;
    int i_dm = indx % DMnx;
    // w.r.t. PR domain
    k_dm += nloc[2];
    j_dm += nloc[1];
    i_dm += nloc[0];

    memset(rho, 0, sizeof(double)*Nd_loc*Nd_loc);
    int count = 0;
    for (int k_loc = -nloc[2]; k_loc <= nloc[2]; k_loc++) {
        for (int j_loc = -nloc[1]; j_loc <= nloc[1]; j_loc++) {
            for (int i_loc = -nloc[0]; i_loc <= nloc[0]; i_loc++) {
                int indx_ = (i_loc+i_dm) + (j_loc+j_dm)*pSQ->DMnx_PR + (k_loc+k_dm)*pSQ->DMnx_PR*pSQ->DMny_PR;
                assert(indx_>=0 && indx_<pSQ->DMnd_PR);
                restrict_Dn(pSQ->Dn_exx[indx_], i_loc, j_loc, k_loc, Nx_loc, Ny_loc, Nz_loc, rho+count*Nd_loc);
                count++;
            }
        }
    }

    // need to transpose density matrix 
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int j = 0; j < pSQ->Nd_loc; j++) {
            for (int i = 0; i < pSQ->Nd_loc; i++) {
                exxPot[i+j*pSQ->Nd_loc] = -rho[j+i*pSQ->Nd_loc] * pSQ->basis[i+j*pSQ->Nd_loc] * pSQ->erfcR[i+j*pSQ->Nd_loc];
            }
        }
    } else {
        for (int j = 0; j < pSQ->Nd_loc; j++) {
            for (int i = 0; i < pSQ->Nd_loc; i++) {
                exxPot[i+j*pSQ->Nd_loc] = -rho[j+i*pSQ->Nd_loc] * pSQ->basis[i+j*pSQ->Nd_loc];
            }
        }
    }
    free(rho);
}


/**
 * @brief   Compute erfc(\mu R) in HSE
 */
void calculate_erfcR(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int rank, size;
    MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);
    MPI_Comm_size(pSQ->dmcomm_SQ, &size);

    int *nloc = pSQ->nloc;
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;    
    int Nd_loc = pSQ->Nd_loc;
    
    int *rcounts = (int *) malloc(sizeof(int) * size);
    int *rdispls = (int *) malloc(sizeof(int) * size);
    assert(rcounts != NULL && rdispls != NULL);
    rdispls[0] = 0;
    for (int i = 0; i < size; i++) {
        rcounts[i] = Nd_loc / size + ((i < Nd_loc % size) ? 1 : 0);
        if (i < size - 1) {
            rdispls[i+1] = rdispls[i] + rcounts[i];
        }
    }

    for (int nd = rdispls[rank]; nd < rdispls[rank]+rcounts[rank]; nd++) {
        double *erfcR = pSQ->erfcR + nd*Nd_loc;        
        int k_dm = nd / (Nx_loc * Ny_loc);
        int j_dm = (nd - k_dm * (Nx_loc * Ny_loc)) / Nx_loc;
        int i_dm = nd % Nx_loc;
        double nx = (i_dm - nloc[0]) * pSPARC->delta_x;
        double ny = (j_dm - nloc[1]) * pSPARC->delta_y;
        double nz = (k_dm - nloc[2]) * pSPARC->delta_z;

        int count = 0;
        for (int k = -nloc[2]; k <= nloc[2]; k++) {
            double rz = k*pSPARC->delta_z;
            for (int j = -nloc[1]; j <= nloc[1]; j++) {
                double ry = j*pSPARC->delta_y;
                for (int i = -nloc[0]; i <= nloc[0]; i++) {
                    double rx = i*pSPARC->delta_x;
                    CalculateDistance(pSPARC, rx, ry, rz, nx, ny, nz, erfcR+count);
                    erfcR[count] = erfc(pSPARC->hyb_range_fock * erfcR[count]);
                    count ++;
                }
            }
        }
    }
    for (int i = 0; i < size; i++) {
        rcounts[i] *= Nd_loc;
        rdispls[i] *= Nd_loc;
    }

    MPI_Allgatherv(MPI_IN_PLACE, 1, MPI_DOUBLE, pSQ->erfcR, rcounts, rdispls, MPI_DOUBLE, pSQ->dmcomm_SQ);
}


/**
 * @brief   Compute exact exchange stress contribution in SQ
 */
void exact_exchange_stress_node_SQ(SPARC_OBJ *pSPARC, double *Dn, 
                double *gradx_Dn, double *grady_Dn, double *gradz_Dn, double *s_exx)
{
    if (pSPARC->ExxAcc == 1) {
        evaluate_exact_exchange_stress_node_ACC_SQ(pSPARC, Dn, gradx_Dn, grady_Dn, gradz_Dn, s_exx);
    } else {
        evaluate_exact_exchange_stress_node_SQ(pSPARC, Dn, gradx_Dn, grady_Dn, gradz_Dn, s_exx);
    }
}


/**
 * @brief   Compute exact exchange stress using basis in SQ for each node
 */
void evaluate_exact_exchange_stress_node_ACC_SQ(SPARC_OBJ *pSPARC, double *Dn, 
                double *gradx_Dn, double *grady_Dn, double *gradz_Dn, double *s_exx)
{
#define s_exx(i,j) s_exx[3*i+j]
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    SQ_OBJ *pSQ = pSPARC->pSQ; 
    int *nloc = pSQ->nloc;
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nd_loc = pSQ->Nd_loc;
    int NxNy_loc = Nx_loc*Ny_loc;
    int center = nloc[0] + nloc[1]*Nx_loc + nloc[2]*NxNy_loc;    
    
    double *rx = (double *) malloc(Nd_loc*sizeof(double));
    double *ry = (double *) malloc(Nd_loc*sizeof(double));
    double *rz = (double *) malloc(Nd_loc*sizeof(double));
    
    int count = 0;    
    for (int k = -nloc[2]; k <= nloc[2]; k++) {
        double z = k*pSPARC->delta_z;
        for (int j = -nloc[1]; j <= nloc[1]; j++) {
            double y = j*pSPARC->delta_y;
            for (int i = -nloc[0]; i <= nloc[0]; i++) {
                double x = i*pSPARC->delta_x;
                rx[count] = x;
                ry[count] = y;
                rz[count] = z;
                count ++;
            }
        }
    }

    // s_exx(x,x)
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(0,0) += 2*Dn[i]*gradx_Dn[i]*rx[i]*pSQ->erfcR[i+center*Nd_loc]*pSQ->basis[center+i*Nd_loc];
        }
    } else {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(0,0) += 2*Dn[i]*gradx_Dn[i]*rx[i]*pSQ->basis[center+i*Nd_loc];
        }
    }
    // s_exx(x,y)
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(0,1) += 2*Dn[i]*gradx_Dn[i]*ry[i]*pSQ->erfcR[i+center*Nd_loc]*pSQ->basis[center+i*Nd_loc];
        }
    } else {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(0,1) += 2*Dn[i]*gradx_Dn[i]*ry[i]*pSQ->basis[center+i*Nd_loc];
        }
    }
    // s_exx(x,z)
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(0,2) += 2*Dn[i]*gradx_Dn[i]*rz[i]*pSQ->erfcR[i+center*Nd_loc]*pSQ->basis[center+i*Nd_loc];
        }
    } else {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(0,2) += 2*Dn[i]*gradx_Dn[i]*rz[i]*pSQ->basis[center+i*Nd_loc];
        }
    }
    // s_exx(y,x)
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(1,0) += 2*Dn[i]*grady_Dn[i]*rx[i]*pSQ->erfcR[i+center*Nd_loc]*pSQ->basis[center+i*Nd_loc];
        }
    } else {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(1,0) += 2*Dn[i]*grady_Dn[i]*rx[i]*pSQ->basis[center+i*Nd_loc];
        }
    }
    // s_exx(y,y)
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(1,1) += 2*Dn[i]*grady_Dn[i]*ry[i]*pSQ->erfcR[i+center*Nd_loc]*pSQ->basis[center+i*Nd_loc];
        }
    } else {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(1,1) += 2*Dn[i]*grady_Dn[i]*ry[i]*pSQ->basis[center+i*Nd_loc];
        }
    }
    // s_exx(y,z)
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(1,2) += 2*Dn[i]*grady_Dn[i]*rz[i]*pSQ->erfcR[i+center*Nd_loc]*pSQ->basis[center+i*Nd_loc];
        }
    } else {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(1,2) += 2*Dn[i]*grady_Dn[i]*rz[i]*pSQ->basis[center+i*Nd_loc];
        }
    }
    // s_exx(z,x)
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(2,0) += 2*Dn[i]*gradz_Dn[i]*rx[i]*pSQ->erfcR[i+center*Nd_loc]*pSQ->basis[center+i*Nd_loc];
        }
    } else {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(2,0) += 2*Dn[i]*gradz_Dn[i]*rx[i]*pSQ->basis[center+i*Nd_loc];
        }
    }
    // s_exx(z,y)
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(2,1) += 2*Dn[i]*gradz_Dn[i]*ry[i]*pSQ->erfcR[i+center*Nd_loc]*pSQ->basis[center+i*Nd_loc];
        }
    } else {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(2,1) += 2*Dn[i]*gradz_Dn[i]*ry[i]*pSQ->basis[center+i*Nd_loc];
        }
    }
    // s_exx(z,z)
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(2,2) += 2*Dn[i]*gradz_Dn[i]*rz[i]*pSQ->erfcR[i+center*Nd_loc]*pSQ->basis[center+i*Nd_loc];
        }
    } else {
        for (int i = 0; i < Nd_loc; i++) {
            s_exx(2,2) += 2*Dn[i]*gradz_Dn[i]*rz[i]*pSQ->basis[center+i*Nd_loc];
        }
    }

#undef s_exx
}


/**
 * @brief   Compute exact exchange stress by solving poisson's equations in SQ for each node
 */
void evaluate_exact_exchange_stress_node_SQ(SPARC_OBJ *pSPARC, double *Dn, 
                double *gradx_Dn, double *grady_Dn, double *gradz_Dn, double *s_exx)
{
#define s_exx(i,j) s_exx[3*i+j]
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    SQ_OBJ *pSQ = pSPARC->pSQ; 
    int *nloc = pSQ->nloc;
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nz_loc = pSQ->Nz_loc;
    int Nd_loc = pSQ->Nd_loc;
    int NxNy_loc = Nx_loc*Ny_loc;
    int center = nloc[0] + nloc[1]*Nx_loc + nloc[2]*NxNy_loc;    
    
    double *rx = (double *) malloc(Nd_loc*sizeof(double));
    double *ry = (double *) malloc(Nd_loc*sizeof(double));
    double *rz = (double *) malloc(Nd_loc*sizeof(double));
    
    int count = 0;    
    for (int k = -nloc[2]; k <= nloc[2]; k++) {
        double z = k*pSPARC->delta_z;
        for (int j = -nloc[1]; j <= nloc[1]; j++) {
            double y = j*pSPARC->delta_y;
            for (int i = -nloc[0]; i <= nloc[0]; i++) {
                double x = i*pSPARC->delta_x;
                rx[count] = x;
                ry[count] = y;
                rz[count] = z;
                count ++;
            }
        }
    }

    int DMVertices[6] = {0, Nx_loc-1, 0, Ny_loc-1, 0, Nz_loc-1}; 
    double *rhs = (double *)malloc( Nd_loc * sizeof(double) );
    double *d_cor = (double *)malloc( Nd_loc * sizeof(double) );
    double *sol = (double *)malloc( Nd_loc * sizeof(double) );

    // s_exx(x,x)
    for (int i = 0; i < Nd_loc; i++) rhs[i] = 2*Dn[i]*gradx_Dn[i]*rx[i];
    poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
    apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
    for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
    LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                rhs, pSQ->kron_lap->inv_eig, sol);
    s_exx(0,0) += sol[center];

    // s_exx(x,y)
    for (int i = 0; i < Nd_loc; i++) rhs[i] = 2*Dn[i]*gradx_Dn[i]*ry[i];
    poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
    apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
    for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
    LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                rhs, pSQ->kron_lap->inv_eig, sol);
    s_exx(0,1) += sol[center];

    // s_exx(x,z)
    for (int i = 0; i < Nd_loc; i++) rhs[i] = 2*Dn[i]*gradx_Dn[i]*rz[i];
    poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
    apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
    for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
    LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                rhs, pSQ->kron_lap->inv_eig, sol);
    s_exx(0,2) += sol[center];

    // s_exx(y,x)
    for (int i = 0; i < Nd_loc; i++) rhs[i] = 2*Dn[i]*grady_Dn[i]*rx[i];
    poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
    apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
    for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
    LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                rhs, pSQ->kron_lap->inv_eig, sol);
    s_exx(1,0) += sol[center];

    // s_exx(y,y)
    for (int i = 0; i < Nd_loc; i++) rhs[i] = 2*Dn[i]*grady_Dn[i]*ry[i];
    poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
    apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
    for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
    LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                rhs, pSQ->kron_lap->inv_eig, sol);
    s_exx(1,1) += sol[center];

    // s_exx(y,z)
    for (int i = 0; i < Nd_loc; i++) rhs[i] = 2*Dn[i]*grady_Dn[i]*rz[i];
    poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
    apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
    for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
    LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                rhs, pSQ->kron_lap->inv_eig, sol);
    s_exx(1,2) += sol[center];

    // s_exx(z,x)
    for (int i = 0; i < Nd_loc; i++) rhs[i] = 2*Dn[i]*gradz_Dn[i]*rx[i];
    poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
    apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
    for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
    LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                rhs, pSQ->kron_lap->inv_eig, sol);
    s_exx(2,0) += sol[center];

    // s_exx(z,y)
    for (int i = 0; i < Nd_loc; i++) rhs[i] = 2*Dn[i]*gradz_Dn[i]*ry[i];
    poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
    apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
    for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
    LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                rhs, pSQ->kron_lap->inv_eig, sol);
    s_exx(2,1) += sol[center];

    // s_exx(z,z)
    for (int i = 0; i < Nd_loc; i++) rhs[i] = 2*Dn[i]*gradz_Dn[i]*rz[i];
    poisson_RHS_hybrid(pSPARC, rhs, Nd_loc);
    apply_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx_loc, Ny_loc, Nz_loc, Nx_loc, Ny_loc, Nz_loc, DMVertices, rhs, d_cor, MPI_COMM_SELF);
    for (int i = 0; i < Nd_loc; i++) rhs[i] -= d_cor[i];        
    LAP_KRON(pSQ->kron_lap->Nx, pSQ->kron_lap->Ny, pSQ->kron_lap->Nz, pSQ->kron_lap->Vx, pSQ->kron_lap->Vy, pSQ->kron_lap->Vz,
                rhs, pSQ->kron_lap->inv_eig, sol);
    s_exx(2,2) += sol[center];
    
    free(rx); free(ry); free(rz);
    free(rhs); free(d_cor); free(sol);
#undef s_exx
}


/**
 * @brief   Compute exact exchange stress in SQ
 */
void Calculate_exact_exchange_stress_SQ(SPARC_OBJ *pSPARC) {
    // exact exchange stress are calculated in SQ force calculation.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG    
    if (!rank){
        printf("\nExact exchange contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_exx, NULL);
    } 
#endif
    return;
}


/**
 * @brief   Compute exact exchange pressure in SQ
 */
void exact_exchange_pressure_SQ(SPARC_OBJ *pSPARC)
{
    if (strcmpi(pSPARC->XC,"HSE")) { // PBE0
        pSPARC->pres_exx = 0;
        return;
    }
    
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int DMnd = pSQ->DMnd_SQ;
    int *nloc = pSQ->nloc;
    int Nd_loc = pSQ->Nd_loc;    
    double mu2 = pSPARC->hyb_range_fock * pSPARC->hyb_range_fock;

    double *r2 = (double *) malloc(sizeof(double) * Nd_loc);
    int count = 0;
    for (int k = -nloc[2]; k <= nloc[2]; k++) {
        double rz = k*pSPARC->delta_z;
        for (int j = -nloc[1]; j <= nloc[1]; j++) {
            double ry = j*pSPARC->delta_y;
            for (int i = -nloc[0]; i <= nloc[0]; i++) {
                double rx = i*pSPARC->delta_x;
                CalculateDistance(pSPARC, rx, ry, rz, 0, 0, 0, r2+count);
                r2[count] *= r2[count];
                count ++;
            }
        }
    }

    double pres_exx = 0;
    for (int nd = 0; nd < DMnd; nd++) {
        for (int i = 0; i < Nd_loc; i++) {
            pres_exx += pSQ->Dn[nd][i]*pSQ->Dn[nd][i]*exp(-mu2*r2[i]);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &pres_exx, 1, MPI_DOUBLE, MPI_SUM, pSQ->dmcomm_SQ);
    pres_exx *= (pSPARC->exx_frac*pSPARC->dV*pSPARC->dV*2*pSPARC->hyb_range_fock/sqrt(M_PI));
    pSPARC->pres_exx = pres_exx;
    free(r2);
}


/**
 * @brief   Compute exact exchange pressure in SQ
 */
void Calculate_exact_exchange_pressure_SQ(SPARC_OBJ *pSPARC) {
    // exact exchange pressure are calculated in SQ force calculation.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG    
    if (!rank){
        printf("Uncounted Pressure contribution from exact exchange: %.15f Ha\n",pSPARC->pres_exx);
    } 
#endif
    return;
}


/**
 * @brief   Free SCF variables for SQ-hybrid
 */
void free_exx_SQ_highT(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;    
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    
    free_kron_Lap(pSQ->kron_lap);
    free(pSQ->kron_lap);
    free_multipole_expansion(pSQ->MpExp_exx, MPI_COMM_SELF);
    free(pSQ->MpExp_exx);
    if (pSPARC->ExxAcc == 1) {
        free(pSQ->basis);
        int nExxPot = (pSPARC->SQ_highT_hybrid_gauss_mem == 1) ? pSQ->DMnd_SQ : 1;        
        for (int i = 0; i < nExxPot; i++) {
            free(pSQ->exxPot[i]);
        }
        free(pSQ->exxPot);
    }
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        free(pSQ->erfcR);
    }
}
