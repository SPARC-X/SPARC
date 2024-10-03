/***
 * @file    sq.c
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

#include "sq.h"
#include "occupation.h"
#include "parallelization.h"
#include "tools.h"
#include "sqNlocVecRoutines.h"
#include "sqParallelization.h"
#include "sqEnergy.h"
#include "tools.h"
#include "electrostatics.h"
#include "sqHighTExactExchange.h"

#ifdef SPARCX_ACCEL
#include "accel.h"
#endif

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)

/**
 * SQ TODO list:
 * 1. Newtown Raphson algorithm for fermi level
 * 2. Energy and forces correction
 * 3. Always storage of nolocal projectors
 * 4. Always use spherical Rcut region
 * 5. Function for cell replication
 */


/**
 * @brief   Gauss Quadrature method
 * 
 * @param scf_iter  The scf iteration counter
 */
void GaussQuadrature(SPARC_OBJ *pSPARC, int SCFCount, int spn_i) {
    if (pSPARC->pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int nd, rank;
    int *nloc, DMnx, DMny, DMnz, DMnd;    
    int Nx_loc, Ny_loc;
    double lambda_min, lambda_max, lambda_min_MIN, lambda_max_MAX, x1, x2, *t0;
    double time1, time2;
    SQ_OBJ  *pSQ = pSPARC->pSQ;

    nloc = pSQ->nloc;
    DMnx = pSQ->DMnx_SQ;
    DMny = pSQ->DMny_SQ;
    DMnz = pSQ->DMnz_SQ;    
    DMnd = pSQ->DMnd_SQ;
    Nx_loc = pSQ->Nx_loc;
    Ny_loc = pSQ->Ny_loc;    
    MPI_Comm_rank(pSQ->dmcomm_SQ, & rank);
    //////////////////////////////////////////////////////////////////////

    #ifdef DEBUG
        time2 = MPI_Wtime();
    #endif

    #ifdef SPARCX_ACCEL
	if (pSPARC->useACCEL == 1 && pSPARC->cell_typ == 0 && pSPARC->usefock <=1)
	{
		ACCEL_SQ_LanczosAlgorithm_gauss(pSPARC, DMnx, DMny, DMnz, &lambda_min, &lambda_max);
	}
	else
	#endif // SPARCX_ACCEL
	{
    	t0 = (double *) malloc(sizeof(double) * pSQ->Nd_loc);
    	int center = nloc[0] + nloc[1]*Nx_loc + nloc[2]*Nx_loc*Ny_loc;    
    	for (nd = 0; nd < DMnd; nd ++) {
        	// initialize t0 as identity vector
        	memset(t0, 0, sizeof(double)*pSQ->Nd_loc);
        	t0[center] = 1.0;

            LanczosAlgorithm_gauss(pSPARC, t0, &lambda_min, &lambda_max, nd, spn_i);
        	pSQ->mineig[nd+spn_i*DMnd]  = lambda_min;
        	pSQ->maxeig[nd+spn_i*DMnd]  = lambda_max;
    	}
		free(t0);
	}

    // This barrier fixed the severe slowdown of MPI communication on hive
    MPI_Barrier(pSQ->dmcomm_SQ);    

    #ifdef DEBUG
        time1 = MPI_Wtime();
        if(!rank) printf("Rank %d finished Lanczos taking %.3f ms\n",rank, (time1-time2)*1e3); 
    #endif
}


/**
 * @brief   Lanczos algorithm for Gauss method
 * 
 * @param vkm1  Initial guess for Lanczos algorithm
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 * @param nd            FD node index in current process domain
 */
void LanczosAlgorithm_gauss(SPARC_OBJ *pSPARC, double *vkm1, double *lambda_min, double *lambda_max, int nd, int spn_i) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    SQ_OBJ* pSQ  = pSPARC->pSQ;    
    double *vk, *vkp1, val, *aa, *bb;

    int DMnd_SQ = pSQ->DMnd_SQ;
    int Nd_loc = pSQ->Nd_loc;
    aa = (double *) malloc(sizeof(double)*pSPARC->SQ_npl_g);
    bb = (double *) malloc(sizeof(double)*pSPARC->SQ_npl_g);
    vk = (double *) malloc(sizeof(double *)*Nd_loc);
    vkp1 = (double *) malloc(sizeof(double *)*Nd_loc);
     
    Vector2Norm(vkm1, Nd_loc, &val, MPI_COMM_SELF);

    size_t le_count = 0;
    double *lanczos_vec = (pSPARC->sqHighTFlag == 1) ? pSQ->lanczos_vec_all[nd + spn_i*DMnd_SQ] : pSQ->lanczos_vec;

    size_t i;
    for (i = 0; i < Nd_loc; i++) {
        vkm1[i] = vkm1[i]/val;
        lanczos_vec[le_count ++] = vkm1[i];
    }

    // vk=Hsub*vkm1. Here vkm1 is the vector and i,j,k are the indices of node in 
    // proc domain to which the Hsub corresponds to and the indices are w.r.t PR domain
    HsubTimesVec(pSPARC, vkm1, nd, vk);

    Vector2Norm(vk, Nd_loc, &val, MPI_COMM_SELF);

    VectorDotProduct(vkm1, vk, Nd_loc, &val, MPI_COMM_SELF);

    aa[0] = val;
    for (i = 0; i < Nd_loc; i++) {
        vk[i] = vk[i] - aa[0] * vkm1[i];
    }

    Vector2Norm(vk, Nd_loc, &val, MPI_COMM_SELF);

    bb[0] = val;

    for (i = 0; i < Nd_loc; i++) {
        vk[i] = vk[i] / bb[0];
    }

    for (i = 0; i < Nd_loc; i++) {
        lanczos_vec[le_count ++] = vk[i];
    }

    int count = 0;
    //double dl, dm, lmin_prev = 0.0, lmax_prev = 0.0;
    while (count < pSPARC->SQ_npl_g - 1) {
        HsubTimesVec(pSPARC, vk, nd, vkp1); // vkp1=Hsub*vk

        VectorDotProduct(vk, vkp1, Nd_loc, &val, MPI_COMM_SELF); // val=vk'*vkp1
        aa[count + 1] = val;
        for (i = 0; i < Nd_loc; i++) {
             vkp1[i] = vkp1[i] - aa[count + 1] * vk[i] - bb[count] * vkm1[i];
        }

        Vector2Norm(vkp1, Nd_loc, & val, MPI_COMM_SELF);

        bb[count + 1] = val;
        for (i = 0; i < Nd_loc; i++) {
            vkm1[i] = vk[i];
            vk[i] = vkp1[i] / val;
        }
        // if (count != pSPARC->SQ_npl_g - 2 && pSPARC->SQ_typ_dm == 2) {
        if (count != pSPARC->SQ_npl_g - 2) {
            for (i = 0; i < Nd_loc; i++) {
                lanczos_vec[le_count ++] = vk[i];
            }
        }
        count = count + 1;
    }

    TridiagEigenSolve_gauss(pSPARC, aa, bb, nd, spn_i, lambda_min , lambda_max);

    free(vk);
    free(vkp1);
    free(aa);
    free(bb);
}

/**
 * @brief   Tridiagonal eigenvalue solver for Gauss method
 * 
 * @param diag          Diagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain 
 * @param subdiag       Subdiagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain 
 * @param nd            FD node index in current process domain
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 */
void TridiagEigenSolve_gauss(SPARC_OBJ *pSPARC, double *diag, double *subdiag, int nd, int spn_i, double *lambda_min, double *lambda_max) {
    int m, l, iter, i, j, k, n;
    double s, r, p, g, f, dd, c, b, *d, *e, **z;
    SQ_OBJ* pSQ  = pSPARC->pSQ;

    int DMnd_SQ = pSQ->DMnd_SQ;
    n = pSPARC->SQ_npl_g;
    d = (double *) malloc(sizeof(double) * n);
    e = (double *) malloc(sizeof(double) *  n);
    z = (double **) malloc(sizeof(double*) * n);
    for (i = 0; i < n; i++)
        z[i] = (double *) malloc(sizeof(double) * n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j)
                z[i][j] = 1.0;
            else
                z[i][j] = 0.0;
        }
    }

    // create copy of diag and subdiag in d and e
    for (i = 0; i < n; i++) {
        d[i] = diag[i];
        e[i] = subdiag[i];
    }

    // e has the subdiagonal elements 
    // ignore last element(n-1) of e, make it zero
    e[n - 1] = 0.0;

    for (l = 0; l <= n - 1; l++) {
        iter = 0;
        do {
            for (m = l; m <= n - 2; m++) {
                dd = fabs(d[m]) + fabs(d[m + 1]);
                if ((double)(fabs(e[m]) + dd) == dd) break;
            }
            if (m != l) {
                if (iter++ == 200) {
                    printf("Too many iterations in Tridiagonal solver\n");
                    exit(1);
                }
                g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                r = sqrt(g * g + 1.0); // pythag
                g = d[m] - d[l] + e[l] / (g + SIGN(r, g)); 
                s = c = 1.0;
                p = 0.0;

                for (i = m - 1; i >= l; i--) {
                    f = s * e[i];
                    b = c * e[i];
                    e[i + 1] = (r = sqrt(g * g + f * f));
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p = s * r);
                    g = c * r - b;
                    // Form eigenvectors (Normalized)
                    for (k = 0; k < n; k++) {
                        f = z[k][i + 1];
                        z[k][i + 1] = s * z[k][i] + c * f;
                        z[k][i] = c * z[k][i] - s * f;
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }

    for (i = 0; i < n; i++) {
        pSQ->gnd[nd+spn_i*DMnd_SQ][i] = d[i] ;
        pSQ->gwt[nd+spn_i*DMnd_SQ][i] = z[0][i] * z[0][i];
    }

    double *w = (pSPARC->sqHighTFlag == 1) ? pSQ->w_all[nd+spn_i*DMnd_SQ] : pSQ->w;
    int count = 0;
    // Save all eigenvectors w 
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            w[count ++] = z[j][i];
        }
    }

    *lambda_min = d[0]; *lambda_max = d[0];

    for (i = 1; i < n; i++) {
        if (d[i] > * lambda_max) { 
            *lambda_max = d[i];
        } else if (d[i] < * lambda_min) { 
            *lambda_min = d[i];
        }
    }
    // free memory
    free(d);
    free(e);
    for (i = 0; i < n; i++)
        free(z[i]);
    free(z);

    return;
}



/**
 * @brief Calculate nodal Hamiltonian times vector routines
 * 
 * @param pSPARC    SPARC object pointer
 * @param x         vector
 * @param nd        number of grid points
 * @param Hx        nodal Hamiltonian times vecotr
 */
void HsubTimesVec(SPARC_OBJ *pSPARC, const double *x, const int nd, double *Hx)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;    
    // Apply Laplacian 
    Lap_vec_mult_orth_SQ(pSPARC, x, -0.5, pSQ->Veff_PR, nd, Hx);
    // Apply nonlocal projector
    Vnl_vec_mult_SQ(pSPARC, pSQ->Nd_loc, pSPARC->Atom_Influence_nloc_SQ[nd], 
                  pSPARC->nlocProj_SQ[nd], x, Hx);
    
    if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0)) {
        exact_exchange_potential_SQ(pSPARC, x, nd, Hx);
    }
}


/**
 * @brief Calculate (a * Laplacian + diag(Veff)) times x 
 * 
 * @param pSPARC    SPARC object pointer
 * @param x         vector
 * @param a         scaling of Laplacian
 * @param Veff      effective potential 
 * @param nd        number of grid points
 * @param Hx        (a * Laplacian + diag(Veff))x 
 */
void Lap_vec_mult_orth_SQ(
    SPARC_OBJ *pSPARC, const double *x, const double a, const double *Veff, int nd, double *Hx)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int FDn = pSPARC->order / 2;

    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nz_loc = pSQ->Nz_loc;
    int NxNy_loc = Nx_loc * Ny_loc;
    
    int Nx_ex = Nx_loc + pSPARC->order;
    int Ny_ex = Ny_loc + pSPARC->order;
    int NxNy_ex = Nx_ex * Ny_ex;

    int DMnx = pSQ->DMnx_SQ;
    int DMny = pSQ->DMny_SQ;
    int DMnxny = DMnx * DMny;

    int k = nd / DMnxny;
    int j = (nd - k * DMnxny) / DMnx;
    int i = nd % DMnx;
    
    // integrate a into coefficients weights
    double *Lap_weights = (double *)malloc(3*(FDn+1)*sizeof(double)); 
    double *Lap_stencil = Lap_weights;
    int p;
    for (p = 0; p < FDn + 1; p++)
    {
        (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
        (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
        (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
    }
    
    double w2_diag;
    w2_diag  = Lap_weights[0];
    w2_diag += Lap_weights[1];
    w2_diag += Lap_weights[2];    

    double *x_ex = pSQ->x_ex;
    // extend into x_ex with padding zeros outside (dirichlet boundary condition)
    // [FDn:Nx_ex-FDn),[FDn:Ny_ex-FDn),[FDn:Nz_ex-FDn)
    restrict_to_subgrid(x, x_ex, 
        Nx_ex, Nx_loc, NxNy_ex, NxNy_loc, 
        FDn, Nx_loc+FDn-1, FDn, Ny_loc+FDn-1, FDn, Nz_loc+FDn-1, 
        0, 0, 0, sizeof(double));

    int stride_y_v = pSQ->DMnx_PR;
    int stride_z_v = pSQ->DMnx_PR * pSQ->DMny_PR;
    // apply a*Lap*x + Veff*x
    stencil_3axis_thread_sq(
        x_ex, FDn, Nx_loc, Nx_ex, NxNy_loc, NxNy_ex, 
        0, Nx_loc, 0, Ny_loc, 0, Nz_loc, FDn, FDn, FDn, 
        Lap_weights, w2_diag, 1.0, Veff, 
        stride_y_v, stride_z_v, i, j, k, Hx );

    free(Lap_weights);
}



/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0)) * x.
 *          Modified from stencil_3axis_thread_v2 function
 *
 * @param x0               : Input domain with extended boundary 
 * @param radius           : Radius of the stencil (radius * 2 = stencil order)
 * @param stride_y         : Distance between y(i, j, k) and y(i, j+1, k)
 * @param stride_y_ex      : Distance between x0(i, j, k) and x0(i, j+1, k)
 * @param stride_z         : Distance between y(i, j, k) and y(i, j, k+1)
 * @param stride_z_ex      : Distance between x0(i, j, k) and x0(i, j, k+1)
 * @param [x_spos, x_epos) : X index range of y that will be computed in this kernel
 * @param [y_spos, y_epos) : Y index range of y that will be computed in this kernel
 * @param [z_spos, z_epos) : Z index range of y that will be computed in this kernel
 * @param x_ex_spos        : X start index in x0 that will be computed in this kernel
 * @param y_ex_spos        : Y start index in x0 that will be computed in this kernel
 * @param z_ex_spos        : Z start index in x0 that will be computed in this kernel
 * @param stencil_coefs    : Stencil coefficients for the stencil points, length radius+1,
 *                           ordered as [x_0 y_0 z_0 x_1 y_1 y_2 ... x_radius y_radius z_radius]
 * @param coef_0           : Stencil coefficient for the center element 
 * @param b                : Scaling factor of v0
 * @param v0               : Values of the diagonal matrix
 * @param stride_y_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param stride_z_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param istart           : start index in x-direction of v0 
 * @param jstart           : start index in y-direction of v0 
 * @param kstart           : start index in z-direction of v0 
 * @param y (OUT)          : Output domain with original boundary
 *
 */
void stencil_3axis_thread_sq(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b, const double *v0, 
    const int stride_y_v, const int stride_z_v, 
    const int istart, const int jstart, const int kstart, 
    double *y
)
{
    switch (radius)
    {
        case 6:
            stencil_3axis_thread_radius6_sq(
                x0, radius, stride_y,  stride_y_ex, stride_z, stride_z_ex,
                x_spos, x_epos, y_spos, y_epos, z_spos, z_epos, x_ex_spos, y_ex_spos, z_ex_spos,
                stencil_coefs, coef_0, b, v0, stride_y_v, stride_z_v, istart, jstart, kstart, y
            );
            return;
            break;

        default:
            stencil_3axis_thread_variable_radius_sq(
                x0, radius, stride_y,  stride_y_ex, stride_z, stride_z_ex,
                x_spos, x_epos, y_spos, y_epos, z_spos, z_epos, x_ex_spos, y_ex_spos, z_ex_spos,
                stencil_coefs, coef_0, b, v0, stride_y_v, stride_z_v, istart, jstart, kstart, y
            );
            return;
            break;
    }
}


/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0)) * x.
 *          Modified from stencil_3axis_thread_v2 function
 *
 * @param x0               : Input domain with extended boundary 
 * @param radius           : Radius of the stencil (radius * 2 = stencil order)
 * @param stride_y         : Distance between y(i, j, k) and y(i, j+1, k)
 * @param stride_y_ex      : Distance between x0(i, j, k) and x0(i, j+1, k)
 * @param stride_z         : Distance between y(i, j, k) and y(i, j, k+1)
 * @param stride_z_ex      : Distance between x0(i, j, k) and x0(i, j, k+1)
 * @param [x_spos, x_epos) : X index range of y that will be computed in this kernel
 * @param [y_spos, y_epos) : Y index range of y that will be computed in this kernel
 * @param [z_spos, z_epos) : Z index range of y that will be computed in this kernel
 * @param x_ex_spos        : X start index in x0 that will be computed in this kernel
 * @param y_ex_spos        : Y start index in x0 that will be computed in this kernel
 * @param z_ex_spos        : Z start index in x0 that will be computed in this kernel
 * @param stencil_coefs    : Stencil coefficients for the stencil points, length radius+1,
 *                           ordered as [x_0 y_0 z_0 x_1 y_1 y_2 ... x_radius y_radius z_radius]
 * @param coef_0           : Stencil coefficient for the center element 
 * @param b                : Scaling factor of v0
 * @param v0               : Values of the diagonal matrix
 * @param stride_y_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param stride_z_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param istart           : start index in x-direction of v0 
 * @param jstart           : start index in y-direction of v0 
 * @param kstart           : start index in z-direction of v0 
 * @param y (OUT)          : Output domain with original boundary
 *
 */
void stencil_3axis_thread_variable_radius_sq(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b, const double *v0, 
    const int stride_y_v, const int stride_z_v, 
    const int istart, const int jstart, const int kstart, 
    double *y
)
{
    int i, j, k, jp, kp, r;
    const int shift_ip = x_ex_spos - x_spos;
    for (k = z_spos, kp = z_ex_spos; k < z_epos; k++, kp++)
    {
        for (j = y_spos, jp = y_ex_spos; j < y_epos; j++, jp++)
        {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = kp * stride_z_ex + jp * stride_y_ex;
            int offset_v = (k+kstart) * stride_z_v + (j+jstart) * stride_y_v;
            #pragma omp simd
            for (i = x_spos; i < x_epos; i++)
            {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_ex = offset_ex + ip;
                int idx_v = offset_v + (i+istart);

                double res = coef_0 * x0[idx_ex];
                for (r = 1; r <= radius; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double res_x = (x0[idx_ex - r]          + x0[idx_ex + r])          * stencil_coefs[3*r];
                    double res_y = (x0[idx_ex - stride_y_r] + x0[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    double res_z = (x0[idx_ex - stride_z_r] + x0[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    res += res_x + res_y + res_z;
                }
                y[idx] = res + b * (v0[idx_v] * x0[idx_ex]); 
            }
        }
    }
}


/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0)) * x.
 *          Modified from stencil_3axis_thread_v2 function
 *
 * @param x0               : Input domain with extended boundary 
 * @param radius           : Radius of the stencil (radius * 2 = stencil order)
 * @param stride_y         : Distance between y(i, j, k) and y(i, j+1, k)
 * @param stride_y_ex      : Distance between x0(i, j, k) and x0(i, j+1, k)
 * @param stride_z         : Distance between y(i, j, k) and y(i, j, k+1)
 * @param stride_z_ex      : Distance between x0(i, j, k) and x0(i, j, k+1)
 * @param [x_spos, x_epos) : X index range of y that will be computed in this kernel
 * @param [y_spos, y_epos) : Y index range of y that will be computed in this kernel
 * @param [z_spos, z_epos) : Z index range of y that will be computed in this kernel
 * @param x_ex_spos        : X start index in x0 that will be computed in this kernel
 * @param y_ex_spos        : Y start index in x0 that will be computed in this kernel
 * @param z_ex_spos        : Z start index in x0 that will be computed in this kernel
 * @param stencil_coefs    : Stencil coefficients for the stencil points, length radius+1,
 *                           ordered as [x_0 y_0 z_0 x_1 y_1 y_2 ... x_radius y_radius z_radius]
 * @param coef_0           : Stencil coefficient for the center element 
 * @param b                : Scaling factor of v0
 * @param v0               : Values of the diagonal matrix
 * @param stride_y_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param stride_z_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param istart           : start index in x-direction of v0 
 * @param jstart           : start index in y-direction of v0 
 * @param kstart           : start index in z-direction of v0 
 * @param y (OUT)          : Output domain with original boundary
 *
 */
void stencil_3axis_thread_radius6_sq(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b, const double *v0, 
    const int stride_y_v, const int stride_z_v, 
    const int istart, const int jstart, const int kstart, 
    double *y
)
{
    int i, j, k, jp, kp, r;

    const int shift_ip = x_ex_spos - x_spos;
    for (k = z_spos, kp = z_ex_spos; k < z_epos; k++, kp++)
    {
        for (j = y_spos, jp = y_ex_spos; j < y_epos; j++, jp++)
        {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = kp * stride_z_ex + jp * stride_y_ex;
            int offset_v = (k+kstart) * stride_z_v + (j+jstart) * stride_y_v;

            #pragma omp simd
            for (i = x_spos; i < x_epos; i++)
            {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_ex = offset_ex + ip;
                int idx_v = offset_v + (i+istart);

                double res = coef_0 * x0[idx_ex];
                for (r = 1; r <= 6; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double res_x = (x0[idx_ex - r]          + x0[idx_ex + r])          * stencil_coefs[3*r];
                    double res_y = (x0[idx_ex - stride_y_r] + x0[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    double res_z = (x0[idx_ex - stride_z_r] + x0[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    res += res_x + res_y + res_z;
                }
                y[idx] = res + b * (v0[idx_v] * x0[idx_ex]); 
            }
        }
    }
}



