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

#include "sq.h"
#include "sqtool.h"
#include "occupation.h"
#include "parallelization.h"
#include "tools.h"
#include "sqNlocVecRoutines.h"
#include "sqParallelization.h"
#include "sqEnergy.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)

/**
 * SQ TODO list:
 * 1. Clenshaw Curtis for energy
 * 2. Newtown Raphson algorithm for fermi level
 * 3. Energy and forces correction
 * 4. Always storage of nolocal projectors
 * 5. Always use spherical Rcut region
 * 6. Function for cell replication
 * 7. Memory estimation 
 */


/**
 * @brief   Gauss Quadrature method
 * 
 * @param scf_iter  The scf iteration counter
 */
void GaussQuadrature(SPARC_OBJ *pSPARC, int SCFCount) {
    if (pSPARC->pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int i, j, k, ii, jj, kk, nd, rank, count;
    int *nloc, DMnx, DMny, DMnz, DMnxny, DMnd;
    double lambda_min, lambda_max, lambda_min_MIN, lambda_max_MAX, x1, x2;
    double ***t0;
    #ifdef DEBUG
    double time1, time2;
    #endif
    SQ_OBJ  *pSQ = pSPARC->pSQ;

    nloc = pSQ->nloc;
    DMnx = pSQ->Nx_d_SQ;
    DMny = pSQ->Ny_d_SQ;
    DMnz = pSQ->Nz_d_SQ;
    DMnxny = DMnx * DMny;
    DMnd = DMnxny * DMnz;
    MPI_Comm_rank(pSQ->dmcomm_SQ, & rank);
    //////////////////////////////////////////////////////////////////////

    // Step 1, get local Veff 
    // Step 2, communicate to get full values for PR domain
    count = 0;
    for (k = 0; k < DMnz; k++) {
        for (j = 0; j < DMny; j++) {
            for (i = 0; i < DMnx; i++) {
                pSQ->Veff_PR[k+nloc[2]][j+nloc[1]][i+nloc[0]] = pSQ->Veff_loc_SQ[count];
                count++;
            }
        }
    }

    // This barrier fixed the severe slowdown of MPI communication on hive
    MPI_Barrier(pSQ->dmcomm_SQ);
    Transfer_Veff_PR(pSPARC, pSQ->Veff_PR, pSQ->SQ_dist_graph_comm);
    #ifdef DEBUG
    time2 = MPI_Wtime();
    #endif

    t0 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
    for (k = 0; k < 2*nloc[2]+1; k++) {        
        t0[k] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
        for (j = 0; j < 2*nloc[1]+1; j++) {
            t0[k][j] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
        }
    }

    nd = 0;
    for (k = nloc[2]; k < DMnz+nloc[2]; k++) {
        for (j = nloc[1]; j < DMny+nloc[1]; j++) {
            for (i = nloc[0]; i < DMnx+nloc[0]; i++) {
                // (i, j, k) is the coordinates of FD nodes in process w.r.t. PR domain
                // initialize t0 
                for (kk = -nloc[2]; kk <= nloc[2]; kk++) {
                    for (jj = -nloc[1]; jj <= nloc[1]; jj++) {
                        for (ii = -nloc[0]; ii <= nloc[0]; ii++) {
                            t0[kk + nloc[2]][jj + nloc[1]][ii + nloc[0]] = 0.0;
                            if (ii == 0 && jj == 0 && kk == 0)
                                t0[kk + nloc[2]][jj + nloc[1]][ii + nloc[0]] = 1.0;
                        }
                    }
                }
                LanczosAlgorithm_gauss(pSPARC, t0, i, j, k, &lambda_min, &lambda_max, nd);
                pSQ->mineig[nd]  = lambda_min;
                pSQ->maxeig[nd]  = lambda_max;
                nd++;
            }
        }
    }

    // This barrier fixed the severe slowdown of MPI communication on hive
    MPI_Barrier(pSQ->dmcomm_SQ);
    
    #ifdef DEBUG
        time1 = MPI_Wtime();
        if(!rank) printf("Rank %d finished Lanczos taking %.3f ms\n",rank, (time1-time2)*1e3); 
    #endif

    // Find Fermi energy (Efermi)
    // For the first SCF without Efermi, communicate across procs to find global  
    // lambda_min and lambda_max to give as input guesses for Brent's algorithm
    // TODO: ADD Newton Raphson method in SQDFT here. 
    if (SCFCount == 0) {
        for (nd = 0; nd < DMnd; nd ++) {
            if (nd == 0) {
                lambda_min_MIN = pSQ->mineig[nd];
                lambda_max_MAX = pSQ->maxeig[nd];
            } else {
                if (pSQ->mineig[nd] < lambda_min_MIN)
                    lambda_min_MIN = pSQ->mineig[nd];
                if (pSQ->maxeig[nd] > lambda_max_MAX)
                    lambda_max_MAX = pSQ->maxeig[nd];
            }
        }
        
        MPI_Allreduce(&lambda_max_MAX, &lambda_max, 1, MPI_DOUBLE, MPI_MAX, pSQ->dmcomm_SQ);
        MPI_Allreduce(&lambda_min_MIN, &lambda_min, 1, MPI_DOUBLE, MPI_MIN, pSQ->dmcomm_SQ);
    }

    x1 = (SCFCount > 0) ? pSPARC->Efermi - 2.0 : min(lambda_min - 1, -6.907755278982137*(1/pSPARC->Beta));
    x2 = (SCFCount > 0) ? pSPARC->Efermi + 2.0 : lambda_max + 1.0;
    pSPARC->Efermi = Calculate_FermiLevel(pSPARC, x1, x2, 1e-12, 100, occ_constraint_SQ_gauss); 

    #ifdef DEBUG
    time2 = MPI_Wtime();
        if(!rank) printf("Rank %d Wating and finding fermi level takes %.3f ms\n",rank, (time2-time1)*1e3); 
    #endif

    // Calculate Electron Density
    double rho_pc, *rho;
    rho = (double *) calloc(sizeof(double), DMnd);

    for (i = 0; i < DMnd; i++) {
        rho_pc = 0.0;
        for (j = 0; j < pSPARC->SQ_npl_g; j++) {
            // rho_pc += FermiDirac(pSQ->gnd[i][j], pSQ->lambda_f, pSPARC->Beta) * pSQ->gwt[i][j];
            rho_pc += pSQ->gwt[i][j] * smearing_function(
                        pSPARC->Beta, pSQ->gnd[i][j], pSPARC->Efermi, pSPARC->elec_T_type);
        }
        rho[i] = 2 * rho_pc / pSPARC->dV;
    }

    #ifdef DEBUG
    time1 = MPI_Wtime();
        if(!rank) printf("Rank %d, calculating electron density takes %.3f ms\n",rank, (time1-time2)*1e3); 
    #endif

    // Transfer rho into phi domain
    TransferDensity_sq2phi(pSPARC, rho, pSPARC->electronDens);

    free(rho);

    for (k = 0; k < 2*nloc[2]+1; k++) {
        for (j = 0; j < 2*nloc[1]+1; j++) {
            free(t0[k][j]);
        }
        free(t0[k]);
    }
    free(t0);
}

/**
 * @brief   Compute nodal Hamiltonian times a vector
 * 
 * @param vec   The 3-d vector multiplied by nodal hamiltonian. Vector size is the nodal Rcut domain
 * @param i     The corresponding x - coordinates of the node w.r.t. PR domain
 * @param j     The corresponding y - coordinates of the node w.r.t. PR domain
 * @param k     The corresponding z - coordinates of the node w.r.t. PR domain
 * @param Hv    The 3-d vector stores the result of Hx
 * Note         This is directly copied from SQDFT. Make it easier to read in the future.
 */
void HsubTimesVec(SPARC_OBJ *pSPARC, double ***vec, int i, int j, int k, double ***Hv) {
    int rank, ii, jj, kk, iii, jjj, kkk, a, p, q, r, ind1, ind2, INDX;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int *nloc = pSQ->nloc;
    int FDn = pSPARC->order / 2;
    
    int tl[3] = {2 * nloc[0] + 1, 2 * nloc[1] + 1, 2 * nloc[2] + 1};
    int tnd = tl[0] * tl[1] * tl[2];

    int tl_ex = tl[0] + 2*FDn;
    int tl_ex2 = (tl[0] + 2*FDn) * (tl[1] + 2*FDn);
    int tnd_ex = tl_ex * (tl[1] + 2*FDn) * (tl[2] + 2*FDn);
    
    double *vec_lap, *vec_nl, temp;

    vec_lap = (double*) calloc(tnd_ex, sizeof(double));
    vec_nl = (double*) calloc(tnd, sizeof(double)); 

    int count = 0;
    for (kk = 0; kk < tl[2]; kk++) {
        r = kk + FDn;
        ind1 = r * tl_ex2;
        for (jj = 0; jj < tl[1]; jj++) {
            q = jj + FDn;
            ind2 = q * tl_ex + ind1;
            for (ii = 0; ii < tl[0]; ii++) {
                p = ii + FDn;
                temp = vec[kk][jj][ii];
                vec_lap[p+ind2] = temp;
                vec_nl[count] = temp;
                count++;
            }
        }
    }
    
    double *VnlVec;
    VnlVec = (double*) calloc(tnd, sizeof(double));
    // (i, j, k) is w.r.t. PR domain
    int DMnx = pSQ->DMVertices_SQ[1] - pSQ->DMVertices_SQ[0] + 1;
    int DMny = pSQ->DMVertices_SQ[3] - pSQ->DMVertices_SQ[2] + 1;
    int nd = (i-nloc[0]) + (j-nloc[1])*DMnx + (k-nloc[2])*DMny*DMnx;
    Vnl_vec_mult_SQ(pSPARC, pSQ->Nd_loc, pSPARC->Atom_Influence_nloc_SQ[nd], 
                  pSPARC->nlocProj_SQ[nd], vec_nl, VnlVec);

    count = 0;
    int kshift = k - nloc[2];
    int jshift = j - nloc[1];
    int ishift = i - nloc[0];
    
    double *w2_x, *w2_y, *w2_z;
    w2_x = pSPARC->D2_stencil_coeffs_x;
    w2_y = pSPARC->D2_stencil_coeffs_y;
    w2_z = pSPARC->D2_stencil_coeffs_z;

    double coeff0 = w2_x[0] + w2_y[0] + w2_z[0];

    int *shifty, *shiftz;
    shifty = (int*) calloc(sizeof(int), FDn+1);
    shiftz = (int*) calloc(sizeof(int), FDn+1);

    shifty[0] = shiftz[0] = 0;
    for (p = 1; p <= FDn; p++) {
        shifty[p] = p * tl_ex;
        shiftz[p] = shifty[p] * (tl[1] + 2*FDn);
    }

    for (kk = 0; kk < tl[2]; kk++) {
        r = kk + FDn;
        kkk = kk + kshift;
        ind1 = r * tl_ex2;
        for (jj = 0; jj < tl[1]; jj++) {
            q = jj + FDn;
            jjj = jj + jshift;
            ind2 = q * tl_ex + ind1;
            #pragma omp simd
            for (ii = 0; ii < tl[0]; ii++) {
                p = ii + FDn;
                iii = ii + ishift;
                INDX = p+ind2;
                temp = vec_lap[INDX] * coeff0;
                for (a = 1; a <= FDn; a++) {
                    temp += (vec_lap[INDX - a] + vec_lap[INDX + a]) * w2_x[a]
                           +(vec_lap[INDX - shifty[a]] + vec_lap[INDX + shifty[a]]) * w2_y[a]
                           +(vec_lap[INDX - shiftz[a]] + vec_lap[INDX + shiftz[a]]) * w2_z[a];
                }

                Hv[kk][jj][ii] = -0.5 * temp + pSQ->Veff_PR[kkk][jjj][iii] * vec_nl[count] + VnlVec[count]; //-0.5Lapl+Veff+Vnloc
                count++;
            }
        }
    }

    // de-allocate memory
    free(vec_lap);
    free(vec_nl);
    free(VnlVec);
    free(shifty);
    free(shiftz);
}

/**
 * @brief   Lanczos algorithm for Gauss method
 * 
 * @param vkm1  Initial guess for Lanczos algorithm
 * @param i     The corresponding x - coordinates of the node w.r.t. PR domain
 * @param j     The corresponding y - coordinates of the node w.r.t. PR domain
 * @param k     The corresponding z - coordinates of the node w.r.t. PR domain
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 * @param nd            FD node index in current process domain
 */
void LanczosAlgorithm_gauss(SPARC_OBJ *pSPARC, double ***vkm1, int i, int j, int k, double *lambda_min, double *lambda_max, int nd) {
    int rank, ii, jj, kk;
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);

    SQ_OBJ* pSQ  = pSPARC->pSQ;    

    int *nloc = pSQ->nloc;
    double ***vk, ***vkp1, val, *aa, *bb;

    int nn = (2*nloc[0]+1)*(2*nloc[1]+1)*(2*nloc[2]+1);
    aa = (double *) calloc(sizeof(double), nn);
    bb = (double *) calloc(sizeof(double), nn);

    vk = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
    vkp1 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        vk[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
        vkp1[kk] =  (double **) calloc(sizeof(double *), 2*nloc[1]+1);
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            vk[kk][jj] = (double *) calloc(sizeof(double ), 2*nloc[0]+1);
            vkp1[kk][jj] = (double *) calloc(sizeof(double ), 2*nloc[0]+1);
        }
    }

    Vector2Norm_local(vkm1, nloc, & val);

    int le_count = 0;
    double *lanczos_vec = (pSPARC->SQ_typ_dm == 2 && pSPARC->SQ_gauss_mem == 1) ? pSQ->lanczos_vec_all[nd] : pSQ->lanczos_vec;

    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            for (ii = 0; ii < 2 * nloc[0] + 1; ii++) {
                vkm1[kk][jj][ii] = vkm1[kk][jj][ii] / val;
                if (pSPARC->SQ_typ_dm == 2) {
                    lanczos_vec[le_count ++] = vkm1[kk][jj][ii];
                }
            }
        }
    }
    // vk=Hsub*vkm1. Here vkm1 is the vector and i,j,k are the indices of node in 
    // proc domain to which the Hsub corresponds to and the indices are w.r.t PR domain
    HsubTimesVec(pSPARC, vkm1, i, j, k, vk); 

    Vector2Norm_local(vk, nloc, & val);

    VectorDotProduct_local(vkm1, vk, nloc, & val);

    aa[0] = val;
    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            for (ii = 0; ii < 2 * nloc[0] + 1; ii++) {
                vk[kk][jj][ii] = vk[kk][jj][ii] - aa[0] * vkm1[kk][jj][ii];
            }
        }
    }
    Vector2Norm_local(vk, nloc, & val);

    bb[0] = val;

    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            for (ii = 0; ii < 2 * nloc[0] + 1; ii++) {
                vk[kk][jj][ii] = vk[kk][jj][ii] / bb[0];
                if (pSPARC->SQ_typ_dm == 2) {
                    lanczos_vec[le_count ++] = vk[kk][jj][ii];
                }
                
            }
        }
    }

    int count = 0;
    //double dl, dm, lmin_prev = 0.0, lmax_prev = 0.0;
    while (count < pSPARC->SQ_npl_g - 1) {
        HsubTimesVec(pSPARC, vk, i, j, k, vkp1); // vkp1=Hsub*vk

        VectorDotProduct_local(vk, vkp1, nloc, & val); // val=vk'*vkp1
        aa[count + 1] = val;
        for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
            for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
                for (ii = 0; ii < 2 * nloc[0] + 1; ii++) {
                    vkp1[kk][jj][ii] = vkp1[kk][jj][ii] - aa[count + 1] * vk[kk][jj][ii] - bb[count] * vkm1[kk][jj][ii];
                }
            }
        }
        Vector2Norm_local(vkp1, nloc, & val);
        bb[count + 1] = val;
        for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
            for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
                for (ii = 0; ii < 2 * nloc[0] + 1; ii++) {
                    vkm1[kk][jj][ii] = vk[kk][jj][ii];
                    vk[kk][jj][ii] = vkp1[kk][jj][ii] / bb[count + 1];
                    
                    if (count != pSPARC->SQ_npl_g - 2) {
                        if (pSPARC->SQ_typ_dm == 2) {
                            lanczos_vec[le_count ++] = vk[kk][jj][ii];
                        }
                    }
                }
            }
        }
        count = count + 1;
    }

    TridiagEigenSolve_gauss(pSPARC, aa, bb, nd, lambda_min , lambda_max);

    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            free(vk[kk][jj]);
            free(vkp1[kk][jj]);
        }
        free(vk[kk]);
        free(vkp1[kk]);
    }
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
void TridiagEigenSolve_gauss(SPARC_OBJ *pSPARC, double *diag, double *subdiag, int nd, double *lambda_min, double *lambda_max) {
    int m, l, iter, i, j, k, n;
    double s, r, p, g, f, dd, c, b, *d, *e, **z;
    SQ_OBJ* pSQ  = pSPARC->pSQ;

    n = pSPARC->SQ_npl_g;
    d = (double *) calloc(sizeof(double), n);
    e = (double *) calloc(sizeof(double), n);
    z = (double **) calloc(sizeof(double*), n);
    for (i = 0; i < n; i++)
        z[i] = (double *) calloc(sizeof(double), n);
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
        pSQ->gnd[nd][i] = d[i] ;
        pSQ->gwt[nd][i] = z[0][i] * z[0][i];
    }

    double *w = (pSPARC->SQ_typ_dm == 2 && pSPARC->SQ_gauss_mem == 1) ? pSQ->w_all[nd] : pSQ->w;
    if (pSPARC->SQ_typ_dm == 2) {
        int count = 0;
        // Save all eigenvectors w 
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                w[count ++] = z[j][i];
            }
        }
    }

    *lambda_min = d[0]; *lambda_max = d[0];

    for (i = 1; i < n; i++) {
        if (d[i] > * lambda_max) { 
            *lambda_max = d[i];
        } else if (d[i] < * lambda_min) { 
            * lambda_min = d[i];
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
 * @brief   Compute chebyshev coefficients Ci (length npl_c+1) to fit the function "func"
 * 
 * NOTE:    THIS FUNCTION CURRENTLY USES DISCRETE ORTHOGONALITY PROPERTY OF CHEBYSHEV POLYNOMIALS WHICH ONLY GIVE "approximate" 
 *          COEFFICIENTS. SO IF THE FUNCTION IS NOT SMOOTH ENOUGH THE COEFFICIENTS WILL NOT BE ACCURATE ENOUGH. 
 *          SO THIS FUNCTION HAS TO BE REPLACED WITH ITS CONTINUOUS COUNTER PART WHICH EVALUATES OSCILLATORY INTEGRALS AND USES FFT.
 */
void ChebyshevCoeff_SQ(int npl_c, double *Ci, double *d, double (*fun)(double, double, double, int), 
                const double beta, const double lambda_f, const int type) {
    int k, j;
    double y, fac1, fac2, sum;

    fac1 = M_PI/(npl_c + 1);
    for (k = 0; k < npl_c + 1; k++) {
        y = cos(fac1 * (k + 0.5));
        // d[k] = (*func)(y, lambda_fp, bet);
        d[k] = fun(beta, y, lambda_f, type);
    }

    fac2 = 2.0 / (npl_c + 1);
    for (j = 0; j < npl_c + 1; j++) {
        sum = 0.0;
        for (k = 0; k < npl_c + 1; k++) {
            // sum = sum + d[k] * cos((M_PI * (j - 1 + 1)) * ((double)((k - 0.5 + 1) / (npl_c + 1))));
            sum = sum + d[k] * cos(fac1 * j * (k + 0.5));
        }
        Ci[j] = fac2 * sum;
    }
    Ci[0] = Ci[0] / 2.0;
}

/**
 * @brief   Lanczos algorithm computing only minimal eigenvales
 * 
 * TODO:    There are 2 different Lanczos algorithm with confusing names. Change them or combine them
 */
void LanczosAlgorithm_new(SPARC_OBJ *pSPARC, int i, int j, int k, double *lambda_min, int nd, int choice) {
    int rank, ii, jj, kk, ll, p, max_iter = 500;
    int *nloc, flag = 0, count;
    double ***vk, ***vkp1, val, *aa, *bb, **lanc_V, lambda_min_temp;
    double lmin_prev = 0.0, dl = 1.0, dm=1.0 ,lmax_prev=0.0;
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    SQ_OBJ *pSQ = pSPARC->pSQ;
    nloc = pSQ->nloc;


    aa = (double *) calloc(sizeof(double), pSQ->Nd_loc);
    bb = (double *) calloc(sizeof(double), pSQ->Nd_loc);

    vk = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
    vkp1 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        vk[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
        vkp1[kk] =  (double **) calloc(sizeof(double *), 2*nloc[1]+1);
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            vk[kk][jj] = (double *) calloc(sizeof(double ), 2*nloc[0]+1);
            vkp1[kk][jj] = (double *) calloc(sizeof(double ), 2*nloc[0]+1);
        }
    }
    
    lanc_V = (double **) calloc(sizeof(double *), max_iter+1);
    for (p = 0; p < max_iter + 1; p++)
        lanc_V[p] = (double *) calloc(sizeof(double), pSQ->Nd_loc);

    Vector2Norm_local(pSQ->vec[nd], nloc, &val);

    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                pSQ->vec[nd][kk][jj][ii] = pSQ->vec[nd][kk][jj][ii] / val;
            }
        }
    }

    HsubTimesVec(pSPARC, pSQ->vec[nd], i, j, k, vk); 
    VectorDotProduct_local(pSQ->vec[nd], vk, nloc, &val);

    aa[0] = val;
    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                vk[kk][jj][ii] = vk[kk][jj][ii] - aa[0] * pSQ->vec[nd][kk][jj][ii];
            }
        }
    }
    Vector2Norm_local(vk, nloc, &val);
    bb[0] = val;
    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                vk[kk][jj][ii] = vk[kk][jj][ii] / bb[0];
                lanc_V[flag][kk * (2 * nloc[0] + 1) * (2 * nloc[1] + 1) + jj * (2 * nloc[0] + 1) + ii] = vk[kk][jj][ii];
            }
        }
    }

    flag += 1;
    count = 0;
    while ((count < max_iter && dl > pSPARC->TOL_LANCZOS) || (choice == 1 && dm > pSPARC->TOL_LANCZOS)) {
        HsubTimesVec(pSPARC, vk, i, j, k, vkp1); // vkp1=Hsub*vk
        VectorDotProduct_local(vk, vkp1, nloc, &val); // val=vk'*vkp1
        aa[count + 1] = val;
        for (kk = 0; kk <= 2 * nloc[2]; kk++) {
            for (jj = 0; jj <= 2 * nloc[1]; jj++) {
                for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                    vkp1[kk][jj][ii] = vkp1[kk][jj][ii] - aa[count + 1] * vk[kk][jj][ii] - bb[count] * pSQ->vec[nd][kk][jj][ii];
                }
            }
        }
        Vector2Norm_local(vkp1, nloc, &val);
        bb[count + 1] = val;

        for (kk = 0; kk <= 2 * nloc[2]; kk++) {
            for (jj = 0; jj <= 2 * nloc[1]; jj++) {
                for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                    pSQ->vec[nd][kk][jj][ii] = vk[kk][jj][ii];
                    vk[kk][jj][ii] = vkp1[kk][jj][ii] / bb[count + 1];
                    lanc_V[flag][kk * (2 * nloc[0] + 1) * (2 * nloc[1] + 1) + jj * (2 * nloc[0] + 1) + ii] = vk[kk][jj][ii];
                }
            }
        }
        flag += 1;
        // Eigendecompose the tridiagonal matrix
        TridiagEigenSolve_new(pSPARC, aa, bb, count + 2, lambda_min, choice, nd);
        
        dl = fabs(( * lambda_min) - lmin_prev);
        dm = fabs((pSQ->lambda_max[nd])-lmax_prev);
        lmin_prev = * lambda_min;
        lmax_prev = pSQ->lambda_max[nd];
        count = count + 1;
    }
    lambda_min_temp = *lambda_min; *lambda_min -= pSPARC->TOL_LANCZOS;
    if (choice == 1) {
        pSQ->lambda_max[nd] += pSPARC->TOL_LANCZOS;
        // free(pSQ->low_eig_vec);
    }
    // if(choice == 0)
    //     free(pSQ->low_eig_vec);
    
    // eigenvector corresponding to lowest eigenvalue of the tridiagonal matrix of the current node
    pSQ->low_eig_vec = (double *) calloc(sizeof(double), count + 1);
    TridiagEigenSolve_new(pSPARC, aa, bb, count + 1, &lambda_min_temp, 2, nd);
    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++)
                pSQ->vec[nd][kk][jj][ii] = 0.0;
        }
    }

    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                for (ll = 0; ll < count + 1; ll++)
                    pSQ->vec[nd][kk][jj][ii] += lanc_V[ll][kk * (2 * nloc[0] + 1) * (2 * nloc[1] + 1) + jj * (2 * nloc[0] + 1) + ii] * pSQ->low_eig_vec[ll];
            }
        }
    }

    free(pSQ->low_eig_vec);

#ifdef DEBUG
    if (!rank && !nd) {
        printf("\nrank %d, nd %d, Lanczos took %d iterations. \n", rank, nd, count);
    }
#endif

    if (count == max_iter)
        printf("WARNING: Lanczos exceeded max_iter. count=%d, dl=%f \n", count, dl);
    
    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            free(vk[kk][jj]);
            free(vkp1[kk][jj]);
        }
        free(vk[kk]);
        free(vkp1[kk]);
    }
    free(vk);
    free(vkp1);
    free(aa);
    free(bb);
    for (p = 0; p < max_iter + 1; p++)
        free(lanc_V[p]);
    free(lanc_V);
}

/**
 * @brief   Tridiagonal solver for eigenvalues. Part of Lanczos algorithm.
 */
void TridiagEigenSolve_new(SPARC_OBJ *pSPARC, double *diag, double *subdiag, 
                            int n, double *lambda_min, int choice, int nd) {

    int i, k, m, l, iter, index;
    double s, r, p, g, f, dd, c, b;
    // d has diagonal and e has subdiagonal
    double *d, *e, **z; 
    SQ_OBJ* pSQ  = pSPARC->pSQ;

    d = (double *) calloc(sizeof(double), n);
    e = (double *) calloc(sizeof(double), n);
    z = (double **) calloc(sizeof(double *), n);

    for (i = 0; i < n; i++)
        z[i] = (double *) calloc(sizeof(double), n);

    if (choice == 2) {
        for (i = 0; i < n; i++) {
            z[i][i] = 1.0;
        }
    }

    //create copy of diag and subdiag in d and e
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
                    if (choice == 2) {
                        for (k = 0; k < n; k++) {
                            f = z[k][i + 1];
                            z[k][i + 1] = s * z[k][i] + c * f;
                            z[k][i] = c * z[k][i] - s * f;
                        }
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }

    // go over the array d to find the smallest and largest eigenvalue
    *lambda_min = d[0];
    if (choice == 1)
        pSQ->lambda_max[nd] = d[0];
    index = 0;
    for (i = 1; i < n; i++) {
        if (choice == 1) {
            if (d[i] > pSQ->lambda_max[nd]) {
                pSQ->lambda_max[nd] = d[i];
            }
        }
        if (d[i] < *lambda_min) { 
            *lambda_min = d[i];
            index = i;
        }
    }

    if (choice == 2) {
        for (i = 0; i < n; i++)
            pSQ->low_eig_vec[i] = z[i][index];
    }

    free(d);
    free(e);
    for (i = 0; i < n; i++)
        free(z[i]);
    free(z);

    return;
}