/**
 * @file    spinOrbitCoupling.c
 * @brief   This file contains functions for spin-orbit coupling (SOC)
 *          calculation.  
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
/* BLAS routines */
#ifdef USE_MKL
    #include <mkl.h> // for cblas_* functions
#else
    #include <cblas.h>
#endif
#include <assert.h>

#include "spinOrbitCoupling.h"
#include "nlocVecRoutines.h"
#include "tools.h"
#include "isddft.h"
#include "initialization.h"
#include "gradVecRoutinesKpt.h"
#include "stress.h"
#include "cyclix_tools.h"

#define TEMP_TOL 1e-12

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)>(b)?(b):(a))



/**
 * @brief   Calculate nonlocal spin-orbit (SO) projectors. 
 */
void CalculateNonlocalProjectors_SOC(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ *nlocProj, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, int *DMVertices, MPI_Comm comm)
{
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double t1, t2, t_tot;
    t_tot = 0.0;
#ifdef DEBUG
    if (rank == 0) printf("Calculate nonlocal spin-orbit (SO) projectors ... \n");
#endif    
    int l, np, lcount, lcount2, m, psd_len, col_count, indx, ityp, iat, ipos, ndc, lloc, lmax, pspsoc;
    int DMnx, DMny, i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *UdV_sort, x2, y2, z2, x, y, z, *Intgwt = NULL;
    double _Complex *Ylm;
    double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;

    if (pSPARC->CyclixFlag) {
        if(comm == pSPARC->kptcomm_topo){
            Intgwt = pSPARC->Intgwt_kpttopo;
        } else{
            Intgwt = pSPARC->Intgwt_psi;
        }
    }

    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;
    // DMnz = DMVertices[5] - DMVertices[4] + 1;  

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
        pspsoc = pSPARC->psd[ityp].pspsoc;
        if (!pspsoc) {
            nlocProj[ityp].nprojso = 0; 
            continue; 
        }
        // allocate memory for projectors
        nlocProj[ityp].Chiso = (double _Complex **)malloc( sizeof(double _Complex *) * Atom_Influence_nloc[ityp].n_atom);
        if (pSPARC->CyclixFlag) {
            nlocProj[ityp].Chiso_cyclix = (double _Complex **)malloc( sizeof(double _Complex *) * Atom_Influence_nloc[ityp].n_atom);
        }
        lloc = pSPARC->localPsd[ityp]; // local projector index
        lmax = pSPARC->psd[ityp].lmax;
        psd_len = pSPARC->psd[ityp].size;
        // number of projectors per atom
        nlocProj[ityp].nprojso = 0;
        for (l = 1; l <= lmax; l++) {
            if (l == lloc) continue;
            nlocProj[ityp].nprojso += pSPARC->psd[ityp].ppl_soc[l-1] * (2 * l + 1);
        }
        if (! nlocProj[ityp].nprojso) continue;
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            // store coordinates of the overlapping atom
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            // grid nodes in (spherical) rc-domain
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            nlocProj[ityp].Chiso[iat] = (double _Complex *)malloc( sizeof(double _Complex) * ndc * nlocProj[ityp].nprojso);
            if (pSPARC->CyclixFlag) {
                nlocProj[ityp].Chiso_cyclix[iat] = (double _Complex *)malloc( sizeof(double _Complex) * ndc * nlocProj[ityp].nprojso);
            }
            rc_pos_x = (double *)malloc( sizeof(double) * ndc );
            rc_pos_y = (double *)malloc( sizeof(double) * ndc );
            rc_pos_z = (double *)malloc( sizeof(double) * ndc );
            rc_pos_r = (double *)malloc( sizeof(double) * ndc );
            Ylm = (double _Complex *)malloc( sizeof(double _Complex) * ndc );
            UdV_sort = (double *)malloc( sizeof(double) * ndc );
            // use spline to fit UdV
            if(pSPARC->cell_typ == 0){
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_nloc[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x2 = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y2 = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z2 = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    rc_pos_x[ipos] = x2;
                    rc_pos_y[ipos] = y2;
                    rc_pos_z[ipos] = z2;
                    x2 *= x2; y2 *= y2; z2 *= z2;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            } else if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_nloc[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    nonCart2Cart_coord(pSPARC, &x, &y, &z);
                    rc_pos_x[ipos] = x;
                    rc_pos_y[ipos] = y;
                    rc_pos_z[ipos] = z;
                    x2 = x * x; y2 = y * y; z2 = z*z;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            } else if(pSPARC->cell_typ > 20 && pSPARC->cell_typ < 30) {
                double y0 = pSPARC->atom_pos[3*Atom_Influence_nloc[ityp].atom_index[iat]+1];
                double z0 = pSPARC->atom_pos[3*Atom_Influence_nloc[ityp].atom_index[iat]+2];
                double ty = (y0 - y0_i)/pSPARC->range_y;
                double tz = (z0 - z0_i)/pSPARC->range_z;                
                RotMat_cyclix(pSPARC, ty, tz);                
                double xi, yi, zi;
                xi = x0_i; yi = y0_i; zi = z0_i;
                nonCart2Cart_coord(pSPARC, &xi, &yi, &zi);
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_nloc[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x = xin + i_DM * pSPARC->delta_x;
                    y = (j_DM + DMVertices[2]) * pSPARC->delta_y;
                    z = (k_DM + DMVertices[4]) * pSPARC->delta_z;
                    nonCart2Cart_coord(pSPARC, &x, &y, &z);
                    x -= xi;
                    y -= yi;
                    z -= zi;
                    rc_pos_x[ipos] = pSPARC->RotM_cyclix[0] * x + pSPARC->RotM_cyclix[1] * y;
                    rc_pos_y[ipos] = pSPARC->RotM_cyclix[3] * x + pSPARC->RotM_cyclix[4] * y;
                    rc_pos_z[ipos] = z;
                    x2 = x * x; y2 = y * y; z2 = z * z;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            }

            lcount = lcount2 = col_count = 0;
            // multiply spherical harmonics and UdV
            for (l = 1; l <= lmax; l++) {
                // skip the local projector
                if  (l == lloc) { lcount2 += pSPARC->psd[ityp].ppl_soc[l-1]; continue;}
                for (np = 0; np < pSPARC->psd[ityp].ppl_soc[l-1]; np++) {
                    // find UdV using spline interpolation
                    if (pSPARC->psd[ityp].is_r_uniform == 1) {
						SplineInterpUniform(pSPARC->psd[ityp].RadialGrid, pSPARC->psd[ityp].UdV_soc+lcount2*psd_len, psd_len, 
						                    rc_pos_r, UdV_sort, ndc, pSPARC->psd[ityp].SplineFitUdV_soc+lcount*psd_len);
					} else {
						SplineInterpNonuniform(pSPARC->psd[ityp].RadialGrid, pSPARC->psd[ityp].UdV_soc+lcount2*psd_len, psd_len, 
						                       rc_pos_r, UdV_sort, ndc, pSPARC->psd[ityp].SplineFitUdV_soc+lcount*psd_len); 
					}
                    for (m = -l; m <= l; m++) {
                        t1 = MPI_Wtime();
                        // find spherical harmonics, Ylm
                        ComplexSphericalHarmonic(ndc, rc_pos_x, rc_pos_y, rc_pos_z, rc_pos_r, l, m, Ylm);
                        t2 = MPI_Wtime();
                        t_tot += t2 - t1;
                        // calculate Chi = UdV * Ylm
                        for (ipos = 0; ipos < ndc; ipos++) {
                            nlocProj[ityp].Chiso[iat][col_count*ndc+ipos] = Ylm[ipos] * UdV_sort[ipos];
                        }
                        if (pSPARC->CyclixFlag) {
                            for (ipos = 0; ipos < ndc; ipos++) {
                                indx = Atom_Influence_nloc[ityp].grid_pos[iat][ipos];
                                nlocProj[ityp].Chiso_cyclix[iat][col_count*ndc+ipos] = Ylm[ipos] * UdV_sort[ipos] * Intgwt[indx];
                            }
                        }
                        col_count++;
                    }
                    lcount++; lcount2++;
                }
            }

            free(rc_pos_x);
            free(rc_pos_y);
            free(rc_pos_z);
            free(rc_pos_r);
            free(Ylm);
            free(UdV_sort);
        }
    }
    
#ifdef DEBUG    
    if(!rank) printf(BLU"rank = %d, Time for complex spherical harmonics: %.3f ms\n"RESET, rank, t_tot*1e3);
#endif    
}


/**
 * @brief   Extract 3 different components from Chiso for further calculation
 */
void CreateChiSOMatrix(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ *nlocProj, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, MPI_Comm comm)
{
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
    if (rank == 0) printf("Extracting nonlocal spin-orbit (SO) projectors for 2 terms... \n");
#endif    
    int l, np, m, ityp, iat, ipos, ndc, lloc, lmax, pspsoc, count1, count2;    

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
        pspsoc = pSPARC->psd[ityp].pspsoc;
        nlocProj[ityp].nprojso_ext = 0;
        if (!pspsoc) continue; 
        // allocate memory for projectors
        nlocProj[ityp].Chisowt0 = (double _Complex **)malloc( sizeof(double _Complex *) * Atom_Influence_nloc[ityp].n_atom);
        nlocProj[ityp].Chisowtl = (double _Complex **)malloc( sizeof(double _Complex *) * Atom_Influence_nloc[ityp].n_atom);
        nlocProj[ityp].Chisowtnl = (double _Complex **)malloc( sizeof(double _Complex *) * Atom_Influence_nloc[ityp].n_atom);
        if (pSPARC->CyclixFlag) {
            nlocProj[ityp].Chisowt0_cyclix = (double _Complex **)malloc( sizeof(double _Complex *) * Atom_Influence_nloc[ityp].n_atom);
            nlocProj[ityp].Chisowtl_cyclix = (double _Complex **)malloc( sizeof(double _Complex *) * Atom_Influence_nloc[ityp].n_atom);
            nlocProj[ityp].Chisowtnl_cyclix = (double _Complex **)malloc( sizeof(double _Complex *) * Atom_Influence_nloc[ityp].n_atom);
        }
        lloc = pSPARC->localPsd[ityp]; // local projector index
        lmax = pSPARC->psd[ityp].lmax;      
        
        for (l = 1; l <= lmax; l++) {
            if (l == lloc) continue;
            nlocProj[ityp].nprojso_ext += pSPARC->psd[ityp].ppl_soc[l-1] * (2 * l);
        }
        // //////////////////////////////////////////////////////
        // printf("nprojso_ext %d\n", nlocProj[ityp].nprojso_ext);
        if (! nlocProj[ityp].nprojso_ext) continue;

        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            // grid nodes in (spherical) rc-domain
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            nlocProj[ityp].Chisowt0[iat] = (double _Complex *)malloc( sizeof(double _Complex) * ndc * nlocProj[ityp].nprojso_ext);
            nlocProj[ityp].Chisowtl[iat] = (double _Complex *)malloc( sizeof(double _Complex) * ndc * nlocProj[ityp].nprojso_ext);
            nlocProj[ityp].Chisowtnl[iat] = (double _Complex *)malloc( sizeof(double _Complex) * ndc * nlocProj[ityp].nprojso_ext);
            if (pSPARC->CyclixFlag) {
                nlocProj[ityp].Chisowt0_cyclix[iat] = (double _Complex *)malloc( sizeof(double _Complex) * ndc * nlocProj[ityp].nprojso_ext);
                nlocProj[ityp].Chisowtl_cyclix[iat] = (double _Complex *)malloc( sizeof(double _Complex) * ndc * nlocProj[ityp].nprojso_ext);
                nlocProj[ityp].Chisowtnl_cyclix[iat] = (double _Complex *)malloc( sizeof(double _Complex) * ndc * nlocProj[ityp].nprojso_ext);
            }

            // extract Chi without m = 0
            count1 = count2 = 0;
            // multiply spherical harmonics and UdV
            for (l = 1; l <= lmax; l++) {
                // skip the local projector
                if  (l == lloc) { continue;}
                for (np = 0; np < pSPARC->psd[ityp].ppl_soc[l-1]; np++) {
                    for (m = -l; m <= l; m++) {
                        if (m == 0) { count1++; continue;}
                        // printf("col indx %d\n",count1+1);
                        for (ipos = 0; ipos < ndc; ipos++) {
                            nlocProj[ityp].Chisowt0[iat][count2*ndc+ipos] = nlocProj[ityp].Chiso[iat][count1*ndc+ipos];
                        }
                        if (pSPARC->CyclixFlag) {
                          for (ipos = 0; ipos < ndc; ipos++) {
                              nlocProj[ityp].Chisowt0_cyclix[iat][count2*ndc+ipos] = nlocProj[ityp].Chiso_cyclix[iat][count1*ndc+ipos];
                          }
                        }
                        count1++; count2++;
                    }
                }
            }

            // extract Chi without m = l
            count1 = count2 = 0;
            // multiply spherical harmonics and UdV
            for (l = 1; l <= lmax; l++) {
                // skip the local projector
                if  (l == lloc) { continue;}
                for (np = 0; np < pSPARC->psd[ityp].ppl_soc[l-1]; np++) {
                    for (m = -l; m <= l; m++) {
                        if (m == l) { count1++; continue;}
                        for (ipos = 0; ipos < ndc; ipos++) {
                            nlocProj[ityp].Chisowtl[iat][count2*ndc+ipos] = nlocProj[ityp].Chiso[iat][count1*ndc+ipos];
                        }
                        if (pSPARC->CyclixFlag) {
                          for (ipos = 0; ipos < ndc; ipos++) {                              
                              nlocProj[ityp].Chisowtl_cyclix[iat][count2*ndc+ipos] = nlocProj[ityp].Chiso_cyclix[iat][count1*ndc+ipos];
                          }
                        }
                        count1++; count2++;
                    }
                }
            }

            // extract Chi without m = -l
            count1 = count2 = 0;
            // multiply spherical harmonics and UdV
            for (l = 1; l <= lmax; l++) {
                // skip the local projector
                if  (l == lloc) { continue;}
                for (np = 0; np < pSPARC->psd[ityp].ppl_soc[l-1]; np++) {
                    for (m = -l; m <= l; m++) {
                        if (m == -l) { count1++; continue;}
                        for (ipos = 0; ipos < ndc; ipos++) {
                            nlocProj[ityp].Chisowtnl[iat][count2*ndc+ipos] = nlocProj[ityp].Chiso[iat][count1*ndc+ipos];
                        }
                        if (pSPARC->CyclixFlag) {
                          for (ipos = 0; ipos < ndc; ipos++) {                              
                              nlocProj[ityp].Chisowtnl_cyclix[iat][count2*ndc+ipos] = nlocProj[ityp].Chiso_cyclix[iat][count1*ndc+ipos];
                          }
                        }
                        count1++; count2++;
                    }
                }
            }
        }
    }
}


/**
 * @brief   Calculate indices for storing nonlocal inner product in an array for SOC. 
 *
 *          We will store the inner product < Chi_Jlm, x_n > in a continuous array "alpha",
 *          the dimensions are in the order: <lm>, n, J. Here we find out the sizes of the 
 *          inner product corresponding to atom J, and the total number of inner products
 *          corresponding to each vector x_n.
 */
void CalculateNonlocalInnerProductIndexSOC(SPARC_OBJ *pSPARC)
{
    int ityp, iat, l, lmax, lloc, atom_index, nproj;
    pSPARC->IP_displ_SOC = (int *)malloc( sizeof(int) * (pSPARC->n_atom+1));
    atom_index = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        lmax = pSPARC->psd[ityp].lmax;
        lloc = pSPARC->localPsd[ityp];
        // number of projectors per atom (of this type)
        nproj = 0;
        for (l = 1; l <= lmax; l++) {
            if (l == lloc) continue;
            nproj += pSPARC->psd[ityp].ppl_soc[l-1] * (2*l);
        }
        pSPARC->IP_displ_SOC[0] = 0;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            pSPARC->IP_displ_SOC[atom_index+1] = pSPARC->IP_displ_SOC[atom_index] + nproj;
            atom_index++;
        }
    }
}


/**
 * @brief   Calculate Vnl SO term 1 times vectors in a matrix-free way with Bloch factor
 * 
 *          0.5*sum_{J,n,lm} m*gamma_{Jln} (sum_{J'} ChiSO_{J'lmn}>)(sum_{J'} <ChiSO_{J'lmn}|x>)
 */
void Vnl_vec_mult_SOC1(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                      NLOC_PROJ_OBJ *nlocProj, int ncol, double _Complex *x, int ldi, double _Complex *Hx, int ldo, int spinor, int kpt, MPI_Comm comm)
{
    int i, n, np, count;
    /* compute nonlocal operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index;
    double x0_i, y0_i, z0_i;
    double _Complex *alpha, *x_rc, *Vnlx;
    alpha = (double _Complex *) calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol, sizeof(double _Complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1 = pSPARC->k1_loc[kpt];
    double k2 = pSPARC->k2_loc[kpt];
    double k3 = pSPARC->k3_loc[kpt];
    double theta;
    double _Complex bloch_fac, a, b;
    double spinorfac;
    
    //first find inner product
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (! nlocProj[ityp].nprojso_ext) continue; 
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            if (pSPARC->CyclixFlag) {
                a = bloch_fac;
            } else {
                a = bloch_fac * pSPARC->dV;
            }
            b = 1.0;
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            x_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    x_rc[n*ndc+i] = x[n*ldi + Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                }
            }
            if (pSPARC->CyclixFlag) {
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nlocProj[ityp].nprojso_ext, ncol, ndc,
                    &a, nlocProj[ityp].Chisowt0_cyclix[iat], ndc, x_rc, ndc, &b,
                    alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext);
            } else {
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nlocProj[ityp].nprojso_ext, ncol, ndc, 
                    &a, nlocProj[ityp].Chisowt0[iat], ndc, x_rc, ndc, &b, 
                    alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext);
            }
            free(x_rc);
        }
    }

    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    }

    spinorfac = (spinor == 0) ? 1.0 : -1.0; 
    // go over all atoms and multiply gamma_Jl to the inner product
    count = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int lloc = pSPARC->localPsd[ityp];
        lmax = pSPARC->psd[ityp].lmax;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            for (n = 0; n < ncol; n++) {
                ldispl = 0;
                for (l = 1; l <= lmax; l++) {
                    // skip the local l
                    if (l == lloc) {
                        ldispl += pSPARC->psd[ityp].ppl_soc[l-1];
                        continue;
                    }
                    for (np = 0; np < pSPARC->psd[ityp].ppl_soc[l-1]; np++) {
                        for (m = -l; m <= l; m++) {
                            if (!m) continue;
                            alpha[count++] *= spinorfac*0.5*m*pSPARC->psd[ityp].Gamma_soc[ldispl+np];
                            // printf("gammaso_Jl %f\n", spinorfac*0.5*m*pSPARC->psd[ityp].Gamma_soc[ldispl+np]);
                        }
                    }
                    ldispl += pSPARC->psd[ityp].ppl_soc[l-1];
                }
            }
        }
    }

    // multiply the inner product and the nonlocal projector
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (! nlocProj[ityp].nprojso_ext) continue; 
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) - sin(theta) * I;
            b = 0.0;
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            Vnlx = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ndc, ncol, nlocProj[ityp].nprojso_ext, &bloch_fac, nlocProj[ityp].Chisowt0[iat], ndc, 
                          alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext, &b, Vnlx, ndc); 
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    Hx[n*ldo + Atom_Influence_nloc[ityp].grid_pos[iat][i]] += Vnlx[n*ndc+i];
                }
            }
            free(Vnlx);
        }
    }
    free(alpha);
}


/**
 * @brief   Calculate Vnl SO term 1 times vectors in a matrix-free way with Bloch factor
 * 
 *          0.5*sum_{J,n,lm} sqrt(l*(l+1)-m*(m+sigma))*gamma_{Jln} *
 *          (sum_{J'} ChiSO_{J'lm+sigma,n}>)(sum_{J'} <ChiSO_{J'lmn}|x_sigma'>)
 */
void Vnl_vec_mult_SOC2(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
        NLOC_PROJ_OBJ *nlocProj, int ncol, double _Complex *xos, int ldi, double _Complex *Hx, int ldo, int spinor, int kpt, MPI_Comm comm)
{
    int i, n, np, count;
    /* compute nonlocal operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index;
    double x0_i, y0_i, z0_i;
    double _Complex *alpha, *x_rc, *Vnlx;
    alpha = (double _Complex *) calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol, sizeof(double _Complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1 = pSPARC->k1_loc[kpt];
    double k2 = pSPARC->k2_loc[kpt];
    double k3 = pSPARC->k3_loc[kpt];
    double theta;
    double _Complex bloch_fac, a, b;
    double _Complex **Chiso, **Chiso_cyclix = NULL;
    
    //first find inner product
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (! nlocProj[ityp].nprojso_ext) continue;
        Chiso = (spinor == 0) ? nlocProj[ityp].Chisowtnl : nlocProj[ityp].Chisowtl; 
        if (pSPARC->CyclixFlag) {
            Chiso_cyclix = (spinor == 0) ? nlocProj[ityp].Chisowtnl_cyclix : nlocProj[ityp].Chisowtl_cyclix;
        }
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            if (pSPARC->CyclixFlag) {
                a = bloch_fac;
            } else {
                a = bloch_fac * pSPARC->dV;
            }
            b = 1.0;
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            x_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    x_rc[n*ndc+i] = xos[n*ldi + Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                }
            }
            if (pSPARC->CyclixFlag) {
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nlocProj[ityp].nprojso_ext, ncol, ndc,
                    &a, Chiso_cyclix[iat], ndc, x_rc, ndc, &b,
                    alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext);
            } else {
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nlocProj[ityp].nprojso_ext, ncol, ndc, 
                    &a, Chiso[iat], ndc, x_rc, ndc, &b, 
                    alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext);
            }
            free(x_rc);
        }
    }

    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    }
    
    // go over all atoms and multiply gamma_Jl to the inner product
    count = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int lloc = pSPARC->localPsd[ityp];
        lmax = pSPARC->psd[ityp].lmax;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            for (n = 0; n < ncol; n++) {
                ldispl = 0;
                for (l = 1; l <= lmax; l++) {
                    // skip the local l
                    if (l == lloc) {
                        ldispl += pSPARC->psd[ityp].ppl_soc[l-1];
                        continue;
                    }
                    for (np = 0; np < pSPARC->psd[ityp].ppl_soc[l-1]; np++) {
                        for (m = -l; m <= l; m++) {
                            if (m == l) continue;
                            // tricks here: sqrt from -l+1 to l is the same as it from -l to l-1
                            alpha[count++] *= 0.5*sqrt(l*(l+1)-m*(m+1))*pSPARC->psd[ityp].Gamma_soc[ldispl+np];
                        }
                    }
                    ldispl += pSPARC->psd[ityp].ppl_soc[l-1];
                }
            }
        }
    }
    
    // multiply the inner product and the nonlocal projector
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (! nlocProj[ityp].nprojso_ext) continue; 
        Chiso = (spinor == 0) ? nlocProj[ityp].Chisowtl : nlocProj[ityp].Chisowtnl; 
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) - sin(theta) * I;
            b = 0.0;
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            Vnlx = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ndc, ncol, nlocProj[ityp].nprojso_ext, &bloch_fac, Chiso[iat], ndc, 
                          alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext, &b, Vnlx, ndc); 
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    Hx[n*ldo + Atom_Influence_nloc[ityp].grid_pos[iat][i]] += Vnlx[n*ndc+i];
                }
            }
            free(Vnlx);
        }
    }
    free(alpha);
}
