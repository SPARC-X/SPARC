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
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *UdV_sort, x2, y2, z2, x, y, z;
    double complex *Ylm;

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
        nlocProj[ityp].Chiso = (double complex **)malloc( sizeof(double complex *) * Atom_Influence_nloc[ityp].n_atom);
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
            nlocProj[ityp].Chiso[iat] = (double complex *)malloc( sizeof(double complex) * ndc * nlocProj[ityp].nprojso);
            rc_pos_x = (double *)malloc( sizeof(double) * ndc );
            rc_pos_y = (double *)malloc( sizeof(double) * ndc );
            rc_pos_z = (double *)malloc( sizeof(double) * ndc );
            rc_pos_r = (double *)malloc( sizeof(double) * ndc );
            Ylm = (double complex *)malloc( sizeof(double complex) * ndc );
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
    
    double t1, t2, t_tot;
    t_tot = 0.0;
#ifdef DEBUG
    if (rank == 0) printf("Extracting nonlocal spin-orbit (SO) projectors for 2 terms... \n");
#endif    
    int l, np, m, psd_len, indx, ityp, iat, ipos, ndc, lloc, lmax, pspsoc, count1, count2;
    int DMnx, DMny, i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *UdV_sort, x2, y2, z2, x, y, z;
    double complex *Ylm;

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
        pspsoc = pSPARC->psd[ityp].pspsoc;
        nlocProj[ityp].nprojso_ext = 0;
        if (!pspsoc) continue; 
        // allocate memory for projectors
        nlocProj[ityp].Chisowt0 = (double complex **)malloc( sizeof(double complex *) * Atom_Influence_nloc[ityp].n_atom);
        nlocProj[ityp].Chisowtl = (double complex **)malloc( sizeof(double complex *) * Atom_Influence_nloc[ityp].n_atom);
        nlocProj[ityp].Chisowtnl = (double complex **)malloc( sizeof(double complex *) * Atom_Influence_nloc[ityp].n_atom);
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
            nlocProj[ityp].Chisowt0[iat] = (double complex *)malloc( sizeof(double complex) * ndc * nlocProj[ityp].nprojso_ext);
            nlocProj[ityp].Chisowtl[iat] = (double complex *)malloc( sizeof(double complex) * ndc * nlocProj[ityp].nprojso_ext);
            nlocProj[ityp].Chisowtnl[iat] = (double complex *)malloc( sizeof(double complex) * ndc * nlocProj[ityp].nprojso_ext);

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
                      NLOC_PROJ_OBJ *nlocProj, int ncol, double complex *x, double complex *Hx, int spinor, int kpt, MPI_Comm comm)
{
    int i, n, np, count;
    /* compute nonlocal operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index;
    double x0_i, y0_i, z0_i;
    double complex *alpha, *x_rc, *Vnlx;
    alpha = (double complex *) calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol, sizeof(double complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1 = pSPARC->k1_loc[kpt];
    double k2 = pSPARC->k2_loc[kpt];
    double k3 = pSPARC->k3_loc[kpt];
    double theta;
    double complex bloch_fac, a, b;
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
            a = bloch_fac * pSPARC->dV;
            b = 1.0;
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            x_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    x_rc[n*ndc+i] = x[n*DMnd + Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                }
            }
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nlocProj[ityp].nprojso_ext, ncol, ndc, 
                &a, nlocProj[ityp].Chisowt0[iat], ndc, x_rc, ndc, &b, 
                alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext);
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
            Vnlx = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ndc, ncol, nlocProj[ityp].nprojso_ext, &bloch_fac, nlocProj[ityp].Chisowt0[iat], ndc, 
                          alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext, &b, Vnlx, ndc); 
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    Hx[n*DMnd + Atom_Influence_nloc[ityp].grid_pos[iat][i]] += Vnlx[n*ndc+i];
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
        NLOC_PROJ_OBJ *nlocProj, int ncol, double complex *xos, double complex *Hx, int spinor, int kpt, MPI_Comm comm)
{
    int i, n, np, count;
    /* compute nonlocal operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index;
    double x0_i, y0_i, z0_i;
    double complex *alpha, *x_rc, *Vnlx;
    alpha = (double complex *) calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol, sizeof(double complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1 = pSPARC->k1_loc[kpt];
    double k2 = pSPARC->k2_loc[kpt];
    double k3 = pSPARC->k3_loc[kpt];
    double theta;
    double complex bloch_fac, a, b;
    double complex **Chiso;
    
    //first find inner product
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (! nlocProj[ityp].nprojso_ext) continue;
        Chiso = (spinor == 0) ? nlocProj[ityp].Chisowtnl : nlocProj[ityp].Chisowtl; 
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            a = bloch_fac * pSPARC->dV;
            b = 1.0;
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            x_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    x_rc[n*ndc+i] = xos[n*DMnd + Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                }
            }
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nlocProj[ityp].nprojso_ext, ncol, ndc, 
                &a, Chiso[iat], ndc, x_rc, ndc, &b, 
                alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext);
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
            Vnlx = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ndc, ncol, nlocProj[ityp].nprojso_ext, &bloch_fac, Chiso[iat], ndc, 
                          alpha+pSPARC->IP_displ_SOC[atom_index]*ncol, nlocProj[ityp].nprojso_ext, &b, Vnlx, ndc); 
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    Hx[n*DMnd + Atom_Influence_nloc[ityp].grid_pos[iat][i]] += Vnlx[n*ndc+i];
                }
            }
            free(Vnlx);
        }
    }
    free(alpha);
}

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////      Forces functions      ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief    Calculate nonlocal force components with kpts in case of spinor wave function.
 */
void Calculate_nonlocal_forces_kpt_spinor_linear(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, k, n, np, ldispl, ndc, ityp, iat, ncol, DMnd, dim, atom_index, count, kpt, Nk;
    int spinor, Nspinor, DMndbyNspinor, size_k, spn_i, nspin, size_s;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    DMndbyNspinor = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    Nspinor = pSPARC->Nspinor;
    size_k = DMnd * ncol;    
    size_s = size_k * Nk;

    double complex *alpha, *alpha_so1, *alpha_so2, *beta;
    double *force_nloc;

    // alpha stores integral in order: Nstate ,image, type, kpt, spin
    alpha = (double complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 4 * Nspinor, sizeof(double complex));
    alpha_so1 = (double complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * 4 * Nspinor, sizeof(double complex));
    alpha_so2 = (double complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * 4 * Nspinor, sizeof(double complex));

    force_nloc = (double *)calloc(3 * pSPARC->n_atom, sizeof(double));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, x0_i, y0_i, z0_i, kpt_vec;
    double complex bloch_fac, a, b;
    
    // Comment: all these changes are made for calculating Dpsi only for one time
#ifdef DEBUG 
    if (!rank) printf("Start Calculating nonlocal forces for spinor wavefunctions...\n");
#endif

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++) {
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SC");
            beta = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SO1");
            beta = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SO2");
            count++;
        }
    }    
    
    /* find inner product <Chi_Jlm, dPsi_n> */
    for (dim = 0; dim < 3; dim++) {
        count = 0;
        for(spn_i = 0; spn_i < nspin; spn_i++) {
            for(kpt = 0; kpt < Nk; kpt++) {
                k1 = pSPARC->k1_loc[kpt];
                k2 = pSPARC->k2_loc[kpt];
                k3 = pSPARC->k3_loc[kpt];
                kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                for (spinor = 0; spinor < Nspinor; spinor++) {
                    // find dPsi in direction dim
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k+spinor*DMndbyNspinor, 
                                            pSPARC->Yorb_kpt+spinor*DMndbyNspinor, dim, kpt_vec, pSPARC->dmcomm);
                }
                beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin * (dim + 1) + count);
                Compute_Integral_Chi_Dpsi_kpt(pSPARC, pSPARC->Yorb_kpt, beta, spn_i, kpt, "SC");
                beta = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin * (dim + 1) + count);
                Compute_Integral_Chi_Dpsi_kpt(pSPARC, pSPARC->Yorb_kpt, beta, spn_i, kpt, "SO1");
                beta = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin * (dim + 1) + count);
                Compute_Integral_Chi_Dpsi_kpt(pSPARC, pSPARC->Yorb_kpt, beta, spn_i, kpt, "SO2");
                count++; 
            }
        }    
    }
        
    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, alpha_so1, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, alpha_so2, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }

    Compute_force_nloc_by_integrals(pSPARC, force_nloc, alpha, "SC");
    Compute_force_nloc_by_integrals(pSPARC, force_nloc, alpha_so1, "SO1");
    Compute_force_nloc_by_integrals(pSPARC, force_nloc, alpha_so2, "SO2");

    free(alpha);
    free(alpha_so1);
    free(alpha_so2);

    // sum over all spin
    if (pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(force_nloc, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
    }

    // sum over all kpoints
    if (pSPARC->npkpt > 1) {
        // The MPI_Reduce solution fails in some cases, switching to Allreduce fixed them
        // if (pSPARC->kptcomm_index == 0)
        //     MPI_Reduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        // else
        //     MPI_Reduce(force_nloc, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(force_nloc, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    }
    
#ifdef DEBUG    
    if (!rank) {
        printf("force_nloc = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", force_nloc[i*3], force_nloc[i*3+1], force_nloc[i*3+2]);
        }
    }    
    if (!rank) {
        printf("force_loc = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", pSPARC->forces[i*3], pSPARC->forces[i*3+1], pSPARC->forces[i*3+2]);
        }
    }
#endif
    
    if (!rank) {
        for (i = 0; i < 3 * pSPARC->n_atom; i++) {
            pSPARC->forces[i] += force_nloc[i];
        }
    }
    
    free(force_nloc);
}


/**
 * @brief   Calculate <Psi_n, Chi_Jlm> for spinor force
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_psi_Chi_kpt(SPARC_OBJ *pSPARC, double complex *beta, int spn_i, int kpt, char *option) 
{
    int i, k, n, np, ldispl, ndc, ityp, iat, ncol, DMnd, dim, atom_index, l, m, lmax, Nk;
    int spinor, Nspinor, DMndbyNspinor, size_k, nspin, size_s, nproj, spinorshift, *IP_displ;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    DMndbyNspinor = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    Nspinor = pSPARC->Nspinor;
    size_k = DMnd * ncol;    
    size_s = size_k * Nk;

    double complex *x_ptr, *dx_ptr, *x_rc, *dx_rc, *x_rc_ptr, *dx_rc_ptr, *beta_x, *beta_y, *beta_z;
    double *force_nloc, fJ_x, fJ_y, fJ_z, val_x, val_y, val_z, val2_x, val2_y, val2_z, g_nk;

    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, x0_i, y0_i, z0_i;
    double complex bloch_fac, a, b, **Chi;

    k1 = pSPARC->k1_loc[kpt];
    k2 = pSPARC->k2_loc[kpt];
    k3 = pSPARC->k3_loc[kpt];

    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        nproj = !strcmpi(option, "SC") ? pSPARC->nlocProj[ityp].nproj : pSPARC->nlocProj[ityp].nprojso_ext;
        if (!strcmpi(option, "SC")) 
            Chi = pSPARC->nlocProj[ityp].Chi_c;
        else if (!strcmpi(option, "SO1")) 
            Chi = pSPARC->nlocProj[ityp].Chisowt0;

        if (! nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) - sin(theta) * I;
            a = bloch_fac * pSPARC->dV;
            b = 1.0;
            ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
            x_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
            /* first find inner product <Psi_n, Chi_Jlm>, here we calculate <Chi_Jlm, Psi_n> instead */
            for (spinor = 0; spinor < Nspinor; spinor++) {
                if (!strcmpi(option, "SO2")) 
                    Chi = (spinor == 0) ? pSPARC->nlocProj[ityp].Chisowtl : pSPARC->nlocProj[ityp].Chisowtnl; 
                for (n = 0; n < ncol; n++) {
                    x_ptr = pSPARC->Xorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd + spinor * DMndbyNspinor;
                    x_rc_ptr = x_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(x_rc_ptr + i) = conj(*(x_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]));
                    }
                }
                spinorshift = IP_displ[pSPARC->n_atom] * ncol * spinor;
                cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nproj, ncol, ndc, &a, Chi[iat], ndc, 
                            x_rc, ndc, &b, beta+spinorshift+IP_displ[atom_index]*ncol, nproj); // multiply dV to get inner-product
            }
            free(x_rc);
        }
    }
}



/**
 * @brief   Calculate <Chi_Jlm, DPsi_n> for spinor force
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_Chi_Dpsi_kpt(SPARC_OBJ *pSPARC, double complex *dpsi, double complex *beta, int spn_i, int kpt, char *option) 
{
    int i, k, n, np, ldispl, ndc, ityp, iat, ncol, DMnd, dim, atom_index, l, m, lmax, Nk;
    int spinor, Nspinor, DMndbyNspinor, size_k, nspin, size_s, spinorshift, *IP_displ, nproj, ispinor;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    DMndbyNspinor = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    Nspinor = pSPARC->Nspinor;
    size_k = DMnd * ncol;    
    size_s = size_k * Nk;

    double complex *x_ptr, *dx_ptr, *x_rc, *dx_rc, *x_rc_ptr, *dx_rc_ptr;
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, x0_i, y0_i, z0_i;
    double complex bloch_fac, a, b, **Chi;

    k1 = pSPARC->k1_loc[kpt];
    k2 = pSPARC->k2_loc[kpt];
    k3 = pSPARC->k3_loc[kpt];

    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        nproj = !strcmpi(option, "SC") ? pSPARC->nlocProj[ityp].nproj : pSPARC->nlocProj[ityp].nprojso_ext;
        if (!strcmpi(option, "SC")) 
            Chi = pSPARC->nlocProj[ityp].Chi_c;
        else if (!strcmpi(option, "SO1")) 
            Chi = pSPARC->nlocProj[ityp].Chisowt0;

        if (! nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            b = 1.0;
            ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat]; 
            dx_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
            for (spinor = 0; spinor < Nspinor; spinor++) {
                if (!strcmpi(option, "SO2")) 
                    Chi = (spinor == 0) ? pSPARC->nlocProj[ityp].Chisowtnl : pSPARC->nlocProj[ityp].Chisowtl; 
                ispinor = !strcmpi(option, "SO2") ? (1 - spinor) : spinor;

                for (n = 0; n < ncol; n++) {                    
                    dx_ptr = dpsi + n * DMnd + ispinor * DMndbyNspinor;
                    dx_rc_ptr = dx_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                    }
                }
                
                /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).
                */      
                spinorshift = IP_displ[pSPARC->n_atom] * ncol * spinor;
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nproj, ncol, ndc, &bloch_fac, Chi[iat], ndc, 
                            dx_rc, ndc, &b, beta+spinorshift+IP_displ[atom_index]*ncol, nproj);   
            }
            free(dx_rc);
        }
    }
}


/**
 * @brief   Compute nonlocal forces using stored integrals
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_force_nloc_by_integrals(SPARC_OBJ *pSPARC, double *force_nloc, double complex *alpha, char *option) 
{
    int i, k, n, np, ldispl, ityp, iat, ncol, atom_index, count, l, m, lmax, Nk;
    int spinor, Nspinor, spn_i, nspin, l_start, mexclude, ppl, *IP_displ;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    Nspinor = pSPARC->Nspinor;

    double complex *beta, *beta_x, *beta_y, *beta_z;
    double fJ_x, fJ_y, fJ_z, val_x, val_y, val_z, val2_x, val2_y, val2_z, g_nk, scaled_gamma_Jl;

    // go over all atoms and find nonlocal force components
    int Ns = pSPARC->Nstates;
    double kpt_spn_fac;

    l_start = !strcmpi(option, "SC") ? 0 : 1;
    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;
    beta_x = alpha + IP_displ[pSPARC->n_atom]*ncol*Nk*nspin*Nspinor;
    beta_y = alpha + IP_displ[pSPARC->n_atom]*ncol*Nk*nspin*Nspinor * 2;
    beta_z = alpha + IP_displ[pSPARC->n_atom]*ncol*Nk*nspin*Nspinor * 3;

    count = 0; 
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            kpt_spn_fac = (2.0/pSPARC->Nspin/Nspinor) * 2.0 * pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
            for (spinor = 0; spinor < Nspinor; spinor++) {
                double spinorfac = (spinor == 0) ? 1.0 : -1.0; 
                atom_index = 0;
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                    int lloc = pSPARC->localPsd[ityp];
                    lmax = pSPARC->psd[ityp].lmax;
                    for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                        fJ_x = fJ_y = fJ_z = 0.0;
                        for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                            g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                            val2_x = val2_y = val2_z = 0.0;
                            ldispl = 0;
                            for (l = l_start; l <= lmax; l++) {
                                mexclude = !strcmpi(option, "SC") ? (l+1) : (!strcmpi(option, "SO1") ? 0 : l);
                                ppl = !strcmpi(option, "SC") ? pSPARC->psd[ityp].ppl[l] : pSPARC->psd[ityp].ppl_soc[l-1];

                                // skip the local l
                                if (l == lloc) {
                                    ldispl += ppl;
                                    continue;
                                }
                                for (np = 0; np < ppl; np++) {
                                    // val_x = val_y = val_z = 0.0;
                                    for (m = -l; m <= l; m++) {
                                        if (m == mexclude) continue;
                                        if (!strcmpi(option, "SC")) scaled_gamma_Jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                        else if (!strcmpi(option, "SO1")) scaled_gamma_Jl = spinorfac*0.5*m*pSPARC->psd[ityp].Gamma_soc[ldispl+np];
                                        else if (!strcmpi(option, "SO2")) scaled_gamma_Jl = 0.5*sqrt(l*(l+1)-m*(m+1))*pSPARC->psd[ityp].Gamma_soc[ldispl+np];

                                        val_x = creal(alpha[count]) * creal(beta_x[count]) - cimag(alpha[count]) * cimag(beta_x[count]);
                                        val_y = creal(alpha[count]) * creal(beta_y[count]) - cimag(alpha[count]) * cimag(beta_y[count]);
                                        val_z = creal(alpha[count]) * creal(beta_z[count]) - cimag(alpha[count]) * cimag(beta_z[count]);
                                        val2_x += scaled_gamma_Jl * val_x;
                                        val2_y += scaled_gamma_Jl * val_y;
                                        val2_z += scaled_gamma_Jl * val_z;

                                        count++;
                                    }
                                }
                                ldispl += ppl;
                            }
                            fJ_x += val2_x * g_nk;
                            fJ_y += val2_y * g_nk;
                            fJ_z += val2_z * g_nk;
                        }
                        force_nloc[atom_index*3  ] -= kpt_spn_fac * fJ_x;
                        force_nloc[atom_index*3+1] -= kpt_spn_fac * fJ_y;
                        force_nloc[atom_index*3+2] -= kpt_spn_fac * fJ_z;
                        atom_index++;
                    }
                }
            }
        }
    }    
}

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////      Stress functions      ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief    Calculate nonlocal + kinetic components of stress.
 */
void Calculate_nonlocal_kinetic_stress_kpt_spinor(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    int i, j, k, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMndbyNspinor, DMnx, DMny, indx, i_DM, j_DM, k_DM;
    int dim, dim2, atom_index, count, count2, l, m, lmax, kpt, Nk, size_k, spn_i, nspin, size_s, spinor, Nspinor, spinorshift;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    DMndbyNspinor = pSPARC->Nd_d_dmcomm;
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor;
    
    double complex *alpha, *alpha_so1, *alpha_so2, *beta, *psi_ptr, *dpsi_ptr, *psi_rc, *psi_rc_ptr, *dpsi_full;
    double complex *beta1_x1, *beta2_x1, *beta2_x2, *beta3_x1, *beta3_x2, *beta3_x3;
    double complex *dpsi_xi, *dpsi_xj, *dpsi_x1, *dpsi_x2, *dpsi_x3, *dpsi_xi_lv, *dpsi_xi_rc, *dpsi_xi_rc_ptr;
    double SJ[6], eJ, temp_k, temp_e, temp_s[6], temp2_e, temp2_s[6], g_nk, gamma_jl, kptwt,  R1, R2, R3, x1_R1, x2_R2, x3_R3;
    double dpsii_dpsij, energy_nl = 0.0, stress_k[6], stress_nl[6], StXmRjp;
    
    for (i = 0; i < 6; i++) SJ[i] = temp_s[i] = temp2_s[i] = stress_nl[i] = stress_k[i] = 0;

    dpsi_full = (double complex *)malloc( 3 * size_s * nspin * sizeof(double complex) );  // dpsi_x, dpsi_y, dpsi_z in cartesian coordinates
    assert(dpsi_full != NULL);
    
    alpha = (double complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 7 * Nspinor, sizeof(double complex));
    alpha_so1 = (double complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * 7 * Nspinor, sizeof(double complex));
    alpha_so2 = (double complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * 7 * Nspinor, sizeof(double complex));
    assert(alpha != NULL && alpha_so1 != NULL && alpha_so2 != NULL);
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, kpt_vec;
    double complex bloch_fac, a, b;
#ifdef DEBUG 
    if (!rank) printf("Start calculating stress contributions from kinetic and nonlocal psp. \n");
#endif

    // find gradient of psi
    if (pSPARC->cell_typ == 0){
        for (dim = 0; dim < 3; dim++) {
            // count = 0;
            for(spn_i = 0; spn_i < nspin; spn_i++) {
                for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                    k1 = pSPARC->k1_loc[kpt];
                    k2 = pSPARC->k2_loc[kpt];
                    k3 = pSPARC->k3_loc[kpt];
                    kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                    for (spinor = 0; spinor < Nspinor; spinor++) {
                        // find dPsi in direction dim
                        dpsi_xi = dpsi_full + dim*size_s*nspin;
                        Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k+spinor*DMndbyNspinor, 
                                                 dpsi_xi+spn_i*size_s+kpt*size_k+spinor*DMndbyNspinor, dim, kpt_vec, pSPARC->dmcomm);
                    }
                }
            }
        }
    } else {
        dpsi_xi_lv = (double complex *)malloc( size_k * sizeof(double complex) );  // dpsi_x, dpsi_y, dpsi_z along lattice vecotrs
        assert(dpsi_xi_lv != NULL);
        dpsi_x1 = dpsi_full;
        dpsi_x2 = dpsi_full + size_s*nspin;
        dpsi_x3 = dpsi_full + 2*size_s*nspin;
        for (dim = 0; dim < 3; dim++) {
            // count = 0;
            for(spn_i = 0; spn_i < nspin; spn_i++) {
                for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                    k1 = pSPARC->k1_loc[kpt];
                    k2 = pSPARC->k2_loc[kpt];
                    k3 = pSPARC->k3_loc[kpt];
                    kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                    for (spinor = 0; spinor < Nspinor; spinor++) {
                        // find dPsi in direction dim along lattice vector directions
                        Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, 
                                                pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k+spinor*DMndbyNspinor, 
                                                dpsi_xi_lv+spinor*DMndbyNspinor, dim, kpt_vec, pSPARC->dmcomm);
                    }
                    // find dPsi in direction dim in cartesian coordinates
                    for (i = 0; i < size_k; i++) {
                        if (dim == 0) {
                            dpsi_x1[i + spn_i*size_s+kpt*size_k] = pSPARC->gradT[0]*dpsi_xi_lv[i];
                            dpsi_x2[i + spn_i*size_s+kpt*size_k] = pSPARC->gradT[1]*dpsi_xi_lv[i];
                            dpsi_x3[i + spn_i*size_s+kpt*size_k] = pSPARC->gradT[2]*dpsi_xi_lv[i];
                        } else {
                            dpsi_x1[i + spn_i*size_s+kpt*size_k] += pSPARC->gradT[0+3*dim]*dpsi_xi_lv[i];
                            dpsi_x2[i + spn_i*size_s+kpt*size_k] += pSPARC->gradT[1+3*dim]*dpsi_xi_lv[i];
                            dpsi_x3[i + spn_i*size_s+kpt*size_k] += pSPARC->gradT[2+3*dim]*dpsi_xi_lv[i];
                        }
                    }
                }
            }
        }
        free(dpsi_xi_lv);
    }

    // find <chi_Jlm, psi>
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++){
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SC");
            beta = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SO1");
            beta = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * count;            
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SO2");
            count++;
        }
    }       

    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    count2 = 1;
    for (dim = 0; dim < 3; dim++) {
        dpsi_xi = dpsi_full + dim*size_s*nspin;
        for (dim2 = dim; dim2 < 3; dim2++) {
            count = 0;
            for(spn_i = 0; spn_i < nspin; spn_i++) {
                for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                    beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin * count2 + count);
                    Compute_Integral_Chi_StXmRjp_beta_Dpsi_kpt(pSPARC, dpsi_xi+spn_i*size_s+kpt*size_k, beta, spn_i, kpt, dim2, "SC");
                    beta = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin * count2 + count);
                    Compute_Integral_Chi_StXmRjp_beta_Dpsi_kpt(pSPARC, dpsi_xi+spn_i*size_s+kpt*size_k, beta, spn_i, kpt, dim2, "SO1");
                    beta = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin * count2 + count);
                    Compute_Integral_Chi_StXmRjp_beta_Dpsi_kpt(pSPARC, dpsi_xi+spn_i*size_s+kpt*size_k, beta, spn_i, kpt, dim2, "SO2");
                    count ++;
                }
            }    
            count2 ++;
        }
    }
    
    double complex *dpsi_xi_ptr, *dpsi_xj_ptr;
    // Kinetic stress
    count = 0;
    for (dim = 0; dim < 3; dim++) {
        dpsi_xi = dpsi_full + dim*size_s*nspin;
        for (dim2 = dim; dim2 < 3; dim2++) {
            dpsi_xj = dpsi_full + dim2*size_s*nspin;
            for(spn_i = 0; spn_i < nspin; spn_i++) {
                for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                    temp_k = 0;
                    for(n = 0; n < ncol; n++){
                        dpsi_xi_ptr = dpsi_xi + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_xi
                        dpsi_xj_ptr = dpsi_xj + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_xj

                        dpsii_dpsij = 0;
                        for(i = 0; i < DMnd; i++){
                            dpsii_dpsij += creal(*(dpsi_xi_ptr + i)) * creal(*(dpsi_xj_ptr + i)) + cimag(*(dpsi_xi_ptr + i)) * cimag(*(dpsi_xj_ptr + i));
                        }
                        g_nk = pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                        temp_k += dpsii_dpsij * g_nk;
                    }
                    stress_k[count] -= (2.0/pSPARC->Nspin/pSPARC->Nspinor) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k;
                }
            }
            count ++;
        }
    }
    free(dpsi_full);

    if (pSPARC->npNd > 1) {
        // MPI_Allreduce will fail randomly in valgrind test for unknown reason
        MPI_Request req0, req1, req2, req3;
        MPI_Status  sta0, sta1, sta2, sta3;
        MPI_Iallreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 7, 
                        MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm, &req0);
        MPI_Iallreduce(MPI_IN_PLACE, alpha_so1, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * Nspinor * nspin * 7, 
                        MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm, &req1);
        MPI_Iallreduce(MPI_IN_PLACE, alpha_so2, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * Nspinor * nspin * 7, 
                        MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm, &req2);
        MPI_Iallreduce(MPI_IN_PLACE, stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm, &req3);
        MPI_Wait(&req0, &sta0);
        MPI_Wait(&req1, &sta1);
        MPI_Wait(&req2, &sta2);
        MPI_Wait(&req3, &sta3);
    }

    /* calculate nonlocal stress */
    // go over all atoms and find nonlocal stress
    Compute_stress_tensor_nloc_by_integrals(pSPARC, stress_nl, alpha, "SC");
    Compute_stress_tensor_nloc_by_integrals(pSPARC, stress_nl, alpha_so1, "SO1");
    Compute_stress_tensor_nloc_by_integrals(pSPARC, stress_nl, alpha_so2, "SO2");
    
    energy_nl = Compute_Nonlocal_Energy_by_integrals(pSPARC, alpha, alpha_so1, alpha_so2);
    
    for(i = 0; i < 6; i++)
        stress_nl[i] *= (2.0/pSPARC->Nspin/pSPARC->Nspinor) * 2.0;
    
    energy_nl *= (2.0/pSPARC->Nspin/pSPARC->Nspinor)/pSPARC->dV;   

    pSPARC->stress_nl[0] = stress_nl[0] - energy_nl;
    pSPARC->stress_nl[1] = stress_nl[1];
    pSPARC->stress_nl[2] = stress_nl[2];
    pSPARC->stress_nl[3] = stress_nl[3] - energy_nl;
    pSPARC->stress_nl[4] = stress_nl[4];
    pSPARC->stress_nl[5] = stress_nl[5] - energy_nl;
    for(i = 0; i < 6; i++)
        pSPARC->stress_k[i] = stress_k[i];
    
    // sum over all spin
    if (pSPARC->npspin > 1) {    
        if (pSPARC->spincomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        } else{
            MPI_Reduce(pSPARC->stress_nl, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
            MPI_Reduce(pSPARC->stress_k, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        }
    }


    // sum over all kpoints
    if (pSPARC->npkpt > 1) {    
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        } else{
            MPI_Reduce(pSPARC->stress_nl, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
            MPI_Reduce(pSPARC->stress_k, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        }
    }

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;

        for(i = 0; i < 6; i++) {
            pSPARC->stress_nl[i] /= cell_measure;
            pSPARC->stress_k[i] /= cell_measure;
        }

    }

#ifdef DEBUG    
    if (!rank){
        printf("\nNon-local contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_nl, NULL);
        printf("\nKinetic contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_k, NULL);  
    } 
#endif
    free(alpha);
    free(alpha_so1);
    free(alpha_so2);
}


/**
 * @brief   Calculate <ChiSC_Jlm, ST(x-RJ')_beta, DPsi_n> for spinor stress
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_Chi_StXmRjp_beta_Dpsi_kpt(SPARC_OBJ *pSPARC, double complex *dpsi_xi, double complex *beta, int spn_i, int kpt, int dim2, char *option) 
{
    int i, n, ndc, ityp, iat, ncol, DMnd, dim, atom_index, l, m, lmax, Nk;
    int spinor, Nspinor, DMndbyNspinor, size_k, nspin, size_s, spinorshift, nproj, ispinor, *IP_displ;
    int indx, i_DM, j_DM, k_DM, DMnx, DMny;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    DMndbyNspinor = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;    
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor;

    double complex *dpsi_xi_rc, *dpsi_ptr, *dpsi_xi_rc_ptr;
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, R1, R2, R3, x1_R1, x2_R2, x3_R3, StXmRjp;
    double complex bloch_fac, a, b, **Chi;
    
    k1 = pSPARC->k1_loc[kpt];
    k2 = pSPARC->k2_loc[kpt];
    k3 = pSPARC->k3_loc[kpt];

    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        nproj = !strcmpi(option, "SC") ? pSPARC->nlocProj[ityp].nproj : pSPARC->nlocProj[ityp].nprojso_ext;
        if (!strcmpi(option, "SC")) 
            Chi = pSPARC->nlocProj[ityp].Chi_c;
        else if (!strcmpi(option, "SO1")) 
            Chi = pSPARC->nlocProj[ityp].Chisowt0;

        if (! nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
            R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3];
            R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
            R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(R1/Lx) * Lx) - k2 * (floor(R2/Ly) * Ly) - k3 * (floor(R3/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            b = 1.0;
            ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
            dpsi_xi_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            assert(dpsi_xi_rc);
            atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
            for (spinor = 0; spinor < Nspinor; spinor++) {
                if (!strcmpi(option, "SO2")) 
                    Chi = (spinor == 0) ? pSPARC->nlocProj[ityp].Chisowtnl : pSPARC->nlocProj[ityp].Chisowtl; 
                ispinor = !strcmpi(option, "SO2") ? (1 - spinor) : spinor;

                for (n = 0; n < ncol; n++) {
                    dpsi_ptr = dpsi_xi + n * DMnd + ispinor * DMndbyNspinor;
                    dpsi_xi_rc_ptr = dpsi_xi_rc + n * ndc;

                    for (i = 0; i < ndc; i++) {
                        indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                        k_DM = indx / (DMnx * DMny);
                        j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                        i_DM = indx % DMnx;
                        x1_R1 = (i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - R1;
                        x2_R2 = (j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - R2;
                        x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                        StXmRjp = pSPARC->LatUVec[0+dim2] * x1_R1 + pSPARC->LatUVec[3+dim2] * x2_R2 + pSPARC->LatUVec[6+dim2] * x3_R3;
                        *(dpsi_xi_rc_ptr + i) = *(dpsi_ptr + indx) * StXmRjp;
                    }
                }
            
                /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).

                */      
                spinorshift = IP_displ[pSPARC->n_atom] * ncol * spinor;
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nproj, ncol, ndc, &bloch_fac, Chi[iat], ndc, 
                            dpsi_xi_rc, ndc, &b, beta+spinorshift+IP_displ[atom_index]*ncol, nproj);                        
            }
            free(dpsi_xi_rc);
        }
    }
}

/**
 * @brief   Compute nonlocal Energy with spin-orbit coupling
 */
double Compute_Nonlocal_Energy_by_integrals(SPARC_OBJ *pSPARC, double complex *alpha, double complex *alpha_so1, double complex *alpha_so2)
{
    int i, j, k, n, np, ldispl, ityp, iat, ncol, Ns;
    int count, count2, l, m, lmax, kpt, Nk, spn_i, nspin, spinor, Nspinor, shift;
    double eJ, temp_e, temp2_e, g_nk, gamma_jl, kptwt, spinorfac, energy_nl, alpha_r, alpha_i, beta_r, beta_i;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    Nspinor = pSPARC->Nspinor;
    shift = pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol;

    energy_nl = 0.0;
    count = count2 = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            for (spinor = 0; spinor < Nspinor; spinor++) {
                spinorfac = (spinor == 0) ? 1.0 : -1.0; 
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                    lmax = pSPARC->psd[ityp].lmax;
                    for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                        eJ = 0.0; 
                        for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                            g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                            temp2_e = 0.0; 

                            // scalar relativistic term
                            ldispl = 0;
                            for (l = 0; l <= lmax; l++) {
                                // skip the local l
                                if (l == pSPARC->localPsd[ityp]) {
                                    ldispl += pSPARC->psd[ityp].ppl[l];
                                    continue;
                                }
                                for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                                    temp_e = 0.0; 
                                    for (m = -l; m <= l; m++) {
                                        alpha_r = creal(alpha[count]); alpha_i = cimag(alpha[count]);
                                        temp_e += alpha_r*alpha_r + alpha_i*alpha_i;
                                        count++;
                                    }
                                    gamma_jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                    temp2_e += temp_e * gamma_jl;
                                }
                                ldispl += pSPARC->psd[ityp].ppl[l];
                            }
                            
                            // spin-orbit coupling term 1
                            ldispl = 0;
                            for (l = 1; l <= lmax; l++) {
                                // skip the local l
                                if (l == pSPARC->localPsd[ityp]) {
                                    ldispl += pSPARC->psd[ityp].ppl_soc[l-1];
                                    continue;
                                }
                                for (np = 0; np < pSPARC->psd[ityp].ppl_soc[l-1]; np++) {
                                    for (m = -l; m <= l; m++) {
                                        if (!m) continue;
                                        alpha_r = creal(alpha_so1[count2]); alpha_i = cimag(alpha_so1[count2]);
                                        temp_e =  alpha_r * alpha_r + alpha_i * alpha_i;
                                        temp2_e += spinorfac*0.5*m*pSPARC->psd[ityp].Gamma_soc[ldispl+np] * temp_e;
                                        count2++;
                                    }
                                }
                                ldispl += pSPARC->psd[ityp].ppl_soc[l-1];
                            }
                            eJ += temp2_e * g_nk;
                        }
                        
                        kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                        energy_nl += kptwt * eJ;
                    }
                }
            }
        }
    }     

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                lmax = pSPARC->psd[ityp].lmax;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    eJ = 0.0; 
                    for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                        g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                        temp2_e = 0.0;
                        // spin-orbit coupling term 2
                        ldispl = 0;
                        for (l = 1; l <= lmax; l++) {
                            // skip the local l
                            if (l == pSPARC->localPsd[ityp]) {
                                ldispl += pSPARC->psd[ityp].ppl_soc[l-1];
                                continue;
                            }
                            for (np = 0; np < pSPARC->psd[ityp].ppl_soc[l-1]; np++) {
                                for (m = -l; m <= l; m++) {
                                    if (m == l) continue;
                                    alpha_r = creal(alpha_so2[count]);       alpha_i = cimag(alpha_so2[count]);
                                    beta_r  = creal(alpha_so2[count+shift]); beta_i  = cimag(alpha_so2[count+shift]);
                                    temp_e =  2*(alpha_r * beta_r + alpha_i * beta_i);
                                    temp2_e += 0.5*sqrt(l*(l+1)-m*(m+1))*pSPARC->psd[ityp].Gamma_soc[ldispl+np] * temp_e;
                                    count++;
                                }
                            }
                            ldispl += pSPARC->psd[ityp].ppl_soc[l-1];
                        }
                        eJ += temp2_e * g_nk;
                    }
                    kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                    energy_nl += kptwt * eJ;
                }
            }
            count += shift;
        }
    }     

    return energy_nl;
}


/**
 * @brief   Compute nonlocal stress tensor with spin-orbit coupling
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_stress_tensor_nloc_by_integrals(SPARC_OBJ *pSPARC, double *stress_nl, double complex *alpha, char *option)
{
    int i, j, k, n, np, ldispl, ityp, iat, ncol, Ns;
    int count, count2, l, m, lmax, kpt, Nk, spn_i, nspin, spinor, Nspinor;
    int l_start, mexclude, ppl, *IP_displ;
    double g_nk, gamma_jl, kptwt, alpha_r, alpha_i, SJ[6], temp_s, temp2_s[6], scaled_gamma_Jl;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    Nspinor = pSPARC->Nspinor;

    l_start = !strcmpi(option, "SC") ? 0 : 1;
    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;

    double complex *beta1_x1, *beta2_x1, *beta3_x1, *beta2_x2, *beta3_x2, *beta3_x3; 
    beta1_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin;
    beta2_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin * 2;
    beta3_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin * 3;
    beta2_x2 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin * 4;
    beta3_x2 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin * 5;
    beta3_x3 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin * 6;

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            for (spinor = 0; spinor < Nspinor; spinor++) {
                double spinorfac = (spinor == 0) ? 1.0 : -1.0; 
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                    lmax = pSPARC->psd[ityp].lmax;
                    for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                        for(i = 0; i < 6; i++) SJ[i] = 0.0;
                        for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                            g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                            for(i = 0; i < 6; i++) temp2_s[i] = 0.0;
                            ldispl = 0;
                            for (l = l_start; l <= lmax; l++) {
                                mexclude = !strcmpi(option, "SC") ? (l+1) : (!strcmpi(option, "SO1") ? 0 : l);
                                ppl = !strcmpi(option, "SC") ? pSPARC->psd[ityp].ppl[l] : pSPARC->psd[ityp].ppl_soc[l-1];

                                // skip the local l
                                if (l == pSPARC->localPsd[ityp]) {
                                    ldispl += ppl;
                                    continue;
                                }
                                for (np = 0; np < ppl; np++) {
                                    for (m = -l; m <= l; m++) {
                                        if (m == mexclude) continue;
                                        if (!strcmpi(option, "SC")) scaled_gamma_Jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                        else if (!strcmpi(option, "SO1")) scaled_gamma_Jl = spinorfac*0.5*m*pSPARC->psd[ityp].Gamma_soc[ldispl+np];
                                        else if (!strcmpi(option, "SO2")) scaled_gamma_Jl = 0.5*sqrt(l*(l+1)-m*(m+1))*pSPARC->psd[ityp].Gamma_soc[ldispl+np];

                                        alpha_r = creal(alpha[count]); alpha_i = cimag(alpha[count]);
                                        temp2_s[0] += scaled_gamma_Jl * (alpha_r * creal(beta1_x1[count]) - alpha_i * cimag(beta1_x1[count]));
                                        temp2_s[1] += scaled_gamma_Jl * (alpha_r * creal(beta2_x1[count]) - alpha_i * cimag(beta2_x1[count]));
                                        temp2_s[2] += scaled_gamma_Jl * (alpha_r * creal(beta3_x1[count]) - alpha_i * cimag(beta3_x1[count]));
                                        temp2_s[3] += scaled_gamma_Jl * (alpha_r * creal(beta2_x2[count]) - alpha_i * cimag(beta2_x2[count]));
                                        temp2_s[4] += scaled_gamma_Jl * (alpha_r * creal(beta3_x2[count]) - alpha_i * cimag(beta3_x2[count]));
                                        temp2_s[5] += scaled_gamma_Jl * (alpha_r * creal(beta3_x3[count]) - alpha_i * cimag(beta3_x3[count]));
                                        count++;
                                    }
                                }
                                ldispl += ppl;
                            }
                            for(i = 0; i < 6; i++)
                                SJ[i] += temp2_s[i] * g_nk;
                        }
                        
                        kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                        for(i = 0; i < 6; i++)
                            stress_nl[i] -= kptwt * SJ[i];
                    }
                }
            }
        }
    }     
}


//////////////////////////////////////////////////////////////////////////////////////////
///////////////////      Pressure functions      /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


/**
 * @brief    Calculate nonlocal pressure components.
 */
void Calculate_nonlocal_pressure_kpt_spinor(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, k, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, i_DM, j_DM, k_DM;
    int dim, count, count2, l, m, lmax, atom_index, kpt, Nk, size_k, spn_i, nspin, size_s, spinor, DMndbyNspinor, Nspinor, spinorshift;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    DMndbyNspinor = pSPARC->Nd_d_dmcomm;
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor;
    
    double complex *alpha, *alpha_so1, *alpha_so2, *beta;
    double R1, R2, R3, pJ, eJ, temp_e, temp_p, temp2_e, temp2_p, g_nk;
    double pressure_nloc = 0.0, energy_nl;

    alpha = (double complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 4, sizeof(double complex));
    alpha_so1 = (double complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 4, sizeof(double complex));
    alpha_so2 = (double complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 4, sizeof(double complex));
    assert(alpha != NULL && alpha_so1 != NULL && alpha_so2 != NULL);

    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, kpt_vec;
    double complex bloch_fac, a, b;
#ifdef DEBUG 
    if (!rank) printf("Start Calculating nonlocal pressure\n");
#endif

    // find <chi_Jlm, psi>
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++){
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SC");
            beta = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SO1");
            beta = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * count;            
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SO2");
            count++;
        }
    }       

    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    count2 = 1;
    for (dim = 0; dim < 3; dim++) {
        count = 0;
        for(spn_i = 0; spn_i < nspin; spn_i++) {
            for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                k1 = pSPARC->k1_loc[kpt];
                k2 = pSPARC->k2_loc[kpt];
                k3 = pSPARC->k3_loc[kpt];
                kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                for (spinor = 0; spinor < Nspinor; spinor++) {
                    // find dPsi in direction dim along lattice vector directions
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k+spinor*DMndbyNspinor, 
                                            pSPARC->Yorb_kpt+spinor*DMndbyNspinor, dim, kpt_vec, pSPARC->dmcomm);
                }
                beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin * count2 + count);
                Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt(pSPARC, pSPARC->Yorb_kpt, beta, spn_i, kpt, dim, "SC");
                beta = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin * count2 + count);
                Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt(pSPARC, pSPARC->Yorb_kpt, beta, spn_i, kpt, dim, "SO1");
                beta = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin * count2 + count);
                Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt(pSPARC, pSPARC->Yorb_kpt, beta, spn_i, kpt, dim, "SO2");
                count ++;
            }
        }    
        count2 ++;
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, alpha_so1, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, alpha_so2, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * Nspinor * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal pressure */
    // go over all atoms and find nonlocal pressure
    Compute_pressure_nloc_by_integrals(pSPARC, &pressure_nloc, alpha, "SC");
    Compute_pressure_nloc_by_integrals(pSPARC, &pressure_nloc, alpha_so1, "SO1");
    Compute_pressure_nloc_by_integrals(pSPARC, &pressure_nloc, alpha_so2, "SO2");
    pressure_nloc *= 2.0/pSPARC->Nspin/pSPARC->Nspinor;

    energy_nl = Compute_Nonlocal_Energy_by_integrals(pSPARC, alpha, alpha_so1, alpha_so2);
    energy_nl *= (2.0/pSPARC->Nspin/pSPARC->Nspinor)/pSPARC->dV;
    pressure_nloc -= energy_nl;
    
    // sum over all spin
    if (pSPARC->npspin > 1) {    
        if (pSPARC->spincomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        } else{
            MPI_Reduce(&pressure_nloc, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        }
    }
        
    // sum over all kpoints
    if (pSPARC->npkpt > 1) {    
        if (pSPARC->kptcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        } else{
            MPI_Reduce(&pressure_nloc, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        }
    }

    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        } else{
            MPI_Reduce(&pressure_nloc, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        }
    }
    
    if (!rank) {
        pSPARC->pres_nl = pressure_nloc;
    }

#ifdef DEBUG    
    if (!rank){
        printf("Pressure contribution from nonlocal pseudopotential: = %.15f Ha\n", pSPARC->pres_nl);
    }    
#endif

    free(alpha);
    free(alpha_so1);
    free(alpha_so2);
}



/**
 * @brief   Calculate <ChiSC_Jlm, (x-RJ')_beta, DPsi_n> for spinor stress
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt(SPARC_OBJ *pSPARC, double complex *dpsi_xi, double complex *beta, int spn_i, int kpt, int dim2, char *option) 
{
    int i, n, ndc, ityp, iat, ncol, DMnd, dim, atom_index, l, m, lmax, Nk;
    int spinor, Nspinor, DMndbyNspinor, size_k, nspin, size_s, spinorshift, nproj, ispinor, *IP_displ;
    int indx, i_DM, j_DM, k_DM, DMnx, DMny;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    DMndbyNspinor = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;    
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor;

    double complex *dpsi_xi_rc, *dpsi_ptr, *dpsi_xi_rc_ptr;
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, R1, R2, R3, x1_R1, x2_R2, x3_R3, XmRjp;
    double complex bloch_fac, a, b, **Chi;
    
    k1 = pSPARC->k1_loc[kpt];
    k2 = pSPARC->k2_loc[kpt];
    k3 = pSPARC->k3_loc[kpt];

    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        nproj = !strcmpi(option, "SC") ? pSPARC->nlocProj[ityp].nproj : pSPARC->nlocProj[ityp].nprojso_ext;
        if (!strcmpi(option, "SC")) 
            Chi = pSPARC->nlocProj[ityp].Chi_c;
        else if (!strcmpi(option, "SO1")) 
            Chi = pSPARC->nlocProj[ityp].Chisowt0;

        if (! nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
            R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3];
            R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
            R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(R1/Lx) * Lx) - k2 * (floor(R2/Ly) * Ly) - k3 * (floor(R3/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            b = 1.0;
            ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
            dpsi_xi_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            assert(dpsi_xi_rc);
            atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
            for (spinor = 0; spinor < Nspinor; spinor++) {
                if (!strcmpi(option, "SO2")) 
                    Chi = (spinor == 0) ? pSPARC->nlocProj[ityp].Chisowtnl : pSPARC->nlocProj[ityp].Chisowtl; 
                ispinor = !strcmpi(option, "SO2") ? (1 - spinor) : spinor;

                for (n = 0; n < ncol; n++) {
                    dpsi_ptr = dpsi_xi + n * DMnd + ispinor * DMndbyNspinor;
                    dpsi_xi_rc_ptr = dpsi_xi_rc + n * ndc;

                    for (i = 0; i < ndc; i++) {
                        indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                        k_DM = indx / (DMnx * DMny);
                        j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                        i_DM = indx % DMnx;
                        x1_R1 = (i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - R1;
                        x2_R2 = (j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - R2;
                        x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                        XmRjp = (dim2 == 0) ? x1_R1 : ((dim2 == 1) ? x2_R2 :x3_R3);
                        *(dpsi_xi_rc_ptr + i) = *(dpsi_ptr + indx) * XmRjp;
                    }
                }
            
                /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).

                */      
                spinorshift = IP_displ[pSPARC->n_atom] * ncol * spinor;
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nproj, ncol, ndc, &bloch_fac, Chi[iat], ndc, 
                            dpsi_xi_rc, ndc, &b, beta+spinorshift+IP_displ[atom_index]*ncol, nproj);                        
            }
            free(dpsi_xi_rc);
        }
    }
}


/**
 * @brief   Compute nonlocal pressure with spin-orbit coupling
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_pressure_nloc_by_integrals(SPARC_OBJ *pSPARC, double *pressure_nloc, double complex *alpha, char *option)
{
    int i, j, k, n, np, ldispl, ityp, iat, ncol, Ns;
    int count, count2, l, m, lmax, kpt, Nk, spn_i, nspin, spinor, Nspinor;
    int l_start, mexclude, ppl, *IP_displ;
    double g_nk, gamma_jl, kptwt, alpha_r, alpha_i, temp_p, pJ, scaled_gamma_Jl;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    Nspinor = pSPARC->Nspinor;

    l_start = !strcmpi(option, "SC") ? 0 : 1;
    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;

    double complex *beta1_x1, *beta2_x2, *beta3_x3; 
    beta1_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin;
    beta2_x2 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin * 2;
    beta3_x3 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin * 3;

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            for (spinor = 0; spinor < Nspinor; spinor++) {
                double spinorfac = (spinor == 0) ? 1.0 : -1.0; 
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                    lmax = pSPARC->psd[ityp].lmax;
                    for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                        pJ = 0;
                        for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                            g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                            temp_p = 0.0;
                            ldispl = 0;
                            for (l = l_start; l <= lmax; l++) {
                                mexclude = !strcmpi(option, "SC") ? (l+1) : (!strcmpi(option, "SO1") ? 0 : l);
                                ppl = !strcmpi(option, "SC") ? pSPARC->psd[ityp].ppl[l] : pSPARC->psd[ityp].ppl_soc[l-1];

                                // skip the local l
                                if (l == pSPARC->localPsd[ityp]) {
                                    ldispl += ppl;
                                    continue;
                                }
                                for (np = 0; np < ppl; np++) {
                                    for (m = -l; m <= l; m++) {
                                        if (m == mexclude) continue;
                                        if (!strcmpi(option, "SC")) scaled_gamma_Jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                        else if (!strcmpi(option, "SO1")) scaled_gamma_Jl = spinorfac*0.5*m*pSPARC->psd[ityp].Gamma_soc[ldispl+np];
                                        else if (!strcmpi(option, "SO2")) scaled_gamma_Jl = 0.5*sqrt(l*(l+1)-m*(m+1))*pSPARC->psd[ityp].Gamma_soc[ldispl+np];

                                        alpha_r = creal(alpha[count]); alpha_i = cimag(alpha[count]);
                                        temp_p += scaled_gamma_Jl * (alpha_r * creal(beta1_x1[count]) - alpha_i * cimag(beta1_x1[count]));
                                        temp_p += scaled_gamma_Jl * (alpha_r * creal(beta2_x2[count]) - alpha_i * cimag(beta2_x2[count]));
                                        temp_p += scaled_gamma_Jl * (alpha_r * creal(beta3_x3[count]) - alpha_i * cimag(beta3_x3[count]));
                                        count++;
                                    }
                                }
                                ldispl += ppl;
                            }
                            pJ += temp_p * g_nk;
                        }
                        kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                        *pressure_nloc -= 2.0 * kptwt * pJ;
                    }
                }
            }
        }
    }     
}