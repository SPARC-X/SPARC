/***
 * @file    sqNlocVecRoutines.c
 * @brief   This file contains the functions for nonlocal routines in SQ method.
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
/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "isddft.h"
#include "sqNlocVecRoutines.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)


/**
 * @brief   This function gets the overlapping nodes of each Rcut cuboidal or spherical domain of 
 *          each FD node in current domain and atoms and their images' rc spherical domain. In current 
 *          implementation, only nloc_mem_flag in SQDFT is implemented, i.e., saving nonlocal projector 
 *          chi matrix for each FD node. 
 * 
 * @TODO     Implement the nloc_mem_flag option if memory issue happens.
 */
void GetNonlocalProjectorsForNode(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ *nlocProj, NLOC_PROJ_OBJ ***nlocProj_SQ, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, ATOM_NLOC_INFLUENCE_OBJ ***Atom_Influence_nloc_SQ, MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) return;
    int rank, nd, ityp, iat, ipos, ndc, iat_, ndc_, ipos_, icol;
    int k_PR, j_PR, i_PR, k_dm, j_dm, i_dm, k_dm_PR, j_dm_PR, i_dm_PR, k_PR_dm, j_PR_dm, i_PR_dm;
    int DMnx, DMny, DMnx_PR, DMny_PR, DMnx_dm, DMny_dm, indx, nproj;
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int *nloc = pSQ->nloc;

    MPI_Comm_rank(comm, &rank);
#ifdef DEBUG
    if (rank == 0) printf("\nStart to get nonlocal projectors for each node.\n");    
#endif
    // number of nodes in the local PR domain
    DMnx_PR = pSQ->DMVertices_PR[1] - pSQ->DMVertices_PR[0] + 1;
    DMny_PR = pSQ->DMVertices_PR[3] - pSQ->DMVertices_PR[2] + 1;
    // number of nodes in the local dmcomm_SQ domain
    DMnx_dm = pSQ->DMVertices_SQ[1] - pSQ->DMVertices_SQ[0] + 1;
    DMny_dm = pSQ->DMVertices_SQ[3] - pSQ->DMVertices_SQ[2] + 1;
    // number of nodes in Rcut domain of one node
    DMnx = 1 + 2 * pSQ->nloc[0];
    DMny = 1 + 2 * pSQ->nloc[1];

    (*nlocProj_SQ) = (NLOC_PROJ_OBJ **)malloc( sizeof(NLOC_PROJ_OBJ *) * pSQ->DMnd_SQ); 
    (*Atom_Influence_nloc_SQ) = (ATOM_NLOC_INFLUENCE_OBJ **)malloc( sizeof(ATOM_NLOC_INFLUENCE_OBJ *) * pSQ->DMnd_SQ); 

    for (nd = 0; nd < pSQ->DMnd_SQ; nd++) {
        // node nd (i_dm, j_dm, k_dm) in local dmcomm_SQ domain
        k_dm = nd / (DMnx_dm * DMny_dm);
        j_dm = (nd - k_dm * (DMnx_dm * DMny_dm)) / DMnx_dm;
        i_dm = nd % DMnx_dm;
        // node nd (i_dm_PR, j_dm_PR, k_dm_PR) in PR domain
        k_dm_PR = k_dm + nloc[2];
        j_dm_PR = j_dm + nloc[1];
        i_dm_PR = i_dm + nloc[0];

        (*nlocProj_SQ)[nd] = (NLOC_PROJ_OBJ *)malloc( sizeof(NLOC_PROJ_OBJ) * pSPARC->Ntypes ); 
        (*Atom_Influence_nloc_SQ)[nd] = (ATOM_NLOC_INFLUENCE_OBJ *)malloc( sizeof(ATOM_NLOC_INFLUENCE_OBJ) * pSPARC->Ntypes); 
        
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
            nproj = (*nlocProj_SQ)[nd][ityp].nproj = nlocProj[ityp].nproj;
            // loop over all images intersecting with local PR
            // already skip instances with n_atom = 0
            int count_natom = 0;
            int n_atom = Atom_Influence_nloc[ityp].n_atom;

            if (n_atom == 0) {
                (*Atom_Influence_nloc_SQ)[nd][ityp].n_atom = 0;
                continue;
            }

            int *iat_list = (int *) malloc(sizeof(int) * n_atom);
            int *ndc_list = (int *) malloc(sizeof(int) * n_atom);
            int **ipos_list = (int **) malloc(sizeof(int*) * n_atom);
            
            for (iat = 0; iat < n_atom; iat++) {
                // ndc must be > 0
                ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
                ipos_list[iat] = (int*) malloc(sizeof(int) * ndc);
                int count_ndc = 0;

                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_nloc[ityp].grid_pos[iat][ipos];
                    // w.r.t. PR domain 
                    k_PR = indx / (DMnx_PR * DMny_PR);
                    j_PR = (indx - k_PR * (DMnx_PR * DMny_PR)) / DMnx_PR;
                    i_PR = indx % DMnx_PR;
                    
                    int flag = 0;
                    double rx2, ry2, rz2, rcut;
                    rz2 = (k_PR - k_dm_PR) * pSPARC->delta_z; rz2 *= rz2;
                    ry2 = (j_PR - j_dm_PR) * pSPARC->delta_y; ry2 *= ry2;
                    rx2 = (i_PR - i_dm_PR) * pSPARC->delta_x; rx2 *= rx2;
                    rcut = max(max(nloc[0]*pSPARC->delta_x, nloc[1]*pSPARC->delta_y), nloc[2]*pSPARC->delta_z);
                    flag = (int) ((rx2 + ry2 + rz2) > (rcut * rcut));

                    if (flag == 0) {
                        // ipos_list[iat] first count_ndc elements are original index in 1 to ndc
                        ipos_list[iat][count_ndc] = ipos;
                        count_ndc ++;            // this node is within current node's Rcut domain
                    }
                }
                if (count_ndc > 0) {
                    iat_list[count_natom] = iat;
                    ndc_list[count_natom] = count_ndc;
                    count_natom ++;          // This image has intersection with current node's Rcut domain
                }
            }
            // count_natom images have intersection 
            (*Atom_Influence_nloc_SQ)[nd][ityp].n_atom = count_natom;
            if (count_natom == 0) {
                for (iat = 0; iat < n_atom; iat++) { 
                    free(ipos_list[iat]);
                }
                free(iat_list);
                free(ndc_list);
                free(ipos_list);
                continue;
            }
            (*Atom_Influence_nloc_SQ)[nd][ityp].coords = (double *) malloc(sizeof(double) * count_natom * 3);
            (*Atom_Influence_nloc_SQ)[nd][ityp].atom_index = (int *) malloc(sizeof(int) * count_natom);
            (*Atom_Influence_nloc_SQ)[nd][ityp].ndc = (int *) malloc(sizeof(int) * count_natom);
            (*Atom_Influence_nloc_SQ)[nd][ityp].grid_pos = (int **) malloc(sizeof(int*) * count_natom);
            (*nlocProj_SQ)[nd][ityp].Chi = (double **)malloc( sizeof(double *) * count_natom);

            for (iat = 0; iat < count_natom; iat++) {
                iat_ = iat_list[iat];
                ndc  = ndc_list[iat];
                ndc_ = Atom_Influence_nloc[ityp].ndc[iat_];
                (*Atom_Influence_nloc_SQ)[nd][ityp].ndc[iat] = ndc;
                (*Atom_Influence_nloc_SQ)[nd][ityp].coords[iat*3] = Atom_Influence_nloc[ityp].coords[iat_*3];
                (*Atom_Influence_nloc_SQ)[nd][ityp].coords[iat*3+1] = Atom_Influence_nloc[ityp].coords[iat_*3+1];
                (*Atom_Influence_nloc_SQ)[nd][ityp].coords[iat*3+2] = Atom_Influence_nloc[ityp].coords[iat_*3+2];
                (*Atom_Influence_nloc_SQ)[nd][ityp].atom_index[iat] = Atom_Influence_nloc[ityp].atom_index[iat_];
                (*Atom_Influence_nloc_SQ)[nd][ityp].grid_pos[iat] = (int *)malloc(sizeof(int) * ndc);
                for (ipos = 0; ipos < ndc; ipos++) {
                    ipos_ = ipos_list[iat_][ipos];
                    indx = Atom_Influence_nloc[ityp].grid_pos[iat_][ipos_];
                    // w.r.t. PR domain 
                    k_PR = indx / (DMnx_PR * DMny_PR);
                    j_PR = (indx - k_PR * (DMnx_PR * DMny_PR)) / DMnx_PR;
                    i_PR = indx % DMnx_PR;
                    // w.r.t. current node Rcut domain
                    k_PR_dm = k_PR - k_dm;
                    j_PR_dm = j_PR - j_dm;
                    i_PR_dm = i_PR - i_dm;
                    indx = i_PR_dm + j_PR_dm * DMnx + k_PR_dm * DMnx * DMny;
                    (*Atom_Influence_nloc_SQ)[nd][ityp].grid_pos[iat][ipos] = indx;
                }

                (*nlocProj_SQ)[nd][ityp].Chi[iat] = (double *)malloc( sizeof(double) * ndc * nproj);
                for (ipos = 0; ipos < ndc; ipos++) {
                    ipos_ = ipos_list[iat_][ipos];
                    for (icol = 0; icol < nproj; icol++) {
                        (*nlocProj_SQ)[nd][ityp].Chi[iat][ipos + icol*ndc] = nlocProj[ityp].Chi[iat_][ipos_ + icol*ndc_];
                    }
                }
            }

            for (iat = 0; iat < n_atom; iat++) { 
                free(ipos_list[iat]);
            }
            free(iat_list);
            free(ndc_list);
            free(ipos_list);
        }
    }
}


/**
 * @brief   Calculate Vnl times vectors in a matrix-free way.
 */
void Vnl_vec_mult_SQ(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                  NLOC_PROJ_OBJ *nlocProj, const double *x, double *Hx)
{
    int i, np, count;
    int ityp, iat, l, m, ldispl, lmax, ndc, nproj, lloc;
    double *alpha, *x_rc;

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        nproj = nlocProj[ityp].nproj;
        if (!nproj) continue; // this is typical for hydrogen
        alpha = (double *)calloc(nproj, sizeof(double));
        lloc = pSPARC->localPsd[ityp];
        lmax = pSPARC->psd[ityp].lmax;
        
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            x_rc = (double *)malloc(ndc * sizeof(double));
            for (i = 0; i < ndc; i++) {
                x_rc[i] = x[Atom_Influence_nloc[ityp].grid_pos[iat][i]];
            }
            
            // Find inner product
            cblas_dgemv (CblasColMajor, CblasTrans, ndc, nproj, pSPARC->dV, 
                    nlocProj[ityp].Chi[iat], ndc, x_rc, 1, 0.0, alpha, 1);
            
            // Apply Gamma
            count = 0;
            ldispl = 0;
            for (l = 0; l <= lmax; l++) {
                // skip the local l
                if (l == lloc) {
                    ldispl += pSPARC->psd[ityp].ppl[l];
                    continue;
                }
                for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                    for (m = -l; m <= l; m++) {
                        // printf("count %d, ldispl+np %d\n",count,  ldispl+np);
                        alpha[count++] *= pSPARC->psd[ityp].Gamma[ldispl+np];
                    }
                }
                ldispl += pSPARC->psd[ityp].ppl[l];
            }
            cblas_dgemv (CblasColMajor, CblasNoTrans, ndc, nproj, 1.0, 
                    nlocProj[ityp].Chi[iat], ndc, alpha, 1, 0.0, x_rc, 1);
            for (i = 0; i < ndc; i++) {
                Hx[Atom_Influence_nloc[ityp].grid_pos[iat][i]] += x_rc[i];
            }
            free(x_rc);
        }
        free(alpha);
    }
}



/**
 * @brief   Calculate Vnl times vectors in a matrix-free way for force calculation.
 */
void Vnl_vec_mult_J_SQ(const SPARC_OBJ *pSPARC, int DMnd, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                  NLOC_PROJ_OBJ *nlocProj, double *x, double *Vx)
{
    int i, np, icol, count, counter_cp = 0, indx;
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index, nproj, *nloc;
    int i_g, j_g, k_g, lloc;
    double x0_i, y0_i, z0_i, x2, y2, z2, rc, rc2;
    double *alpha, *x_rc;
    SQ_OBJ *pSQ = pSPARC->pSQ;

    int DMnx = pSQ->DMnx_SQ;
    int DMny = pSQ->DMny_SQ;
    int DMnxny = DMnx * DMny;

    int k_dm = nd / DMnxny;
    int j_dm = (nd - k_dm * DMnxny) / DMnx;
    int i_dm = nd % DMnx;

    nloc = pSQ->nloc;
    indx = nloc[0] + nloc[1] * (2*nloc[0]+1) + nloc[2] * (2*nloc[0]+1)*(2*nloc[1]+1);
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        nproj = nlocProj[ityp].nproj;
        if (!nproj) continue; // this is typical for hydrogen
        alpha = (double *)calloc(nproj, sizeof(double));
        lloc = pSPARC->localPsd[ityp];
        lmax = pSPARC->psd[ityp].lmax;
        rc = 0.0;
        // find max rc 
        for (i = 0; i <= pSPARC->psd[ityp].lmax; i++) {
            rc = max(rc, pSPARC->psd[ityp].rc[i]);
        }
        rc2 = rc * rc;

        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            i_g = i_dm + pSQ->DMVertices_SQ[0];
            j_g = j_dm + pSQ->DMVertices_SQ[2];
            k_g = k_dm + pSQ->DMVertices_SQ[4];
            x2 = i_g * pSPARC->delta_x - x0_i;  x2 *= x2;
            y2 = j_g * pSPARC->delta_y - y0_i;  y2 *= y2;
            z2 = k_g * pSPARC->delta_z - z0_i;  z2 *= z2;
            if ((x2 + y2 + z2) > rc2) continue;
            
            // find counter_pc
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            for (i = 0; i < ndc; i++) {
                if (Atom_Influence_nloc[ityp].grid_pos[iat][i] == indx) {
                    counter_cp = i;
                }
            }

            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            x_rc = (double *)malloc(ndc * sizeof(double));
            
            for (i = 0; i < ndc; i++) {
                x_rc[i] = x[Atom_Influence_nloc[ityp].grid_pos[iat][i]];
            }
            // Find inner product
            cblas_dgemv (CblasColMajor, CblasTrans, ndc, nproj, pSPARC->dV, 
                    nlocProj[ityp].Chi[iat], ndc, x_rc, 1, 0.0, alpha, 1);
            
            // Apply Gamma
            count = 0;
            ldispl = 0;
            for (l = 0; l <= lmax; l++) {
                // skip the local l
                if (l == lloc) {
                    ldispl += pSPARC->psd[ityp].ppl[l];
                    continue;
                }
                for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                    for (m = -l; m <= l; m++) {
                        alpha[count++] *= pSPARC->psd[ityp].Gamma[ldispl+np];
                    }
                }
                ldispl += pSPARC->psd[ityp].ppl[l];
            }
            
            for (icol = 0; icol < nproj; icol++) {
                Vx[atom_index] += alpha[icol] * nlocProj[ityp].Chi[iat][counter_cp + icol*ndc];
            }
            free(x_rc);
        }
        free(alpha);
    }
    return;
}


/**
 * @brief   Calculate Vnl times vectors in a matrix-free way for stress calculation.
 */
void Vnl_vec_mult_dir_SQ(const SPARC_OBJ *pSPARC, int DMnd, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                  NLOC_PROJ_OBJ *nlocProj, int dir, double *x, double *Vx)
{
    int i, np, icol, count, counter_cp = 0, indx, pos;
    int ityp, iat, l, m, ldispl, lmax, ndc, nproj, *nloc;
    int i_g, j_g, k_g, DMnx, DMny, lloc;
    double x0_i, y0_i, z0_i, x2, y2, z2, rc, rc2;
    double *alpha, *x_rc, *disp;
    SQ_OBJ *pSQ = pSPARC->pSQ;

    int DMnx_SQ = pSQ->DMnx_SQ;
    int DMny_SQ = pSQ->DMny_SQ;
    int DMnxny_SQ = DMnx_SQ * DMny_SQ;

    int k_dm = nd / DMnxny_SQ;
    int j_dm = (nd - k_dm * DMnxny_SQ) / DMnx_SQ;
    int i_dm = nd % DMnx_SQ;

    if (dir < 1 || dir > 3) {
        printf("ERROR in Vnl_vec_mult_dir_SQ:\n"
            "\tdir should be 1, 2, or 3 indicating x, y, or z direction!\n");
        exit(-1);
    }

    nloc = pSQ->nloc;
    DMnx = 2 * nloc[0] + 1;
    DMny = 2 * nloc[1] + 1;    
    indx = nloc[0] + nloc[1] * DMnx + nloc[2] * DMnx*DMny;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        nproj = nlocProj[ityp].nproj;
        if (!nproj) continue; // this is typical for hydrogen
        alpha = (double *)calloc(nproj, sizeof(double));
        lloc = pSPARC->localPsd[ityp];
        lmax = pSPARC->psd[ityp].lmax;
        rc = 0.0;
        // find max rc 
        for (i = 0; i <= pSPARC->psd[ityp].lmax; i++) {
            rc = max(rc, pSPARC->psd[ityp].rc[i]);
        }
        rc2 = rc * rc;

        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            i_g = i_dm + pSQ->DMVertices_SQ[0];
            j_g = j_dm + pSQ->DMVertices_SQ[2];
            k_g = k_dm + pSQ->DMVertices_SQ[4];
            x2 = i_g * pSPARC->delta_x - x0_i;  x2 *= x2;
            y2 = j_g * pSPARC->delta_y - y0_i;  y2 *= y2;
            z2 = k_g * pSPARC->delta_z - z0_i;  z2 *= z2;
            // distance (i_dm, j_dm, k_dm) to atom image
            if ((x2 + y2 + z2) > rc2) continue;
            
            // find counter_pc
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            x_rc = (double *)malloc( ndc * sizeof(double));
            disp = (double *)malloc( ndc * sizeof(double));

            for (i = 0; i < ndc; i++) {
                pos = Atom_Influence_nloc[ityp].grid_pos[iat][i];
                if (pos == indx) {
                    counter_cp = i;
                }
                // coordinates of FD node in local Rcut domain
                k_g = pos / (DMnx * DMny);
                j_g = (pos - k_g * (DMnx * DMny)) / DMnx;
                i_g = pos % DMnx;
                if (dir == 1) {
                    disp[i] = (i_g + i_dm + pSQ->DMVertices_PR[0]) * pSPARC->delta_x - x0_i;
                } else if (dir == 2) {
                    disp[i] = (j_g + j_dm + pSQ->DMVertices_PR[2]) * pSPARC->delta_y - y0_i;
                } else {
                    disp[i] = (k_g + k_dm + pSQ->DMVertices_PR[4]) * pSPARC->delta_z - z0_i;
                }
            }
            
            for (i = 0; i < ndc; i++) {
                x_rc[i] = x[Atom_Influence_nloc[ityp].grid_pos[iat][i]] * disp[i];
            }

            // Find inner product
            cblas_dgemv (CblasColMajor, CblasTrans, ndc, nproj, pSPARC->dV, 
                    nlocProj[ityp].Chi[iat], ndc, x_rc, 1, 0.0, alpha, 1);

            // Apply Gamma
            count = 0;
            ldispl = 0;
            for (l = 0; l <= lmax; l++) {
                // skip the local l
                if (l == lloc) {
                    ldispl += pSPARC->psd[ityp].ppl[l];
                    continue;
                }
                for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                    for (m = -l; m <= l; m++) {
                        alpha[count++] *= pSPARC->psd[ityp].Gamma[ldispl+np];
                    }
                }
                ldispl += pSPARC->psd[ityp].ppl[l];
            }
            
            for (icol = 0; icol < nproj; icol++) {
                *Vx += alpha[icol] * nlocProj[ityp].Chi[iat][counter_cp + icol*ndc];
            }
            free(x_rc);
            free(disp);
        }
        free(alpha);
    }
    return;
}

