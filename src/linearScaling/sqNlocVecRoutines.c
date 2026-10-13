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
 * @brief   The nonlocal projectors of grid point nd: for every atom image of the PR-domain set
 *          (Atom_Influence_nloc_kptcomm, nlocProj_kptcomm) the rows of its projectors Chi at the PR nodes within the
 *          point's Rcut sphere and box, with the node positions relative to the point's Rcut box, into
 *          pSQ->nlocProj_pt / pSQ->Atom_Influence_nloc_pt. Only one point's projectors are stored at a time: the
 *          function runs at the start of each Lanczos run (and in the force step for the force terms), the buffers
 *          (coords, atom_index, ndc, grid_pos, Chi of the images per type, flat storage nlpt_pos / nlpt_chi) only grow.
 *          xs of the point's set holds the image's index in the PR-domain set.
 */
void GetNonlocalProjectorsForOnePoint(SPARC_OBJ *pSPARC, int nd)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;
    NLOC_PROJ_OBJ *nlocProj = pSPARC->nlocProj_kptcomm, *nlpt = pSQ->nlocProj_pt;
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc = pSPARC->Atom_Influence_nloc_kptcomm, *aipt = pSQ->Atom_Influence_nloc_pt;
    int ityp, iat, iat_, ipos, icol, indx, ndc, ndc_, nproj;
    int k_PR, j_PR, i_PR, k_dm, j_dm, i_dm, k_dm_PR, j_dm_PR, i_dm_PR;
    int *nloc = pSQ->nloc;
    size_t npos = 0, nchi = 0, off, offc;

    // number of nodes in the local PR domain, in the local dmcomm_SQ domain, and in the Rcut domain of one node
    int DMnx_PR = pSQ->DMnx_PR, DMny_PR = pSQ->DMny_PR;
    int DMnx_dm = pSQ->DMnx_SQ, DMny_dm = pSQ->DMny_SQ;
    int DMnx = 1 + 2 * nloc[0], DMny = 1 + 2 * nloc[1];
    // node nd (i_dm, j_dm, k_dm) in local dmcomm_SQ domain, (i_dm_PR, j_dm_PR, k_dm_PR) in PR domain
    k_dm = nd / (DMnx_dm * DMny_dm);
    j_dm = (nd - k_dm * (DMnx_dm * DMny_dm)) / DMnx_dm;
    i_dm = nd % DMnx_dm;
    k_dm_PR = k_dm + nloc[2];
    j_dm_PR = j_dm + nloc[1];
    i_dm_PR = i_dm + nloc[0];
    double rcut = max(max(nloc[0]*pSPARC->delta_x, nloc[1]*pSPARC->delta_y), nloc[2]*pSPARC->delta_z), rcut2 = rcut * rcut;

    // pass 1: the selected rows of every image (indices in the image's PR node list), image after image in nlpt_pos
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int n_atom = Atom_Influence_nloc[ityp].n_atom, count_natom = 0;
        nlpt[ityp].nproj = nlocProj[ityp].nproj;
        aipt[ityp].n_atom = 0;
        if (n_atom == 0) continue;
        if (pSQ->nlpt_cap[ityp] < n_atom) {                                     // room for all images of the type
            aipt[ityp].coords = (double *) realloc(aipt[ityp].coords, (size_t)n_atom * 3 * sizeof(double));
            aipt[ityp].atom_index = (int *) realloc(aipt[ityp].atom_index, (size_t)n_atom * sizeof(int));
            aipt[ityp].ndc = (int *) realloc(aipt[ityp].ndc, (size_t)n_atom * sizeof(int));
            aipt[ityp].xs = (int *) realloc(aipt[ityp].xs, (size_t)n_atom * sizeof(int));
            aipt[ityp].grid_pos = (int **) realloc(aipt[ityp].grid_pos, (size_t)n_atom * sizeof(int *));
            nlpt[ityp].Chi = (double **) realloc(nlpt[ityp].Chi, (size_t)n_atom * sizeof(double *));
            assert(aipt[ityp].coords != NULL && aipt[ityp].atom_index != NULL && aipt[ityp].ndc != NULL
                && aipt[ityp].xs != NULL && aipt[ityp].grid_pos != NULL && nlpt[ityp].Chi != NULL);
            pSQ->nlpt_cap[ityp] = n_atom;
        }
        for (iat = 0; iat < n_atom; iat++) {
            int *pos_ = Atom_Influence_nloc[ityp].grid_pos[iat], *sel, count_ndc = 0;
            ndc_ = Atom_Influence_nloc[ityp].ndc[iat];
            if (npos + ndc_ > pSQ->nlpt_pos_cap) {
                pSQ->nlpt_pos_cap = max(2 * pSQ->nlpt_pos_cap, npos + ndc_);
                pSQ->nlpt_pos = (int *) realloc(pSQ->nlpt_pos, pSQ->nlpt_pos_cap * sizeof(int));
                assert(pSQ->nlpt_pos != NULL);
            }
            sel = pSQ->nlpt_pos + npos;
            for (ipos = 0; ipos < ndc_; ipos++) {
                indx = pos_[ipos];
                // w.r.t. PR domain 
                k_PR = indx / (DMnx_PR * DMny_PR);
                j_PR = (indx - k_PR * (DMnx_PR * DMny_PR)) / DMnx_PR;
                i_PR = indx % DMnx_PR;

                int flag = 0;
                double rx2, ry2, rz2;
                rz2 = (k_PR - k_dm_PR) * pSPARC->delta_z; rz2 *= rz2;
                ry2 = (j_PR - j_dm_PR) * pSPARC->delta_y; ry2 *= ry2;
                rx2 = (i_PR - i_dm_PR) * pSPARC->delta_x; rx2 *= rx2;
                flag = (int) ((rx2 + ry2 + rz2) > rcut2)                            // outside the node's Rcut sphere,
                    || abs(i_PR - i_dm_PR) > nloc[0] || abs(j_PR - j_dm_PR) > nloc[1] // or outside its box (possible
                    || abs(k_PR - k_dm_PR) > nloc[2];                                // when the mesh spacings differ)

                if (flag == 0) {
                    sel[count_ndc] = ipos;
                    count_ndc ++;            // this node is within current node's Rcut domain
                }
            }
            if (count_ndc == 0) continue;    // no intersection with current node's Rcut domain
            aipt[ityp].ndc[count_natom] = count_ndc;
            aipt[ityp].xs[count_natom] = iat;
            aipt[ityp].coords[count_natom*3] = Atom_Influence_nloc[ityp].coords[iat*3];
            aipt[ityp].coords[count_natom*3+1] = Atom_Influence_nloc[ityp].coords[iat*3+1];
            aipt[ityp].coords[count_natom*3+2] = Atom_Influence_nloc[ityp].coords[iat*3+2];
            aipt[ityp].atom_index[count_natom] = Atom_Influence_nloc[ityp].atom_index[iat];
            npos += count_ndc;
            nchi += (size_t)count_ndc * nlpt[ityp].nproj;
            count_natom ++;
        }
        aipt[ityp].n_atom = count_natom;
    }

    // pass 2: the Chi rows into nlpt_chi, the node positions relative to the point's Rcut box
    if (nchi > pSQ->nlpt_chi_cap) {
        pSQ->nlpt_chi_cap = max(2 * pSQ->nlpt_chi_cap, nchi);
        pSQ->nlpt_chi = (double *) realloc(pSQ->nlpt_chi, pSQ->nlpt_chi_cap * sizeof(double));
        assert(pSQ->nlpt_chi != NULL);
    }
    off = offc = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        nproj = nlpt[ityp].nproj;
        for (iat = 0; iat < aipt[ityp].n_atom; iat++) {
            int *sel = pSQ->nlpt_pos + off, *pos_;
            double *chi = pSQ->nlpt_chi + offc;
            const double *chi_;
            ndc = aipt[ityp].ndc[iat];
            iat_ = aipt[ityp].xs[iat];
            ndc_ = Atom_Influence_nloc[ityp].ndc[iat_];
            pos_ = Atom_Influence_nloc[ityp].grid_pos[iat_];
            chi_ = nlocProj[ityp].Chi[iat_];
            for (icol = 0; icol < nproj; icol++)
                for (ipos = 0; ipos < ndc; ipos++)
                    chi[ipos + icol*ndc] = chi_[sel[ipos] + icol*ndc_];
            for (ipos = 0; ipos < ndc; ipos++) {
                indx = pos_[sel[ipos]];
                // w.r.t. PR domain 
                k_PR = indx / (DMnx_PR * DMny_PR);
                j_PR = (indx - k_PR * (DMnx_PR * DMny_PR)) / DMnx_PR;
                i_PR = indx % DMnx_PR;
                // w.r.t. current node Rcut domain
                sel[ipos] = (i_PR - i_dm) + (j_PR - j_dm) * DMnx + (k_PR - k_dm) * DMnx * DMny;
            }
            aipt[ityp].grid_pos[iat] = sel;
            nlpt[ityp].Chi[iat] = chi;
            off += ndc;
            offc += (size_t)ndc * nproj;
        }
    }
}



/**
 * @brief   Calculate Vnl times vectors in a matrix-free way.
 */
void Vnl_vec_mult_SQ(const SPARC_OBJ *pSPARC, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
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
void Vnl_vec_mult_J_SQ(const SPARC_OBJ *pSPARC, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
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
void Vnl_vec_mult_dir_SQ(const SPARC_OBJ *pSPARC, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
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

