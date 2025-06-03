/**
 * @file occupationMatrix.c 
 * @brief This file contains the routines required to calculate the local occupation matrices and mixing them during the SCF
 * @author  Sayan Bhowmik <sbhowmik9@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Reference:   Dudarev, S. L., et al. "Electron-energy-loss spectra and the structural stability of nickel oxide:  An LSDA+U study"
 *              Phys. Rev. B 57, 1505
 */

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "occupationMatrix.h"
#include "isddft.h"
#include "tools.h"
#include "electronDensity.h"
#include "parallelization.h"
#include "readfiles.h"
#include "initialization.h"
#include "md.h"

#define TEMP_TOL 1e-12

/** 
 * @brief Refresh occupation matrix history at the start of every scf cycle (required for mixing)
 */
void RefreshOccMatHistory(SPARC_OBJ *pSPARC) {
    int atmcount, ityp, iat, angnum;
    atmcount = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        angnum = pSPARC->AtmU[ityp].angnum;

        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            memset(pSPARC->rho_mn_X[atmcount], 0, sizeof(double)* pSPARC->Nspinor*angnum*angnum*pSPARC->MixingHistory);
            memset(pSPARC->rho_mn_F[atmcount], 0, sizeof(double)* pSPARC->Nspinor*angnum*angnum*pSPARC->MixingHistory);

            atmcount++;
        }
    }
}

/**
 * @brief Initialize occupation matrix at the start of every scf cycle
 */
void init_occ_mat_scf(SPARC_OBJ *pSPARC) {
    int ityp, iat, angnum, atmcount, spinor, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if( (pSPARC->elecgs_Count - pSPARC->StressCount) == 0){
        // For the first step of scf re-initialize rho_mn
        atmcount = 0;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->atom_solve_flag[ityp]) {
                atmcount += pSPARC->nAtomv[ityp];
                continue;
            }

            angnum = pSPARC->AtmU[ityp].angnum;

            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {

                for (spinor = 0; spinor < pSPARC->Nspinor; spinor++){
                    assert(pSPARC->rho_mn_at[atmcount][spinor] != NULL);
                    memcpy(pSPARC->rho_mn[atmcount][spinor],pSPARC->rho_mn_at[atmcount][spinor],angnum*angnum*sizeof(double));
                }
                atmcount++;
            }
        }

        if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1){
            for(int i = 0; i < 3 * pSPARC->n_atom; i++) 
                pSPARC->atom_pos_0dt_occM[i] = pSPARC->atom_pos[i];
        }
    } else {
        if ( (pSPARC->elecgs_Count - pSPARC->StressCount) >= 3 && (pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1) ) { 
            // use extrapolation
            #ifdef DEBUG
                if(!rank) printf("Using occupation matrix extrapolation.\n");
            #endif

            atmcount = 0;
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                if(!pSPARC->atom_solve_flag[ityp]) {
                    atmcount += pSPARC->nAtomv[ityp];
                    continue;
                }

                angnum = pSPARC->AtmU[ityp].angnum;

                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                        for (int col = 0; col < angnum; col++) {
                            for (int row = 0; row < angnum; row++) {
                                pSPARC->rho_mn[atmcount][spinor][col * angnum + row] 
                                    = pSPARC->rho_mn_at[atmcount][spinor][col * angnum + row] 
                                    + pSPARC->drho_mn[atmcount][spinor][col * angnum + row];
                            }
                        }
                    }
                }
                atmcount++;
            }
        }
    }
}

/**
 * @brief Extrapolate occupation matrix for MD or RELAX_FLAG 1
 */
void occMatExtrapolation(SPARC_OBJ *pSPARC) {
    int ityp, iat, angnum, spinor, ii, matrank, count = 0, atm, atmcount;
    double alpha, beta, *coord_temp1, *coord_temp2;

    atmcount = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        angnum = pSPARC->AtmU[ityp].angnum;

        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                memcpy(pSPARC->drho_mn_2dt[atmcount][spinor], 
                    pSPARC->drho_mn_1dt[atmcount][spinor],angnum*angnum*sizeof(double));
                memcpy(pSPARC->drho_mn_1dt[atmcount][spinor], 
                    pSPARC->drho_mn_0dt[atmcount][spinor],angnum*angnum*sizeof(double));

                for (int col = 0; col < angnum; col++) {
                    for (int row = 0; row < angnum; row++) {
                        pSPARC->drho_mn_0dt[atmcount][spinor][col * angnum + row]
                            = pSPARC->rho_mn[atmcount][spinor][col * angnum + row]
                            - pSPARC->rho_mn_at[atmcount][spinor][col * angnum + row];
                    }
                }
            }
            atmcount++;
        }
    }

     if(pSPARC->MDFlag == 1){
        if(pSPARC->MDCount == 1) {
            for(atm = 0; atm < pSPARC->n_atom; atm++){
                pSPARC->atom_pos_nm_occM[count * 3] = pSPARC->atom_pos[count * 3];
                pSPARC->atom_pos_nm_occM[count * 3 + 1] = pSPARC->atom_pos[count * 3 + 1];
                pSPARC->atom_pos_nm_occM[count * 3 + 2] = pSPARC->atom_pos[count * 3 + 2];
                count ++;
            }   
        } else{
            coord_temp1 = (double *) malloc(3*pSPARC->n_atom*sizeof(double));
            coord_temp2 = (double *) malloc(3*pSPARC->n_atom*sizeof(double));

            for(atm = 0; atm < 3*pSPARC->n_atom; atm++){
                coord_temp1[atm] = pSPARC->atom_pos_nm_occM[atm];
                coord_temp2[atm] = pSPARC->atom_pos[atm];
            }
            wraparound_dynamics(pSPARC, coord_temp1, 0);
            if(pSPARC->cell_typ != 0){
                for(atm = 0; atm < pSPARC->n_atom; atm++){
                    Cart2nonCart_coord(pSPARC, coord_temp2+3*atm, coord_temp2+3*atm+1, coord_temp2+3*atm+2);
                    Cart2nonCart_coord(pSPARC, pSPARC->atom_pos_nm_occM+3*atm, pSPARC->atom_pos_nm_occM+3*atm+1, pSPARC->atom_pos_nm_occM+3*atm+2);
                }
            }
            for(atm = 0; atm < 3*pSPARC->n_atom; atm++){
                pSPARC->atom_pos_nm_occM[atm] += coord_temp2[atm] - coord_temp1[atm];
            }

            

            if(pSPARC->cell_typ != 0){
                for(atm = 0; atm < pSPARC->n_atom; atm++){
                    nonCart2Cart_coord(pSPARC, pSPARC->atom_pos_nm_occM+3*atm, pSPARC->atom_pos_nm_occM+3*atm+1, pSPARC->atom_pos_nm_occM+3*atm+2);
                }
            }

            free(coord_temp1);
            free(coord_temp2);
        }   
    } else if(pSPARC->RelaxFlag == 1){
        if((pSPARC->elecgs_Count - pSPARC->StressCount) == 1) {
            for(atm = 0; atm < pSPARC->n_atom; atm++){
                pSPARC->atom_pos_nm_occM[count * 3] = pSPARC->atom_pos[count * 3];
                pSPARC->atom_pos_nm_occM[count * 3 + 1] = pSPARC->atom_pos[count * 3 + 1];
                pSPARC->atom_pos_nm_occM[count * 3 + 2] = pSPARC->atom_pos[count * 3 + 2];
                count ++;
            }
        } else{
            coord_temp1 = (double *) malloc(3*pSPARC->n_atom*sizeof(double));
            coord_temp2 = (double *) malloc(3*pSPARC->n_atom*sizeof(double));

            for(atm = 0; atm < 3*pSPARC->n_atom; atm++){
                coord_temp1[atm] = pSPARC->atom_pos_nm_occM[atm];
                coord_temp2[atm] = pSPARC->atom_pos[atm];
            }
            wraparound_dynamics(pSPARC, coord_temp1, 0);
            if(pSPARC->cell_typ != 0){
                for(atm = 0; atm < pSPARC->n_atom; atm++){
                    Cart2nonCart_coord(pSPARC, coord_temp2+3*atm, coord_temp2+3*atm+1, coord_temp2+3*atm+2);
                    Cart2nonCart_coord(pSPARC, pSPARC->atom_pos_nm_occM+3*atm, pSPARC->atom_pos_nm_occM+3*atm+1, pSPARC->atom_pos_nm_occM+3*atm+2);
                }
            }
            for(atm = 0; atm < 3*pSPARC->n_atom; atm++){
                pSPARC->atom_pos_nm_occM[atm] += coord_temp2[atm] - coord_temp1[atm];
            }

            if(pSPARC->cell_typ != 0){
                for(atm = 0; atm < pSPARC->n_atom; atm++)
                    nonCart2Cart_coord(pSPARC, pSPARC->atom_pos_nm_occM+3*atm, pSPARC->atom_pos_nm_occM+3*atm+1, pSPARC->atom_pos_nm_occM+3*atm+2);
            }

            free(coord_temp1);
            free(coord_temp2);
        }   
    }

    if((pSPARC->elecgs_Count - pSPARC->StressCount) >= 3){ 
       double *FtF, *Ftf, *s, temp1, temp2, temp3; // 2x2 matrix and 2x1 vectors in (FtF)*(svec) = Ftf
        FtF = (double *)calloc( 4 , sizeof(double) );
        Ftf = (double *)calloc( 2 , sizeof(double) );
        s = (double *)calloc( 2 , sizeof(double) );
        for(ii = 0; ii < 3 * pSPARC->n_atom; ii++){
            temp1 = pSPARC->atom_pos_0dt_occM[ii] - pSPARC->atom_pos_1dt_occM[ii];
            temp2 = pSPARC->atom_pos_1dt_occM[ii] - pSPARC->atom_pos_2dt_occM[ii];
            temp3 = pSPARC->atom_pos_nm_occM[ii] - pSPARC->atom_pos_0dt_occM[ii];
            FtF[0] += temp1*temp1;
            FtF[1] += temp1*temp2;
            FtF[3] += temp2*temp2;
            Ftf[0] += temp1*temp3;
            Ftf[1] += temp2*temp3;
        }
        FtF[2] = FtF[1];
        // Find inv(FtF)*Ftf, LAPACKE_dgelsd stores the answer in Ftf vector
        LAPACKE_dgelsd(LAPACK_COL_MAJOR, 2, 2, 1, FtF, 2, Ftf, 2, s, -1.0, &matrank);
        alpha = Ftf[0];
        beta = Ftf[1];
        // Extrapolation 
        atmcount = 0;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->atom_solve_flag[ityp]) {
                atmcount += pSPARC->nAtomv[ityp];
                continue;
            }

            angnum = pSPARC->AtmU[ityp].angnum;
            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                    for (int col = 0; col < angnum; col++) {
                        for (int row = 0; row < angnum; row++) {
                            pSPARC->drho_mn[atmcount][spinor][col*angnum + row]
                                = (1 + alpha)*pSPARC->drho_mn_0dt[atmcount][spinor][col*angnum+row] 
                                + (beta - alpha)*pSPARC->drho_mn_1dt[atmcount][spinor][col*angnum+row]
                                - beta*pSPARC->drho_mn_2dt[atmcount][spinor][col*angnum+row];
                        }
                    }
                }
            }
        }

        free(FtF);
        free(Ftf);
        free(s);            
    }

    for(ii = 0; ii < 3 * pSPARC->n_atom; ii++){
        pSPARC->atom_pos_2dt_occM[ii] = pSPARC->atom_pos_1dt_occM[ii];
        pSPARC->atom_pos_1dt_occM[ii] = pSPARC->atom_pos_0dt_occM[ii];
        pSPARC->atom_pos_0dt_occM[ii] = pSPARC->atom_pos_nm_occM[ii];
    }
}

/**
 * @brief   Required for inner products. 
 *          If occupation matrices are stored in contiguous array for all atoms then this gives the start index of occupation matrix for each atom.
 */
void CalculateOccMatAtomIndex(SPARC_OBJ *pSPARC)
{
    int ityp, iat, l, lmax, atom_index, atom_index2, angnum, id;

    pSPARC->rho_mn_displ = (int *)malloc(sizeof(int) * (pSPARC->n_atom + 1));
    memset(pSPARC->rho_mn_displ, -1, sizeof(int) * (pSPARC->n_atom + 1));

    atom_index = atom_index2 = id = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atom_index += pSPARC->nAtomv[ityp];
            if (id == 0) atom_index2 = atom_index;
            continue;
        }

        lmax = pSPARC->AtmU[ityp].max_l;

        if (id == 0) {
            pSPARC->rho_mn_displ[atom_index] = 0;
            id++;
        }

        // number of projectors per atom (of this type)
        int nproj = 0;
        for (l = 0; l <= lmax; l++)
        {
            if (pSPARC->AtmU[ityp].U[l] == 0)
                continue;                                     // Skip for zero U values for a particular l orbital
            nproj += pSPARC->AtmU[ityp].ppl[l] * (2 * l + 1); // pSPARC->AtmU[ityp].ppl[l] > 1 if core states are included
        }

        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++)
        {
            if (iat == 0 && id !=0) {
                pSPARC->rho_mn_displ[atom_index] = pSPARC->rho_mn_displ[atom_index2];
            } 
            pSPARC->rho_mn_displ[atom_index + 1] = pSPARC->rho_mn_displ[atom_index] + nproj*nproj;
            atom_index++;
        }
        atom_index2 = atom_index;
    }
}

/**
 * @brief Distribute based on gamma or k-points.
 */
void Calculate_Occupation_Matrix(SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj) {
    if (pSPARC->isGammaPoint) {
        Calculate_occMat(pSPARC, Atom_Influence_loc, locProj);
    } else {
        Calculate_occMat_kpt(pSPARC, Atom_Influence_loc, locProj);
    }
}

/**
 * @brief Calculate occupation matrix at every scf iteration for gamma point
 *        \Sum_n g_n < phi_m | psi_n > < psi_n | phi_m > , m x m matrix per atom
 */
void Calculate_occMat(SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj)
{

    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int i, n, k, Ns, count, nstart, nend, spinor, DMnd;
    double **g_n;
    DMnd = pSPARC->Nd_d_dmcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    Ns = pSPARC->Nstates;
    int Nspinor = pSPARC->Nspinor;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ncol = nend - nstart + 1;
    int Nkpts = pSPARC->Nkpts_kptcomm;

    // Extract local g_n as array
    g_n = (double **)calloc((pSPARC->Nspinor_spincomm), sizeof(double*));
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        g_n[spinor] = (double *)calloc(ncol, sizeof(double));
    }
    
 
    count = 0;
    for (n = nstart; n <= nend; n++) {
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
            int spinor_g = spinor + pSPARC->spinor_start_indx;
            double *occ = pSPARC->occ; 
            if (pSPARC->spin_typ == 1) occ += spinor*Ns;
            g_n[spinor][count] = occ[n];
        }
        count++;
    }

    // Extract local bands on local domain
    double **X = (double **)calloc((pSPARC->Nspinor), sizeof(double*));
    if (pSPARC->Nspinor_spincomm > 1) {
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
            X[spinor] = (double *)calloc(ncol * DMnd, sizeof(double));
        }
    } else {
        spinor = pSPARC->spincomm_index;
        X[spinor] = (double *)calloc(ncol * DMnd, sizeof(double));
    }

    if (pSPARC->Nspinor_spincomm > 1) {
        count = 0;
        
        for (n = 0; n < ncol; n++) {
            for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
                for (i = 0; i < DMnd; i++) {
                    X[spinor][n*DMnd + i] = pSPARC->Xorb[count++];
                }
            }
        }
        
    } else {
        spinor = pSPARC->spincomm_index;
        count = 0;
        
        for (n = 0; n < ncol; n++) {
            for (i = 0; i < DMnd; i++) {
                X[spinor][n*DMnd + i] = pSPARC->Xorb[count++];
            }
        }
        
    }

    // calculate local rho_mn
    int n_atom = pSPARC->n_atom;
    int atm_idx;
    for (int JJ = n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    double **alpha = (double **)calloc((pSPARC->Nspinor), sizeof(double*));
    double **alpha_gn = (double **)calloc((pSPARC->Nspinor), sizeof(double*));
    double **rho_mn = (double **)calloc((pSPARC->Nspinor), sizeof(double*));
    int sp, ndc, atom_index;
    double *x_rc;

    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (pSPARC->Nspinor_spincomm == 1) {
            sp = pSPARC->spincomm_index;
        } else {
            sp = spinor;
        }
        alpha[sp] = (double *)calloc(pSPARC->IP_displ_U[atm_idx] * ncol, sizeof(double));
        alpha_gn[sp] = (double *)calloc(pSPARC->IP_displ_U[atm_idx] * ncol, sizeof(double));
    }

    for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
        rho_mn[spinor] = (double *)calloc(pSPARC->rho_mn_displ[atm_idx], sizeof(double));
    }

    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) {
            continue;
        }

        if (!locProj[ityp].nproj)
            continue;

        for (int iat = 0; iat < Atom_Influence_loc[ityp].n_atom; iat++)
        {
            for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
                if (pSPARC->Nspinor_spincomm == 1) {
                    sp = pSPARC->spincomm_index;
                } else {
                    sp = spinor;
                }

                ndc = Atom_Influence_loc[ityp].ndc[iat];
                x_rc = (double *)malloc(ndc * ncol * sizeof(double));
                atom_index = Atom_Influence_loc[ityp].atom_index[iat];
                for (n = 0; n < ncol; n++)
                {
                    for (i = 0; i < ndc; i++)
                    {
                        x_rc[n * ndc + i] = X[sp][n * DMnd + Atom_Influence_loc[ityp].grid_pos[iat][i]];
                        x_rc[n * ndc + i] /= sqrt(pSPARC->dV);
                    }
                }
                // < Orb_Jlm, x_n > for each J (size: m x Ns_loc)
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, locProj[ityp].nproj, ncol, ndc,
                            pSPARC->dV, locProj[ityp].Orb[iat], ndc, x_rc, ndc, 1.0,
                            alpha[sp] + pSPARC->IP_displ_U[atom_index] * ncol, locProj[ityp].nproj);
                free(x_rc);
            }

        }
    }   
    
    // if there are domain parallelization over each band then sum over all processes over dmcomm
    // < Orb_Jlm, x_n > but reduced over domain
    int commsize;
    MPI_Comm_size(pSPARC->dmcomm, &commsize);
    if (commsize > 1) {
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
            if (pSPARC->Nspinor_spincomm == 1) {
                sp = pSPARC->spincomm_index;
            } else {
                sp = spinor;
            }
            MPI_Allreduce(MPI_IN_PLACE, alpha[sp], pSPARC->IP_displ_U[atm_idx] * ncol, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
        }
    }
    

    int angnum;
    int atmcount = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        if (!locProj[ityp].nproj) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        angnum = pSPARC->AtmU[ityp].angnum;
        for (int iat = 0; iat < pSPARC->nAtomv[ityp]; iat++)
        {
            for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
                if (pSPARC->Nspinor_spincomm == 1) {
                    sp = pSPARC->spincomm_index;
                } else {
                    sp = spinor;
                }

                // Scale each column of < Orb_Jlm, x_n > with g_n (size: m x Ns_loc)
                cblas_dcopy(locProj[ityp].nproj * ncol, alpha[sp] + pSPARC->IP_displ_U[atmcount] * ncol, 1, 
                            alpha_gn[sp] + pSPARC->IP_displ_U[atmcount] * ncol, 1); // copy into alpha_gn
                

                for (n = 0; n < ncol; n++) { // scale each column
                    cblas_dscal(locProj[ityp].nproj, g_n[spinor][n], 
                                alpha_gn[sp] + pSPARC->IP_displ_U[atmcount] * ncol + n*locProj[ityp].nproj, 1);
                }

                // g_n \times < Orb_Jlm, x_n > \times < x_n, Orb_Jlm >
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, locProj[ityp].nproj, locProj[ityp].nproj, ncol,
                    1.0, alpha_gn[sp] + pSPARC->IP_displ_U[atmcount] * ncol, locProj[ityp].nproj,
                    alpha[sp] + pSPARC->IP_displ_U[atmcount] * ncol, locProj[ityp].nproj, 0.0,
                    rho_mn[sp] + pSPARC->rho_mn_displ[atmcount], locProj[ityp].nproj);
            }
            atmcount++; 
        }
    }

    // reduce over bands
    if (pSPARC->npband) {
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
            if (pSPARC->Nspinor_spincomm == 1) {
                sp = pSPARC->spincomm_index;
            } else {
                sp = spinor;
            }
            MPI_Allreduce(MPI_IN_PLACE, rho_mn[sp], pSPARC->rho_mn_displ[atm_idx], MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        }
    }

    // Sendrecv rho_mn across spin bridge comm if there is spin paral
    if (pSPARC->npspin > 1) {
        int rank_sp_bridge;
        MPI_Comm_rank(pSPARC->spin_bridge_comm, &rank_sp_bridge);
        int lenbuf;
        if (rank_sp_bridge == 0) {
            lenbuf = pSPARC->rho_mn_displ[atm_idx];
            MPI_Sendrecv(rho_mn[0], lenbuf, MPI_DOUBLE, 1, 0,
                         rho_mn[1], lenbuf, MPI_DOUBLE, 1, 0,
                         pSPARC->spin_bridge_comm, MPI_STATUS_IGNORE);
        } else if (rank_sp_bridge == 1) {
            lenbuf = pSPARC->rho_mn_displ[atm_idx];
            MPI_Sendrecv(rho_mn[1], lenbuf, MPI_DOUBLE, 0, 0,
                         rho_mn[0], lenbuf, MPI_DOUBLE, 0, 0,
                         pSPARC->spin_bridge_comm, MPI_STATUS_IGNORE);
        }
    }

    // store rho_mn in pSPARC->rho_mn
    atmcount = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        angnum = pSPARC->AtmU[ityp].angnum;
        for (int iat = 0; iat < pSPARC->nAtomv[ityp]; iat++)
        {
            for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                memcpy(pSPARC->rho_mn[atmcount][spinor], 
                    rho_mn[spinor] + pSPARC->rho_mn_displ[atmcount], 
                    angnum*angnum*sizeof(double));

                // Make rho_mn hermitean
                for (int c = 0; c < angnum; c++) {
                    for (int r = c; r < angnum; r++) {
                        pSPARC->rho_mn[atmcount][spinor][c * angnum + r] = pSPARC->rho_mn[atmcount][spinor][r * angnum + c];
                    }
                }
            }
            atmcount++;
        }
    }   


    // free memory
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (pSPARC->Nspinor_spincomm == 1) {
            sp = pSPARC->spincomm_index;
        } else {
            sp = spinor;
        }
        if (g_n[spinor] != NULL) free(g_n[spinor]);

        if (alpha[sp] != NULL) free(alpha[sp]);

        if (alpha_gn[sp] != NULL) free(alpha_gn[sp]);

        if (X[sp] != NULL) free(X[sp]);
 
    }

    free(g_n); free(X); free(alpha); free(alpha_gn);

    for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
        if (rho_mn[spinor] != NULL) free(rho_mn[spinor]);
    }
    free(rho_mn);
}

/**
 * @brief Calculate occupation matrix at every scf iteration for k-point
 *        \Sum_k w_k \Sum_n g_n < phi_m | psi_nk > < psi_nk | phi_m > , m x m matrix per atom
 */
void Calculate_occMat_kpt(SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj) {
    
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int i, n, k, Ns, count, nstart, nend, spinor, DMnd;
    double x0_i, y0_i, z0_i, theta, k1, k2, k3;
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;

    double **g_nk;
    DMnd = pSPARC->Nd_d_dmcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    Ns = pSPARC->Nstates;
    int Nspinor = pSPARC->Nspinor;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ncol = nend - nstart + 1;
    int Nkpts = pSPARC->Nkpts_kptcomm;

    // Extract local g_n as array
    g_nk = (double **)calloc((pSPARC->Nspinor_spincomm), sizeof(double*));
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        g_nk[spinor] = (double *)calloc(Nkpts*ncol, sizeof(double));
    }
    
 
    count = 0;
    for (k = 0; k < Nkpts; k++) {
        for (n = nstart; n <= nend; n++) {
            double woccfac = (pSPARC->kptWts_loc[k] / pSPARC->Nkpts);
            for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
                double *occ = pSPARC->occ + k*Ns; 
                if (pSPARC->spin_typ == 1) occ += spinor*Ns*Nkpts;
                g_nk[spinor][count] = woccfac * occ[n];
            }
            count++;
        }
    }

    // Extract local bands on local domain
    double _Complex **X = (double _Complex **)calloc((pSPARC->Nspinor), sizeof(double _Complex *));
    if (pSPARC->Nspinor_spincomm > 1) {
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
            X[spinor] = (double _Complex *)calloc(Nkpts * ncol * DMnd, sizeof(double _Complex));
        }
    } else {
        spinor = pSPARC->spincomm_index;
        X[spinor] = (double _Complex *)calloc(Nkpts * ncol * DMnd, sizeof(double _Complex));
    }

    if (pSPARC->Nspinor_spincomm > 1) {
        count = 0;
        
        for (k = 0; k < Nkpts; k++){
            for (n = 0; n < ncol; n++) {
                for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
                    for (i = 0; i < DMnd; i++) {
                        X[spinor][k*ncol*DMnd + n*DMnd + i] = pSPARC->Xorb_kpt[count++];
                    }
                }
            }
        }
        
    } else {
        spinor = pSPARC->spincomm_index;
        count = 0;
        
        for (k = 0; k < Nkpts; k++){
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < DMnd; i++) {
                    X[spinor][k*ncol*DMnd + n*DMnd + i] = pSPARC->Xorb_kpt[count++];
                }
            }
        }
        
    }

    // calculate local rho_mn
    int n_atom = pSPARC->n_atom;
    int atm_idx;
    for (int JJ = n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    double _Complex **alpha = (double _Complex **)calloc((pSPARC->Nspinor), sizeof(double _Complex *));
    double _Complex **alpha_gnk = (double _Complex **)calloc((pSPARC->Nspinor), sizeof(double _Complex *));
    double _Complex **rho_mn = (double _Complex **)calloc((pSPARC->Nspinor), sizeof(double _Complex *));
    int sp, ndc, atom_index;
    double _Complex *x_rc;
    double _Complex a, b, bloch_fac;

    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (pSPARC->Nspinor_spincomm == 1) {
            sp = pSPARC->spincomm_index;
        } else {
            sp = spinor;
        }
        alpha[sp] = (double _Complex *)calloc(pSPARC->IP_displ_U[atm_idx] * ncol, sizeof(double _Complex));
        alpha_gnk[sp] = (double _Complex *)calloc(pSPARC->IP_displ_U[atm_idx] * ncol, sizeof(double _Complex));
    }

    for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
        rho_mn[spinor] = (double _Complex *)calloc(pSPARC->rho_mn_displ[atm_idx], sizeof(double _Complex));
    }

    for (k = 0; k < Nkpts; k++) {
        k1 = pSPARC->k1_loc[k];
        k2 = pSPARC->k2_loc[k];
        k3 = pSPARC->k3_loc[k];
        // Set memory of alpha and alpha_gnk to 0
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
            if (pSPARC->Nspinor_spincomm == 1) {
                sp = pSPARC->spincomm_index;
            } else {
                sp = spinor;
            }
            memset(alpha[sp],0,pSPARC->IP_displ_U[atm_idx] * ncol*sizeof(double _Complex));
            memset(alpha_gnk[sp],0,pSPARC->IP_displ_U[atm_idx] * ncol*sizeof(double _Complex));

        }

        for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->atom_solve_flag[ityp]) {
                continue;
            }

            if (!locProj[ityp].nproj)
                continue;

            for (int iat = 0; iat < Atom_Influence_loc[ityp].n_atom; iat++)
            {
                for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
                    if (pSPARC->Nspinor_spincomm == 1) {
                        sp = pSPARC->spincomm_index;
                    } else {
                        sp = spinor;
                    }

                    x0_i = Atom_Influence_loc[ityp].coords[iat*3  ];
                    y0_i = Atom_Influence_loc[ityp].coords[iat*3+1];
                    z0_i = Atom_Influence_loc[ityp].coords[iat*3+2];
                    theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
                    bloch_fac = cos(theta) + sin(theta) * I;

                    // theta = k1 * (floor(x0_i/Lx) * Lx) + k2 * (floor(y0_i/Ly) * Ly) + k3 * (floor(z0_i/Lz) * Lz);
                    // bloch_fac = cos(theta) - sin(theta) * I;

                    a = bloch_fac * pSPARC->dV;
                    b = 1.0;

                    ndc = Atom_Influence_loc[ityp].ndc[iat];
                    x_rc = (double _Complex *)malloc(ndc * ncol * sizeof(double _Complex));
                    atom_index = Atom_Influence_loc[ityp].atom_index[iat];
                    for (n = 0; n < ncol; n++)
                    {
                        for (i = 0; i < ndc; i++)
                        {
                            x_rc[n * ndc + i] = X[sp][k*ncol*DMnd + n * DMnd + Atom_Influence_loc[ityp].grid_pos[iat][i]];
                            x_rc[n * ndc + i] /= sqrt(pSPARC->dV);
                        }
                    }
                    // < Orb_Jlm, x_n > for each J (size: m x Ns_loc)
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, locProj[ityp].nproj, ncol, ndc,
                                &a, locProj[ityp].Orb_c[iat], ndc, x_rc, ndc, &b,
                                alpha[sp] + pSPARC->IP_displ_U[atom_index] * ncol, locProj[ityp].nproj);
                    free(x_rc);
                }

            }
        }

        // if there are domain parallelization over each band then sum over all processes over dmcomm
        // < Orb_Jlm, x_n > but reduced over domain
        int commsize;
        MPI_Comm_size(pSPARC->dmcomm, &commsize);
        if (commsize > 1) {
            for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
                if (pSPARC->Nspinor_spincomm == 1) {
                    sp = pSPARC->spincomm_index;
                } else {
                    sp = spinor;
                }
                MPI_Allreduce(MPI_IN_PLACE, alpha[sp], pSPARC->IP_displ_U[atm_idx] * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
            }
        }

        int angnum;
        int atmcount = 0;
        for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->atom_solve_flag[ityp]) {
                atmcount += pSPARC->nAtomv[ityp];
                continue;
            }

            if (!locProj[ityp].nproj) {
                atmcount += pSPARC->nAtomv[ityp];
                continue;
            }

            // angnum = pSPARC->AtmU[ityp].angnum;
            angnum = locProj[ityp].nproj;
            for (int iat = 0; iat < pSPARC->nAtomv[ityp]; iat++)
            {

                for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
                    if (pSPARC->Nspinor_spincomm == 1) {
                        sp = pSPARC->spincomm_index;
                    } else {
                        sp = spinor;
                    }

                    // Scale each column of < Orb_Jlm, x_n > with g_n (size: m x Ns_loc)
                    cblas_zcopy(locProj[ityp].nproj * ncol, alpha[sp] + pSPARC->IP_displ_U[atmcount] * ncol, 1, 
                                alpha_gnk[sp] + pSPARC->IP_displ_U[atmcount] * ncol, 1); // copy into alpha_gn
                    

                    for (n = 0; n < ncol; n++) { // scale each column
                        double _Complex scale_f = g_nk[spinor][k*ncol + n];
                        cblas_zscal(locProj[ityp].nproj, &scale_f, 
                                    alpha_gnk[sp] + pSPARC->IP_displ_U[atmcount] * ncol + n*locProj[ityp].nproj, 1);
                    }

                    // g_n \times < Orb_Jlm, x_n > \times < x_n, Orb_Jlm >
                    a = 1.0; b = 1.0;
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, locProj[ityp].nproj, locProj[ityp].nproj, ncol,
                        &a, alpha_gnk[sp] + pSPARC->IP_displ_U[atmcount] * ncol, locProj[ityp].nproj,
                        alpha[sp] + pSPARC->IP_displ_U[atmcount] * ncol, locProj[ityp].nproj, &b,
                        rho_mn[sp] + pSPARC->rho_mn_displ[atmcount], locProj[ityp].nproj);
                }
                atmcount++; 
            }
        }
    }

    // reduce over k-point group
    if (pSPARC->npkpt > 1){
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
            if (pSPARC->Nspinor_spincomm == 1) {
                sp = pSPARC->spincomm_index;
            } else {
                sp = spinor;
            }
            MPI_Allreduce(MPI_IN_PLACE, rho_mn[sp], pSPARC->rho_mn_displ[atm_idx], MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->kpt_bridge_comm);
        }
    }

    // reduce over bands
    if (pSPARC->npband) {
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
            if (pSPARC->Nspinor_spincomm == 1) {
                sp = pSPARC->spincomm_index;
            } else {
                sp = spinor;
            }
            MPI_Allreduce(MPI_IN_PLACE, rho_mn[sp], pSPARC->rho_mn_displ[atm_idx], MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->blacscomm);
        }
    }

    // Sendrecv rho_mn across spin bridge comm if there is spin paral
    if (pSPARC->npspin > 1) {
        int rank_sp_bridge;
        MPI_Comm_rank(pSPARC->spin_bridge_comm, &rank_sp_bridge);
        int lenbuf;
        if (rank_sp_bridge == 0) {
            lenbuf = pSPARC->rho_mn_displ[atm_idx];
            MPI_Sendrecv(rho_mn[0], lenbuf, MPI_DOUBLE_COMPLEX, 1, 0,
                         rho_mn[1], lenbuf, MPI_DOUBLE_COMPLEX, 1, 0,
                         pSPARC->spin_bridge_comm, MPI_STATUS_IGNORE);
        } else if (rank_sp_bridge == 1) {
            lenbuf = pSPARC->rho_mn_displ[atm_idx];
            MPI_Sendrecv(rho_mn[1], lenbuf, MPI_DOUBLE_COMPLEX, 0, 0,
                         rho_mn[0], lenbuf, MPI_DOUBLE_COMPLEX, 0, 0,
                         pSPARC->spin_bridge_comm, MPI_STATUS_IGNORE);
        }
    }

    // New hermitean logic
    int atmcount = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        int angnum = locProj[ityp].nproj;

        for (int iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                double _Complex *rho_block = rho_mn[spinor] + pSPARC->rho_mn_displ[atmcount];

                // Make rho_mn[spinor] Hermitian in-place
                for (int c = 0; c < angnum; c++) {
                    for (int r = c + 1; r < angnum; r++) {
                        double _Complex avg = 0.5 * (rho_block[c * angnum + r] + conj(rho_block[r * angnum + c]));
                        rho_block[c * angnum + r] = avg;
                        rho_block[r * angnum + c] = conj(avg);
                    }
                    // Diagonal elements must be real
                    rho_block[c * angnum + c] = creal(rho_block[c * angnum + c]);
                }

                // Now copy the real parts into pSPARC->rho_mn
                for (int c = 0; c < angnum; c++) {
                    for (int r = 0; r < angnum; r++) {
                        pSPARC->rho_mn[atmcount][spinor][c * angnum + r] = creal(rho_block[c * angnum + r]);
                    }
                }
            }
            atmcount++;
        }
    }


    // free memory
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (pSPARC->Nspinor_spincomm == 1) {
            sp = pSPARC->spincomm_index;
        } else {
            sp = spinor;
        }
        if (g_nk[spinor] != NULL) free(g_nk[spinor]);

        if (alpha[sp] != NULL) free(alpha[sp]);

        if (alpha_gnk[sp] != NULL) free(alpha_gnk[sp]);

        if (X[sp] != NULL) free(X[sp]);
 
    }

    free(g_nk); free(X); free(alpha); free(alpha_gnk);

    for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
        if (rho_mn[spinor] != NULL) free(rho_mn[spinor]);
    }
    free(rho_mn);
}


/**
* @brief Performs mixing of the occupation matrices for each atom
*/
void mixing_rho_mn(SPARC_OBJ *pSPARC, int iter_count)
{
#define R(i,j) (*(R+(j)*N+(i)))
#define F(i,j) (*(F+(j)*N+(i)))

    if (pSPARC->MixingVariable != 0) {
        printf("Hubbard DFT works only with density mixing currently!\n");
        exit(EXIT_FAILURE);
    }

    int m = pSPARC->MixingHistory;
    int p = pSPARC->PulayFrequency;
    double beta = pSPARC->MixingParameter;
    double omega = pSPARC->MixingParameterSimple;

    // First broadcast the mix_Gamma into psi-domain
    communicate_mix_gamma(pSPARC);

    // flag for Pulay (Anderson extrapolation) mixing, otherwise apply simple mixing 
    int Pulay_mixing_flag = (int) ((iter_count+1) % p == 0 && iter_count > 0);

    // decide which mixing parameter to use
    double amix;
    if (Pulay_mixing_flag) { // pulay mixing
        amix = beta; 
    } else { // simple (linear) mixing
        amix = omega; 
    }

    // loop over all atom types
    int atmcount = 0;
    int angnum, N, i_hist;
    double *g_k, *x_kp1;
    double *x_k, *x_km1, *f_k, *f_km1, *R, *F, *Pf;
    double *f_wavg, *x_wavg;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if(!pSPARC->atom_solve_flag[ityp]) { // if no hubbard correction then continue
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        angnum = pSPARC->AtmU[ityp].angnum;
        N = angnum*angnum*pSPARC->Nspinor;
        g_k = (double *)malloc(N*sizeof(double));
        x_kp1 = (double *)malloc(N*sizeof(double));
        x_wavg = (double *)malloc(N*sizeof(double));
        f_wavg = (double *)malloc(N*sizeof(double));
        Pf = (double *)malloc(N*sizeof(double));

        // loop over all atom of type ityp
        for (int iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            for (int spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                memcpy(g_k + spinor*angnum*angnum, pSPARC->rho_mn[atmcount][spinor], 
                       angnum*angnum*sizeof(double));
                memcpy(x_kp1 + spinor*angnum*angnum, pSPARC->rho_mn[atmcount][spinor], 
                        angnum*angnum*sizeof(double));
            }

            x_k = pSPARC->rho_mn_xk[atmcount];          // the current mixed var x (x^{in}_k)
            x_km1 = pSPARC->rho_mn_xkm1[atmcount];      // x_{k-1}
            f_k = pSPARC->rho_mn_fk[atmcount];          // f_k = g(x_k) - x_k
            f_km1 = pSPARC->rho_mn_fkm1[atmcount];      // f_{k-1}
            R = pSPARC->rho_mn_X[atmcount];             // [x_{k-m+1} - x_{k-m}, ... , x_k - x_{k-1}]
            F = pSPARC->rho_mn_F[atmcount];             // [f_{k-m+1} - f_{k-m}, ... , f_k - f_{k-1}]

            // update old residual f_{k-1}
            if (iter_count > 0) {
                for (int i = 0; i < N; i++) f_km1[i] = f_k[i];
            }

            // compute current residual     
            for (int i = 0; i < N; i++) f_k[i] = g_k[i] - x_k[i];

            // store residual & iteration history //
            if (iter_count > 0) {
                i_hist = (iter_count - 1) % m;
                if (pSPARC->PulayRestartFlag && i_hist == 0) {
                    for (int i = 0; i < N*(m-1); i++) {
                        R[N+i] = F[N+i] = 0.0; // set all cols to 0 (except for 1st col)
                    }

                    // set 1st cols of R and F
                    for (int i = 0; i < N; i++) {
                        R[i] = x_k[i] - x_km1[i];
                        F[i] = f_k[i] - f_km1[i];
                    }
                } else {
                    for (int i = 0; i < N; i++) {
                        R(i, i_hist) = x_k[i] - x_km1[i];
                        F(i, i_hist) = f_k[i] - f_km1[i];
                    }
                }
            }

            if (Pulay_mixing_flag) {
                // Anderson extrapolation: Using same Gamma as from density mixing
                // find weighted average x_{k+1} = x_k - X*Gamma
                for (int i = 0; i < N; i++) x_wavg[i] = x_k[i];
                cblas_dgemv(CblasColMajor, CblasNoTrans, N, m, -1.0, R,
                            N, pSPARC->mix_Gamma, 1, 1.0, x_wavg, 1);
                
                // find weighted average f_{k+1} = f_k - F*Gamma
                for (int i = 0; i < N; i++) f_wavg[i] = f_k[i];
                cblas_dgemv(CblasColMajor, CblasNoTrans, N, m, -1.0, F,
                            N, pSPARC->mix_Gamma, 1, 1.0, f_wavg, 1);
            } else {
                // Simple (linear) extrapolation
                for (int i = 0; i < N; i++) {
                    x_wavg[i] = x_k[i];
                    f_wavg[i] = f_k[i];
                }
            }

            for (int i = 0; i < N; i++) {
                Pf[i] = amix * f_wavg[i];
            }

            // find x_{k+1} := x_wavg + Pf (mixing param is included in Pf)
            for (int i = 0; i < N; i++) x_kp1[i] = x_wavg[i] + Pf[i];

            // Copy x_kp1 back into original rho_mn
            for (int spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                memcpy(pSPARC->rho_mn[atmcount][spinor], x_kp1 + spinor*angnum*angnum,
                        angnum*angnum*sizeof(double));

                // Make rho_mn hermitean
                for (int c = 0; c < angnum; c++) {
                    for (int r = c; r < angnum; r++) {
                        pSPARC->rho_mn[atmcount][spinor][c * angnum + r] = pSPARC->rho_mn[atmcount][spinor][r * angnum + c];
                    }
                }
            }

            // Update x_km1 and x_k
            for (int i = 0; i < N; i++) {
                x_km1[i] = x_k[i];
                x_k[i] = x_kp1[i];
            }

            atmcount++;
        } // end of loop for all atoms of type ityp
        
        // free variables
        free(g_k); free(x_kp1); free(x_wavg); free(f_wavg); free(Pf);
    }
#undef R
#undef F
}

/**
 * @brief Broadcast mix_Gamma when phi_domain and psi_domain are different.
 */
void communicate_mix_gamma(SPARC_OBJ *pSPARC) {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    int is_in_dmcomm_phi = (pSPARC->dmcomm_phi != MPI_COMM_NULL);
    int dm_rank = -1; // local rank in dmcomm_phi if inside
    int sender_rank_world = 0; // Root rank in MPI_COMM-WORLD

    if (is_in_dmcomm_phi) {
        MPI_Comm_rank(pSPARC->dmcomm_phi, &dm_rank);
        if (dm_rank == 0) {
            sender_rank_world = global_rank; // Set the sender to global rank of rank 0 in dmcomm_phi
        }
    }

    // Broadcast the sender's world rank to all processes
    MPI_Bcast(&sender_rank_world, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Now, let the sender broadcast mix_gamma
    MPI_Bcast(pSPARC->mix_Gamma, pSPARC->MixingHistory, MPI_DOUBLE, sender_rank_world, MPI_COMM_WORLD);
}

/**
 * @brief Print occupation matrix for each atom and its eigen-decomposition.
 */
void print_Occ_mat(SPARC_OBJ *pSPARC) {
    double *rho_mn, *eval, *evec, *eval_imag, *evec_left;
    int atmcount, atmcount2, angnum;
    atmcount = atmcount2 = 0;
    double trace;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        angnum = pSPARC->AtmU[ityp].angnum;
        eval = (double *)calloc(angnum, sizeof(double));
        evec = (double *)calloc(angnum*angnum, sizeof(double));
        eval_imag = (double *)calloc(angnum, sizeof(double));
        rho_mn = (double *)malloc(angnum*angnum * sizeof(double));
        evec_left = NULL;

        for (int iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            printf("-----------------Atom %d-----------------\n",atmcount2+1);
            for (int spinor  = 0; spinor < pSPARC->Nspinor; spinor++) {
                for (int c = 0; c < angnum; c++) {
                    for (int r = 0; r < angnum; r++) {
                        rho_mn[c*angnum + r] = pSPARC->rho_mn[atmcount][spinor][c*angnum + r];
                    }
                }

                if (pSPARC->Nspin > 1) printf("SPIN %d\n",spinor);

                // Print eigenvalues, total local occupation of the atom, eigenvectors
                trace = 0.0;
                for (int m = 0; m < angnum; m++) trace += rho_mn[m * angnum + m];
                printf("Trace[rho_mn]: %6.6f\n",pSPARC->occfac*trace);

                int info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', angnum, rho_mn, angnum,
                                        eval, eval_imag, evec_left, angnum, evec, angnum);
                printf("Eigenvalues:\n");
                for (int JJ = 0; JJ < angnum; JJ++) {
                    printf("% 6.6f    ",eval[JJ]);
                }
                
                printf("\nEigenvectors:\n");
                for (int r = 0 ; r < angnum; r++) {
                    for (int c = 0; c < angnum; c++) {
                        printf("% 6.6f    ", evec[c * angnum + r]);
                    }
                    printf("\n");
                }

                printf("Occupation matrix:\n");
                for (int r = 0; r < angnum; r++) {
                    for (int c = 0; c < angnum; c++) {
                        printf("% 6.6f    ", pSPARC->rho_mn[atmcount][spinor][c * angnum + r]);
                    }
                    printf("\n");
                }
            }
            atmcount++; atmcount2++;
        }

        free(eval); free(eval_imag); free(evec); free(rho_mn);
    }
}