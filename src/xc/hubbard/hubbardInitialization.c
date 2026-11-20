/**
 * @file hubbardinitialization.c 
 * @brief This file contains initialization routines required for DFT+U.
 * @author  Sayan Bhowmik <sbhowmik9@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <assert.h>
// this is for checking existence of files
# include <unistd.h>
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "hubbardInitialization.h"
#include "isddft.h"
#include "tools.h"
#include "parallelization.h"

#define TEMP_TOL 1e-12

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

#define N_MEMBR 241

/**
* @brief Broadcast atom solution info in SPARC using MPI_Pack & MPI_Unpack
*/
void bcast_SPARC_Atom_Soln(SPARC_OBJ *pSPARC) {
    int i, l, rank, position, l_buff, Ntypes, atmcount, atmcount2, nproj, lmax_sum, size_sum, nprojsize_sum, nproj_sum;
    int *tempbuff, *lmaxv, *sizev, *nspinorv, *vallenv, *pplv, *ppl_sdispl;
    char *buff;
    double *Uarr;

#ifdef DEBUG
    double t1, t2;
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Ntypes = pSPARC->Ntypes;

    /* Broadcast the atom solve flag */
    if (rank != 0) {
        pSPARC->atom_solve_flag = (int *)malloc(Ntypes * sizeof(int));
    }
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    MPI_Bcast(pSPARC->atom_solve_flag, Ntypes, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Bcast atom solve flag took %0.3f ms\n",(t2-t1)*1000);
#endif

    /* Broadcasting structure */
    atmcount = 0;
    for (i = 0; i < Ntypes; i++) {
        if (pSPARC->atom_solve_flag[i]) atmcount++;
    }
    lmaxv = (int *)malloc(atmcount*sizeof(int));
    sizev = (int *)malloc(atmcount*sizeof(int));
    nspinorv = (int *)malloc(atmcount*sizeof(int));
    vallenv = (int *)malloc(atmcount*sizeof(int));
    Uarr = (double *)malloc(5*atmcount*sizeof(double)); // per type: U of s, p, d, f and U_RCUT
    ppl_sdispl = (int *)malloc((atmcount+1)*sizeof(int));
    tempbuff = (int *)malloc((4*atmcount)*sizeof(int));
    assert(lmaxv != NULL && sizev != NULL && ppl_sdispl != NULL && tempbuff != NULL);

    // send lmax, size
    if (rank == 0) {
        // pack info into temp buffer
        atmcount2 = 0;
        for (i = 0; i < Ntypes; i++) {
            if (pSPARC->atom_solve_flag[i]) {
                lmaxv[atmcount2] = pSPARC->AtmU[i].max_l;
                sizev[atmcount2] = pSPARC->AtmU[i].size;
                nspinorv[atmcount2] = pSPARC->AtmU[i].nspinor;
                vallenv[atmcount2] = pSPARC->AtmU[i].val_len;
                for (l = 0; l <= 3; l++) {
                    Uarr[atmcount2*5 + l] = pSPARC->AtmU[i].U[l];
                }
                Uarr[atmcount2*5 + 4] = pSPARC->AtmU[i].rcut_user;

                tempbuff[atmcount2] = lmaxv[atmcount2];
                tempbuff[atmcount2 + atmcount] = sizev[atmcount2];
                tempbuff[atmcount2 + 2*atmcount] = nspinorv[atmcount2];
                tempbuff[atmcount2 + 3*atmcount] = vallenv[atmcount2];
                // for (l = 0; l <= 3; l++) {
                //     tempbuff[atmcount2 + 4*atmcount + l] = Uarr[atmcount2 + l];
                // }
                atmcount2++;
            }
        }
        MPI_Bcast(tempbuff, 4*atmcount, MPI_INT, 0, MPI_COMM_WORLD);

        // Send U
        MPI_Bcast(Uarr, 5*atmcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // pack AtmU[i].ppl[l] and bcast
        ppl_sdispl[0] = 0;
        for (i = 0; i < atmcount; i++) {
            ppl_sdispl[i+1] = ppl_sdispl[i] + lmaxv[i] + 1;
        }
        
        pplv = (int *)malloc( ppl_sdispl[atmcount] * sizeof(int));
        
        assert(pplv != NULL);
        atmcount2 = 0;
        for (i = 0; i < Ntypes; i++) {
            if (!pSPARC->atom_solve_flag[i])
                continue;

            for (l = 0; l <= lmaxv[atmcount2]; l++) {
                pplv[ppl_sdispl[atmcount2] + l] = pSPARC->AtmU[i].ppl[l];
            }
            atmcount2++;
        }
        MPI_Bcast(pplv, ppl_sdispl[atmcount], MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        // allocate AtmU array for receiver process
        pSPARC->AtmU = (HUB_DUDAREV_OBJ *)malloc(pSPARC->Ntypes*sizeof(HUB_DUDAREV_OBJ));
        assert(pSPARC->AtmU != NULL);
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        MPI_Bcast(tempbuff, 4*atmcount, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(Uarr, 5*atmcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Bcast atom solution pre-info. took %0.3f ms\n",(t2-t1)*1000);
#endif

        // unpack info.
        atmcount2 = 0;
        for (i = 0; i < Ntypes; i++) {
            if (!pSPARC->atom_solve_flag[i])
                continue;

            lmaxv[atmcount2] = tempbuff[atmcount2];
            sizev[atmcount2] = tempbuff[atmcount2 + atmcount];
            nspinorv[atmcount2] = tempbuff[atmcount2 + 2*atmcount];
            vallenv[atmcount2] = tempbuff[atmcount2 + 3*atmcount];
            // for (l = 0; l <= 3; l++) {
            //     Uarr[atmcount2 + l] = tempbuff[atmcount2 + 4*atmcount + l];
            // }
            pSPARC->AtmU[i].max_l = lmaxv[atmcount2];
            pSPARC->AtmU[i].size = sizev[atmcount2];
            pSPARC->AtmU[i].nspinor = nspinorv[atmcount2];
            pSPARC->AtmU[i].val_len = vallenv[atmcount2];
            for (l = 0; l <= 3; l++) {
                pSPARC->AtmU[i].U[l] = Uarr[atmcount2*5 + l];
            }
            pSPARC->AtmU[i].rcut_user = Uarr[atmcount2*5 + 4];
            atmcount2++;
        }

        // bcast AtmU[i].ppl[l] and unpack
        ppl_sdispl[0] = 0;
        for (i = 0; i < atmcount; i++) {
            ppl_sdispl[i+1] = ppl_sdispl[i] + lmaxv[i] + 1;
        }

        pplv = (int *)malloc(ppl_sdispl[atmcount] * sizeof(int));
        
        MPI_Bcast(pplv, ppl_sdispl[atmcount], MPI_INT, 0, MPI_COMM_WORLD);

        atmcount2 = 0;
        for (i = 0; i < Ntypes; i++) {
            if (!pSPARC->atom_solve_flag[i])
                continue;

            pSPARC->AtmU[i].ppl = (int *)malloc((lmaxv[atmcount2] + 1) * sizeof(int));
            for (l = 0; l <= lmaxv[atmcount2]; l++) {
                pSPARC->AtmU[i].ppl[l] = pplv[ppl_sdispl[atmcount2] + l];
            }
            atmcount2++;
        }
        
    }

    // allocate memory for buff, the extra 16*atmcount byte is spare memory
    lmax_sum = 0; size_sum = 0; nproj_sum = 0; nprojsize_sum = 0;
    atmcount2 = 0;
    for (i = 0; i < Ntypes; i++) {
        if (!pSPARC->atom_solve_flag[i])
            continue;

        lmax_sum += lmaxv[atmcount2]+1;
        size_sum += sizev[atmcount2];
        nproj = pSPARC->AtmU[i].val_len;
        nproj_sum += nproj* nspinorv[atmcount2];
        nprojsize_sum += nproj * sizev[atmcount2] * nspinorv[atmcount2];
        atmcount2++;
    }

    // l_buff = (nprojsize_sum + size_sum + atmcount) * sizeof(double) 
    //          + nproj_sum * sizeof(int)
    //          + 0*(atmcount) *16; // last term is spare memory in case
    l_buff = (nprojsize_sum + size_sum) * sizeof(double) 
    + nproj_sum * sizeof(int)
    + 0*(atmcount) *16; // last term is spare memory in case
    buff = (char *)malloc(l_buff*sizeof(char));
    assert(buff != NULL);

    if (rank == 0) {
        // pack the variables
        position = 0; atmcount2 = 0;
        for (i = 0; i < Ntypes; i++) {
            if (!pSPARC->atom_solve_flag[i])
                continue;

            nproj = vallenv[atmcount2]*nspinorv[atmcount2];
            MPI_Pack(pSPARC->AtmU[i].orbitals, nproj*sizev[atmcount2], MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->AtmU[i].RadialGrid, sizev[atmcount2], MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->AtmU[i].occ, nproj, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
            // MPI_Pack(pSPARC->AtmU[i].rc, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            atmcount2++;
        }
        // Broadcast the packed buffer
        MPI_Bcast(buff, l_buff, MPI_PACKED, 0, MPI_COMM_WORLD);
    } else {
        /* allocate memory for receiver processes */
        atmcount2 = 0;
        for (i = 0; i < Ntypes; i++) {
            if (!pSPARC->atom_solve_flag[i])
                continue;

            nproj = vallenv[atmcount2]*nspinorv[atmcount2];
            pSPARC->AtmU[i].orbitals = (double *)malloc(nproj*sizev[atmcount2]*sizeof(double));
            pSPARC->AtmU[i].RadialGrid = (double *)malloc(sizev[atmcount2]*sizeof(double));
            pSPARC->AtmU[i].occ = (int *)malloc(nproj*sizeof(int));
            if (pSPARC->AtmU[i].orbitals == NULL || pSPARC->AtmU[i].RadialGrid == NULL ||
                pSPARC->AtmU[i].occ == NULL) {
                    printf("\nmemory cannot be allocated5\n");
                    exit(EXIT_FAILURE);
            }
            atmcount2++;
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
        
        // unpack variables
        position = 0; atmcount2 = 0;
        for (i = 0; i < Ntypes; i++) {
            if(!pSPARC->atom_solve_flag[i])
                continue;

            nproj = vallenv[atmcount2]*nspinorv[atmcount2];
            MPI_Unpack(buff, l_buff, &position, pSPARC->AtmU[i].orbitals, nproj*sizev[atmcount2], MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->AtmU[i].RadialGrid, sizev[atmcount2], MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->AtmU[i].occ, nproj, MPI_INT, MPI_COMM_WORLD);
            // MPI_Unpack(buff, l_buff, &position, pSPARC->AtmU[i].rc, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            atmcount2++;
        }
    }

    // deallocate memory
    free(tempbuff); free(lmaxv); free(sizev); free(nspinorv); free(vallenv); 
    free(Uarr); free(pplv); free(ppl_sdispl);
    free(buff);

    // Set rc
    set_rc_loc_orbitals(pSPARC);
}

/**
* @brief Sets the cutoff radius for the local orbitals of atoms with U correction.
*/
void set_rc_loc_orbitals(SPARC_OBJ *pSPARC) {
    int ityp, i, j, l, Ntypes, lmax, orb_len;
    double *Orbital;
    Ntypes = pSPARC->Ntypes;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Note: Cutoff s.t. |psi|<5e-3 beyond rc - error w.r.t. QE 1e-4 ha/atom in total energy
    double rc = 6.325; 
    
    double cutoff = 5e-3; // May adjust this later depending on desired accuracy
    double rc_old = 0.0;
    for (ityp = 0; ityp < Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]){ // condition to check if U corrections are desired for atomtype
            continue;
        }

        lmax = pSPARC->AtmU[ityp].max_l;
        orb_len = pSPARC->AtmU[ityp].size; // vector length of each orbital from atom solve

        int lcount = 0;
        for (l = 0; l <= lmax; l++) {
            if (pSPARC->AtmU[ityp].U[l] == 0){
                lcount += pSPARC->AtmU[ityp].ppl[l];
                continue; // Skip for zero U values for a particular l orbital
            }

            Orbital = pSPARC->AtmU[ityp].orbitals + lcount * orb_len;

            for (j = orb_len - 1; j >= 0; j--) {
                if (fabs(Orbital[j]) > cutoff) {
                    rc = pSPARC->AtmU[ityp].RadialGrid[j];
                    break;
                }
            }

            if (rc < rc_old) {
                rc = rc_old;
            } else {
                rc_old = rc;
            }
        }
    }

    // Add buffer to rc
    rc = ceil(rc);

    #ifdef DEBUG
    if (!rank)
        printf("THE LOCAL ORBITAL CUTOFF IS %f.\n",rc);
    #endif

    // Variables needed for non-ortho cells
    double LV_inv[9], U[9], S[3], VT[9], superb[2], temp[9], LatVec[9];
    double S_inv[9] = {0};
    int m = 3;
    double row_norm_LV[3], col_norm_LV_inv[3];
    int num_bins;

    for (ityp = 0; ityp < Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp])
            continue;

        // cutoff of this type: U_RCUT if given (used as is), otherwise the default rule (rc above)
        double rct = (pSPARC->AtmU[ityp].rcut_user > 0.0) ? pSPARC->AtmU[ityp].rcut_user : rc;
        double *rg = pSPARC->AtmU[ityp].RadialGrid; int len = pSPARC->AtmU[ityp].size;
        if (rct > rg[len - 1]) {
            if (!rank) printf("WARNING: Hubbard orbital cutoff %.3f bohr of atom type %s exceeds the radial grid (%.3f bohr); using %.3f.\n",
                              rct, &pSPARC->atomType[ityp * L_ATMTYPE], rg[len - 1], rg[len - 1]);
            rct = rg[len - 1];
        }
        // largest orbital weight outside the cutoff, int_rc^inf R^2 r^2 dr (trapezoid), over the U orbitals of this type
        double wout_max = 0.0; int lc = 0;
        for (l = 0; l <= pSPARC->AtmU[ityp].max_l; l++) {
            for (int pp = 0; pp < pSPARC->AtmU[ityp].ppl[l]; pp++, lc++) {
                if (pSPARC->AtmU[ityp].U[l] == 0) continue;
                double *orb = pSPARC->AtmU[ityp].orbitals + lc * len, win = 0.0, wall = 0.0;
                for (j = 1; j < len; j++) {
                    double a = orb[j-1]*orb[j-1]*rg[j-1]*rg[j-1], b = orb[j]*orb[j]*rg[j]*rg[j], seg = 0.5*(a+b)*(rg[j]-rg[j-1]);
                    wall += seg; if (rg[j] <= rct) win += seg;
                }
                if (wall - win > wout_max) wout_max = wall - win;
            }
        }
        pSPARC->AtmU[ityp].rc_used = rct; pSPARC->AtmU[ityp].rc_wout = wout_max;
        if (!rank && wout_max > 1e-2)
            printf("WARNING: more than 1%% of the Hubbard orbital weight of atom type %s lies outside its cutoff (%.3f bohr).\n",
                   &pSPARC->atomType[ityp * L_ATMTYPE], rct);

        pSPARC->AtmU[ityp].rc2 = rct*rct;
        if (pSPARC->cell_typ == 0) {
            pSPARC->AtmU[ityp].rc[0] = rct;
            pSPARC->AtmU[ityp].rc[1] = rct;
            pSPARC->AtmU[ityp].rc[2] = rct;
        } else if (pSPARC->cell_typ >= 20) {
            printf("\nThis cell type is not supported in Hubbard calculations.\n");
            exit(EXIT_FAILURE);
        } else {
            /* Calculating pinv(LatVec): This is necessary because for extremely skewed LV, the LV maybe close to singular */
            // Compute SVD of LatVec(LV) first: LV = U * S * V^T
            for (int JJ = 0; JJ < 9; JJ++) LatVec[JJ] = pSPARC->LatVec[JJ];   // dgesvd overwrites its input: fresh copy per type
            LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, m, LatVec, m, S, U, m, VT, m, superb);

            // First compute S^{-1}
            for (i = 0; i < 3; i++) {
                if (S[i] > TEMP_TOL) S_inv[i * 3 + i] = 1.0/S[i];
            }

            // Compute LV^{-1} = V * S^{-1} * U^T
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, m, m, 1.0, VT, m, S_inv, m, 0.0, temp, m);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, m, m, 1.0, temp, m, U, m, 0.0, LV_inv, m);

            // Compute norms of each row of both LV and LV_inv
            // Note: LV = [a; b; c] and LV_inv = [a* b* c*] where a* is reciprocal vector of a
            //       Both LV and LV_inv are stored in a row major format
            for (i = 0; i < 3; i++) {
                row_norm_LV[i] = 0.0;
                col_norm_LV_inv[i] = 0.0;
                for (j = 0; j < 3; j++) {
                    row_norm_LV[i] += pSPARC->LatVec[i*3 + j]*pSPARC->LatVec[i*3 + j];
                    col_norm_LV_inv[i] += LV_inv[j*3 + i]*LV_inv[j*3 + i];
                }
                row_norm_LV[i] = sqrt(row_norm_LV[i]);
                col_norm_LV_inv[i] = sqrt(col_norm_LV_inv[i]);
            }

            // Compute number of cells needed in each (LV) direction to traverse rc distance
            // Note: Nx = ceil(rc * ||a*||), Ny = ceil(rc * ||b*||), Nz = ceil(rc * ||c*||)
            //       rc_x = Nx * ||a||, rc_y = Ny * ||b||, rc_z = Nx * ||c||
            for (i = 0; i < 3; i++) {
                num_bins = (int) ceil(rct*col_norm_LV_inv[i]);
                pSPARC->AtmU[ityp].rc[i] = ceil(num_bins * row_norm_LV[i]);
            }
        }
    }
}

/**
* @brief Spline interpolation of stored orbitals
*/
void Calculate_SplineDerivLocOrb(SPARC_OBJ *pSPARC) {
    int ityp, l, lmax, lcount, np, ppl_sum, orb_len;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]){ // condition to check if U corrections are desired for atomtype
            continue;
        }

        orb_len = pSPARC->AtmU[ityp].size;
        ppl_sum = 0; 
        lmax = pSPARC->AtmU[ityp].max_l;
        for (l = 0; l <= lmax; l++)
        {
            ppl_sum += pSPARC->AtmU[ityp].ppl[l]; // pSPARC->AtmU[ityp].ppl[l] is >1 only when core states are involved
        }
        pSPARC->AtmU[ityp].SplineFitOrb = (double *)malloc(sizeof(double) * orb_len * ppl_sum);
        if (pSPARC->AtmU[ityp].SplineFitOrb == NULL) {
            printf("Memory allocation failed!\n");
            exit(EXIT_FAILURE);
        }

        lcount = 0;
        for (l = 0; l <= lmax; l++) {
            for (np = 0; np < pSPARC->AtmU[ityp].ppl[l]; np++) {
                getYD_gen(pSPARC->AtmU[ityp].RadialGrid, pSPARC->AtmU[ityp].orbitals+lcount*orb_len, 
                          pSPARC->AtmU[ityp].SplineFitOrb+lcount*orb_len, orb_len);
                lcount++;
            }
        }
    }
}

/**
 * @brief Initialize the occupation matrix for all hubbard atoms, including the variables required for extrapolation
 */
void init_occ_mat(SPARC_OBJ *pSPARC) {
    int ityp, iat, l, lmax, ldispl, mcount, atmcount, Ntypes, nAtomv, spinor, isMagnetic, maj_sp, min_sp, angnum;
    double atomMag;
    int *l_states, *l_occ;

    // ERROR: If non collinear spin is invoked
    if (pSPARC->spin_typ == 2) {
        printf("\nERROR: Hubbard calculations with non collinear spin currently not supported.\n");
        exit(EXIT_FAILURE);
    }

    // ERROR: If spin-orbit coupling is invoked
    if (pSPARC->SOC_Flag == 1) {
        printf("\nERROR: Hubbard calculations with spin-orbit coupling currently not supported.\n");
        exit(EXIT_FAILURE);
    }

    // ERROR: If cyclix is invoked
    if (pSPARC->CyclixFlag == 1) {
        printf("\nERROR: Hubbard calculations with cyclix currently not supported.\n");
        exit(EXIT_FAILURE);
    }

    // ERROR: If a hybrid functional is invoked
    if (pSPARC->usefock > 0) {
        printf("\nERROR: Hubbard calculations with hybrid functionals not supported.\n");
        exit(EXIT_FAILURE);
    }

    Ntypes = pSPARC->Ntypes;

    // Calculate size of rho_mn for each atom type
    for (ityp = 0; ityp < Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) continue;

        angnum = 0; lmax = pSPARC->AtmU[ityp].max_l;
        for (l = 0; l <= lmax; l++) {
            if (pSPARC->AtmU[ityp].U[l] < TEMP_TOL) continue;

            angnum += 2*l + 1;
        }
        pSPARC->AtmU[ityp].angnum = angnum;
    }

    // Store Uval
    for (ityp = 0; ityp < Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) continue;

        angnum = 0; lmax = pSPARC->AtmU[ityp].max_l;
        pSPARC->AtmU[ityp].Uval = (double *)calloc(pSPARC->AtmU[ityp].angnum, sizeof(double));
        for (l = 0; l <= lmax; l++) {
            if (pSPARC->AtmU[ityp].U[l] < TEMP_TOL) continue;

            for (int JJ = -l ; JJ <= l; JJ++) {
                pSPARC->AtmU[ityp].Uval[angnum] = pSPARC->AtmU[ityp].U[l];
                angnum++;
            }
        }
    }


    // allocate memory for rho_mn, rho_mn_at, rho_mn_0dt, rho_mn_1dt, rho_mn_2dt
    pSPARC->rho_mn = (double ***)calloc(pSPARC->n_atom, sizeof(double **));
    pSPARC->rho_mn_at = (double ***)calloc(pSPARC->n_atom, sizeof(double **));

    if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1 || pSPARC->RelaxFlag == 3){
        pSPARC->drho_mn = (double ***)calloc(pSPARC->n_atom, sizeof(double **));
        pSPARC->drho_mn_0dt = (double ***)calloc(pSPARC->n_atom, sizeof(double **));
        pSPARC->drho_mn_1dt = (double ***)calloc(pSPARC->n_atom, sizeof(double **));
        pSPARC->drho_mn_2dt = (double ***)calloc(pSPARC->n_atom, sizeof(double **));

        pSPARC->atom_pos_nm_occM = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
        assert(pSPARC->atom_pos_nm_occM != NULL);
        pSPARC->atom_pos_0dt_occM = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
        assert(pSPARC->atom_pos_0dt_occM != NULL);
        pSPARC->atom_pos_1dt_occM = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
        assert(pSPARC->atom_pos_1dt_occM != NULL);
        pSPARC->atom_pos_2dt_occM = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
        assert(pSPARC->atom_pos_2dt_occM != NULL);
    }

    atmcount = 0;
    for (ityp = 0; ityp < Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }
        angnum = pSPARC->AtmU[ityp].angnum;
        for (iat  = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            pSPARC->rho_mn[atmcount] = (double **)calloc(pSPARC->Nspinor, sizeof(double *));
            pSPARC->rho_mn_at[atmcount] = (double **)calloc(pSPARC->Nspinor, sizeof(double *));

            if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1 || pSPARC->RelaxFlag == 3){
                pSPARC->drho_mn[atmcount] = (double **)calloc(pSPARC->Nspinor, sizeof(double *));
                pSPARC->drho_mn_0dt[atmcount] = (double **)calloc(pSPARC->Nspinor, sizeof(double *));
                pSPARC->drho_mn_1dt[atmcount] = (double **)calloc(pSPARC->Nspinor, sizeof(double *));
                pSPARC->drho_mn_2dt[atmcount] = (double **)calloc(pSPARC->Nspinor, sizeof(double *));
            }
            
            for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                pSPARC->rho_mn[atmcount][spinor] = (double *)calloc(angnum * angnum, sizeof(double));
                pSPARC->rho_mn_at[atmcount][spinor] = (double *)calloc(angnum * angnum, sizeof(double));

                if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1 || pSPARC->RelaxFlag == 3){
                    pSPARC->drho_mn[atmcount][spinor] = (double *)calloc(angnum * angnum, sizeof(double));
                    pSPARC->drho_mn_0dt[atmcount][spinor] = (double *)calloc(angnum * angnum, sizeof(double));
                    pSPARC->drho_mn_1dt[atmcount][spinor] = (double *)calloc(angnum * angnum, sizeof(double));
                    pSPARC->drho_mn_2dt[atmcount][spinor] = (double *)calloc(angnum * angnum, sizeof(double));
                }
            }
            atmcount++;
        }
    }
    
    // populate rho_mn and rho_mn_at
    atmcount = 0;
    for (ityp = 0; ityp < Ntypes; ityp++) { // loop over atom types
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        nAtomv = pSPARC->nAtomv[ityp]; // cumulative sum of number of atoms of type ityp
        angnum = pSPARC->AtmU[ityp].angnum;

        for (iat = 0; iat < nAtomv; iat++) { // loop over atoms of type ityp
            isMagnetic = 0;
            
            // check if the atom in question is magnetic (initial guess or user input) in spin polarized cases
            if (pSPARC->spin_typ == 1) {
                atomMag = pSPARC->atom_spin[3*atmcount + 2]; // z spin

                if (atomMag > 0){
                    isMagnetic = 1;
                    maj_sp = 0; // up spin filled first
                    min_sp = 1;
                } else if (atomMag < 0) {
                    isMagnetic = 1;
                    maj_sp = 1; // dw spin filled first
                    min_sp = 0;
                }     
            }

            // store number of magnetic states and occupations per azimuthal quantum number for valid U correction on the corresponding state
            l_states = (int *)calloc(angnum, sizeof(int));
            l_occ = (int *)calloc(angnum, sizeof(int));

            mcount = 0;
            for (l = 0; l <= pSPARC->AtmU[ityp].max_l; l++) {
                if (pSPARC->AtmU[ityp].U[l] < TEMP_TOL) continue;

                ldispl = -1;
                for (int ll = 0; ll <= l; ll++) {
                    ldispl += pSPARC->AtmU[ityp].ppl[ll];
                }

                for (int m = -l; m <= l; m++) {
                    l_states[mcount] = 2*l + 1;
                    l_occ[mcount] = pSPARC->AtmU[ityp].occ[ldispl];
                    mcount++;
                }
            }

            // populate the rho_mn matrix and rho_mn_at
            if (isMagnetic) {// magnetic case
                for (int m = 0; m < angnum; m++) { 
                    if (l_occ[m] >= l_states[m]) { // total occupation > allowed half fillings
                        pSPARC->rho_mn[atmcount][maj_sp][m * angnum + m] = 1.0;
                        pSPARC->rho_mn[atmcount][min_sp][m * angnum + m] = (double)(l_occ[m] - l_states[m])/l_states[m];

                        
                        pSPARC->rho_mn_at[atmcount][maj_sp][m * angnum + m] = 1.0;
                        pSPARC->rho_mn_at[atmcount][min_sp][m * angnum + m] = (double)(l_occ[m] - l_states[m])/l_states[m];
                    
                    } else { 
                        pSPARC->rho_mn[atmcount][maj_sp][m * angnum + m] = (double)(l_occ[m] - l_states[m])/l_states[m];

                        
                        pSPARC->rho_mn_at[atmcount][maj_sp][m * angnum + m] = (double)(l_occ[m] - l_states[m])/l_states[m];
                    }
                }
                
            } else { // non magnetic case
                for (int spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                    for (int m = 0; m < angnum; m++) {
                        pSPARC->rho_mn[atmcount][spinor][m * angnum + m] = 0.5*l_occ[m]/l_states[m];

                        
                        pSPARC->rho_mn_at[atmcount][spinor][m * angnum + m] = 0.5*l_occ[m]/l_states[m];
                    }
                }
            }

            free(l_occ); free(l_states);

            atmcount++;

        } // end of loop for atoms of type ityp
    } // end of loop for atom types

    // Also initialise memory for the history vectors
    pSPARC->rho_mn_X = (double **)calloc(pSPARC->n_atom, sizeof(double *));
    pSPARC->rho_mn_F = (double **)calloc(pSPARC->n_atom, sizeof(double *));
    pSPARC->rho_mn_xkm1 = (double **)calloc(pSPARC->n_atom, sizeof(double *));
    pSPARC->rho_mn_fkm1 = (double **)calloc(pSPARC->n_atom, sizeof(double *));
    pSPARC->rho_mn_xk = (double **)calloc(pSPARC->n_atom, sizeof(double *));
    pSPARC->rho_mn_fk = (double **)calloc(pSPARC->n_atom, sizeof(double *));

    atmcount = 0;
    for (ityp = 0; ityp < Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }
        angnum = pSPARC->AtmU[ityp].angnum;
        for (iat  = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            pSPARC->rho_mn_X[atmcount] = (double *)calloc(pSPARC->Nspinor*angnum*angnum*pSPARC->MixingHistory, sizeof(double));
            pSPARC->rho_mn_F[atmcount] = (double *)calloc(pSPARC->Nspinor*angnum*angnum*pSPARC->MixingHistory, sizeof(double));
            pSPARC->rho_mn_xkm1[atmcount] = (double *)calloc(pSPARC->Nspinor*angnum*angnum, sizeof(double));
            pSPARC->rho_mn_fkm1[atmcount] = (double *)calloc(pSPARC->Nspinor*angnum*angnum, sizeof(double));
            pSPARC->rho_mn_xk[atmcount] = (double *)calloc(pSPARC->Nspinor*angnum*angnum, sizeof(double));
            pSPARC->rho_mn_fk[atmcount] = (double *)calloc(pSPARC->Nspinor*angnum*angnum, sizeof(double));

            for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                memcpy(pSPARC->rho_mn_xkm1[atmcount] + spinor*angnum*angnum, pSPARC->rho_mn[atmcount][spinor],
                    angnum*angnum*sizeof(double));

            }

            memcpy(pSPARC->rho_mn_xk[atmcount], pSPARC->rho_mn_xkm1[atmcount],
                pSPARC->Nspinor*angnum*angnum*sizeof(double));
            
            atmcount++;
        }
    }
}

#define LAYOUT_BANDS 30      // target bands per band group for the default layout
#define LAYOUT_MIN_EDGE 24   // smallest subdomain edge (grid points) the default layout creates
#define LAYOUT_MIN_COVER 3.0 // apply the layout only if the U-atom spheres cover each grid point this often on average
#define LAYOUT_RC 7.0        // nominal orbital cutoff (bohr) for that estimate (the real cutoffs are set later)

/**
 * @brief   DFT+U-aware default layout, called from Setup_Comms (parallelization.c) after SPARC's own search. That search
 *          puts as many processes as possible into band groups; the Hubbard product (locOrbRoutines.c) has a fixed cost per
 *          call that only domains reduce. Among the splits of the same processes into (band groups) x (domains), pick the
 *          one with bands per band group closest to LAYOUT_BANDS (on a log scale; ties go to more band groups), never more
 *          band groups than SPARC chose, and only domain grids SPARC_Dims_create accepts with every subdomain edge
 *          >= LAYOUT_MIN_EDGE points. Applied only at the Gamma point and when the Hubbard product is a large part of the
 *          work: the U-atom spheres (nominal radius LAYOUT_RC) cover each grid point at least LAYOUT_MIN_COVER times on
 *          average (rutile TiO2: 6.8; one Mo in a 1385 bohr^3 MoO3 cell: 1.0).
 */
void hubbard_default_layout(const SPARC_OBJ *pSPARC, int *npband, int *npNd, const int *gridsizes, int minsize)
{
    if (!pSPARC->is_hubbard || !pSPARC->isGammaPoint || pSPARC->usefock || pSPARC->CyclixFlag) return;
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // At this point only rank 0 holds atom_solve_flag (the other ranks receive it in the Hubbard set-up), so rank 0
    // computes the coverage and broadcasts it; every rank of MPI_COMM_WORLD runs SPARC's default layout search.
    double cover = 0.0;
    if (world_rank == 0) {
        int nU = 0;
        for (int t = 0; t < pSPARC->Ntypes; t++) if (pSPARC->atom_solve_flag[t]) nU += pSPARC->nAtomv[t];
        double vol = pSPARC->Jacbdet * pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;   // cell volume (any cell shape)
        cover = vol > 0 ? nU * (4.0 / 3.0) * M_PI * pow(LAYOUT_RC, 3) / vol : 0.0;
    }
    MPI_Bcast(&cover, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (cover < LAYOUT_MIN_COVER) return;
    int Pk = (*npband) * (*npNd), Ns = pSPARC->Nstates, best = *npband;
    double score_best = fabs(log((double)Ns / best / LAYOUT_BANDS));
    for (int d = 1; d <= Pk && d <= Ns && d <= *npband; d++) {
        if (Pk % d) continue;
        int nd = Pk / d, dims[3], ierr = 0;
        if (nd > 1) {
            SPARC_Dims_create(nd, 3, gridsizes, minsize, dims, &ierr);
            if (ierr || gridsizes[0] / dims[0] < LAYOUT_MIN_EDGE || gridsizes[1] / dims[1] < LAYOUT_MIN_EDGE
                || gridsizes[2] / dims[2] < LAYOUT_MIN_EDGE) continue;
        }
        double score = fabs(log((double)Ns / d / LAYOUT_BANDS));
        if (score < score_best - 1e-12 || (fabs(score - score_best) <= 1e-12 && d > best)) { score_best = score; best = d; }
    }
    *npNd = Pk / best; *npband = best;
}
