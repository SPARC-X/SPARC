/**
 * @file occupationMatrix.c 
 * @brief This file contains the routines required to calculate hubbard contribution to pressure without calculating stress tensor.
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
#include <math.h>
#include <mpi.h>
#include <assert.h>

#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#include "hubbardPressure.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "lapVecRoutines.h"
#include "tools.h" 
#include "isddft.h"
#include "hubbardForce.h"
#include "stress.h"
#include "hubbardStress.h"

#define TEMP_TOL 1e-12

/**
 * @brief Calculate hubbard pressure
 */
void Calculate_hubbard_pressure(SPARC_OBJ *pSPARC) {
    if (pSPARC->isGammaPoint) {
        Calculate_hubbard_pressure_linear(pSPARC);
    } else {
        Calculate_hubbard_pressure_kpt(pSPARC);
    }
}

/**
 * @brief Calculate gamma point hubbard pressure
 */
void Calculate_hubbard_pressure_linear(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    int i, ncol, DMnd, DMndsp;
    int dim, dim2, count, size_k, spinor, Nspinor;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    size_k = DMndsp * ncol;
    
    double *alpha, *beta;
    double energy_hub = 0.0, pressure_hub = 0.0;
    

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    alpha = (double *)calloc( pSPARC->IP_displ_U[atm_idx] * ncol * 4 * Nspinor, sizeof(double));
    assert(alpha != NULL);

    #ifdef DEBUG 
        if (!rank) printf("Start calculating pressure contributions from hubbard part. \n");
    #endif

    // find <Orb_Jlm, psi>
    beta = alpha;
    Compute_Integral_psi_Orb(pSPARC, beta, pSPARC->Xorb);

    /* find inner product <Orb_Jlm, dPsi_n.(x-R_J)> */
    count = 1;
    for (dim = 0; dim < 3; dim++) {
        for (spinor = 0; spinor < Nspinor; spinor++) {
            // find dPsi in direction dim along lattice vector directions
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spinor*DMnd, DMndsp, 
                pSPARC->Yorb+spinor*DMnd, DMndsp, dim, pSPARC->dmcomm);
        }
        beta = alpha + pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * count;
        Compute_Integral_Orb_XmRjp_beta_Dpsi(pSPARC, pSPARC->Yorb, beta, dim);
        count++;
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * 4, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate hubbard pressure */
    Compute_pressure_hubbard_by_integrals(pSPARC, &pressure_hub, alpha);

    free(alpha);

    // sum over all spin
    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pressure_hub, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pressure_hub, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    /* compute hubbard energy */
    energy_hub = Compute_Hubbard_Energy(pSPARC);
    pressure_hub -= energy_hub; // check about multiplying by 3 later

    if (!rank) {
        pSPARC->pres_hub = pressure_hub;
    }

    #ifdef DEBUG    
        if (!rank){
            printf("Pressure contribution from hubbard part: = %.15f Ha\n", pSPARC->pres_hub); 

            // Define measure of unit cell
            double cell_measure = pSPARC->Jacbdet;
            if(pSPARC->BCx == 0)
                cell_measure *= pSPARC->range_x;
            if(pSPARC->BCy == 0)
                cell_measure *= pSPARC->range_y;
            if(pSPARC->BCz == 0)
                cell_measure *= pSPARC->range_z;

            printf("Pressure contribution from hubbard part: = %6.2f kbar\n", 10*CONST_HA_BOHR3_GPA*(pSPARC->pres_hub/(-3*cell_measure)));
        } 
    #endif
}

/**
 * @brief   Calculate <Orb_Jlm, (x-RJ')_beta, DPsi_n> for spinor psi
 */
void Compute_Integral_Orb_XmRjp_beta_Dpsi(SPARC_OBJ *pSPARC, double *dpsi_xi, double *beta, int dim2) {
    int i, n, ndc, ityp, iat, ncol, DMnd, atom_index;
    int spinor, Nspinor, DMndsp, spinorshift;
    int indx, i_DM, j_DM, k_DM, DMnx, DMny;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;

    double *dpsi_xi_rc, *dpsi_ptr, *dpsi_xi_rc_ptr;
    double R1, R2, R3, x1_R1, x2_R2, x3_R3, XmRjp;

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) continue;

        if (! pSPARC->locProj[ityp].nproj) continue; 

        for (iat = 0; iat < pSPARC->Atom_Influence_loc_orb[ityp].n_atom; iat++) {
            R1 = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3];
            R2 = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3+1];
            R3 = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3+2];
            ndc = pSPARC->Atom_Influence_loc_orb[ityp].ndc[iat];
            dpsi_xi_rc = (double *)malloc( ndc * ncol * sizeof(double));
            assert(dpsi_xi_rc);
            atom_index = pSPARC->Atom_Influence_loc_orb[ityp].atom_index[iat];
            for (spinor = 0; spinor < Nspinor; spinor++) {
                for (n = 0; n < ncol; n++) {
                    dpsi_ptr = dpsi_xi + n * DMndsp + spinor * DMnd;
                    dpsi_xi_rc_ptr = dpsi_xi_rc + n * ndc;

                    for (i = 0; i < ndc; i++) {
                        indx = pSPARC->Atom_Influence_loc_orb[ityp].grid_pos[iat][i];
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
                spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * spinor;
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->locProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->locProj[ityp].Orb[iat], ndc, 
                            dpsi_xi_rc, ndc, 1.0, beta+spinorshift+pSPARC->IP_displ_U[atom_index]*ncol, pSPARC->locProj[ityp].nproj);
            }
            free(dpsi_xi_rc);
        }
    }
}

/**
 * @brief Compute hubbard pressure for gamma point using integrals
 */
void Compute_pressure_hubbard_by_integrals(SPARC_OBJ *pSPARC,double *pressure_hub,double *alpha) {
    int n, ityp, iat, ncol, Ns;
    int count, spinor, Nspinor;
    int *IP_displ_U;
    double term_temp;
    int nstart = pSPARC->band_start_indx;
    int nend = pSPARC->band_end_indx;

    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;    
    Nspinor = pSPARC->Nspinor_spincomm;
    IP_displ_U = pSPARC->IP_displ_U;

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    // Extract local g_n as array
    double **g_n = (double **)calloc((pSPARC->Nspinor_spincomm), sizeof(double*));
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        g_n[spinor] = (double *)calloc(ncol, sizeof(double));
    }
    
    count = 0;
    for (n = nstart; n <= nend; n++) {
        // double woccfac = pSPARC->occfac;
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
            double *occ = pSPARC->occ; 
            if (pSPARC->spin_typ == 1) occ += spinor*Ns;
            
            // g_n[spinor][count] = woccfac * occ[n];
            g_n[spinor][count] = occ[n];
        }
        count++;
    }
    // End of extract local g_n as array

    // g_n <Orb_Jlm | Psi> stored here
    double *alpha_gn = (double *)calloc( pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor, sizeof(double));
    int atmcount;
    for (spinor = 0; spinor < Nspinor; spinor++) {
        int spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * spinor;

        atmcount = 0; 

        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->atom_solve_flag[ityp]) continue;

            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                // Scale each column of < Orb_Jlm, x_n > with g_n (size: m x Ns_loc)
                cblas_dcopy(pSPARC->locProj[ityp].nproj * ncol, alpha +spinorshift+pSPARC->IP_displ_U[atmcount]*ncol, 1, 
                    alpha_gn + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, 1); // copy into alpha_gn
                
                for (n = 0; n < ncol; n++) { // scale each column
                    cblas_dscal(pSPARC->locProj[ityp].nproj, g_n[spinor][n], 
                                alpha_gn +spinorshift+pSPARC->IP_displ_U[atmcount]*ncol + n*pSPARC->locProj[ityp].nproj, 1);
                }

                atmcount++;
            }
        }
    }
    // End of g_n <Orb_Jlm | Psi> storage

    double *beta1_x1, *beta2_x2, *beta3_x3; 
    beta1_x1 = alpha + IP_displ_U[atm_idx]*ncol*Nspinor;
    beta2_x2 = alpha + IP_displ_U[atm_idx]*ncol*Nspinor * 2;
    beta3_x3 = alpha + IP_displ_U[atm_idx]*ncol*Nspinor * 3;

    double *rho_block, *pre_fac, *tf_xx, *tf_yy, *tf_zz;
    double *pre_fac_tf_xx, *pre_fac_tf_yy, *pre_fac_tf_zz;
    int angnum;

    int hub_sp = 0;
    for (spinor = 0; spinor < Nspinor; spinor++) {
        int spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * spinor;

        if (pSPARC->Nspinor_spincomm == 1) {
            hub_sp = pSPARC->spincomm_index;
        } else {
            hub_sp = spinor;
        }

        atmcount = 0;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->atom_solve_flag[ityp]) {
                atmcount += pSPARC->nAtomv[ityp];
                continue;
            }

            angnum = pSPARC->locProj[ityp].nproj;
            pre_fac = (double *)calloc(angnum*angnum, sizeof(double));

            // Initialise tf_xx, tf_yy, tf_zz
            tf_xx = (double *)calloc(angnum*angnum, sizeof(double)); 
            tf_yy = (double *)calloc(angnum*angnum, sizeof(double));
            tf_zz = (double *)calloc(angnum*angnum, sizeof(double));

            // Initialise pre_fac times (tf_xx, tf_yy, tf_zz)
            pre_fac_tf_xx = (double *)calloc(angnum*angnum, sizeof(double));
            pre_fac_tf_yy = (double *)calloc(angnum*angnum, sizeof(double));
            pre_fac_tf_zz = (double *)calloc(angnum*angnum, sizeof(double));

            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                // First reset pre_fac and pre_fac_rho_mn to 0
                memset(pre_fac, 0, angnum*angnum*sizeof(double));

                rho_block = pSPARC->rho_mn[atmcount][hub_sp];

                // Calculate prefactor U(\delta_{mn} - rho_{mn})
                for (int col = 0; col < angnum; col++) {
                    for (int row = 0; row < angnum; row++) {
                        if (row == col) {
                            pre_fac[angnum*col + row] += 0.5;
                        }
                        pre_fac[angnum*col + row] -= rho_block[angnum*col + row];
                    }
                }

                // Scale the rows by Uval[row]
                for (int row = 0; row < angnum; row++) {
                    for (int col = 0; col < angnum; col++) {
                        pre_fac[angnum*col + row] *= pSPARC->AtmU[ityp].Uval[row];
                    }
                }

                // Calculate the terms
                // xx
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, 2.0,
                            beta1_x1 + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                            alpha_gn + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                            0.0, tf_xx, angnum);
                
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, 1.0,
                            pre_fac, angnum, tf_xx, angnum, 0.0, pre_fac_tf_xx, angnum);

                term_temp = 0.0;
                for (int col = 0; col < angnum; col++) term_temp += pre_fac_tf_xx[angnum*col + col];
                *pressure_hub -= pSPARC->occfac*term_temp;

                // yy
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, 2.0,
                    beta2_x2 + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                    alpha_gn + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                    0.0, tf_yy, angnum);
                    
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, 1.0,
                            pre_fac, angnum, tf_yy, angnum, 0.0, pre_fac_tf_yy, angnum);

                term_temp = 0.0;
                for (int col = 0; col < angnum; col++) term_temp += pre_fac_tf_yy[angnum*col + col];
                *pressure_hub -= pSPARC->occfac*term_temp;

                // zz
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, 2.0,
                    beta3_x3 + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                    alpha_gn + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                    0.0, tf_zz, angnum);
                    
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, 1.0,
                            pre_fac, angnum, tf_zz, angnum, 0.0, pre_fac_tf_zz, angnum);

                term_temp = 0.0;
                for (int col = 0; col < angnum; col++) term_temp += pre_fac_tf_zz[angnum*col + col];
                *pressure_hub -= pSPARC->occfac*term_temp;

                atmcount++;
            }
            free(pre_fac); 
            free(tf_xx); free(tf_yy); free(tf_zz);
            free(pre_fac_tf_xx); free(pre_fac_tf_yy); free(pre_fac_tf_zz);
        }
    }

    // free memory
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (g_n[spinor] != NULL) free(g_n[spinor]);
    }

    free(g_n); free(alpha_gn);
}

/**
 * @brief Calculate hubbard pressure for k-points using integrals
 */
void Calculate_hubbard_pressure_kpt(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    int i, ncol, DMnd, DMndsp;
    int dim, dim2, count, kpt, Nk, size_k, spinor, Nspinor;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    size_k = DMndsp * ncol;

    double _Complex *alpha, *beta;
    alpha = NULL;
    double energy_hub = 0.0, pressure_hub = 0.0;

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    alpha = (double _Complex *)calloc( pSPARC->IP_displ_U[atm_idx] * ncol * Nk * Nspinor * 4, sizeof(double _Complex));
    assert(alpha != NULL);

    double k1, k2, k3, kpt_vec;
#ifdef DEBUG 
    if (!rank) printf("Start calculating pressure contributions from hubbard part. \n");
#endif

    // find <Orb_Jlm, psi>
    for (kpt = 0; kpt < Nk; kpt++) {
        beta = alpha + pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * kpt;
        Compute_Integral_psi_Orb_kpt(pSPARC, beta, pSPARC->Xorb_kpt+kpt*size_k, kpt);
    }

    // find inner product <Orb_Jlm, dPsi_n.(x-R_J)>
    count = 1;
    for (dim = 0; dim < 3; dim++) {
        for (kpt = 0; kpt < Nk; kpt++) {
            k1 = pSPARC->k1_loc[kpt];
            k2 = pSPARC->k2_loc[kpt];
            k3 = pSPARC->k3_loc[kpt];
            kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
            for (spinor = 0; spinor < Nspinor; spinor++) {
                // find dPsi in direction dim along lattice vector directions
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+kpt*size_k+spinor*DMnd, DMndsp, 
                                        pSPARC->Yorb_kpt+spinor*DMnd, DMndsp, dim, &kpt_vec, pSPARC->dmcomm);
            }
            beta = alpha + pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * (Nk * count + kpt);
            Compute_Integral_Orb_XmRjp_beta_Dpsi_kpt(pSPARC, pSPARC->Yorb_kpt, beta, kpt, dim);
        }
        count++;
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate hubbard pressure */
    Compute_pressure_hubbard_by_integrals_kpt(pSPARC, &pressure_hub, alpha);

    free(alpha);

    // sum over all spin
    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pressure_hub, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    // sum over all kpoints
    if (pSPARC->npkpt > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pressure_hub, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pressure_hub, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    /* compute hubbard energy */
    energy_hub = Compute_Hubbard_Energy(pSPARC);
    pressure_hub -= energy_hub; // check about multiplying by 3 later

    if (!rank) {
        pSPARC->pres_hub = pressure_hub;
    }

    #ifdef DEBUG    
        if (!rank){
            printf("Pressure contribution from hubbard part: = %.15f Ha\n", pSPARC->pres_hub); 

            // Define measure of unit cell
            double cell_measure = pSPARC->Jacbdet;
            if(pSPARC->BCx == 0)
                cell_measure *= pSPARC->range_x;
            if(pSPARC->BCy == 0)
                cell_measure *= pSPARC->range_y;
            if(pSPARC->BCz == 0)
                cell_measure *= pSPARC->range_z;

            printf("Pressure contribution from hubbard part: = %6.2f kbar\n", 10*CONST_HA_BOHR3_GPA*(pSPARC->pres_hub/(-3*cell_measure)));
        } 
    #endif
}

/**
 * @brief Calculate <orb_Jlm, (x-RJ')_beta, DPsi_n> for spinor pressure
 */
void Compute_Integral_Orb_XmRjp_beta_Dpsi_kpt(SPARC_OBJ *pSPARC, double _Complex *dpsi_xi, double _Complex *beta, int kpt, int dim2) {
    int i, n, ndc, ityp, iat, ncol, DMnd, atom_index;
    int spinor, Nspinor, DMndsp, spinorshift, nproj, ispinor, *IP_displ_U;
    int indx, i_DM, j_DM, k_DM, DMnx, DMny;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;

    double _Complex *dpsi_xi_rc, *dpsi_ptr, *dpsi_xi_rc_ptr;
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, R1, R2, R3, x1_R1, x2_R2, x3_R3, XmRjp;
    double _Complex bloch_fac, b, **Orb = NULL;
    
    k1 = pSPARC->k1_loc[kpt];
    k2 = pSPARC->k2_loc[kpt];
    k3 = pSPARC->k3_loc[kpt];

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    IP_displ_U = pSPARC->IP_displ_U;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (!pSPARC->atom_solve_flag[ityp]) continue;

        nproj = pSPARC->locProj[ityp].nproj;

        Orb = pSPARC->locProj[ityp].Orb_c;

        for (iat = 0; iat < pSPARC->Atom_Influence_loc_orb[ityp].n_atom; iat++) {
            R1 = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3  ];
            R2 = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3+1];
            R3 = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3+2];
            theta = -k1 * (floor(R1/Lx) * Lx) - k2 * (floor(R2/Ly) * Ly) - k3 * (floor(R3/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            b = 1.0;
            ndc = pSPARC->Atom_Influence_loc_orb[ityp].ndc[iat];
            dpsi_xi_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
            assert(dpsi_xi_rc);
            atom_index = pSPARC->Atom_Influence_loc_orb[ityp].atom_index[iat];
            for (spinor = 0; spinor < Nspinor; spinor++) {
                ispinor = spinor;

                for (n = 0; n < ncol; n++) {
                    dpsi_ptr = dpsi_xi + n * DMndsp + ispinor * DMnd;
                    dpsi_xi_rc_ptr = dpsi_xi_rc + n * ndc;

                    for (i = 0; i < ndc; i++) {
                        indx = pSPARC->Atom_Influence_loc_orb[ityp].grid_pos[iat][i];
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

                spinorshift = IP_displ_U[atm_idx] * ncol * spinor;
                // No need to multiply dV, accounted for in Compute_Integral_psi_Orb_kpt
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nproj, ncol, ndc, &bloch_fac, Orb[iat], ndc, 
                    dpsi_xi_rc, ndc, &b, beta+spinorshift+IP_displ_U[atom_index]*ncol, nproj); 
            }
            free(dpsi_xi_rc);
        }
    }
}

/**
 * @brief Compute hubbard pressure with k-points by integrals
 */
void Compute_pressure_hubbard_by_integrals_kpt(SPARC_OBJ *pSPARC, double *pressure_hub, double _Complex *alpha) {
    int k, n, ityp, iat, ncol, Ns;
    int count, Nk, spinor, Nspinor, hub_sp;
    int *IP_displ_U;
    double term_temp;
    int nstart = pSPARC->band_start_indx;
    int nend = pSPARC->band_end_indx;

    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;    
    Nk = pSPARC->Nkpts_kptcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    IP_displ_U = pSPARC->IP_displ_U;

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    // Extract local g_nk as array
    double **g_nk = (double **)calloc((pSPARC->Nspinor_spincomm), sizeof(double*));
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        g_nk[spinor] = (double *)calloc(Nk*ncol, sizeof(double));
    }
    
    count = 0;
    for (k = 0; k < Nk; k++) {
        for (n = nstart; n <= nend; n++) {
            // double woccfac = pSPARC->occfac*(pSPARC->kptWts_loc[k] / pSPARC->Nkpts);
            double woccfac = (pSPARC->kptWts_loc[k] / pSPARC->Nkpts);
            for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
                double *occ = pSPARC->occ + k*Ns; 
                if (pSPARC->spin_typ == 1) occ += spinor*Ns*Nk;
                g_nk[spinor][count] = woccfac * occ[n];
            }
            count++;
        }
    }
    // Finished extracting g_nk

    // g_nk <Orb_Jlm | Psi> stored here
    double _Complex *alpha_gnk = (double _Complex *)calloc( pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * Nk, sizeof(double _Complex));
    int angnum, atmcount;

    for (k = 0; k < Nk; k++) {
        for (spinor = 0; spinor < Nspinor; spinor++) {
            int spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * k + pSPARC->IP_displ_U[atm_idx] * ncol * spinor;

            atmcount = 0;

            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                if (!pSPARC->atom_solve_flag[ityp]) continue;

                angnum = pSPARC->AtmU[ityp].angnum;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    // Scale each column of < Orb_Jlm, x_n > with g_n (size: m x Ns_loc)
                    cblas_zcopy(pSPARC->locProj[ityp].nproj * ncol, alpha + spinorshift + pSPARC->IP_displ_U[atmcount]*ncol, 1, 
                        alpha_gnk + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, 1); // copy into alpha_gn

                    for (n = 0; n < ncol; n++) { // scale each column
                        double _Complex scale_f = g_nk[spinor][k*ncol + n];
                        cblas_zscal(pSPARC->locProj[ityp].nproj, &scale_f, 
                                    alpha_gnk+spinorshift+pSPARC->IP_displ_U[atmcount]*ncol + n*pSPARC->locProj[ityp].nproj, 1);
                    }

                    atmcount++;
                }
            }
        }
    }
    // End of g_nk <Orb_Jlm | Psi> storage

    double _Complex *beta1_x1, *beta2_x2, *beta3_x3; 
    beta1_x1 = alpha + IP_displ_U[atm_idx]*ncol*Nspinor*Nk;
    beta2_x2 = alpha + IP_displ_U[atm_idx]*ncol*Nspinor*Nk * 2;
    beta3_x3 = alpha + IP_displ_U[atm_idx]*ncol*Nspinor*Nk * 3;

    double _Complex *rho_block, *pre_fac, *tf_xx, *tf_yy, *tf_zz;
    double _Complex *pre_fac_tf_xx, *pre_fac_tf_yy, *pre_fac_tf_zz;
    double _Complex mult_a, mult_b, mult_c;

    for (k = 0; k < Nk; k++) {
        hub_sp = 0;
        for (spinor = 0; spinor < Nspinor; spinor++) {
            int spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * k + pSPARC->IP_displ_U[atm_idx] * ncol * spinor;

            if (pSPARC->Nspinor_spincomm == 1) {
                hub_sp = pSPARC->spincomm_index;
            } else {
                hub_sp = spinor;
            }

            atmcount = 0;
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                if (!pSPARC->atom_solve_flag[ityp]) {
                    atmcount += pSPARC->nAtomv[ityp];
                    continue;
                }

                angnum = pSPARC->locProj[ityp].nproj;
                pre_fac = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));

                // Initialise tf_xx, tf_xy, tf_xz, tf_yy, tf_yz, tf_zz
                tf_xx = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex)); 
                tf_yy = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));
                tf_zz = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));

                // Initialise pre_fac times (tf_xx, tf_xy, tf_xz, tf_yy, tf_yz, tf_zz)
                pre_fac_tf_xx = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex)); 
                pre_fac_tf_yy = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));
                pre_fac_tf_zz = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));

                // Initialize rho block
                rho_block = (double _Complex*)calloc(angnum*angnum, sizeof(double _Complex));

                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    // First reset pre_fac and pre_fac_rho_mn to 0
                    memset(pre_fac, 0, angnum*angnum*sizeof(double _Complex));

                    // rho_block = pSPARC->rho_mn[atmcount][hub_sp];

                    // Store rho_mn in rho_block
                    for (int col = 0; col < angnum; col++) {
                        for (int row = 0; row < angnum; row++) {
                            rho_block[angnum*col + row] = pSPARC->rho_mn[atmcount][hub_sp][angnum*col + row];
                        }
                    }

                    // Calculate prefactor U(\delta_{mn} - rho_{mn})
                    for (int col = 0; col < angnum; col++) {
                        for (int row = 0; row < angnum; row++) {
                            if (row == col) {
                                pre_fac[angnum*col + row] += 0.5;
                            }
                            pre_fac[angnum*col + row] -= rho_block[angnum*col + row];
                        }
                    }

                    // Scale the rows by Uval[row]
                    for (int row = 0; row < angnum; row++) {
                        for (int col = 0; col < angnum; col++) {
                            pre_fac[angnum*col + row] *= pSPARC->AtmU[ityp].Uval[row];
                        }
                    }

                    // Calculate the terms
                    // xx
                    mult_a = 2.0, mult_b = 0.0, mult_c = 1.0;
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, &mult_a,
                        beta1_x1 + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                        alpha_gnk + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                        &mult_b, tf_xx, angnum);
            
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, &mult_c,
                                pre_fac, angnum, tf_xx, angnum, &mult_b, pre_fac_tf_xx, angnum);

                    term_temp = 0.0;
                    for (int col = 0; col < angnum; col++) term_temp += creal(pre_fac_tf_xx[angnum*col + col]);
                    *pressure_hub -= pSPARC->occfac*term_temp;

                    // yy
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, &mult_a,
                        beta2_x2 + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                        alpha_gnk + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                        &mult_b, tf_yy, angnum);
                        
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, &mult_c,
                                pre_fac, angnum, tf_yy, angnum, &mult_b, pre_fac_tf_yy, angnum);

                    term_temp = 0.0;
                    for (int col = 0; col < angnum; col++) term_temp += creal(pre_fac_tf_yy[angnum*col + col]);
                    *pressure_hub -= pSPARC->occfac*term_temp;

                    // zz
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, &mult_a,
                        beta3_x3 + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                        alpha_gnk + spinorshift + pSPARC->IP_displ_U[atmcount] * ncol, angnum,
                        &mult_b, tf_zz, angnum);
                        
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, &mult_c,
                                pre_fac, angnum, tf_zz, angnum, &mult_b, pre_fac_tf_zz, angnum);

                    term_temp = 0.0;
                    for (int col = 0; col < angnum; col++) term_temp += creal(pre_fac_tf_zz[angnum*col + col]);
                    *pressure_hub -= pSPARC->occfac*term_temp;

                    atmcount++;

                }
                free(pre_fac); free(rho_block);
                free(tf_xx); free(tf_yy); free(tf_zz);
                free(pre_fac_tf_xx); free(pre_fac_tf_yy); free(pre_fac_tf_zz);
            }
        }
    }

    // free memory
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (g_nk[spinor] != NULL) free(g_nk[spinor]);
    }

    free(g_nk); free(alpha_gnk);
}