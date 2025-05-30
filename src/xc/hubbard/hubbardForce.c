/**
 * @file hubbardForce.c 
 * @brief This file contains the routines required to calculate DFT+U forces.
 * @author  Sayan Bhowmik <sbhowmik9@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Reference:   Dudarev, S. L., et al. "Electron-energy-loss spectra and the structural stability of nickel oxide:  An LSDA+U study"
 *              Phys. Rev. B 57, 1505
 */

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
/* BLAS routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#include "hubbardForce.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "lapVecRoutines.h"
#include "tools.h" 
#include "isddft.h"
#include "initialization.h"
// #include "hubbardInitialization.h"
// #include "locOrbRoutines.h"
// #include "occupationMatrix.h"

#define TEMP_TOL 1e-12

/**
 * @brief Distribute based on gamma or k-points.
 */
void Calculate_hubbard_forces(SPARC_OBJ *pSPARC) {
    if (pSPARC->isGammaPoint) {
        Calculate_hubbard_forces_linear(pSPARC);
    } else {
        Calculate_hubbard_forces_kpt(pSPARC);
    }
}

/**
 * @brief Calculate DFT+U forces for gamma point.
 */
void Calculate_hubbard_forces_linear(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, ncol, DMnd, DMndsp, spinor, Nspinor, dim;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;

    double *force_hub, *alpha, *beta;
    force_hub = (double *)calloc(3 * pSPARC->n_atom, sizeof(double));

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    alpha = (double *)calloc( pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * 4, sizeof(double));

    #ifdef DEBUG 
        if (!rank) printf("Start Calculating hubbard forces\n");
    #endif

    beta = alpha;
    Compute_Integral_psi_Orb(pSPARC, beta, pSPARC->Xorb);

    for (dim = 0; dim < 3; dim++) {
        for (spinor = 0; spinor < Nspinor; spinor++) {
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, 
                pSPARC->Xorb+spinor*DMnd, DMndsp, pSPARC->Yorb+spinor*DMnd, DMndsp, 
                dim, pSPARC->dmcomm);
        }
        beta = alpha + pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * (dim + 1);
        Compute_Integral_Orb_Dpsi(pSPARC, pSPARC->Yorb, beta);
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * 4, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate hubbard force */
    Compute_force_hubbard_by_integrals(pSPARC, force_hub, alpha);
    free(alpha);

    // sum over all spin
    if (pSPARC->npspin > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, force_hub, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

    // sum over all bands
    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, force_hub, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);        
    }

    #ifdef DEBUG
    if (!rank) {
        printf("force_hubbard = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", force_hub[i*3], force_hub[i*3+1], force_hub[i*3+2]);
        }
    }
    #endif

    if (!rank) {
        for (i = 0; i < 3 * pSPARC->n_atom; i++) {
            pSPARC->forces[i] += force_hub[i];
        }
    }

    free(force_hub);
}

/**
 * @brief Compute < Psi_n | Phi_m >
 */
void Compute_Integral_psi_Orb(SPARC_OBJ *pSPARC, double *beta, double *Xorb) {
    int i, n, ndc, ityp, iat, ncol, DMnd, atom_index;
    int spinor, Nspinor, DMndsp, *IP_displ_U, spinorshift;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;

    double *x_ptr, *x_rc, *x_rc_ptr;
    IP_displ_U = pSPARC->IP_displ_U;
    double alpha = pSPARC->dV;

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if(!pSPARC->atom_solve_flag[ityp]) continue;
        if (! pSPARC->locProj[ityp].nproj) continue; // this is typical for hydrogen
        double **Orb = pSPARC->locProj[ityp].Orb;
        for (iat = 0; iat < pSPARC->Atom_Influence_loc_orb[ityp].n_atom; iat++) {
            ndc = pSPARC->Atom_Influence_loc_orb[ityp].ndc[iat]; 
            x_rc = (double *)malloc( ndc * ncol * sizeof(double));
            atom_index = pSPARC->Atom_Influence_loc_orb[ityp].atom_index[iat];
            /* first find inner product <Psi_n, Orb_Jlm>, here we calculate <Orb_Jlm, Psi_n> instead */
            for (spinor = 0; spinor < Nspinor; spinor++) {
                for (n = 0; n < ncol; n++) {
                    x_ptr = Xorb + n * DMndsp + spinor * DMnd;
                    x_rc_ptr = x_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(x_rc_ptr + i) = *(x_ptr + pSPARC->Atom_Influence_loc_orb[ityp].grid_pos[iat][i]);
                    }
                }
                spinorshift = IP_displ_U[atm_idx] * ncol * spinor;
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->locProj[ityp].nproj, ncol, ndc, alpha, Orb[iat], ndc, 
                            x_rc, ndc, 1.0, beta+spinorshift+IP_displ_U[atom_index]*ncol, pSPARC->locProj[ityp].nproj); // multiply dV to get inner-product
            }
            free(x_rc);
        }
    }
}

/**
 * @brief Calculate < Phi_m | \nabla Psi_n >
 */
void Compute_Integral_Orb_Dpsi(SPARC_OBJ *pSPARC, double *dpsi, double *beta) {
    int i, n, ndc, ityp, iat, ncol, DMnd, atom_index;
    int spinor, Nspinor, DMndsp;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    double *dx_ptr, *dx_rc, *dx_rc_ptr;

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if(!pSPARC->atom_solve_flag[ityp]) continue;
        if (! pSPARC->locProj[ityp].nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < pSPARC->Atom_Influence_loc_orb[ityp].n_atom; iat++) {
            ndc = pSPARC->Atom_Influence_loc_orb[ityp].ndc[iat]; 
            dx_rc = (double *)malloc( ndc * ncol * sizeof(double));
            atom_index = pSPARC->Atom_Influence_loc_orb[ityp].atom_index[iat];
            for (spinor = 0; spinor < Nspinor; spinor++) {
                for (n = 0; n < ncol; n++) {
                    dx_ptr = dpsi + n * DMndsp + spinor * DMnd;
                    dx_rc_ptr = dx_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_loc_orb[ityp].grid_pos[iat][i]);
                    }
                }
                /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).
                */
                int spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * spinor;
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->locProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->locProj[ityp].Orb[iat], ndc, 
                            dx_rc, ndc, 1.0, beta+spinorshift+pSPARC->IP_displ_U[atom_index]*ncol, pSPARC->locProj[ityp].nproj); 
            }
            free(dx_rc);
        }
    }
}

/**
 * @brief Calculate forces in gamma point using integrals.
 */
void Compute_force_hubbard_by_integrals(SPARC_OBJ *pSPARC, double *force_hub, double *alpha) {
    int n, ityp, iat, ncol, atom_index, atom_index2, count;
    int spinor, Nspinor, hub_sp;
    double *pre_fac;
    int angnum;

    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Nspinor = pSPARC->Nspinor_spincomm;

    double *beta_x, *beta_y, *beta_z;
    double *integral_x, *integral_y, *integral_z;
    double *tf_x, *tf_y, *tf_z;
    double fJ_x, fJ_y, fJ_z;

    // go over all atoms and find hubbard force components
    int Ns = pSPARC->Nstates;

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    // <Orb_Jlm | Dpsi_i> are stored here
    beta_x = alpha + pSPARC->IP_displ_U[atm_idx]*ncol*Nspinor;
    beta_y = alpha + pSPARC->IP_displ_U[atm_idx]*ncol*Nspinor * 2;
    beta_z = alpha + pSPARC->IP_displ_U[atm_idx]*ncol*Nspinor * 3;

    // Extract local g_n as array
    int nstart = pSPARC->band_start_indx;
    int nend = pSPARC->band_end_indx;
    double **g_n = (double **)calloc((pSPARC->Nspinor_spincomm), sizeof(double*));
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        g_n[spinor] = (double *)calloc(ncol, sizeof(double));
    }
    
    count = 0;
    for (n = nstart; n <= nend; n++) {
        double woccfac = pSPARC->occfac;
        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
            double *occ = pSPARC->occ; 
            if (pSPARC->spin_typ == 1) occ += spinor*Ns;
            
            g_n[spinor][count] = woccfac * occ[n];
        }
        count++;
    }
    // End of extract local g_n as array

    // g_n <Orb_Jlm | Psi> stored here
    double *alpha_gn = (double *)calloc( pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor, sizeof(double));
    // hub_sp = 0;
    for (spinor = 0; spinor < Nspinor; spinor++) {
        int spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * spinor;

        atom_index = 0; 

        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->atom_solve_flag[ityp]) continue;

            angnum = pSPARC->AtmU[ityp].angnum;
            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                // Scale each column of < Orb_Jlm, x_n > with g_n (size: m x Ns_loc)
                cblas_dcopy(pSPARC->locProj[ityp].nproj * ncol, alpha +spinorshift+pSPARC->IP_displ_U[atom_index]*ncol, 1, 
                    alpha_gn + spinorshift + pSPARC->IP_displ_U[atom_index] * ncol, 1); // copy into alpha_gn
                
                for (n = 0; n < ncol; n++) { // scale each column
                    cblas_dscal(pSPARC->locProj[ityp].nproj, g_n[spinor][n], 
                                alpha_gn +spinorshift+pSPARC->IP_displ_U[atom_index]*ncol + n*pSPARC->locProj[ityp].nproj, 1);
                }

                atom_index++;
            }
        }
    }
    // End of g_n <Orb_Jlm | Psi> storage
    
    hub_sp = 0;
    for (spinor = 0; spinor < Nspinor; spinor++) {
        int spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * spinor;

        if (pSPARC->Nspinor_spincomm == 1) {
            hub_sp = pSPARC->spincomm_index;
        } else {
            hub_sp = spinor;
        }

        atom_index = 0; atom_index2 = 0;

        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->atom_solve_flag[ityp]) {
                continue;
                atom_index += pSPARC->nAtomv[ityp];
            }

            angnum = pSPARC->AtmU[ityp].angnum;

            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                fJ_x = fJ_y = fJ_z = 0.0;

                // Calculate prefactor U(\delta_{mn} - rho_{mn})
                pre_fac = (double *)calloc(angnum*angnum, sizeof(double));
                for (int col = 0; col < angnum; col++) {
                    for (int row = 0; row < angnum; row++) {
                        if (row == col) {
                            pre_fac[angnum*col + row] += 0.5;
                        }
                        pre_fac[angnum*col + row] -= pSPARC->rho_mn[atom_index][hub_sp][angnum*col + row];
                    }
                }

                // Scale the rows by Uval[row]
                for (int row = 0; row < angnum; row++) {
                    for (int col = 0; col < angnum; col++) {
                        pre_fac[angnum*col + row] *= pSPARC->AtmU[ityp].Uval[row];
                    }
                }

                // Initialise
                integral_x = (double *)calloc(angnum*angnum, sizeof(double));
                tf_x = (double *)calloc(angnum*angnum, sizeof(double));

                integral_y = (double *)calloc(angnum*angnum, sizeof(double));
                tf_y = (double *)calloc(angnum*angnum, sizeof(double));

                integral_z = (double *)calloc(angnum*angnum, sizeof(double));
                tf_z = (double *)calloc(angnum*angnum, sizeof(double));

                // x - direction
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, 2.0,
                            alpha_gn + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, 
                            beta_x + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, 0.0,
                            integral_x, angnum);

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, 1.0,
                            pre_fac, angnum, integral_x, angnum, 0.0, tf_x, angnum);

                // y - direction
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, 2.0,
                    alpha_gn + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, 
                    beta_y + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, 0.0,
                    integral_y, angnum);

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, 1.0,
                            pre_fac, angnum, integral_y, angnum, 0.0, tf_y, angnum);

                // z - direction
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, 2.0,
                    alpha_gn + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, 
                    beta_z + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, 0.0,
                    integral_z, angnum);

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, 1.0,
                            pre_fac, angnum, integral_z, angnum, 0.0, tf_z, angnum);

                // Calculate the force contributions
                for (int row = 0; row < angnum; row++) {
                    fJ_x += tf_x[row*angnum + row];
                    fJ_y += tf_y[row*angnum + row];
                    fJ_z += tf_z[row*angnum + row];
                }

                // Update force vectors
                force_hub[atom_index*3  ] -= fJ_x;
                force_hub[atom_index*3+1] -= fJ_y;
                force_hub[atom_index*3+2] -= fJ_z;

                // free
                free(pre_fac); 
                free(integral_x); free(tf_x);
                free(integral_y); free(tf_y);
                free(integral_z); free(tf_z);

                // Update counters
                atom_index++; atom_index2++;
            }
        }
    }

    // free memory
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (g_n[spinor] != NULL) free(g_n[spinor]);
    }

    free(g_n); free(alpha_gn);
}

/**
 * @brief Calculate DFT+U forces in k-point cases.
 */
void Calculate_hubbard_forces_kpt(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i, ncol, DMnd, dim, kpt, Nk;
    int spinor, Nspinor, DMndsp, size_k;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    Nk = pSPARC->Nkpts_kptcomm;
    size_k = DMndsp * ncol;

    double _Complex *alpha, *beta;
    alpha = NULL;
    double *force_hub;

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    alpha = (double _Complex *)calloc( pSPARC->IP_displ_U[atm_idx] * ncol * Nk * Nspinor * 4, sizeof(double _Complex));

    force_hub = (double *)calloc(3 * pSPARC->n_atom, sizeof(double));
    double k1, k2, k3, kpt_vec[3];

    #ifdef DEBUG 
    if (!rank) printf("Start Calculating hubbard forces for spinor wavefunctions...\n");
    #endif

    for(kpt = 0; kpt < Nk; kpt++) {
        beta = alpha + pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * kpt;
        Compute_Integral_psi_Orb_kpt(pSPARC, beta, pSPARC->Xorb_kpt+kpt*size_k, kpt);
    }

    for (dim = 0; dim < 3; dim++) {
        for (kpt = 0; kpt < Nk; kpt++) {
            k1 = pSPARC->k1_loc[kpt];
            k2 = pSPARC->k2_loc[kpt];
            k3 = pSPARC->k3_loc[kpt];
            *kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);

            for (spinor = 0; spinor < Nspinor; spinor++) {
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+kpt*size_k+spinor*DMnd, DMndsp, 
                    pSPARC->Yorb_kpt+spinor*DMnd, DMndsp, dim, kpt_vec, pSPARC->dmcomm);
            }
            beta = alpha + pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * (Nk * (dim + 1) + kpt);
            Compute_Integral_Orb_Dpsi_kpt(pSPARC, pSPARC->Yorb_kpt, beta, kpt);
        }
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ_U[atm_idx] * ncol * Nk * Nspinor * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }

    Compute_force_hubbard_by_integrals_kpt(pSPARC, force_hub, alpha);
    free(alpha);

    // sum over all spin
    if (pSPARC->npspin > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, force_hub, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

    // sum over all kpoints
    if (pSPARC->npkpt > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, force_hub, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, force_hub, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);        
    }

    #ifdef DEBUG
    if (!rank) {
        printf("force_hubbard = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", force_hub[i*3], force_hub[i*3+1], force_hub[i*3+2]);
        }
    } 
    #endif

    if (!rank) {
        for (i = 0; i < 3 * pSPARC->n_atom; i++) {
            pSPARC->forces[i] += force_hub[i];
        }
    }
    free(force_hub);
}

/**
 * @brief Calculate < Psi_nk | Phi_m > for k-pts
 */
void Compute_Integral_psi_Orb_kpt(SPARC_OBJ *pSPARC, double _Complex *beta, double _Complex *Xorb_kpt, int kpt) {
    int i, n, ndc, ityp, iat, ncol, DMnd, atom_index;
    int spinor, Nspinor, DMndsp, nproj, spinorshift, *IP_displ_U;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;

    double _Complex *x_ptr, *x_rc, *x_rc_ptr;

    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, x0_i, y0_i, z0_i;
    double _Complex bloch_fac, a, b, **Orb = NULL;

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

        if(!nproj) continue;
        for (iat = 0; iat < pSPARC->Atom_Influence_loc_orb[ityp].n_atom; iat++) {
            x0_i = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3  ];
            y0_i = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3+1];
            z0_i = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) - sin(theta) * I;
            a = (bloch_fac * pSPARC->dV);
            b = 1.0;
            ndc = pSPARC->Atom_Influence_loc_orb[ityp].ndc[iat];
            x_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
            atom_index = pSPARC->Atom_Influence_loc_orb[ityp].atom_index[iat];
            /* first find inner product <Psi_n, orb_Jlm>, here we calculate <Orb_Jlm, Psi_n> instead */
            for (spinor = 0; spinor < Nspinor; spinor++) {
                for (n = 0; n < ncol; n++) {
                    x_ptr = Xorb_kpt + n * DMndsp + spinor * DMnd;
                    x_rc_ptr = x_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(x_rc_ptr + i) = conj(*(x_ptr + pSPARC->Atom_Influence_loc_orb[ityp].grid_pos[iat][i]));
                    }
                }

                spinorshift = IP_displ_U[atm_idx] * ncol * spinor;
                cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nproj, ncol, ndc, &a, Orb[iat], ndc, 
                            x_rc, ndc, &b, beta+spinorshift+IP_displ_U[atom_index]*ncol, nproj); // multiply dV to get inner-product
            }
            free(x_rc);
        }
    }
}

/**
 * @brief Calculate < Phi_m | \nabla Psi_nk > for k points.
 */
void Compute_Integral_Orb_Dpsi_kpt(SPARC_OBJ *pSPARC, double _Complex *dpsi, double _Complex *beta, int kpt) {
    int i, n, ndc, ityp, iat, ncol, DMnd, atom_index;
    int spinor, Nspinor, DMndsp, spinorshift, *IP_displ_U, nproj;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;

    double _Complex *dx_ptr, *dx_rc, *dx_rc_ptr;
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, x0_i, y0_i, z0_i;
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

        if(!nproj) continue;

        for (iat = 0; iat < pSPARC->Atom_Influence_loc_orb[ityp].n_atom; iat++) {
            x0_i = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3  ];
            y0_i = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3+1];
            z0_i = pSPARC->Atom_Influence_loc_orb[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            b = 1.0;
            ndc = pSPARC->Atom_Influence_loc_orb[ityp].ndc[iat]; 
            dx_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
            atom_index = pSPARC->Atom_Influence_loc_orb[ityp].atom_index[iat];
            for (spinor = 0; spinor < Nspinor; spinor++) {
                for (n = 0; n < ncol; n++) {
                    dx_ptr = dpsi + n * DMndsp + spinor * DMnd;
                    dx_rc_ptr = dx_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_loc_orb[ityp].grid_pos[iat][i]);
                    }
                }

                /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).
                */
                spinorshift = IP_displ_U[atm_idx] * ncol * spinor;
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nproj, ncol, ndc, &bloch_fac, Orb[iat], ndc, 
                    dx_rc, ndc, &b, beta+spinorshift+IP_displ_U[atom_index]*ncol, nproj);
            }
            free(dx_rc);
        }
    }
}

/**
 * @brief Calculate k-point DFT+U forces using integral.
 */
void Compute_force_hubbard_by_integrals_kpt(SPARC_OBJ *pSPARC, double *force_hub, double _Complex *alpha) {
    int k, n, ityp, iat, ncol, atom_index, atom_index2, count, Nk;
    int spinor, Nspinor, hub_sp, *IP_displ_U;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Nk = pSPARC->Nkpts_kptcomm;
    Nspinor = pSPARC->Nspinor_spincomm;

    double _Complex *pre_fac, *integral_x, *integral_y, *integral_z, *tf_x, *tf_y, *tf_z, *beta_x, *beta_y, *beta_z;
    double fJ_x, fJ_y, fJ_z;
    int angnum;

    // go over all atoms and find nonlocal force components
    int Ns = pSPARC->Nstates;
    // double k1, k2, k3;
    // double _Complex a, b, bloch_fac;

    int atm_idx = -1;
    for (int JJ = pSPARC->n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }
    
    IP_displ_U = pSPARC->IP_displ_U;
    // <Orb_Jlm | Psi> stored here
    beta_x = alpha + IP_displ_U[atm_idx]*ncol*Nk*Nspinor;
    beta_y = alpha + IP_displ_U[atm_idx]*ncol*Nk*Nspinor * 2;
    beta_z = alpha + IP_displ_U[atm_idx]*ncol*Nk*Nspinor * 3;

    // Extract local g_nk as array
    int nstart = pSPARC->band_start_indx;
    int nend = pSPARC->band_end_indx;
    double **g_nk = (double **)calloc((pSPARC->Nspinor_spincomm), sizeof(double*));
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        g_nk[spinor] = (double *)calloc(Nk*ncol, sizeof(double));
    }
    
    count = 0;
    for (k = 0; k < Nk; k++) {
        for (n = nstart; n <= nend; n++) {
            double woccfac = pSPARC->occfac*(pSPARC->kptWts_loc[k] / pSPARC->Nkpts);
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

    for (k = 0; k < Nk; k++) {
        for (spinor = 0; spinor < Nspinor; spinor++) {
            int spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * k + pSPARC->IP_displ_U[atm_idx] * ncol * spinor;

            atom_index = 0;

            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                if (!pSPARC->atom_solve_flag[ityp]) continue;

                angnum = pSPARC->AtmU[ityp].angnum;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    // Scale each column of < Orb_Jlm, x_n > with g_n (size: m x Ns_loc)
                    cblas_zcopy(pSPARC->locProj[ityp].nproj * ncol, alpha + spinorshift + pSPARC->IP_displ_U[atom_index]*ncol, 1, 
                        alpha_gnk + spinorshift + pSPARC->IP_displ_U[atom_index] * ncol, 1); // copy into alpha_gn

                    for (n = 0; n < ncol; n++) { // scale each column
                        double _Complex scale_f = g_nk[spinor][k*ncol + n];
                        cblas_zscal(pSPARC->locProj[ityp].nproj, &scale_f, 
                                    alpha_gnk+spinorshift+pSPARC->IP_displ_U[atom_index]*ncol + n*pSPARC->locProj[ityp].nproj, 1);
                    }

                    atom_index++;
                }
            }
        }
    }
    // End of g_nk <Orb_Jlm | Psi> storage

    for (k = 0; k < Nk; k++) {
        hub_sp = 0;
        for (spinor = 0; spinor < Nspinor; spinor++) {
            int spinorshift = pSPARC->IP_displ_U[atm_idx] * ncol * Nspinor * k + pSPARC->IP_displ_U[atm_idx] * ncol * spinor;

            if (pSPARC->Nspinor_spincomm == 1) {
                hub_sp = pSPARC->spincomm_index;
            } else {
                hub_sp = spinor;
            }

            atom_index = 0; atom_index2 = 0;

            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                if (!pSPARC->atom_solve_flag[ityp]) {
                    continue;
                    atom_index += pSPARC->nAtomv[ityp];
                }

                angnum = pSPARC->AtmU[ityp].angnum;

                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    fJ_x = fJ_y = fJ_z = 0.0;

                    // Calculate prefactor U(\delta_{mn} - rho_{mn})
                    pre_fac = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));
                    for (int col = 0; col < angnum; col++) {
                        for (int row = 0; row < angnum; row++) {
                            if (row == col) {
                                pre_fac[angnum*col + row] += 0.5;
                            }
                            pre_fac[angnum*col + row] -= pSPARC->rho_mn[atom_index][hub_sp][angnum*col + row];
                        }
                    }

                    // Scale the rows by Uval[row]
                    for (int row = 0; row < angnum; row++) {
                        for (int col = 0; col < angnum; col++) {
                            pre_fac[angnum*col + row] *= pSPARC->AtmU[ityp].Uval[row];
                        }
                    }

                    // Initialise
                    integral_x = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));
                    tf_x = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));

                    integral_y = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));
                    tf_y = (double _Complex*)calloc(angnum*angnum, sizeof(double _Complex));

                    integral_z = (double _Complex*)calloc(angnum*angnum, sizeof(double _Complex));
                    tf_z = (double _Complex*)calloc(angnum*angnum, sizeof(double _Complex));

                    double _Complex mult_a = 2.0, mult_b = 0.0, mult_c = 1.0;

                    // x - direction
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, &mult_a,
                                alpha_gnk + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, 
                                beta_x + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, &mult_b,
                                integral_x, angnum);

                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, &mult_c,
                                pre_fac, angnum, integral_x, angnum, &mult_b, tf_x, angnum);

                    // y - direction
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, &mult_a,
                        alpha_gnk + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, 
                        beta_y + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, &mult_b,
                        integral_y, angnum);

                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, &mult_c,
                                pre_fac, angnum, integral_y, angnum, &mult_b, tf_y, angnum);

                    // z - direction
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, angnum, angnum, ncol, &mult_a,
                        alpha_gnk + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, 
                        beta_z + spinorshift + pSPARC->IP_displ_U[atom_index2] * ncol, angnum, &mult_b,
                        integral_z, angnum);

                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, angnum, angnum, &mult_c,
                                pre_fac, angnum, integral_z, angnum, &mult_b, tf_z, angnum);

                    // Calculate the force contributions
                    for (int row = 0; row < angnum; row++) {
                        fJ_x += creal(tf_x[row*angnum + row]);
                        fJ_y += creal(tf_y[row*angnum + row]);
                        fJ_z += creal(tf_z[row*angnum + row]);
                    }

                    // Update force vectors
                    force_hub[atom_index*3  ] -= fJ_x;
                    force_hub[atom_index*3+1] -= fJ_y;
                    force_hub[atom_index*3+2] -= fJ_z;

                    // free
                    free(pre_fac); 
                    free(integral_x); free(tf_x);
                    free(integral_y); free(tf_y);
                    free(integral_z); free(tf_z);

                    // Update counters
                    atom_index++; atom_index2++;
                }
            }
        }
    }

    // free memory
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (g_nk[spinor] != NULL) free(g_nk[spinor]);
    }

    free(g_nk); free(alpha_gnk);
}