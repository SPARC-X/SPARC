/**
 * @file    forces.c
 * @brief   This file contains the functions for calculating forces.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
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

#include "forces.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "lapVecRoutines.h"
#include "tools.h" 
#include "isddft.h"
#include "initialization.h"
#include "electrostatics.h"
#include "spinOrbitCoupling.h"
#include "sqProperties.h"
#include "d3forceStress.h"

#define TEMP_TOL 1e-12

/**
 * @brief    Calculate atomic forces.
 */
void Calculate_EGS_Forces(SPARC_OBJ *pSPARC)
{
    int rank, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG    
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    // find local force components
    Calculate_local_forces(pSPARC);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Time for calculating local force components: %.3f ms\n", (t2 - t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // find nonlocal force components
    Calculate_nonlocal_forces(pSPARC);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Time for calculating nonlocal force components: %.3f ms\n", (t2 - t1)*1e3);
#endif

    // Convert the forces from non cartesian to cartesian coordinates
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(i = 0; i < pSPARC->n_atom; i++){
             nonCart2Cart_grad(pSPARC, pSPARC->forces + 3*i, pSPARC->forces + 3*i+1, pSPARC->forces + 3*i+2);
        }
    }      
    
    // calculate f_xc for non-linear core correction (NLCC)
    if (pSPARC->NLCC_flag) {
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        double *forces_xc = (double *)calloc(3*pSPARC->n_atom, sizeof(double));
        void Calculate_forces_xc(SPARC_OBJ *pSPARC, double *forces_xc);
        Calculate_forces_xc(pSPARC, forces_xc); // note that forces_xc is already in Cartesian basis
        for (i = 0; i < 3 * pSPARC->n_atom; i++) {
            pSPARC->forces[i] += forces_xc[i];
        }
#ifdef DEBUG
        if (rank == 0) {
            // print forces_xc
            printf(GRN "forces_xc: \n" RESET);
            for (i = 0; i < pSPARC->n_atom; i++) {
                printf(GRN "%18.14f %18.14f %18.14f\n" RESET, 
                    forces_xc[3*i],forces_xc[3*i+1],forces_xc[3*i+2]);
            }
        }
        t2 = MPI_Wtime();
        if (!rank) printf("Time for calculating XC forces components: %.3f ms\n", (t2-t1)*1e3);
#endif
        free(forces_xc);
    }  

    if (pSPARC->d3Flag == 1) 
        add_d3_forces(pSPARC);

    // make the sum of the forces zero
    Symmetrize_forces(pSPARC);

    // Apply constraint on motion of atoms for Relaxation\MD
    if(pSPARC->RelaxFlag == 1 || pSPARC->MDFlag == 1) {
        for(i = 0; i < 3*pSPARC->n_atom; i++)
            pSPARC->forces[i] *= pSPARC->mvAtmConstraint[i];
    }
    
#ifdef DEBUG    
    if (!rank) {
        printf(" Cartesian force = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n",pSPARC->forces[i*3], pSPARC->forces[i*3+1], pSPARC->forces[i*3+2]);
        }
    }
#endif
}



/**
 * @brief    Symmetrize the force components so that sum of forces is zero.
 */
void Symmetrize_forces(SPARC_OBJ *pSPARC)
{
    // consider broadcasting the force components or force residual
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 
    int i, n_atom;
    double shift_fx, shift_fy, shift_fz;
    shift_fx = shift_fy = shift_fz = 0.0;
    
    n_atom = pSPARC->n_atom;
    for (i = 0; i < n_atom; i++) {
        shift_fx += pSPARC->forces[i*3];
        shift_fy += pSPARC->forces[i*3+1];
        shift_fz += pSPARC->forces[i*3+2];
    }
    
    shift_fx /= n_atom;
    shift_fy /= n_atom;
    shift_fz /= n_atom;
    for (i = 0; i < n_atom; i++) {
        pSPARC->forces[i*3] -= shift_fx;
        pSPARC->forces[i*3+1] -= shift_fy;
        pSPARC->forces[i*3+2] -= shift_fz;
    }
}


/**
 * @brief    Calculate nonlocal force components.
 */
void Calculate_nonlocal_forces(SPARC_OBJ *pSPARC)
{
    if (pSPARC->isGammaPoint) {
        if (pSPARC->SQFlag == 1) {
            Calculate_nonlocal_forces_SQ(pSPARC);
        } else {
            Calculate_nonlocal_forces_linear(pSPARC);
        }
    } else {
        if (pSPARC->Nspinor == 1)
            Calculate_nonlocal_forces_kpt_linear(pSPARC); 
        else if (pSPARC->Nspinor == 2) 
            Calculate_nonlocal_forces_kpt_spinor_linear(pSPARC); 
    }
}


/**
 * @brief    Calculate nonlocal force components.
 */
void Calculate_nonlocal_forces_linear(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, n, np, ldispl, ndc, ityp, iat, ncol, DMnd, dim, atom_index, count, l, m, lmax, spn_i, nspin, size_s;
    nspin = pSPARC->Nspin_spincomm; // number of spin in my spin communicator
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    size_s = ncol * DMnd;
    double *force_nloc, *alpha, *beta, *x_ptr, *dx_ptr, *x_rc, *dx_rc, *x_rc_ptr, *dx_rc_ptr;
    double fJ_x, fJ_y, fJ_z, val_x, val_y, val_z, val2_x, val2_y, val2_z, g_nk, *beta_x, *beta_y,
           *beta_z;
    
    force_nloc = (double *)calloc(3 * pSPARC->n_atom, sizeof(double));
    alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 4, sizeof(double));
#ifdef DEBUG 
    if (!rank) printf("Start Calculating nonlocal forces\n");
#endif
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * count;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            //lmax = pSPARC->psd[ityp].lmax;
            if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
            for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat]; 
                x_rc = (double *)malloc( ndc * ncol * sizeof(double));
                atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                /* first find inner product <Psi_n, Chi_Jlm>, here we calculate <Chi_Jlm, Psi_n> instead */
                for (n = 0; n < ncol; n++) {
                    x_ptr = pSPARC->Xorb + spn_i * size_s + n * DMnd;
                    x_rc_ptr = x_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        // x_rc[n*ndc+i] = pSPARC->Xorb[n*DMnd+pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                        *(x_rc_ptr + i) = *(x_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                    }
                }
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, pSPARC->dV, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                            x_rc, ndc, 1.0, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiply dV to get inner-product
                //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncol, pSPARC->nlocProj[ityp].nproj, ndc, pSPARC->dV, x_rc, ndc, 
                //            pSPARC->nlocProj[ityp].Chi[iat], ndc, 1.0, alpha+pSPARC->IP_displ[atom_index]*ncol, ncol); // this calculates <Psi_n, Chi_Jlm>
                free(x_rc);
                // /* find inner product <Chi_Jlm, dPsi_n> */
                // dx_rc = (double *)malloc( ndc * ncol * sizeof(double));
                // for (dim = 0; dim < 3; dim++) {
                //     // find dPsi in direction dim
                //     Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb, pSPARC->Yorb, dim, pSPARC->dmcomm);
                //     beta = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*(dim+1);
                //     atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                //     for (n = 0; n < ncol; n++) {
                //         dx_ptr = pSPARC->Yorb + n * DMnd;
                //         dx_rc_ptr = dx_rc + n * ndc;
                //         for (i = 0; i < ndc; i++) {
                //             // dx_rc[n*ndc+i] = pSPARC->Yorb[n*DMnd+pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                //             *(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                //         }
                //     }
                //     /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                //     *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                //     *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                //     *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).
                //     */      
                //     cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                //     dx_rc, ndc, 1.0, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); 
                // }
                // free(dx_rc);
            }
        }
        count++;
    }    
    
    /* find inner product <Chi_Jlm, dPsi_n> */
    for (dim = 0; dim < 3; dim++) {
        count = 0;
        for(spn_i = 0; spn_i < nspin; spn_i++) {
            // find dPsi in direction dim
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, pSPARC->Yorb, dim, pSPARC->dmcomm);
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (nspin * (dim + 1) + count);
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                //lmax = pSPARC->psd[ityp].lmax;
                if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat]; 
                    dx_rc = (double *)malloc( ndc * ncol * sizeof(double));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    for (n = 0; n < ncol; n++) {
                        dx_ptr = pSPARC->Yorb + n * DMnd;
                        dx_rc_ptr = dx_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            // dx_rc[n*ndc+i] = pSPARC->Yorb[n*DMnd+pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                            *(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                        }
                    }
                    /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                     *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                     *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                     *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).
                     */      
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                                dx_rc, ndc, 1.0, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); 
                    free(dx_rc);
                }
            }
            count++;
        }    
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 4, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal force */
    // go over all atoms and find nonlocal force components
    beta_x = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin;
    beta_y = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 2;
    beta_z = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 3;
    count = 0; atom_index = 0;
    double spn_fac;


    for(spn_i = 0; spn_i < nspin; spn_i++) {
        atom_index = 0;
        spn_fac = 2.0/pSPARC->Nspin * 2.0;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            int lloc = pSPARC->localPsd[ityp];
            lmax = pSPARC->psd[ityp].lmax;
            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                fJ_x = fJ_y = fJ_z = 0.0;
                //alpha_J = alpha + pSPARC->IP_displ[atom_index]*ncol;
                //beta_Jx = beta_x + pSPARC->IP_displ[atom_index]*ncol;
                //beta_Jy = beta_y + pSPARC->IP_displ[atom_index]*ncol;
                //beta_Jz = beta_z + pSPARC->IP_displ[atom_index]*ncol;
                for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                    g_nk = pSPARC->occ[spn_i*pSPARC->Nstates+n];
                    val2_x = val2_y = val2_z = 0.0;
                    ldispl = 0;
                    for (l = 0; l <= lmax; l++) {
                        // skip the local l
                        if (l == lloc) {
                            ldispl += pSPARC->psd[ityp].ppl[l];
                            continue;
                        }
                        for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                            val_x = val_y = val_z = 0.0;
                            for (m = -l; m <= l; m++) {
                                val_x += alpha[count] * beta_x[count];
                                val_y += alpha[count] * beta_y[count];
                                val_z += alpha[count] * beta_z[count];
                                count++;
                            }
                            val2_x += val_x * pSPARC->psd[ityp].Gamma[ldispl+np];
                            val2_y += val_y * pSPARC->psd[ityp].Gamma[ldispl+np];
                            val2_z += val_z * pSPARC->psd[ityp].Gamma[ldispl+np];
                        }
                        ldispl += pSPARC->psd[ityp].ppl[l];
                    }
                    fJ_x += val2_x * g_nk;
                    fJ_y += val2_y * g_nk;
                    fJ_z += val2_z * g_nk;
                }
                
                force_nloc[atom_index*3  ] -= spn_fac * fJ_x;
                force_nloc[atom_index*3+1] -= spn_fac * fJ_y;
                force_nloc[atom_index*3+2] -= spn_fac * fJ_z;
                atom_index++;
            }
        }
    }    
    
    // sum over all spin
    if (pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(force_nloc, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
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
    free(alpha);
}


/**
 * @brief    Calculate nonlocal force components with kpts.
 */
void Calculate_nonlocal_forces_kpt_linear(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, k, n, np, ldispl, ndc, ityp, iat, ncol, DMnd, dim, atom_index, count, l, m, lmax, kpt, Nk, size_k, spn_i, nspin, size_s;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;    
    size_s = size_k * Nk;
    double complex *alpha, *beta, *x_ptr, *dx_ptr, *x_rc, *dx_rc, *x_rc_ptr, *dx_rc_ptr, *beta_x, *beta_y, *beta_z;
    double *force_nloc, fJ_x, fJ_y, fJ_z, val_x, val_y, val_z, val2_x, val2_y, val2_z, g_nk;
    
    force_nloc = (double *)calloc(3 * pSPARC->n_atom, sizeof(double));
    alpha = (double complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 4, sizeof(double complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, kpt_vec, x0_i, y0_i, z0_i;
    double complex bloch_fac, a, b;
    
#ifdef DEBUG 
    if (!rank) printf("Start Calculating nonlocal forces\n");
#endif
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++) {
            k1 = pSPARC->k1_loc[kpt];
            k2 = pSPARC->k2_loc[kpt];
            k3 = pSPARC->k3_loc[kpt];
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * count;
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                //lmax = pSPARC->psd[ityp].lmax;
                if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
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
                    for (n = 0; n < ncol; n++) {
                        x_ptr = pSPARC->Xorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd;
                        x_rc_ptr = x_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            // x_rc[n*ndc+i] = pSPARC->Xorb[n*DMnd+pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                            //printf("grid_pos % d\n", pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                            *(x_rc_ptr + i) = conj(*(x_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]));
                        }
                    }
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &a, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                x_rc, ndc, &b, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiply dV to get inner-product
                    //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncol, pSPARC->nlocProj[ityp].nproj, ndc, pSPARC->dV, x_rc, ndc, 
                    //            pSPARC->nlocProj[ityp].Chi[iat], ndc, 1.0, alpha+pSPARC->IP_displ[atom_index]*ncol, ncol); // this calculates <Psi_n, Chi_Jlm>
                    free(x_rc);
                }
            }
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
                // find dPsi in direction dim
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, pSPARC->Yorb_kpt, dim, kpt_vec, pSPARC->dmcomm);
                beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin * (dim + 1) + count);
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                    //lmax = pSPARC->psd[ityp].lmax;
                    if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
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
                        
                        for (n = 0; n < ncol; n++) {
                            dx_ptr = pSPARC->Yorb_kpt + n * DMnd;
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
                        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                    dx_rc, ndc, &b, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);   
                        
                        free(dx_rc);
                    }
                }
                count++; 
            }
        }    
    }
        
    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }
    
    /* calculate nonlocal force */
    // go over all atoms and find nonlocal force components
    int Ns = pSPARC->Nstates;
    double kpt_spn_fac;
    beta_x = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin;
    beta_y = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 2;
    beta_z = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 3;
    count = 0; 
    
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            kpt_spn_fac = (2.0/pSPARC->Nspin) * 2.0 * pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
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
                        for (l = 0; l <= lmax; l++) {
                            // skip the local l
                            if (l == lloc) {
                                ldispl += pSPARC->psd[ityp].ppl[l];
                                continue;
                            }
                            for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                                val_x = val_y = val_z = 0.0;
                                for (m = -l; m <= l; m++) {
                                    val_x += creal(alpha[count]) * creal(beta_x[count]) - cimag(alpha[count]) * cimag(beta_x[count]);
                                    val_y += creal(alpha[count]) * creal(beta_y[count]) - cimag(alpha[count]) * cimag(beta_y[count]);
                                    val_z += creal(alpha[count]) * creal(beta_z[count]) - cimag(alpha[count]) * cimag(beta_z[count]);
                                    count++;
                                }
                                val2_x += val_x * pSPARC->psd[ityp].Gamma[ldispl+np];
                                val2_y += val_y * pSPARC->psd[ityp].Gamma[ldispl+np];
                                val2_z += val_z * pSPARC->psd[ityp].Gamma[ldispl+np];
                            }
                            ldispl += pSPARC->psd[ityp].ppl[l];
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
    free(alpha);
}



/**
 * @brief    Calculate local force components.
 */
void Calculate_local_forces(SPARC_OBJ *pSPARC)
{
    Calculate_local_forces_linear(pSPARC);
}



/**
 * @brief    Calculate local force components
 */ 
void Calculate_local_forces_linear(SPARC_OBJ *pSPARC)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; // consider broadcasting the force components or force residual
    
    int ityp, iat, i, j, k, p, ip, jp, kp, i_DM, j_DM, k_DM, FDn, count, count_interp,
        DMnx, DMny, DMnd, nx, ny, nz, nd, nxp, nyp, nzp, nd_ex, 
        icor, jcor, kcor, atom_index, *ind_interp, dK, dJ, dI;
    double x0_i, y0_i, z0_i, x0_i_shift, y0_i_shift, z0_i_shift, x, y, z, *R,
           *VJ, *VJ_ref, *VcJ,  
           DVcJ_x_val, DVcJ_y_val, DVcJ_z_val, force_x, force_y, force_z, force_corr_x, 
           force_corr_y, force_corr_z, *R_interp, *VJ_interp;
    double inv_4PI = 0.25 / M_PI, w2_diag, rchrg;
    int *pshifty_ex, *pshiftz_ex;
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
#ifdef DEBUG    
    if (!rank) printf("Start calculating local components of forces ...\n");
#endif
    double t1, t2, t_sort = 0.0;
    double *Lap_wt, *Lap_stencil;
    
    FDn = pSPARC->order / 2;
    w2_diag = (pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_y[0] + pSPARC->D2_stencil_coeffs_z[0]) * -inv_4PI;
    if(pSPARC->cell_typ == 0){
        Lap_wt = (double *)malloc((3*(FDn+1))*sizeof(double));
        Lap_stencil = Lap_wt;
    } else{
        Lap_wt = (double *)malloc((5*(FDn+1))*sizeof(double));
        Lap_stencil = Lap_wt+5;
        //nonCart2Cart_coord(pSPARC, &delta_x, &delta_y, &delta_z);
    }
    Lap_stencil_coef_compact(pSPARC, FDn, Lap_stencil, -inv_4PI);

    // Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz;
    DMnx = pSPARC->Nx_d; DMny = pSPARC->Ny_d; // DMnz = pSPARC->Nz_d;
    DMnd = pSPARC->Nd_d;
    
    // initialize force components to zero
    for (i = 0; i < 3 * pSPARC->n_atom; i++) {
        pSPARC->forces[i] = 0.0;
    }
    
    // Create indices for laplacian
    pshifty_ex = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz_ex = (int *)malloc( (FDn+1) * sizeof(int));
    
    if (pshifty_ex == NULL || pshiftz_ex == NULL) {
        printf("\nMemory allocation failed in local forces!\n");
        exit(EXIT_FAILURE);
    }
    
    // find gradient of phi
    double *Dphi_x, *Dphi_y, *Dphi_z, *DVc_x, *DVc_y, *DVc_z;
    Dphi_x = (double *)malloc( DMnd * sizeof(double));
    Dphi_y = (double *)malloc( DMnd * sizeof(double));
    Dphi_z = (double *)malloc( DMnd * sizeof(double));
    DVc_x = (double *)malloc( DMnd * sizeof(double));
    DVc_y = (double *)malloc( DMnd * sizeof(double));
    DVc_z = (double *)malloc( DMnd * sizeof(double));
    
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_z, 2, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->Vc, DVc_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->Vc, DVc_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->Vc, DVc_z, 2, pSPARC->dmcomm_phi);
    
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        for (iat = 0; iat < pSPARC->Atom_Influence_local[ityp].n_atom; iat++) {
            // coordinates of the image atom
            x0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3];
            y0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 1];
            z0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 2];
            // original atom index this image atom corresponds to
            atom_index = pSPARC->Atom_Influence_local[ityp].atom_index[iat];
            
            // number of finite-difference nodes in each direction of overlap rb region
            nx = pSPARC->Atom_Influence_local[ityp].xe[iat] - pSPARC->Atom_Influence_local[ityp].xs[iat] + 1;
            ny = pSPARC->Atom_Influence_local[ityp].ye[iat] - pSPARC->Atom_Influence_local[ityp].ys[iat] + 1;
            nz = pSPARC->Atom_Influence_local[ityp].ze[iat] - pSPARC->Atom_Influence_local[ityp].zs[iat] + 1;
            nd = nx * ny * nz;
            // number of finite-difference nodes in each direction of extended rb (+ order/2) region
            nxp = nx + pSPARC->order;
            nyp = ny + pSPARC->order;
            nzp = nz + pSPARC->order;
            nd_ex = nxp * nyp * nzp; // total number of nodes
            
            pshifty_ex[0] = pshiftz_ex[0] = 0;
            for (p = 1; p <= FDn; p++) {
                // for x_ex
                pshifty_ex[p] = p * nxp;
                pshiftz_ex[p] = pshifty_ex[p] * nyp;
            }
            
            // number of finite-difference nodes in each direction of extended rb (+ order) region
            //nx2p = nxp + pSPARC->order;
            //ny2p = nyp + pSPARC->order;
            //nz2p = nzp + pSPARC->order;
            //nd_2ex = nx2p * ny2p * nz2p; // total number of nodes
            
            // radii^2 of the finite difference grids of the FDn-extended-rb-region
            R  = (double *)malloc(sizeof(double) * nd_ex);
            if (R == NULL) {
                printf("\nMemory allocation failed!\n");
                exit(EXIT_FAILURE);
            }

            // left corner of the 2FDn-extended-rb-region
            icor = pSPARC->Atom_Influence_local[ityp].xs[iat] - FDn;
            jcor = pSPARC->Atom_Influence_local[ityp].ys[iat] - FDn;
            kcor = pSPARC->Atom_Influence_local[ityp].zs[iat] - FDn;
            
            // relative coordinate of image atoms
            x0_i_shift =  x0_i - pSPARC->delta_x * icor; 
            y0_i_shift =  y0_i - pSPARC->delta_y * jcor;
            z0_i_shift =  z0_i - pSPARC->delta_z * kcor;
            
            // find distance between atom and finite-difference grids
            count = 0; count_interp = 0;
            if(pSPARC->cell_typ == 0) {    
                for (k = 0; k < nzp; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < nyp; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nxp; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            } else {
                for (k = 0; k < nzp; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < nyp; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nxp; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt(pSPARC->metricT[0] * (x*x) + pSPARC->metricT[1] * (x*y) + pSPARC->metricT[2] * (x*z) 
                                          + pSPARC->metricT[4] * (y*y) + pSPARC->metricT[5] * (y*z) + pSPARC->metricT[8] * (z*z) );
                            //R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            } 

            VJ_ref = (double *)malloc( nd_ex * sizeof(double) );
            if (VJ_ref == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            
            // Calculate pseudopotential reference
            Calculate_Pseudopot_Ref(R, nd_ex, pSPARC->REFERENCE_CUTOFF, -pSPARC->Znucl[ityp], VJ_ref);
            
            VJ = (double *)malloc( nd_ex * sizeof(double) );
            if (VJ == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            
            // avoid sorting positions larger than rchrg
            VJ_interp = (double *)malloc( count_interp * sizeof(double) );
            R_interp = (double *)malloc( count_interp * sizeof(double) );
            ind_interp = (int *)malloc( count_interp * sizeof(int) );
            count = 0;
            for (i = 0; i < nd_ex; i++) {
                if (R[i] <= rchrg) {
                    ind_interp[count] = i; // store index
                    R_interp[count] = R[i]; // store radius value
                    count++;
                } else {
                    VJ[i] = -pSPARC->Znucl[ityp] / R[i];
                }
            }

            t1 = MPI_Wtime();
            
            // sort R_interp and then apply cubic spline interpolation to find VJ
            // notice here we extract out positions within radius rchrg
            if (pSPARC->psd[ityp].is_r_uniform == 1) {
            	SplineInterpUniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
				                    R_interp, VJ_interp, count_interp, pSPARC->psd[ityp].SplinerVlocD); 
            } else {
				// SortSplineInterp(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
				//                  R_interp, VJ_interp, count_interp, pSPARC->psd[ityp].SplinerVlocD); 
				SplineInterpNonuniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
				                    R_interp, VJ_interp, count_interp, pSPARC->psd[ityp].SplinerVlocD); 
			}

            t2 = MPI_Wtime();
            t_sort += t2 - t1;
            
            for (i = 0; i < count_interp; i++) {
                if (R_interp[i] < TEMP_TOL) {
                    VJ[ind_interp[i]] = pSPARC->psd[ityp].Vloc_0;
                } else {
                    VJ[ind_interp[i]] =  VJ_interp[i]/R_interp[i];
                }
            }
            free(VJ_interp); VJ_interp = NULL;
            free(R_interp); R_interp = NULL;
            free(ind_interp); ind_interp = NULL;
            free(R); R = NULL;
            
            // calculate VcJ in the extended-rb-domain
            VcJ = (double *)malloc( nd_ex * sizeof(double) );
            if (VcJ == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            for (i = 0; i < nd_ex; i++) {
                VcJ[i] = VJ_ref[i] - VJ[i];
            }
            
            // calculate bJ, bJ_ref and gradient of VcJ in the rb-domain
            dK = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dJ = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            dI = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];

            double *bJ = (double*)malloc(nd * sizeof(double));
            double *bJ_ref = (double*)malloc(nd * sizeof(double));
                        
            Calc_lapV(pSPARC, VJ, FDn, nxp, nyp, nzp, nx, ny, nz, Lap_wt, w2_diag, 0.0, -inv_4PI, bJ);
            Calc_lapV(pSPARC, VJ_ref, FDn, nxp, nyp, nzp, nx, ny, nz, Lap_wt, w2_diag, 0.0, -inv_4PI, bJ_ref);

            force_x = force_y = force_z = 0.0;
            force_corr_x = force_corr_y = force_corr_z = 0.0;
            for (k = 0; k < nz; k++) {
                kp = k + FDn;
                k_DM = k + dK;
                int kshift_DM = k_DM * DMnx * DMny;
                int kshift_p = kp * nxp * nyp;
                int kshift = k * nx * ny; 
                for (j = 0; j < ny; j++) {
                    jp = j + FDn;
                    j_DM = j + dJ;
                    int jshift_DM = kshift_DM + j_DM * DMnx;
                    int jshift_p = kshift_p + jp * nxp;
                    int jshift = kshift + j * nx;
                    //#pragma simd
                    for (i = 0; i < nx; i++) {
                        ip = i + FDn;
                        i_DM = i + dI;
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;
                        int ishift = jshift + i;
                        DVcJ_x_val = DVcJ_y_val = DVcJ_z_val = 0.0;
                        //bJ_val = VJ[ishift_p] * w2_diag;
                        //bJ_ref_val = VJ_ref[ishift_p] * w2_diag;
                        for (p = 1; p <= FDn; p++) {
                            DVcJ_x_val += (VcJ[ishift_p+p] - VcJ[ishift_p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                            DVcJ_y_val += (VcJ[ishift_p+pshifty_ex[p]] - VcJ[ishift_p-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                            DVcJ_z_val += (VcJ[ishift_p+pshiftz_ex[p]] - VcJ[ishift_p-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];
                        }
                        // find integrals in the force expression
                        double b_plus_b_ref = pSPARC->psdChrgDens[ishift_DM] + pSPARC->psdChrgDens_ref[ishift_DM];
                        double bJ_plus_bJ_ref = bJ[ishift] + bJ_ref[ishift];
                        
                        force_x -= bJ[ishift] * Dphi_x[ishift_DM];
                        force_corr_x += DVcJ_x_val * b_plus_b_ref - DVc_x[ishift_DM] * bJ_plus_bJ_ref;
                        force_y -= bJ[ishift] * Dphi_y[ishift_DM];
                        force_corr_y += DVcJ_y_val * b_plus_b_ref - DVc_y[ishift_DM] * bJ_plus_bJ_ref;
                        force_z -= bJ[ishift] * Dphi_z[ishift_DM];
                        force_corr_z += DVcJ_z_val * b_plus_b_ref - DVc_z[ishift_DM] * bJ_plus_bJ_ref;
                    }
                }
            }
                       
            pSPARC->forces[atom_index*3  ] += (force_x + 0.5 * force_corr_x) * pSPARC->dV;
            pSPARC->forces[atom_index*3+1] += (force_y + 0.5 * force_corr_y) * pSPARC->dV;
            pSPARC->forces[atom_index*3+2] += (force_z + 0.5 * force_corr_z) * pSPARC->dV;
            
            free(VJ); VJ = NULL;
            free(VJ_ref); VJ_ref = NULL;
            free(bJ); bJ = NULL;
            free(bJ_ref); bJ_ref = NULL;
            free(VcJ); VcJ = NULL;
        }   
    }

    free(Lap_wt);            
    free(Dphi_x);
    free(Dphi_y);
    free(Dphi_z);
    free(DVc_x);
    free(DVc_y);
    free(DVc_z);
    free(pshifty_ex);
    free(pshiftz_ex);
    
    t1 = MPI_Wtime();
    // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
    MPI_Allreduce(MPI_IN_PLACE, pSPARC->forces, 3*pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("time for sorting and interpolate pseudopotential: %.3f ms, time for Allreduce/Reduce: %.3f ms \n", t_sort*1e3, (t2-t1)*1e3);
#endif
}


/**
 * @brief    Calculate xc force components for nonlinear core 
 *           correction (NLCC).
 */ 
void Calculate_forces_xc(SPARC_OBJ *pSPARC, double *forces_xc) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);

    int ityp, iat, i, j, k, p, ip, jp, kp, di, dj, dk, i_DM, j_DM, k_DM, FDn, count, count_interp,
        DMnx, DMny, nx, ny, nz, nxp, nyp, nzp, nd_ex, nx2p, ny2p, nz2p, nd_2ex, 
        icor, jcor, kcor, *pshifty, *pshiftz, *pshifty_ex, *pshiftz_ex, *ind_interp;
    double x0_i, y0_i, z0_i, x0_i_shift, y0_i_shift, z0_i_shift, x, y, z, *R,
           *R_interp, *VJ_interp;
    double rchrg;
    double t1, t2, t_sort = 0.0;

    FDn = pSPARC->order / 2;
    
    DMnx = pSPARC->Nx_d;
    DMny = pSPARC->Ny_d;

    // Create indices for laplacian
    pshifty = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz = (int *)malloc( (FDn+1) * sizeof(int));
    pshifty_ex = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz_ex = (int *)malloc( (FDn+1) * sizeof(int));
    
    if (pshifty == NULL || pshiftz == NULL || 
        pshifty_ex == NULL || pshiftz_ex == NULL) {
        printf("\nMemory allocation failed in local forces!\n");
        exit(EXIT_FAILURE);
    }

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        for (iat = 0; iat < pSPARC->Atom_Influence_local[ityp].n_atom; iat++) {
            // coordinates of the image atom
            x0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3];
            y0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 1];
            z0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 2];

            // original atom index this image atom corresponds to
            int atom_index = pSPARC->Atom_Influence_local[ityp].atom_index[iat];
            
            // number of finite-difference nodes in each direction of overlap rb region
            nx = pSPARC->Atom_Influence_local[ityp].xe[iat] - pSPARC->Atom_Influence_local[ityp].xs[iat] + 1;
            ny = pSPARC->Atom_Influence_local[ityp].ye[iat] - pSPARC->Atom_Influence_local[ityp].ys[iat] + 1;
            nz = pSPARC->Atom_Influence_local[ityp].ze[iat] - pSPARC->Atom_Influence_local[ityp].zs[iat] + 1;

            // number of finite-difference nodes in each direction of extended_rb (rb + order/2) region
            nxp = nx + pSPARC->order;
            nyp = ny + pSPARC->order;
            nzp = nz + pSPARC->order;
            nd_ex = nxp * nyp * nzp; // total number of nodes
            // number of finite-difference nodes in each direction of extended_extended_rb (rb + order) region
            nx2p = nxp + pSPARC->order;
            ny2p = nyp + pSPARC->order;
            nz2p = nzp + pSPARC->order;
            nd_2ex = nx2p * ny2p * nz2p; // total number of nodes
            
            // radii^2 of the finite difference grids of the extended_extended_rb region
            R  = (double *)malloc(sizeof(double) * nd_2ex);
            assert(R != NULL);
            
            // left corner of the 2FDn-extended-rb-region
            icor = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->order;
            jcor = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->order;
            kcor = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->order;
                        
            // relative coordinate of image atoms
            x0_i_shift =  x0_i - pSPARC->delta_x * icor; 
            y0_i_shift =  y0_i - pSPARC->delta_y * jcor;
            z0_i_shift =  z0_i - pSPARC->delta_z * kcor;
            
            // find distance between atom and finite-difference grids
            count = 0; count_interp = 0;
            if(pSPARC->cell_typ == 0) {    
                for (k = 0; k < nz2p; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < ny2p; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nx2p; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            } else {
                for (k = 0; k < nz2p; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < ny2p; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nx2p; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt(pSPARC->metricT[0] * (x*x) + pSPARC->metricT[1] * (x*y) + pSPARC->metricT[2] * (x*z) 
                                          + pSPARC->metricT[4] * (y*y) + pSPARC->metricT[5] * (y*z) + pSPARC->metricT[8] * (z*z) );
                            //R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            } 
            
            
            // VJ = (double *)malloc( nd_2ex * sizeof(double) );
            double *rhocJ = (double *)calloc( nd_2ex,sizeof(double) );
            assert(rhocJ != NULL);
            
            // avoid interpolating positions larger than rchrg
            VJ_interp = (double *)malloc( count_interp * sizeof(double) );
            R_interp = (double *)malloc( count_interp * sizeof(double) );
            ind_interp = (int *)malloc( count_interp * sizeof(int) );
            double *rhocJ_interp = (double *)calloc(count_interp, sizeof(double));
            count = 0;
            for (i = 0; i < nd_2ex; i++) {
                if (R[i] <= rchrg) {
                    ind_interp[count] = i; // store index
                    R_interp[count] = R[i]; // store radius value
                    count++;
                } else {
                    // VJ[i] = -pSPARC->Znucl[ityp] / R[i];
                }
            }
            
            t1 = MPI_Wtime();
            // sort R_interp and then apply cubic spline interpolation to find rhocJ
            SplineInterpMain(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rho_c_table, pSPARC->psd[ityp].size, 
                         R_interp, rhocJ_interp, count_interp, pSPARC->psd[ityp].SplineRhocD,pSPARC->psd[ityp].is_r_uniform);
            t2 = MPI_Wtime();
            t_sort += t2 - t1;

            for (i = 0; i < count_interp; i++) {
                rhocJ[ind_interp[i]] = rhocJ_interp[i];
            }

            free(rhocJ_interp); rhocJ_interp = NULL;
            free(VJ_interp); VJ_interp = NULL;
            free(R_interp); R_interp = NULL;
            free(ind_interp); ind_interp = NULL;
            free(R); R = NULL;
            
            // shift vectors initialized
            pshifty[0] = pshiftz[0] = pshifty_ex[0] = pshiftz_ex[0] = 0;
            for (p = 1; p <= FDn; p++) {
                pshifty[p] = p * nxp;
                pshiftz[p] = pshifty[p] * nyp;
                pshifty_ex[p] = p * nx2p;
                pshiftz_ex[p] = pshifty_ex[p] *ny2p;
            }

            // calculate gradient of bJ, bJ_ref, VJ, VJ_ref in the rb-domain
            dk = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dj = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            di = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];

            // calculate drhocJ, 3 components
            double *drhocJ_x = malloc(nd_ex * sizeof(double));
            double *drhocJ_y = malloc(nd_ex * sizeof(double));
            double *drhocJ_z = malloc(nd_ex * sizeof(double));
            assert(drhocJ_x != NULL && drhocJ_y != NULL && drhocJ_z != NULL);
            
            for (int k2p = FDn, kp = 0; k2p < nz2p-FDn; k2p++,kp++) {
                int kshift_2p = k2p * nx2p * ny2p;
                int kshift_p = kp * nxp * nyp;
                for (int j2p = FDn, jp = 0; j2p < ny2p-FDn; j2p++,jp++) {
                    int jshift_2p = kshift_2p + j2p * nx2p;
                    int jshift_p = kshift_p + jp * nxp;
                    for (int i2p = FDn, ip = 0; i2p < nx2p-FDn; i2p++,ip++) {
                        int ishift_2p = jshift_2p + i2p;
                        int ishift_p = jshift_p + ip;
                        double drhocJ_x_val, drhocJ_y_val, drhocJ_z_val;
                        drhocJ_x_val = drhocJ_y_val = drhocJ_z_val = 0.0;
                        for (p = 1; p <= FDn; p++) {
                            drhocJ_x_val += (rhocJ[ishift_2p+p] - rhocJ[ishift_2p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                            drhocJ_y_val += (rhocJ[ishift_2p+pshifty_ex[p]] - rhocJ[ishift_2p-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                            drhocJ_z_val += (rhocJ[ishift_2p+pshiftz_ex[p]] - rhocJ[ishift_2p-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];
                        }
                        drhocJ_x[ishift_p] = drhocJ_x_val;
                        drhocJ_y[ishift_p] = drhocJ_y_val;
                        drhocJ_z[ishift_p] = drhocJ_z_val;
                    }
                }
            }

            // find int Vxc(x) * drhocJ(x) dx
            double *Vxc = pSPARC->XCPotential;
            double force_xc_x, force_xc_y, force_xc_z;
            force_xc_x = force_xc_y = force_xc_z = 0.0;
            for (k = 0, kp = FDn, k_DM = dk; k < nz; k++, kp++, k_DM++) {
                int kshift_DM = k_DM * DMnx * DMny;
                int kshift_p = kp * nxp * nyp; 
                for (j = 0, jp = FDn, j_DM = dj; j < ny; j++, jp++, j_DM++) {
                    int jshift_DM = kshift_DM + j_DM * DMnx;
                    int jshift_p = kshift_p + jp * nxp;
                    for (i = 0, ip = FDn, i_DM = di; i < nx; i++, ip++, i_DM++) {
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;
                        double drhocJ_x_val = drhocJ_x[ishift_p];
                        double drhocJ_y_val = drhocJ_y[ishift_p];
                        double drhocJ_z_val = drhocJ_z[ishift_p];
                        if (pSPARC->cell_typ != 0)
                            nonCart2Cart_grad(pSPARC, &drhocJ_x_val, &drhocJ_y_val, &drhocJ_z_val);
                        double Vxc_val; 
                        if (pSPARC->Nspin == 1)
                            Vxc_val = Vxc[ishift_DM];
                        else
                            Vxc_val = 0.5 * (Vxc[ishift_DM] + Vxc[pSPARC->Nd_d+ishift_DM]);
                        force_xc_x += Vxc_val * drhocJ_x_val;
                        force_xc_y += Vxc_val * drhocJ_y_val;
                        force_xc_z += Vxc_val * drhocJ_z_val;
                    }
                }
            }
            forces_xc[atom_index*3  ] += force_xc_x  * pSPARC->dV;
            forces_xc[atom_index*3+1] += force_xc_y  * pSPARC->dV;
            forces_xc[atom_index*3+2] += force_xc_z  * pSPARC->dV;
            free(rhocJ);
            free(drhocJ_x);
            free(drhocJ_y);
            free(drhocJ_z);
        }
    }

    // sum over all domains
    MPI_Allreduce(MPI_IN_PLACE, forces_xc, 3*pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);

    free(pshifty);
    free(pshiftz);
    free(pshifty_ex);
    free(pshiftz_ex);
}
