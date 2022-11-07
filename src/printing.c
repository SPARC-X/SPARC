/**
 * @file    printing.c
 * @brief   This file contains the functions for printing properties.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <complex.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

#include "printing.h"
#include "tools.h"
#include "parallelization.h"
#include "exchangeCorrelation.h"
#include "exactExchangeEnergyDensity.h"
#include "MGGAexchangeCorrelation.h"


/**
 * @brief   Print initial electron density guess and converged density.
 */
void printElecDens(SPARC_OBJ *pSPARC) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int nproc_dmcomm_phi, rank_dmcomm_phi, DMnd;
    MPI_Comm_size(pSPARC->dmcomm_phi, &nproc_dmcomm_phi);
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank_dmcomm_phi);
    
    int Nd = pSPARC->Nd;
    DMnd = pSPARC->Nd_d;
    
    double *rho_at, *rho, *b_ref, *b;
    rho_at = NULL;
    rho = NULL;
    b_ref = NULL;
    b = NULL;
    if (nproc_dmcomm_phi > 1) { // if there's more than one process, need to collect rho first
        // use DD2DD to collect distributed data
        int gridsizes[3], sdims[3], rdims[3], rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_dmcomm_phi) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int dims[3] = {1,1,1}, periods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &recv_comm);
        }
        D2D_OBJ d2d_sender, d2d_recvr;
        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;
        sdims[0] = pSPARC->npNdx_phi;
        sdims[1] = pSPARC->npNdy_phi;
        sdims[2] = pSPARC->npNdz_phi;
        rdims[0] = rdims[1] = rdims[2] = 1;
        rDMVert[0] = 0; rDMVert[1] = pSPARC->Nx-1;
        rDMVert[2] = 0; rDMVert[3] = pSPARC->Ny-1;
        rDMVert[4] = 0; rDMVert[5] = pSPARC->Nz-1;
        
        // set up D2D targets
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, rDMVert, pSPARC->dmcomm_phi, 
                       sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        if (rank_dmcomm_phi == 0) {
            int n_rho = 1;
            if(pSPARC->Nspin > 1) { // for spin polarized systems
                n_rho = 3; // rho, rho_up, rho_down
            }
            rho_at = (double*)malloc(pSPARC->Nd * n_rho * sizeof(double));
            rho    = (double*)malloc(pSPARC->Nd * n_rho * sizeof(double));
            b_ref  = (double*)malloc(pSPARC->Nd * sizeof(double));
            b      = (double*)malloc(pSPARC->Nd * sizeof(double));
        }
        // send rho_at, rho and b_ref
        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->electronDens_at, rDMVert, 
            rho_at, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        
        if (pSPARC->Nspin > 1) { // send rho_at_up, rho_at_down
            D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->electronDens_at+DMnd, rDMVert, 
                rho_at+Nd, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
            D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->electronDens_at+2*DMnd, rDMVert, 
                rho_at+2*Nd, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        }

        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->electronDens, rDMVert, 
            rho, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        
        if (pSPARC->Nspin > 1) { // send rho_up, rho_down
            D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->electronDens+DMnd, rDMVert, 
                rho+Nd, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
            D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->electronDens+2*DMnd, rDMVert, 
                rho+2*Nd, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        }

        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->psdChrgDens_ref, rDMVert, 
            b_ref, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        
        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->psdChrgDens, rDMVert, 
            b, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        
        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, pSPARC->dmcomm_phi, recv_comm);
        if (rank_dmcomm_phi == 0) 
            MPI_Comm_free(&recv_comm);
    } else {
        rho_at = pSPARC->electronDens_at;
        rho    = pSPARC->electronDens;
        b_ref  = pSPARC->psdChrgDens_ref;
        b      = pSPARC->psdChrgDens;
    }
    
    if (rank_dmcomm_phi == 0) {
        if (pSPARC->Nspin == 1) {
            // printing total electron density in cube format
            printDens_cube(pSPARC, rho, pSPARC->DensTCubFilename, "Electron density");
        } else {
            // printing total, spin-up and spin-down electron density in cube format
            printDens_cube(pSPARC, rho, pSPARC->DensTCubFilename, "Total electron density");
            printDens_cube(pSPARC, rho+Nd, pSPARC->DensUCubFilename, "Spin-up electron density");
            printDens_cube(pSPARC, rho+2*Nd, pSPARC->DensDCubFilename, "Spin-down electron density");
        }
    }

    // free the collected data after printing to file
    if (nproc_dmcomm_phi > 1) {
        if (rank_dmcomm_phi == 0) {
            free(rho_at);
            free(rho);
            free(b_ref);
            free(b);
        }
    }
}

/**
 * @brief   Print converged density in cube format.
 */
void printDens_cube(SPARC_OBJ *pSPARC, double *rho, char *fname, char *rhoname) {
#define rho(i,j,k) rho[(i)+(j)*Nx+(k)*Nx*Ny]
    
    FILE *output_fp = fopen(fname,"w");
    if (output_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",fname);
        exit(EXIT_FAILURE);
    }    
    int Nx = pSPARC->Nx;
    int Ny = pSPARC->Ny;
    int Nz = pSPARC->Nz;
    double dx = pSPARC->delta_x;
    double dy = pSPARC->delta_y;
    double dz = pSPARC->delta_z;

    // printing headers
    time_t current_time;
    time(&current_time);
    char *c_time_str = ctime(&current_time);
    // ctime includes a newline char '\n', remove manually
    if (c_time_str[strlen(c_time_str)-1] == '\n') 
        c_time_str[strlen(c_time_str)-1] = '\0'; 
    fprintf(output_fp, "%s in Cube format printed by SPARC-X (Print time: %s)\n", rhoname, c_time_str);
    fprintf(output_fp, "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z.\n");
    fprintf(output_fp, "%5d %11.6f  %11.6f  %11.6f\n", pSPARC->n_atom, 0.0, 0.0, 0.0);
    fprintf(output_fp, "%5d %11.6f  %11.6f  %11.6f\n", Nx, pSPARC->LatUVec[0]*dx, pSPARC->LatUVec[1]*dx, pSPARC->LatUVec[2]*dx);
    fprintf(output_fp, "%5d %11.6f  %11.6f  %11.6f\n", Ny, pSPARC->LatUVec[3]*dy, pSPARC->LatUVec[4]*dy, pSPARC->LatUVec[5]*dy);
    fprintf(output_fp, "%5d %11.6f  %11.6f  %11.6f\n", Nz, pSPARC->LatUVec[6]*dz, pSPARC->LatUVec[7]*dz, pSPARC->LatUVec[8]*dz);
    
    int atmcount = 0, i, j, k, ityp;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int zatom = pSPARC->Zatom[ityp];
        int zion = pSPARC->Znucl[ityp];
        for (i = 0; i < pSPARC->nAtomv[ityp]; i++) {
            // get atom positions
            double x0 = pSPARC->atom_pos[3*atmcount];
            double y0 = pSPARC->atom_pos[3*atmcount+1];
            double z0 = pSPARC->atom_pos[3*atmcount+2];
            atmcount++;
            fprintf(output_fp, "%5d %11.6f %11.6f  %11.6f  %11.6f\n", zatom, (double)zion, x0, y0, z0);
        }
    }

    // printing rho
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < Ny; j++) {
            for (k = 0; k < Nz; k++) {
                fprintf(output_fp, "  %.6E", rho(i,j,k));
                if (k % 6 == 5)
                    fprintf(output_fp, "\n");
            }
            fprintf(output_fp, "\n");
        }
    }

    fclose(output_fp);
#undef rho
}

/**
 * @brief   Print final eigenvalues and occupation numbers.
 */
void printEigen(SPARC_OBJ *pSPARC) {
    int rank, rank_spincomm, rank_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

    // only root processes of kptcomms will enter
    if (pSPARC->kptcomm_index < 0 || rank_kptcomm != 0) return; 

    int Nk = pSPARC->Nkpts_kptcomm;
    int Ns = pSPARC->Nstates;
    double occfac = 2.0/pSPARC->Nspin/pSPARC->Nspinor;
    // number of kpoints assigned to each kptcomm
    int    *Nk_i   = (int    *)malloc(pSPARC->npkpt * sizeof(int)); 
    double *kred_i = (double *)malloc(pSPARC->Nkpts_sym * 3 * sizeof(double));
    int *kpt_displs= (int    *)malloc((pSPARC->npkpt+1) * sizeof(int));

    char EigenFilename[L_STRING];
    snprintf(EigenFilename, L_STRING, "%s", pSPARC->EigenFilename);
    
    FILE *output_fp;
    // first create an empty file
    if (rank == 0) {
        output_fp = fopen(EigenFilename,"w");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",EigenFilename);
            exit(EXIT_FAILURE);
        } 
        fprintf(output_fp, "Final eigenvalues (Ha) and occupation numbers\n");
        fclose(output_fp);   
    }

    // gather all eigenvalues and occupation number to root process in spin 
    int sendcount, *recvcounts, *displs;
    double *recvbuf_eig, *recvbuf_occ;
    sendcount = 0;
    recvcounts = NULL;
    displs = NULL;
    recvbuf_eig = NULL;
    recvbuf_occ = NULL;

    // first collect eigval/occ over spin
    if (pSPARC->npspin > 1) {
        // set up receive buffer and receive counts in kptcomm roots with spin up
        if (pSPARC->spincomm_index == 0) { 
            recvbuf_eig = (double *)malloc(pSPARC->Nspin * Nk * Ns * sizeof(double));
            recvbuf_occ = (double *)malloc(pSPARC->Nspin * Nk * Ns * sizeof(double));
            recvcounts  = (int *)   malloc(pSPARC->npspin * sizeof(int)); // npspin is 2
            displs      = (int *)   malloc((pSPARC->npspin+1) * sizeof(int)); 
            int i;
            displs[0] = 0;
            for (i = 0; i < pSPARC->npspin; i++) {
                recvcounts[i] = pSPARC->Nspin_spincomm * Nk * Ns;
                displs[i+1] = displs[i] + recvcounts[i];
            }
        } 
        // set up send info
        sendcount = pSPARC->Nspin_spincomm * Nk * Ns;
        MPI_Gatherv(pSPARC->lambda_sorted, sendcount, MPI_DOUBLE,
                    recvbuf_eig, recvcounts, displs,
                    MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
        MPI_Gatherv(pSPARC->occ_sorted, sendcount, MPI_DOUBLE,
                    recvbuf_occ, recvcounts, displs,
                    MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
        if (pSPARC->spincomm_index == 0) { 
            free(recvcounts);
            free(displs);
        }
    } else {
        recvbuf_eig = pSPARC->lambda_sorted;
        recvbuf_occ = pSPARC->occ_sorted;
    }
 
    double *eig_all = NULL, *occ_all = NULL;
    int *displs_all;
    displs_all = (int *)malloc((pSPARC->npkpt+1) * sizeof(int));  

    // next collect eigval/occ over all kpoints
    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
        // set up receive buffer and receive counts in kptcomm roots with spin up
        if (pSPARC->kptcomm_index == 0) {
            int i;
            eig_all = (double *)malloc(pSPARC->Nspin * pSPARC->Nkpts_sym * Ns * sizeof(double));
            occ_all = (double *)malloc(pSPARC->Nspin * pSPARC->Nkpts_sym * Ns * sizeof(double));
            recvcounts = (int *)malloc(pSPARC->npkpt * sizeof(int));
            // collect all the number of kpoints assigned to each kptcomm
            MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT,
               0, pSPARC->kpt_bridge_comm);
            displs_all[0] = 0;
            for (i = 0; i < pSPARC->npkpt; i++) {
                recvcounts[i] = Nk_i[i] * pSPARC->Nspin * Ns;
                displs_all[i+1] = displs_all[i] + recvcounts[i];
            }
            // collect all the kpoints assigend to each kptcomm
            // first set up sendbuf and recvcounts
            double *kpt_sendbuf = (double *)malloc(Nk * 3 * sizeof(double));
            int *kpt_recvcounts = (int *)malloc(pSPARC->npkpt * sizeof(int));
            // int *kpt_displs     = (int *)malloc((pSPARC->npkpt+1) * sizeof(int));
            for (i = 0; i < Nk; i++) {
                kpt_sendbuf[3*i  ] = pSPARC->k1_loc[i]*pSPARC->range_x/(2.0*M_PI);
                kpt_sendbuf[3*i+1] = pSPARC->k2_loc[i]*pSPARC->range_y/(2.0*M_PI);
                kpt_sendbuf[3*i+2] = pSPARC->k3_loc[i]*pSPARC->range_z/(2.0*M_PI);
            } 
            kpt_displs[0] = 0; 
            for (i = 0; i < pSPARC->npkpt; i++) {
                kpt_recvcounts[i]  = Nk_i[i] * 3;
                kpt_displs[i+1] = kpt_displs[i] + kpt_recvcounts[i];
            }
            // collect reduced kpoints from all kptcomms
            MPI_Gatherv(kpt_sendbuf, Nk*3, MPI_DOUBLE, 
                kred_i, kpt_recvcounts, kpt_displs, 
                MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
            free(kpt_sendbuf);
            free(kpt_recvcounts);
        } else {
            // collect all the number of kpoints assigned to each kptcomm
            MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT,
               0, pSPARC->kpt_bridge_comm);
            // collect all the kpoints assigend to each kptcomm
            double *kpt_sendbuf = (double *)malloc(Nk * 3 * sizeof(double));
            int kpt_recvcounts[1]={0}, i;
            for (i = 0; i < Nk; i++) {
                kpt_sendbuf[3*i  ] = pSPARC->k1_loc[i]*pSPARC->range_x/(2.0*M_PI);
                kpt_sendbuf[3*i+1] = pSPARC->k2_loc[i]*pSPARC->range_y/(2.0*M_PI);
                kpt_sendbuf[3*i+2] = pSPARC->k3_loc[i]*pSPARC->range_z/(2.0*M_PI);
            }
            // collect reduced kpoints from all kptcomms
            MPI_Gatherv(kpt_sendbuf, Nk*3, MPI_DOUBLE, 
                kred_i, kpt_recvcounts, kpt_displs, 
                MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
            free(kpt_sendbuf);
        }   
        // set up send info
        sendcount = pSPARC->Nspin * Nk * Ns;
        MPI_Gatherv(recvbuf_eig, sendcount, MPI_DOUBLE,
                    eig_all, recvcounts, displs_all,
                    MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
        MPI_Gatherv(recvbuf_occ, sendcount, MPI_DOUBLE,
                    occ_all, recvcounts, displs_all,
                    MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
        if (pSPARC->kptcomm_index == 0) {
            free(recvcounts);
            //free(displs_all);
        }
    } else {
        int i;
        Nk_i[0] = Nk; // only one kptcomm
        kpt_displs[0] = 0;
        displs_all[0] = 0;
        if (pSPARC->BC != 1) {
            for (i = 0; i < Nk; i++) {
                kred_i[3*i  ] = pSPARC->k1_loc[i]*pSPARC->range_x/(2.0*M_PI);
                kred_i[3*i+1] = pSPARC->k2_loc[i]*pSPARC->range_y/(2.0*M_PI);
                kred_i[3*i+2] = pSPARC->k3_loc[i]*pSPARC->range_z/(2.0*M_PI);
            }
        } else {
            kred_i[0] = kred_i[1] = kred_i[2] = 0.0;
        }
        eig_all = recvbuf_eig;
        occ_all = recvbuf_occ;
    }

    // let root process print eigvals and occs to .eigen file
    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            // write to .eig file
            output_fp = fopen(EigenFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",EigenFilename);
                exit(EXIT_FAILURE);
            }
            int k, Kcomm_indx, i;
            if (pSPARC->Nspin == 1) {
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        int kred_index = kpt_displs[Kcomm_indx]/3+k+1;
                        fprintf(output_fp,
                                "\n"
                                "kred #%d = (%f,%f,%f)\n"
                                "n        eigval                 occ\n",
                                kred_index,
                                kred_i[kpt_displs[Kcomm_indx]+3*k], 
                                kred_i[kpt_displs[Kcomm_indx]+3*k+1], 
                                kred_i[kpt_displs[Kcomm_indx]+3*k+2]);
                        for (i = 0; i < pSPARC->Nstates; i++) {
                            fprintf(output_fp, "%-7d%20.12E %18.12f\n", 
                                i+1,
                                eig_all[displs_all[Kcomm_indx] + k*Ns + i],
                                occfac * occ_all[displs_all[Kcomm_indx] + k*Ns + i]);
                        }
                    }
                }
            } else if (pSPARC->Nspin == 2) {
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        int kred_index = kpt_displs[Kcomm_indx]/3+k+1;
                        fprintf(output_fp,
                                "\n"
                                "kred #%d = (%f,%f,%f)\n"
                                "                       Spin-up                                    Spin-down\n"
                                "n        eigval                 occ                 eigval                 occ\n",
                                kred_index,
                                kred_i[kpt_displs[Kcomm_indx]+3*k], 
                                kred_i[kpt_displs[Kcomm_indx]+3*k+1], 
                                kred_i[kpt_displs[Kcomm_indx]+3*k+2]);
                        for (i = 0; i < pSPARC->Nstates; i++) {
                            fprintf(output_fp, "%-7d%20.12E %18.12f    %20.12E %18.12f\n", 
                                i+1,
                                eig_all[displs_all[Kcomm_indx] + k*Ns + i],
                                occfac * occ_all[displs_all[Kcomm_indx] + k*Ns + i],
                                eig_all[displs_all[Kcomm_indx] + (Nk_Kcomm_indx + k)*Ns + i],
                                occfac * occ_all[displs_all[Kcomm_indx] + (Nk_Kcomm_indx + k)*Ns + i]);
                        }
                    }
                }
            }
            fclose(output_fp);
        }
    }

    free(Nk_i);
    free(kred_i);
    free(kpt_displs);
    free(displs_all);

    if (pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0) { 
            free(recvbuf_eig);
            free(recvbuf_occ);
        }
    }

    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            free(eig_all);
            free(occ_all);
        }
    }
}


/**
 * @brief   Print Kohn-Sham orbitals
 */
void print_orbitals(SPARC_OBJ *pSPARC) {
    int gridsizes[3], rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int flag = pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL;
    MPI_Comm alldmcomm;
    int color = (flag == 1) ? MPI_UNDEFINED: 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &alldmcomm);
    if (flag) return;

    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int Nd = pSPARC->Nd;

    double dx = pSPARC->delta_x;
    double dy = pSPARC->delta_y;
    double dz = pSPARC->delta_z;
    double dV = pSPARC->dV;

    int DMnd = pSPARC->Nd_d_dmcomm;
    int size_k = DMnd * pSPARC->Nband_bandcomm;
    int size_s = size_k * pSPARC->Nkpts_kptcomm;

    int spin_start = max(pSPARC->PrintPsiFlag[1],0);
    int spin_end = min(pSPARC->PrintPsiFlag[2], pSPARC->Nspin-1);
    int kpt_start = max(pSPARC->PrintPsiFlag[3],0);
    int kpt_end = min(pSPARC->PrintPsiFlag[4], pSPARC->Nkpts-1);
    int band_start = max(pSPARC->PrintPsiFlag[5],0);
    int band_end = min(pSPARC->PrintPsiFlag[6], pSPARC->Nstates-1);

    char fname[L_STRING];
    snprintf(fname,  L_STRING, "%s", pSPARC->OrbitalsFilename);
    if (rank == 0) {
        FILE *output_fp = fopen(fname,"wb");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",fname);
            exit(EXIT_FAILURE);
        }
        // system info
        fwrite(gridsizes, sizeof(int), 1, output_fp);       // Nx
        fwrite(gridsizes+1, sizeof(int), 1, output_fp);     // Ny
        fwrite(gridsizes+2, sizeof(int), 1, output_fp);     // Nz
        fwrite(&Nd, sizeof(int), 1, output_fp);             // Nd
        fwrite(&dx, sizeof(double), 1, output_fp);          // dx
        fwrite(&dy, sizeof(double), 1, output_fp);          // dy
        fwrite(&dz, sizeof(double), 1, output_fp);          // dz
        fwrite(&dV, sizeof(double), 1, output_fp);          // dV
        fwrite(&pSPARC->isGammaPoint, sizeof(int), 1, output_fp);     // isGamma
        // number of spin, kpt and band
        int nspin = spin_end - spin_start + 1;
        fwrite(&nspin, sizeof(int), 1, output_fp);
        int nkpt = kpt_end - kpt_start + 1;
        fwrite(&nkpt, sizeof(int), 1, output_fp);
        int nband = band_end - band_start + 1;
        fwrite(&nband, sizeof(int), 1, output_fp);
        fclose(output_fp);
    }

    for (int spin = spin_start; spin <= spin_end; spin++) {
        int spin_flag = (pSPARC->spin_start_indx <= spin && spin <= pSPARC->spin_end_indx);
        int spin_shift = spin - pSPARC->spin_start_indx;

        for (int kpt = kpt_start; kpt <= kpt_end; kpt++) {
            int kpt_flag = (pSPARC->kpt_start_indx <= kpt && kpt <= pSPARC->kpt_end_indx);
            int kpt_shift = kpt - pSPARC->kpt_start_indx;

            for (int band = band_start; band <= band_end; band++) {
                int band_flag = (pSPARC->band_start_indx <= band && band <= pSPARC->band_end_indx);
                int band_shift = band - pSPARC->band_start_indx;
                
                if (spin_flag && kpt_flag && band_flag) {
                    double kpt_vec[3] = {pSPARC->k1_loc[kpt_shift]*pSPARC->range_x/(2.0*M_PI),
                                         pSPARC->k2_loc[kpt_shift]*pSPARC->range_y/(2.0*M_PI),
                                         pSPARC->k3_loc[kpt_shift]*pSPARC->range_z/(2.0*M_PI) };
                    if (pSPARC->isGammaPoint) {
                        print_orbital_real(pSPARC->Xorb + band_shift*DMnd + kpt_shift*size_k + spin_shift*size_s, gridsizes, 
                            pSPARC->DMVertices_dmcomm, pSPARC->dV, fname, spin, kpt, kpt_vec, band, pSPARC->dmcomm);
                    } else {
                        print_orbital_complex(pSPARC->Xorb_kpt + band_shift*DMnd + kpt_shift*size_k + spin_shift*size_s, gridsizes, 
                            pSPARC->DMVertices_dmcomm, pSPARC->dV, fname, spin, kpt, kpt_vec, band, pSPARC->dmcomm);
                    }
                }
                MPI_Barrier(alldmcomm);
            }
        }
    }
    MPI_Comm_free(&alldmcomm);
}

/**
 * @brief   Print real Kohn-Sham orbitals
 */
void print_orbital_real(
    double *x, int *gridsizes, int *DMVertices, double dV,
    char *fname, int spin_index, int kpt_index, double *kpt_vec, int band_index, MPI_Comm comm
) 
{
    if (comm == MPI_COMM_NULL) return;
    int nproc_comm, rank_comm;
    MPI_Comm_size(comm, &nproc_comm);
    MPI_Comm_rank(comm, &rank_comm);

    // global size of the vector
    int Nx = gridsizes[0];
    int Ny = gridsizes[1];
    int Nz = gridsizes[2];
    int Nd = Nx * Ny * Nz;
    double *x_global = NULL;
    if (rank_comm == 0) {
        x_global = (double*)malloc(Nd * sizeof(double));
    }

    if (nproc_comm > 1) { // if there's more than one process, need to collect x first
        int sdims[3], periods[3], my_coords[3];
        MPI_Cart_get(comm, 3, sdims, periods, my_coords);
        /* use DD2DD to collect distributed data */
        // create a cartesian topology on one process (rank 0)
        int rdims[3] = {1,1,1}, rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_comm) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int rperiods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, rdims, rperiods, 0, &recv_comm);
        }
        
        D2D_OBJ d2d_sender, d2d_recvr;
        rDMVert[0] = 0; rDMVert[1] = Nx-1;
        rDMVert[2] = 0; rDMVert[3] = Ny-1;
        rDMVert[4] = 0; rDMVert[5] = Nz-1;
        
        // set up D2D targets, note that this is time consuming if 
        // number of processes is large (> 1000), in that case, do
        // this step only once and keep the d2d target objects
        Set_D2D_Target(
            &d2d_sender, &d2d_recvr, gridsizes, DMVertices, 
            rDMVert, comm, sdims, recv_comm, rdims, comm
        );
        
        // collect vector to one process   
        D2D(&d2d_sender, &d2d_recvr, gridsizes, DMVertices, x, rDMVert, 
            x_global, comm, sdims, recv_comm, rdims, comm);
        
        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, comm, recv_comm);

        if (!rank_comm) MPI_Comm_free(&recv_comm);
    } else {
        memcpy(x_global, x, sizeof(double)*Nd);
    }

    if (rank_comm == 0) {
        // scale psi to make it L2-norm = 1
        for (int i = 0; i < Nd; i++) x_global[i] /= sqrt(dV);

        FILE *output_fp = fopen(fname,"ab");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",fname);
            exit(EXIT_FAILURE);
        }

        fwrite(&spin_index, sizeof(int), 1, output_fp);
        fwrite(&kpt_index, sizeof(int), 1, output_fp);
        fwrite(kpt_vec, 3, sizeof(double) , output_fp);
        fwrite(&band_index, sizeof(int), 1, output_fp);
        fwrite(x_global, Nd, sizeof(double) , output_fp);
        fclose(output_fp);
    }
    
    // free the collected data after printing to file
    if (rank_comm == 0) {
        free(x_global);
    }
}


/**
 * @brief   Print complex Kohn-Sham orbitals
 */
void print_orbital_complex(
    double complex *x, int *gridsizes, int *DMVertices, double dV,
    char *fname, int spin_index, int kpt_index, double *kpt_vec, int band_index, MPI_Comm comm
) 
{
    if (comm == MPI_COMM_NULL) return;
    int nproc_comm, rank_comm;
    MPI_Comm_size(comm, &nproc_comm);
    MPI_Comm_rank(comm, &rank_comm);

    // global size of the vector
    int Nx = gridsizes[0];
    int Ny = gridsizes[1];
    int Nz = gridsizes[2];
    int Nd = Nx * Ny * Nz;
    double complex *x_global = NULL;

    if (rank_comm == 0) {
        x_global = (double complex *)malloc(Nd * sizeof(double complex));
    }

    if (nproc_comm > 1) { // if there's more than one process, need to collect x first
        int sdims[3], periods[3], my_coords[3];
        MPI_Cart_get(comm, 3, sdims, periods, my_coords);
        /* use DD2DD to collect distributed data */
        // create a cartesian topology on one process (rank 0)
        int rdims[3] = {1,1,1}, rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_comm) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int rperiods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, rdims, rperiods, 0, &recv_comm);
        }
        
        D2D_OBJ d2d_sender, d2d_recvr;
        rDMVert[0] = 0; rDMVert[1] = Nx-1;
        rDMVert[2] = 0; rDMVert[3] = Ny-1;
        rDMVert[4] = 0; rDMVert[5] = Nz-1;
        
        // set up D2D targets, note that this is time consuming if 
        // number of processes is large (> 1000), in that case, do
        // this step only once and keep the d2d target objects
        Set_D2D_Target(
            &d2d_sender, &d2d_recvr, gridsizes, DMVertices, 
            rDMVert, comm, sdims, recv_comm, rdims, comm
        );
        
        // collect vector to one process   
        D2D_kpt(&d2d_sender, &d2d_recvr, gridsizes, DMVertices, x, rDMVert, 
            x_global, comm, sdims, recv_comm, rdims, comm);
        
        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, comm, recv_comm);
        
        if (!rank_comm) MPI_Comm_free(&recv_comm);
    } else {
        memcpy(x_global, x, sizeof(double complex) * Nd);
    }
    
    if (rank_comm == 0) {
        // scale psi to make it L2-norm = 1
        for (int i = 0; i < Nd; i++) x_global[i] /= sqrt(dV);

        FILE *output_fp = fopen(fname,"ab");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",fname);
            exit(EXIT_FAILURE);
        }

        fwrite(&spin_index, sizeof(int), 1, output_fp);
        fwrite(&kpt_index, sizeof(int), 1, output_fp);
        fwrite(kpt_vec, 3, sizeof(double) , output_fp);
        fwrite(&band_index, sizeof(int), 1, output_fp);
        fwrite( x_global, Nd, sizeof(double complex) , output_fp );
        fclose(output_fp);
    }
    
    // free the collected data after printing to file
    if (rank_comm == 0) {
        free(x_global);
    }
}



/**
 * @brief   Print Energy density
 */
void printEnergyDensity(SPARC_OBJ *pSPARC)
{
    int rank, nproc_dmcomm, rank_dmcomm, nproc_dmcomm_phi, rank_dmcomm_phi;
    int DMnd, Nd;
    double *KineticRho, *ExxRho, *ExcRho;
    KineticRho = ExxRho = ExcRho = NULL;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Nd = pSPARC->Nd;
    DMnd = pSPARC->Nd_d_dmcomm;

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    // compute kinetic energy density
    pSPARC->KineticRho = (double *) calloc(pSPARC->Nd_d_dmcomm * (2*pSPARC->Nspin-1), sizeof(double));
    compute_Kinetic_Density_Tau(pSPARC, pSPARC->KineticRho);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Time for calculating kinetic energy density: %.3f ms\n", (t2-t1)*1e3);
#endif

    // compute exchange correlation energy density
    pSPARC->ExcRho = (double *) calloc( pSPARC->Nd_d, sizeof(double));
    Calculate_xc_energy_density(pSPARC, pSPARC->ExcRho);
#ifdef DEBUG
    t1 = MPI_Wtime();
if (rank == 0) printf("Time for calculating exchange correlation energy density: %.3f ms\n", (t1-t2)*1e3);
#endif

    if (pSPARC->usefock > 0) {
        // compute exact exchange energy density 
        pSPARC->ExxRho = (double *) calloc( pSPARC->Nd_d_dmcomm * (2*pSPARC->Nspin-1), sizeof(double));
        computeExactExchangeEnergyDensity(pSPARC, pSPARC->ExxRho);
#ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Time for calculating exact exchange energy density: %.3f ms\n", (t2-t1)*1e3);
#endif
    }

    // start printing
    int n_rho = 1;
    if(pSPARC->Nspin > 1) { // for spin polarized systems
        n_rho = 3; // rho, rho_up, rho_down
    }

    // Assume rank-0 core has spin, kpt, band index = 0 and rank_dmcomm = 0
    if (rank == 0) {
        if ((pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 
            && pSPARC->bandcomm_index == 0 && rank_dmcomm == 0) == 0) {
            if (rank != 0) {
                printf("ERROR: Assumption that rank-0 core has spin, kpt, band index = 0 and rank_dmcomm = 0 failed!\n"
                    "       Further implementation is required!\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    if (pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);
        MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);
    } else {
        nproc_dmcomm = 0;
        rank_dmcomm = -1;
    }

    // gather densities in dmcomm
    if (rank == 0) {
        KineticRho = (double*)malloc(Nd * n_rho * sizeof(double));
        ExcRho = (double*)malloc(Nd * sizeof(double));
        if (pSPARC->usefock > 0) {
            ExxRho = (double*)malloc(Nd * n_rho * sizeof(double));
        }
    }
    
    GatherEnergyDensity_dmcomm(pSPARC, pSPARC->KineticRho, KineticRho);
    if (pSPARC->Nspin > 1) {
        GatherEnergyDensity_dmcomm(pSPARC, pSPARC->KineticRho + DMnd, KineticRho + Nd);
        GatherEnergyDensity_dmcomm(pSPARC, pSPARC->KineticRho + 2*DMnd, KineticRho + 2*Nd);
    }
    if (pSPARC->usefock > 0) {
        GatherEnergyDensity_dmcomm(pSPARC, pSPARC->ExxRho, ExxRho);
        if (pSPARC->Nspin > 1) {
            GatherEnergyDensity_dmcomm(pSPARC, pSPARC->ExxRho + DMnd, ExxRho + Nd);
            GatherEnergyDensity_dmcomm(pSPARC, pSPARC->ExxRho + 2*DMnd, ExxRho + 2*Nd);
        }
    }

    // gather densities in dmcomm_phi
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        MPI_Comm_size(pSPARC->dmcomm_phi, &nproc_dmcomm_phi);
        MPI_Comm_rank(pSPARC->dmcomm_phi, &rank_dmcomm_phi);
    } else {
        nproc_dmcomm_phi = 0;
        rank_dmcomm_phi = -1;
    }
    
    // Assume rank-0 core has rank_dmcomm_phi = 0
    GatherEnergyDensity_dmcomm_phi(pSPARC, pSPARC->ExcRho, ExcRho);
    
    if (rank == 0) {
        // print in cube format
        if (pSPARC->Nspin == 1) {
            printDens_cube(pSPARC, KineticRho, pSPARC->KinEnDensTCubFilename, "Kinetic energy density");
        } else {
            printDens_cube(pSPARC, KineticRho, pSPARC->KinEnDensTCubFilename, "Total kinetic energy density");
            printDens_cube(pSPARC, KineticRho + Nd, pSPARC->KinEnDensUCubFilename, "Spin-up kinetic energy density");
            printDens_cube(pSPARC, KineticRho + 2*Nd, pSPARC->KinEnDensDCubFilename, "Spin-down kinetic energy density");
        }
        printDens_cube(pSPARC, ExcRho, pSPARC->XcEnDensCubFilename, "Exchange correlation energy density (without hybrid contribution)");
        if (pSPARC->usefock > 0) {
            if (pSPARC->Nspin == 1) {
                printDens_cube(pSPARC, ExxRho, pSPARC->ExxEnDensTCubFilename, "Exact exchange energy density");
            } else {
                printDens_cube(pSPARC, ExxRho, pSPARC->ExxEnDensTCubFilename, "Total Exact exchange energy density");
                printDens_cube(pSPARC, ExxRho + Nd, pSPARC->ExxEnDensUCubFilename, "Spin-up exact exchange energy density");
                printDens_cube(pSPARC, ExxRho + 2*Nd, pSPARC->ExxEnDensDCubFilename, "Spin-down exact exchange energy density");
            }
        }
    }

    if (rank == 0) {
        free(KineticRho);
        free(ExcRho);
        if (pSPARC->usefock > 0) {
            free(ExxRho);
        }
    }

    free(pSPARC->ExcRho);
    free(pSPARC->KineticRho);
    if (pSPARC->usefock > 0) {
        free(pSPARC->ExxRho);
    }
}


/**
 * @brief   Gather Energy Density from dmcomm
 */
void GatherEnergyDensity_dmcomm(SPARC_OBJ *pSPARC, double *rho_send, double *rho_recv)
{
    if (!(pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0 && pSPARC->dmcomm != MPI_COMM_NULL)) {
        return;
    }

    int rank, nproc_dmcomm, rank_dmcomm, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);
    MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);

    if (nproc_dmcomm > 1) {
        int gridsizes[3], sdims[3], rdims[3], rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_dmcomm) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int dims[3] = {1,1,1}, periods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &recv_comm);
        }
        D2D_OBJ d2d_sender, d2d_recvr;
        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;
        sdims[0] = pSPARC->npNdx;
        sdims[1] = pSPARC->npNdy;
        sdims[2] = pSPARC->npNdz;
        rdims[0] = rdims[1] = rdims[2] = 1;
        rDMVert[0] = 0; rDMVert[1] = pSPARC->Nx-1;
        rDMVert[2] = 0; rDMVert[3] = pSPARC->Ny-1;
        rDMVert[4] = 0; rDMVert[5] = pSPARC->Nz-1;
        
        // set up D2D targets
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices_dmcomm, rDMVert, pSPARC->dmcomm, 
                       sdims, recv_comm, rdims, pSPARC->dmcomm);

        // send Kinetic energy density
        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices_dmcomm, rho_send, rDMVert, 
            rho_recv, pSPARC->dmcomm, sdims, recv_comm, rdims, pSPARC->dmcomm);
        
        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, pSPARC->dmcomm, recv_comm);
        if (rank_dmcomm == 0) 
            MPI_Comm_free(&recv_comm);
    } else {
        for (i = 0; i < pSPARC->Nd; i++) {
            rho_recv[i] = rho_send[i];
        }
    }
}


/**
 * @brief   Gather Energy Density from dmcomm_phi
 */
void GatherEnergyDensity_dmcomm_phi(SPARC_OBJ *pSPARC, double *rho_send, double *rho_recv)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;

    int rank, nproc_dmcomm_phi, rank_dmcomm_phi, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(pSPARC->dmcomm_phi, &nproc_dmcomm_phi);
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank_dmcomm_phi);

    if (nproc_dmcomm_phi > 1) {
        int gridsizes[3], sdims[3], rdims[3], rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_dmcomm_phi) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int dims[3] = {1,1,1}, periods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &recv_comm);
        }
        D2D_OBJ d2d_sender, d2d_recvr;
        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;
        sdims[0] = pSPARC->npNdx_phi;
        sdims[1] = pSPARC->npNdy_phi;
        sdims[2] = pSPARC->npNdz_phi;
        rdims[0] = rdims[1] = rdims[2] = 1;
        rDMVert[0] = 0; rDMVert[1] = pSPARC->Nx-1;
        rDMVert[2] = 0; rDMVert[3] = pSPARC->Ny-1;
        rDMVert[4] = 0; rDMVert[5] = pSPARC->Nz-1;
        
        // set up D2D targets
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, rDMVert, pSPARC->dmcomm_phi, 
                       sdims, recv_comm, rdims, pSPARC->dmcomm_phi);

        // send Exchange correlation energy density
        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, rho_send, rDMVert, 
            rho_recv, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);

        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, pSPARC->dmcomm_phi, recv_comm);
        if (rank_dmcomm_phi == 0) 
            MPI_Comm_free(&recv_comm);
    } else {
        for (i = 0; i < pSPARC->Nd; i++) {
            rho_recv[i] = rho_send[i];
        }
    }
}