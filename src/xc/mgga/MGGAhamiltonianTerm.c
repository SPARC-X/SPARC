/**
 * @file    MGGAhamiltonianTerm.c
 * @brief   This file contains the metaGGA term of Hamiltonian*Wavefunction, -1/2 * \nabla*(rho*(\part epsilon / \part tau) * \nabla psi)
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "isddft.h"
#include "MGGAhamiltonianTerm.h"
#include "tools.h"
#include "parallelization.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"

/**
 * @brief   the function to compute the mGGA term in Hamiltonian, called by Hamiltonian_vectors_mult
 */
void compute_mGGA_term_hamil(const SPARC_OBJ *pSPARC, double *x, int ncol, int colLength, int *DMVertices, double *vxcMGGA3_dm, double *mGGAterm, int spin, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    double *Dx_x = (double *) calloc(colLength, sizeof(double));
    assert(Dx_x != NULL);
    double *Dx_y = (double *) calloc(colLength, sizeof(double));
    assert(Dx_y != NULL);
    double *Dx_z = (double *) calloc(colLength, sizeof(double));
    assert(Dx_z != NULL);
    double *Dvxc3Dx_x = (double *) calloc(colLength, sizeof(double));
    assert(Dvxc3Dx_x != NULL);
    double *Dvxc3Dx_y = (double *) calloc(colLength, sizeof(double));
    assert(Dvxc3Dx_y != NULL);
    double *Dvxc3Dx_z = (double *) calloc(colLength, sizeof(double));
    assert(Dvxc3Dx_z != NULL);

    int i, j;
    for (i = 0; i < ncol; i++) {
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, &(x[i*(unsigned)colLength]), Dx_x, 0, comm);
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, &(x[i*(unsigned)colLength]), Dx_y, 1, comm);
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, &(x[i*(unsigned)colLength]), Dx_z, 2, comm);

        if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){ // transform for unorthogonal cell
            double DxAfter[3], DxBefore[3];

            DxAfter[0] = 0.0; DxAfter[1] = 0.0; DxAfter[2] = 0.0;
            for (j = 0; j < colLength; j++) {
                DxBefore[0] = Dx_x[j]; DxBefore[1] = Dx_y[j]; DxBefore[2] = Dx_z[j];

                DxAfter[0] = DxBefore[0] * pSPARC->lapcT[0] + DxBefore[1] * pSPARC->lapcT[1] + DxBefore[2] * pSPARC->lapcT[2];
                DxAfter[1] = DxBefore[0] * pSPARC->lapcT[3] + DxBefore[1] * pSPARC->lapcT[4] + DxBefore[2] * pSPARC->lapcT[5];
                DxAfter[2] = DxBefore[0] * pSPARC->lapcT[6] + DxBefore[1] * pSPARC->lapcT[7] + DxBefore[2] * pSPARC->lapcT[8];

                Dx_x[j] = DxAfter[0]; 
                Dx_y[j] = DxAfter[1]; 
                Dx_z[j] = DxAfter[2]; 
            }
        }

        for (j = 0; j < colLength; j++) {
            Dx_x[j] *= vxcMGGA3_dm[j]; 
            Dx_y[j] *= vxcMGGA3_dm[j]; 
            Dx_z[j] *= vxcMGGA3_dm[j]; // Now the vectors are Vxc3*gradX
        }
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, Dx_x, Dvxc3Dx_x, 0, comm);
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, Dx_y, Dvxc3Dx_y, 1, comm);
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, Dx_z, Dvxc3Dx_z, 2, comm);
        
        for (j = 0; j < colLength; j++) {
            mGGAterm[j + i*colLength] = Dvxc3Dx_x[j] + Dvxc3Dx_y[j] + Dvxc3Dx_z[j];
        }
    }

    #ifdef DEBUG
    t2 = MPI_Wtime();
        if (rank == 0) printf("end of Calculating mGGA term in Hamiltonian, took %.3f ms\n", (t2 - t1)*1000);
    #endif
    free(Dx_x); free(Dx_y); free(Dx_z);
    free(Dvxc3Dx_x); free(Dvxc3Dx_y); free(Dvxc3Dx_z);
}

/**
 * @brief   the function to compute the mGGA term in Hamiltonian, called by Hamiltonian_vectors_mult_kpt
 */
void compute_mGGA_term_hamil_kpt(const SPARC_OBJ *pSPARC, double _Complex *x, int ncol, int colLength, int *DMVertices, double *vxcMGGA3_dm, double _Complex *mGGAterm, int spin, int kpt, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    
    int size_k = colLength * ncol;
    double _Complex *Dx_x_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dx_x_kpt != NULL);
    double _Complex *Dx_y_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dx_y_kpt != NULL);
    double _Complex *Dx_z_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dx_z_kpt != NULL);
    double _Complex *Dvxc3Dx_x_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dvxc3Dx_x_kpt != NULL);
    double _Complex *Dvxc3Dx_y_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dvxc3Dx_y_kpt != NULL);
    double _Complex *Dvxc3Dx_z_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dvxc3Dx_z_kpt != NULL);

    int j; // seems that in computations having k-point, there is no need to loop over bands

    double _Complex *X_kpt = x;
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, X_kpt, Dx_x_kpt, 0, pSPARC->k1_loc[kpt], comm);
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, X_kpt, Dx_y_kpt, 1, pSPARC->k2_loc[kpt], comm);
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, X_kpt, Dx_z_kpt, 2, pSPARC->k3_loc[kpt], comm);
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){ // transform for unorthogonal cell
        double _Complex DxAfter[3] = {0.0};
        for (j = 0; j < size_k; j++) {
            double _Complex DxBefore[3] = {Dx_x_kpt[j], Dx_y_kpt[j], Dx_z_kpt[j]};
            // matrixTimesVec_3d_complex(lapcT, DxBefore, DxAfter);
            DxAfter[0] = DxBefore[0] * pSPARC->lapcT[0] + DxBefore[1] * pSPARC->lapcT[1] + DxBefore[2] * pSPARC->lapcT[2];
            DxAfter[1] = DxBefore[0] * pSPARC->lapcT[3] + DxBefore[1] * pSPARC->lapcT[4] + DxBefore[2] * pSPARC->lapcT[5];
            DxAfter[2] = DxBefore[0] * pSPARC->lapcT[6] + DxBefore[1] * pSPARC->lapcT[7] + DxBefore[2] * pSPARC->lapcT[8];
            Dx_x_kpt[j] = DxAfter[0]; 
            Dx_y_kpt[j] = DxAfter[1]; 
            Dx_z_kpt[j] = DxAfter[2]; 
        }
    }

    for (j = 0; j < size_k; j++) {
        Dx_x_kpt[j] *= vxcMGGA3_dm[j%colLength]; 
        Dx_y_kpt[j] *= vxcMGGA3_dm[j%colLength]; 
        Dx_z_kpt[j] *= vxcMGGA3_dm[j%colLength]; // Now the vectors are Vxc3*gradX
    }
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, Dx_x_kpt, Dvxc3Dx_x_kpt, 0, pSPARC->k1_loc[kpt], comm);
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, Dx_y_kpt, Dvxc3Dx_y_kpt, 1, pSPARC->k2_loc[kpt], comm);
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, Dx_z_kpt, Dvxc3Dx_z_kpt, 2, pSPARC->k3_loc[kpt], comm);
    
    for (j = 0; j < size_k; j++) {
        mGGAterm[j] = Dvxc3Dx_x_kpt[j] + Dvxc3Dx_y_kpt[j] + Dvxc3Dx_z_kpt[j];
    }
    // if ((pSPARC->countPotentialCalculate == 5) && (kpt == 1) && (comm == pSPARC->dmcomm)) {
    //     FILE *compute_mGGA_term_hamil_kpt = fopen("X_mGGA_term_hamil_kpt.txt","w");
    //     fprintf(compute_mGGA_term_hamil_kpt, "SCF %d, ncol %d, colLength %d, spin %d, kpt %d\n", 
    //     pSPARC->countPotentialCalculate, ncol, colLength, spin, kpt);
    //     int index;
    //     fprintf(compute_mGGA_term_hamil_kpt, "SCF 5, vxcMGGA3_dm is listed below\n");
    //     for (index = 0; index < colLength; index++) {
    //         fprintf(compute_mGGA_term_hamil_kpt, "%10.9E\n", vxcMGGA3_dm[index]);
    //     }
        
    //     fprintf(compute_mGGA_term_hamil_kpt, "SCF 5, kpt 1, 2nd column [colLength + index] of x is listed below\n");
    //     for (index = 0; index < colLength; index++) {
    //         fprintf(compute_mGGA_term_hamil_kpt, "%10.9E %10.9E\n", creal(x[colLength + index]), cimag(x[colLength + index]));
    //     }
        
    //     fprintf(compute_mGGA_term_hamil_kpt, "2nd column [colLength + index] of mGGAterm is listed below\n");
    //     for (index = 0; index < colLength; index++) {
    //         fprintf(compute_mGGA_term_hamil_kpt, "%10.9E %10.9E\n", creal(mGGAterm[colLength + index]), cimag(mGGAterm[colLength + index]));
    //     }
    //     fclose(compute_mGGA_term_hamil_kpt);
    // }

    #ifdef DEBUG
    t2 = MPI_Wtime();
        if (rank == 0) printf("end of Calculating mGGA term in Hamiltonian, took %.3f ms\n", (t2 - t1)*1000);
    #endif
    
    free(Dx_x_kpt); free(Dx_y_kpt); free(Dx_z_kpt);
    free(Dvxc3Dx_x_kpt); free(Dvxc3Dx_y_kpt); free(Dvxc3Dx_z_kpt);
}