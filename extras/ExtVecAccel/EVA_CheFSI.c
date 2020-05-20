/**
 * @file    EVA_CheFSI.c
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for Chebyshev Filtering Subspace Iteration 
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#include "isddft.h"
#include "EVA_buff.h"
#include "EVA_Hamil_MV.h"

#ifdef USE_EVA_CUDA_MODULE
#include "EVA_CheFSI_CUDA.h"
#endif

extern EVA_buff_t EVA_buff;

// Perform Chebyshev filtering on CPU
void EVA_Chebyshev_Filtering_CPU(
    const SPARC_OBJ *pSPARC, const int *DMVertices, const int ncol, 
    const int m, const double a, const double b, const double a0, 
    MPI_Comm comm, double *X, double *Y
)
{  
    #ifdef DEBUG   
    if (EVA_buff->my_global_rank == 0) printf("Start Chebyshev filtering on CPU ... \n");
    #endif
    
    int DMnd;
    DMnd  = (1 - DMVertices[0] + DMVertices[1]);
    DMnd *= (1 - DMVertices[2] + DMVertices[3]); 
    DMnd *= (1 - DMVertices[4] + DMVertices[5]);
    int X_len_tot = DMnd * ncol;    
    
    double e = 0.5 * (b - a);
    double c = 0.5 * (b + a);
    double sigma  = e / (a0 - c);
    double gamma  = 2.0 / sigma;
    double vscal1 = sigma / e;
    double sigma1 = sigma;
    double sigma2, vscal2; 
    
    size_t Ynew_size = sizeof(double) * (size_t)(X_len_tot);
    if (Ynew_size > EVA_buff->Ynew_size)
    {
        free(EVA_buff->Ynew);
        EVA_buff->Ynew_size = Ynew_size;
        EVA_buff->Ynew = (double *) malloc(Ynew_size);
        assert(EVA_buff->Ynew != NULL);
    }
    double *Ynew = EVA_buff->Ynew;
    
    EVA_Hamil_MatVec(
        pSPARC, DMnd, DMVertices, 
        ncol, -c, pSPARC->Veff_loc_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj,
        X, Y, comm
    );

    // Scale Y by (sigma1 / e)
    #pragma omp parallel for simd
    for (int i = 0; i < X_len_tot; i++) Y[i] *= vscal1;

    double *tmp, *Y0 = Y;  // Backup the original Y pointer
    
    for (int j = 1; j < m; j++) 
    {
        sigma2 = 1.0 / (gamma - sigma);

        // Ynew = (H - c*I) * Y
        EVA_Hamil_MatVec(
            pSPARC, DMnd, DMVertices,
            ncol, -c, pSPARC->Veff_loc_dmcomm,
            pSPARC->Atom_Influence_nloc, pSPARC->nlocProj,
            Y, Ynew, comm
        );
        
        // Ynew = (2*sigma2/e) * Ynew - (sigma*sigma2) * X
        vscal1 = 2.0 * sigma2 / e; 
        vscal2 = sigma * sigma2;
        #pragma omp parallel for simd
        for (int i = 0; i < X_len_tot; i++) 
        {
            Ynew[i] *= vscal1;
            Ynew[i] -= vscal2 * X[i];
        }
        
        // Update X and Y: X = Y, Y = Ynew; swap pointers to reduce data movement
        tmp = X; X = Y; Y = Ynew; Ynew = tmp;
        
        sigma = sigma2;
    }
    
    // Need to copy results to the correct pointer
    if (Y != Y0) 
    {
        #pragma omp parallel for simd
        for (int i = 0; i < X_len_tot; i++) Y0[i] = Y[i];
    }
}

// Perform Chebyshev filtering
void EVA_Chebyshev_Filtering(
    const SPARC_OBJ *pSPARC, const int *DMVertices, const int ncol, 
    const int m, const double a, const double b, const double a0, 
    MPI_Comm comm, double *X, double *Y
)
{
    if (comm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;
   
    int nproc;
    MPI_Comm_size(comm, &nproc);
    
    if (nproc == 1 && EVA_buff->use_EVA_CUDA == 1) 
    {
        #ifdef USE_EVA_CUDA_MODULE
        EVA_Chebyshev_Filtering_CUDA(pSPARC, DMVertices, ncol, m, a, b, a0, comm, X, Y);
        #else
        if (EVA_buff->my_global_rank == 0) printf("How can you trigger this??");
        assert(EVA_buff->use_EVA_CUDA == 0);
        #endif
    } else {
        EVA_Chebyshev_Filtering_CPU(pSPARC, DMVertices, ncol, m, a, b, a0, comm, X, Y);
    }
}
