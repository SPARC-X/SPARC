/**
 * @file    d3finalization.c
 * @brief   This file contains the functions for initializing and finalizing variables and coefficients of DFT-D3.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * S.Grimme, J.Antony, S.Ehrlich, H.Krieg, A consistent and accurate ab
 * initio parametrization of density functional dispersion correction
 * (DFT-D) for the 96 elements H-Pu
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "isddft.h"
#include "readfiles.h"
#include "initialization.h"
#include "d3finalization.h"


void free_D3_coefficients(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 0) fclose(pSPARC->d3Output);

    int inputMaxci[95] = {0,
        2,1,2,3,5,5,4,3,2,1,
        2,3,4,5,4,3,2,1,2,3,
        3,3,3,3,3,3,4,4,2,2,
        4,5,4,3,2,1,2,3,3,3,
        3,3,3,3,3,3,2,2,4,5,
        4,3,2,1,2,3,3,1,2,2,
        2,2,2,2,2,2,2,2,2,2,
        2,3,3,3,3,3,3,3,2,2,
        4,5,4,3,2,1,2,3,2,2,
        2,2,2,2};
    
    free(pSPARC->atomicNumbers);
    free(pSPARC->atomScaledR2R4);
    free(pSPARC->atomScaledRcov);
    free(pSPARC->atomCN);
    free(pSPARC->atomMaxci);
    free(pSPARC->d3Grads);
    
    int iat, jat, iadr, jadr;
    for (iat = 1; iat < 95; iat++) { // atomic number begin from 1
        for (jat = 1; jat < 95; jat++) { 
            for (iadr = 0; iadr < inputMaxci[iat]; iadr++) { // iadr begin from 0
                for (jadr = 0; jadr < inputMaxci[jat]; jadr++) {
                    free(*(*(*(*(pSPARC->c6ab + iat) + jat) + iadr) + jadr));
                }
                free(*(*(*(pSPARC->c6ab + iat) + jat) + iadr));
            }
            free(*(*(pSPARC->c6ab + iat) + jat));
        }
        free(*(pSPARC->c6ab + iat));
    }
    free(pSPARC->c6ab);
}