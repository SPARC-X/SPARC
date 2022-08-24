/**
 * @file    d3initialization.c
 * @brief   This file contains the function for initializing variables and coefficients of DFT-D3.
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
#include "atomdata.h"
#include "tools.h"
#include "initialization.h"
#include "readfiles.h"
#include "d3copyC6.h"
#include "d3initialization.h"

void set_D3_coefficients(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double scaledR2R4[95] = {0,
    2.00734898,  1.56637132,  5.01986934,  3.85379032,  3.64446594,
    3.10492822,  2.71175247,  2.59361680,  2.38825250,  2.21522516,
    6.58585536,  5.46295967,  5.65216669,  4.88284902,  4.29727576,
    4.04108902,  3.72932356,  3.44677275,  7.97762753,  7.07623947,
    6.60844053,  6.28791364,  6.07728703,  5.54643096,  5.80491167,
    5.58415602,  5.41374528,  5.28497229,  5.22592821,  5.09817141,
    6.12149689,  5.54083734,  5.06696878,  4.87005108,  4.59089647,
    4.31176304,  9.55461698,  8.67396077,  7.97210197,  7.43439917,
    6.58711862,  6.19536215,  6.01517290,  5.81623410,  5.65710424,
    5.52640661,  5.44263305,  5.58285373,  7.02081898,  6.46815523,
    5.98089120,  5.81686657,  5.53321815,  5.25477007, 11.02204549,
    10.15679528,  9.35167836,  9.06926079,  8.97241155,  8.90092807,
    8.85984840,  8.81736827,  8.79317710,  7.89969626,  8.80588454,
    8.42439218,  8.54289262,  8.47583370,  8.45090888,  8.47339339,
    7.83525634,  8.20702843,  7.70559063,  7.32755997,  7.03887381,
    6.68978720,  6.05450052,  5.88752022,  5.70661499,  5.78450695,
    7.79780729,  7.26443867,  6.78151984,  6.67883169,  6.39024318,
    6.09527958, 11.79156076, 11.10997644,  9.51377795,  8.67197068,
    8.77140725,  8.65402716,  8.53923501,  8.85024712};

    double scaledRcov[95] = {0,
    0.80628308, 1.15903197, 3.02356173, 2.36845659, 1.94011865,
    1.88972601, 1.78894056, 1.58736983, 1.61256616, 1.68815527,
    3.52748848, 3.14954334, 2.84718717, 2.62041997, 2.77159820,
    2.57002732, 2.49443835, 2.41884923, 4.43455700, 3.88023730,
    3.35111422, 3.07395437, 3.04875805, 2.77159820, 2.69600923,
    2.62041997, 2.51963467, 2.49443835, 2.54483100, 2.74640188,
    2.82199085, 2.74640188, 2.89757982, 2.77159820, 2.87238349,
    2.94797246, 4.76210950, 4.20778980, 3.70386304, 3.50229216,
    3.32591790, 3.12434702, 2.89757982, 2.84718717, 2.84718717,
    2.72120556, 2.89757982, 3.09915070, 3.22513231, 3.17473967,
    3.17473967, 3.09915070, 3.32591790, 3.30072128, 5.26603625,
    4.43455700, 4.08180818, 3.70386304, 3.98102289, 3.95582657,
    3.93062995, 3.90543362, 3.80464833, 3.82984466, 3.80464833,
    3.77945201, 3.75425569, 3.75425569, 3.72905937, 3.85504098,
    3.67866672, 3.45189952, 3.30072128, 3.09915070, 2.97316878,
    2.92277614, 2.79679452, 2.82199085, 2.84718717, 3.32591790,
    3.27552496, 3.27552496, 3.42670319, 3.30072128, 3.47709584,
    3.57788113, 5.06446567, 4.56053862, 4.20778980, 3.98102289,
    3.82984466, 3.85504098, 3.88023730, 3.90543362};

    pSPARC->d3Grads = (double*)malloc(sizeof(double) * 3 * pSPARC->n_atom);

    pSPARC->atomicNumbers = (int*)malloc(sizeof(int) * pSPARC->n_atom);
    pSPARC->atomScaledR2R4 = (double*)malloc(sizeof(double) * pSPARC->n_atom);
    pSPARC->atomScaledRcov = (double*)malloc(sizeof(double) * pSPARC->n_atom);
    pSPARC->atomCN = (double*)malloc(sizeof(double) * pSPARC->n_atom);
    pSPARC->atomMaxci = (int*)malloc(sizeof(int) * pSPARC->n_atom);
    
    // double *selfAtomC6 = (double*)malloc(sizeof(double) * pSPARC->n_atom);

    char elemType[8]; int ityp; int atm; int thisAtomNumber; int count = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        // first identify element type
        find_element(elemType, &pSPARC->atomType[L_ATMTYPE*ityp]);
        for (atm = 0; atm < pSPARC->nAtomv[ityp]; atm++) {
            atomdata_number(elemType, &(pSPARC->atomicNumbers[count]));
            thisAtomNumber = pSPARC->atomicNumbers[count];
            pSPARC->atomScaledR2R4[count] = scaledR2R4[thisAtomNumber];
            pSPARC->atomScaledRcov[count] = scaledRcov[thisAtomNumber];
            count++;
        }
    }

    if (pSPARC->BCx == 0) {// Periodic boundary condition
        pSPARC->BCtype[0] = 1;
    }
    else pSPARC->BCtype[0] = 0; // D boundary
    if (pSPARC->BCy == 0) {// Periodic boundary condition
        pSPARC->BCtype[1] = 1;
    }
    else pSPARC->BCtype[1] = 0; // D boundary
    if (pSPARC->BCz == 0) {// Periodic boundary condition
        pSPARC->BCtype[2] = 1;
    }
    else pSPARC->BCtype[2] = 0; // D boundary
    pSPARC->periodicBCFlag = pSPARC->BCtype[0] + pSPARC->BCtype[1] + pSPARC->BCtype[2]; // !=0: PBC

    if (pSPARC->d3Rthr < pSPARC->d3Cn_thr) {
        if (rank == 0)
            printf(RED "ERROR: D3_RTHR should not be smaller than D3_CN_THR. Please reset these two radius!\n" RESET);
        exit(EXIT_FAILURE); 
    }

    // coefficients of zero-damping and PBE functional setfuncpar.f
    if (strcmpi(pSPARC->XC, "GGA_PBE") == 0) {
        pSPARC->d3Rs6 = 1.217; // formula 4
        pSPARC->d3S18 = 0.722;
    } else if (strcmpi(pSPARC->XC, "GGA_PBEsol") == 0) {
        pSPARC->d3Rs6 = 1.345; // formula 4
        pSPARC->d3S18 = 0.612;
    } else if (strcmpi(pSPARC->XC, "GGA_RPBE") == 0) {
        pSPARC->d3Rs6 = 0.872; // formula 4
        pSPARC->d3S18 = 0.514;
    } else if (strcmpi(pSPARC->XC, "PBE0") == 0) { // TO BE ADDED
        pSPARC->d3Rs6 = 1.287; // formula 4
        pSPARC->d3S18 = 0.928;
    } else if (strcmpi(pSPARC->XC, "HSE") == 0) { // TO BE ADDED
        pSPARC->d3Rs6 = 1.129; // formula 4
        pSPARC->d3S18 = 0.109;
        if (rank == 0)
            printf("WARNING: the DFT-D3 parameters for HSE only fit HSE06. Please make sure HSE06 is the applied xc.\n");
    } else {
        if (rank == 0) 
            printf(RED "ERROR: Cannot find D3 coefficients for this functional. DFT-D3 correction calculation canceled!\n" RESET);
        exit(EXIT_FAILURE);
    }

    pSPARC->c6ab = copyC6();

}