/**
 * @file    d3Correction.c
 * @brief   This file contains the functions for DFT-D3 correction.
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
#include <math.h>

#include "isddft.h"
#include "readfiles.h"
#include "atomdata.h"
#include "tools.h"
#include "initialization.h"

#include "d3Correction.h"

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

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

    pSPARC->d3Grads = (double*)malloc(sizeof(double) * 3 * pSPARC->n_atom);

    pSPARC->atomicNumbers = (int*)malloc(sizeof(int) * pSPARC->n_atom);
    pSPARC->atomScaledR2R4 = (double*)malloc(sizeof(double) * pSPARC->n_atom);
    pSPARC->atomScaledRcov = (double*)malloc(sizeof(double) * pSPARC->n_atom);
    pSPARC->atomCN = (double*)malloc(sizeof(double) * pSPARC->n_atom);
    pSPARC->atomMaxci = (int*)malloc(sizeof(int) * pSPARC->n_atom);
    int *ImageCN = pSPARC->nImageCN;
    int *ImageEG = pSPARC->nImageEG;
    
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
    } else {
        if (rank == 0) printf("WARNING: Cannot find D3 coefficients for this functional. Default value is rs6 = 1.217 and s18 = 0.722.\n");
        pSPARC->d3Rs6 = 1.217;
        pSPARC->d3S18 = 0.722;
    }

    pSPARC->c6ab = copyC6();

}

void d3_set_criteria(int *nImage, double rLimit, int *periodicType, double *inputLatt) {
    double cos[3];
    double rCutoff = sqrt(rLimit);
    cos[0] = solve_cos(*(inputLatt+0), *(inputLatt+1), *(inputLatt+2), *(inputLatt+3), *(inputLatt+4), *(inputLatt+5), *(inputLatt+6), *(inputLatt+7), *(inputLatt+8));
    cos[1] = solve_cos(*(inputLatt+3), *(inputLatt+4), *(inputLatt+5), *(inputLatt+6), *(inputLatt+7), *(inputLatt+8), *(inputLatt+0), *(inputLatt+1), *(inputLatt+2));
    cos[2] = solve_cos(*(inputLatt+6), *(inputLatt+7), *(inputLatt+8), *(inputLatt+0), *(inputLatt+1), *(inputLatt+2), *(inputLatt+3), *(inputLatt+4), *(inputLatt+5));
    int row;
    for (row = 0; row < 3; row++) {
        if (*(periodicType + row) == 1) {
            *(nImage + row) = floor(rCutoff / cos[row]) + 1;
        }
        else
            *(nImage + row) = 0;
    }
}

double solve_cos(double latF1, double latF2, double latF3, double latS1, double latS2, double latS3, double latT1, double latT2, double latT3){
    double cross[3];
    cross[0] = latS2*latT3 - latT2*latS3;
    cross[1] = latS3*latT1 - latT3*latS1;
    cross[2] = latS1*latT2 - latT1*latS2;
    double normCross = sqrt(pow(cross[0], 2) + pow(cross[1], 2) + pow(cross[2], 2));
    double cos = (latF1*cross[0] + latF2*cross[1] + latF3*cross[2]) / normCross;
    return cos;
}

void d3_CN(SPARC_OBJ *pSPARC, double K1) {
    int i, j, x, y, z;
    int imageNumber = (pSPARC->nImageCN[0]*2 + 1) * (pSPARC->nImageCN[1]*2 + 1) * (pSPARC->nImageCN[2]*2 + 1);
    int imageIndex = 0;
    double *imageVec = (double*)malloc(sizeof(double) * imageNumber * 3);
    for (x = -(pSPARC->nImageCN[0]); x < pSPARC->nImageCN[0] + 1; x++) {
        for (y = -(pSPARC->nImageCN[1]); y < pSPARC->nImageCN[1] + 1; y++) {
            for (z = -(pSPARC->nImageCN[2]); z < pSPARC->nImageCN[2] + 1; z++) {
                imageVec[imageIndex*3 + 0] = x*(pSPARC->lattice[0]) + y*(pSPARC->lattice[3]) + z*(pSPARC->lattice[6]);
                imageVec[imageIndex*3 + 1] = x*(pSPARC->lattice[1]) + y*(pSPARC->lattice[4]) + z*(pSPARC->lattice[7]);
                imageVec[imageIndex*3 + 2] = x*(pSPARC->lattice[2]) + y*(pSPARC->lattice[5]) + z*(pSPARC->lattice[8]);
                imageIndex++;
            }
        }
    }
    double iCoord[3]; double jCoord[3]; double relativeVec[3];
    double dist, rr;
    double CNrCutoff = sqrt(pSPARC->d3Cn_thr);
    double *atomPosition = pSPARC->atom_pos;
    for (i = 0; i < pSPARC->n_atom; i++) {
        pSPARC->atomCN[i] = 0.0;
        iCoord[0] = *(atomPosition + i*3 + 0); iCoord[1] = *(atomPosition + i*3 + 1); iCoord[2] = *(atomPosition + i*3 + 2);
        for (j = 0; j < pSPARC->n_atom; j++) {
            jCoord[0] = *(atomPosition + j*3 + 0); jCoord[1] = *(atomPosition + j*3 + 1); jCoord[2] = *(atomPosition + j*3 + 2);
            for (imageIndex = 0; imageIndex < imageNumber; imageIndex++) {
                relativeVec[0] = iCoord[0] - jCoord[0] - imageVec[imageIndex*3 + 0];
                relativeVec[1] = iCoord[1] - jCoord[1] - imageVec[imageIndex*3 + 1];
                relativeVec[2] = iCoord[2] - jCoord[2] - imageVec[imageIndex*3 + 2];
                dist = sqrt(pow(relativeVec[0], 2) + pow(relativeVec[1], 2) + pow(relativeVec[2], 2));
                if ((dist < CNrCutoff) && (dist > 1e-10)) {
                    rr = (pSPARC->atomScaledRcov[i] + pSPARC->atomScaledRcov[j]) / dist;
                    pSPARC->atomCN[i] += 1.0 / (1.0 + exp(-K1 * (rr - 1.0)));
                }
            }
        }
    }
    free(imageVec);
}

double d3_getC6(int *atomMaxci, int *atomicNumbers, double *CN, int atomI, int atomJ, double *****c6ab, double k3) {
    double c6mem = -1e-99;
    double weightSum = 0.0;
    double c6termSum = 0.0;
    double r_save = 1e99;
    double c6, CNri, CNrj, r, weight, C6result;
    int sampleI, sampleJ;
    int atominNumI = atomicNumbers[atomI];    int atominNumJ = atomicNumbers[atomJ];
    int maxciI = atomMaxci[atomI];    int maxciJ = atomMaxci[atomJ];
    for (sampleI = 0; sampleI < maxciI; sampleI++) {
        for (sampleJ = 0; sampleJ < maxciJ; sampleJ++) {
            c6 = c6ab[atominNumI][atominNumJ][sampleI][sampleJ][0];
            if (c6 > 0.0) {
                CNri = c6ab[atominNumI][atominNumJ][sampleI][sampleJ][1];
                CNrj = c6ab[atominNumI][atominNumJ][sampleI][sampleJ][2];
                r = pow((CNri - CN[atomI]), 2.0) + pow((CNrj - CN[atomJ]), 2.0);
                if (r < r_save) {
                    r_save = r;
                    c6mem = c6;
                }
                weight = exp(k3*r);
                weightSum += weight;
                c6termSum += weight * c6;
            }
        }
    }
    if (weightSum > 1e-99) 
        C6result = c6termSum / weightSum;
    else
        C6result = c6mem;
    return C6result;
}

void d3_comp_dC6_dCNij(double *C6dC6pairIJ, int *atomMaxci, int *atomicNumbers, double *CN, int atomI, int atomJ, double *****c6ab, double k3) {
    double c6mem = -1e-20;
    double weightSum = 0.0;
    double c6termSum = 0.0;
    double r_save = 1e20;
    double dC6termSumI = 0.0;
    double dWeightSumI = 0.0;
    double dC6termSumJ = 0.0;
    double dWeightSumJ = 0.0;
    double c6, CNri, CNrj, r, weight, C6result, dC6I, dC6J;
    int sampleI, sampleJ;
    int atominNumI = atomicNumbers[atomI];    int atominNumJ = atomicNumbers[atomJ];
    int maxciI = atomMaxci[atomI];    int maxciJ = atomMaxci[atomJ];
    double dWeightI, dWeightJ;
    for (sampleI = 0; sampleI < maxciI; sampleI++) {
        for (sampleJ = 0; sampleJ < maxciJ; sampleJ++) {
            c6 = c6ab[atominNumI][atominNumJ][sampleI][sampleJ][0];
            if (c6 > 0.0) {
                CNri = c6ab[atominNumI][atominNumJ][sampleI][sampleJ][1];
                CNrj = c6ab[atominNumI][atominNumJ][sampleI][sampleJ][2];
                r = pow((CNri - CN[atomI]), 2.0) + pow((CNrj - CN[atomJ]), 2.0);
                // if ((atomI == 2) && (atomJ == 1)) { // !!!!why commenting it or not will influence the result???
                //  printf("sampleI %d, sampleJ %d, CNri %12.9E, CNrj %12.9E, r %12.9E, c6 %12.9E\n", sampleI, sampleJ, CNri, CNrj, r, c6);
                // }
                double compare = r - r_save;
                // if (r < r_save) {
                if (compare < 0.0) {
                    r_save = r;
                    c6mem = c6;
                }
                weight = exp(k3*r);
                weightSum += weight;
                c6termSum += weight * c6;
                dWeightI = weight * 2.0 * k3 * (CN[atomI] - CNri);
                dC6termSumI += dWeightI*c6;
                dWeightSumI += dWeightI;
                dWeightJ = weight * 2.0 * k3 * (CN[atomJ] - CNrj);
                dC6termSumJ += dWeightJ*c6;
                dWeightSumJ += dWeightJ;
            }
        }
    }
    if (weightSum > 1e-99) {
        C6result = c6termSum / weightSum;
        dC6I = (dC6termSumI*weightSum - dWeightSumI*c6termSum) / pow(weightSum, 2.0);
        dC6J = (dC6termSumJ*weightSum - dWeightSumJ*c6termSum) / pow(weightSum, 2.0);
    }
    else {
        C6result = c6mem;
        dC6I = 0;
        dC6J = 0;
    }
    
    C6dC6pairIJ[0] = C6result;
    C6dC6pairIJ[1] = dC6I;
    C6dC6pairIJ[2] = dC6J;
}


/**
 * @brief The main function of DFT-D3. D3 energy, force are computed here. Called by Calculate_electronicGroundState in electronicGroundState.c
 */
void d3_energy_gradient(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    int natom = pSPARC->n_atom;
    double s8 = pSPARC->d3S18;
    double e = 0.0;
    double e6 = 0.0;
    double e8 = 0.0;
    double e63 = 0.0;
    double s6 = 1.0;
    double rs6 = pSPARC->d3Rs6;
    double rs8 = 1.0;
    double rs10 = rs8;
    double alp6 = 14.0;
    double alp8 = alp6 + 2;
    double alp10 = alp8 + 2;
    double K1 = 16.0;
    double K3 = -4.0;
    double cn_thr = pSPARC->d3Cn_thr;
    double *sqrtC6matrix = (double*)malloc(sizeof(double) * natom * natom);

    
    // find the number of cells to be checked in periodic directions
    double cell[3] = {pSPARC->range_x, pSPARC->range_y, pSPARC->range_z};
    int row, col;
    for (row = 0; row < 3; row++) {
        pSPARC->lattice[row*3] = pSPARC->LatUVec[row*3] * cell[row];
        pSPARC->lattice[row*3 + 1] = pSPARC->LatUVec[row*3 + 1] * cell[row];
        pSPARC->lattice[row*3 + 2] = pSPARC->LatUVec[row*3 + 2] * cell[row];
    }
    if (pSPARC->periodicBCFlag == 0) {
        for (row = 0; row < 3; row++) {
            pSPARC->nImageCN[row] = 0;
            pSPARC->nImageEG[row] = 0;
        }
    }
    else {
        d3_set_criteria(pSPARC->nImageCN, pSPARC->d3Cn_thr, pSPARC->BCtype, pSPARC->lattice);
        d3_set_criteria(pSPARC->nImageEG, pSPARC->d3Rthr, pSPARC->BCtype, pSPARC->lattice);
    }

    #ifdef DEBUG
    if (rank == 0) {
        printf("d3 nImageCN %d %d %d\n", pSPARC->nImageCN[0], pSPARC->nImageCN[1], pSPARC->nImageCN[2]);
        printf("d3 nImageEG %d %d %d\n", pSPARC->nImageEG[0], pSPARC->nImageEG[1], pSPARC->nImageEG[2]);
    } 
    #endif

    int atm;
    // at first, transfer non-cartesian coordinates to cartesian coordinates
    for (atm = 0; atm < pSPARC->n_atom; atm++) {
        nonCart2Cart_coord(pSPARC, &pSPARC->atom_pos[atm*3], &pSPARC->atom_pos[atm*3 + 1], &pSPARC->atom_pos[atm*3 + 2]);
    }
    
    d3_CN(pSPARC, K1);

    // self C6 test
    double *selfAtomC6 = (double*)malloc(sizeof(double) * pSPARC->n_atom);
    int thisAtomNumber;
    for (atm = 0; atm < pSPARC->n_atom; atm++) {
     thisAtomNumber = pSPARC->atomicNumbers[atm];
        pSPARC->atomMaxci[atm] = inputMaxci[thisAtomNumber]; // maybe there is mistake
        pSPARC->d3Grads[atm*3 + 0] = 0.0; pSPARC->d3Grads[atm*3 + 1] = 0.0; pSPARC->d3Grads[atm*3 + 2] = 0.0;
        selfAtomC6[atm] = d3_getC6(pSPARC->atomMaxci, pSPARC->atomicNumbers, pSPARC->atomCN, atm, atm, pSPARC->c6ab, K3);
    }

    int *numImageEG = pSPARC->nImageEG;
    int imageNumberEG = (numImageEG[0]*2 + 1) * (numImageEG[1]*2 + 1) * (numImageEG[2]*2 + 1);
    double *imageVecEG = (double*)malloc(sizeof(double) * imageNumberEG * 3);
    double ***drij = (double***)malloc(sizeof(double**) * imageNumberEG);
    int imageIndex, i, j;
    for (imageIndex = 0; imageIndex < imageNumberEG; imageIndex++) {
        *(drij + imageIndex) = (double**)malloc(sizeof(double*) * natom);
        for (i = 0; i < natom; i++) {
            *(*(drij + imageIndex) + i) = (double*)calloc(natom, sizeof(double));
        }
    }
    double *dC6ij = (double*)malloc(sizeof(double*) * natom * natom);
    double *FdivR6mulDC6 = (double*)calloc(natom, sizeof(double));
    
    // below is the code computing D3 energy and its analytical gradient in 2-atom pairs
    int imageX, imageY, imageZ;
    double *Lattice = pSPARC->lattice;
    double *atomPosition = pSPARC->atom_pos;
    imageIndex = 0;
    for (imageX = -numImageEG[0]; imageX < numImageEG[0] + 1; imageX++) {
        for (imageY = -numImageEG[1]; imageY < numImageEG[1] + 1; imageY++) {
            for (imageZ = -numImageEG[2]; imageZ < numImageEG[2] + 1; imageZ++) {
                imageVecEG[imageIndex*3 + 0] = imageX*Lattice[0] + imageY*Lattice[3] + imageZ*Lattice[6];
                imageVecEG[imageIndex*3 + 1] = imageX*Lattice[1] + imageY*Lattice[4] + imageZ*Lattice[7];
                imageVecEG[imageIndex*3 + 2] = imageX*Lattice[2] + imageY*Lattice[5] + imageZ*Lattice[8];
                imageIndex++;
            }
        }
    }
    
    double setC6dC6pairIJ[3]; double *C6dC6pairIJ = setC6dC6pairIJ;
    double c6, r0abNow, mulR2R4, rr, t6, t8, damp6, damp8, de6dr, de8dr, c8, FdivR6;
    double iDist2, iDist, iDist6, iDist7, iDist8, iDist9; 
    for (i = 0; i < natom; i++) {
        for (j = 0; j < i + 1; j++) {
            d3_comp_dC6_dCNij(C6dC6pairIJ, pSPARC->atomMaxci, pSPARC->atomicNumbers, pSPARC->atomCN, i, j, pSPARC->c6ab, K3);
            c6 = C6dC6pairIJ[0];

            if (i == j) {
                dC6ij[i*natom + i] = C6dC6pairIJ[1];
                sqrtC6matrix[i*natom + i] = sqrt(C6dC6pairIJ[0]);
            }
            else {
                dC6ij[i*natom + j] = C6dC6pairIJ[1];
                dC6ij[j*natom + i] = C6dC6pairIJ[2];
                sqrtC6matrix[i*natom + j] = sqrt(C6dC6pairIJ[0]);
                sqrtC6matrix[j*natom + i] = sqrtC6matrix[i*natom + j];
            }
            r0abNow = find_r0ab(pSPARC->atomicNumbers[i], pSPARC->atomicNumbers[j]);
            mulR2R4 = pSPARC->atomScaledR2R4[i] * pSPARC->atomScaledR2R4[j];
            for (imageIndex = 0; imageIndex < imageNumberEG; imageIndex++) {
                iDist2 = pow(atomPosition[i*3 + 0] - atomPosition[j*3 + 0] - imageVecEG[imageIndex*3 + 0], 2.0)
                        +pow(atomPosition[i*3 + 1] - atomPosition[j*3 + 1] - imageVecEG[imageIndex*3 + 1], 2.0)
                        +pow(atomPosition[i*3 + 2] - atomPosition[j*3 + 2] - imageVecEG[imageIndex*3 + 2], 2.0);
                if ((iDist2 > pSPARC->d3Rthr) || (iDist2 < 1e-15)) continue;
                iDist = sqrt(iDist2);
                iDist6 = pow(iDist2, 3.0);
                iDist7 = iDist6 * iDist;
                iDist8 = iDist6 * iDist2;
                iDist9 = iDist8 * iDist;
                rr = r0abNow / iDist;
                t6 = pow((rs6*rr), alp6);
                t8 = pow((rs8*rr), alp8);
                damp6 = 1.0 / (1.0 + 6 * t6);
                damp8 = 1.0 / (1.0 + 6 * t8);
                de6dr = s6 * 6 * damp6 * c6 / iDist7;
                de8dr = s8 * 6 * c6 * mulR2R4 * damp8 / iDist9;
                
                if (i == j) {
                    drij[imageIndex][i][j] -= (de6dr + 4 * de8dr) / 2.0;
                    drij[imageIndex][i][j] += (de6dr*alp6*t6*damp6 + 3*de8dr*alp8*t8*damp8) / 2.0;
                    FdivR6 = (s6 / iDist6 * damp6 + 3 * s8 * mulR2R4 / iDist8 * damp8) / 2.0;
                    e6 -= s6 / iDist6 * damp6 * c6 / 2.0;
                    e8 -= 3 * s8 * mulR2R4 / iDist8 * damp8 * c6 / 2.0;
                    FdivR6mulDC6[i] += FdivR6 * (C6dC6pairIJ[1] + C6dC6pairIJ[2]);
                }
                else {
                    drij[imageIndex][i][j] -= de6dr + 4 * de8dr;
                    drij[imageIndex][i][j] += de6dr*alp6*t6*damp6 + 3*de8dr*alp8*t8*damp8;
                    drij[imageIndex][j][i] = drij[imageIndex][i][j];
                    FdivR6 = (s6 / iDist6 * damp6 + 3 * s8 * mulR2R4 / iDist8 * damp8);
                    e6 -= s6 / iDist6 * damp6 * c6;
                    e8 -= 3 * s8 * mulR2R4 / iDist8 * damp8 * c6;
                    FdivR6mulDC6[i] += FdivR6 * C6dC6pairIJ[1];
                    FdivR6mulDC6[j] += FdivR6 * C6dC6pairIJ[2];
                }
            }
        }
    }
    
    // below is the code computing D3 energy and its analytical gradient in 3-atom pairs
    int *numImageCN = pSPARC->nImageCN;
    int imageNumberCN = (numImageCN[0]*2 + 1) * (numImageCN[1]*2 + 1) * (numImageCN[2]*2 + 1);
    double *imageVecCN = (double*)malloc(sizeof(double) * imageNumberCN * 3); // the array saves the vector from center cell to the (imageX, imageY, imageZ) image cell
    int *imageIndexCN = (int*)malloc(sizeof(int) * imageNumberCN * 3); // the array saves the indexes of the (imageX, imageY, imageZ) image cell
    imageIndex = 0;
    for (imageX = -numImageCN[0]; imageX < numImageCN[0] + 1; imageX++) {
        for (imageY = -numImageCN[1]; imageY < numImageCN[1] + 1; imageY++) {
            for (imageZ = -numImageCN[2]; imageZ < numImageCN[2] + 1; imageZ++) {
                imageVecCN[imageIndex*3 + 0] = imageX*Lattice[0] + imageY*Lattice[3] + imageZ*Lattice[6];
                imageVecCN[imageIndex*3 + 1] = imageX*Lattice[1] + imageY*Lattice[4] + imageZ*Lattice[7];
                imageVecCN[imageIndex*3 + 2] = imageX*Lattice[2] + imageY*Lattice[5] + imageZ*Lattice[8];
                imageIndexCN[imageIndex*3 + 0] = imageX;
                imageIndexCN[imageIndex*3 + 1] = imageY;
                imageIndexCN[imageIndex*3 + 2] = imageZ;
                imageIndex++;
            }
        }
    }
    double ijVec[3]; double ikVec[3]; double jkVec[3]; 
    int scopeImageIK[3][2]; // the array saves the scope of image cells of the 3rd atom k. If a cell out of the scope, we will not compute interactions between the atom i, j and any atom in this cell
    int imageIndexIJ[3];
    double ijDist, ikDist, jkDist, ijDist2, ikDist2, jkDist2, r0abij, r0abik, r0abjk, rr0ij, rr0ik, rr0jk;
    double c9, geoMean, damp3, t1, t2, t3, tDenomin, ang;
    int k, imageIJindex, imageIKindex, imageIndexEsIJ, imageIndexEsJK, imageIndexEsIK;
    double dFdamp, dAngIJ, dAngJK, dAngIK, dC9i, dC9j, dC9k;
    for (i = 0; i < natom; i++) {
        for (j = 0; j < i + 1; j++) {
            ijVec[0] = atomPosition[j*3 + 0] - atomPosition[i*3 + 0];
            ijVec[1] = atomPosition[j*3 + 1] - atomPosition[i*3 + 1];
            ijVec[2] = atomPosition[j*3 + 2] - atomPosition[i*3 + 2];
            r0abij = find_r0ab(pSPARC->atomicNumbers[i], pSPARC->atomicNumbers[j]);
            for (imageIJindex = 0; imageIJindex < imageNumberCN; imageIJindex++) {
                ijDist2 = pow(ijVec[0] + imageVecCN[imageIJindex*3 + 0], 2.0)
                        + pow(ijVec[1] + imageVecCN[imageIJindex*3 + 1], 2.0)
                        + pow(ijVec[2] + imageVecCN[imageIJindex*3 + 2], 2.0);
                ijDist = sqrt(ijDist2);
                if ((ijDist2 > cn_thr)|| (ijDist2 < 1e-15)) continue; //
                rr0ij = ijDist / r0abij;
                scopeImageIK[0][0] = max(-numImageCN[0], imageIndexCN[imageIJindex*3 + 0] - numImageCN[0]);
                scopeImageIK[0][1] = min( numImageCN[0], imageIndexCN[imageIJindex*3 + 0] + numImageCN[0]);
                scopeImageIK[1][0] = max(-numImageCN[1], imageIndexCN[imageIJindex*3 + 1] - numImageCN[1]);
                scopeImageIK[1][1] = min( numImageCN[1], imageIndexCN[imageIJindex*3 + 1] + numImageCN[1]);
                scopeImageIK[2][0] = max(-numImageCN[2], imageIndexCN[imageIJindex*3 + 2] - numImageCN[2]);
                scopeImageIK[2][1] = min( numImageCN[2], imageIndexCN[imageIJindex*3 + 2] + numImageCN[2]);
                int imageNumberIK = (scopeImageIK[0][1] - scopeImageIK[0][0] + 1) * 
                (scopeImageIK[1][1] - scopeImageIK[1][0] + 1) * (scopeImageIK[2][1] - scopeImageIK[2][0] + 1);
                double *imageVecIK = (double*)malloc(sizeof(double) * imageNumberIK * 3);
                int *imageIndexIK = (int*)malloc(sizeof(int) * imageNumberIK * 3);
                imageIndex = 0;
                for (imageX = scopeImageIK[0][0]; imageX < scopeImageIK[0][1] + 1; imageX++) {
                    for (imageY = scopeImageIK[1][0]; imageY < scopeImageIK[1][1] + 1; imageY++) {
                        for (imageZ = scopeImageIK[2][0]; imageZ < scopeImageIK[2][1] + 1; imageZ++) {
                            imageVecIK[imageIndex*3 + 0] = imageX*Lattice[0] + imageY*Lattice[3] + imageZ*Lattice[6];
                            imageVecIK[imageIndex*3 + 1] = imageX*Lattice[1] + imageY*Lattice[4] + imageZ*Lattice[7];
                            imageVecIK[imageIndex*3 + 2] = imageX*Lattice[2] + imageY*Lattice[5] + imageZ*Lattice[8];
                            imageIndexIK[imageIndex*3 + 0] = imageX;
                            imageIndexIK[imageIndex*3 + 1] = imageY;
                            imageIndexIK[imageIndex*3 + 2] = imageZ;
                            imageIndex++;
                        }
                    }
                }
                for (k = 0; k < j + 1; k++) {
                    ikVec[0] = atomPosition[k*3 + 0] - atomPosition[i*3 + 0];
                    ikVec[1] = atomPosition[k*3 + 1] - atomPosition[i*3 + 1];
                    ikVec[2] = atomPosition[k*3 + 2] - atomPosition[i*3 + 2];
                    jkVec[0] = atomPosition[k*3 + 0] - atomPosition[j*3 + 0] - imageVecCN[imageIJindex*3 + 0];
                    jkVec[1] = atomPosition[k*3 + 1] - atomPosition[j*3 + 1] - imageVecCN[imageIJindex*3 + 1];
                    jkVec[2] = atomPosition[k*3 + 2] - atomPosition[j*3 + 2] - imageVecCN[imageIJindex*3 + 2];
                    c9 = sqrtC6matrix[i*natom + j] * sqrtC6matrix[i*natom + k] * sqrtC6matrix[j*natom + k];
                    r0abik = find_r0ab(pSPARC->atomicNumbers[i], pSPARC->atomicNumbers[k]);
                    r0abjk = find_r0ab(pSPARC->atomicNumbers[j], pSPARC->atomicNumbers[k]);
                    for (imageIKindex = 0; imageIKindex < imageNumberIK; imageIKindex++) {
                        ikDist2 = pow(ikVec[0] + imageVecIK[imageIKindex*3 + 0], 2.0)
                                + pow(ikVec[1] + imageVecIK[imageIKindex*3 + 1], 2.0)
                                + pow(ikVec[2] + imageVecIK[imageIKindex*3 + 2], 2.0);
                        ikDist = sqrt(ikDist2);
                        jkDist2 = pow(jkVec[0] + imageVecIK[imageIKindex*3 + 0], 2.0)
                                + pow(jkVec[1] + imageVecIK[imageIKindex*3 + 1], 2.0)
                                + pow(jkVec[2] + imageVecIK[imageIKindex*3 + 2], 2.0);
                        jkDist = sqrt(jkDist2);
                        if ((ikDist2 > cn_thr) || (ikDist2 < 1e-15)) continue; //
                        if ((jkDist2 > cn_thr) || (jkDist2 < 1e-15)) continue; //
                        rr0ik = ikDist / r0abik;
                        rr0jk = jkDist / r0abjk;
                        geoMean = pow(rr0ij*rr0ik*rr0jk, (1.0/3.0));
                        damp3 = 1.0 / (1.0 + 6 * pow((4.0/3.0) / geoMean, alp8));
                        t1 = ijDist2 + jkDist2 - ikDist2;
                        t2 = ijDist2 + ikDist2 - jkDist2;
                        t3 = ikDist2 + jkDist2 - ijDist2;
                        tDenomin = ijDist2 * jkDist2 * ikDist2;
                        ang = (0.375*t1*t2*t3/tDenomin + 1.0) / pow(tDenomin, 1.5);
                        if ((i == j) && (i == k))
                            e63 -= damp3*c9*ang/6.0;
                        else if ((i == j) || (j == k))
                            e63 -= damp3*c9*ang/2.0;
                        else
                            e63 -= damp3*c9*ang;
                        
                        // now compute the derivatives
                        dFdamp = -2.0*alp8 * pow((4.0/3.0)/geoMean, alp8) * pow(damp3, 2.0); //d(f_dmp)/d(r_ij)
                        // derivative of i, j
                        dAngIJ = d3_dAng(ijDist2, jkDist2, ikDist2, tDenomin);
                        imageIndexEsIJ = d3_count_image(imageIndexCN[imageIJindex*3 + 0], 
                                                    imageIndexCN[imageIJindex*3 + 1], 
                                                    imageIndexCN[imageIJindex*3 + 2], numImageEG);
                        // derivative of j, k
                        dAngJK = d3_dAng(jkDist2, ikDist2, ijDist2, tDenomin);
                        imageIndexEsJK = d3_count_image(imageIndexIK[imageIKindex*3 + 0] - imageIndexCN[imageIJindex*3 + 0], 
                                                    imageIndexIK[imageIKindex*3 + 1] - imageIndexCN[imageIJindex*3 + 1], 
                                                    imageIndexIK[imageIKindex*3 + 2] - imageIndexCN[imageIJindex*3 + 2], numImageEG);
                        // derivative of i, k
                        dAngIK = d3_dAng(ikDist2, jkDist2, ijDist2, tDenomin);
                        imageIndexEsIK = d3_count_image(imageIndexIK[imageIKindex*3 + 0],
                                                    imageIndexIK[imageIKindex*3 + 1], 
                                                    imageIndexIK[imageIKindex*3 + 2], numImageEG);
                        //dC9/dCN
                        dC9i = -0.5*c9*(dC6ij[i*natom + j]/(sqrtC6matrix[i*natom + j]*sqrtC6matrix[i*natom + j]) + dC6ij[i*natom + k]/(sqrtC6matrix[i*natom + k]*sqrtC6matrix[i*natom + k]));
                        dC9j = -0.5*c9*(dC6ij[j*natom + i]/(sqrtC6matrix[i*natom + j]*sqrtC6matrix[i*natom + j]) + dC6ij[j*natom + k]/(sqrtC6matrix[j*natom + k]*sqrtC6matrix[j*natom + k]));
                        dC9k = -0.5*c9*(dC6ij[k*natom + i]/(sqrtC6matrix[i*natom + k]*sqrtC6matrix[i*natom + k]) + dC6ij[k*natom + j]/(sqrtC6matrix[j*natom + k]*sqrtC6matrix[j*natom + k]));

                        if ((i == j) && (i == k)) { // three image atoms injecting to the same atom in the cell
                            drij[imageIndexEsIJ][i][j] += dFdamp / ijDist * c9 * ang / 6.0;
                            drij[imageIndexEsIJ][i][j] -= dAngIJ * c9 * damp3 / 6.0;
                            // drij[imageIndexEsIJ][j][i] = drij[imageIndexEsIJ][i][j];

                            drij[imageIndexEsJK][j][k] += dFdamp / jkDist * c9 * ang / 6.0;
                            drij[imageIndexEsJK][j][k] -= dAngJK * c9 * damp3 / 6.0;
                            // drij[imageIndexEsJK][k][j] = drij[imageIndexEsJK][k][j];

                            drij[imageIndexEsIK][i][k] += dFdamp / ikDist * c9 * ang / 6.0;
                            drij[imageIndexEsIK][i][k] -= dAngIK * c9 * damp3 / 6.0;
                            // drij[imageIndexEsIK][k][i] = drij[imageIndexEsIK][i][k];
                            FdivR6 = ang * damp3 / 6.0; 
                        }
                        else if ((i == j) || (j == k)) { // two of three image atoms injecting to the same atom in the cell
                            drij[imageIndexEsIJ][i][j] += dFdamp / ijDist * c9 * ang / 2.0;
                            drij[imageIndexEsIJ][i][j] -= dAngIJ * c9 * damp3 / 2.0;
                            drij[imageIndexEsIJ][j][i] = drij[imageIndexEsIJ][i][j];

                            drij[imageIndexEsJK][j][k] += dFdamp / jkDist * c9 * ang / 2.0;
                            drij[imageIndexEsJK][j][k] -= dAngJK * c9 * damp3 / 2.0;
                            drij[imageIndexEsJK][k][j] = drij[imageIndexEsJK][k][j];

                            drij[imageIndexEsIK][i][k] += dFdamp / ikDist * c9 * ang / 2.0;
                            drij[imageIndexEsIK][i][k] -= dAngIK * c9 * damp3 / 2.0;
                            drij[imageIndexEsIK][k][i] = drij[imageIndexEsIK][i][k];

                            FdivR6 = ang * damp3 / 2.0; 
                        }
                        else {
                            drij[imageIndexEsIJ][i][j] += dFdamp / ijDist * c9 * ang;
                            drij[imageIndexEsIJ][i][j] -= dAngIJ * c9 * damp3;
                            drij[imageIndexEsIJ][j][i] = drij[imageIndexEsIJ][i][j];

                            drij[imageIndexEsJK][j][k] += dFdamp / jkDist * c9 * ang;
                            drij[imageIndexEsJK][j][k] -= dAngJK * c9 * damp3;
                            drij[imageIndexEsJK][k][j] = drij[imageIndexEsJK][k][j];

                            drij[imageIndexEsIK][i][k] += dFdamp / ikDist * c9 * ang;
                            drij[imageIndexEsIK][i][k] -= dAngIK * c9 * damp3;
                            drij[imageIndexEsIK][k][i] = drij[imageIndexEsIK][i][k];

                            FdivR6 = ang * damp3; 
                        }
                        FdivR6mulDC6[i] += FdivR6*dC9i;
                        FdivR6mulDC6[j] += FdivR6*dC9j;
                        FdivR6mulDC6[k] += FdivR6*dC9k;
                    }  
                }
                free(imageVecIK);
                free(imageIndexIK);
            }
        }
    } 
    e = e6 + e8 - e63;
    pSPARC->d3Energy[0] = e; // totalD3energy
    pSPARC->d3Energy[1] = e6; // e6
    pSPARC->d3Energy[2] = e8; // e8
    pSPARC->d3Energy[3] = -e63; // e63
    #ifdef DEBUG
    if (rank == 0) printf("d3 energy %12.9f, inside e6 %12.9f, e8 %12.9f, e63 %12.9f\n", pSPARC->d3Energy[0], pSPARC->d3Energy[1], pSPARC->d3Energy[2], pSPARC->d3Energy[3]);
    #endif
    // summary: gradient for all atoms and pressure of space
    
    for (row = 0; row < 3; row++) {
        for (col = 0; col < 3; col++) {
            pSPARC->d3Sigma[row*3 + col] = 0;
        }
    }
    double gi[3]; double iCoord[3]; double jCoord[3]; double initialijImageVecs[3]; double ijImageVecs[3]; double vec[3];
    double rcovij, expterm, dCN, xi;
    for (i = 0; i < natom; i++) {
        pSPARC->d3Grads[i*3 + 0] = 0.0; pSPARC->d3Grads[i*3 + 1] = 0.0; pSPARC->d3Grads[i*3 + 2] = 0.0;
    }
    for (i = 0; i < natom; i++) {
        gi[0] = 0.0; gi[1] = 0.0; gi[2] = 0.0;
        for (j = 0; j <= i; j++) {
            iCoord[0] = atomPosition[i*3 + 0]; iCoord[1] = atomPosition[i*3 + 1]; iCoord[2] = atomPosition[i*3 + 2];
            jCoord[0] = atomPosition[j*3 + 0]; jCoord[1] = atomPosition[j*3 + 1]; jCoord[2] = atomPosition[j*3 + 2];
            initialijImageVecs[0] = jCoord[0] - iCoord[0]; initialijImageVecs[1] = jCoord[1] - iCoord[1]; initialijImageVecs[2] = jCoord[2] - iCoord[2];
            for (imageIndex = 0; imageIndex < imageNumberEG; imageIndex++) {
                ijImageVecs[0] = initialijImageVecs[0] + imageVecEG[imageIndex*3 + 0];
                ijImageVecs[1] = initialijImageVecs[1] + imageVecEG[imageIndex*3 + 1];
                ijImageVecs[2] = initialijImageVecs[2] + imageVecEG[imageIndex*3 + 2];
                rcovij = pSPARC->atomScaledRcov[i] + pSPARC->atomScaledRcov[j];
                iDist2 = ijImageVecs[0]*ijImageVecs[0] + ijImageVecs[1]*ijImageVecs[1] + ijImageVecs[2]*ijImageVecs[2];
                if (i != j) {
                    if ((iDist2 > pSPARC->d3Rthr) || (iDist2 < 0.5)) continue;
                }
                else {
                    if (iDist2 < 1e-15) continue; // need tp discuss
                }
                iDist = sqrt(iDist2);
                if ((iDist2 < cn_thr) && (iDist2 > 1e-15)) {
                    expterm = exp(-K1*(rcovij/iDist - 1));
                    dCN = -K1*rcovij*expterm / pow(iDist*(expterm + 1), 2.0);
                }
                else {
                    dCN=0.0;
                }
                // imageIndexEs = count_image(imageIndex(imagej, 1), imageIndex(imagej, 2), imageIndex(imagej, 3), nImageEG);
                if (i != j) {
                    xi = drij[imageIndex][i][j] + dCN*(FdivR6mulDC6[i] + FdivR6mulDC6[j]);
                }
                else {
                    xi = drij[imageIndex][i][j] + dCN*FdivR6mulDC6[i];
                }
                vec[0] = xi * ijImageVecs[0] / iDist;
                vec[1] = xi * ijImageVecs[1] / iDist;
                vec[2] = xi * ijImageVecs[2] / iDist;
                if (i != j) {
                    gi[0] += vec[0]; gi[1] += vec[1]; gi[2] += vec[2];
                    pSPARC->d3Grads[j*3 + 0] -= vec[0]; pSPARC->d3Grads[j*3 + 1] -= vec[1]; pSPARC->d3Grads[j*3 + 2] -= vec[2]; 
                }
                for (row = 0; row < 3; row++) {
                    for (col = 0; col < 3; col++) {
                        pSPARC->d3Sigma[row*3 + col] += vec[row]*ijImageVecs[col];
                    }
                }
            }
        }
        pSPARC->d3Grads[i*3 + 0] += gi[0]; pSPARC->d3Grads[i*3 + 1] += gi[1]; pSPARC->d3Grads[i*3 + 2] += gi[2];
    }
    
    pSPARC->Etot += e; // add total d3 energy onto total energy
    #ifdef DEBUG
    if (rank == 0) {
        printf("D3 force =\n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", -pSPARC->d3Grads[3*i + 0], -pSPARC->d3Grads[3*i + 1], -pSPARC->d3Grads[3*i + 2]);
        }
    }
    #endif

    // at last, transfer cartesian coordinates back to non-cartesian coordinates
    for (atm = 0; atm < pSPARC->n_atom; atm++) {
        Cart2nonCart_coord(pSPARC, &pSPARC->atom_pos[atm*3], &pSPARC->atom_pos[atm*3 + 1], &pSPARC->atom_pos[atm*3 + 2]);
    }

    free(selfAtomC6);

    free(sqrtC6matrix);
    free(imageVecEG);
    for (imageIndex = 0; imageIndex < imageNumberEG; imageIndex++) {
        for (i = 0; i < natom; i++) {
            free(*(*(drij + imageIndex) + i));
        }
        free(*(drij + imageIndex));
    }
    free(drij);
    free(dC6ij);
    free(FdivR6mulDC6);
    free(imageVecCN);
    free(imageIndexCN);
}

double d3_dAng(double abDist2, double bcDist2, double acDist2, double tDenomin) {
    double dAng = -0.375*(abDist2*abDist2*abDist2 + abDist2*abDist2*(bcDist2 + acDist2)
        + abDist2*(3.0*bcDist2*bcDist2 + 2.0*bcDist2*acDist2 + 3.0*acDist2*acDist2)
        - 5.0*(bcDist2 - acDist2)*(bcDist2 - acDist2)*(bcDist2 + acDist2))
        / (sqrt(abDist2)*pow(tDenomin, 2.5));
    return dAng;
}

int d3_count_image(int imageX, int imageY, int imageZ, int *nImage) {
    int index = (imageX + nImage[0])*(2*nImage[1] + 1)*(2*nImage[2] + 1) + (imageY + nImage[1])*(2*nImage[2] + 1) + (imageZ + nImage[2]);
    return index;
}

void d3_grad_cell_stress(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double str[9];
    double detLattice = 0.0;
    double invLattice[9];
    int i, j, k, row;
    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
            for(k = 0; k < 3; k++){
                if(i != j && j != k && k != i)
                    detLattice += ((i - j) * (j - k) * (k - i)/2) * pSPARC->lattice[3 * i] * pSPARC->lattice[3 * j + 1] * pSPARC->lattice[3 * k + 2];
            }
        }
    }
    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
           invLattice[3*j + i] = (pSPARC->lattice[3 * ((j+1) % 3) + (i+1) % 3] * pSPARC->lattice[3 * ((j+2) % 3) + (i+2) % 3] - pSPARC->lattice[3 * ((j+1) % 3) + (i+2) % 3] * pSPARC->lattice[3 * ((j+2) % 3) + (i+1) % 3])/detLattice;
        }
    }

    str[0] = -pSPARC->d3Sigma[0]*invLattice[0] - pSPARC->d3Sigma[1]*invLattice[1] - pSPARC->d3Sigma[2]*invLattice[2]; // 6642-6648 in dftd3.f
    str[1] = -pSPARC->d3Sigma[0]*invLattice[3] - pSPARC->d3Sigma[1]*invLattice[4] - pSPARC->d3Sigma[2]*invLattice[5];
    str[2] = -pSPARC->d3Sigma[0]*invLattice[6] - pSPARC->d3Sigma[1]*invLattice[7] - pSPARC->d3Sigma[2]*invLattice[8];
    str[3] = -pSPARC->d3Sigma[3]*invLattice[0] - pSPARC->d3Sigma[4]*invLattice[1] - pSPARC->d3Sigma[5]*invLattice[2];
    str[4] = -pSPARC->d3Sigma[3]*invLattice[3] - pSPARC->d3Sigma[4]*invLattice[4] - pSPARC->d3Sigma[5]*invLattice[5];
    str[5] = -pSPARC->d3Sigma[3]*invLattice[6] - pSPARC->d3Sigma[4]*invLattice[7] - pSPARC->d3Sigma[5]*invLattice[8];
    str[6] = -pSPARC->d3Sigma[6]*invLattice[0] - pSPARC->d3Sigma[7]*invLattice[1] - pSPARC->d3Sigma[8]*invLattice[2];
    str[7] = -pSPARC->d3Sigma[6]*invLattice[3] - pSPARC->d3Sigma[7]*invLattice[4] - pSPARC->d3Sigma[8]*invLattice[5];
    str[8] = -pSPARC->d3Sigma[6]*invLattice[6] - pSPARC->d3Sigma[7]*invLattice[7] - pSPARC->d3Sigma[8]*invLattice[8];

    double d3CellGrad[9]; // compare this variable with the output file of dftd3.f: dftd3_cellgradient
    d3CellGrad[0] = str[0]; d3CellGrad[1] = str[3]; d3CellGrad[2] = str[6]; 
    d3CellGrad[3] = str[1]; d3CellGrad[4] = str[4]; d3CellGrad[5] = str[7]; 
    d3CellGrad[6] = str[2]; d3CellGrad[7] = str[5]; d3CellGrad[8] = str[8]; 

    #ifdef DEBUG
    if (rank == 0) {
        printf("the gradient of D3 energy regarding to (lattice vectors * side length) tensor of cell (Ha/Bohr):\n");
        for (row = 0; row < 3; row++) {
            printf("%18.14f %18.14f %18.14f\n", d3CellGrad[3*row + 0], d3CellGrad[3*row + 1], d3CellGrad[3*row + 2]);
        }
    }
    #endif
    // another reference from QE: qe-6.7>dft-d3>dftd3_qe.f, 92-93
    for (row = 0; row < 3; row++) {
        pSPARC->d3Stress[row*3 + 0] = 0.0; pSPARC->d3Stress[row*3 + 1] = 0.0; pSPARC->d3Stress[row*3 + 2] = 0.0;
    }
    for (row = 0; row < 3; row++) {
        pSPARC->d3Stress[row*3 + 0] = -pSPARC->d3Sigma[row*3 + 0]/detLattice; 
        pSPARC->d3Stress[row*3 + 1] = -pSPARC->d3Sigma[row*3 + 1]/detLattice; 
        pSPARC->d3Stress[row*3 + 2] = -pSPARC->d3Sigma[row*3 + 2]/detLattice;
    }
    #ifdef DEBUG
    if (rank == 0) {
        printf("\nthe DFT-D3 stresses (GPa):\n");
        for (row = 0; row < 3; row++) {
            printf("%18.14f %18.14f %18.14f\n", 
                pSPARC->d3Stress[3*row + 0]*29421.02648438959, pSPARC->d3Stress[3*row + 1]*29421.02648438959, pSPARC->d3Stress[3*row + 2]*29421.02648438959);
        }
    }
    #endif
    if (pSPARC->Calc_stress == 1) {
        pSPARC->stress[0] += pSPARC->d3Stress[0];
        pSPARC->stress[3] += pSPARC->d3Stress[4];
        pSPARC->stress[5] += pSPARC->d3Stress[8];
        pSPARC->stress[1] += pSPARC->d3Stress[1];
        pSPARC->stress[2] += pSPARC->d3Stress[2];
        pSPARC->stress[4] += pSPARC->d3Stress[5];
        pSPARC->pres -= (pSPARC->d3Stress[0] + pSPARC->d3Stress[4] + pSPARC->d3Stress[8])/3;
    }
    else if (pSPARC->Calc_pres == 1) {
        pSPARC->pres -= (pSPARC->d3Stress[0] + pSPARC->d3Stress[4] + pSPARC->d3Stress[8])/3;
    }
    
}

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

void add_d3_forces(SPARC_OBJ *pSPARC) {
    for (int i = 0; i < pSPARC->n_atom; i++) { // force is -grad
        pSPARC->forces[3*i + 0] -= pSPARC->d3Grads[3*i + 0]; pSPARC->forces[3*i + 1] -= pSPARC->d3Grads[3*i + 1]; pSPARC->forces[3*i + 2] -= pSPARC->d3Grads[3*i + 2];
    } // add d3 forces onto atom forces
}