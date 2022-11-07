/**
 * @file    d3forceStress.c
 * @brief   This file contains the functions for adding forces and stresses from DFT-D3 into total forces and stresses.
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

#include "d3forceStress.h"

void d3_grad_cell_stress(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double detLattice = 0.0;
    int i, j, k, row;
    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
            for(k = 0; k < 3; k++){
                if(i != j && j != k && k != i)
                    detLattice += ((i - j) * (j - k) * (k - i)/2) * pSPARC->lattice[3 * i] * pSPARC->lattice[3 * j + 1] * pSPARC->lattice[3 * k + 2];
            }
        }
    }

    #ifdef DEBUG
    double invLattice[9];
    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
           invLattice[3*j + i] = (pSPARC->lattice[3 * ((j+1) % 3) + (i+1) % 3] * pSPARC->lattice[3 * ((j+2) % 3) + (i+2) % 3] - pSPARC->lattice[3 * ((j+1) % 3) + (i+2) % 3] * pSPARC->lattice[3 * ((j+2) % 3) + (i+1) % 3])/detLattice;
        }
    }

    double str[9];
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
        printf("\nthe DFT-D3 stresses, which is a part of XC stress contribution (GPa):\n");
        for (row = 0; row < 3; row++) {
            printf("%18.14f %18.14f %18.14f\n", 
                pSPARC->d3Stress[3*row + 0]*29421.02648438959, pSPARC->d3Stress[3*row + 1]*29421.02648438959, pSPARC->d3Stress[3*row + 2]*29421.02648438959);
        }
    }
    #endif
    if (pSPARC->Calc_stress == 1) {
        pSPARC->stress_xc[0] += pSPARC->d3Stress[0];
        pSPARC->stress_xc[3] += pSPARC->d3Stress[4];
        pSPARC->stress_xc[5] += pSPARC->d3Stress[8];
        pSPARC->stress_xc[1] += pSPARC->d3Stress[1];
        pSPARC->stress_xc[2] += pSPARC->d3Stress[2];
        pSPARC->stress_xc[4] += pSPARC->d3Stress[5];
        // pSPARC->pres -= (pSPARC->d3Stress[0] + pSPARC->d3Stress[4] + pSPARC->d3Stress[8])/3;
    }
    else if (pSPARC->Calc_pres == 1) {
        pSPARC->pres_xc += (pSPARC->d3Stress[0] + pSPARC->d3Stress[4] + pSPARC->d3Stress[8]) * detLattice;
    }
    if (!rank) printf("detLattice %9.6E, cell measure %9.6E\n", detLattice, pSPARC->Jacbdet*pSPARC->range_x*pSPARC->range_y*pSPARC->range_z);
}

void add_d3_forces(SPARC_OBJ *pSPARC) {
    for (int i = 0; i < pSPARC->n_atom; i++) { // force is -grad
        pSPARC->forces[3*i + 0] -= pSPARC->d3Grads[3*i + 0]; pSPARC->forces[3*i + 1] -= pSPARC->d3Grads[3*i + 1]; pSPARC->forces[3*i + 2] -= pSPARC->d3Grads[3*i + 2];
    } // add d3 forces onto atom forces
}