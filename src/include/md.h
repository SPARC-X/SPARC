/**
 * @file    md.h
 * @brief   This file declares the functions for performing molecular dynamics.
 *
 * @authors Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef MD_H
#define MD_H

#include "isddft.h"

/**
* @ brief Main function of molecular dynamics
*/
void main_MD(SPARC_OBJ *pSPARC);

/**
* @ brief: function to initialize velocities and accelerations for Molecular Dynamics (MD)
 **/
void Initialize_MD(SPARC_OBJ *pSPARC);

/**
 * @brief   Performs Molecular Dynamics using NVE.
 */
void NVE(SPARC_OBJ *pSPARC);

/**
 * @brief   Performs Molecular Dynamics using NVT.
 */
void NVT_NH(SPARC_OBJ *pSPARC);

/*
@ brief: function to perform first step of velocity verlet algorithm
*/
void VVerlet1(SPARC_OBJ* pSPARC);

/*
@ brief: function to perform second step of velocity verlet algorithm
*/
void VVerlet2(SPARC_OBJ* pSPARC);

/*
* @ brief: function to update position of atoms using Leapfrog method
*/
void Leapfrog_part1(SPARC_OBJ *pSPARC);

/*
* @ brief: function to update velocity of atoms using Leapfrog method
*/
void Leapfrog_part2(SPARC_OBJ *pSPARC);

/**
  @ brief:  Perform molecular dynamics keeping number of particles, volume of the cell and kinetic energy constant i.e. NVK with Gaussian thermostat. 
            It is based on the implementation in ABINIT (ionmov=12)
 **/
void NVK_G(SPARC_OBJ *pSPARC);

/*
 @ brief: calculate velocity at next half time step for isokinetic ensemble with Gaussian thermostat
*/

void Calc_vel1_G(SPARC_OBJ *pSPARC);


/*
 @ brief: calculate velocity at next full time step for isokinetic ensemble with Gaussian thermostat
*/

void Calc_vel2_G(SPARC_OBJ *pSPARC); 

/*
* @ brief: function to wrap around atom positions that lie outside main domain in case of PBC and check if the atoms are too close to the boundary in case of bounded domain
*/
void Check_atomlocation(SPARC_OBJ *pSPARC);

/*
 @ brief: function to write all relevant DFT quantities generated during MD simulation
*/
void Print_fullMD(SPARC_OBJ *pSPARC, FILE *output_md, double *avgvel, double *maxvel, double *mindis);

/* 
 @ brief function to evaluate the qunatities of interest in a MD simulation
*/
void MD_QOI(SPARC_OBJ *pSPARC, double *avgvel, double *maxvel, double *mindis); 

/*
 @ brief: function to write all relevant quantities needed for MD restart
*/
void PrintMD(SPARC_OBJ *pSPARC, int Flag, int print_restart_typ);

/*
@ brief function to read the restart file for MD restart
*/
void RestartMD(SPARC_OBJ *pSPARC);

/* 
@ brief: function to rename the restart file 
*/
void Rename_restart(SPARC_OBJ *pSPARC);

#endif // MD_H
