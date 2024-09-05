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
* @ brief: function to check if the atoms are too close to the boundary in case of bounded domain or to each other in general
*/
void Check_atomlocation(SPARC_OBJ *pSPARC);

/*
* @ brief: function to wraparound atom positions for PBC
*/
void wraparound_dynamics(SPARC_OBJ *pSPARC, double *coord, int opt);

/*
* @ brief: function to wraparound velocities in MD and displacement vectors in relaxation for PBC
*/
void wraparound_velocity(SPARC_OBJ *pSPARC, double shift, int dir, int loc);

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

/* 
@ brief: Performs Molecular Dynamics using NPT_NH.
*/
void NPT_NH(SPARC_OBJ *pSPARC);

/* 
@ brief: Updating accelerations and velocities of thermostat and barostat variables.
*/
void IsoPress(SPARC_OBJ *pSPARC);

/* 
@ brief: Updating accelerations and velocities of particles in the first half step.
*/
void AccelVelocityParticle (SPARC_OBJ *pSPARC);

/* 
@ brief: Updating velocities of particles in the second half step.
*/
void VelocityParticle (SPARC_OBJ *pSPARC);

/* 
@ brief: Updating positions of particles, size of unit cell and position of thermostat variables.
*/
void PositionParticleCell(SPARC_OBJ *pSPARC);

/**
 * @brief   Write the re-initialized parameters into the output file.
 */
void write_output_reinit_NPT(SPARC_OBJ *pSPARC);

/* 
@ brief: reinitialize related variables after the size changing of cell.
*/
void reinitialize_mesh_NPT(SPARC_OBJ *pSPARC);

/* 
@ brief: calculate Hamiltonian of the NPT system.
*/
void hamiltonian_NPT_NH(SPARC_OBJ *pSPARC);

/* 
@ brief: Performs Molecular Dynamics using NPT_NP.
*/
void NPT_NP(SPARC_OBJ *pSPARC);

/* 
@ brief: calculate Hamiltonian of the NPT_NP system.
*/
void initialize_Hamiltonian(SPARC_OBJ *pSPARC);

/* 
@ brief: updating momentums of thermostat and barostat variables and particles in the first half step in NPT_NP.
*/
void updateMomentum_FirstHalf(SPARC_OBJ *pSPARC);

/* 
@ brief: updating momentums of thermostat and barostat variables and particles in the second half step in NPT_NP.
*/
void updateMomentum_SecondHalf(SPARC_OBJ *pSPARC);

/* 
@ brief: updating value of thermostat variable and length of cell and positions of particles in NPT_NP.
*/
void updatePosition(SPARC_OBJ *pSPARC);

/**
 * @ brief: function to convert non cartesian to cartesian coordinates and velocities, from initialization.c
 */
void nonCart2Cart(double *LatUVec, double *carCoord, double *nonCarCoord);

/**
 * @brief: function to convert cartesian to non cartesian coordinates and velocities, from initialization.c
 */
void Cart2nonCart(double *gradT, double *carCoord, double *nonCarCoord);

#endif // MD_H
