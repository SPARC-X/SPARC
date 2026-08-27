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
#include <stdbool.h>
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
void Print_fullMD(SPARC_OBJ *pSPARC, FILE *output_md, double *avgvel, double *maxvel, double *mindis); // avgvel/maxvel needed for DEBUG-mode printout

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

/*
 @brief   Write the re-initialized parameters into the output file.
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
@ brief: Performs Molecular Dynamics using NPT_NP or NPH.
*/
void NPT_NP_and_NPH(SPARC_OBJ *pSPARC,  double *avgvel, double *maxvel, double *mindis);

/*
@ brief: Do one full step of 'NPT_NP' or 'NPH' ensemble i.e. Solve equation 18a to 18g of the Hernandez paper and calculate the Hamiltonian of the same system.
*/
void NPT_NPH_main(SPARC_OBJ *pSPARC, double *avgvel, double *maxvel, double *mindis);

/*
@ brief: function to calculate the initial hamiltonian in 'NPT_NP' or 'NPH'.
*/
void NPT_NP_and_NPH_init_hamiltonian(SPARC_OBJ *pSPARC);

/* 
@ brief: Initializes cell ingredients during the start of 'NPT_NP' or 'NPH', or updates the cell after each step.
         Cell ingredients include but not limited to: cell angles, reciprocal lattice vectors, metric tensor and reciprocal metric tensors, for use in NPT_NP and NPH dynamics
*/
void fetch_MD_cell_ingredients(SPARC_OBJ *pSPARC, bool update_cell);

/*
@ brief: Initializes cell ingredients during the restart of 'NPT_NP' or 'NPH'. To be called only once in the first step after restart.
*/
void fetch_MD_cell_ingredients_restart(SPARC_OBJ *pSPARC);

/*
@ brief: function to transpose a matrix and add to itself 
*/
void transpose_and_add(double *matrix1);

/*
@ brief: function to compute the kinetic energy of the ionic particles in 'NPT_NP' or 'NPH'
*/
void Calculate_Ionic_particles_Kinetic_energy(SPARC_OBJ *pSPARC);

/*
@ brief: function to compute the kinetic stress of the ionic particles in 'NPT_NP' or 'NPH'
*/
void Calculate_Kinetic_stress(SPARC_OBJ *pSPARC);

/*
@brief: function to impose constraints on the flexibility of the cell in the 'NPT_NP' or 'NPH'
*/
void compute_constraint_stress(SPARC_OBJ *pSPARC, int NPT_NPHconstraintFlag, int *NPT_NPHscaleVecs);

/*
@brief: function to update the components of the metric tensor in one full step iteratively in 'NPT_NP' or 'NPH'.
*/
void Update_metric_tensor_components_iteratively_full_step(SPARC_OBJ *pSPARC, double S_new, int NPT_NPH_ANGLES, int NPT_NPHconstraintFlag);

/*
@brief: function to update the momentum of the metric tensor in one half step iteratively in 'NPT_NP' or 'NPH'.
*/
void Update_metric_tensor_momenta_iteratively_half_step(SPARC_OBJ *pSPARC);

/**
 * @ brief: function to convert non cartesian to cartesian coordinates and velocities, from initialization.c
 */
void nonCart2Cart(double *LatUVec, double *carCoord, double *nonCarCoord);
//void Cart2nonCart_transformMat_MD(SPARC_OBJ *pSPARC);

/*
 * @brief: function to convert cartesian to non cartesian coordinates and velocities, from initialization.c
 */
void Cart2nonCart(double *gradT, double *carCoord, double *nonCarCoord);

#endif // MD_H
