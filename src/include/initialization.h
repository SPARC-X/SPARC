/**
 * @file    initialization.h
 * @brief   This file contains the function declarations for initialization.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef INITIALIZATION_H
#define INITIALIZATION_H 

#include "isddft.h"


/**
 * @brief   Initializes parameters and objects.
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure SPARC.
 * @param argc  The number of arguments that are passed to the program.
 * @param argv  The array of strings representing command line arguments.
 */
void Initialize(SPARC_OBJ *pSPARC, int argc, char *argv[]);


/**
 * @brief   Prints usage of SPARC through command line.
 */
void print_usage();


/**
 * @brief   Create MPI struct type SPARC_INPUT_MPI.
 *
 *          This function creates a user-defined new MPI datatype, which is a
 *          structure type. By defining this new MPI datatype, one can send a
 *          collection of variables packed in this struct by just one MPI_Send
 *          call. 
 *
 * @param pSPARC_INPUT_MPI (output)    The pointer to the new MPI datatype.
 */
void SPARC_Input_MPI_create(MPI_Datatype *pSPARC_INPUT_MPI); 


/**
 * @brief   Check input arguments and read input filename.
 *
 * @param pSPARC_Input  The pointer that points to SPARC_INPUT_OBJ type structure.
 * @param argc          The number of arguments that are passed to the program.
 * @param argv          The array of strings representing command line arguments.
 */
void check_inputs(SPARC_INPUT_OBJ *pSPARC_Input, int argc, char *argv[]);


/**
 * @brief   Set default values for SPARC members.
 *
 * @param pSPARC_Input  The pointer that points to SPARC_INPUT_OBJ type structure.
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure.
 */
void set_defaults(SPARC_INPUT_OBJ *pSPARC_Input, SPARC_OBJ *pSPARC);


/**
 * @brief Broadcast atom info. in SPARC using MPI_Pack & MPI_Unpack.
 */
void bcast_SPARC_Atom(SPARC_OBJ *pSPARC);


/**
 * @brief   Copy the data read from input files into struct SPARC.
 *
 * @param pSPARC        The pointer that points to SPARC_OBJ type structure SPARC.
 * @param pSPARC_Input  The pointer that points to SPARC_Input_OBJ type structure.
 */
void SPARC_copy_input(SPARC_OBJ *pSPARC, SPARC_INPUT_OBJ *pSPARC_Input);


/**
 * @brief   Call Spline to calculate derivatives of the tabulated functions and 
 *          store them for later use (during interpolation).
 */
void Calculate_SplineDerivRadFun(SPARC_OBJ *pSPARC);


/** 
@ brief: function to calculate the det(Jacobian) for integration and transformation matrices for distance, gradient, and laplacian in a non cartesian coordinate system 
**/

void Cart2nonCart_transformMat(SPARC_OBJ *pSPARC);

/**
 * @brief   Write the initialized parameters into the output file.
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure.
 */
 
/*
 * @brief   function to convert non cartesian to cartesian coordinates
 */

void nonCart2Cart_coord(const SPARC_OBJ *pSPARC, double *x, double *y, double *z);

/*
 * @brief   function to convert cartesian to non cartesian coordinates
 */

void Cart2nonCart_coord(const SPARC_OBJ *pSPARC, double *x, double *y, double *z);

/*
 * @brief   function to convert gradients along lattice directions to cartesian gradients
 */

void nonCart2Cart_grad(SPARC_OBJ *pSPARC, double *x, double *y, double *z);



/*
@brief: function to calculate the distance btween two points
*/

void CalculateDistance(SPARC_OBJ *pSPARC, double x, double y, double z, double xref, double yref, double zref, double *d);


/**
 * @brief   Write the initialized parameters into the output file.
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure.
 */
void write_output_init(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate k-points and the associated weights.
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure SPARC.
 */
void Calculate_kpoints(SPARC_OBJ *pSPARC);

void Calculate_local_kpoints(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate the weight of a given k-point..
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure SPARC.
 */
double kpointWeight(double kx,double ky,double kz);

/**
 * @brief   Estimate the memory required for the simulation.
 */
double estimate_memory(const SPARC_OBJ *pSPARC);

/**
 * @brief   Find equivalent mesh size to a given Ecut.
 *
 * @param Ecut  Energy cutoff used in plane-wave codes, in Hartree.
 * @param FDn   Finite difference order divided by 2.
 */
double Ecut2h(double Ecut, int FDn);

/**
 * @brief   Computing nearest neighbohr distance
 */
double compute_nearest_neighbor_dist(SPARC_OBJ *pSPARC, char CorN);

/**
 * @brief   Simple linear model for selecting maximum number of 
 *          processors for eigenvalue solver. 
 */
int parallel_eigensolver_max_processor(int N, char RorC, char SorG);
#endif // INITIALIZATION_H 




