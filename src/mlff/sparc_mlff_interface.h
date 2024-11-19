#ifndef SPARC_MLFF_INTERFACE_H
#define SPARC_MLFF_INTERFACE_H

#include "isddft.h"
#include "mlff_types.h"

/**
 * @brief      Governing function that calls appropriate subroutines based on the MLFF flag and state of simulation
 */

void MLFF_call(SPARC_OBJ* pSPARC);

/**
 * @brief      Initializes the MLFF structure. 
 */
// Updated
void init_MLFF_structure(SPARC_OBJ *pSPARC);

/**
 * @brief      Initializes a new MLFF simulation depending on the flag. 
 */
// New
void init_MLFF_simulation(SPARC_OBJ *pSPARC);

/**
 * @brief      Initializes a new MLFF simulation from scratch. 
 */

void init_new_MLFF(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);

/**
 * @brief      Initializes a new MLFF simulation from a known model that can be updated. 
 */

void init_existing_trainable_MLFF(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);

/**
 * @brief      Initializes a new MLFF simulation from a known model that remains static. 
 */

void init_static_MLFF(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);

/**
 * @brief      Frees memory allocated for MLFF structure.  
 */
// Updated
void free_MLFF(MLFF_Obj *mlff_str);

  /**
 * @brief      Constructs design matrices from saved reference structures and reference atomic descriptors.
 */

void pretrain_MLFF_model(
  MLFF_Obj *mlff_str,
  SPARC_OBJ *pSPARC, 
  double **cell_data, 
  double **LatUVec_data,
  double **apos_data, 
  double *Etot_data, 
  double **F_data, 
  double **stress_data, 
  int *natom_data, 
  int **natom_elem_data);

  /**
 * @brief      Gets the number of atoms on each processor
 */

  void get_domain_decompose_mlff_natom(int natom,
  int nelem,
  int *nAtomv,
  int nprocs,
  int rank,
  int *natom_domain);

  /**
 * @brief      Returns the atom and element indices for parallelization in MLFF
 */

void get_domain_decompose_mlff_idx(
  int natom,
  int nelem,
  int *nAtomv,
  int nprocs,
  int rank,
  int natom_domain,
  int *atom_idx_domain,
  int *el_idx_domain);

/**
 * @brief      Returns predictions from MLFF or DFT depending on size of training set or UQ. Updates MLFF model as necessary. 
 */

void MLFF_train_predict(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);

/** 
 * @brief Performs MLFF calculation without UQ for DFT check
*/

void MLFF_only_predict(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);

/**
 * @brief      This function is used to add the first MD step to the MLFF training set
 */

void sparc_mlff_interface_initialDFT(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);

/**
 * @brief Add DFT data to the MLFF model
 */

void sparc_mlff_interface_addDFT(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);

/**
 * @brief      Predict energies, forces, stresses and Bayesian error from MLFF
 */

void sparc_mlff_interface_predict(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str, double *E_predict, double *F_predict, double *stress_predict, double *bayesian_error);

/**
 * @brief      Write the MLFF results to the output and static files
 */

void write_MLFF_results(SPARC_OBJ *pSPARC);

/**
 * @brief      Reshape the stress tensor based on the cell type and boundary conditions for MLFF training
 */

void reshape_stress(int cell_typ, int *BC, int *index);

#endif 