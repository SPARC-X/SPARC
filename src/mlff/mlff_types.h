#ifndef MLFF_TYPES_H
#define MLFF_TYPES_H

#include "ddbp_types.h"
#include <stdio.h>


#define L_STRING 512
// typedef int value_type;
// /*
//  * @brief  Data type for dynamic array.
//  */
// typedef struct {
//     value_type *array;
//     size_t len;  // length of the array (used)
//     size_t capacity; // total capacity of memory available
// } dyArray;


typedef struct NeighList NeighList;
typedef struct SoapObj SoapObj;
typedef struct DescriptorObj DescriptorObj;
typedef struct MLFF_Obj MLFF_Obj;
typedef struct GMPObj GMPObj;

/**
 * @brief   This structure type is for Neighbor list
 */
typedef struct NeighList
{
  // # of atoms 
  int natom;
  int nelem;
  int cell_typ;
  int BC[3];
  int* natom_elem;  
  double cell_len[3];
  double LatUVec[9];
  double twist;                     // System geometry

  int natom_domain;
  int *atom_idx_domain;
  int *el_idx_domain;               // parallelization

  double rcut;                      // cutoff radius

  int* Nneighbors;                  // # of neighbour in the fundamental cell for each atom (excludes neighbours in periodic images)
  int* unique_Nneighbors;           // # of neighbours of a given element speciy for each atoms
  int** Nneighbors_elemWise;        // # of neigbours of a given element speciy in the fundamental cell for each atom (excludes neighbours in periodic images) 
  int** unique_Nneighbors_elemWise; // list of neighbours of all atoms
  int *if_self_image;               // If the neighbor is a self-image of the atoms in unit cell

  dyArray* neighborList;            // list of neighbours of a given element speciy
  dyArray** neighborList_elemWise;  // list of unique neighbours of all atoms
  dyArray* unique_neighborList;     // list of unique neighbous of a given speciy
  dyArray** unique_neighborList_elemWise;   // list of the image ID in x-direction of neighbours for all atoms

  dyArray* neighborList_imgX;  
  dyArray* neighborList_imgY;
  dyArray* neighborList_imgZ;        // list of the image ID in x,y,z-direction of neighbours of a given element speciy for all atoms

  dyArray** neighborList_elemWise_imgX;
  dyArray** neighborList_elemWise_imgY;  
  dyArray** neighborList_elemWise_imgZ; // list of the image ID in x,y,z-direction of neighbours of a given element speciy for all atoms

  
  dyArray* localID_neighbours; // stores an ID of all the neighbours of the atoms. [IDs are integers ranging from 0 to N-1 whre N is number of unique neighbours of the concerning atom]
  dyArray** localID_neighbours_elem;  // list atom type of neighbours of a given atom
  dyArray* neighborAtmTyp;  // type of the neighbor
} NeighList;


typedef struct DescriptorObj
{
   
  int natom;  
  int nelem;
  int* natom_elem;  // geometry and system info
  
  double rcut; // cutoff distance (Bohr)
  
  //double cell_len[3];
  double cell_measure; // dimensions of the fundamental unit cell (Bohr)

  int natom_domain;
  int *atom_idx_domain;
  int *el_idx_domain;  // paralleization over atoms

  int *if_self_image;  // number of self images in the nlist 
  
  int descriptor_typ;  // Descriptor type (Will be removed)

  // SOAP variables
  int Lmax; 
  int Nmax; // SOAP basis function size

  int size_X3;  // size of descriptors
  double xi_3;   // exponent in the dot product kernel of SOAP

  int N_rgrid; // # number of grids points used in the spline interpolation of h_nl

  int* Nneighbors;
  // array of number of unqiue neighbours for all the atoms
  int* unique_Nneighbors;
  dyArray* unique_neighborList;
  int** unique_Nneighbors_elemWise;

  double **X3; // array of 2, 3-body descriptor SOAP, X3[natom][size_X2,3]


  double ***dX3_dF; // array of 2,3-body descriptor derivatives for x-dir force, dX2,3_dX[natom][1+neigh][size_X2,3]
  

  double ***dX3_dX;
  double ***dX3_dY;
  double ***dX3_dZ;  // Derivative of descriptors in x,y and z-directions

  // size of GMP descriptor
  int size_X;
  // size of the 2-body descriptor
  
  
} DescriptorObj;


typedef struct MLFF_Obj
{

  int natom_domain;
  int nelem;
  int natom;
  int *atom_idx_domain;
  int *el_idx_domain;
  int *Znucl;
  int *natm_train_elemwise;   // System geometry


  double condK_min;
  int if_sparsify_before_train;
  double **K_train;
  double *b_no_norm;    // Covariance matrix anf E,F,sigma vectors
  double *AtA_SVD_U;
  double *AtA_SVD_Vt;
  double *AtA_SingVal;
  double *AtA;          
  double *weights;     
  double *cov_train;
  dyArray atom_idx_addtrain;   // matrices required during BLR
  int print_mlff_flag;
  
  
     
  int descriptor_typ;  // type of descriptor


  int natm_train_total;
  int *natm_typ_train;
  int n_str;
  int n_str_max;
  int n_train_max; 
  int size_X3;   
  double xi_3;
  double rcut;  
  double sigma_atom;// size of local environment and reference structures

  int Nmax;   
  int Lmax;
  int kernel_typ;   // Kernel and descriptor parameters

  
  double F_tol;    // tolerance of force error to decide to skip DFT

  int n_rows;
  int n_cols;    // rows and cols in the K_train

  
  int N_rgrid;   // number of radial points in hnl

  double error_train_E;
  double error_train_F;
  double *error_train_stress;  // Training errors

  

  int *E_row_idx;
  
  
  double mu_E;
  double std_E;
  double std_F;
  double *std_stress;   // mean and std dev of prediction variables
  
  double sigma_w;
  double sigma_v;       // noise and weight prior hyperparameters in the regression
  
  double E_scale;
  double F_scale;
  double *stress_scale;
  double relative_scale_F;
  double *relative_scale_stress;  // scaling to be applied on the design and prediction matrices
  
  double *F_store;
  double *E_store;
  double **stress_store;   // All energy, forces, stresses of DFT run stored



  int E_store_counter;
  int F_store_counter;   // No of energy, forces, stresses of DFT run stored


  double *internal_energy_DFT;
  double *free_energy_DFT;
  double internal_energy_model_weights[2];
  double internal_energy_model_R2;
  int internal_energy_DFT_count;
  int mlff_internal_energy_flag;
  int mlff_pressure_train_flag;
 
  double **X3_traindataset; // stores 3-body SOAP descriptor for all local descriptors in the training dataset
  // stores SOAP descriptors and other related variables for all reference structures in the training dataset

  // ********** Memory Bottleneck *****************
  DescriptorObj *descriptor_strdataset;
  // precalculated values of h_nl and its derivative on a sparse grid. Later used in spline interpolation.
  double* rgrid;
  double* h_nl;
  double* dh_nl;
  char ref_atom_name[L_STRING];
  char ref_str_name[L_STRING];
  char restart_name[L_STRING];
  int stress_len;
  int mlff_flag;
  FILE *fp_mlff;

  // GMP variables
  int **params_i;
  double **params_d;
  double **atom_gaussians_p;
  double *sigmas;
  int *ngaussians;
  int *element_index_to_order_p;
  int *atom_type_to_indices_p;
  int *atom_indices_p;
} MLFF_Obj;

#endif 
