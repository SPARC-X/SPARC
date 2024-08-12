#ifndef COVARIANCE_MATRIX_H
#define COVARIANCE_MATRIX_H

#include "mlff_types.h"
#include "isddft.h"


/*
Calculates the kernel k(xi, xj) for a given structure for the polynomial kernel

Input:
1. desc_str: Descriptor of the structure
2. X3_traindataset: The training columns
3. natm_typ_train: Atom type of cols in X3_traindataset
4. n_cols: number of columns

Output:
1. kernel_ij: the kernel between xi and xi (x1 \cdot xj)^(xi_3)
*/
void calculate_polynomial_kernel_ij(DescriptorObj *desc_str, double **X3_traindataset, int *natm_typ_train, int n_cols, double **kernel_ij);


/*
Calculates the zit required to calculate the derivatives of the kernel

Input:
1. desc_str: Descriptor of the structure
2. X3_traindataset: The training columns
3. natm_typ_train: Atom type of cols in X3_traindataset
4. n_cols: number of columns

Output:
1. z_it: zit term (Eq 11a in delta paper)
*/
void calculate_zit(DescriptorObj *desc_str, double **X3_traindataset, int *natm_typ_train, int n_cols, double ***zit);


/*
copy_descriptors function copies the content of one SoapObj to another

[Input]
1. desc_str: DescriptorObj structure to be copied
[Output]
1. desc_str_MLFF: DescriptorObj structure where it needs to be copied
*/
void copy_descriptors(DescriptorObj *desc_str_MLFF, DescriptorObj *desc_str);


/*
add_firstMD function updates the MLFF_Obj by updating design matrix, b vector etc. for the first MD

[Input]
1. desc_str: DescriptorObj structure of the first MD
2. nlist: NeighList strcuture of the first MD
3. mlff_str: MLFF_Obj structure
4. E: energy per atom of first MD structure (Ha/atom)
5. F: atomic foces of first MD structure (Ha/bohr) [ColMajor]
6. stress: stress of first MD structure (GPa)
[Output]
1. mlff_str: MLFF_Obj structure
*/
void add_firstMD(DescriptorObj *desc_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double* F, double* stress_sparc);


/*
add_newstr_rows function updates the MLFF_Obj by updating design matrix, b vector etc. for a new reference structure

[Input]
1. desc_str: DescriptorObj structure of the reference structure to be added
2. nlist: NeighList strcuture of the first MD
3. mlff_str: MLFF_Obj structure
4. E: energy per atom of the reference structure to be added (Ha/atom)
5. F: atomic foces of the reference structure to be added (Ha/bohr) [ColMajor]
6. stress: stress of the reference structure to be added (GPa)
[Output]
1. mlff_str: MLFF_Obj structure
*/
void add_newstr_rows(DescriptorObj *desc_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double *F, double* stress_sparc);


/*
calculate_Kpredict function calculate the design matrix for prediction for a new structure

[Input]
1. desc_str: DescriptorObj structure of the new structure
2. nlist: NeighList strcuture of the new structure
3. mlff_str: MLFF_Obj structure
[Output]
1. K_predict: design prediction matrix
*/
void calculate_Kpredict(DescriptorObj *desc_str, NeighList *nlist, MLFF_Obj *mlff_str, double **K_predict);


/*
add_newtrain_cols function updates the MLFF_Obj by updating design matrix columns etc. for a new local confiugration

[Input]
1. mlff_str: MLFF_Obj structure
2. X2: 2-body ddescriptor of the new local confiugration
3. X3: 3-body ddescriptor of the new local confiugration
4. elem_typ: Element type of the new local confiugration
[Output]
1. mlff_str: MLFF_Obj structure
*/
void add_newtrain_cols(double *X3, int elem_typ, MLFF_Obj *mlff_str);


/*
remove_str_rows function removes a given reference structure from the training dataset

[Input]
1. mlff_str: MLFF_Obj structure
2. str_ID: ID of the reference structure in training dataset
[Output]
1. mlff_str: MLFF_Obj structure
*/
void remove_str_rows(MLFF_Obj *mlff_str, int str_ID);


/*
remove_train_cols function removes a given local confiugration from the training dataset

[Input]
1. mlff_str: MLFF_Obj structure
2. col_ID: ID of the local confiugration in training dataset
[Output]
1. mlff_str: MLFF_Obj structure
*/
void remove_train_cols(MLFF_Obj *mlff_str, int col_ID);

/*
get_N_r_hnl function calculates the number of grid points in hnl_file

[Input]
1. pSPARC: SPARC structure
[Output]
1. pSPARC: SPARC structure
*/
void get_N_r_hnl(SPARC_OBJ *pSPARC);

// double mlff_kernel(int kernel_typ, double **X2_str, double **X3_str, double **X2_tr, double **X3_tr, int atom, int atom_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

// double der_mlff_kernel(int kernel_typ, double ***dX2_str, double ***dX3_str, double **X2_str, double **X3_str, double **X2_tr, double **X3_tr, int atom, int atom_tr, int neighbor,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

// double soap_kernel_polynomial(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

// double soap_kernel_Gaussian(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

// double soap_kernel_Laplacian(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

// double der_soap_kernel_polynomial(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

// double der_soap_kernel_Gaussian(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

// double der_soap_kernel_Laplacian(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

// double GMP_kernel_polynomial(double *X2_str, double *X2_tr,
// 			  double xi_3, int size_X2);

// double GMP_kernel_Gaussian(double *X2_str, double *X2_tr,
// 			  double xi_3, int size_X2);

// double der_GMP_kernel_polynomial(double *dX2_str, double *X2_str, double *X2_tr,
// 			double xi_3, int size_X2);

// double der_GMP_kernel_Gaussian(double *dX2_str, double *X2_str, double *X2_tr,
// 			 double xi_3, int size_X2);
#endif
