#ifndef REGRESSION_H
#define REGRESSION_H

#include "mlff_types.h"


/*
Sparsification of columns in the training dataset

Input:
1. mlff_str: MLFF structure

Output:
1. mlff_str: MLFF structure
*/
void CUR_sparsify_before_training(MLFF_Obj *mlff_str);



/*
Bayesian linear regression to find the weights

Input:
1. mlff_str: MLFF structure

Output:
1. mlff_str: MLFF structure
*/
void mlff_train_Bayesian(MLFF_Obj *mlff_str);



/*
Hyperparameter optimization (noise parameter and the prior on w)

Input:
1. btb_reduced: b is the vector containing energy, forces and stresses
2. AtA: A is the covariance matrix (K_train)
3. M_total_rows: total number of rows (including energym forces and stresses from all structures in the training)
4. condK_min: Minimum conition number of K'K allowed after regularization

Output:
1. mlff_str: MLFF structure
*/
void hyperparameter_Bayesian(double btb_reduced, double *AtA, double *Atb, MLFF_Obj *mlff_str, int M, double condK_min);


/*
Finding regularization constant to satisfy the condK_min

Input:
1. A: The K_train matrix
2. size: size of K'K matrix
3. condK_min: Minimum conition number of K'K allowed after regularization

Output:
1. reg_final: Regularization constant
*/
double get_regularization_min(double *A, int size, double condK_min);


/*
Calculate Energy, force, stress/pressure for prediction

Input:
1. K_predict: The K matrix or prediction structure
2. mlff_str: MLFF structure
3. condK_min: Minimum conition number of K'K allowed after regularization

Output:
1. E, F, stress, error_bayesian: Outputs
*/
void mlff_predict(double *K_predict, MLFF_Obj *mlff_str, double *E,  double* F, double* stress, double* error_bayesian, int natoms );
#endif
