#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "tools_mlff.h"
#include "spherical_harmonics.h"
#include "soap_descriptor.h"
#include "mlff_types.h"
#include "covariance_matrix.h"
#include "sparsification.h"
#include "ddbp_tools.h"
#include "tools.h"
#include "regression.h"
#include "mlff_read_write.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


/*
Sparsification of columns in the training dataset

Input:
1. mlff_str: MLFF structure

Output:
1. mlff_str: MLFF structure
*/
void CUR_sparsify_before_training(MLFF_Obj *mlff_str){
	int count, count1;
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;

    FILE *fp_mlff;
    if (mlff_str->print_mlff_flag==1 && rank==0){
    	// fp_mlff = fopen("mlff.log","a");
    	fp_mlff = mlff_str->fp_mlff;
    }
t1 = MPI_Wtime();
    double **X3_cur;
    int *org_idx;
    int ncols_to_remove;
    int *cols_to_remove;
    dyArray *highrank_ID_descriptors;
    highrank_ID_descriptors = (dyArray *) malloc(sizeof(dyArray)*mlff_str->nelem);

    for (int i=0; i < mlff_str->nelem; i++){
    	init_dyarray(&highrank_ID_descriptors[i]);
    	X3_cur = (double **) malloc(sizeof(double*)*mlff_str->natm_train_elemwise[i]);
    	org_idx = (int *)malloc(sizeof(int)*mlff_str->natm_train_elemwise[i]);

    	for (int j=0; j < mlff_str->natm_train_elemwise[i]; j++){
    		X3_cur[j] = (double *) malloc(sizeof(double)*mlff_str->size_X3);
    	}

    	count=0;
    	for (int j=0; j < mlff_str->natm_train_total; j++){
    		 if (mlff_str->natm_typ_train[j]==i){
    		 	org_idx[count] = j;
    		 	for (int k=0; k<mlff_str->size_X3; k++){
    		 		X3_cur[count][k] = mlff_str->X3_traindataset[j][k];
    		 	}
    		 	count++;
    		 }
    	}
    	if (count != mlff_str->natm_train_elemwise[i]){
    		printf("something wrong in CUR in mlff_train_Bayesian in rank %d\n",rank);
    		exit(1);
    	}

    	mlff_CUR_sparsify(mlff_str->kernel_typ, X3_cur, mlff_str->natm_train_elemwise[i], mlff_str->size_X3, mlff_str->xi_3, &highrank_ID_descriptors[i], 0);
    	ncols_to_remove = mlff_str->natm_train_elemwise[i] - (&highrank_ID_descriptors[i])->len;

    	if (mlff_str->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "For element %d, CUR removed %d columns out of a total of %d columns\n",i,ncols_to_remove,mlff_str->natm_train_elemwise[i]);
		}
    	cols_to_remove = (int *) malloc(sizeof(int)*ncols_to_remove);

    	count1=0;

    	for (int l=0; l < mlff_str->natm_train_elemwise[i]; l++){
    		count=0;
    		for (int k = 0; k <(&highrank_ID_descriptors[i])->len; k++){
    			if (l==(&highrank_ID_descriptors[i])->array[k]){
    				count++;
    			}
    		}
    		if (count==0){
    			cols_to_remove[count1] = org_idx[l];
    			count1++;
    		}
    	}
    	
    	if (count1 != ncols_to_remove){
    		printf("something wrong in CUR! in rank %d\n", rank);
    		exit(1);
    	}

    	int col_idx_toremove;
    	int cols_before_remove = mlff_str->n_cols;

    	for (int k=0; k < ncols_to_remove; k++){
 		
    		remove_train_cols(mlff_str, cols_to_remove[k]-k); 
    	}

    	delete_dyarray(&highrank_ID_descriptors[i]);
t2 = MPI_Wtime();
		if (mlff_str->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "sparsification before training done! Time taken: %.3f s\n", t2-t1);
		}


    	for (int j=0; j < mlff_str->natm_train_elemwise[i]; j++){
    		free(X3_cur[j]);
    	}
    	free(X3_cur);
    	free(org_idx);
    	free(cols_to_remove);
    }
    free(highrank_ID_descriptors);

}

/*
Bayesian linear regression to find the weights

Input:
1. mlff_str: MLFF structure

Output:
1. mlff_str: MLFF structure
*/
void mlff_train_Bayesian(MLFF_Obj *mlff_str){

	int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
double t1, t2, t3, t4;
t3 = MPI_Wtime();

t1 = MPI_Wtime();

	if (mlff_str->if_sparsify_before_train){
		CUR_sparsify_before_training(mlff_str);
	}

t2 = MPI_Wtime();
#ifdef DEBUG
	if (rank == 0) {
	    printf("CUR_sparsify_before_training took %.3f s.\n", t2 - t1); 
	}
#endif 
t1 = MPI_Wtime();

	FILE *fp_mlff;
    if (mlff_str->print_mlff_flag==1 && rank==0){
    	// fp_mlff = fopen("mlff.log","a");
    	fp_mlff = mlff_str->fp_mlff;
    }

	double *a_scaled, *b_scaled;
	int info, m = mlff_str->n_rows, n = mlff_str->n_cols;

	mlff_str->E_scale = 1.0;
	mlff_str->F_scale = mlff_str->std_E/mlff_str->std_F;
	for (int i = 0; i < mlff_str->stress_len; i++){
		mlff_str->stress_scale[i] = mlff_str->std_E/mlff_str->std_stress[i];
	}

	a_scaled = (double *) malloc(m*n * sizeof(double));
	b_scaled = (double *) malloc(m * sizeof(double));

	double scale =0;

	

	if (rank==0){
		for (int i = 0; i < m; i++ ){
			int quot = i%(1+mlff_str->stress_len+3*mlff_str->natom_domain);
			if (quot==0){
				scale = mlff_str->E_scale ;
				b_scaled[i] = (1.0/mlff_str->std_E)*(mlff_str->b_no_norm[i] - mlff_str->mu_E);
			} else if (quot>0 && quot < 1+mlff_str->stress_len) {
				scale = mlff_str->stress_scale[quot-1]* mlff_str->relative_scale_stress[quot-1];
				b_scaled[i] = (1.0/mlff_str->std_stress[quot-1])*(mlff_str->b_no_norm[i])* mlff_str->relative_scale_stress[quot-1];
			} else {
				scale = mlff_str->F_scale* mlff_str->relative_scale_F;
				b_scaled[i] = (1.0/mlff_str->std_F)*(mlff_str->b_no_norm[i])* mlff_str->relative_scale_F;
				
			}
			for (int j = 0; j < n; j++){
				a_scaled[j*m+i] = scale * mlff_str->K_train[i][j];
			}
		}
	} else {
		for (int i = 0; i < m; i++ ){
			b_scaled[i] = (1.0/mlff_str->std_F)*(mlff_str->b_no_norm[i])* mlff_str->relative_scale_F;
			for (int j = 0; j < n; j++){
				a_scaled[j*m+i] = mlff_str->F_scale* mlff_str->relative_scale_F * mlff_str->K_train[i][j];
			}
		}
	}

t2 = MPI_Wtime();
#ifdef DEBUG
	if (rank == 0) {
	    printf("Scaling K_train and b-vector took %.3f s.\n", t2 - t1); 
	}
#endif 
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Scaling K_train and b-vector done! Time taken: %.3f s\n", t2-t1);
	}
t1 = MPI_Wtime();

	double *AtA, *Atb, *AtA_h, *Atb_h;
	double btb=0.0;

	for (int i=0; i < mlff_str->n_rows; i++){
		btb += b_scaled[i] * b_scaled[i];
	}

	double btb_reduced;
	MPI_Allreduce(&btb, &btb_reduced, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	AtA = (double *) malloc(sizeof(double)* mlff_str->n_cols * mlff_str->n_cols);
	Atb = (double *) malloc(sizeof(double)* mlff_str->n_cols);

	AtA_h = (double *) malloc(sizeof(double)* mlff_str->n_cols * mlff_str->n_cols);
	Atb_h = (double *) malloc(sizeof(double)* mlff_str->n_cols);

	if ( mlff_str->n_rows > 0){
		// AtA MKL call
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mlff_str->n_cols, mlff_str->n_cols,
                    mlff_str->n_rows, 1.0, a_scaled, mlff_str->n_rows, a_scaled, mlff_str->n_rows,
                     0.0, AtA, mlff_str->n_cols);
		// Atb MKL call
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mlff_str->n_cols, 1, mlff_str->n_rows,
                    1.0, a_scaled, mlff_str->n_rows, b_scaled, mlff_str->n_rows,
                     0.0, Atb, mlff_str->n_cols);
		for (int i=0; i < mlff_str->n_cols * mlff_str->n_cols; i++){
			 AtA_h[i] = AtA[i];
		}
		for (int i=0; i < mlff_str->n_cols; i++){
			Atb_h[i] = Atb[i];
		}

	} else {
		for (int i=0; i < mlff_str->n_cols * mlff_str->n_cols; i++){
			AtA[i] = 0.0;
			AtA_h[i] = 0.0;
		}
		for (int i=0; i < mlff_str->n_cols; i++){
			Atb_h[i] = 0.0;
			Atb[i] = 0.0;
		}
	}

t2 = MPI_Wtime();
#ifdef DEBUG
	if (rank == 0) {
	    printf("Calculating btb, AtA, Atb took %.3f s.\n", t2 - t1); 
	}
#endif 

	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Calculation btb, AtA, Atb done! Time taken: %.3f s\n", t2-t1);
	}
	
t1 = MPI_Wtime();

	double *AtA_reduced, *Atb_reduced, *AtA_h_reduced, *Atb_h_reduced;
	AtA_reduced = (double *) malloc(sizeof(double)* mlff_str->n_cols * mlff_str->n_cols);
	Atb_reduced = (double *) malloc(sizeof(double)* mlff_str->n_cols);

	AtA_h_reduced = (double *) malloc(sizeof(double)* mlff_str->n_cols * mlff_str->n_cols);
	Atb_h_reduced = (double *) malloc(sizeof(double)* mlff_str->n_cols);

	MPI_Allreduce(AtA, AtA_reduced, mlff_str->n_cols * mlff_str->n_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(Atb, Atb_reduced, mlff_str->n_cols , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	for (int i=0; i < mlff_str->n_cols * mlff_str->n_cols; i++){
		AtA_h_reduced[i] = AtA_reduced[i];
	}
	for (int i=0; i < mlff_str->n_cols; i++){
		Atb_h_reduced[i] = Atb_reduced[i];
	}

t2 = MPI_Wtime();
#ifdef DEBUG
	if (rank == 0) {
	    printf("Doing MPI_Allreduce for AtA and Atb took %.3f s.\n", t2 - t1); 
	}
#endif 
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Doing MPI_Allreduce for AtA and Atb done! Time taken: %.3f s\n", t2-t1);
	}
	free(mlff_str->AtA);
	mlff_str->AtA = (double *) malloc(sizeof(double)*mlff_str->n_cols*mlff_str->n_cols);
	for (int i= 0; i < mlff_str->n_cols*mlff_str->n_cols; i++){
		mlff_str->AtA[i] = AtA_reduced[i];
	}

	int ipiv[mlff_str->n_cols];
	int M_total_rows;
	MPI_Allreduce(&mlff_str->n_rows, &M_total_rows, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	free(mlff_str->AtA_SVD_U);
	free(mlff_str->AtA_SVD_Vt);
	free(mlff_str->AtA_SingVal);
	mlff_str->AtA_SVD_U = (double *)malloc(mlff_str->n_cols*mlff_str->n_cols*sizeof(double));
	mlff_str->AtA_SVD_Vt = (double *)malloc(mlff_str->n_cols*mlff_str->n_cols*sizeof(double));
	mlff_str->AtA_SingVal = (double *)malloc(mlff_str->n_cols*sizeof(double));

t1 = MPI_Wtime();
	if (rank==0){
		double sigma_w, sigma_v;
		int dohyperparameter = 1;
		if (dohyperparameter){
			hyperparameter_Bayesian(btb_reduced, AtA_h_reduced, Atb_h_reduced, mlff_str, M_total_rows, mlff_str->condK_min);
		}

		MPI_Bcast(&mlff_str->sigma_v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mlff_str->sigma_w, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->weights, mlff_str->n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->AtA_SingVal, mlff_str->n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->AtA_SVD_U, mlff_str->n_cols*mlff_str->n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->AtA_SVD_Vt, mlff_str->n_cols*mlff_str->n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast(&mlff_str->sigma_v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mlff_str->sigma_w, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->weights, mlff_str->n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->AtA_SingVal, mlff_str->n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->AtA_SVD_U, mlff_str->n_cols*mlff_str->n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->AtA_SVD_Vt, mlff_str->n_cols*mlff_str->n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

t2 = MPI_Wtime();
#ifdef DEBUG
	if (rank == 0) {
	    printf("Doing hyperparameter_Bayesian and broadcasting weights and SVD took %.3f s.\n", t2 - t1); 
	}	
#endif 
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "hyperparameter_Bayesian and broadcasting weights done! Time taken: %.3f s\n", t2-t1);
	}
t1 = MPI_Wtime();
	double *b_predict, *error_b_scaled;
	double error_train_F=0.0, error_train_E=0.0, error_train_maxF;
	double error_train_stress[mlff_str->stress_len];
	for (int i = 0; i < mlff_str->stress_len; i++){
		error_train_stress[i] = 0.0;
	}
	b_predict = (double *) malloc(mlff_str->n_rows *sizeof(double));
	error_b_scaled = (double *) malloc(mlff_str->n_rows *sizeof(double));
	if (mlff_str->n_rows > 0){
	    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
	            mlff_str->n_rows, 1, mlff_str->n_cols, 1.0, a_scaled, mlff_str->n_rows,
	             mlff_str->weights, mlff_str->n_cols, 0.0, b_predict, mlff_str->n_rows);
	    for (int i=0; i < mlff_str->n_rows; i++){
			error_b_scaled[i] = fabs(b_predict[i] - b_scaled[i]);
	    }
	    if (rank==0){
	    	for (int i=0; i < mlff_str->n_rows; i++){
	    		int quot = i%(1+3*mlff_str->natom_domain+mlff_str->stress_len);
	    		if (quot==0){
	    			error_b_scaled[i] = error_b_scaled[i] * mlff_str->std_E;
	    			if (error_b_scaled[i]>error_train_E){
	    				error_train_E = error_b_scaled[i];
	    			}
	    		} else if (quot>0 && quot < 1+mlff_str->stress_len){
	    			error_b_scaled[i] = error_b_scaled[i] * mlff_str->std_stress[quot-1] / mlff_str->relative_scale_stress[quot-1];
	    			if (error_b_scaled[i] > error_train_stress[quot-1]){
	    				error_train_stress[quot-1] = error_b_scaled[i];
	    			}
	    		} else {
	    			error_b_scaled[i] = error_b_scaled[i] * mlff_str->std_F / mlff_str->relative_scale_F;
	    			if (error_b_scaled[i]>error_train_F){
	    				error_train_F = error_b_scaled[i];
	    			}
	    		}
	    	}
	    } else {
	    	for (int i=0; i < mlff_str->n_rows; i++){
	    		error_b_scaled[i] = error_b_scaled[i] * mlff_str->std_F / mlff_str->relative_scale_F;
	    		if (error_b_scaled[i]>error_train_F){
	    			error_train_F = error_b_scaled[i];
	    		}
	    	}
	    }
	} else {
		error_train_F = 0.0;
		error_train_E = 0.0;
	}

	MPI_Bcast(&error_train_E, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(error_train_stress, mlff_str->stress_len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Allreduce(&error_train_F, &error_train_maxF, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	mlff_str->error_train_E = error_train_E;
	mlff_str->error_train_F = error_train_maxF;
	for (int i=0; i < mlff_str->stress_len; i++){
		mlff_str->error_train_stress[i] = error_train_stress[i];
	}


t2 = MPI_Wtime();
#ifdef DEBUG
	if (rank == 0) {
	    printf("Getting training errors took %.3f s.\n", t2 - t1); 
	}
#endif
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Training errors done! Time taken: %.3f s\n", t2-t1);
	}

	print_restart_MLFF(mlff_str);
	if (rank==0) print_ref_atom_MLFF(mlff_str);

	free(a_scaled); free(AtA_reduced); free(Atb_reduced);free(AtA_h_reduced); free(Atb_h_reduced);
	free(b_scaled);
	free(AtA);
	free(Atb);
	free(AtA_h);
	free(Atb_h);
	free(b_predict);
	free(error_b_scaled);
t4 = MPI_Wtime();
#ifdef DEBUG
	if (rank == 0) {
	    printf("mlff_train_Bayesian took %.3f s.\n", t4 - t3); 
	}
#endif
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "mlff_train_Bayesian done! Time taken: %.3f s\n", t2-t1);
		// fclose(fp_mlff);
	}
}


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
void hyperparameter_Bayesian(double btb_reduced, double *AtA, double *Atb, MLFF_Obj *mlff_str, int M_total_rows, double condK_min){
	// printf("Inside hyperparameter_Bayesian std_F: %f, std_E: %f, mu_E: %f\n",mlff_str->std_F, mlff_str->std_E, mlff_str->mu_E);
	int rank = 0;
double t1, t2, t3, t4;
	FILE *fp_mlff;
    if (mlff_str->print_mlff_flag==1 && rank==0){
    	// fp_mlff = fopen("mlff.log","a");
    	fp_mlff = mlff_str->fp_mlff;
    }


t1 = MPI_Wtime();

	double *AtA_h, *Atb_h;

	AtA_h = (double *) malloc(mlff_str->n_cols*mlff_str->n_cols*sizeof(double));
	Atb_h = (double *) malloc(mlff_str->n_cols*sizeof(double));

	for (int i =0; i <mlff_str->n_cols; i++){
		Atb_h[i] = Atb[i];
		for (int j=0; j <mlff_str->n_cols; j++){
			 AtA_h[j*mlff_str->n_cols+i] = AtA[j*mlff_str->n_cols+i];
		}
	}

	int ipiv[mlff_str->n_cols], info;

	// info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', mlff_str->n_cols, AtA_h, mlff_str->n_cols, lambda_0);

	double *u, *vt, *S_0, *S, *superb;
	S_0 = (double *) malloc(sizeof(double)* mlff_str->n_cols);
	S = (double *) malloc(sizeof(double)* mlff_str->n_cols);
	u = (double *)malloc(sizeof(double)*mlff_str->n_cols*mlff_str->n_cols);
	vt = (double *)malloc(sizeof(double)*mlff_str->n_cols*mlff_str->n_cols);
	superb = (double *)malloc(sizeof(double)*mlff_str->n_cols);

t2 = MPI_Wtime();
#ifdef DEBUG
	printf("Doing hyperparameter_Bayesian and broadcasting weights and SVD took %.3f s.\n", t2 - t1); 
#endif

	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "broadcasting weights and SVD done! Time taken: %.3f s\n", t2-t1);
	}

t1 = MPI_Wtime();

	info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', mlff_str->n_cols, mlff_str->n_cols, 
						  AtA_h, mlff_str->n_cols, S_0, u, mlff_str->n_cols, vt, mlff_str->n_cols,superb);

t2 = MPI_Wtime();
#ifdef DEBUG
	printf("Doing SVD of AtA took %.3f s.\n", t2 - t1); 
#endif
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "SVD of AtA done! Time taken: %.3f s\n", t2-t1);
	}
t1 = MPI_Wtime();

	if (info==0){
		printf("SVD inside hyperparameter_Bayesian is successful!!\n");
	} else if (info < 0){
		
		printf("the %d-th parameter in SVD inside hyperparameter_Bayesian had an illegal value.\n", info);
		exit(1);
	} else {
		printf(" ?bdsqr did not converge in SVD inside hyperparameter_Bayesian.\n");
		exit(1);
	}

	for (int i=0; i < mlff_str->n_cols; i++){
		mlff_str->AtA_SingVal[i] = S_0[i];
	}

	for (int i=0; i < mlff_str->n_cols * mlff_str->n_cols; i++){
		mlff_str->AtA_SVD_U[i] = u[i];
		mlff_str->AtA_SVD_Vt[i] = vt[i];
	}

	double error_w = 10, error_v=1, error_tol_v = 0.001, error_tol_w = 1, sigma_v, sigma_w;
	double sigma_v0, sigma_w0;
	double *term2, *term3, *AtAw;
	double *weights, *Ax, *e;
	double *s_loop;
	int matrank;

	weights = (double *) malloc(mlff_str->n_cols*sizeof(double));
	Ax = (double *) malloc(sizeof(double)*mlff_str->n_rows);
	e = (double *) malloc(sizeof(double)*mlff_str->n_rows);

	term2 = (double *)malloc(sizeof(double)*1);
	term3 = (double *)malloc(sizeof(double)*1);
	AtAw = (double *)malloc(sizeof(double)*mlff_str->n_cols);
    
    s_loop = (double *)malloc(sizeof(double)*mlff_str->n_cols);

    double regul_min = get_regularization_min(AtA, mlff_str->n_cols, condK_min);

    sigma_v0 = mlff_str->sigma_v;
    sigma_w0 =  mlff_str->sigma_w;
#ifdef DEBUG   
    printf("Regularization minimum: 1e%f\n", log10(regul_min));
    printf("Loop for Bayesian regression hyperparameters optimization: \n");
    printf("Initial guess; sigma_v0: %f, sigma_w0: %f\n",sigma_v0,sigma_w0);
#endif
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Regularization minimum: 1e%f\n", log10(regul_min));
		fprintf(fp_mlff, "Loop for Bayesian regression hyperparameters optimization: \n");
		fprintf(fp_mlff, "Initial guess; sigma_v0: %f, sigma_w0: %f\n",sigma_v0,sigma_w0);
	}
t2 = MPI_Wtime();
printf("Saving and other initialization before hyperparamter SCF loop took %.3f s.\n", t2 - t1); 
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Saving and other initialization before hyperparamter done! Time taken: %.3f s\n", t2-t1);
	}
t1 = MPI_Wtime();
    int count=0;
    while( (error_w > error_tol_w || error_v > error_tol_v) && count <10) {

t3 = MPI_Wtime();
    	double gamma = 0.0;
    	for (int k = 0; k < mlff_str->n_cols; k++){
    		S[k] = S_0[k]/(sigma_v0*sigma_v0);
    		if (S_0[k] > 1e-10){
    			gamma += S[k]/(S[k] + 1.0/(sigma_w0*sigma_w0));
    		}
    	}
t4 = MPI_Wtime();
#ifdef DEBUG 
		printf("Gamma calculation took %.3f s.\n", t4 - t3); 
    	printf("Iter number: %d\n",count);
    	printf("Gamma: %f\n",gamma);

#endif
    	if (mlff_str->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "Gamma calculation done! It took %.3f s.\n", t4 - t3);
			fprintf(fp_mlff, "Iter number: %d\n",count);
			fprintf(fp_mlff, "Gamma: %f\n",gamma);
		}
t3 = MPI_Wtime();

    	for (int i = 0; i < mlff_str->n_cols*mlff_str->n_cols; i++ ){
    		AtA_h[i] = AtA[i];
    	}

    	for (int i =0; i <mlff_str->n_cols; i++){
    		AtA_h[i*mlff_str->n_cols+i] +=  (sigma_v0*sigma_v0)/(sigma_w0*sigma_w0);
    	}

t4 = MPI_Wtime();
#ifdef DEBUG 
	printf("Copying AtA and regularizing took %.3f s.\n", t4 - t3);
#endif 
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Copying AtA and regularizing took %.3f s.\n", t4 - t3);
	}
    	// info = LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', mlff_str->n_cols, 1, AtA_h, mlff_str->n_cols, &ipiv[0], Atb_h, mlff_str->n_cols);

t3 = MPI_Wtime();
    	LAPACKE_dgelsd(LAPACK_COL_MAJOR, mlff_str->n_cols, mlff_str->n_cols, 1, AtA_h,
    	 mlff_str->n_cols, Atb_h, mlff_str->n_cols, s_loop, -1.0, &matrank);

t4 = MPI_Wtime();
#ifdef DEBUG
	printf("Linear system solving took %.3f s.\n", t4 - t3);
#endif 
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Linear system solving took %.3f s.\n", t4 - t3);
	}
t3 = MPI_Wtime();

    	for (int i=0; i <mlff_str->n_cols; i++){
    		weights[i] = Atb_h[i];
    	}

    	 for (int i =0; i <mlff_str->n_cols; i++) {
    	 	Atb_h[i] = Atb[i];
    	 }

    	double norm_w2 = dotProduct(weights, weights, mlff_str->n_cols);

t4 = MPI_Wtime();
#ifdef DEBUG
	printf("Copying weights and calculating norm_w2 took %.3f s.\n", t4 - t3);
#endif 
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Copying weights and calculating norm_w2 took %.3f s.\n", t4 - t3);
	}
t3 = MPI_Wtime();

    	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                1, 1, mlff_str->n_cols, 1.0, weights, mlff_str->n_cols,
                 Atb, mlff_str->n_cols, 0.0, term2, 1);

    	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                mlff_str->n_cols, 1, mlff_str->n_cols, 1.0, AtA, 
                mlff_str->n_cols, weights, mlff_str->n_cols, 0.0, AtAw, mlff_str->n_cols);

    	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                1, 1, mlff_str->n_cols, 1.0, weights, mlff_str->n_cols, 
                AtAw, mlff_str->n_cols, 0.0, term3, 1);

    	double norm_er2 = btb_reduced - 2*term2[0] + term3[0];

    	// printf("term2: %f, term3: %f, btb_reduced: %f\n",term2[0], term3[0],btb_reduced);

    	sigma_w = sqrt(norm_w2/gamma);
    	sigma_v = sqrt(norm_er2/(M_total_rows - gamma));

t4 = MPI_Wtime();
#ifdef DEBUG
		printf("Calculating sigma_w and sigma_w for the next iterations took %.3f s.\n", t4 - t3);
    	printf("norm_er2: %f, M_total_rows: %d, gamma: %f\n",norm_er2,M_total_rows,gamma);
    	printf("sigma_v: %f and sigma_w: %f before check for cond\n",sigma_v,sigma_w);
#endif 
    	if (mlff_str->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "Calculating sigma_w and sigma_w for the next iterations took %.3f s.\n", t4 - t3);
			fprintf(fp_mlff, "norm_er2: %f, M_total_rows: %d, gamma: %f\n",norm_er2,M_total_rows,gamma);
			fprintf(fp_mlff, "sigma_v: %f and sigma_w: %f before check for cond\n",sigma_v,sigma_w);
		}
    	if ((sigma_v*sigma_v)/(sigma_w*sigma_w) < regul_min){
#ifdef DEBUG    		
    		printf("regularization: %f is too small as computed from the loop\n", (sigma_v*sigma_v/(sigma_w*sigma_w)));
#endif     		
    		if (mlff_str->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "regularization: %f is too small as computed from the loop\n", (sigma_v*sigma_v/(sigma_w*sigma_w)));
			}
			sigma_w = sqrt(sigma_v*sigma_v/regul_min);
			
		}

		error_w = fabs(sigma_w-sigma_w0);
		error_v = fabs(sigma_v-sigma_v0);
		sigma_w0 = sigma_w;
		sigma_v0 = sigma_v;
#ifdef DEBUG  		
		printf("After condK: sigma_v: %f, sigma_w: %f\n",sigma_v,sigma_w);
#endif		
		if (mlff_str->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "After condK: sigma_v: %f, sigma_w: %f\n",sigma_v,sigma_w);
		}
		count++;

    }

t2 = MPI_Wtime();
#ifdef DEBUG
	printf("Hyperparameter SCF loop took %.3f s.\n", t2 - t1); 
#endif
	if (mlff_str->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Hyperparameter SCF loop took %.3f s.\n", t2 - t1);
	}
    mlff_str->sigma_w = sigma_w;
    mlff_str->sigma_v = sigma_v;

	for (int i=0; i < mlff_str->n_cols; i++){
		mlff_str->weights[i] = weights[i];
	}

	if (error_w< error_tol_w &&  error_v < error_tol_v){
#ifdef DEBUG
	    printf("sigma_v and sigma_w converged in %d iteration\n",count);
#endif
	    if (mlff_str->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "sigma_v and sigma_w converged in %d iteration\n",count);
		}
	}
	else{
	    printf("WARNING: sigma_v and sigma_w did not converge and error was %10.9f, %10.9f\n",error_v, error_w);
	}

	// if (mlff_str->print_mlff_flag == 1 && rank ==0){
	// 	fclose(fp_mlff);
	// }

	free(AtA_h);
	free(Atb_h);
	free(weights);
	free(Ax);
	free(e);
	free(S_0);
	free(S);
	free(u);
	free(vt);
	free(superb);
	free(s_loop);
	free(term2);
	free(term3);
	free(AtAw);

}

/*
Finding regularization constant to satisfy the condK_min

Input:
1. A: The K_train matrix
2. size: size of K'K matrix
3. condK_min: Minimum conition number of K'K allowed after regularization

Output:
1. reg_final: Regularization constant
*/
double get_regularization_min(double *A, int size, double condK_min){
	double *a_copy;
	int ipiv[size], info;
	double rcond, norm;
	double reg_final;
	a_copy = (double*)malloc(size*size*sizeof(double));

	double reg_temp[16] = {1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
	reg_final = 1e-16;
	for (int k=0; k < 16; k++){
		for (int i=0; i < size*size; i++){
			a_copy[i] =A[i];
		}

		for (int i=0; i < size; i++){
			a_copy[i+size*i] += reg_temp[k];
		}

		norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'I', size, size, a_copy, size);
		info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, size, size, a_copy, size, &ipiv[0]);
		LAPACKE_dgecon(LAPACK_COL_MAJOR, 'I', size, a_copy, size, norm, &rcond);
#ifdef DEBUG
		printf("Regularization: 1e-%f, Condition number reciprocal: 1e%f\n",log10(reg_temp[k]), log10(rcond) );
#endif
		if (rcond > condK_min){
			reg_final = reg_temp[k];
			 break;
		}
	}
	free(a_copy);
	return reg_final;

}



/*
Calculate Energy, force, stress/pressure for prediction

Input:
1. K_predict: The K matrix or prediction structure
2. mlff_str: MLFF structure
3. condK_min: Minimum conition number of K'K allowed after regularization

Output:
1. E, F, stress, error_bayesian: Outputs
*/
void mlff_predict(double *K_predict, MLFF_Obj *mlff_str, double *E,  double* F, double *stress, double* error_bayesian, int natoms ){
	int rank;
	int quot;
	double regul;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int rows, cols;

double t1, t2, t3, t4;

t1 = MPI_Wtime();

	if (rank==0){
		rows = 3*mlff_str->natom_domain + 1 + mlff_str->stress_len;
		cols = mlff_str->n_cols;
	} else {
		rows = 3*mlff_str->natom_domain;
		cols = mlff_str->n_cols;
	}

	double *b_predict;
	b_predict = (double *) malloc(rows *sizeof(double));
	double *KV = (double *) malloc(rows *cols* sizeof(double));
	double *DiagMat = (double *) malloc(cols *cols* sizeof(double));
	double *KV_times_DiagMat = (double *) malloc(rows *cols* sizeof(double));
	double *KV_times_DiagMat_times_Ut = (double *) malloc(rows *cols* sizeof(double));
	double *KV_times_DiagMat_times_Ut_times_Kt = (double *) malloc(rows *rows* sizeof(double));

	if((mlff_str->mlff_flag == 1) || (mlff_str->mlff_flag == 22)){
		for (int i = 0; i < cols *cols; i++){
			DiagMat[i] = 0.0;
		}

		regul = (mlff_str->sigma_v*mlff_str->sigma_v)/(mlff_str->sigma_w*mlff_str->sigma_w);
		double tol_pinv = 1e-15 * mlff_str->n_cols * largest(mlff_str->AtA_SingVal, mlff_str->n_cols);
		for (int i = 0; i < cols; i++){
			if (mlff_str->AtA_SingVal[i]> tol_pinv)
				DiagMat[i*cols +i] = 1.0/(regul + mlff_str->AtA_SingVal[i]);
		}
	}
	

	if (rows > 0){
	    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
	            rows, 1, cols, 1.0, K_predict, rows, mlff_str->weights, cols, 0.0, b_predict, rows);
	}

t2 = MPI_Wtime();
#ifdef DEBUG
	if (rank==0){
		printf("b_predict calculation took %.3f s.\n", t2 - t1); 
	}
#endif
t1 = MPI_Wtime();

	if((mlff_str->mlff_flag == 1) || (mlff_str->mlff_flag == 22)){
		if (rows > 0){
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
			rows, cols, cols, 1.0, K_predict, rows, mlff_str->AtA_SVD_Vt, cols, 0.0, KV, rows);

			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
			rows, cols, cols, 1.0, KV, rows, DiagMat, cols, 0.0, KV_times_DiagMat, rows);

			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
			rows, cols, cols, 1.0, KV_times_DiagMat, rows, mlff_str->AtA_SVD_U, cols, 0.0, KV_times_DiagMat_times_Ut, rows);

			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
			rows, rows, cols, 1.0, KV_times_DiagMat_times_Ut, rows, K_predict, rows, 0.0, KV_times_DiagMat_times_Ut_times_Kt, rows);

			for (int i = 0; i < rows *rows; i++){
				KV_times_DiagMat_times_Ut_times_Kt[i] = KV_times_DiagMat_times_Ut_times_Kt[i] * (mlff_str->sigma_v*mlff_str->sigma_v);
			}
		}

		if (rank==0){
			for (int i=0; i<rows; i++){
				if (i==0){
					error_bayesian[i] = (mlff_str->std_E)*sqrt(mlff_str->sigma_v*mlff_str->sigma_v +KV_times_DiagMat_times_Ut_times_Kt[i*rows+i]);
				} else if (i > 0 && i < 1+mlff_str->stress_len){
					error_bayesian[i] = (1.0/mlff_str->relative_scale_stress[i-1])*(mlff_str->std_stress[i-1])*sqrt(mlff_str->sigma_v*mlff_str->sigma_v + KV_times_DiagMat_times_Ut_times_Kt[i*rows+i]);
				} else{
					error_bayesian[i] = (1.0/mlff_str->relative_scale_F)*(mlff_str->std_F)*sqrt(mlff_str->sigma_v*mlff_str->sigma_v+KV_times_DiagMat_times_Ut_times_Kt[i*rows+i]);
				}
			}
		} else {
			for (int i=0; i<rows; i++){
				error_bayesian[i] = (1.0/mlff_str->relative_scale_F)*(mlff_str->std_F)*sqrt(mlff_str->sigma_v*mlff_str->sigma_v + KV_times_DiagMat_times_Ut_times_Kt[i*rows+i]);
			}
		}

	}

t2 = MPI_Wtime();
#ifdef DEBUG
	if (rank==0){
		printf("error_bayesian calculation took %.3f s.\n", t2 - t1); 
	}
#endif
t1 = MPI_Wtime();

	if (rank==0){
		E[0] = b_predict[0] * mlff_str->std_E + mlff_str->mu_E;
		for (int istress = 0; istress < mlff_str->stress_len; istress++){
			stress[istress] = b_predict[1+istress] * mlff_str->std_stress[istress] * (1.0/mlff_str->relative_scale_stress[istress]);
		}
		for (int i=0; i<mlff_str->natom_domain; i++){
	  		for (int j=0; j<3; j++){
	    		F[i*3+j] = b_predict[i*3+j+1+mlff_str->stress_len] * mlff_str->std_F * (1.0/mlff_str->relative_scale_F);
	  		}
		}
	} else {
		E[0] = 0.0;
		for (int istress = 0; istress < mlff_str->stress_len; istress++){
			stress[istress] = 0.0;
		}
		for (int i=0; i<mlff_str->natom_domain; i++){
	  		for (int j=0; j<3; j++){
	    		F[i*3+j] = b_predict[i*3+j] * mlff_str->std_F * (1.0/mlff_str->relative_scale_F);
	  		}
		}
	}
t2 = MPI_Wtime();
#ifdef DEBUG
	if (rank==0){
		printf("E, and F calculation took %.3f s.\n", t2 - t1); 
	}
#endif
	free(b_predict);
	free(KV);
	free(DiagMat);
	free(KV_times_DiagMat);
	free(KV_times_DiagMat_times_Ut);
	free(KV_times_DiagMat_times_Ut_times_Kt);

}
