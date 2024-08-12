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
#include "descriptor.h"
#include "mlff_types.h"
#include "sparsification.h"
#include "regression.h"
#include "isddft.h"
#include "ddbp_tools.h"
#include "covariance_matrix.h"
#include "sparc_mlff_interface.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
//#define au2GPa 29421.02648438959


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

void calculate_polynomial_kernel_ij(DescriptorObj *desc_str, double **X3_traindataset, int *natm_typ_train, int n_cols, double **kernel_ij){

	int size_X3 = desc_str->size_X3;
	int natom_domain = desc_str->natom_domain;


	double norm_X3_str[natom_domain], norm_X3_tr[n_cols];

	for (int i = 0; i < natom_domain; i++){
		norm_X3_str[i] = sqrt(dotProduct(desc_str->X3[i], desc_str->X3[i], size_X3));
	}

	for (int i = 0; i < n_cols; i++){
		norm_X3_tr[i] = sqrt(dotProduct(X3_traindataset[i], X3_traindataset[i], size_X3));
	}


	
	double **X3_str_unitvec, **X3_tr_unitvec;
	X3_str_unitvec = (double **)malloc(sizeof(double*)*natom_domain);
	X3_tr_unitvec = (double **)malloc(sizeof(double*)*n_cols);
	for (int i = 0; i < natom_domain; i++){
		X3_str_unitvec[i] = (double *)malloc(sizeof(double)*size_X3);
	}
	for (int i = 0; i < n_cols; i++){
		X3_tr_unitvec[i] = (double *)malloc(sizeof(double)*size_X3);
	}

	for (int i = 0; i < natom_domain; i++){
		for (int j = 0; j < size_X3; j++){
			X3_str_unitvec[i][j] = desc_str->X3[i][j]/norm_X3_str[i];
		}
	}

	for (int i = 0; i < n_cols; i++){
		for (int j = 0; j < size_X3; j++){
			X3_tr_unitvec[i][j] = X3_traindataset[i][j]/norm_X3_tr[i];
		}
	}


	for (int i = 0; i < natom_domain; i++){
		int el_str = desc_str->el_idx_domain[i];
		for (int j = 0; j < n_cols; j++){
			int el_tr = natm_typ_train[j];
			if (el_str == el_tr){
				kernel_ij[i][j] = pow(dotProduct(X3_str_unitvec[i], X3_tr_unitvec[j], size_X3), desc_str->xi_3);
			}
		}
	}

	for (int i = 0; i < natom_domain; i++){
		free(X3_str_unitvec[i]);
	}
	for (int i = 0; i < n_cols; i++){
		free(X3_tr_unitvec[i]);
	}
	free(X3_str_unitvec);
	free(X3_tr_unitvec);
}

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

void calculate_zit(DescriptorObj *desc_str, double **X3_traindataset, int *natm_typ_train, int n_cols, double ***zit){
	int size_X3 = desc_str->size_X3;
	int natom_domain = desc_str->natom_domain;


	double norm_X3_str[natom_domain]; 
	for (int i = 0; i < natom_domain; i++){
		norm_X3_str[i] = sqrt(dotProduct(desc_str->X3[i], desc_str->X3[i], size_X3));
	}

	double **norm_str_tr = (double **) malloc(sizeof(double*)*natom_domain);
	for (int i = 0; i < natom_domain; i++){
		norm_str_tr[i] = (double *) malloc(sizeof(double)*n_cols);
		for (int j = 0; j < n_cols; j++){
			norm_str_tr[i][j] = 0.0;
		}
	}

	for (int i = 0; i < natom_domain; i++){
		int el_str = desc_str->el_idx_domain[i];
		for (int j = 0; j < n_cols; j++){
			int el_tr = natm_typ_train[j];
			if (el_str == el_tr){
				norm_str_tr[i][j] = dotProduct(desc_str->X3[i], X3_traindataset[j], size_X3);
			}
		}
	}

	for (int i = 0; i < natom_domain; i++){
		int el_str = desc_str->el_idx_domain[i];
		for (int j = 0; j < n_cols; j++){
			int el_tr = natm_typ_train[j];
			if (el_str == el_tr){
				for (int k = 0; k < size_X3; k++){
					zit[i][j][k] = X3_traindataset[j][k]/norm_str_tr[i][j] - desc_str->X3[i][k]/(norm_X3_str[i]*norm_X3_str[i]); // -ve of force
				}
			}
		}
	}

	for (int i = 0; i < natom_domain; i++){
		free(norm_str_tr[i]);
	}
	free(norm_str_tr);
}

/*
copy_descriptors function copies the content of one SoapObj to another

[Input]
1. desc_str: DescriptorObj structure to be copied
[Output]
1. desc_str_MLFF: DescriptorObj structure where it needs to be copied
*/

// These functions will become problematic if different species are used or if parallelization changes
void copy_descriptors(DescriptorObj *desc_str_MLFF, DescriptorObj *desc_str){

	memcpy(desc_str_MLFF->atom_idx_domain, desc_str->atom_idx_domain, desc_str->natom_domain*sizeof(int));
	memcpy(desc_str_MLFF->el_idx_domain, desc_str->el_idx_domain, desc_str->natom_domain*sizeof(int));
	memcpy(desc_str_MLFF->Nneighbors, desc_str->Nneighbors, desc_str->natom_domain*sizeof(int));
	memcpy(desc_str_MLFF->unique_Nneighbors, desc_str->unique_Nneighbors, desc_str->natom_domain*sizeof(int));
	memcpy(desc_str_MLFF->natom_elem, desc_str->natom_elem, desc_str->nelem*sizeof(int));
	for (int i = 0; i < desc_str->natom_domain; i++){
		memcpy(desc_str_MLFF->unique_Nneighbors_elemWise[i], desc_str->unique_Nneighbors_elemWise[i], desc_str->nelem*sizeof(int));
	}

	for (int i = 0; i < desc_str->natom_domain; i++){
		free(desc_str_MLFF->unique_neighborList[i].array);
		desc_str_MLFF->unique_neighborList[i].capacity = desc_str->unique_neighborList[i].capacity;
		desc_str_MLFF->unique_neighborList[i].array = (int *)malloc(desc_str->unique_neighborList[i].capacity*sizeof(int));
		desc_str_MLFF->unique_neighborList[i].len = desc_str->unique_neighborList[i].len;
		for (int k =0; k<desc_str->unique_neighborList[i].len; k++){
			desc_str_MLFF->unique_neighborList[i].array[k] = desc_str->unique_neighborList[i].array[k];
		}
	}


	for (int i = 0; i < desc_str->natom_domain; i++){
		memcpy(desc_str_MLFF->X3[i], desc_str->X3[i], desc_str->size_X3*sizeof(double));
		int uniq_natms = desc_str->unique_Nneighbors[i];
		// Large rcuts result in self image in neighbor list
		if (uniq_natms == desc_str->natom) uniq_natms -= 1;
		for (int j = 0; j < 1+uniq_natms; j++){
			memcpy(desc_str_MLFF->dX3_dX[i][j], desc_str->dX3_dX[i][j], desc_str->size_X3*sizeof(double));
			memcpy(desc_str_MLFF->dX3_dY[i][j], desc_str->dX3_dY[i][j], desc_str->size_X3*sizeof(double));
			memcpy(desc_str_MLFF->dX3_dZ[i][j], desc_str->dX3_dZ[i][j], desc_str->size_X3*sizeof(double));
		}
		for (int j = 0; j < 6; j++){
			memcpy(desc_str_MLFF->dX3_dF[i][j], desc_str->dX3_dF[i][j], desc_str->size_X3*sizeof(double));
		}
	}

}



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


void add_firstMD(DescriptorObj *desc_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double *F, double *stress_sparc) {

	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	
	
	int natom = desc_str->natom;
	int nelem = desc_str->nelem;
	double xi_3 = desc_str->xi_3;

	int size_X3;
	size_X3 = desc_str->size_X3;

	
	

	int row_idx, col_idx, count, atom_idx, idx;
	int iatm_str, istress;

	// calculate mean and std deviation to do the normalization
	mlff_str->E_store[mlff_str->E_store_counter] = E;
	for (int i = 0; i < mlff_str->stress_len; i++)
		mlff_str->stress_store[i][mlff_str->E_store_counter] = stress_sparc[i];

	for (int i=0; i < 3*natom; i++){
		mlff_str->F_store[mlff_str->F_store_counter+i] = F[i];
	}
	mlff_str->E_store_counter += 1;
	mlff_str->F_store_counter += 3*natom;

	int kernel_typ = mlff_str->kernel_typ;
	mlff_str->mu_E  = E;
	mlff_str->std_E  = 1;
	mlff_str->std_F  = sqrt(get_variance(F, 3*natom));
	for (int i = 0; i < mlff_str->stress_len; i++){
		mlff_str->std_stress[i]  = 1;
	}

	// populate bvec
	if (rank==0){
		mlff_str->b_no_norm[0] = E;
		for (int istress=0; istress < mlff_str->stress_len; istress++){
			mlff_str->b_no_norm[1+istress] =  stress_sparc[istress];
		}

		for (int i = 0; i < mlff_str->natom_domain; i++){
			idx = mlff_str->atom_idx_domain[i];
			mlff_str->b_no_norm[3*i+1+mlff_str->stress_len] = F[3*idx];
			mlff_str->b_no_norm[3*i+2+mlff_str->stress_len] = F[3*idx+1];
			mlff_str->b_no_norm[3*i+3+mlff_str->stress_len] = F[3*idx+2];

		}
	} else{
		for (int i = 0; i < mlff_str->natom_domain; i++){
			idx = mlff_str->atom_idx_domain[i];
			mlff_str->b_no_norm[3*i] = F[3*idx];
			mlff_str->b_no_norm[3*i+1] = F[3*idx+1];
			mlff_str->b_no_norm[3*i+2] = F[3*idx+2];
		}
	}

	int *cum_natm_elem = (int *)malloc(sizeof(int)*nelem);
	
	//Stores the starting index wrt SPARC for each element type - sequential
	cum_natm_elem[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem[i] = cum_natm_elem[i-1]+desc_str->natom_elem[i-1];
	}

	double  *X3_gathered;
	X3_gathered = (double *) malloc(sizeof(double)*size_X3*desc_str->natom);

	double *X3_local;
	X3_local = (double *) malloc(sizeof(double)*size_X3*mlff_str->natom_domain);

	for (int i=0; i < mlff_str->natom_domain; i++){
		for (int j=0; j < size_X3; j++){
			X3_local[i*size_X3+j] = desc_str->X3[i][j];
		}
	}

	int local_natoms[nprocs];
	MPI_Allgather(&mlff_str->natom_domain, 1, MPI_INT, local_natoms, 1, MPI_INT, MPI_COMM_WORLD);

	int recvcounts_X3[nprocs], displs_X3[nprocs];
	displs_X3[0] = 0;
	for (int i=0; i < nprocs; i++){
		recvcounts_X3[i] = local_natoms[i]*size_X3;
		if (i>0){
			displs_X3[i] = displs_X3[i-1]+local_natoms[i-1]*size_X3;
		}
	}


	MPI_Allgatherv(X3_local, size_X3*mlff_str->natom_domain, MPI_DOUBLE, X3_gathered, recvcounts_X3, displs_X3, MPI_DOUBLE, MPI_COMM_WORLD);

	double **X3_gathered_2D = (double **) malloc(sizeof(double*)*natom);
	for (int i=0; i < natom; i++){
		X3_gathered_2D[i] = (double *) malloc(sizeof(double)*size_X3);
		for (int j=0; j < size_X3; j++){
			X3_gathered_2D[i][j] = X3_gathered[i*size_X3+j];
		}
	}


	dyArray *highrank_ID_descriptors = (dyArray *) malloc(sizeof(dyArray)*nelem);
	for (int i=0; i <nelem; i++){
		int N_low_min = desc_str->natom_elem[i] - 500;
		init_dyarray(&highrank_ID_descriptors[i]);
		mlff_CUR_sparsify(kernel_typ, &X3_gathered_2D[cum_natm_elem[i]],
				 desc_str->natom_elem[i], size_X3, xi_3, &highrank_ID_descriptors[i], N_low_min);
	}

	

	
	for (int i = 0; i < nelem; i++){
		mlff_str->natm_train_elemwise[i] = (highrank_ID_descriptors[i]).len;
	}


	

	count=0;
	for (int i = 0; i < nelem; i++){
		for (int j = 0; j < (highrank_ID_descriptors[i]).len; j++){
			mlff_str->natm_typ_train[count] = i;
			for(int jj = 0; jj < size_X3; jj++){
				mlff_str->X3_traindataset[count][jj] = X3_gathered_2D[cum_natm_elem[i]+(highrank_ID_descriptors[i]).array[j]][jj];
			}
			count++;
		}
	}





	initialize_descriptor(mlff_str->descriptor_strdataset, mlff_str, nlist);


	copy_descriptors(mlff_str->descriptor_strdataset, desc_str);



	mlff_str->n_str = 1;
	mlff_str->natm_train_total = count;
	if (rank==0){
		mlff_str->n_rows = 3*desc_str->natom_domain + 1 + mlff_str->stress_len;
	} else{
		mlff_str->n_rows = 3*desc_str->natom_domain;
	}
	

	mlff_str->n_cols = count;


	int *cum_natm_elem1 = (int *)malloc(sizeof(int)*nelem);
	cum_natm_elem1[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem1[i] = cum_natm_elem1[i-1] + mlff_str->natm_train_elemwise[i-1];
	}


	double *K_train_local = (double *) malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len)); // row major;
	for (int i=0; i < mlff_str->n_cols*(3*natom+1+mlff_str->stress_len); i++){
		K_train_local[i] = 0.0;
	}


	double **kernel_ij = (double **) malloc(sizeof(double*)*desc_str->natom_domain);
	for (int i = 0; i < desc_str->natom_domain; i++){
		kernel_ij[i] = (double *) malloc(sizeof(double)*mlff_str->n_cols);
		for (int j = 0; j < mlff_str->n_cols; j++){
			kernel_ij[i][j] = 0.0;
		}
	}

	double ***zit = (double ***) malloc(sizeof(double**)*desc_str->natom_domain);
	for (int i = 0; i < desc_str->natom_domain; i++){
		zit[i] = (double **) malloc(sizeof(double*)*mlff_str->n_cols);
		for (int j = 0; j < mlff_str->n_cols; j++){
			zit[i][j] = (double *) malloc(sizeof(double)*mlff_str->size_X3);
			for (int k = 0; k < mlff_str->size_X3; k++){
				zit[i][j][k] = 0.0;
			}
		}
	}



	

	calculate_polynomial_kernel_ij(desc_str, mlff_str->X3_traindataset, mlff_str->natm_typ_train, mlff_str->n_cols, kernel_ij);
	calculate_zit(desc_str, mlff_str->X3_traindataset, mlff_str->natm_typ_train, mlff_str->n_cols, zit);




	// Energy term to populate K_train matrix
	int el_type;
	double temp = (1.0/ (double) desc_str->natom);
	for (int i=0; i < desc_str->natom_domain; i++){
		for (int j = 0; j < mlff_str->n_cols; j++){
			K_train_local[j] += temp * kernel_ij[i][j];
		}
	}
	
	// Force term to populate K_train matrix
	int atm_idx;
	for (int i=0; i < desc_str->natom_domain; i++){
		el_type = desc_str->el_idx_domain[i];
		atm_idx = desc_str->atom_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_elem1[el_type] + iatm_el_tr;
			row_idx = 3*atm_idx+1;

			K_train_local[row_idx*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dX[i][0], size_X3);
			K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dY[i][0], size_X3);
			K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dZ[i][0], size_X3);
			// printf("%f %.6E %.6E\n", xi_3, kernel_ij[i][col_idx], dotProduct(zit[i][col_idx], desc_str->dX3_dX[i][0], size_X3));
			// unique neighbor list never contains self image
			for (int neighs =0; neighs < desc_str->unique_Nneighbors[i]; neighs++){
				row_idx = 3*nlist->unique_neighborList[i].array[neighs]+1;

				K_train_local[row_idx*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dX[i][1+neighs], size_X3);
				K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dY[i][1+neighs], size_X3);
				K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dZ[i][1+neighs], size_X3);
				// printf("%f %.6E %.6E\n", xi_3, kernel_ij[i][col_idx], dotProduct(zit[i][col_idx], desc_str->dX3_dX[i][1+neighs], size_X3));
			}
			
		}
	}



	double volume = desc_str->cell_measure;

	if (mlff_str->mlff_pressure_train_flag == 0){
		for (int i=0; i < desc_str->natom_domain; i++){
			el_type = desc_str->el_idx_domain[i];
			atm_idx = desc_str->atom_idx_domain[i];
			for (int iatm_el_tr = 0; iatm_el_tr < (highrank_ID_descriptors[el_type]).len; iatm_el_tr++){
				col_idx = cum_natm_elem1[el_type] + iatm_el_tr;
				for (int istress = 0; istress < mlff_str->stress_len; istress++){
					row_idx = 3*desc_str->natom+1+istress;
					K_train_local[row_idx*mlff_str->n_cols + col_idx] += (1.0/volume)*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dF[i][istress], size_X3);
				}
			}
		}
	} else {
		for (int i=0; i < desc_str->natom_domain; i++){
			el_type = desc_str->el_idx_domain[i];
			atm_idx = desc_str->atom_idx_domain[i];
			for (int iatm_el_tr = 0; iatm_el_tr < (highrank_ID_descriptors[el_type]).len; iatm_el_tr++){
				col_idx = cum_natm_elem1[el_type] + iatm_el_tr;
				int idx_st[3]={0,3,5};
				for (int istress = 0; istress < 3; istress++){
					row_idx = 3*desc_str->natom+1+0;
					K_train_local[row_idx*mlff_str->n_cols + col_idx] -= (1.0/3.0)*(1.0/volume)*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dF[i][idx_st[istress]], size_X3);
				}
			}
		}
	}
	

	
	
	


	double *K_train_assembled;
	K_train_assembled = (double *)malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len));

	MPI_Allreduce(K_train_local, K_train_assembled, mlff_str->n_cols*(3*natom+1+mlff_str->stress_len), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	int r_idx;
	if (rank==0){
		for (int i=0; i < mlff_str->n_cols; i++){
			mlff_str->K_train[0][i] = K_train_assembled[i];
		}
		for (int istress =0; istress < mlff_str->stress_len; istress++){
			r_idx = 3*desc_str->natom+1+istress;
			for (int i=0; i < mlff_str->n_cols; i++){
				mlff_str->K_train[1+istress][i] = K_train_assembled[r_idx*mlff_str->n_cols + i];
			}
		}
	}

	
	
	for (int i = 0; i < desc_str->natom_domain; i++){
		atm_idx = desc_str->atom_idx_domain[i];
		r_idx = 3*atm_idx+1;

		for (int j=0; j < mlff_str->n_cols; j++){
			if (rank==0){
				mlff_str->K_train[3*i+1+mlff_str->stress_len][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				mlff_str->K_train[3*i+2+mlff_str->stress_len][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				mlff_str->K_train[3*i+3+mlff_str->stress_len][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			} else{
				mlff_str->K_train[3*i][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				mlff_str->K_train[3*i+1][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				mlff_str->K_train[3*i+2][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			}
			
		}
	}


	

	for (int i=0; i <nelem; i++){  
		delete_dyarray(&highrank_ID_descriptors[i]);
	} 
	free(highrank_ID_descriptors);
	free(cum_natm_elem);
	free(cum_natm_elem1);
	free(X3_local);
	free(X3_gathered);
	for (int i=0; i < natom; i++){
		free(X3_gathered_2D[i]);
	}
	free(X3_gathered_2D); 
	free(K_train_local); 
	free(K_train_assembled);

	for (int i = 0; i < desc_str->natom_domain; i++){
		free(kernel_ij[i]);
	}	
	free(kernel_ij);

	for (int i = 0; i < desc_str->natom_domain; i++){
		for (int j = 0; j < mlff_str->n_cols; j++){
			free(zit[i][j]);
		}
		free(zit[i]);
	}
	free(zit);


}

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

void add_newstr_rows(DescriptorObj *desc_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double *F, double *stress_sparc) {
	
	int row_idx, col_idx, atom_idx;
	int num_Fterms_newstr, iel, iatm_str;

	int  natom = desc_str->natom;
	int nelem = desc_str->nelem	;
	int size_X3 = desc_str->size_X3;
	double xi_3 = desc_str->xi_3;
	int kernel_typ = mlff_str->kernel_typ;

	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    mlff_str->E_store[mlff_str->E_store_counter] = E;
    for (int i = 0; i < mlff_str->stress_len; i++)
		mlff_str->stress_store[i][mlff_str->E_store_counter] = stress_sparc[i];
	for (int i=0; i < 3*natom; i++){
		mlff_str->F_store[mlff_str->F_store_counter+i] = F[i];
	}
	mlff_str->E_store_counter += 1;
	mlff_str->F_store_counter += 3*natom;

	if (mlff_str->n_str > 1) {
		mlff_str->mu_E  = get_mean(mlff_str->E_store, mlff_str->E_store_counter);
		mlff_str->std_E  = sqrt(get_variance(mlff_str->E_store, mlff_str->E_store_counter));
		mlff_str->std_F  = sqrt(get_variance(mlff_str->F_store, mlff_str->F_store_counter));
		for (int i = 0; i < mlff_str->stress_len; i++){
			mlff_str->std_stress[i] = sqrt(get_variance(mlff_str->stress_store[i], mlff_str->E_store_counter));
		}
	} else {
		mlff_str->mu_E  = mlff_str->E_store[0];
		mlff_str->std_E  = 1.0;
		mlff_str->std_F  = sqrt(get_variance(mlff_str->F_store, mlff_str->F_store_counter));
		for (int i = 0; i < mlff_str->stress_len; i++){
			mlff_str->std_stress[i] = 1.0;
		}
	}	
	
	int idx;
	if (rank==0){
		mlff_str->b_no_norm[mlff_str->n_rows] = E;
		for (int istress=0; istress < mlff_str->stress_len; istress++){
			mlff_str->b_no_norm[mlff_str->n_rows+1+istress] = stress_sparc[istress];
		}
		for (int i = 0; i < desc_str->natom_domain; i++){
			idx = mlff_str->atom_idx_domain[i];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+1+mlff_str->stress_len] = F[3*idx];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+2+mlff_str->stress_len] = F[3*idx+1];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+3+mlff_str->stress_len] = F[3*idx+2];
		}
	} else{
		for (int i = 0; i < desc_str->natom_domain; i++){
			idx = mlff_str->atom_idx_domain[i];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i] = F[3*idx];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+1] = F[3*idx+1];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+2] = F[3*idx+2];
		}
	}
	
	initialize_descriptor(mlff_str->descriptor_strdataset+mlff_str->n_str, mlff_str, nlist);
	copy_descriptors(mlff_str->descriptor_strdataset+mlff_str->n_str, desc_str);

	int *cum_natm_elem = (int *)malloc(sizeof(int)*nelem);
	cum_natm_elem[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem[i] = cum_natm_elem[i-1] + desc_str->natom_elem[i-1];
	}

	int **cum_natm_ele_cols = (int **)malloc(sizeof(int*)*nelem);
	for (int i = 0; i <nelem; i++){
		cum_natm_ele_cols[i] = (int *)malloc(sizeof(int)*mlff_str->natm_train_elemwise[i]);
	}

	for (int i = 0; i < nelem; i++){
		int count=0;
		for (int j=0; j < mlff_str->natm_train_total; j++){
			if (mlff_str->natm_typ_train[j] == i){
				cum_natm_ele_cols[i][count] = j;
				count++;
			}
		}
	}

	double temp = (1.0/ (double) natom);
	double *K_train_local = (double *) malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len)); // row major
	for (int i=0; i < mlff_str->n_cols*(3*natom+1+mlff_str->stress_len); i++){
		K_train_local[i] = 0.0;
	}


	double **kernel_ij = (double **) malloc(sizeof(double*)*desc_str->natom_domain);
	for (int i = 0; i < desc_str->natom_domain; i++){
		kernel_ij[i] = (double *) malloc(sizeof(double)*mlff_str->n_cols);
		for (int j = 0; j < mlff_str->n_cols; j++){
			kernel_ij[i][j] = 0.0;
		}
	}

	double ***zit = (double ***) malloc(sizeof(double**)*desc_str->natom_domain);
	for (int i = 0; i < desc_str->natom_domain; i++){
		zit[i] = (double **) malloc(sizeof(double*)*mlff_str->n_cols);
		for (int j = 0; j < mlff_str->n_cols; j++){
			zit[i][j] = (double *) malloc(sizeof(double)*mlff_str->size_X3);
			for (int k = 0; k < mlff_str->size_X3; k++){
				zit[i][j][k] = 0.0;
			}
		}
	}

	calculate_polynomial_kernel_ij(desc_str, mlff_str->X3_traindataset, mlff_str->natm_typ_train, mlff_str->n_cols, kernel_ij);
	calculate_zit(desc_str, mlff_str->X3_traindataset, mlff_str->natm_typ_train, mlff_str->n_cols, zit);


	int el_type;
	for (int i=0; i < desc_str->natom_domain; i++){
		el_type = desc_str->el_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			K_train_local[col_idx] += temp * kernel_ij[i][col_idx];
		}
	}

	int atm_idx;
	for (int i=0; i <desc_str->natom_domain; i++){
		el_type = desc_str->el_idx_domain[i];
		atm_idx = desc_str->atom_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			row_idx = 3*atm_idx+1;

			K_train_local[row_idx*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dX[i][0], size_X3);
			K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dY[i][0], size_X3);
			K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dZ[i][0], size_X3);

			for (int neighs =0; neighs < desc_str->unique_Nneighbors[i]; neighs++){
				row_idx = 3*nlist->unique_neighborList[i].array[neighs]+1;

				K_train_local[row_idx*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dX[i][1+neighs], size_X3);
				K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dY[i][1+neighs], size_X3);
				K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] += xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dZ[i][1+neighs], size_X3);
			
			}
			
		}
	}


	double volume = desc_str->cell_measure;
	
	if (mlff_str->mlff_pressure_train_flag == 0){
		for (int i=0; i < desc_str->natom_domain; i++){
			el_type = desc_str->el_idx_domain[i];
			atm_idx = desc_str->atom_idx_domain[i];
			for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
				col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
				for (int istress = 0; istress < mlff_str->stress_len; istress++){
						row_idx = 3*desc_str->natom+1+istress;
						K_train_local[row_idx*mlff_str->n_cols + col_idx] += (1.0/volume)*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dF[i][istress], size_X3);
				}
			}
		}
	} else {
		for (int i=0; i < desc_str->natom_domain; i++){
			el_type = desc_str->el_idx_domain[i];
			atm_idx = desc_str->atom_idx_domain[i];
			for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
				col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
				int idx_st[3]={0,3,5};
				for (int istress = 0; istress < 3; istress++){
						row_idx = 3*desc_str->natom+1+0;
						K_train_local[row_idx*mlff_str->n_cols + col_idx] -= (1.0/3.0)*(1.0/volume)*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dF[i][idx_st[istress]], size_X3);
				}
			}
		}
	}
	


	double *K_train_assembled;
	K_train_assembled = (double *)malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len));

	MPI_Allreduce(K_train_local, K_train_assembled, mlff_str->n_cols*(3*natom+1+mlff_str->stress_len), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	int r_idx;

	if (rank==0){
		for (int i=0; i < mlff_str->n_cols; i++){
			mlff_str->K_train[mlff_str->n_rows][i] = K_train_assembled[i];
		}
		for (int istress =0; istress < mlff_str->stress_len; istress++){
			r_idx = 3*desc_str->natom+1+istress;
			for (int i=0; i < mlff_str->n_cols; i++){
				mlff_str->K_train[mlff_str->n_rows+1+istress][i] = K_train_assembled[r_idx*mlff_str->n_cols + i];
			}
		}
	}

	
	for (int i = 0; i < desc_str->natom_domain; i++){
		atm_idx = desc_str->atom_idx_domain[i];
		r_idx = 3*atm_idx+1;

		for (int j=0; j < mlff_str->n_cols; j++){
			if (rank==0){
				mlff_str->K_train[mlff_str->n_rows+3*i+1+mlff_str->stress_len][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				mlff_str->K_train[mlff_str->n_rows+3*i+2+mlff_str->stress_len][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				mlff_str->K_train[mlff_str->n_rows+3*i+3+mlff_str->stress_len][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			} else{
				mlff_str->K_train[mlff_str->n_rows+3*i][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				mlff_str->K_train[mlff_str->n_rows+3*i+1][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				mlff_str->K_train[mlff_str->n_rows+3*i+2][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			}
			
		}
	}
	free(K_train_local);
	free(K_train_assembled);

	


	// updating other MLFF parameters such as number of structures, number of training environment, element typ of training env
	mlff_str->n_str += 1;
	if (rank==0){
		mlff_str->n_rows += 3*desc_str->natom_domain + 1 + mlff_str->stress_len;
	} else {
		mlff_str->n_rows += 3*desc_str->natom_domain;
	}
	free(cum_natm_elem);
	for (int i = 0; i <nelem; i++){
		free(cum_natm_ele_cols[i]);
	}
	free(cum_natm_ele_cols);
	// free(stress);

	for (int i = 0; i < desc_str->natom_domain; i++){
		free(kernel_ij[i]);
	}	
	free(kernel_ij);

	for (int i = 0; i < desc_str->natom_domain; i++){
		for (int j = 0; j < mlff_str->n_cols; j++){
			free(zit[i][j]);
		}
		free(zit[i]);
	}
	free(zit);
	// exit(13);

}


/*
calculate_Kpredict function calculate the design matrix for prediction for a new structure

[Input]
1. desc_str: DescriptorObj structure of the new structure
2. nlist: NeighList strcuture of the new structure
3. mlff_str: MLFF_Obj structure
[Output]
1. K_predict: design prediction matrix
*/
//TODO: Obtain the polynomial to do the evaluation of energy and forces
void calculate_Kpredict(DescriptorObj *desc_str, NeighList *nlist, MLFF_Obj *mlff_str, double **K_predict){

 	int row_idx, col_idx, atom_idx, iel, iatm_str, istress;
 	double E_scale, F_scale, *stress_scale;

 	int natom = desc_str->natom;
 	int nelem = desc_str->nelem;
	int size_X3 = desc_str->size_X3;
	double xi_3 = desc_str->xi_3;
	stress_scale = (double*) malloc(mlff_str->stress_len*sizeof(double));


	int rank, nproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

 	int kernel_typ = mlff_str->kernel_typ;

 	E_scale = mlff_str->E_scale;
 	F_scale = mlff_str->F_scale * mlff_str->relative_scale_F;

 	for (int i=0; i < mlff_str->stress_len; i++){
 		stress_scale[i] =  mlff_str->stress_scale[i] * mlff_str->relative_scale_stress[i];

 	}

 	int *cum_natm_elem = (int *)malloc(sizeof(int)*nelem);
	cum_natm_elem[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem[i] = cum_natm_elem[i-1] + desc_str->natom_elem[i-1];
	}

	int **cum_natm_ele_cols = (int **)malloc(sizeof(int*)*nelem);
	for (int i = 0; i <nelem; i++){
		cum_natm_ele_cols[i] = (int *)malloc(sizeof(int)*mlff_str->natm_train_elemwise[i]);
	}

	for (int i = 0; i < nelem; i++){
		int count=0;
		for (int j=0; j < mlff_str->natm_train_total; j++){
			if (mlff_str->natm_typ_train[j] == i){
				cum_natm_ele_cols[i][count] = j;
				count++;
			}
		}
	}

	double temp = (1.0/ (double) desc_str->natom);
	double *K_train_local = (double *) malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len)); // row major;
	for (int i=0; i < mlff_str->n_cols*(3*natom+1+mlff_str->stress_len); i++){
		K_train_local[i] = 0.0;
	}

	double **kernel_ij = (double **) malloc(sizeof(double*)*desc_str->natom_domain);
	for (int i = 0; i < desc_str->natom_domain; i++){
		kernel_ij[i] = (double *) malloc(sizeof(double)*mlff_str->n_cols);
		for (int j = 0; j < mlff_str->n_cols; j++){
			kernel_ij[i][j] = 0.0;
		}
	}

	double ***zit = (double ***) malloc(sizeof(double**)*desc_str->natom_domain);
	for (int i = 0; i < desc_str->natom_domain; i++){
		zit[i] = (double **) malloc(sizeof(double*)*mlff_str->n_cols);
		for (int j = 0; j < mlff_str->n_cols; j++){
			zit[i][j] = (double *) malloc(sizeof(double)*mlff_str->size_X3);
			for (int k = 0; k < mlff_str->size_X3; k++){
				zit[i][j][k] = 0.0;
			}
		}
	}

	calculate_polynomial_kernel_ij(desc_str, mlff_str->X3_traindataset, mlff_str->natm_typ_train, mlff_str->n_cols, kernel_ij);
	calculate_zit(desc_str, mlff_str->X3_traindataset, mlff_str->natm_typ_train, mlff_str->n_cols, zit);

	int el_type;
	for (int i=0; i <desc_str->natom_domain; i++){
		el_type = desc_str->el_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			K_train_local[col_idx] += temp * kernel_ij[i][col_idx];
		}
	}

	int atm_idx;
	for (int i=0; i <desc_str->natom_domain; i++){
		el_type = desc_str->el_idx_domain[i];
		atm_idx = desc_str->atom_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			row_idx = 3*atm_idx+1;

			K_train_local[row_idx*mlff_str->n_cols + col_idx] += F_scale*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dX[i][0], size_X3);
			K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] += F_scale*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dY[i][0], size_X3);
			K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] += F_scale*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dZ[i][0], size_X3);
			
			for (int neighs =0; neighs < desc_str->unique_Nneighbors[i]; neighs++){
				row_idx = 3*nlist->unique_neighborList[i].array[neighs]+1;

				K_train_local[row_idx*mlff_str->n_cols + col_idx] += F_scale*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dX[i][1+neighs], size_X3);
				K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] += F_scale*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dY[i][1+neighs], size_X3);
				K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] += F_scale*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dZ[i][1+neighs], size_X3);
			}
			
		}
	}

	
	double volume = desc_str->cell_measure;

	if (mlff_str->mlff_pressure_train_flag == 0){
		for (int i=0; i < desc_str->natom_domain; i++){
			el_type = desc_str->el_idx_domain[i];
			atm_idx = desc_str->atom_idx_domain[i];
			for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
				col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
				for (int istress = 0; istress < mlff_str->stress_len; istress++){
					row_idx = 3*desc_str->natom+1+istress;
					K_train_local[row_idx*mlff_str->n_cols + col_idx] += (1.0/volume)*stress_scale[istress]*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dF[i][istress], size_X3);
				}
			}
		}
	} else {
		for (int i=0; i < desc_str->natom_domain; i++){
			el_type = desc_str->el_idx_domain[i];
			atm_idx = desc_str->atom_idx_domain[i];
			for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
				col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
				int idx_st[3]={0,3,5};
				for (int istress = 0; istress <3; istress++){
					row_idx = 3*desc_str->natom+1+0;
					K_train_local[row_idx*mlff_str->n_cols + col_idx] -= (1.0/3.0)*(1.0/volume)*stress_scale[0]*xi_3*kernel_ij[i][col_idx]*dotProduct(zit[i][col_idx], desc_str->dX3_dF[i][idx_st[istress]], size_X3);
				}
			}
		}
	}
	

	


	double *K_train_assembled;
	K_train_assembled = (double *)malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len));


	MPI_Allreduce(K_train_local, K_train_assembled, mlff_str->n_cols*(3*natom+1+mlff_str->stress_len), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	int r_idx;
	if (rank==0){
		for (int i=0; i < mlff_str->n_cols; i++){
			K_predict[0][i] = K_train_assembled[i];
		}
		for (int istress =0; istress < mlff_str->stress_len; istress++){
			r_idx = 3*desc_str->natom+1+istress;
			for (int i=0; i < mlff_str->n_cols; i++){
				K_predict[1+istress][i] = K_train_assembled[r_idx*mlff_str->n_cols + i];
			}
		}
	}
	


	for (int i = 0; i < desc_str->natom_domain; i++){
		atm_idx = desc_str->atom_idx_domain[i];
		r_idx = 3*atm_idx+1;

		for (int j=0; j < mlff_str->n_cols; j++){
			if (rank==0){
				K_predict[3*i+1+mlff_str->stress_len][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				K_predict[3*i+2+mlff_str->stress_len][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				K_predict[3*i+3+mlff_str->stress_len][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			} else{
				K_predict[3*i][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				K_predict[3*i+1][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				K_predict[3*i+2][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			}
			
		}
	}


    free(K_train_local);
	free(K_train_assembled);

	free(cum_natm_elem);
	for (int i = 0; i <nelem; i++){
		free(cum_natm_ele_cols[i]);
	}
	free(cum_natm_ele_cols);
	free(stress_scale);

	for (int i = 0; i < desc_str->natom_domain; i++){
		free(kernel_ij[i]);
	}	
	free(kernel_ij);

	for (int i = 0; i < desc_str->natom_domain; i++){
		for (int j = 0; j < mlff_str->n_cols; j++){
			free(zit[i][j]);
		}
		free(zit[i]);
	}
	free(zit);

 }

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

 void add_newtrain_cols(double *X3, int elem_typ, MLFF_Obj *mlff_str){
	int row_idx, col_idx, atom_idx, istress;
	int nelem = mlff_str->nelem;
	int natom = mlff_str->natom;
	int size_X3 = mlff_str->size_X3;
	int kernel_typ = mlff_str->kernel_typ;
	double xi_3 = mlff_str->xi_3;

	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	// copy the X2 and X3 into train history data 


	for(int j = 0; j < size_X3; j++){
		mlff_str->X3_traindataset[mlff_str->n_cols][j] = X3[j];
	}


	double **X3_input;
	int *natm_typ_train = (int *) malloc(1*sizeof(int));
	X3_input = (double **) malloc(sizeof(double*)*1);
	X3_input[0] = (double *) malloc(sizeof(double)*size_X3);
	for (int i = 0; i < size_X3; i++){
		X3_input[0][i] = X3[i];
	}
	natm_typ_train[0] = elem_typ;


	double *k_local;
	int total_rows = 0;
	for (int istr = 0; istr < mlff_str->n_str; istr++){
		total_rows = total_rows + 1 + 3*(mlff_str->descriptor_strdataset+istr)->natom + mlff_str->stress_len;
	}
	k_local = (double *)malloc(sizeof(double)*total_rows);
	for (int i=0; i< total_rows; i++){
		k_local[i] = 0.0;
	}

	row_idx=0;
	int el_type;
	double temp;
	double **kernel_ij;
	for (int istr = 0; istr < mlff_str->n_str; istr++){

		kernel_ij = (double **) malloc(sizeof(double*)*mlff_str->natom_domain);
		for (int i = 0; i < mlff_str->natom_domain; i++){
			kernel_ij[i] = (double *) malloc(sizeof(double)*1);
			for (int j = 0; j < 1; j++){
				kernel_ij[i][j] = 0.0;
			}
		}
		calculate_polynomial_kernel_ij(mlff_str->descriptor_strdataset+istr, X3_input, natm_typ_train, 1, kernel_ij);

		temp = 1.0/ ((double) (mlff_str->descriptor_strdataset+istr)->natom);
		for (int i=0; i <(mlff_str->descriptor_strdataset+istr)->natom_domain; i++){
			el_type = (mlff_str->descriptor_strdataset+istr)->el_idx_domain[i];
			if (el_type==elem_typ){
				k_local[row_idx] +=  temp* kernel_ij[i][0];
			}
		}
		row_idx += 1+3*(mlff_str->descriptor_strdataset+istr)->natom+mlff_str->stress_len;


		for (int i = 0; i < mlff_str->natom_domain; i++){
			free(kernel_ij[i]);
		}	
		free(kernel_ij);
	}

	

	// Force term to populate K_train matrix
	row_idx = 0;
	int row_index_F;
	double ***zit;
	for (int istr = 0; istr < mlff_str->n_str; istr++){

		kernel_ij = (double **) malloc(sizeof(double*)*mlff_str->natom_domain);
		for (int i = 0; i < mlff_str->natom_domain; i++){
			kernel_ij[i] = (double *) malloc(sizeof(double)*1);
			for (int j = 0; j < 1; j++){
				kernel_ij[i][j] = 0.0;
			}
		}

		zit = (double ***) malloc(sizeof(double**)*mlff_str->natom_domain);
		for (int i = 0; i < mlff_str->natom_domain; i++){
			zit[i] = (double **) malloc(sizeof(double*)*1);
			for (int j = 0; j < 1; j++){
				zit[i][j] = (double *) malloc(sizeof(double)*mlff_str->size_X3);
				for (int k = 0; k < mlff_str->size_X3; k++){
					zit[i][j][k] = 0.0;
				}
			}
		}

		calculate_polynomial_kernel_ij(mlff_str->descriptor_strdataset+istr, X3_input, natm_typ_train, 1, kernel_ij);
		calculate_zit(mlff_str->descriptor_strdataset+istr, X3_input, natm_typ_train, 1, zit);


		for (int i = 0; i < (mlff_str->descriptor_strdataset+istr)->natom_domain; i++){
			el_type = (mlff_str->descriptor_strdataset+istr)->el_idx_domain[i];
			if (el_type==elem_typ){
				atom_idx = (mlff_str->descriptor_strdataset+istr)->atom_idx_domain[i];
				row_index_F = row_idx + 3*atom_idx+1;
				// x-component (w.r.t itself) "because an atom is not considered it's neighbour hence dealt outside the neighs loop"
				k_local[row_index_F] += xi_3*kernel_ij[i][0]*dotProduct(zit[i][0], (mlff_str->descriptor_strdataset+istr)->dX3_dX[i][0], size_X3); 
				// y-component (w.r.t itself) "because an atom is not considered it's neighbour hence dealt outside the neighs loop"
				k_local[row_index_F+1] +=  xi_3*kernel_ij[i][0]*dotProduct(zit[i][0], (mlff_str->descriptor_strdataset+istr)->dX3_dY[i][0], size_X3);
				// z-component (w.r.t itself) "because an atom is not considered it's neighbour hence dealt outside the neighs loop"
				k_local[row_index_F+2] +=  xi_3*kernel_ij[i][0]*dotProduct(zit[i][0], (mlff_str->descriptor_strdataset+istr)->dX3_dZ[i][0], size_X3);
				for (int neighs = 0; neighs < (mlff_str->descriptor_strdataset+istr)->unique_Nneighbors[i]; neighs++){
					row_index_F = row_idx + 3*(mlff_str->descriptor_strdataset+istr)->unique_neighborList[i].array[neighs]+1;
					k_local[row_index_F] +=  xi_3*kernel_ij[i][0]*dotProduct(zit[i][0], (mlff_str->descriptor_strdataset+istr)->dX3_dX[i][1+neighs], size_X3);
					// y-component (w.r.t neighs neighbour)
					k_local[row_index_F+1] += xi_3*kernel_ij[i][0]*dotProduct(zit[i][0], (mlff_str->descriptor_strdataset+istr)->dX3_dY[i][1+neighs], size_X3); 
					// z-component (w.r.t neighs neighbour)
					k_local[row_index_F+2] += xi_3*kernel_ij[i][0]*dotProduct(zit[i][0], (mlff_str->descriptor_strdataset+istr)->dX3_dZ[i][1+neighs], size_X3);
				}
			}
		}
		row_idx += 1+3*(mlff_str->descriptor_strdataset+istr)->natom+mlff_str->stress_len;

		for (int i = 0; i < mlff_str->natom_domain; i++){
			free(kernel_ij[i]);
		}	
		free(kernel_ij);
		for (int i = 0; i < mlff_str->natom_domain; i++){
			for (int j = 0; j < 1; j++){
				free(zit[i][j]);
			}
			free(zit[i]);
		}
		free(zit);
	}

	

	int row_index_stress;
	double volume;
	row_idx = 0;
	for (int istr = 0; istr < mlff_str->n_str; istr++){

		kernel_ij = (double **) malloc(sizeof(double*)*mlff_str->natom_domain);
		for (int i = 0; i < mlff_str->natom_domain; i++){
			kernel_ij[i] = (double *) malloc(sizeof(double)*1);
			for (int j = 0; j < 1; j++){
				kernel_ij[i][j] = 0.0;
			}
		}

		zit = (double ***) malloc(sizeof(double**)*mlff_str->natom_domain);
		for (int i = 0; i < mlff_str->natom_domain; i++){
			zit[i] = (double **) malloc(sizeof(double*)*1);
			for (int j = 0; j < 1; j++){
				zit[i][j] = (double *) malloc(sizeof(double)*mlff_str->size_X3);
				for (int k = 0; k < mlff_str->size_X3; k++){
					zit[i][j][k] = 0.0;
				}
			}
		}

		calculate_polynomial_kernel_ij(mlff_str->descriptor_strdataset+istr, X3_input, natm_typ_train, 1, kernel_ij);
		calculate_zit(mlff_str->descriptor_strdataset+istr, X3_input, natm_typ_train,1, zit);


		volume = (mlff_str->descriptor_strdataset+istr)->cell_measure;

		if (mlff_str->mlff_pressure_train_flag==0){
			for (int i = 0; i < (mlff_str->descriptor_strdataset+istr)->natom_domain; i++){
				el_type = (mlff_str->descriptor_strdataset+istr)->el_idx_domain[i];
				if (el_type==elem_typ){
					for (int istress = 0; istress < mlff_str->stress_len; istress++){
						row_index_stress = row_idx + 3*(mlff_str->descriptor_strdataset+istr)->natom + 1 + istress;

						k_local[row_index_stress] += (1.0/volume)* xi_3*kernel_ij[i][0]*dotProduct(zit[i][0], (mlff_str->descriptor_strdataset+istr)->dX3_dF[i][istress], size_X3);			
					}
				}
			}
		} else {
			for (int i = 0; i < (mlff_str->descriptor_strdataset+istr)->natom_domain; i++){
				el_type = (mlff_str->descriptor_strdataset+istr)->el_idx_domain[i];
				if (el_type==elem_typ){
					int idx_st[3]={0,3,5};
					for (int istress = 0; istress < 3; istress++){
						row_index_stress = row_idx + 3*(mlff_str->descriptor_strdataset+istr)->natom + 1 + 0;

						k_local[row_index_stress] -= (1.0/3.0)*(1.0/volume)* xi_3*kernel_ij[i][0]*dotProduct(zit[i][0], (mlff_str->descriptor_strdataset+istr)->dX3_dF[i][idx_st[istress]], size_X3);			
					}
				}
			}
		}
		
		row_idx += 1+3*(mlff_str->descriptor_strdataset+istr)->natom+mlff_str->stress_len;

		for (int i = 0; i < mlff_str->natom_domain; i++){
			free(kernel_ij[i]);
		}	
		free(kernel_ij);
		for (int i = 0; i < mlff_str->natom_domain; i++){
			for (int j = 0; j < 1; j++){
				free(zit[i][j]);
			}
			free(zit[i]);
		}
		free(zit);
	}


	double *K_train_assembled;
	K_train_assembled = (double *)malloc(sizeof(double)*total_rows);

	MPI_Allreduce(k_local, K_train_assembled, total_rows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	int temp_idx1 = 0, temp_idx2 = 0;
	if (rank==0){
		for (int i=0; i < mlff_str->n_str; i++){
			
			mlff_str->K_train[temp_idx1][mlff_str->n_cols] = K_train_assembled[temp_idx2];
			for (int istress =0; istress < mlff_str->stress_len; istress++){
				mlff_str->K_train[temp_idx1+1+istress][mlff_str->n_cols] = K_train_assembled[temp_idx2 + 3*((mlff_str->descriptor_strdataset+i)->natom)+1+istress];
			}
			temp_idx1 = temp_idx1 + 1 + mlff_str->stress_len + 3*((mlff_str->descriptor_strdataset+i)->natom_domain);
			temp_idx2 = temp_idx2 + 1 + mlff_str->stress_len + 3*((mlff_str->descriptor_strdataset+i)->natom);
		}
	}

	int r_idx, temp_idx, atm_idx;
	temp_idx=0;
	temp_idx1 = 0;
	for (int s=0; s < mlff_str->n_str; s++){
		for (int i = 0; i < (mlff_str->descriptor_strdataset+s)->natom_domain; i++){
			atm_idx = (mlff_str->descriptor_strdataset+s)->atom_idx_domain[i];
			r_idx = 3*atm_idx+1;
			if (rank==0){
				mlff_str->K_train[temp_idx+3*i+1+mlff_str->stress_len][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx];
				mlff_str->K_train[temp_idx+3*i+2+mlff_str->stress_len][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx+1];
				mlff_str->K_train[temp_idx+3*i+3+mlff_str->stress_len][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx+2];
			} else {
				mlff_str->K_train[temp_idx+3*i][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx];
				mlff_str->K_train[temp_idx+3*i+1][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx+1];
				mlff_str->K_train[temp_idx+3*i+2][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx+2];
			}
		}
		if (rank==0){
			temp_idx =  temp_idx + 1 + 3*((mlff_str->descriptor_strdataset+s)->natom_domain) +mlff_str->stress_len;
		} else {
			temp_idx =  temp_idx + 3*((mlff_str->descriptor_strdataset+s)->natom_domain);
		}
		temp_idx1 =  temp_idx1 + 1 + mlff_str->stress_len + 3*((mlff_str->descriptor_strdataset+s)->natom);
	}

	// updating other MLFF parameters such as number of structures, number of training environment, element typ of training env
	mlff_str->natm_typ_train[mlff_str->n_cols] = elem_typ;
	mlff_str->natm_train_total += 1;
	mlff_str->n_cols += 1;
	mlff_str->natm_train_elemwise[elem_typ] += 1;

	free(K_train_assembled); 
	free(k_local);




}

/*
remove_str_rows function removes a given reference structure from the training dataset

[Input]
1. mlff_str: MLFF_Obj structure
2. str_ID: ID of the reference structure in training dataset
[Output]
1. mlff_str: MLFF_Obj structure
*/

void remove_str_rows(MLFF_Obj *mlff_str, int str_ID){
	int start_idx = 0, end_idx = 0, i, j, istr, rows_to_delete;

	for (i = 0; i < str_ID; i++){
		start_idx += 3*(mlff_str->descriptor_strdataset+i)->natom + 1;
	}
	end_idx = start_idx + 3*(mlff_str->descriptor_strdataset+str_ID)->natom + 1;

	rows_to_delete = end_idx - start_idx;
	

	for (istr = str_ID + 1; istr < mlff_str->n_str; istr++){
		copy_descriptors(mlff_str->descriptor_strdataset + istr - 1, mlff_str->descriptor_strdataset + istr );
	}
	delete_descriptor(mlff_str->descriptor_strdataset + mlff_str->n_str);
	mlff_str->n_str = mlff_str->n_str - 1;

	for (i = start_idx; i < end_idx; i++) {
		mlff_str->b_no_norm[i] = mlff_str->b_no_norm[i + rows_to_delete];
		for (j = 0; j < mlff_str->n_cols; j++){
			mlff_str->K_train[i][j] = mlff_str->K_train[i + rows_to_delete][j];
		}
	}

	for (i = mlff_str->n_rows - rows_to_delete; i < mlff_str->n_rows; i++) {
		mlff_str->b_no_norm[i] = 0.0;
		for (j = 0; j < mlff_str->n_cols; j++){
			mlff_str->K_train[i][j] = 0.0;
		}
	}
	mlff_str->n_rows = mlff_str->n_rows - rows_to_delete;
}

/*
remove_train_cols function removes a given local confiugration from the training dataset

[Input]
1. mlff_str: MLFF_Obj structure
2. col_ID: ID of the local confiugration in training dataset
[Output]
1. mlff_str: MLFF_Obj structure
*/
void remove_train_cols(MLFF_Obj *mlff_str, int col_ID){
	int i, j;
	for (i = col_ID; i < mlff_str->n_cols-1; i++){
		for (j = 0; j < mlff_str->n_rows; j++){
			mlff_str->K_train[j][i] = mlff_str->K_train[j][i+1];
		}
	}

	for (j = 0; j < mlff_str->n_rows; j++){
		mlff_str->K_train[j][mlff_str->n_cols-1] = 0.0;
	}

	for (i =col_ID; i < mlff_str->n_cols-1; i++){
		for (j=0; j < mlff_str->size_X3; j++){
			mlff_str->X3_traindataset[i][j] = mlff_str->X3_traindataset[i+1][j];
		}
	}

	for (j=0; j < mlff_str->size_X3; j++){
		mlff_str->X3_traindataset[mlff_str->n_cols-1][j] = 0.0;
	}


	int atom_typ = mlff_str->natm_typ_train[col_ID];
	mlff_str->natm_train_elemwise[atom_typ] = mlff_str->natm_train_elemwise[atom_typ] -1;
	mlff_str->natm_train_total = mlff_str->natm_train_total -1;

	for (i =col_ID; i < mlff_str->n_cols-1; i++){
		mlff_str->natm_typ_train[i] = mlff_str->natm_typ_train[i+1];
	}
	
	mlff_str->n_cols = mlff_str->n_cols - 1;
}


/*
get_N_r_hnl function calculates the number of grid points in hnl_file

[Input]
1. pSPARC: SPARC structure
[Output]
1. pSPARC: SPARC structure
*/
void get_N_r_hnl(SPARC_OBJ *pSPARC){
	int i, j, info;
	FILE *fp;
	char line[512];
	char a1[512], a2[512], a3[512], a4[512];
	int count1=0, count2=0;

	// fp = fopen("hnl.txt","r");
	fp = fopen(pSPARC->hnl_file_name,"r");

	fgets(line, sizeof (line), fp);
	sscanf(line, "%s%s%s%s", a1, a2, a3, a4);
	int N_r = atoi(a4);
	pSPARC->N_rgrid_MLFF = N_r;
	fclose(fp);
}

// The functions below are outdated. 
// /*
// mlff_kernel function computes the SOAP kernel between two descriptors {X2_str, X3_str} and {X2_tr, X3_tr}

// [Input]
// 1. X2_str: pointer to the first descriptor (2-body)
// 2. X3_str: pointer to the first descriptor (3-body)
// 3. X2_tr: pointer to the second descriptor (2-body)
// 4. X3_tr: pointer to the second descriptor (3-body)
// 5. beta_2: weight to the 2-body term in the kernel
// 6. beta_3: weight to the 3-body term in the kernel
// 7. xi_3: exponent in the kernel
// 8. size_X2: length of the 2-body kernel
// 9. size_X3: length of the 3-body kernel
// [Output]
// 1. kernel_val: Value of the kernel
// */

// double mlff_kernel(int kernel_typ, double **X2_str, double **X3_str, double **X2_tr, double **X3_tr, int atom, int atom_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {
// 	double kernel_val;
// 	if (size_X3 > 0){
// 		if (kernel_typ ==0){// polynomial kernel implemented in VASP MLFF scheme
// 			kernel_val = soap_kernel_polynomial(X2_str[atom], X3_str[atom], X2_tr[atom_tr], X3_tr[atom_tr], beta_2, beta_3, xi_3, size_X2, size_X3);
// 		} else if (kernel_typ==1){//  Gaussiam Kernel
// 			kernel_val = soap_kernel_Gaussian(X2_str[atom], X3_str[atom], X2_tr[atom_tr], X3_tr[atom_tr], beta_2, beta_3, xi_3, size_X2, size_X3);
// 		} else{// Laplacian Kernel	
// 			kernel_val = soap_kernel_Laplacian(X2_str[atom], X3_str[atom], X2_tr[atom_tr], X3_tr[atom_tr], beta_2, beta_3, xi_3, size_X2, size_X3);
// 		} 	
// 		if (kernel_val>1.0+1e-5){
// 			printf("Error in soap kernel evaluation(>1) error %f\n", kernel_val);
// 			exit(1);
// 		}
// 	} else {
// 		if (kernel_typ ==0){// polynomial kernel implemented in VASP MLFF scheme
// 			kernel_val = GMP_kernel_polynomial(X2_str[atom],  X2_tr[atom_tr],  xi_3, size_X2);
// 		} else if (kernel_typ==1){//  Gaussiam Kernel
// 			kernel_val = GMP_kernel_Gaussian(X2_str[atom],  X2_tr[atom_tr],  xi_3, size_X2);
// 		} else{// Laplacian Kernel	
// 			printf("Laplacian kernel not implemented with GMP\n");
// 			exit(1);
// 		} 	
// 		if (kernel_val>1.0+1e-5){
// 			printf("Error in soap kernel evaluation(>1) error %f\n", kernel_val);
// 			exit(1);
// 		}
// 	}
// 	return kernel_val; 

// }

// double der_mlff_kernel(int kernel_typ, double ***dX2_str, double ***dX3_str, double **X2_str, double **X3_str, double **X2_tr, double **X3_tr, int atom, int atom_tr, int neighbor,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {
// 	if (size_X3 > 0){
// 		if (kernel_typ ==0)
// 			return der_soap_kernel_polynomial(dX2_str[atom][neighbor], dX3_str[atom][neighbor], X2_str[atom], X3_str[atom], X2_tr[atom_tr], X3_tr[atom_tr], beta_2, beta_3, xi_3, size_X2, size_X3);
// 		else if (kernel_typ==1)
// 			return der_soap_kernel_Gaussian(dX2_str[atom][neighbor], dX3_str[atom][neighbor], X2_str[atom], X3_str[atom], X2_tr[atom_tr], X3_tr[atom_tr], beta_2, beta_3, xi_3, size_X2, size_X3);
// 		else
// 			return der_soap_kernel_Laplacian(dX2_str[atom][neighbor], dX3_str[atom][neighbor], X2_str[atom], X3_str[atom], X2_tr[atom_tr], X3_tr[atom_tr], beta_2, beta_3, xi_3, size_X2, size_X3);
// 	} else {
// 		if (kernel_typ ==0)
// 			return der_GMP_kernel_polynomial(dX2_str[atom][neighbor], X2_str[atom], X2_tr[atom_tr], xi_3, size_X2);
// 		else if (kernel_typ==1)
// 			return der_GMP_kernel_Gaussian(dX2_str[atom][neighbor], X2_str[atom], X2_tr[atom_tr], xi_3, size_X2);
// 		else{
// 			printf("Laplacian kernel not implemented with GMP\n");
// 			exit(1);
// 		}
// 	}
// }

// double soap_kernel_polynomial(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

// 	double norm_X3_str, norm_X3_tr, X3_str_temp[size_X3], X3_tr_temp[size_X3], kernel_val;

// 	norm_X3_str = sqrt(dotProduct(X3_str, X3_str, size_X3));
// 	norm_X3_tr = sqrt(dotProduct(X3_tr, X3_tr, size_X3));
// 	for (int i = 0; i<size_X3; i++){
// 		X3_str_temp[i] = X3_str[i]/norm_X3_str;
// 		X3_tr_temp[i] = X3_tr[i]/norm_X3_tr;
// 	}

// 	kernel_val = beta_2 * dotProduct(X2_str, X2_tr, size_X2) + beta_3 * pow(dotProduct(X3_tr_temp, X3_str_temp, size_X3), xi_3);
// 	return kernel_val;
// }

// double soap_kernel_Gaussian(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

// 	double norm_X3_str, norm_X3_tr, X3_str_temp[size_X3], X3_tr_temp[size_X3], kernel_val;
// 	int i;
// 	norm_X3_str = sqrt(dotProduct(X3_str, X3_str, size_X3));
// 	norm_X3_tr = sqrt(dotProduct(X3_tr, X3_tr, size_X3));
// 	for (i = 0; i<size_X3; i++){
// 		X3_str_temp[i] = X3_str[i]/norm_X3_str;
// 		X3_tr_temp[i] = X3_tr[i]/norm_X3_tr;
// 	}
// 	double er_X2[size_X2], er_X3[size_X3];
// 	for (i=0; i <size_X2; i++)
// 		er_X2[i] = X2_str[i]-X2_tr[i];
// 	for (i=0; i <size_X3; i++)
// 		er_X3[i] = X3_str_temp[i]-X3_tr_temp[i];
// 	kernel_val = beta_2 * exp(-0.5*dotProduct(er_X2, er_X2, size_X2)) + beta_3 * exp(-0.5*dotProduct(er_X3, er_X3, size_X3));
// 	// printf("Kernel Gausian: %f\n",kernel_val);
// 	// kernel_val = beta_2 * dotProduct(X2_str, X2_tr, size_X2) + beta_3 * pow(dotProduct(X3_tr_temp, X3_str_temp, size_X3), xi_3);
// 	return kernel_val;
// }

// double soap_kernel_Laplacian(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

// 	double norm_X3_str, norm_X3_tr, X3_str_temp[size_X3], X3_tr_temp[size_X3], kernel_val;
// 	int i;

// 	double er_X2[size_X2], er_X3[size_X3];
// 	for (i=0; i <size_X2; i++)
// 		er_X2[i] = X2_str[i]-X2_tr[i];
// 	for (i=0; i <size_X3; i++)
// 		er_X3[i] = X3_str[i]-X3_tr[i];
// 	kernel_val = beta_2 * exp(-0.5*sqrt(dotProduct(er_X2, er_X2, size_X2))) + beta_3 * exp(-0.5*sqrt(dotProduct(er_X3, er_X3, size_X3)));
// 	// kernel_val = beta_2 * dotProduct(X2_str, X2_tr, size_X2) + beta_3 * pow(dotProduct(X3_tr_temp, X3_str_temp, size_X3), xi_3);
// 	return kernel_val;
// }

// /*
// der_soap_kernel function computes the derivative of the kernel w.r.t to some variable

// [Input]
// 1. dX2_str: pointer to the derivative of first descriptor w.r.t to the given variable (2-body)
// 2. dX3_str: pointer to the derivative of first descriptor w.r.t to the given variable (3-body)
// 3. X2_str: pointer to the first descriptor (2-body)
// 4. X3_str: pointer to the first descriptor (3-body)
// 5. X2_tr: pointer to the second descriptor (2-body)
// 6. X3_tr: pointer to the second descriptor (3-body)
// 7. beta_2: weight to the 2-body term in the kernel
// 8. beta_3: weight to the 3-body term in the kernel
// 9. xi_3: exponent in the kernel
// 10. size_X2: length of the 2-body kernel
// 11. size_X3: length of the 3-body kernel
// [Output]
// 1. der_val: derivative of the kernel
// */

// double der_soap_kernel_polynomial(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

// 	double norm_X3_str, norm_X3_tr, const1, der_val, const2, X3_str_temp[size_X3], X3_tr_temp[size_X3], temp, temp1, temp0;

// 	norm_X3_str = sqrt(dotProduct(X3_str, X3_str, size_X3));
// 	norm_X3_tr = sqrt(dotProduct(X3_tr, X3_tr, size_X3));

// 	for (int i = 0; i<size_X3; i++){
// 		X3_str_temp[i] = X3_str[i]/norm_X3_str;
// 		X3_tr_temp[i] = X3_tr[i]/norm_X3_tr;
// 	}

// 	temp = pow(norm_X3_str, -1-xi_3);

// 	temp0 = dotProduct(X3_str, X3_tr_temp, size_X3);

// 	temp1 = pow(temp0, xi_3-1);

// 	const1 = -1.0 * beta_3* xi_3 * temp * temp1*temp0;
// 	const2 = beta_3 * temp * norm_X3_str * xi_3 * temp1;

// 	der_val = beta_2 * dotProduct(X2_tr, dX2_str, size_X2) + 
// 			const1*dotProduct(X3_str_temp, dX3_str, size_X3) + const2*dotProduct(X3_tr_temp, dX3_str, size_X3);
	
// 	return der_val;
// }

// double der_soap_kernel_Gaussian(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

// 	double norm_X3_str, norm_X3_tr, const1, der_val, const2, X3_str_temp[size_X3], X3_tr_temp[size_X3], temp, temp1, temp0;

// 	double temp2[size_X2], temp3[size_X3];

// 	double er_X2[size_X2], er_X3[size_X3];

// 	norm_X3_str = sqrt(dotProduct(X3_str, X3_str, size_X3));
// 	norm_X3_tr = sqrt(dotProduct(X3_tr, X3_tr, size_X3));

// 	double norm_X3_str2 = norm_X3_str*norm_X3_str, norm_X3_str3 = norm_X3_str*norm_X3_str*norm_X3_str;

// 	for (int i = 0; i<size_X3; i++){
// 		X3_str_temp[i] = X3_str[i]/norm_X3_str;
// 		X3_tr_temp[i] = X3_tr[i]/norm_X3_tr;
// 	}

// 	for (int i=0; i <size_X2; i++)
// 		er_X2[i] = X2_str[i]-X2_tr[i];
// 	for (int i=0; i <size_X3; i++)
// 		er_X3[i] = X3_str_temp[i]-X3_tr_temp[i];

// 	double exp_temp3 = exp(-0.5*dotProduct(er_X3, er_X3, size_X3));
// 	double exp_temp2 = exp(-0.5*dotProduct(er_X2, er_X2, size_X2));
// 	temp = dotProduct(X3_str_temp,X3_tr_temp,size_X3);

// 	for (int i=0; i <size_X2; i++)
// 		temp2[i] = -(X2_str[i]-X2_tr[i])*exp_temp2;
// 	for (int i=0; i <size_X3; i++){
// 		temp3[i] = -exp_temp3*(- X3_tr_temp[i] + X3_str_temp[i]*temp)*(1.0/norm_X3_str);
// 	}
// 		// temp3[i] = -(X3_str[i]-X3_tr[i])*exp_temp3;

// 	der_val = beta_2 * dotProduct(temp2, dX2_str, size_X2) + beta_3 * dotProduct(temp3, dX3_str, size_X3);


// 	// temp = pow(norm_X3_str, -1-xi_3);

// 	// temp0 = dotProduct(X3_str, X3_tr_temp, size_X3);

// 	// temp1 = pow(temp0, xi_3-1);

// 	// const1 = -1 * beta_3* xi_3 * temp * temp1*temp0;
// 	// const2 = beta_3 * temp * norm_X3_str * xi_3 * temp1;

// 	// der_val = beta_2 * dotProduct(X2_tr, dX2_str, size_X2) + 
// 	// 		const1*dotProduct(X3_str_temp, dX3_str, size_X3) + const2*dotProduct(X3_tr_temp, dX3_str, size_X3);
	
// 	return der_val;
// }



// double der_soap_kernel_Laplacian(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
// 			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

// 	double norm_X3_str, norm_X3_tr, const1, der_val, const2, X3_str_temp[size_X3], X3_tr_temp[size_X3], temp, temp1, temp0;


// 	double temp2[size_X2], temp3[size_X3];
// 	double er_X2[size_X2], er_X3[size_X3];
// 	for (int i=0; i <size_X2; i++)
// 		er_X2[i] = X2_str[i]-X2_tr[i];
// 	for (int i=0; i <size_X3; i++)
// 		er_X3[i] = X3_str[i]-X3_tr[i];

// 	double exp_temp2 = exp(-0.5*sqrt(dotProduct(er_X2, er_X2, size_X2)));
// 	double exp_temp3 = exp(-0.5*sqrt(dotProduct(er_X3, er_X3, size_X3)));

// 	for (int i=0; i <size_X2; i++){
// 		if ((X2_str[i]-X2_tr[i])>0)
// 			temp = -0.5;
// 		else if ((X2_str[i]-X2_tr[i])<0)
// 			temp = 0.5;
// 		else
// 			temp = 0.0;
// 		temp2[i] = temp*exp_temp2;
// 	}
// 	for (int i=0; i <size_X3; i++){
// 		if ((X3_str[i]-X3_tr[i])>0)
// 			temp = -0.5;
// 		else if ((X3_str[i]-X3_tr[i])<0)
// 			temp = 0.5;
// 		else
// 			temp = 0.0;
// 		temp3[i] = temp*exp_temp3;
// 	}

// 	der_val = beta_2 * dotProduct(temp2, dX2_str, size_X2) + beta_3 * dotProduct(temp3, dX3_str, size_X3);



// 	// temp = pow(norm_X3_str, -1-xi_3);

// 	// temp0 = dotProduct(X3_str, X3_tr_temp, size_X3);

// 	// temp1 = pow(temp0, xi_3-1);

// 	// const1 = -1 * beta_3* xi_3 * temp * temp1*temp0;
// 	// const2 = beta_3 * temp * norm_X3_str * xi_3 * temp1;

// 	// der_val = beta_2 * dotProduct(X2_tr, dX2_str, size_X2) + 
// 	// 		const1*dotProduct(X3_str_temp, dX3_str, size_X3) + const2*dotProduct(X3_tr_temp, dX3_str, size_X3);
	
// 	return der_val;
// }

// double GMP_kernel_polynomial(double *X2_str, double *X2_tr,
// 			  double xi_3, int size_X2) {

// 	double norm_X2_str, norm_X2_tr, X2_str_temp[size_X2], X2_tr_temp[size_X2], kernel_val;
// 	int i;
// 	norm_X2_str = sqrt(dotProduct(X2_str, X2_str, size_X2));
// 	norm_X2_tr = sqrt(dotProduct(X2_tr, X2_tr, size_X2));
// 	for (i = 0; i<size_X2; i++){
// 		X2_str_temp[i] = X2_str[i]/norm_X2_str;
// 		X2_tr_temp[i] = X2_tr[i]/norm_X2_tr;
// 	}

// 	kernel_val = pow(dotProduct(X2_tr_temp, X2_str_temp, size_X2), xi_3);
// 	return kernel_val;
// }

// double GMP_kernel_Gaussian(double *X2_str, double *X2_tr,
// 			  double xi_3, int size_X2) {

// 	double norm_X2_str, norm_X2_tr, X2_str_temp[size_X2], X2_tr_temp[size_X2], kernel_val;
// 	int i;
// 	norm_X2_str = sqrt(dotProduct(X2_str, X2_str, size_X2));
// 	norm_X2_tr = sqrt(dotProduct(X2_tr, X2_tr, size_X2));
// 	for (i = 0; i<size_X2; i++){
// 		X2_str_temp[i] = X2_str[i]/norm_X2_str;
// 		X2_tr_temp[i] = X2_tr[i]/norm_X2_tr;
// 	}
// 	double er_X[size_X2];
// 	for (i=0; i <size_X2; i++)
// 		er_X[i] = X2_str[i]-X2_tr[i];

// 	kernel_val = exp(-0.5*dotProduct(er_X, er_X, size_X2)) ;
// 	// printf("Kernel Gausian: %f\n",kernel_val);
// 	// kernel_val = beta_2 * dotProduct(X2_str, X2_tr, size_X2) + beta_3 * pow(dotProduct(X3_tr_temp, X3_str_temp, size_X3), xi_3);
// 	return kernel_val;
// }

// double der_GMP_kernel_polynomial(double *dX2_str, double *X2_str, double *X2_tr,
// 			double xi_3, int size_X2) {

// 	double norm_X2_str, norm_X2_tr, const1, der_val, const2, X2_str_temp[size_X2], X2_tr_temp[size_X2], temp, temp1, temp0;

// 	norm_X2_str = sqrt(dotProduct(X2_str, X2_str, size_X2));
// 	norm_X2_tr = sqrt(dotProduct(X2_tr, X2_tr, size_X2));

// 	for (int i = 0; i<size_X2; i++){
// 		X2_str_temp[i] = X2_str[i]/norm_X2_str;
// 		X2_tr_temp[i] = X2_tr[i]/norm_X2_tr;
// 	}

// 	temp = pow(norm_X2_str, -1-xi_3);

// 	temp0 = dotProduct(X2_str, X2_tr_temp, size_X2);

// 	temp1 = pow(temp0, xi_3-1);
// 	int beta_3 = 1.0;
// 	const1 = -1 * beta_3* xi_3 * temp * temp1*temp0;
// 	const2 = beta_3 * temp * norm_X2_str * xi_3 * temp1;

// 	der_val =  const1*dotProduct(X2_str_temp, dX2_str, size_X2) + const2*dotProduct(X2_tr_temp, dX2_str, size_X2);
	
// 	return der_val;
// }

// double der_GMP_kernel_Gaussian(double *dX2_str, double *X2_str, double *X2_tr,
// 			 double xi_3, int size_X2) {

// 	double norm_X2_str, norm_X2_tr, const1, der_val, const2, X2_str_temp[size_X2], X2_tr_temp[size_X2], temp, temp1, temp0;

// 	double temp2[size_X2], temp3[size_X2];

// 	double er_X[size_X2];

// 	norm_X2_str = sqrt(dotProduct(X2_str, X2_str, size_X2));
// 	norm_X2_tr = sqrt(dotProduct(X2_tr, X2_tr, size_X2));

// 	double norm_X2_str2 = norm_X2_str*norm_X2_str, norm_X2_str3 = norm_X2_str*norm_X2_str*norm_X2_str;

// 	for (int i = 0; i<size_X2; i++){
// 		X2_str_temp[i] = X2_str[i]/norm_X2_str;
// 		X2_tr_temp[i] = X2_tr[i]/norm_X2_tr;
// 	}


// 	for (int i=0; i <size_X2; i++)
// 		er_X[i] = X2_str_temp[i]-X2_tr_temp[i];

// 	double exp_temp = exp(-0.5*dotProduct(er_X, er_X, size_X2));

// 	temp = dotProduct(X2_str_temp,X2_tr_temp,size_X2);


// 	for (int i=0; i <size_X2; i++){
// 		temp3[i] = -exp_temp*(- X2_tr_temp[i] + X2_str_temp[i]*temp)*(1.0/norm_X2_str);
// 	}
// 		// temp3[i] = -(X3_str[i]-X3_tr[i])*exp_temp3;

// 	der_val = dotProduct(temp3, dX2_str, size_X2);


// 	// temp = pow(norm_X3_str, -1-xi_3);

// 	// temp0 = dotProduct(X3_str, X3_tr_temp, size_X3);

// 	// temp1 = pow(temp0, xi_3-1);

// 	// const1 = -1 * beta_3* xi_3 * temp * temp1*temp0;
// 	// const2 = beta_3 * temp * norm_X3_str * xi_3 * temp1;

// 	// der_val = beta_2 * dotProduct(X2_tr, dX2_str, size_X2) + 
// 	// 		const1*dotProduct(X3_str_temp, dX3_str, size_X3) + const2*dotProduct(X3_tr_temp, dX3_str, size_X3);
	
// 	return der_val;
// }