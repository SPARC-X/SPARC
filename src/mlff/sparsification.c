#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
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
#include "isddft.h"
#include "tools.h"
#include "ddbp_tools.h"
#include "sparsification.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/*
SOAP_CUR_sparsify function computes the IDs of local descriptors which can be removed from the training dataset
 ** All the descriptors here corresponds to same element type
[Input]
1. X2: 2-body local descriptors in the training dataset
2. X3: 2-body local descriptors in the training dataset
3. n_descriptor: number of descriptors in the training datset
4. size_X2: length of 2-body descriptors
5. size_X2: length of 3-body descriptors
6. beta_2, beta_3, xi_3: parameters in SOAP kernel
[Output]
1. K: Gram matrix
2. highrank_ID_descriptors: ID's of high rank descriptors
*/

void mlff_CUR_sparsify(int kernel_typ, double **X3, int n_descriptor, int size_X3, double xi_3, dyArray *highrank_ID_descriptors, int N_low_min){
	
	int count=0, info, i, j;
	double w[n_descriptor], w1[n_descriptor], *K;
	K = (double *) malloc(n_descriptor*n_descriptor*sizeof(double));
	for (i = 0; i < n_descriptor; i++){
		w1[i] = 0.0;
		for (j = 0; j < n_descriptor; j++){
			K[count] = 0.0;
			count++;
		}
	}

	for (int i=0; i < n_descriptor; i++){
		for (int j = 0; j < n_descriptor; j++){
				K[i*n_descriptor + j] = mlff_kernel_eval(kernel_typ, X3[i], X3[j], xi_3, size_X3);
		}
	}


	// FILE* fid;
	// fid = fopen("Ktemp.txt","w");
	// for (int i=0; i < n_descriptor; i++){
	// 	for (int j = 0; j < n_descriptor; j++){
	// 			fprintf(fid,"%10.9f ",K[i*n_descriptor + j]);
	// 	}
	// 	fprintf(fid,"\n");
	// }
	// fclose(fid);
	// exit(3);



	info = LAPACKE_dsyevd( LAPACK_ROW_MAJOR, 'V', 'U', n_descriptor, K, n_descriptor, w );
    /* Check for convergence */
    if( info > 0 ) {
            printf( "LAPACKE_dsyev in SOAP_gram_matrix failed to compute eigenvalues.\n" );
            exit( 1 );
    }
    int N_low =0;
    for (i = 0; i < n_descriptor; i++){
    	if (w[i] < 0.0000000001){
    		N_low += 1;
    		// append_dyarray(highrank_ID_descriptors, i);
    	}
    }

    N_low = max(N_low_min, N_low);

    for (int i= 0; i < n_descriptor; i++){
    	for (int j = 0; j < n_descriptor; j++){
    		if (w[j] < 0.0000000001){
    			w1[i] += (1.0/N_low) * K[i*n_descriptor+j] * K[i*n_descriptor+j];
    		}
    	}

    }
    double temp;
    int temp_idx;


    for (int i = 0; i < n_descriptor - N_low; i++){
    	temp = smallest(w1, n_descriptor);
    	temp_idx = lin_search_double(w1, n_descriptor, temp);
    	w1[temp_idx] = 10000000000;
    	append_dyarray(highrank_ID_descriptors, temp_idx);
    	// printf("N_low: %d, n_descriptor: %d, highrank_ID_descriptors->len: %d\n",N_low,n_descriptor,highrank_ID_descriptors->len);
    }
    free(K);

}


double mlff_kernel_eval(int kernel_typ, double *X3_i, double *X3_j, double xi_3, int size_X3){
	double kernel_val = 0.0;

	double X3_i_unit[size_X3], X3_j_unit[size_X3];

	double norm_X3_i=0.0, norm_X3_j=0.0;

	for (int i = 0; i < size_X3; i++){
		norm_X3_i += X3_i[i]*X3_i[i];
		norm_X3_j += X3_j[i]*X3_j[i];
	}
	norm_X3_i = sqrt(norm_X3_i);
	norm_X3_j = sqrt(norm_X3_j);


	for (int i = 0; i < size_X3; i++){
		X3_i_unit[i] = X3_i[i]/norm_X3_i;
		X3_j_unit[i] = X3_j[i]/norm_X3_j;
	}

	for (int i = 0; i < size_X3; i++){
		kernel_val += X3_i_unit[i]*X3_j_unit[i];
	}

	kernel_val = pow(kernel_val, xi_3);

	return kernel_val;

}