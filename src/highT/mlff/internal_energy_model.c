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


#include "mlff_types.h"
#include "isddft.h"
#include "covariance_matrix.h"
#include "internal_energy_model.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
/*
Obtain the weights for the internal energy linear regression model
fitting y = alpha + beta x
*/

void train_internal_energy_model(MLFF_Obj *mlff_str){

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);


    double alpha, beta;

    double sum_y=0.0;
    double sum_x_square = 0.0;
    double sum_y_square = 0.0;
    double sum_x = 0.0;
    double sum_xy = 0.0;


    int N =  mlff_str->internal_energy_DFT_count;

    for (int i = 0; i < N; i++){
        sum_y += mlff_str->internal_energy_DFT[i];
        sum_x_square += mlff_str->free_energy_DFT[i] * mlff_str->free_energy_DFT[i];
        sum_y_square += mlff_str->internal_energy_DFT[i] * mlff_str->internal_energy_DFT[i];
        sum_x += mlff_str->free_energy_DFT[i];
        sum_xy += mlff_str->free_energy_DFT[i] * mlff_str->internal_energy_DFT[i];
        // sum_y += mlff_str->internal_energy_DFT[i];
    }

    alpha = (sum_y * sum_x_square - sum_x * sum_xy)/(N*sum_x_square - sum_x * sum_x);
    beta = (N*sum_xy - sum_x * sum_y)/(N*sum_x_square - sum_x * sum_x);

    mlff_str->internal_energy_model_weights[0] = alpha;
    mlff_str->internal_energy_model_weights[1] = beta;

    double R = ((sum_xy/N) - (sum_x/N)*(sum_y/N))/sqrt((sum_x_square/N - (sum_x/N)*(sum_x/N))*(sum_y_square/N - (sum_y/N)*(sum_y/N)));
    mlff_str->internal_energy_model_R2 = R*R;

    FILE *fp_mlff;
    if (mlff_str->print_mlff_flag==1 && rank==0){
        // fp_mlff = fopen("mlff.log","a");
        fp_mlff = mlff_str->fp_mlff;
    }

    if (mlff_str->print_mlff_flag == 1 && rank ==0){
        fprintf(fp_mlff, "Internal energy model trained on %d data points! R^2: %f\n", N, mlff_str->internal_energy_model_R2);
        fprintf(fp_mlff, "Internal energy model weights: y = %f + %fx\n", alpha, beta);
    }

    // store the training data so that restarts (MLFF_FLAG 21/22) can rebuild the same model
    print_internal_energy_model_data(mlff_str);
}

/*
Write the (free energy, internal energy) training pairs to mlff_str->ie_model_name (rank 0)
*/
void print_internal_energy_model_data(MLFF_Obj *mlff_str){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;

    FILE *fptr = fopen(mlff_str->ie_model_name, "w");
    if (fptr == NULL) {
        printf("Error opening the file %s!\n", mlff_str->ie_model_name);
        exit(1);
    }
    fprintf(fptr, "SPARC MLFF internal energy model data: free energy (Ha), internal energy (Ha)\n");
    fprintf(fptr, "N_data:\n");
    fprintf(fptr, "%d\n", mlff_str->internal_energy_DFT_count);
    for (int i = 0; i < mlff_str->internal_energy_DFT_count; i++) {
        fprintf(fptr, "%.15e %.15e\n", mlff_str->free_energy_DFT[i], mlff_str->internal_energy_DFT[i]);
    }
    fclose(fptr);
}

/*
Read the training pairs of a previous run from mlff_str->ie_model_name and broadcast them
*/
void read_internal_energy_model_data(MLFF_Obj *mlff_str, int n_max){
    int rank, N = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        FILE *fptr = fopen(mlff_str->ie_model_name, "r");
        if (fptr == NULL) {
            printf("\nERROR: MLFF_INTERNAL_ENERGY_FLAG: 1 needs the internal energy model data \"%s\" of the previous run, which cannot be opened.\n", mlff_str->ie_model_name);
            exit(EXIT_FAILURE);
        }
        char str[L_STRING];
        if (fgets(str, L_STRING, fptr) == NULL || fscanf(fptr, "%s %d", str, &N) != 2 || N < 2 || N > n_max) {
            printf("\nERROR: invalid internal energy model data in \"%s\" (need 2 to %d data points).\n", mlff_str->ie_model_name, n_max);
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < N; i++) {
            if (fscanf(fptr, "%lf %lf", &mlff_str->free_energy_DFT[i], &mlff_str->internal_energy_DFT[i]) != 2) {
                printf("\nERROR: could not read data point %d from \"%s\".\n", i+1, mlff_str->ie_model_name);
                exit(EXIT_FAILURE);
            }
        }
        fclose(fptr);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mlff_str->free_energy_DFT, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(mlff_str->internal_energy_DFT, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    mlff_str->internal_energy_DFT_count = N;
}