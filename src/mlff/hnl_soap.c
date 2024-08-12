#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#include "tools_mlff.h"
#include "spherical_harmonics.h"
#include "soap_descriptor.h"
#include "mlff_types.h"
#include "ddbp_tools.h"
#include "tools.h"
#include "sparc_mlff_interface.h"
#include "cyclix_mlff_tools.h"
// #include "besselfunctions.h"
#include "bessel_NR.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define temp_tol 1.0E-10


double compute_fcut(double r, double sigma_atom, double rcut){
	double fc;

	if (r >= rcut){
		fc = 0.0;
	} else {
		fc = 0.5*(cos(M_PI*r/rcut)+1);
	}
	return fc;
}

double compute_der_fcut(double r, double sigma_atom, double rcut){
	double d_fc;

	if (r >= rcut){
		d_fc = 0.0;
	} else {
		d_fc = -0.5*(M_PI/rcut)*sin(M_PI*r/rcut);
	}
	return d_fc;
}


double compute_hnl(int n, int l, double r, double rcut, double sigma_atom){
	double fc = compute_fcut(r, sigma_atom, rcut);

	double Cr = 4.0*M_PI*fc/sqrt(2*sigma_atom*sigma_atom*M_PI);



	double dr = 0.001;

	int N_integration_grid = floor(rcut/dr);

	dr = (rcut-0.001)/(N_integration_grid-1);

	double rgrid_integration[N_integration_grid];

	double rgrid_min = 0.001, rgrid_max = rcut;

	for (int i = 0; i < N_integration_grid; i++){
		rgrid_integration[i] = rgrid_min + i*dr;
	}

	double roots[30];
	roots_sph_bessel(l, roots);
	double root_n = roots[n];

	double q_nl = root_n/rcut;

	double jl_term[N_integration_grid], exp_term[N_integration_grid], ll_term[N_integration_grid];
	double sj, sy, sjp, syp, sj1;

	sphbes(l+1, q_nl*rcut, &sj1, &sy, &sjp, &syp);
	double C_sph_Bessel = sqrt((1.0/2.0/M_PI)*(1.0/rcut/rcut/rcut)*(1.0/sj1/sj1));
	for (int i = 0; i < N_integration_grid; i++){
		sphbes(l, q_nl*rgrid_integration[i], &jl_term[i], &sy, &sjp, &syp);
		jl_term[i] = C_sph_Bessel*jl_term[i];

		exp_term[i] = exp(-1.0/(2*sigma_atom*sigma_atom) * (r*r + rgrid_integration[i]*rgrid_integration[i]));
		sphbes_mod(l, r*rgrid_integration[i]/(sigma_atom*sigma_atom), &ll_term[i], &sy, &sjp, &syp);
	}

	

	double hnl_val = 0.0;

	for (int i = 0; i < N_integration_grid; i++){
		hnl_val += jl_term[i]*exp_term[i]*ll_term[i]*rgrid_integration[i]*rgrid_integration[i];
	}

	hnl_val = hnl_val * Cr * (rgrid_integration[1]-rgrid_integration[0]);
	return hnl_val;

}

double compute_d_hnl(int n, int l, double r, double rcut, double sigma_atom){
	double fc = compute_fcut(r, sigma_atom, rcut);

	double Cr = 4.0*M_PI*fc/sqrt(2*sigma_atom*sigma_atom*M_PI);

	double dr = 0.001;

	int N_integration_grid = floor(rcut/dr);

	dr = (rcut-0.001)/(N_integration_grid-1);

	double rgrid_integration[N_integration_grid];

	double rgrid_min = 0.001, rgrid_max = rcut;

	for (int i = 0; i < N_integration_grid; i++){
		rgrid_integration[i] = rgrid_min + i*dr;
	}
	
	double roots[30];
	roots_sph_bessel(l, roots);
	double root_n = roots[n];

	double q_nl = root_n/rcut;

	double jl_term[N_integration_grid], exp_term[N_integration_grid], ll_term[N_integration_grid];

	double sj, sy, sjp, syp, sj1;
	sphbes(l+1, q_nl*rcut, &sj1, &sy, &sjp, &syp);
	double C_sph_Bessel = sqrt((1.0/2.0/M_PI)*(1.0/rcut/rcut/rcut)*(1.0/sj1/sj1));
	for (int i = 0; i < N_integration_grid; i++){
		sphbes(l, q_nl*rgrid_integration[i], &jl_term[i], &sy, &sjp, &syp);
		jl_term[i] = C_sph_Bessel*jl_term[i];
		exp_term[i] = exp(-1.0/(2*sigma_atom*sigma_atom) * (r*r + rgrid_integration[i]*rgrid_integration[i]));
		sphbes_mod(l, r*rgrid_integration[i]/(sigma_atom*sigma_atom), &ll_term[i], &sy, &sjp, &syp);
	}

	double d_fc, d_Cr, d_exp_term[N_integration_grid], d_ll_term[N_integration_grid];

	d_fc = compute_der_fcut(r, sigma_atom, rcut);
	d_Cr = 4.0*M_PI*d_fc/sqrt(2*sigma_atom*sigma_atom*M_PI);

	for (int i = 0; i < N_integration_grid; i++){
		d_exp_term[i] = exp_term[i] * (-1.0*r)/(sigma_atom*sigma_atom);
		sphbes_mod(l, r*rgrid_integration[i]/(sigma_atom*sigma_atom), &sj, &sy, &d_ll_term[i], &syp);
		d_ll_term[i] = d_ll_term[i] * (rgrid_integration[i])/(sigma_atom*sigma_atom);
		
	} 
	
	double d_hnl_val = 0.0;

	for (int i = 0; i < N_integration_grid; i++){
		d_hnl_val += d_Cr*jl_term[i]*exp_term[i]*ll_term[i]*rgrid_integration[i]*rgrid_integration[i] 
				   + Cr*jl_term[i]*d_exp_term[i]*ll_term[i]*rgrid_integration[i]*rgrid_integration[i]
				   + Cr*jl_term[i]*exp_term[i]*d_ll_term[i]*rgrid_integration[i]*rgrid_integration[i];
	} 
	d_hnl_val = d_hnl_val *  (rgrid_integration[1]-rgrid_integration[0]);

	return d_hnl_val;
}

void compute_hnl_soap(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str){

	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n_grid_local = 0;

	if (nprocs > pSPARC->N_rgrid_MLFF){
		if (rank < pSPARC->N_rgrid_MLFF){
			n_grid_local = 1;
		}
	} else {
		n_grid_local = 1 + floor(pSPARC->N_rgrid_MLFF/nprocs);
		pSPARC->N_rgrid_MLFF = n_grid_local * nprocs;
	}
	
	mlff_str->rgrid = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF);
	double dr = (mlff_str->rcut-0.001)/(pSPARC->N_rgrid_MLFF-1);
	double rgrid_min = 0.001, rgrid_max = mlff_str->rcut;
	for (int i = 0; i < pSPARC->N_rgrid_MLFF; i++){
		mlff_str->rgrid[i] = rgrid_min + i*dr;
	}

	double *rgrid_local = (double *) malloc(sizeof(double)* n_grid_local);
	for (int i = 0; i < n_grid_local; i++){
		rgrid_local[i] = mlff_str->rgrid[rank*n_grid_local + i];
	}

	

	double ***h_nl_local, ***dh_nl_local;

	h_nl_local = (double ***) malloc(sizeof(double**)* pSPARC->N_max_SOAP);
	dh_nl_local = (double ***) malloc(sizeof(double**)* pSPARC->N_max_SOAP);
	for (int i = 0; i < pSPARC->N_max_SOAP; i++){
		h_nl_local[i] = (double **) malloc(sizeof(double*)* (pSPARC->L_max_SOAP+1));
		dh_nl_local[i] = (double **) malloc(sizeof(double*)* (pSPARC->L_max_SOAP+1));
		for (int j = 0; j < (pSPARC->L_max_SOAP+1); j++){
			h_nl_local[i][j] = (double *) malloc(sizeof(double)* n_grid_local);
			dh_nl_local[i][j] = (double *) malloc(sizeof(double)* n_grid_local);
			for (int k = 0; k < n_grid_local; k++){
				h_nl_local[i][j][k]=0.0;
				dh_nl_local[i][j][k]=0.0;
			}
		}
	}
	for (int n = 0; n < pSPARC->N_max_SOAP; n++){
		for (int l = 0; l < pSPARC->L_max_SOAP+1; l++){
			for (int r_idx = 0; r_idx < n_grid_local; r_idx++){
				h_nl_local[n][l][r_idx] = compute_hnl(n, l, rgrid_local[r_idx], mlff_str->rcut, mlff_str->sigma_atom);
				dh_nl_local[n][l][r_idx] = compute_d_hnl(n, l, rgrid_local[r_idx], mlff_str->rcut, mlff_str->sigma_atom);
			}
		}
	}


	

	double ***h_nl_gathered, ***dh_nl_gathered;
	h_nl_gathered = (double ***) malloc(sizeof(double**)* pSPARC->N_max_SOAP);
	dh_nl_gathered = (double ***) malloc(sizeof(double**)* pSPARC->N_max_SOAP);
	for (int i = 0; i < pSPARC->N_max_SOAP; i++){
		h_nl_gathered[i] = (double **) malloc(sizeof(double*)* (pSPARC->L_max_SOAP+1));
		dh_nl_gathered[i] = (double **) malloc(sizeof(double*)* (pSPARC->L_max_SOAP+1));
		for (int j = 0; j < (pSPARC->L_max_SOAP+1); j++){
			h_nl_gathered[i][j] = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF);
			dh_nl_gathered[i][j] = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF);
			for (int k = 0; k < pSPARC->N_rgrid_MLFF; k++){
				h_nl_gathered[i][j][k]=0.0;
				dh_nl_gathered[i][j][k]=0.0;
			}
		}
	}

	int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = (int *)malloc(nprocs * sizeof(int));
        displs = (int *)malloc(nprocs * sizeof(int));
    }
    MPI_Gather(&n_grid_local, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < nprocs; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }

	for (int i = 0; i < pSPARC->N_max_SOAP; i++){
		for (int j = 0; j < (pSPARC->L_max_SOAP+1); j++){
			MPI_Gatherv(h_nl_local[i][j], n_grid_local, MPI_DOUBLE, h_nl_gathered[i][j], recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gatherv(dh_nl_local[i][j], n_grid_local, MPI_DOUBLE, dh_nl_gathered[i][j], recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
	}
	


	for (int i = 0; i < pSPARC->N_max_SOAP; i++){
		for (int j = 0; j < (pSPARC->L_max_SOAP+1); j++){
			MPI_Bcast(h_nl_gathered[i][j], pSPARC->N_rgrid_MLFF, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(dh_nl_gathered[i][j], pSPARC->N_rgrid_MLFF, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
	}

	

	mlff_str->h_nl = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));
	mlff_str->dh_nl = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));

	int count = 0;
	for (int n = 0; n < pSPARC->N_max_SOAP; n++){
		for (int l = 0; l < pSPARC->L_max_SOAP+1; l++){
			for (int r_idx = 0; r_idx < pSPARC->N_rgrid_MLFF; r_idx++){
				mlff_str->h_nl[count] = h_nl_gathered[n][l][r_idx];
				mlff_str->dh_nl[count] = dh_nl_gathered[n][l][r_idx];
				count=count+1;
			}
		}
	}

	

	for (int i = 0; i < pSPARC->N_max_SOAP; i++){
		for (int j = 0; j < (pSPARC->L_max_SOAP+1); j++){
			if (n_grid_local>0){
				free(h_nl_local[i][j]);
				free(dh_nl_local[i][j]);
			}
			
		}
		free(h_nl_local[i]);
		free(dh_nl_local[i]);
	}
	free(h_nl_local);
	free(dh_nl_local);
	if (n_grid_local>0){
		free(rgrid_local);
	}

	
	


	for (int i = 0; i < pSPARC->N_max_SOAP; i++){
		for (int j = 0; j < (pSPARC->L_max_SOAP+1); j++){
			free(h_nl_gathered[i][j]);
			free(dh_nl_gathered[i][j]);
		}
		free(h_nl_gathered[i]);
		free(dh_nl_gathered[i]);
	}
	free(h_nl_gathered);
	free(dh_nl_gathered);

	free(recvcounts);
	free(displs);


}