#include <string.h>
#include <stdlib.h>
#include "ddbp_tools.h"
#include "descriptor.h"
#include "soap_descriptor.h"


void initialize_descriptor(DescriptorObj *desc_str, MLFF_Obj *mlff_str, NeighList *nlist) {

    int nelem = mlff_str->nelem, natom = mlff_str->natom;
    int Lmax = mlff_str->Lmax, Nmax = mlff_str->Nmax;
    double xi_3 = mlff_str->xi_3;
    desc_str->descriptor_typ = mlff_str->descriptor_typ;
    desc_str->xi_3 = xi_3;
    desc_str->natom = natom;
    desc_str->natom_domain = nlist->natom_domain;
    desc_str->atom_idx_domain = (int *) malloc(nlist->natom_domain * sizeof(int));
    desc_str->el_idx_domain = (int *) malloc(nlist->natom_domain * sizeof(int));
    desc_str->if_self_image = (int *) malloc(nlist->natom_domain * sizeof(int));
    memcpy(desc_str->atom_idx_domain, nlist->atom_idx_domain, nlist->natom_domain * sizeof(int));
    memcpy(desc_str->el_idx_domain, nlist->el_idx_domain, nlist->natom_domain * sizeof(int));
    memcpy(desc_str->if_self_image, nlist->if_self_image, nlist->natom_domain * sizeof(int));
    desc_str->rcut = nlist->rcut;
    desc_str->Lmax = Lmax;
    desc_str->Nmax = Nmax;
    desc_str->nelem = nelem;
    // desc_str->beta_2 = 1 - mlff_str->beta_3;
    // desc_str->beta_3 = mlff_str->beta_3;
    desc_str->Nneighbors = (int *) malloc(nlist->natom_domain * sizeof(int));
    desc_str->unique_Nneighbors = (int *) malloc(nlist->natom_domain * sizeof(int));
    memcpy(desc_str->Nneighbors, nlist->Nneighbors, nlist->natom_domain * sizeof(int));
    memcpy(desc_str->unique_Nneighbors, nlist->unique_Nneighbors, nlist->natom_domain * sizeof(int));
    desc_str->unique_Nneighbors_elemWise = (int **) malloc(nlist->natom_domain * sizeof(int*));
    for (int i = 0; i < nlist->natom_domain; i++) {
        desc_str->unique_Nneighbors_elemWise[i] = (int *) malloc(nelem * sizeof(int));
        memcpy(desc_str->unique_Nneighbors_elemWise[i], nlist->unique_Nneighbors_elemWise[i], nelem * sizeof(int));
    }
    desc_str->natom_elem = (int *) malloc(nelem * sizeof(int));
    memcpy(desc_str->natom_elem, nlist->natom_elem, nelem * sizeof(int));

    desc_str->unique_neighborList = (dyArray *) malloc(sizeof(dyArray)*nlist->natom_domain);
	for (int i =0; i < nlist->natom_domain; i++){
		desc_str->unique_neighborList[i].len = nlist->unique_neighborList[i].len;
		desc_str->unique_neighborList[i].capacity = nlist->unique_neighborList[i].capacity;
		desc_str->unique_neighborList[i].array = (int *)malloc(sizeof(int)*nlist->unique_neighborList[i].capacity);
		for (int k =0; k<nlist->unique_neighborList[i].len; k++){
			desc_str->unique_neighborList[i].array[k] = nlist->unique_neighborList[i].array[k];
		}
	}

    if (nlist->cell_typ < 20) {
		double Jacbdet = 0.0;
		for(int i = 0; i < 3; i++){
        	for(int j = 0; j < 3; j++){
            	for(int k = 0; k < 3; k++){
                	if(i != j && j != k && k != i)
                    	Jacbdet += ((i - j) * (j - k) * (k - i)/2) * nlist->LatUVec[3 * i] * nlist->LatUVec[3 * j + 1] * nlist->LatUVec[3 * k + 2];
            	}
        	}
    	}
		desc_str->cell_measure = Jacbdet;
	    if(nlist->BC[0] == 0)	
	    	desc_str->cell_measure *= nlist->cell_len[0];
	    if(nlist->BC[1] == 0)
	    	desc_str->cell_measure *= nlist->cell_len[1];
	    if(nlist->BC[2] == 0)
	    	desc_str->cell_measure *= nlist->cell_len[2];
	} else {
		desc_str->cell_measure = (nlist->cell_len[2] / (2*M_PI/nlist->cell_len[1]));
	}

    if (mlff_str->descriptor_typ < 2) {
        desc_str->size_X3 = mlff_str->size_X3;
        desc_str->N_rgrid = mlff_str->N_rgrid;

        desc_str->X3 = (double **) malloc(nlist->natom_domain * sizeof(double*));
        desc_str->dX3_dF = (double ***) malloc(nlist->natom_domain * sizeof(double**));
        desc_str->dX3_dX = (double ***) malloc(nlist->natom_domain * sizeof(double**));
        desc_str->dX3_dY = (double ***) malloc(nlist->natom_domain * sizeof(double**));
        desc_str->dX3_dZ = (double ***) malloc(nlist->natom_domain * sizeof(double**));
        for (int i=0; i < nlist->natom_domain; i++) {
            desc_str->X3[i] = (double *) calloc(mlff_str->size_X3, sizeof(double));
            int uniq_natms = nlist->unique_Nneighbors[i];

            desc_str->dX3_dX[i] = (double **) malloc((1 + uniq_natms) * sizeof(double*));
            desc_str->dX3_dY[i] = (double **) malloc((1 + uniq_natms) * sizeof(double*));
            desc_str->dX3_dZ[i] = (double **) malloc((1 + uniq_natms) * sizeof(double*));
            desc_str->dX3_dF[i] = (double **) malloc(6 * sizeof(double*));
            for (int j=0; j < 1+uniq_natms; j++) {
                desc_str->dX3_dX[i][j] = (double *) calloc(mlff_str->size_X3, sizeof(double));
                desc_str->dX3_dY[i][j] = (double *) calloc(mlff_str->size_X3, sizeof(double));
                desc_str->dX3_dZ[i][j] = (double *) calloc(mlff_str->size_X3, sizeof(double));
            }
            for (int j=0; j < 6; j++) {
                desc_str->dX3_dF[i][j] = (double *) calloc(mlff_str->size_X3, sizeof(double));
            }
        }
    } 
    // put GMP in else here
}

void build_descriptor(DescriptorObj *desc_str, NeighList *nlist, MLFF_Obj *mlff_str, double *atom_pos) {

    initialize_descriptor(desc_str, mlff_str, nlist);
    if (mlff_str->descriptor_typ < 2) {
        build_soapObj(desc_str, nlist, mlff_str->rgrid, mlff_str->h_nl, mlff_str->dh_nl, atom_pos, mlff_str->Nmax, mlff_str->Lmax,  mlff_str->xi_3, mlff_str->N_rgrid);
    }  // put GMP in else here
}

void delete_descriptor(DescriptorObj *desc_str) {

    if (desc_str->descriptor_typ < 2) {
        delete_soapObj(desc_str);
    }  // put GMP in else here
    for (int i=0; i < desc_str->natom_domain; i++) {
        free(desc_str->unique_Nneighbors_elemWise[i]);
        delete_dyarray(desc_str->unique_neighborList+i);
    }
    free(desc_str->unique_neighborList);
    free(desc_str->unique_Nneighbors);
    free(desc_str->atom_idx_domain);
    free(desc_str->el_idx_domain);
    free(desc_str->if_self_image);
    free(desc_str->Nneighbors);
    free(desc_str->unique_Nneighbors_elemWise);
    free(desc_str->natom_elem);
}