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

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define temp_tol 1.0E-10








void read_h_nl(const int N, const int L, double *rgrid, double *h_nl, double *dh_nl, SPARC_OBJ *pSPARC){
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

	for (i = 0; i < N_r; i++){
		info = fscanf(fp, "%s", a1);
		rgrid[i] = atof(a1);

	}

	for (i =0; i < N*(L+1); i++){
		info = fscanf(fp, "%s %s", a1, a2);

		for (j = 0; j < N_r; j++){
			info = fscanf(fp, "%s", a1);
			h_nl[count1] = atof(a1);
			count1++;	
		}
		for (j=0; j<N_r; j++){
			info = fscanf(fp, "%s", a1);
			dh_nl[count2] = atof(a1);
			count2++;
		}
	}


	fclose(fp);
}





/*
initialize_nlist function initializes the a objects in the NeighList structure and also allocates memory for the dunamic arrays.

[Input]
1. natom: Number of atoms in the system
2. rcut: Cutoff distance (bohr)
3. nelem: Number of element species 
[Output]
1. nlist: pointer to Neighlist structure to be initialized
*/

void initialize_nlist(NeighList* nlist, int cell_typ, int *BC, double *cell_len, double *LatUVec, double twist, const int natom, const double rcut, const int nelem , const int natom_domain, int *atom_idx_domain, int *el_idx_domain){

    nlist->cell_typ = cell_typ;
    nlist->BC[0] = BC[0]; nlist->BC[1] = BC[1]; nlist->BC[2] = BC[2];
    nlist->cell_len[0] = cell_len[0]; nlist->cell_len[1] = cell_len[1]; nlist->cell_len[2] = cell_len[2];
    for(int i = 0; i < 9; i++){
    	nlist->LatUVec[i] = LatUVec[i];
    }
    nlist->twist = twist;
    nlist->natom = natom;
 	nlist->natom_domain = natom_domain;
 	nlist->atom_idx_domain = (int *) malloc(sizeof(int)*natom_domain);  
 	nlist->el_idx_domain = (int *) malloc(sizeof(int)*natom_domain);
 	nlist->if_self_image = (int *) malloc(sizeof(int)*natom_domain);
 	for (int i=0; i < natom_domain; i++){
 		nlist->atom_idx_domain[i] = atom_idx_domain[i];
 		nlist->el_idx_domain[i] = el_idx_domain[i];
 		nlist->if_self_image[i] = 0;
 	}

 	nlist->nelem = nelem;
	nlist->rcut = rcut;
	nlist->natom_elem = (int *) malloc(nelem*sizeof(int));
	for (int i=0; i<nelem; i++){
		nlist->natom_elem[i] = 0;
	}

	nlist->Nneighbors = (int *) malloc(natom_domain*sizeof(int));
	nlist->unique_Nneighbors = (int *) malloc(natom_domain*sizeof(int));
	nlist->Nneighbors_elemWise = (int **) malloc(natom_domain*sizeof(int*));
	nlist->unique_Nneighbors_elemWise = (int **) malloc(natom_domain*sizeof(int*));
	for (int i=0; i<natom_domain; i++){
		nlist->Nneighbors[i] = 0;
		nlist->unique_Nneighbors[i] = 0;
		nlist->Nneighbors_elemWise[i] = (int *) malloc(nelem*sizeof(int));
		nlist->unique_Nneighbors_elemWise[i] = (int *) malloc(nelem*sizeof(int));
		for (int j = 0; j<nelem; j++){
			nlist->Nneighbors_elemWise[i][j] = 0;
			nlist->unique_Nneighbors_elemWise[i][j] = 0;
		}
	}
	
	nlist->neighborList = (dyArray *) malloc(sizeof(dyArray)*natom_domain);
	nlist->unique_neighborList = (dyArray *) malloc(sizeof(dyArray)*natom_domain);
	nlist->neighborList_elemWise = (dyArray **) malloc(sizeof(dyArray*)*natom_domain);
	nlist->unique_neighborList_elemWise = (dyArray **) malloc(sizeof(dyArray*)*natom_domain);
	nlist->neighborAtmTyp = (dyArray *) malloc(sizeof(dyArray)*natom_domain);
	nlist->neighborList_imgX = (dyArray *) malloc(sizeof(dyArray)*natom_domain);
	nlist->neighborList_imgY = (dyArray *) malloc(sizeof(dyArray)*natom_domain);
	nlist->neighborList_imgZ = (dyArray *) malloc(sizeof(dyArray)*natom_domain);
	nlist->neighborList_elemWise_imgX = (dyArray **) malloc(sizeof(dyArray*)*natom_domain);
	nlist->neighborList_elemWise_imgY = (dyArray **) malloc(sizeof(dyArray*)*natom_domain);
	nlist->neighborList_elemWise_imgZ = (dyArray **) malloc(sizeof(dyArray*)*natom_domain);
	nlist->localID_neighbours = (dyArray *) malloc(sizeof(dyArray)*natom_domain);
	nlist->localID_neighbours_elem = (dyArray **) malloc(sizeof(dyArray*)*natom_domain);


	
	for (int i =0; i < natom_domain; i++){
		init_dyarray(nlist->neighborList + i);
		init_dyarray(nlist->unique_neighborList + i);
		init_dyarray(nlist->neighborAtmTyp + i);
		init_dyarray(nlist->neighborList_imgX + i);
		init_dyarray(nlist->neighborList_imgY + i);
		init_dyarray(nlist->neighborList_imgZ + i);
		init_dyarray(nlist->localID_neighbours + i);


		nlist->neighborList_elemWise[i] = (dyArray *) malloc(sizeof(dyArray)*nelem);
		nlist->unique_neighborList_elemWise[i] = (dyArray *) malloc(sizeof(dyArray)*nelem);
		nlist->neighborList_elemWise_imgX[i] = (dyArray *) malloc(sizeof(dyArray)*nelem);
		nlist->neighborList_elemWise_imgY[i] = (dyArray *) malloc(sizeof(dyArray)*nelem);
		nlist->neighborList_elemWise_imgZ[i] = (dyArray *) malloc(sizeof(dyArray)*nelem);
		nlist->localID_neighbours_elem[i] = (dyArray *) malloc(sizeof(dyArray)*nelem);

		for (int j=0; j<nelem; j++){
			init_dyarray(&(nlist->neighborList_elemWise[i][j]));
			init_dyarray(&(nlist->unique_neighborList_elemWise[i][j]));
			init_dyarray(&(nlist->neighborList_elemWise_imgX[i][j]));
			init_dyarray(&(nlist->neighborList_elemWise_imgY[i][j]));
			init_dyarray(&(nlist->neighborList_elemWise_imgZ[i][j]));
			init_dyarray(&(nlist->localID_neighbours_elem[i][j]));

		}
	}
}

/*
lin_search function searches for the first occurence of an element in the array 

[Input]
1. arr: pointer to array
2. n: length of array
3. x: element to search

[Output]
1. i: ID of the element in the array
*/

int lin_search(int *arr, int n, int x)
{
    int i;
    for (i = 0; i < n; i++)
        if (arr[i] == x)
            return i;
    return -1;
}

/*
build_nlist function calculate the list of neighbours for each atoms in the structure.

[Input]
1. rcut: Cutoff distance (bohr) 
2. nelem: Number of element species 
3. natom: Number of atoms in the system
4. atompos: Pointer to the array containing the atom positions of the atoms in the fundamental cell (stored RowMajor format)
5. atomtyp: Pointer to the array containing the element type of the atoms in the fundamental cell (stored RowMajor format)
6. BC[]: Boundary condition (0 for Dirchelet, 1 for Periodic) [Not used currently, add later]
7. cell: 3*1 array to store the length of fundamental cell. [Currently only orthogonal cells are implemented]

[Output]
1. nlist: pointer to Neighlist structure
*/

//TODO: Replace geometric ratio!
void build_nlist(const double rcut, const int nelem, const int natom, const double * const atompos,
				 int * atomtyp, int cell_typ, int *BC, double *cell_len, double *LatUVec, double twist, double *geometric_ratio, NeighList* nlist, const int natom_domain, int *atom_idx_domain, int *el_idx_domain) {
	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	double L1 = cell_len[0];
	double L2 = cell_len[1];
	double L3 = cell_len[2];
	double rcut_x = rcut;
	double rcut_y = rcut * geometric_ratio[0];
	double rcut_z = rcut * geometric_ratio[1];
	initialize_nlist(nlist, cell_typ, BC, cell_len, LatUVec, twist, natom, rcut, nelem, natom_domain, atom_idx_domain, el_idx_domain);	

    	
	for (int i=0; i<natom; i++){
		nlist->natom_elem[atomtyp[i]] = nlist->natom_elem[atomtyp[i]] +1;
	}
	
	for (int i = 0; i < natom_domain; i++){
		int atm_idx = atom_idx_domain[i];
		int el_idx = el_idx_domain[i];
		double xi = atompos[3*atm_idx];
		double yi = atompos[3*atm_idx+1];
		double zi = atompos[3*atm_idx+2];

		int img_px = 0;
		int img_nx = 0;
		int img_py = 0;
		int img_ny = 0;
		int img_pz = 0;
		int img_nz = 0;


		if (BC[0] == 0) {
			img_px = max(0,ceil((rcut_x - L1 + xi) /L1));
			img_nx = max(0,ceil((rcut_x - xi) /L1));
        }
        if (BC[1] == 0) {
			img_py = max(0,ceil((rcut_y - L2 + yi) /L2));
			img_ny = max(0,ceil((rcut_y - yi) /L2));    
        }
        if (BC[2] == 0) {
			img_pz = max(0,ceil((rcut_z - L3 + zi) /L3));
			img_nz = max(0,ceil((rcut_z - zi) /L3));
		}

		if (cell_typ > 20 && cell_typ < 30) {
			get_img_cyclix(L2, temp_tol, &img_ny, &img_py);
			// img_ny = 0;
			// img_py = floor(2*M_PI/L2+temp_tol) - 1; // all atoms and images in cyclic direction
		}
		
		// count_neighs = 0;
		for (int j = 0; j < natom; j++){
			int count_unique=0;
			double xj_u = atompos[3*j];
			double yj_u = atompos[3*j+1];
			double zj_u = atompos[3*j+2];
			// if (j==atm_idx) continue;
			for (int img_x = -img_nx; img_x <= img_px; img_x++){
				double xj = xj_u + img_x*L1;	
				//if (fabs( xj - xi) >= rcut_x) continue;
				for (int img_y = -img_ny; img_y <= img_py; img_y++){
					double yj = yj_u + img_y*L2;
					//if ((cell_typ > 20 && cell_typ < 30) ? (fabs( yj - yi + twist*(zj-zi)) >= rcut_y && (2*M_PI - fabs( yj - yi + twist*(zj-zi))) >= rcut_y) : fabs( yj - yi) >= rcut_y) continue;
					for (int img_z = -img_nz; img_z <= img_pz; img_z++){
						double zj = zj_u + img_z*L3;
						//if (fabs(zj - zi) >= rcut_z) continue;
						if ((j==atm_idx) && (img_x==0) && (img_y==0) && (img_z==0)) continue;

						double dx = 0.0;
						double dy = 0.0;
						double dz = 0.0;
						double dr = 0.0;
						
						if (cell_typ > 20 && cell_typ < 30) {
							// dx = xj*cos(yj+twist*zj) - xi*cos(yi+twist*zi);
							// dy = xj*sin(yj+twist*zj) - xi*sin(yi+twist*zi);
							// dz = zj - zi;
							get_cartesian_dist_cyclix(xi, yi, zi, xj, yj, zj, twist, &dx, &dy, &dz);
						} else if (cell_typ > 10 && cell_typ < 20){
							double dx1 = xj - xi;
							double dx2 = yj - yi;
							double dx3 = zj - zi;
					        dx = LatUVec[0] * dx1 + LatUVec[3] * dx2 + LatUVec[6] * dx3;
					        dy = LatUVec[1] * dx1 + LatUVec[4] * dx2 + LatUVec[7] * dx3;
					        dz = LatUVec[2] * dx1 + LatUVec[5] * dx2 + LatUVec[8] * dx3;
						} else if (cell_typ == 0) { // add for nonorthogonal
							dx = xj - xi;
							dy = yj - yi;
							dz = zj - zi;
						} 
						
						dr = sqrt(dx*dx + dy*dy + dz*dz);
						if (dr >= rcut || dr ==0) continue;

						if (j == atm_idx){
							nlist->if_self_image[i] = 1;
						}

						nlist->Nneighbors[i] += 1;
						nlist->Nneighbors_elemWise[i][atomtyp[j]] += 1;

						if (count_unique==0) {
							if (j !=atm_idx){
								nlist->unique_Nneighbors[i] += 1;
								nlist->unique_Nneighbors_elemWise[i][atomtyp[j]] += 1;
								append_dyarray(nlist->unique_neighborList + i, j);
								append_dyarray(&(nlist->unique_neighborList_elemWise[i][atomtyp[j]]), j);
							}
						}

	

						append_dyarray(&(nlist->neighborList_elemWise_imgX[i][atomtyp[j]]), img_x);
						append_dyarray(&(nlist->neighborList_elemWise_imgY[i][atomtyp[j]]), img_y);
						append_dyarray(&(nlist->neighborList_elemWise_imgZ[i][atomtyp[j]]), img_z);

						append_dyarray(nlist->neighborList + i, j);
						append_dyarray(&(nlist->neighborList_elemWise[i][atomtyp[j]]), j);



						append_dyarray(nlist->neighborList_imgX + i, img_x);
						append_dyarray(nlist->neighborList_imgY + i, img_y);
						append_dyarray(nlist->neighborList_imgZ + i, img_z);

						append_dyarray(nlist->neighborAtmTyp+i, atomtyp[j]);

						count_unique++;
					}
				}
			} 	
		}
	}

	



	// nlist->localID_neighbours[i] is an array which stores the index of nlist->neighborList_elemWise[i][j].array[k]
	// where j is the element type and k is the neightbour index. The index is stored in (nlist->unique_neighborList_elemWise[i][j])

	for (int i = 0; i < natom_domain; i++) {
		for (int j = 0; j < nelem; j++){
			for (int k = 0; k < nlist->Nneighbors_elemWise[i][j]; k++){
				int idx_neigh =  nlist->neighborList_elemWise[i][j].array[k];
				int ID = lin_search((nlist->unique_neighborList_elemWise[i][j]).array, nlist->unique_Nneighbors_elemWise[i][j], idx_neigh);
				append_dyarray(&(nlist->localID_neighbours[i]), ID);
			}
		}
	}

	for (int i = 0; i < natom_domain; i++) {
		for (int j = 0; j < nlist->unique_Nneighbors[i]; j++) {
			int idx_neigh =  nlist->unique_neighborList[i].array[j];
			for (int k = 0; k < nelem; k++){
				int ID = lin_search((nlist->unique_neighborList_elemWise[i][k]).array, nlist->unique_Nneighbors_elemWise[i][k], idx_neigh);
				append_dyarray(&(nlist->localID_neighbours_elem[i][k]), ID);
			}
		}
	}
}


/*
clear_nlist function frees the memory dynamically allocated for Neighlist 

[Input]
1. nlist: pointer to Neighlist structure

[Output]
1. nlist: pointer to Neighlist structure
*/

void clear_nlist(NeighList* nlist, int natom_domain) {
	free(nlist->natom_elem);
	free(nlist->Nneighbors);
	free(nlist->unique_Nneighbors);

	for (int i=0; i < natom_domain; i++){
		free(nlist->Nneighbors_elemWise[i]);
		free(nlist->unique_Nneighbors_elemWise[i]);
		delete_dyarray(nlist->neighborAtmTyp+i);
		delete_dyarray(nlist->neighborList+i);
		delete_dyarray(nlist->unique_neighborList+i);
		delete_dyarray(nlist->neighborList_imgX+i);
		delete_dyarray(nlist->neighborList_imgY+i);
		delete_dyarray(nlist->neighborList_imgZ+i);
		delete_dyarray(nlist->localID_neighbours+i);
		for (int j=0; j < nlist->nelem; j++){
			delete_dyarray(&(nlist->neighborList_elemWise[i][j]));
			delete_dyarray(&(nlist->unique_neighborList_elemWise[i][j]));
			delete_dyarray(&(nlist->neighborList_elemWise_imgX[i][j]));
			delete_dyarray(&(nlist->neighborList_elemWise_imgY[i][j]));
			delete_dyarray(&(nlist->neighborList_elemWise_imgZ[i][j]));
			delete_dyarray(&(nlist->localID_neighbours_elem[i][j]));
		}
		free(nlist->neighborList_elemWise[i]);
		free(nlist->unique_neighborList_elemWise[i]);
		free(nlist->neighborList_elemWise_imgX[i]);
		free(nlist->neighborList_elemWise_imgY[i]);
		free(nlist->neighborList_elemWise_imgZ[i]);
		free(nlist->localID_neighbours_elem[i]);
	}
	free(nlist->if_self_image);
	free(nlist->Nneighbors_elemWise);
	free(nlist->atom_idx_domain);
	free(nlist->el_idx_domain);
	free(nlist->unique_Nneighbors_elemWise);

	free(nlist->localID_neighbours);
	free(nlist->localID_neighbours_elem);
	free(nlist->neighborList_elemWise);
	free(nlist->unique_neighborList_elemWise);
	free(nlist->neighborList);
	free(nlist->unique_neighborList);
	free(nlist->neighborAtmTyp);
	free(nlist->neighborList_imgX);
	free(nlist->neighborList_imgY);
	free(nlist->neighborList_imgZ);
	free(nlist->neighborList_elemWise_imgX);
	free(nlist->neighborList_elemWise_imgY);
	free(nlist->neighborList_elemWise_imgZ);
}



/*
uniqueEle function finds number of unique entries in an array

[Input]
1. a: pointer to the array
2. n: size of array

[Output]
1. count: Number of unique entries
*/

int uniqueEle(int* a, int n)      //Function Definition
{
   int i, j, count = 1;
   //Traverse the array
   for (i = 1; i < n; i++)      //hold an array element
   {
      for (j = 0; j < i; j++)   
      {
         if (a[i] == a[j])    //Check for duplicate elements
         {
            break;             //If duplicate elements found then break
         }
      }
      if (i == j)
      {
         count++;     //increment the number of distinct elements
      }
   }
   return count;      //Return the number of distinct elements
}

/*
delete_soapObj function frees the memory dynamically allocated for SoapObj 

[Input]
1. soap_str: pointer to SoapObj structure

[Output]
1. soap_str: pointer to SoapObj structure
*/

void delete_soapObj(DescriptorObj *soap_str){
	int i, j;

	for (i=0; i<soap_str->natom_domain; i++){
		// free(soap_str->X2[i]);
		free(soap_str->X3[i]);	
		int uniq_natms = soap_str->unique_Nneighbors[i];
		for (j=0; j<1+uniq_natms; j++){
			free(soap_str->dX3_dX[i][j]);
			free(soap_str->dX3_dY[i][j]);
			free(soap_str->dX3_dZ[i][j]);
		}
		for (j=0; j<6; j++){
			free(soap_str->dX3_dF[i][j]);
		}

		free(soap_str->dX3_dX[i]);
		free(soap_str->dX3_dY[i]);
		free(soap_str->dX3_dZ[i]);
		free(soap_str->dX3_dF[i]);
		
	}


	free(soap_str->X3);
	free(soap_str->dX3_dX);
	free(soap_str->dX3_dY);
	free(soap_str->dX3_dZ);
	free(soap_str->dX3_dF);

}




/*
build_soapObj function calculates the SOAP descriptors and their derivatives w.r.t the atom positions and the deformation gradient

[Input]
1. nlist: pointer to the NeighList structure
2. rgrid: pointer to the rgrid for the spline interpolation
3. h_nl: pointer to the h_nl function for spline interpolation
4. dh_nl: pointer to the derivative of h_nl for spline interpolation
5. atompos: pointer to the atom positions [stored in ColMajor]

[Output]
1. soap_str: pointer to the SoapObj structure
*/
//TODO: implement real spherical harmonics
//TODO: implement spherical harmonics in Cartesian
void build_soapObj(DescriptorObj *soap_str, NeighList *nlist, double* rgrid, double* h_nl, double* dh_nl, double *atompos, int Nmax, int Lmax,  double xi_3, int N_rgrid) {


	// const double PI = 3.141592653589793;
	int natom= nlist->natom;
	int nelem = nlist->nelem;
	int uniq_el_atms, local_index;
	int size_cnlm =  (Lmax + 1) * (Lmax + 2) * Nmax;   // just \sum_{l=1}^{Lmax} (l+1) only store (m = 0 to l)
	size_cnlm = size_cnlm/2;

    int cell_typ = nlist->cell_typ;
    double twist = nlist->twist;

	double complex ***cnlm = (double complex***) malloc(nlist->natom_domain*sizeof(double complex**));
	double complex ****dcnlm_dX = (double complex****) malloc(nlist->natom_domain*sizeof(double complex***));
	double complex ****dcnlm_dY = (double complex****) malloc(nlist->natom_domain*sizeof(double complex***));
	double complex ****dcnlm_dZ = (double complex****) malloc(nlist->natom_domain*sizeof(double complex***));
	double complex ****dcnlm_dF = (double complex****) malloc(nlist->natom_domain*sizeof(double complex***));

	for (int i = 0; i < nlist->natom_domain; i++){
		cnlm[i] = (double complex**) malloc(nelem*sizeof(double complex*));
		for (int j = 0; j < nelem; j++){
			cnlm[i][j] = (double complex*) malloc(size_cnlm*sizeof(double complex));
			for (int k = 0; k < size_cnlm; k++){
				cnlm[i][j][k] = 0.0 + 0.0*I;
			}
		}
		dcnlm_dX[i] = (double complex***) malloc(nelem*sizeof(double complex**));
		dcnlm_dY[i] = (double complex***) malloc(nelem*sizeof(double complex**));
		dcnlm_dZ[i] = (double complex***) malloc(nelem*sizeof(double complex**));
		dcnlm_dF[i] = (double complex***) malloc(nelem*sizeof(double complex**));

		for (int j = 0; j < nelem; j++){
			uniq_el_atms = nlist->unique_Nneighbors_elemWise[i][j];
			dcnlm_dX[i][j] = (double complex**) malloc((1+uniq_el_atms)*sizeof(double complex*));
			dcnlm_dY[i][j] = (double complex**) malloc((1+uniq_el_atms)*sizeof(double complex*));
			dcnlm_dZ[i][j] = (double complex**) malloc((1+uniq_el_atms)*sizeof(double complex*));
			dcnlm_dF[i][j] = (double complex**) malloc(6*sizeof(double complex*));
			for (int k = 0; k < 1+uniq_el_atms; k++){
				dcnlm_dX[i][j][k] = (double complex*) malloc(size_cnlm*sizeof(double complex));
				dcnlm_dY[i][j][k] = (double complex*) malloc(size_cnlm*sizeof(double complex));
				dcnlm_dZ[i][j][k] = (double complex*) malloc(size_cnlm*sizeof(double complex));
				for (int l = 0; l < size_cnlm; l++){
					dcnlm_dX[i][j][k][l] = 0.0 + 0.0*I;
					dcnlm_dY[i][j][k][l] = 0.0 + 0.0*I;
					dcnlm_dZ[i][j][k][l] = 0.0 + 0.0*I;
				}
			}
			for (int k = 0; k < 6; k++){
				dcnlm_dF[i][j][k] = (double complex*) malloc(size_cnlm*sizeof(double complex));
				for (int l = 0; l < size_cnlm; l++){
					dcnlm_dF[i][j][k][l] = 0.0 + 0.0*I;
				}
			}
		}
	}


	int N_r = soap_str->N_rgrid;
	double L1 = nlist->cell_len[0];
	double L2 = nlist->cell_len[1];
	double L3 = nlist->cell_len[2];
	double *LatUVec = nlist->LatUVec;


	double complex *Ylm = (double complex *) malloc((Lmax+1)*(Lmax+1)*sizeof(double complex));
	double complex *dYlm_theta = (double complex *) malloc((Lmax+1)*(Lmax+1)*sizeof(double complex));
	double complex *dYlm_phi = (double complex *) malloc((Lmax+1)*(Lmax+1)*sizeof(double complex));
	double *hnl_temp = (double *) malloc(sizeof(double)*Nmax*(Lmax+1));
	double *dhnl_temp = (double *) malloc(sizeof(double)*Nmax*(Lmax+1));
	double *rtemp = (double *) malloc(sizeof(double)*1);

	double *YD_hnl = (double *) malloc(N_r*(Nmax)*(Lmax+1)*sizeof(double));
	double *YD_dhnl = (double *) malloc(N_r*(Nmax)*(Lmax+1)*sizeof(double));
	
	// derivatives of tabulated h_nl, dh_nl required for spline interpolation (used later in the inner loop).
	for (int i =0; i < Nmax; i++){
		for (int j =0; j < Lmax+1; j++){
			getYD_gen(rgrid, h_nl + (i*(Lmax+1)+j)*N_r, YD_hnl + (i*(Lmax+1)+j)*N_r, N_r);
			getYD_gen(rgrid, dh_nl + (i*(Lmax+1)+j)*N_r, YD_dhnl + (i*(Lmax+1)+j)*N_r, N_r);
		}
	}


	// Reshape stress indices
	int index[6] = {0,1,2,3,4,5};
	reshape_stress(cell_typ, nlist->BC, index);
	double complex *temp_dcnlm_dF = (double complex*) malloc(6 * sizeof(double complex));

	int counter1 = 0, counter2 = 0, counter3 = 0;
	for (int i = 0; i < soap_str->natom_domain; i++){
		int atm_idx = soap_str->atom_idx_domain[i];
		double xi = atompos[3*atm_idx];
		double yi = atompos[3*atm_idx+1];
		double zi = atompos[3*atm_idx+2];
		int *ntemp = (nlist->neighborList +i)->array;
		int *imgx_temp=(nlist->neighborList_imgX + i)->array;
		int *imgy_temp=(nlist->neighborList_imgY + i)->array;
		int *imgz_temp=(nlist->neighborList_imgZ + i)->array;
		int *elemtyp_temp=(nlist->neighborAtmTyp +i)->array;

		for (int j = 0; j < nlist->Nneighbors[i]; j++){
			int idx_neigh = ntemp[j];
			int elem_typ = elemtyp_temp[j];// e_tilde
			double xj = atompos[3*idx_neigh] + L1 * imgx_temp[j];
			double yj = atompos[3*idx_neigh+1] + L2 * imgy_temp[j];
			double zj = atompos[3*idx_neigh+2] + L3 * imgz_temp[j];

			double dx = 0.0;
			double dy = 0.0;
			double dz = 0.0;

			double dr = 0.0;
			double dtheta = 0.0;
			double dphi = 0.0;

			double ty = 0.0;
			double tz = 0.0;
			double rot11 = 0.0;
			double rot12 = 0.0;

			if (cell_typ > 20 && cell_typ < 30) { // add for nonorthogonal
				// dx = xj*cos(yj+twist*zj) - xi*cos(yi+twist*zi);
				// dy = xj*sin(yj+twist*zj) - xi*sin(yi+twist*zi);
				// dz = zj - zi;
				get_cartesian_dist_cyclix(xi, yi, zi, xj, yj, zj, twist, &dx, &dy, &dz);
				ty = -floor(yj/L2);
				tz = -floor(zj/L3);
				rot11 = cos(ty*L2 + tz*twist*L3);
				rot12 = -sin(ty*L2 + tz*twist*L3);
			} else if (cell_typ > 10 && cell_typ < 20){
				double dx1 = xj - xi;
				double dx2 = yj - yi;
				double dx3 = zj - zi;
		        dx = LatUVec[0] * dx1 + LatUVec[3] * dx2 + LatUVec[6] * dx3;
		        dy = LatUVec[1] * dx1 + LatUVec[4] * dx2 + LatUVec[7] * dx3;
		        dz = LatUVec[2] * dx1 + LatUVec[5] * dx2 + LatUVec[8] * dx3;
			} else if (cell_typ == 0) {
				dx = xj - xi;
				dy = yj - yi;
				dz = zj - zi;
			}
			dr = sqrt(dx*dx + dy*dy + dz*dz);
			if (fabs(dr) < 1.0E-15){
				printf("dr==0 Error in SOAP descriptor!!\n");
				exit(1);
			}
			double dth_dxi[3] = {0.0, 0.0, 0.0};
			double dphi_dxi[3] = {0.0, 0.0, 0.0};
			double dr_dF[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
			calculate_dtheta_dphi(dx, dy, dz, dr, &dtheta, &dphi, dth_dxi, dphi_dxi);
			sph_harmonics(dtheta, dphi, soap_str->Lmax, Ylm, dYlm_theta, dYlm_phi);

			// derivative of d(Fdr)/dF at F = I
			dr_dF[0] = dx*dx/dr;
			dr_dF[1] = dx*dy/dr;
			dr_dF[2] = dx*dz/dr;
			dr_dF[3] = dy*dy/dr;
			dr_dF[4] = dy*dz/dr;
			dr_dF[5] = dz*dz/dr;

			double dth_dr[6] = {dth_dxi[0]*dx, dth_dxi[0]*dy, dth_dxi[0]*dz, dth_dxi[1]*dy, dth_dxi[1]*dz, dth_dxi[2]*dz};
			double dphi_dr[6] = {dphi_dxi[0]*dx, dphi_dxi[0]*dy, dphi_dxi[0]*dz, dphi_dxi[1]*dy, dphi_dxi[1]*dz, dphi_dxi[2]*dz};
			double dxi_dr[3] = {dx/dr, dy/dr, dz/dr};
			rtemp[0] = dr;
			// Calculate h_nl and dh_nl from dpline interpoation for all combinations of n and l
			for (int n=0; n < Nmax; n++){
				for (int l=0; l < Lmax+1; l++){
					SplineInterp(rgrid, h_nl+(n*(Lmax+1)+l)*N_r,
						N_r, rtemp, hnl_temp + n*(Lmax+1)+l, 1, YD_hnl+(n*(Lmax+1)+l)*N_r);

					SplineInterp(rgrid, dh_nl+(n*(Lmax+1)+l)*N_r,
						N_r, rtemp, dhnl_temp + n*(Lmax+1)+l, 1, YD_dhnl+(n*(Lmax+1)+l)*N_r);
				}
			}
			// int local_index = lin_search((nlist->unique_neighborList_elemWise[i][elem_typ]).array,
			 							// nlist->unique_Nneighbors_elemWise[i][elem_typ], idx_neigh);
			local_index = 1+nlist->localID_neighbours[i].array[j];

			for (int n=0; n < Nmax; n++){
				for (int l=0; l < Lmax+1; l++){
					for (int m=0; m < l+1; m++){
						int idx1 = (n*(Lmax+1)*(Lmax+2))/2 + (l*(l+1))/2 + m;
						int idx2 = n*(Lmax+1)+l;
						int idx3 = l*l + l + m;

						double hnl_val =  hnl_temp[idx2];
						double complex Ylm_val = conj(Ylm[idx3]);

						double dhnl_val = dhnl_temp[idx2];
						double complex dYlm_theta_val = conj(dYlm_theta[idx3]);
						double complex dYlm_phi_val = conj(dYlm_phi[idx3]);
						// cnlm is coefficient of expansion of atom density 
						cnlm[i][elem_typ][idx1] +=  hnl_val * Ylm_val;

						double complex dFi[3] = {0.0+0.0*I,0.0+0.0*I,0.0+0.0*I};
						// derivative of cnlm of i^th atom with respect to the atom position of j^th neighbor 
						if (atm_idx != idx_neigh){ // all neighbors except self-images

								if ((fabs(dtheta) < temp_tol || fabs(dtheta - M_PI) < temp_tol)){
									double fact = 1.0;
									if ((l%2 == 0) && (fabs(dtheta - M_PI) < temp_tol)) 
										fact = -1.0;
										dFi[0] =  dhnl_val * dxi_dr[0] * Ylm_val + hnl_val * dYlm_theta_val * (fact/dr) ;
										dFi[1] =  dhnl_val * dxi_dr[1] * Ylm_val - hnl_val * dYlm_theta_val * 1.0*I * (fact/dr);
										dFi[2] = dhnl_val * dxi_dr[2] * Ylm_val + hnl_val * (dYlm_theta_val * dth_dxi[2] + dYlm_phi_val * dphi_dxi[2]);	
								} else {
									for (int iforce = 0; iforce < 3; iforce++){			
										dFi[iforce] = dhnl_val * dxi_dr[iforce] * Ylm_val + hnl_val * (dYlm_theta_val * dth_dxi[iforce] + dYlm_phi_val * dphi_dxi[iforce]);		
									}
								}
								dcnlm_dX[i][elem_typ][0][idx1] -= dFi[0];
								dcnlm_dY[i][elem_typ][0][idx1] -= dFi[1];
								dcnlm_dZ[i][elem_typ][0][idx1] -= dFi[2];

    							if (cell_typ > 20 && cell_typ < 30) {
                           dcnlm_dX[i][elem_typ][local_index][idx1] += dFi[0] * rot11 + dFi[1] * rot12;
                           dcnlm_dY[i][elem_typ][local_index][idx1] += -dFi[0] * rot12 + dFi[1] * rot11;
                       } else {
                       	dcnlm_dX[i][elem_typ][local_index][idx1] += dFi[0];
                           dcnlm_dY[i][elem_typ][local_index][idx1] += dFi[1];
                       }
    							dcnlm_dZ[i][elem_typ][local_index][idx1] += dFi[2];
                  } else { // self-images + cyclix (zero for orthogonal/nonorthogonal)

                  	if ((fabs(dtheta) < temp_tol || fabs(dtheta - M_PI) < temp_tol)){

						double fact = 1.0;
						if ((l%2 == 0) && (fabs(dtheta - M_PI) < temp_tol)) 
							fact = -1.0;
						dFi[0] =  dhnl_val * dxi_dr[0] * Ylm_val + hnl_val * dYlm_theta_val *(fact/dr) ;
						dFi[1] =  dhnl_val * dxi_dr[1] * Ylm_val - hnl_val * dYlm_theta_val * 1.0*I   *(fact/dr);
						dFi[2] = dhnl_val * dxi_dr[2] * Ylm_val + hnl_val * (dYlm_theta_val * dth_dxi[2] + dYlm_phi_val * dphi_dxi[2]);	
					} else {
						for (int iforce = 0; iforce < 3; iforce++){			
							dFi[iforce] = dhnl_val * dxi_dr[iforce] * Ylm_val + hnl_val * (dYlm_theta_val * dth_dxi[iforce] + dYlm_phi_val * dphi_dxi[iforce]);	
						}
					}
					if (cell_typ > 20 && cell_typ < 30){
						dcnlm_dX[i][elem_typ][0][idx1] += dFi[0] * (rot11-1.0) + dFi[1] * rot12;
                      	dcnlm_dY[i][elem_typ][0][idx1] += -dFi[0] * rot12 + dFi[1] * (rot11-1.0);
					}
                      

               }

                        // derivative of cnlm of i^th atom with respect to F for stress calculations
						for (int istress = 0; istress < 6; istress++){
							int ind = index[istress];
							if (ind == 0)
								dcnlm_dF[i][elem_typ][istress][idx1] += dFi[0] * dx; //dhnl_val*dr_dF[ind] * (Ylm_val) + hnl_val * (dYlm_theta_val*dth_dr[ind] + dYlm_phi_val*dphi_dr[ind]);
							else if (ind == 1)
								dcnlm_dF[i][elem_typ][istress][idx1] += dFi[0] * dy; 
							else if (ind == 2)
								dcnlm_dF[i][elem_typ][istress][idx1] += dFi[0] * dz;
							else if (ind == 3)
								dcnlm_dF[i][elem_typ][istress][idx1] += dFi[1] * dy;
							else if (ind == 4)
								dcnlm_dF[i][elem_typ][istress][idx1] += dFi[1] * dz;
							else
								dcnlm_dF[i][elem_typ][istress][idx1] += dFi[2] * dz;
						}
					}
				}
			}

			
		}
	}


	



	double complex el1_xder, el1_yder, el1_zder, el2_xder, el2_yder, el2_zder,el_xder,el_yder,el_zder;
	int count_X2, neigh, count_X3, n2, el2;
	
	int local_index2, local_index1;
	double const_X3, temp1, temp2, temp3;	

	for (int na = 0; na < soap_str->natom_domain; na++){
		count_X3=0;
		for (int el1 = 0; el1 < nelem; el1++){
			for (int n1 = 0; n1 < soap_str->Nmax; n1++){
				temp1 = (n1*(Lmax+1)*(Lmax+2))/2;
				for (int n2_el2 = el1*soap_str->Nmax + n1; n2_el2 < nelem*soap_str->Nmax; n2_el2++){
					n2 = n2_el2 % soap_str->Nmax;
					el2 = n2_el2 / soap_str->Nmax;
					temp2 = (n2*(Lmax+1)*(Lmax+2))/2;
					for (int l = 0; l < soap_str->Lmax+1; l++){
						const_X3 = sqrt(8*M_PI*M_PI/(2*l+1));
						temp3 = (l*(l+1))/2;
						for (int m = 0; m < l+1; m++){
							int idx1 = temp1 + temp3 + m;
							int idx2 = temp2 + temp3 + m;
							if (m==0){
								soap_str->X3[na][count_X3] += const_X3 * (double) (cnlm[na][el1][idx1]*conj(cnlm[na][el2][idx2]));
							} else {
								soap_str->X3[na][count_X3] += const_X3 * (double) (2* creal(cnlm[na][el1][idx1]*conj(cnlm[na][el2][idx2])));
							}
							// The derivative w.r.t to itself
							el1_xder = dcnlm_dX[na][el1][0][idx1];
							el1_yder = dcnlm_dY[na][el1][0][idx1];
							el1_zder = dcnlm_dZ[na][el1][0][idx1];
							el2_xder = dcnlm_dX[na][el2][0][idx2];
							el2_yder = dcnlm_dY[na][el2][0][idx2];
							el2_zder = dcnlm_dZ[na][el2][0][idx2];
							if (m==0){
								soap_str->dX3_dX[na][0][count_X3]  += const_X3 * (double)( (el1_xder*conj(cnlm[na][el2][idx2])) +
														  (cnlm[na][el1][idx1]*conj(el2_xder)));

								soap_str->dX3_dY[na][0][count_X3]  += const_X3 * (double)( (el1_yder*conj(cnlm[na][el2][idx2])) +
														  (cnlm[na][el1][idx1]*conj(el2_yder)));

								soap_str->dX3_dZ[na][0][count_X3]  += const_X3 * (double)( (el1_zder*conj(cnlm[na][el2][idx2])) +
														  (cnlm[na][el1][idx1]*conj(el2_zder)));
							} else {
								soap_str->dX3_dX[na][0][count_X3]  += const_X3 * 2*(double)creal( (el1_xder*conj(cnlm[na][el2][idx2])) +
														  (cnlm[na][el1][idx1]*conj(el2_xder)));

								soap_str->dX3_dY[na][0][count_X3]  += const_X3 * 2*(double)creal( (el1_yder*conj(cnlm[na][el2][idx2])) +
														  (cnlm[na][el1][idx1]*conj(el2_yder)));

								soap_str->dX3_dZ[na][0][count_X3]  += const_X3 * 2*(double)creal( (el1_zder*conj(cnlm[na][el2][idx2])) +
														  (cnlm[na][el1][idx1]*conj(el2_zder)));
							}
							// The derivative w.r.t to itself end
							for (int neigh = 0; neigh < nlist->unique_Nneighbors[na]; neigh++){
								local_index1 = 1+nlist->localID_neighbours_elem[na][el1].array[neigh];
								local_index2 = 1+nlist->localID_neighbours_elem[na][el2].array[neigh];
								if (local_index1 == 0){
									el1_xder = 0.0 + 0.0*I;
									el1_yder = 0.0 + 0.0*I;
									el1_zder = 0.0 + 0.0*I;
								} else{
									el1_xder = dcnlm_dX[na][el1][local_index1][idx1];
									el1_yder = dcnlm_dY[na][el1][local_index1][idx1];
									el1_zder = dcnlm_dZ[na][el1][local_index1][idx1];
								}
								if (local_index2 == 0){
									el2_xder = 0.0 + 0.0*I;
									el2_yder = 0.0 + 0.0*I;
									el2_zder = 0.0 + 0.0*I;
								} else{
									el2_xder = dcnlm_dX[na][el2][local_index2][idx2];
									el2_yder = dcnlm_dY[na][el2][local_index2][idx2];
									el2_zder = dcnlm_dZ[na][el2][local_index2][idx2];
								}

 
								if (m==0){
									soap_str->dX3_dX[na][1+neigh][count_X3]  += const_X3 * (double)( (el1_xder*conj(cnlm[na][el2][idx2])) +
															  (cnlm[na][el1][idx1]*conj(el2_xder)));

									soap_str->dX3_dY[na][1+neigh][count_X3]  += const_X3 * (double)( (el1_yder*conj(cnlm[na][el2][idx2])) +
															  (cnlm[na][el1][idx1]*conj(el2_yder)));

									soap_str->dX3_dZ[na][1+neigh][count_X3]  += const_X3 * (double)( (el1_zder*conj(cnlm[na][el2][idx2])) +
															  (cnlm[na][el1][idx1]*conj(el2_zder)));
								} else {
									soap_str->dX3_dX[na][1+neigh][count_X3]  += const_X3 * 2*(double)creal( (el1_xder*conj(cnlm[na][el2][idx2])) +
															  (cnlm[na][el1][idx1]*conj(el2_xder)));

									soap_str->dX3_dY[na][1+neigh][count_X3]  += const_X3 * 2*(double)creal( (el1_yder*conj(cnlm[na][el2][idx2])) +
															  (cnlm[na][el1][idx1]*conj(el2_yder)));

									soap_str->dX3_dZ[na][1+neigh][count_X3]  += const_X3 * 2*(double)creal( (el1_zder*conj(cnlm[na][el2][idx2])) +
															  (cnlm[na][el1][idx1]*conj(el2_zder)));

								}
							}
							if (m==0){
								for (int istress = 0; istress < 6; istress++)
									soap_str->dX3_dF[na][istress][count_X3] += const_X3 * (double)( (dcnlm_dF[na][el1][istress][idx1]*conj(cnlm[na][el2][idx2])) +
															 (cnlm[na][el1][idx1]*conj(dcnlm_dF[na][el2][istress][idx2])));
							} else {
								for (int istress = 0; istress < 6; istress++)
									soap_str->dX3_dF[na][istress][count_X3] += const_X3 * 2*(double)creal( (dcnlm_dF[na][el1][istress][idx1]*conj(cnlm[na][el2][idx2])) +
															 (cnlm[na][el1][idx1]*conj(dcnlm_dF[na][el2][istress][idx2])));
							}
						}
						count_X3++;
					}
				}
			}
			
		}

	}



	free(Ylm);
	free(dYlm_theta);
	free(dYlm_phi);
	free(YD_hnl);
	free(YD_dhnl);
	free(hnl_temp);
	free(dhnl_temp);
	free(rtemp);

	// free the memory for cnlm
	for (int i = 0; i < soap_str->natom_domain; i++){
		for (int j = 0; j < nelem; j++){
			free(cnlm[i][j]);
		}
		free(cnlm[i]);

		for (int j = 0; j < nelem; j++){
			uniq_el_atms = nlist->unique_Nneighbors_elemWise[i][j];
			for (int k = 0; k < 1+uniq_el_atms; k++){
				free(dcnlm_dX[i][j][k]); free(dcnlm_dY[i][j][k]); free(dcnlm_dZ[i][j][k]);
			}
			for (int k = 0; k < 6; k++){
				free(dcnlm_dF[i][j][k]);
			}
			free(dcnlm_dX[i][j]); free(dcnlm_dY[i][j]); free(dcnlm_dZ[i][j]); free(dcnlm_dF[i][j]); 
		}
		free(dcnlm_dX[i]); free(dcnlm_dY[i]); free(dcnlm_dZ[i]); free(dcnlm_dF[i]);
	}
	free(dcnlm_dX); free(dcnlm_dY); free(dcnlm_dZ); free(dcnlm_dF); free(cnlm);

	free(temp_dcnlm_dF);


}


/*
Calculates derivative of theta and phi with x,y, and z

[Input]
1. dx, dy, dz, dr: Cartesian coordinates

[Output]
1. dth_dxi, dphi_dxi: Derivatives
*/

void calculate_dtheta_dphi(double dx, double dy, double dz, double dr, double *dtheta, double *dphi, double *dth_dxi, double *dphi_dxi){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (fabs(dz) <= temp_tol){
		*dtheta = M_PI/2.0;
	}  else {
		*dtheta = atan(sqrt(dx*dx + dy*dy)/dz);
	}
	if (*dtheta<0.0) *dtheta += M_PI;

	double cth = cos(*dtheta);
	double sth = sin(*dtheta);

	double cph, sph;

	*dphi = atan(dy/dx);
	if (dx>0.0 && dy>0.0) *dphi = *dphi;  // 1st Quandrant
	if (dx<0.0 && dy>0.0) *dphi += M_PI;  // 2nd Quandrant
	if (dx<0.0 && dy<0.0) *dphi += M_PI;  // 3rd Quadrant
	if (dx>0.0 && dy<0.0) *dphi += 2.0*M_PI;  // 4th Quadrant

	if (fabs(dx) ==0 && dy >0.0){
		*dphi = M_PI/2.0;
	}
	if (fabs(dx) == 0&& dy <0.0){
		*dphi = 3.0*M_PI/2.0;
	}
	if (fabs(dx) == 0 && fabs(dy) == 0.0){
		*dphi = 0.0;
	}

	// if (fabs(*dtheta) < temp_tol || fabs(*dtheta - M_PI) < temp_tol){ // This condition is for z-axis (dphi undefined, 1/sth -> infty)
	// // dth not needed as dY_dth is always zero for this condition.
	// // dphi not needed as dY_dph is wlways zero for this condition.
	// 	// srand(rank+1);
	// 	// double rand_min = 0.0, rand_max = 2.0*M_PI;
	// 	// *dphi =  rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX; // or 1.0 // to avoid default NAN, will be random as dx and dy are both ~ 0
	// 	*dphi = 0.0;
	// } 
	cph = cos(*dphi);
	sph = sin(*dphi); 

	dth_dxi[0] = (cth * cph)/dr;
	dth_dxi[1] = (cth * sph)/dr;


	dphi_dxi[0] = -sph/(dr*sth);
	dphi_dxi[1] = cph/(dr*sth);


	dth_dxi[2] = (-sth)/dr;
	dphi_dxi[2] = 0.0;


}
			
			
