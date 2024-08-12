#ifndef SOAP_DESCRIPTOR_H
#define SOAP_DESCRIPTOR_H

#include "mlff_types.h"
#include "isddft.h"



void read_h_nl(const int N, const int L, double *rgrid, double *h_nl, double *dh_nl, SPARC_OBJ *pSPARC);


/*
initialize_nlist function initializes the a objects in the NeighList structure and also allocates memory for the dunamic arrays.

[Input]
1. natom: Number of atoms in the system
2. rcut: Cutoff distance (bohr)
3. nelem: Number of element species 
[Output]
1. nlist: pointer to Neighlist structure to be initialized
*/
void initialize_nlist(NeighList* nlist, int cell_typ, int *BC, double *cell_len, double *LatUVec, double twist, const int natom, const double rcut, const int nelem , const int natom_domain, int *atom_idx_domain, int *el_idx_domain);


/*
lin_search function searches for the first occurence of an element in the array 

[Input]
1. arr: pointer to array
2. n: length of array
3. x: element to search

[Output]
1. i: ID of the element in the array
*/
int lin_search(int *arr, int n, int x);


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
void build_nlist(const double rcut, const int nelem, const int natom, const double * const atompos,
				 int * atomtyp, int cell_typ, int *BC, double *cell_len, double *LatUVec, double twist, double *geometric_ratio, NeighList* nlist, const int natom_domain, int *atom_idx_domain, int *el_idx_domain);


/*
clear_nlist function frees the memory dynamically allocated for Neighlist 

[Input]
1. nlist: pointer to Neighlist structure

[Output]
1. nlist: pointer to Neighlist structure
*/
void clear_nlist(NeighList* nlist, int natom_domain);


/*
uniqueEle function finds number of unique entries in an array

[Input]
1. a: pointer to the array
2. n: size of array

[Output]
1. count: Number of unique entries
*/
int uniqueEle(int* a, int n);



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
void build_soapObj(DescriptorObj *soap_str, NeighList *nlist, double* rgrid, double* h_nl, double* dh_nl, double *atompos, int Nmax, int Lmax,  double xi_3, int N_rgrid);


/*
delete_soapObj function frees the memory dynamically allocated for SoapObj 

[Input]
1. soap_str: pointer to SoapObj structure

[Output]
1. soap_str: pointer to SoapObj structure
*/
void delete_soapObj(DescriptorObj *soap_str);


/*
Calculates derivative of theta and phi with x,y, and z

[Input]
1. dx, dy, dz, dr: Cartesian coordinates

[Output]
1. dth_dxi, dphi_dxi: Derivatives
*/
void calculate_dtheta_dphi(double dx, double dy, double dz, double dr, double *dtheta, double *dphi, double *dth_dxi, double *dphi_dxi);

#endif
