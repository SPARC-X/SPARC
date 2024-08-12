#ifndef SPARC_MLFF_INTERFACE_H
#define SPARC_MLFF_INTERFACE_H

#include "isddft.h"
#include "mlff_types.h"
// Updated
void init_MLFF(SPARC_OBJ *pSPARC);
// Updated
void free_MLFF(MLFF_Obj *mlff_str);
// Updated
void MLFF_main(SPARC_OBJ *pSPARC);


void get_domain_decompose_mlff_natom(int natom,
  int nelem,
  int *nAtomv,
  int nprocs,
  int rank,
  int *natom_domain);

void get_domain_decompose_mlff_idx(
  int natom,
  int nelem,
  int *nAtomv,
  int nprocs,
  int rank,
  int natom_domain,
  int *atom_idx_domain,
  int *el_idx_domain);

void pretrain_MLFF_model(
  MLFF_Obj *mlff_str,
  SPARC_OBJ *pSPARC, 
  double **cell_data, 
  double **LatUVec_data,
  double **apos_data, 
  double *Etot_data, 
  double **F_data, 
  double **stress_data, 
  int *natom_data, 
  int **natom_elem_data);


void MLFF_call(SPARC_OBJ* pSPARC);


void MLFF_call_from_MD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);


void MLFF_call_from_MD_only_predict(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);


void sparc_mlff_interface_firstMD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);


void sparc_mlff_interface_addMD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);


void sparc_mlff_interface_predict(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str, double *E_predict, double *F_predict, double *stress_predict, double *bayesian_error);


void write_MLFF_results(SPARC_OBJ *pSPARC);


void reshape_stress(int cell_typ, int *BC, int *index);

void coordinatetransform_map(SPARC_OBJ *pSPARC, int natom, double *coord);

#endif 