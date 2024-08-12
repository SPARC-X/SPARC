#ifndef MLFF_READ_WRITE_H
#define MLFF_READ_WRITE_H

#include "mlff_types.h"
#include "isddft.h"

void intialize_print_MLFF(MLFF_Obj *mlff_str, SPARC_OBJ *pSPARC);
void print_new_ref_structure_MLFF(MLFF_Obj *mlff_str, int nstr, NeighList *nlist, double *atompos, double Etot, double *force, double *stress);
void print_restart_MLFF(MLFF_Obj *mlff_str);
void read_MLFF_files(MLFF_Obj *mlff_str, SPARC_OBJ *pSPARC);
void print_ref_atom_MLFF(MLFF_Obj *mlff_str);
void read_structures_MLFF_data(
    char *fname,
    int nstr,
    int nelem, 
    double **cell_data, 
    double **LatUVec_data,
    double **apos_data, 
    double *Etot_data, 
    double **F_data, 
    double **stress_data,
    int *natom_data, 
    int **natom_elem_data);
#endif
