#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include "mlff_types.h"
#include "isddft.h"

void initialize_descriptor(DescriptorObj *desc_str, MLFF_Obj *mlff_str, NeighList *nlist);

void build_descriptor(DescriptorObj *desc_str, NeighList *nlist, MLFF_Obj *mlff_str, double *atom_pos);

void delete_descriptor(DescriptorObj *desc_str);

#endif