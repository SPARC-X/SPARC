#ifndef INTERNAL_TOOLS_h
#define INTERNAL_TOOLS_h

#include "mlff_types.h"

void train_internal_energy_model(MLFF_Obj *mlff_str);

/*
Write the (free energy, internal energy) training pairs to mlff_str->ie_model_name (rank 0)
*/
void print_internal_energy_model_data(MLFF_Obj *mlff_str);

/*
Read the training pairs of a previous run from mlff_str->ie_model_name and broadcast them
*/
void read_internal_energy_model_data(MLFF_Obj *mlff_str, int n_max);

#endif