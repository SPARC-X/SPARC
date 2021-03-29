#ifndef HAMSTRUCT_H
#define HAMSTRUCT_H

#include <enum.h>
#include <libpce_structs.h>

typedef struct
{
    Hybrid_Decomp* const hd;
    FD_Info* const fd_info;
    Veff_Info* const veff_info;
    NonLocal_Info* const nonlocal_info;
    device_type communication_device;
    device_type compute_device;
    MPI_Comm comm;
    double veff_scaling;
    double laplacian_scaling;
    int do_nonlocal;
} Our_Hamiltonian_Struct;

void Our_Hamiltonian(const void* ham_struct, const Psi_Info* const psi_in, Psi_Info* psi_out, double c);
#endif
