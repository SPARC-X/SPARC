#include <math.h>
#include "isddft.h"
#include <libpce.h>
#include "hamstruct.h"

#ifdef __cplusplus
extern "C" {
#endif

void Our_Hamiltonian(const void* ham_struct, const Psi_Info* const psi_in, Psi_Info* psi_out, double c)
{
  Our_Hamiltonian_Struct* ohs = (Our_Hamiltonian_Struct*)ham_struct;



  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if USE_GPU
if(ohs->compute_device == DEVICE_TYPE_DEVICE) {
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  }
#endif

  const double start = MPI_Wtime();

  const double lap_start = MPI_Wtime();

#if DEBUG
  if(PCE_Internal_debug_level() <= DEBUG_TRACE) {
    double loc_psi = PCE_Internal_sum_abs(psi_in->data,  ohs->hd->local_num_fd,ohs->hd->local_num_cols, ohs->compute_device);
    double glob_psi = 0;
    printf("%i Loc psi pre : %f\n", rank, loc_psi);
    printf("ohs->lap_scaling: %f\n", ohs->laplacian_scaling);

    MPI_Allreduce(&loc_psi, &glob_psi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0) {
    printf("Pre Laplacian psi in: %f\n",glob_psi);
    }
  }
#endif

  // Perform second order laplacian
  PCE_Laplacian(ohs->hd, ohs->fd_info, psi_in, psi_out, 
      ohs->laplacian_scaling,
      ohs->communication_device,
      ohs->compute_device,
      ohs->comm);  // Nabla
#if USE_GPU
if(ohs->compute_device == DEVICE_TYPE_DEVICE) {
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  }
#endif

  const double lap_end = MPI_Wtime();

#if DEBUG
  if(PCE_Internal_debug_level() <= DEBUG_TRACE) {
    printf("hd: %i x %i\n", ohs->hd->local_num_fd, ohs->hd->local_num_cols);
    double loc_psi = PCE_Internal_sum_abs(psi_out->data,  ohs->hd->local_num_fd,ohs->hd->local_num_cols, ohs->compute_device);
    double glob_psi = 0;
    printf("%i post lap Loc psi : %f\n", rank, loc_psi);

    MPI_Allreduce(&loc_psi, &glob_psi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0) {
    printf("%i Post Laplacian psi out: %f\n",rank, glob_psi);
    }
  }
#endif

  if((PCE_Internal_debug_level() <= DEBUG_VERBOSE) && (rank==0)) {
    printf("Laplacian time: %f\n", lap_end - lap_start);
  }

#if DEBUG
  if(PCE_Internal_debug_level() <= DEBUG_TRACE) {
    printf("Post Lap psi out: %f\n",
        PCE_Internal_sum_abs(psi_out->data,  ohs->hd->local_num_fd,ohs->hd->local_num_cols, ohs->compute_device));
  }
#endif

  const double veff_start = MPI_Wtime();

  // Apply Veff
  if(fabs(ohs->veff_scaling) > 1e-12) {
  PCE_Veff_Apply_Inplace(ohs->veff_info, ohs->hd->local_num_cols, ohs->hd->local_num_fd, psi_in, psi_out,
      ohs->veff_scaling, ohs->compute_device);  // V_XC, V_H
  }
#if USE_GPU
if(ohs->compute_device == DEVICE_TYPE_DEVICE) {
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  }
#endif


  const double veff_end = MPI_Wtime();



  if((PCE_Internal_debug_level() <= DEBUG_VERBOSE) && (rank==0)) {
    printf("Veff time: %f\n", veff_end - veff_start);
  }

#if DEBUG
  if(PCE_Internal_debug_level() <= DEBUG_TRACE) {
    double loc_psi = PCE_Internal_sum_abs(psi_out->data,  ohs->hd->local_num_fd,ohs->hd->local_num_cols, ohs->compute_device);
    double glob_psi = 0;
    MPI_Allreduce(&loc_psi, &glob_psi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0) {
    printf("Post Veff psi out: %f\n",glob_psi);
    }
  }
#endif

  if(c != 0) {
    const double shift_start = MPI_Wtime();
    PCE_Internal_Shift(psi_in, psi_out, ohs->hd->local_num_cols * ohs->hd->local_num_fd, -c);
#if USE_GPU
if(ohs->compute_device == DEVICE_TYPE_DEVICE) {
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  }
#endif
    const double shift_end = MPI_Wtime();

    if((PCE_Internal_debug_level() <= DEBUG_VERBOSE) && (rank==0)) {
      printf("shift time: %f\n", shift_end - shift_start);
    }

  }


  const double nl_start = MPI_Wtime();

  if(ohs->do_nonlocal) {
  PCE_NonLocal_Apply(ohs->nonlocal_info, ohs->hd, psi_in, psi_out, ohs->communication_device, ohs->compute_device,
      ohs->comm);  // V_ion
  }

#if USE_GPU
if(ohs->compute_device == DEVICE_TYPE_DEVICE) {
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  }
#endif

  const double nl_end = MPI_Wtime();

  if((PCE_Internal_debug_level() <= DEBUG_VERBOSE) && (rank==0)) {
    printf("NonLocal time: %f\n", nl_end - nl_start);
  }

#if DEBUG
  if(PCE_Internal_debug_level() <= DEBUG_TRACE) {
    printf("Post nonlocal out: %f\n",
        PCE_Internal_sum_abs(psi_out->data,  ohs->hd->local_num_fd,ohs->hd->local_num_cols, ohs->compute_device));
  }
#endif

  const double end = MPI_Wtime();

  if((PCE_Internal_debug_level() <= DEBUG_VERBOSE) && (rank==0)) {
    printf("HAM time: %f\n", end - start);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#if USE_GPU
if(ohs->compute_device == DEVICE_TYPE_DEVICE) {
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  }
#endif
}

#ifdef __cplusplus
}
#endif
