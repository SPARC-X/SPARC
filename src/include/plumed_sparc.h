/**
 * @file    plumed_sparc.h
 * @brief   Optional PLUMED integration for ab initio MD (implemented in plumed_sparc.c when USE_PLUMED).
 */

#ifndef PLUMED_SPARC_H
#define PLUMED_SPARC_H

struct _SPARC_OBJ;

/**
 * Call once after MD is initialized (e.g. after Initialize_MD), before the main MD loop.
 */
void Plumed_Init(struct _SPARC_OBJ *pSPARC);

/**
 * Once per MD iteration after ionic forces are available: run PLUMED for the current structure—
 * collective variables, any monitoring or file output declared in the input, and optional biases
 * that augment pSPARC->forces (the integrator moves ions, not this call). For NPT_NH/NPT_NP,
 * virial after calc() is mapped back to stress/pres (BC==2) and MPI_Bcast. Step index: elecgs_Count.
 */
void Plumed_Evaluate(struct _SPARC_OBJ *pSPARC);

/**
 * Call when MD finishes or is aborted, after the main loop (pair with Plumed_Init).
 */
void Plumed_Finalize(struct _SPARC_OBJ *pSPARC);

#endif /* PLUMED_SPARC_H */
