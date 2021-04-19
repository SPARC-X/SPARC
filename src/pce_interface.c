#include <string.h>
#include "isddft.h"
#include <libpce.h>
#include <memory.h>

#if USE_GPU
#include <vnl_gpu.h>
#include <cuda.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if USE_GPU

void NONLOCAL_GPU(min_SPARC_OBJ *pSPARC,  ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,   NLOC_PROJ_OBJ *nlocProj,
    min_SPARC_OBJ **d_SPARC, ATOM_NLOC_INFLUENCE_OBJ **d_Atom_Influence_nloc, NLOC_PROJ_OBJ **d_locProj)
{
    double t1 = MPI_Wtime();
    //TODO: GARBAGE COLLECTION
    cudaMalloc((void **)d_SPARC,                sizeof(min_SPARC_OBJ));
    cudaMalloc((void **)d_Atom_Influence_nloc,  sizeof(ATOM_NLOC_INFLUENCE_OBJ) * pSPARC->Ntypes);


    *d_locProj = (NLOC_PROJ_OBJ*) malloc(sizeof(NLOC_PROJ_OBJ) * pSPARC->Ntypes);
    interface_gpu(pSPARC,              *d_SPARC,
                               Atom_Influence_nloc, *d_Atom_Influence_nloc,
                               nlocProj,            *d_locProj);  

}
#endif

void SPARC2NONLOCAL_interface(const SPARC_OBJ *pSPARC, NonLocal_Info *nl, device_type device)
{
    nl->pSPARC = (min_SPARC_OBJ*) malloc( sizeof(min_SPARC_OBJ) );
    min_SPARC_OBJ *min_SPARC = nl->pSPARC;

    min_SPARC->n_atom = pSPARC->n_atom;
    printf("NAtom: %i\n", nl->pSPARC->n_atom);
    min_SPARC->dV = pSPARC->dV;
    min_SPARC->Ntypes = pSPARC->Ntypes;

    min_SPARC->localPsd = (int *)malloc( pSPARC->Ntypes * sizeof(int) );
    min_SPARC->nAtomv   = (int *)malloc( pSPARC->Ntypes * sizeof(int) );
    min_SPARC->IP_displ = (int *)malloc( sizeof(int) * (pSPARC->n_atom+1));
    //copy
    memcpy(min_SPARC->localPsd,pSPARC->localPsd, sizeof(int)*pSPARC->Ntypes);
    memcpy(min_SPARC->nAtomv,pSPARC->nAtomv, sizeof(int)*pSPARC->Ntypes);
    memcpy(min_SPARC->IP_displ,pSPARC->IP_displ, sizeof(int)*(pSPARC->n_atom+1));


    min_SPARC->lmax = (int *) malloc( pSPARC->Ntypes * sizeof(int) );
    //min_SPARC->partial_sum = (int *) malloc( pSPARC->Ntypes * sizeof(int) );
    min_SPARC->ppl = (int **)malloc( sizeof(int*) * pSPARC->Ntypes );
    min_SPARC->Gamma = (double **)malloc( sizeof(double*) * pSPARC->Ntypes );

    for(int i = 0; i < pSPARC->Ntypes; i++)
    {
        min_SPARC->lmax[i] = pSPARC->psd[i].lmax;
        min_SPARC->ppl[i] = (int*) malloc( sizeof(int) * (pSPARC->psd[i].lmax+1) );
        int ppl_sum = 0;
        for (int j = 0; j <= pSPARC->psd[i].lmax; j++)
        {
          printf("J: %i\n", j);
            (min_SPARC->ppl[i])[j] = pSPARC->psd[i].ppl[j];
            ppl_sum += pSPARC->psd[i].ppl[j];
	          //min_SPARC->partial_sum[j] = ppl_sum;
        }

        min_SPARC->Gamma[i] = (double*) malloc( sizeof(double) * ppl_sum );
        memcpy(min_SPARC->Gamma[i],pSPARC->psd[i].Gamma,sizeof(double) * ppl_sum);
    }

    nl->Atom_Influence_nloc = pSPARC->Atom_Influence_nloc;
    nl->nlocProj = pSPARC->nlocProj;

#if USE_GPU
  if(device == DEVICE_TYPE_DEVICE) {
//TODO Garbage collection
    printf("GPUUUUUUUU\n");


    NONLOCAL_GPU(min_SPARC, nl->Atom_Influence_nloc, nl->nlocProj, &(nl->d_SPARC), &(nl->d_Atom_Influence_nloc), &(nl->d_locProj));

    nl->pSPARC->d_xrc = OUR_MALLOC(nl->pSPARC->Ntypes, double**, DEVICE_TYPE_HOST);
    //printf("ALLOCED d_xrc\n");

    for(int i = 0; i < nl->pSPARC->Ntypes; i++) {
        nl->pSPARC->d_xrc[i] = NULL;
    }
}
#endif


    return;
}

#ifdef __cplusplus
}
#endif
