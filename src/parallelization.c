/**
 * @file    parallelization.c
 * @brief   This file contains the functions related to parallelization.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#define _XOPEN_SOURCE 500 // For srand48(), drand48(), usleep()

// For sched_setaffinity
#define  _GNU_SOURCE
#include <sched.h>
#include <unistd.h>  // Also for usleep()

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <complex.h>
/* ScaLAPACK routines */
#ifdef USE_MKL
    #include "blacs.h"     // Cblacs_*
    #include <mkl_blacs.h>
    #include <mkl_pblas.h>
    #include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
    #include "blacs.h"     // Cblacs_*
    #include "scalapack.h" // include ScaLAPACK function declarations
#endif

#include "parallelization.h"
#include "tools.h"
#include "isddft.h"
#include "initialization.h"
#include "electrostatics.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/**
 * @brief   Set up sub-communicators.
 */
void Setup_Comms(SPARC_OBJ *pSPARC) {
    int i, j, dims[3] = {0, 0, 0}, periods[3], ierr;
    int nproc, rank;
    int size_spincomm, rank_spincomm;
    int nproc_kptcomm, rank_kptcomm, size_kptcomm;
    int nproc_bandcomm, rank_bandcomm, size_bandcomm, NP_BANDCOMM, NB;
    int npNd, gridsizes[3], minsize, coord_dmcomm[3], rank_dmcomm;
    int color;
#ifdef DEBUG
    double t1, t2;
#endif
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Set up communicators.\n");
#endif

    // in the case user doesn't set any parallelization parameters, search for a good
    // combination in advance
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    npNd = pSPARC->npNdx * pSPARC->npNdy * pSPARC->npNdz;
    minsize = pSPARC->order/2;
    if (pSPARC->npspin == 0 && pSPARC->npkpt == 0 && 
        pSPARC->npband == 0 && npNd == 0) 
    {
        dims_divide_skbd(pSPARC->Nspin, pSPARC->Nkpts_sym, pSPARC->Nstates, 
            gridsizes, nproc, &pSPARC->npspin, &pSPARC->npkpt, &pSPARC->npband, &npNd, minsize, pSPARC->usefock);
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        pSPARC->npNdx = dims[0];
        pSPARC->npNdy = dims[1];
        pSPARC->npNdz = dims[2];
        pSPARC->useDefaultParalFlag = 1;
    } else {
        if (!rank) printf("WARNING: Default parallelization not used. This could result in degradation of performance.\n");
        pSPARC->useDefaultParalFlag = 0;
    }

    //------------------------------------------------//
    //                 set up spincomm                //
    //------------------------------------------------//
    // Allocate number of spin communicators
    // if npspin is not provided by user, or the given npspin is too large
    if (pSPARC->npspin == 0) {
        pSPARC->npspin = min(nproc, pSPARC->Nspin);
    } else if (pSPARC->npspin > pSPARC->Nspin || pSPARC->npspin > nproc) {
        pSPARC->npspin = min(nproc, pSPARC->Nspin);
        if (rank == 0) {
            printf("WARNING: npspin is larger than pSPARC->Nspin or nproc!\n"
                   "         Forcing npspin = min(nproc, pSPARC->Nspin) = %d.\n\n",pSPARC->npspin);
        }
    }

    // Allocate number of processors and their local indices in a spin communicator
    size_spincomm = nproc / pSPARC->npspin;
    if (rank < (nproc - nproc % pSPARC->npspin))
        pSPARC->spincomm_index = rank / size_spincomm;
    else
        pSPARC->spincomm_index = -1;

    // calculate number of spin assigned to each spincomm
    if (rank < (nproc - nproc % pSPARC->npspin)) {
        pSPARC->Nspin_spincomm = pSPARC->Nspin / pSPARC->npspin;
        pSPARC->Nspinor_spincomm = pSPARC->Nspinor / pSPARC->npspin;
    } else {
        pSPARC->Nspin_spincomm = 0;
        pSPARC->Nspinor_spincomm = 0;
    }

    // calculate start and end indices of the spin obtained by each spincomm
    if (pSPARC->spincomm_index == -1) {
        pSPARC->spin_start_indx = 0;
        pSPARC->spinor_start_indx = 0;
    } else {
        pSPARC->spin_start_indx = pSPARC->spincomm_index * pSPARC->Nspin_spincomm;
        pSPARC->spinor_start_indx = pSPARC->spincomm_index * pSPARC->Nspinor_spincomm;
    }
    pSPARC->spin_end_indx = pSPARC->spin_start_indx + pSPARC->Nspin_spincomm - 1;
    pSPARC->spinor_end_indx = pSPARC->spinor_start_indx + pSPARC->Nspinor_spincomm - 1;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // split the MPI_COMM_WORLD into spincomms using color = spincomm_index
    color = (pSPARC->spincomm_index >= 0) ? pSPARC->spincomm_index : INT_MAX; 
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &pSPARC->spincomm);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up spincomm took %.3f ms\n",(t2-t1)*1000);
#endif
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);

    //------------------------------------------------------//
    //              set up spin_bridge_comm                  //
    //------------------------------------------------------//
    // spin_bridge_comm contains all processes with the same rank_spincomm
    if (rank < nproc - nproc % pSPARC->npspin) {
        color = rank_spincomm;
    } else {
        color = INT_MAX;
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, pSPARC->spincomm_index, &pSPARC->spin_bridge_comm);

    //------------------------------------------------//
    //                 set up kptcomm                 //
    //------------------------------------------------//
    // if npkpt is not provided by user, or the given npkpt is too large
    if (pSPARC->npkpt == 0) {
        pSPARC->npkpt = min(size_spincomm, pSPARC->Nkpts_sym); // paral over k-points as much as possible
    } else if (pSPARC->npkpt > pSPARC->Nkpts_sym || pSPARC->npkpt > size_spincomm) {
        pSPARC->npkpt = min(size_spincomm, pSPARC->Nkpts_sym);
        if (rank == 0) {
            printf("WARNING: npkpt is larger than number of k-points after symmetry reduction or size_spincomm!\n"
                   "         Forcing npkpt = min(size_spincomm, Nkpts_sym) = %d.\n\n",pSPARC->npkpt);
        }
    }

    size_kptcomm = size_spincomm / pSPARC->npkpt; // size of kptcomm
    if (pSPARC->spincomm_index != -1 && rank_spincomm < (size_spincomm - size_spincomm % pSPARC->npkpt))
        pSPARC->kptcomm_index = rank_spincomm / size_kptcomm;
    else
        pSPARC->kptcomm_index = -1;

    // calculate number of k-points assigned to each kptcomm
    if (rank_spincomm < (size_spincomm - size_spincomm % pSPARC->npkpt))
        pSPARC->Nkpts_kptcomm = pSPARC->Nkpts_sym / pSPARC->npkpt + (int) (pSPARC->kptcomm_index < (pSPARC->Nkpts_sym % pSPARC->npkpt));
    else
        pSPARC->Nkpts_kptcomm = 0;

    // calculate start and end indices of the k-points obtained by each kptcomm
    if (pSPARC->kptcomm_index == -1) {
        pSPARC->kpt_start_indx = 0;
    } else if (pSPARC->kptcomm_index < (pSPARC->Nkpts_sym % pSPARC->npkpt)) {
        pSPARC->kpt_start_indx = pSPARC->kptcomm_index * pSPARC->Nkpts_kptcomm;
    } else {
        pSPARC->kpt_start_indx = pSPARC->kptcomm_index * pSPARC->Nkpts_kptcomm + pSPARC->Nkpts_sym % pSPARC->npkpt;
    }
    pSPARC->kpt_end_indx = pSPARC->kpt_start_indx + pSPARC->Nkpts_kptcomm - 1;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // split the pSPARC->spincomm into several kptcomms using color = kptcomm_index
    color = (pSPARC->kptcomm_index >= 0) ? pSPARC->kptcomm_index : INT_MAX;
    MPI_Comm_split(pSPARC->spincomm, color, 0, &pSPARC->kptcomm);

    //setup_core_affinity(pSPARC->kptcomm);
    //bind_proc_to_phys_core();

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up kptcomm took %.3f ms\n",(t2-t1)*1000);
#endif

    // Local k-points array
    if (pSPARC->Nkpts >= 1 && pSPARC->kptcomm_index != -1) {
        // allocate memory for storing local k-points and the weights for k-points
        //if (pSPARC->BC != 1) {
            pSPARC->kptWts_loc = (double *)malloc(pSPARC->Nkpts_kptcomm * sizeof(double));
            pSPARC->k1_loc = (double *)malloc(pSPARC->Nkpts_kptcomm * sizeof(double));
            pSPARC->k2_loc = (double *)malloc(pSPARC->Nkpts_kptcomm * sizeof(double));
            pSPARC->k3_loc = (double *)malloc(pSPARC->Nkpts_kptcomm * sizeof(double));
            // calculate the k point weights
            Calculate_local_kpoints(pSPARC);
        //}
    }

    //-------------------------------------------------------//
    //    set up a Cartesian topology within each kptcomm    //
    //-------------------------------------------------------//
    MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);
    if (pSPARC->kptcomm_index < 0) nproc_kptcomm = size_kptcomm; // let all proc know the size

    /* first calculate the best dimensions using SPARC_Dims_create */
    // gridsizes = {Nx, Ny, Nz}, minsize, and periods are the same as above
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
    minsize = pSPARC->order/2;

    // calculate dims[]
    SPARC_Dims_create(nproc_kptcomm, 3, gridsizes, minsize, dims, &ierr);

    pSPARC->npNdx_kptcomm = dims[0];
    pSPARC->npNdy_kptcomm = dims[1];
    pSPARC->npNdz_kptcomm = dims[2];

#ifdef DEBUG
    if (!rank)
    printf("\n kpt_topo #%d, kptcomm topology dims = {%d, %d, %d}, nodes/proc = {%.2f,%.2f,%.2f}\n", pSPARC->kptcomm_index,
            dims[0],dims[1],dims[2],(double)gridsizes[0]/dims[0],(double)gridsizes[1]/dims[1],(double)gridsizes[2]/dims[2]);
#endif

    if (pSPARC->kptcomm_index >= 0) {
        //Processes in kptcomm with rank >= dims[0]*dims[1]*…*dims[d-1] will return MPI_COMM_NULL
        MPI_Cart_create(pSPARC->kptcomm, 3, dims, periods, 1, &pSPARC->kptcomm_topo); // 1 is to reorder rank
    } else {
        pSPARC->kptcomm_topo = MPI_COMM_NULL;
    }

    int rank_kpt_topo, coord_kpt_topo[3];
    if (pSPARC->kptcomm_topo != MPI_COMM_NULL) {
        MPI_Comm_rank(pSPARC->kptcomm_topo, &rank_kpt_topo);
        MPI_Cart_coords(pSPARC->kptcomm_topo, rank_kpt_topo, 3, coord_kpt_topo);

        // find size of distributed domain over kptcomm_topo
        pSPARC->Nx_d_kptcomm = block_decompose(gridsizes[0], dims[0], coord_kpt_topo[0]);
        pSPARC->Ny_d_kptcomm = block_decompose(gridsizes[1], dims[1], coord_kpt_topo[1]);
        pSPARC->Nz_d_kptcomm = block_decompose(gridsizes[2], dims[2], coord_kpt_topo[2]);
        pSPARC->Nd_d_kptcomm = pSPARC->Nx_d_kptcomm * pSPARC->Ny_d_kptcomm * pSPARC->Nz_d_kptcomm;

        // find corners of the distributed domain in kptcomm_topo
        pSPARC->DMVertices_kptcomm[0] = block_decompose_nstart(gridsizes[0], dims[0], coord_kpt_topo[0]);
        pSPARC->DMVertices_kptcomm[1] = pSPARC->DMVertices_kptcomm[0] + pSPARC->Nx_d_kptcomm - 1;
        pSPARC->DMVertices_kptcomm[2] = block_decompose_nstart(gridsizes[1], dims[1], coord_kpt_topo[1]);
        pSPARC->DMVertices_kptcomm[3] = pSPARC->DMVertices_kptcomm[2] + pSPARC->Ny_d_kptcomm - 1;
        pSPARC->DMVertices_kptcomm[4] = block_decompose_nstart(gridsizes[2], dims[2], coord_kpt_topo[2]);
        pSPARC->DMVertices_kptcomm[5] = pSPARC->DMVertices_kptcomm[4] + pSPARC->Nz_d_kptcomm - 1;
    } else {
        rank_kpt_topo = -1;
        coord_kpt_topo[0] = -1; coord_kpt_topo[1] = -1; coord_kpt_topo[2] = -1;
        pSPARC->Nx_d_kptcomm = 0;
        pSPARC->Ny_d_kptcomm = 0;
        pSPARC->Nz_d_kptcomm = 0;
        pSPARC->Nd_d_kptcomm = 0;
        pSPARC->DMVertices_kptcomm[0] = 0;
        pSPARC->DMVertices_kptcomm[1] = 0;
        pSPARC->DMVertices_kptcomm[2] = 0;
        pSPARC->DMVertices_kptcomm[3] = 0;
        pSPARC->DMVertices_kptcomm[4] = 0;
        pSPARC->DMVertices_kptcomm[5] = 0;
        //pSPARC->Nband_bandcomm = 0;
    }

    if(pSPARC->cell_typ != 0) {
        if(pSPARC->kptcomm_topo != MPI_COMM_NULL) {
            int rank_chk, dir, tmp;
            int ncoords[3];
            int nnproc = 1; // No. of neighboring processors in each direction
            int nneighb = pow(2*nnproc+1,3) - 1; // 6 faces + 12 edges + 8 corners
            int *neighb;
            neighb = (int *) malloc(nneighb * sizeof (int));
            int count = 0, i, j, k;
            for(k = -nnproc; k <= nnproc; k++){
                for(j = -nnproc; j <= nnproc; j++){
                    for(i = -nnproc; i <= nnproc; i++){
                        tmp = 0;
                        if(i == 0 && j == 0 && k == 0){
                            continue;
                        } else{
                            ncoords[0] = coord_kpt_topo[0] + i;
                            ncoords[1] = coord_kpt_topo[1] + j;
                            ncoords[2] = coord_kpt_topo[2] + k;
                            for(dir = 0; dir < 3; dir++){
                                if(periods[dir]){
                                    if(ncoords[dir] < 0)
                                        ncoords[dir] += dims[dir];
                                    else if(ncoords[dir] >= dims[dir])
                                        ncoords[dir] -= dims[dir];
                                    tmp = 1;
                                } else{
                                    if(ncoords[dir] < 0 || ncoords[dir] >= dims[dir]){
                                        rank_chk = MPI_PROC_NULL;
                                        tmp = 0;
                                        break;
                                    }
                                    else
                                        tmp = 1;
                                }
                            }
                            //TODO: For dirchlet give rank = MPI_PROC_NULL for out of bounds coordinates
                            if(tmp == 1)
                                MPI_Cart_rank(pSPARC->kptcomm_topo,ncoords,&rank_chk); // proc rank corresponding to ncoords_mapped

                            neighb[count] = rank_chk;
                            count++;
                        }
                    }
                }
            }
            MPI_Dist_graph_create_adjacent(pSPARC->kptcomm_topo,nneighb,neighb,(int *)MPI_UNWEIGHTED,nneighb,neighb,(int *)MPI_UNWEIGHTED,MPI_INFO_NULL,0,&pSPARC->kptcomm_topo_dist_graph);
            free(neighb);
        }
    }

    //------------------------------------------------------------------------------------//
    //    set up inter-communicators between the Cart topology and the rest in kptcomm    //
    //------------------------------------------------------------------------------------//
    int nproc_kptcomm_topo;
    nproc_kptcomm_topo = pSPARC->npNdx_kptcomm * pSPARC->npNdy_kptcomm * pSPARC->npNdz_kptcomm;
    if (nproc_kptcomm_topo < nproc_kptcomm && pSPARC->kptcomm_index >= 0) {
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        // first create a comm that includes all the processes that are excluded from the Cart topology
        MPI_Group kptgroup, kptgroup_excl;
        MPI_Comm_group(pSPARC->kptcomm, &kptgroup);
        int *incl_ranks, count;
        incl_ranks = (int *)malloc((nproc_kptcomm - nproc_kptcomm_topo) * sizeof(int));
        count = 0;
        for (i = nproc_kptcomm_topo; i < nproc_kptcomm; i++) {
            incl_ranks[count] = i; count++;
        }
        MPI_Group_incl(kptgroup, count, incl_ranks, &kptgroup_excl);
        MPI_Comm_create_group(pSPARC->kptcomm, kptgroup_excl, 110, &pSPARC->kptcomm_topo_excl);

        // now create an inter-comm between kptcomm_topo and kptcomm_topo_excl
        if (pSPARC->kptcomm_topo != MPI_COMM_NULL) {
            MPI_Intercomm_create(pSPARC->kptcomm_topo, 0, pSPARC->kptcomm, nproc_kptcomm_topo, 111, &pSPARC->kptcomm_inter);
        } else {
            MPI_Intercomm_create(pSPARC->kptcomm_topo_excl, 0, pSPARC->kptcomm, 0, 111, &pSPARC->kptcomm_inter);
        }

        free(incl_ranks);
        MPI_Group_free(&kptgroup);
        MPI_Group_free(&kptgroup_excl);

#ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\n--set up kptcomm_inter took %.3f ms\n" ,(t2-t1)*1000);
#endif
    } else {
        pSPARC->kptcomm_topo_excl = MPI_COMM_NULL;
        pSPARC->kptcomm_inter = MPI_COMM_NULL;
    }

    //------------------------------------------------------//
    //              set up kpt_bridge_comm                  //
    //------------------------------------------------------//
    // kpt_bridge_comm contains all processes with the same rank_kptcomm
    if (rank_spincomm < size_spincomm - size_spincomm % pSPARC->npkpt) {
        color = rank_kptcomm;
    } else {
        color = INT_MAX;
    }
    MPI_Comm_split(pSPARC->spincomm, color, pSPARC->kptcomm_index, &pSPARC->kpt_bridge_comm); // TODO: exclude null kptcomms

    //------------------------------------------------//
    //               set up bandcomm                  //
    //------------------------------------------------//
    // in each kptcomm, create sub-communicators over bands
    // note: different from kptcomms, we require the bandcomms in the same kptcomm to be of the same size
    if (pSPARC->npband == 0) {
        pSPARC->npband = min(nproc_kptcomm, pSPARC->Nstates); // paral over band as much as possible
    } else if (pSPARC->npband > nproc_kptcomm) { 
        // here we allow user to give npband larger than Nstates!
        pSPARC->npband = min(nproc_kptcomm, pSPARC->Nstates);
    }

    size_bandcomm = nproc_kptcomm / pSPARC->npband; // size of each bandcomm
    NP_BANDCOMM = pSPARC->npband * size_bandcomm; // number of processors that belong to a bandcomm, others are excluded
    // calculate which bandcomm processor with RANK = rank_kptcomm belongs to
    if (rank_kptcomm < NP_BANDCOMM && pSPARC->kptcomm_index != -1) {
        //pSPARC->bandcomm_index = rank_kptcomm / size_bandcomm; // assign processes column-wisely
        pSPARC->bandcomm_index = rank_kptcomm % pSPARC->npband; // assign processes row-wisely
    } else {
        pSPARC->bandcomm_index = -1; // these processors won't be used to do calculations
    }
    // calculate number of bands assigned to each bandcomm, this is a special case of block-cyclic distribution (*,block)
    if (pSPARC->bandcomm_index == -1) {
        pSPARC->Nband_bandcomm = 0;
    } else {
        NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // this is equal to ceil(Nstates/npband), for int inputs only
        pSPARC->Nband_bandcomm = pSPARC->bandcomm_index < (pSPARC->Nstates / NB) ? NB : (pSPARC->bandcomm_index == (pSPARC->Nstates / NB) ? (pSPARC->Nstates % NB) : 0);
    }

    // calculate start and end indices of the bands obtained by each kptcomm
    if (pSPARC->bandcomm_index == -1) {
        pSPARC->band_start_indx = 0;
    } else if (pSPARC->bandcomm_index <= (pSPARC->Nstates / NB)) {
        pSPARC->band_start_indx = pSPARC->bandcomm_index * NB;
    } else {
        pSPARC->band_start_indx = pSPARC->Nstates; // TODO: this might be dangerous, consider using 0, instead of Ns here
    }
    pSPARC->band_end_indx = pSPARC->band_start_indx + pSPARC->Nband_bandcomm - 1;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // split the kptcomm into several bandcomms using color = bandcomm_index
    color = (pSPARC->bandcomm_index >= 0) ? pSPARC->bandcomm_index : INT_MAX;
    MPI_Comm_split(pSPARC->kptcomm, color, 0, &pSPARC->bandcomm);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up bandcomm took %.3f ms\n",(t2-t1)*1000);
#endif

    //------------------------------------------------//
    //               set up domaincomm                //
    //------------------------------------------------//
    MPI_Comm_size(pSPARC->bandcomm, &nproc_bandcomm);
    MPI_Comm_rank(pSPARC->bandcomm, &rank_bandcomm);
    // let all proc know the size of bandcomm
    if (pSPARC->bandcomm_index < 0) nproc_bandcomm = size_bandcomm; 

    npNd = pSPARC->npNdx * pSPARC->npNdy * pSPARC->npNdz;
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    minsize = pSPARC->order/2;

    if (npNd == 0)  {
        // when user does not provide domain decomposition parameters
        npNd = nproc_bandcomm;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        pSPARC->npNdx = dims[0];
        pSPARC->npNdy = dims[1];
        pSPARC->npNdz = dims[2];
    } else if (npNd < 0 || npNd > nproc_bandcomm || pSPARC->Nx / pSPARC->npNdx < minsize ||
               pSPARC->Ny / pSPARC->npNdy < minsize || pSPARC->Nz / pSPARC->npNdz < minsize) {
        // when domain decomposition parameters are not reasonable
        npNd = nproc_bandcomm;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        pSPARC->npNdx = dims[0];
        pSPARC->npNdy = dims[1];
        pSPARC->npNdz = dims[2];
    } else {
        dims[0] = pSPARC->npNdx;
        dims[1] = pSPARC->npNdy;
        dims[2] = pSPARC->npNdz;
    }

    // recalculate number of processors in each dmcomm
    npNd = pSPARC->npNdx * pSPARC->npNdy * pSPARC->npNdz;
    pSPARC->npNd = npNd;

    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
#ifdef DEBUG
    if (!rank) printf("rank = %d, dmcomm dims = {%d, %d, %d}\n", rank, pSPARC->npNdx, pSPARC->npNdy, pSPARC->npNdz);
    t1 = MPI_Wtime();
#endif

    // Processes in bandcomm with rank >= dims[0]*dims[1]*…*dims[d-1] will return MPI_COMM_NULL
    if (pSPARC->bandcomm_index != -1) {
        MPI_Cart_create(pSPARC->bandcomm, 3, dims, periods, 1, &pSPARC->dmcomm); // 1 is to reorder rank
    } else {
        pSPARC->dmcomm = MPI_COMM_NULL;
    }

#ifdef DEBUG
    if (!rank) printf("gridsizes = [%d, %d, %d], Nstates = %d, dmcomm dims = [%d, %d, %d]\n",
        gridsizes[0],gridsizes[1],gridsizes[2],pSPARC->Nstates,dims[0],dims[1],dims[2]);
#endif

    if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index != -1) {
        MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);
        MPI_Cart_coords(pSPARC->dmcomm, rank_dmcomm, 3, coord_dmcomm);

        // find size of distributed domain over dmcomm
        pSPARC->Nx_d_dmcomm = block_decompose(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->Ny_d_dmcomm = block_decompose(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->Nz_d_dmcomm = block_decompose(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->Nd_d_dmcomm = pSPARC->Nx_d_dmcomm * pSPARC->Ny_d_dmcomm * pSPARC->Nz_d_dmcomm;

        // find corners of the distributed domain in dmcomm
        pSPARC->DMVertices_dmcomm[0] = block_decompose_nstart(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->DMVertices_dmcomm[1] = pSPARC->DMVertices_dmcomm[0] + pSPARC->Nx_d_dmcomm - 1;
        pSPARC->DMVertices_dmcomm[2] = block_decompose_nstart(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->DMVertices_dmcomm[3] = pSPARC->DMVertices_dmcomm[2] + pSPARC->Ny_d_dmcomm - 1;
        pSPARC->DMVertices_dmcomm[4] = block_decompose_nstart(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->DMVertices_dmcomm[5] = pSPARC->DMVertices_dmcomm[4] + pSPARC->Nz_d_dmcomm - 1;

    } else {
        rank_dmcomm = -1;
        coord_dmcomm[0] = -1; coord_dmcomm[1] = -1; coord_dmcomm[2] = -1;
        pSPARC->Nx_d_dmcomm = 0;
        pSPARC->Ny_d_dmcomm = 0;
        pSPARC->Nz_d_dmcomm = 0;
        pSPARC->Nd_d_dmcomm = 0;
        pSPARC->DMVertices_dmcomm[0] = 0;
        pSPARC->DMVertices_dmcomm[1] = 0;
        pSPARC->DMVertices_dmcomm[2] = 0;
        pSPARC->DMVertices_dmcomm[3] = 0;
        pSPARC->DMVertices_dmcomm[4] = 0;
        pSPARC->DMVertices_dmcomm[5] = 0;
        //pSPARC->Nband_bandcomm = 0;
    }
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up dmcomm took %.3f ms\n",(t2-t1)*1000);
#endif

    // Set up a 26 neighbor communicator for nonorthogonal systems
    // TODO: Modify the communicator based on number of non zero enteries in off diagonal of pSPARC->lapcT
    // TODO: Take into account more than 26 Neighbors
    if(pSPARC->cell_typ != 0) {
        if(pSPARC->dmcomm != MPI_COMM_NULL) {
            int rank_chk, dir, tmp;
            int ncoords[3];
            int nnproc = 1; // No. of neighboring processors in each direction
            int nneighb = pow(2*nnproc+1,3) - 1; // 6 faces + 12 edges + 8 corners
            int *neighb;
            neighb = (int *) malloc(nneighb * sizeof (int));
            int count = 0, i, j, k;

            for(k = -nnproc; k <= nnproc; k++){
                for(j = -nnproc; j <= nnproc; j++){
                    for(i = -nnproc; i <= nnproc; i++){
                        tmp = 0;
                        if(i == 0 && j == 0 && k == 0){
                            continue;
                        } else{
                            ncoords[0] = coord_dmcomm[0] + i;
                            ncoords[1] = coord_dmcomm[1] + j;
                            ncoords[2] = coord_dmcomm[2] + k;
                            for(dir = 0; dir < 3; dir++){
                                if(periods[dir]){
                                    if(ncoords[dir] < 0)
                                        ncoords[dir] += dims[dir];
                                    else if(ncoords[dir] >= dims[dir])
                                        ncoords[dir] -= dims[dir];
                                    tmp = 1;
                                } else{
                                    if(ncoords[dir] < 0 || ncoords[dir] >= dims[dir]){
                                        rank_chk = MPI_PROC_NULL;
                                        tmp = 0;
                                        break;
                                    }
                                    else
                                        tmp = 1;
                                }
                            }
                            //TODO: For dirchlet give rank = MPI_PROC_NULL for out of bounds coordinates
                            if(tmp == 1)
                                MPI_Cart_rank(pSPARC->dmcomm,ncoords,&rank_chk); // proc rank corresponding to ncoords_mapped

                            neighb[count] = rank_chk;
                            count++;
                        }
                    }
                }
            }
            MPI_Dist_graph_create_adjacent(pSPARC->dmcomm,nneighb,neighb,(int *)MPI_UNWEIGHTED,nneighb,neighb,(int *)MPI_UNWEIGHTED,MPI_INFO_NULL,0,&pSPARC->comm_dist_graph_psi); // creates a distributed graph topology (adjacent, cartesian cubical)
            //pSPARC->dmcomm_phi = pSPARC->comm_dist_graph_phi;
            free(neighb);
        }
    }

    //------------------------------------------------//
    //                set up blacscomm                //
    //------------------------------------------------//
    // There are two ways to set up the blacscomm:
    // 1. Let the group of all processors dealing with the same local domain (but different
    //    bands) be a blacscomm. We will find Psi_i' * Psi_i in each blacscomm, and sum
    //    over all local domains (dmcomm) to find Psi' * Psi. This way, however, doesn't
    //    seem to be efficient since the sum (Allreduce) part will take a lot of time. In
    //    some cases even more than the computation time.
    // 2. In stead of having many parallel blacscomm in each kptcomm, we create only one
    //    blacscomm, containing all the parallel blacscomm described above. This way
    //    uses a little trick of linear algebra: since the rows are reordered when we
    //    do domain parallelization, the matrix Psi is interpretated as P * Psi by the
    //    ScaLAPACK routines distributed block-wisely, where P is a permutation matrix
    //    (with 0's and 1's). But, since P is orthogonal, we could still preceed and find
    //    (P * Psi)' * (P * Psi) = Psi' * (P'*P) * Psi = Psi' * Psi, and the result will
    //    still be the same as Psi'*Psi. P.S., however, this requires the domain partition
    //    to be UNIFORM!

    // The following code sets up blacscomm in the 1st way described above
    color = rank_dmcomm;
    if (pSPARC->bandcomm_index == -1 || pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->kptcomm_index == -1)   color = INT_MAX;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // split the kptcomm into several cblacscomms using color = rank_dmcomm
    color = (color >= 0) ? color : INT_MAX;
    MPI_Comm_split(pSPARC->kptcomm, color, pSPARC->bandcomm_index, &pSPARC->blacscomm);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up blacscomm took %.3f ms\n",(t2-t1)*1000);
#endif

#if defined(USE_MKL) || defined(USE_SCALAPACK)
    // copy this when need to create cblacs context using Cblacs_gridmap
    //int *usermap, ldumap;
    int size_blacscomm, DMnd, DMndsp, DMndspe;
    int *usermap, *usermap_0, *usermap_1;
    int info, bandsizes[2], nprow, npcol, myrow, mycol;

    size_blacscomm = pSPARC->npband;
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    DMndspe = DMnd * pSPARC->Nspinor_eig;

    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        usermap = (int *)malloc(sizeof(int)*size_blacscomm);
        usermap_0 = (int *)malloc(sizeof(int)*size_blacscomm);
        usermap_1 = (int *)malloc(sizeof(int)*size_blacscomm);
        for (i = 0; i < size_blacscomm; i++) {
            usermap[i] = usermap_0[i] = usermap_1[i] = i + rank - rank_kptcomm + rank_dmcomm * pSPARC->npband;
        }

        // in order to use a subgroup of blacscomm, use the following
        // to get a good number of subgroup processes
        bandsizes[0] = ((pSPARC->Nd-1)/pSPARC->npNd+1) * pSPARC->Nspinor_eig;
        bandsizes[1] = pSPARC->Nstates;
        //SPARC_Dims_create(pSPARC->npband, 2, bandsizes, 1, dims, &ierr);
        ScaLAPACK_Dims_2D_BLCYC(size_blacscomm, bandsizes, dims);
#ifdef DEBUG
        if (!rank) printf("rank = %d, size_blacscomm = %d, ScaLAPACK topology Dims = (%d, %d)\n", rank, size_blacscomm, dims[0], dims[1]);
#endif
        // TODO: make it able to use a subgroup of the blacscomm! For now just enforce it.
        if (dims[0] * dims[1] != size_blacscomm) {
            dims[0] = size_blacscomm;
            dims[1] = 1;
        }
    } else {
        usermap = (int *)malloc(sizeof(int)*1);
        usermap_0 = (int *)malloc(sizeof(int)*1);
        usermap_1 = (int *)malloc(sizeof(int)*1);
        usermap[0] = usermap_0[0] = usermap_1[0] = rank;
        dims[0] = dims[1] = 1;
    }

#ifdef DEBUG
    if (!rank) {
        printf("nproc = %d, size_blacscomm = %d = dims[0] * dims[1] = (%d, %d)\n", nproc, size_blacscomm, dims[0], dims[1]);
    }
#endif
    // TODO: CHANGE USERMAP TO COLUMN MAJOR!
    int myrank_mpi, nprocs_mpi;
    Cblacs_pinfo( &myrank_mpi, &nprocs_mpi );

    // the following commands will create a context with handle ictxt_blacs
    Cblacs_get( -1, 0, &pSPARC->ictxt_blacs );
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        Cblacs_gridmap( &pSPARC->ictxt_blacs, usermap_0, 1, 1, pSPARC->npband); // row topology
    } else {
        Cblacs_gridmap( &pSPARC->ictxt_blacs, usermap_0, 1, 1, dims[0] * dims[1]); // row topology
    }
    free(usermap_0);

    // create ictxt_blacs topology
    Cblacs_get( -1, 0, &pSPARC->ictxt_blacs_topo );
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        // create usermap_1 = reshape(usermap,[dims[0], dims[1]])
        for (j = 0; j < dims[1]; j++) {
            for (i = 0; i < dims[0]; i++) {
                usermap_1[j*dims[0]+i] = usermap[i*dims[1]+j];
            }
        }
    }
    Cblacs_gridmap( &pSPARC->ictxt_blacs_topo, usermap_1, dims[0], dims[0], dims[1] ); // Cart topology
    free(usermap_1);
    free(usermap);

    // get coord of each process in original context
    Cblacs_gridinfo( pSPARC->ictxt_blacs, &nprow, &npcol, &myrow, &mycol );
    int ZERO = 0, mb, nb, llda;
    mb = max(1, DMndspe);
    nb = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // equal to ceil(Nstates/npband), for int only
    // set up descriptor for storage of orbitals in ictxt_blacs (original)
    llda = max(1, DMndsp);
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        descinit_(&pSPARC->desc_orbitals[0], &DMndspe, &pSPARC->Nstates,
                  &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs, &llda, &info);
    } else {
        for (i = 0; i < 9; i++)
            pSPARC->desc_orbitals[i] = 0;
    }
#ifdef DEBUG
    int temp_r, temp_c;
    temp_r = numroc_( &DMndspe, &mb, &myrow, &ZERO, &nprow);
    temp_c = numroc_( &pSPARC->Nstates, &nb, &mycol, &ZERO, &npcol);
    if (!rank) printf("rank = %2d, my blacs rank = %d, BLCYC size (%d, %d), actual size (%d, %d)\n", rank, pSPARC->bandcomm_index, temp_r, temp_c, DMndsp, pSPARC->Nband_bandcomm);
#endif
    // get coord of each process in block cyclic topology context
    Cblacs_gridinfo( pSPARC->ictxt_blacs_topo, &nprow, &npcol, &myrow, &mycol );
    pSPARC->nprow_ictxt_blacs_topo = nprow;
    pSPARC->npcol_ictxt_blacs_topo = npcol;

    // set up descriptor for block-cyclic format storage of orbitals in ictxt_blacs
    // TODO: make block-cyclic parameters mb and nb input variables!
    mb = max(1, DMndspe / dims[0]); // this is only block, no cyclic! Tune this to improve efficiency!
    nb = max(1, pSPARC->Nstates / dims[1]); // this is only block, no cyclic!

    // find number of rows/cols of the local distributed orbitals
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        pSPARC->nr_orb_BLCYC = numroc_( &DMndspe, &mb, &myrow, &ZERO, &nprow);
        pSPARC->nc_orb_BLCYC = numroc_( &pSPARC->Nstates, &nb, &mycol, &ZERO, &npcol);
    } else {
        pSPARC->nr_orb_BLCYC = 1;
        pSPARC->nc_orb_BLCYC = 1;
    }
    llda = max(1, pSPARC->nr_orb_BLCYC);
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        descinit_(&pSPARC->desc_orb_BLCYC[0], &DMndspe, &pSPARC->Nstates,
                  &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &llda, &info);
    } else {
        for (i = 0; i < 9; i++)
            pSPARC->desc_orb_BLCYC[i] = 0;
    }

    // set up distribution of projected Hamiltonian and the corresponding overlap matrix
    // TODO: Find optimal distribution of the projected Hamiltonian and mass matrix!
    //       For now Hp and Mp are distributed as follows: we distribute them in the same
    //       context topology as the bands.
    //       Note that mb = nb!

    // the maximum Nstates up to which we will use LAPACK to solve
    // the subspace eigenproblem in serial
    // int MAX_NS = 2000;
    int MAX_NS = pSPARC->eig_serial_maxns;
    pSPARC->useLAPACK = (pSPARC->Nstates <= MAX_NS) ? 1 : 0;

    int mbQ, nbQ, lldaQ;
    
    // block size for storing Hp and Mp
    if (pSPARC->useLAPACK == 1) {
        // in this case we will call LAPACK instead to solve the subspace eigenproblem
        mb = nb = pSPARC->Nstates;
        mbQ = nbQ = 64; // block size for storing subspace eigenvectors
    } else {
        // in this case we will use ScaLAPACK to solve the subspace eigenproblem
        mb = nb = pSPARC->eig_paral_blksz;
        mbQ = nbQ = pSPARC->eig_paral_blksz; // block size for storing subspace eigenvectors
    }
#ifdef DEBUG
    if (!rank) printf("rank = %d, mb = nb = %d, mbQ = nbQ = %d\n", rank, mb, mbQ);
#endif
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        pSPARC->nr_Hp_BLCYC = pSPARC->nr_Mp_BLCYC = numroc_( &pSPARC->Nstates, &mb, &myrow, &ZERO, &nprow);
        pSPARC->nr_Hp_BLCYC = pSPARC->nr_Mp_BLCYC = max(1, pSPARC->nr_Mp_BLCYC);
        pSPARC->nc_Hp_BLCYC = pSPARC->nc_Mp_BLCYC = numroc_( &pSPARC->Nstates, &nb, &mycol, &ZERO, &npcol);
        pSPARC->nc_Hp_BLCYC = pSPARC->nc_Mp_BLCYC = max(1, pSPARC->nc_Mp_BLCYC);
        pSPARC->nr_Q_BLCYC = numroc_( &pSPARC->Nstates, &mbQ, &myrow, &ZERO, &nprow);
        pSPARC->nc_Q_BLCYC = numroc_( &pSPARC->Nstates, &nbQ, &mycol, &ZERO, &npcol);
    } else {
        pSPARC->nr_Hp_BLCYC = pSPARC->nc_Hp_BLCYC = 1;
        pSPARC->nr_Mp_BLCYC = pSPARC->nc_Mp_BLCYC = 1;
        pSPARC->nr_Q_BLCYC  = pSPARC->nc_Q_BLCYC  = 1;
    }

    llda = max(1, pSPARC->nr_Hp_BLCYC);
    lldaQ= max(1, pSPARC->nr_Q_BLCYC);
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        descinit_(&pSPARC->desc_Hp_BLCYC[0], &pSPARC->Nstates, &pSPARC->Nstates,
                  &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &llda, &info);
        for (i = 0; i < 9; i++) {
            //pSPARC->desc_Q_BLCYC[i] = pSPARC->desc_Mp_BLCYC[i] = pSPARC->desc_Hp_BLCYC[i];
            pSPARC->desc_Mp_BLCYC[i] = pSPARC->desc_Hp_BLCYC[i];
        }
        descinit_(&pSPARC->desc_Q_BLCYC[0], &pSPARC->Nstates, &pSPARC->Nstates,
                  &mbQ, &nbQ, &ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &lldaQ, &info);
    } else {
        for (i = 0; i < 9; i++) {
            pSPARC->desc_Q_BLCYC[i] = pSPARC->desc_Mp_BLCYC[i] = pSPARC->desc_Hp_BLCYC[i] = 0;
        }
    }

#ifdef DEBUG
    if (!rank) printf("rank = %d, nr_Hp = %d, nc_Hp = %d\n", rank, pSPARC->nr_Hp_BLCYC, pSPARC->nc_Hp_BLCYC);
#endif

    // allocate memory for block cyclic distribution of projected Hamiltonian and mass matrix
    if (pSPARC->isGammaPoint){
        pSPARC->Hp = (double *)malloc(pSPARC->nr_Hp_BLCYC * pSPARC->nc_Hp_BLCYC * sizeof(double));
        pSPARC->Mp = (double *)malloc(pSPARC->nr_Mp_BLCYC * pSPARC->nc_Mp_BLCYC * sizeof(double));
        pSPARC->Q  = (double *)malloc(pSPARC->nr_Q_BLCYC * pSPARC->nc_Q_BLCYC * sizeof(double));
    } else{
        pSPARC->Hp_kpt = (double _Complex *) malloc(pSPARC->nr_Hp_BLCYC * pSPARC->nc_Hp_BLCYC * sizeof(double _Complex));
        pSPARC->Mp_kpt = (double _Complex *) malloc(pSPARC->nr_Mp_BLCYC * pSPARC->nc_Mp_BLCYC * sizeof(double _Complex));
        pSPARC->Q_kpt  = (double _Complex *) malloc(pSPARC->nr_Q_BLCYC * pSPARC->nc_Q_BLCYC * sizeof(double _Complex));
    }
#else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    pSPARC->useLAPACK = 1;
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    //------------------------------------------------//
    //               set up poisson domain            //
    //------------------------------------------------//
    // some variables are reused here
    npNd = pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi;
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    minsize = pSPARC->order/2;

    if (npNd == 0)  {
        // when user does not provide domain decomposition parameters
        npNd = nproc; // try to use all processors
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        if (ierr == 1 && rank == 0) {
            printf("WARNING: error occured when calculating best domain distribution."
                   "Please check if your gridsizes are less than MINSIZE\n");
        }
        pSPARC->npNdx_phi = dims[0];
        pSPARC->npNdy_phi = dims[1];
        pSPARC->npNdz_phi = dims[2];
    } else if (npNd < 0 || npNd > nproc || pSPARC->Nx / pSPARC->npNdx_phi < minsize ||
               pSPARC->Ny / pSPARC->npNdy_phi < minsize || pSPARC->Nz / pSPARC->npNdz_phi < minsize) {
        // when domain decomposition parameters are not reasonable
        npNd = nproc;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        if (ierr == 1 && rank == 0) {
            printf("WARNING: error occured when calculating best domain distribution."
                   "Please check if your gridsizes are less than MINSIZE\n");
        }
        pSPARC->npNdx_phi = dims[0];
        pSPARC->npNdy_phi = dims[1];
        pSPARC->npNdz_phi = dims[2];
    } else {
        dims[0] = pSPARC->npNdx_phi;
        dims[1] = pSPARC->npNdy_phi;
        dims[2] = pSPARC->npNdz_phi;
    }

    // recalculate number of processors in dmcomm_phi
    npNd = pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi;

    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    // Processes in bandcomm with rank >= dims[0]*dims[1]*…*dims[d-1] will return MPI_COMM_NULL
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &pSPARC->dmcomm_phi); // 1 is to reorder rank

#ifdef DEBUG
    if (rank == 0) {
        printf("========================================================================\n"
                   "Poisson domain decomposition:"
                   "np total = %d, {Nx, Ny, Nz} = {%d, %d, %d}\n"
                   "nproc used = %d = {%d, %d, %d}, nodes/proc = {%.2f, %.2f, %.2f}\n\n",
                   nproc,pSPARC->Nx,pSPARC->Ny,pSPARC->Nz,dims[0]*dims[1]*dims[2],dims[0],dims[1],dims[2],pSPARC->Nx/(double)dims[0],pSPARC->Ny/(double)dims[1],pSPARC->Nz/(double)dims[2]);
    }
#endif

    // find the vertices of the domain in each processor
    // pSPARC->DMVertices[6] = [xs,xe,ys,ye,zs,ze]
    // int Nx_dist, Ny_dist, Nz_dist;
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        MPI_Comm_rank(pSPARC->dmcomm_phi, &rank_dmcomm);
        MPI_Cart_coords(pSPARC->dmcomm_phi, rank_dmcomm, 3, coord_dmcomm);

        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;

        // find size of distributed domain
        pSPARC->Nx_d = block_decompose(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->Ny_d = block_decompose(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->Nz_d = block_decompose(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->Nd_d = pSPARC->Nx_d * pSPARC->Ny_d * pSPARC->Nz_d;

        // find corners of the distributed domain
        pSPARC->DMVertices[0] = block_decompose_nstart(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->DMVertices[1] = pSPARC->DMVertices[0] + pSPARC->Nx_d - 1;
        pSPARC->DMVertices[2] = block_decompose_nstart(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->DMVertices[3] = pSPARC->DMVertices[2] + pSPARC->Ny_d - 1;
        pSPARC->DMVertices[4] = block_decompose_nstart(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->DMVertices[5] = pSPARC->DMVertices[4] + pSPARC->Nz_d - 1;
    } else {
        rank_dmcomm = -1;
        coord_dmcomm[0] = -1; coord_dmcomm[1] = -1; coord_dmcomm[2] = -1;
        pSPARC->Nx_d = 0;
        pSPARC->Ny_d = 0;
        pSPARC->Nz_d = 0;
        pSPARC->Nd_d = 0;
        pSPARC->DMVertices[0] = 0;
        pSPARC->DMVertices[1] = 0;
        pSPARC->DMVertices[2] = 0;
        pSPARC->DMVertices[3] = 0;
        pSPARC->DMVertices[4] = 0;
        pSPARC->DMVertices[5] = 0;
    }

    // Set up a 26 neighbor communicator for nonorthogonal systems
    // TODO: Modify the communicator based on number of non zero enteries in off diagonal of pSPARC->lapcT
    // TODO: Take into account more than 26 Neighbors
    if(pSPARC->cell_typ != 0) {
        if(pSPARC->dmcomm_phi != MPI_COMM_NULL) {
            int rank_chk, dir, tmp;
            int ncoords[3];
            int nnproc = 1; // No. of neighboring processors in each direction
            int nneighb = pow(2*nnproc+1,3) - 1; // 6 faces + 12 edges + 8 corners
            int *neighb;
            neighb = (int *) malloc(nneighb * sizeof (int));
            int count = 0, i, j, k;
            for(k = -nnproc; k <= nnproc; k++){
                for(j = -nnproc; j <= nnproc; j++){
                    for(i = -nnproc; i <= nnproc; i++){
                        tmp = 0;
                        if(i == 0 && j == 0 && k == 0){
                            continue;
                        } else{
                            ncoords[0] = coord_dmcomm[0] + i;
                            ncoords[1] = coord_dmcomm[1] + j;
                            ncoords[2] = coord_dmcomm[2] + k;
                            for(dir = 0; dir < 3; dir++){
                                if(periods[dir]){
                                    if(ncoords[dir] < 0)
                                        ncoords[dir] += dims[dir];
                                    else if(ncoords[dir] >= dims[dir])
                                        ncoords[dir] -= dims[dir];
                                    tmp = 1;
                                } else{
                                    if(ncoords[dir] < 0 || ncoords[dir] >= dims[dir]){
                                        rank_chk = MPI_PROC_NULL;
                                        tmp = 0;
                                        break;
                                    }
                                    else
                                        tmp = 1;
                                }
                            }
                            //TODO: For dirchlet give rank = MPI_PROC_NULL for out of bounds coordinates
                            if(tmp == 1)
                                MPI_Cart_rank(pSPARC->dmcomm_phi,ncoords,&rank_chk); // proc rank corresponding to ncoords_mapped

                            neighb[count] = rank_chk;
                            count++;
                        }
                    }
                }
            }
            MPI_Dist_graph_create_adjacent(pSPARC->dmcomm_phi,nneighb,neighb,(int *)MPI_UNWEIGHTED,nneighb,neighb,(int *)MPI_UNWEIGHTED,MPI_INFO_NULL,0,&pSPARC->comm_dist_graph_phi); // creates a distributed graph topology (adjacent, cartesian cubical)
            //pSPARC->dmcomm_phi = pSPARC->comm_dist_graph_phi;
            free(neighb);
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up dmcomm_phi took %.3f ms\n",(t2-t1)*1000);
#endif

    // allocate memory for storing eigenvalues
    pSPARC->lambda = (double *)calloc(pSPARC->Nstates * pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof(double));
    assert(pSPARC->lambda != NULL);

    pSPARC->lambda_sorted = pSPARC->lambda;

    // allocate memory for storing eigenvalues
    pSPARC->occ = (double *)calloc(pSPARC->Nstates * pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof(double));
    assert(pSPARC->occ != NULL);

    pSPARC->occ_sorted = pSPARC->occ;

    pSPARC->eigmin = (double *) calloc(pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof (double));
    pSPARC->eigmax = (double *) calloc(pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof (double));

    /* allocate memory for storing atomic forces*/
    pSPARC->forces = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
    assert(pSPARC->forces != NULL);

    if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) {
        pSPARC->Veff_loc_dmcomm = (double *)malloc( pSPARC->Nd_d_dmcomm * pSPARC->Nspden * sizeof(double) );
        assert(pSPARC->Veff_loc_dmcomm != NULL);
    }

    pSPARC->Veff_loc_kptcomm_topo = (double *)malloc( pSPARC->Nd_d_kptcomm * ((pSPARC->spin_typ == 2) ? 4 : 1) *  sizeof(double) );
    assert(pSPARC->Veff_loc_kptcomm_topo != NULL);

    // allocate memory for initial guess vector for Lanczos
    if (pSPARC->isGammaPoint && pSPARC->kptcomm_topo != MPI_COMM_NULL) {
        pSPARC->Lanczos_x0 = (double *)malloc(pSPARC->Nd_d_kptcomm * pSPARC->Nspinor_eig * sizeof(double));
        assert(pSPARC->Lanczos_x0 != NULL);
    }

    if (pSPARC->isGammaPoint != 1 && pSPARC->kptcomm_topo != MPI_COMM_NULL) {
        pSPARC->Lanczos_x0_complex = (double _Complex *)malloc(pSPARC->Nd_d_kptcomm * pSPARC->Nspinor_eig * sizeof(double _Complex));
        assert(pSPARC->Lanczos_x0_complex != NULL);
    }

    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        /* allocate memory for electrostatics calculation */
        int DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
        int DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
        int DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
        int DMnd = DMnx * DMny * DMnz;
        // allocate memory for electron density (sum of atom potential) and charge density
        pSPARC->electronDens_at = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->electronDens_core = (double *)calloc( DMnd, sizeof(double) );
        pSPARC->psdChrgDens = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->psdChrgDens_ref = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->Vc = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->electronDens_core != NULL);
        assert(pSPARC->electronDens_at != NULL && pSPARC->psdChrgDens != NULL &&
               pSPARC->psdChrgDens_ref != NULL && pSPARC->Vc != NULL);
        // allocate memory for electron density
        pSPARC->electronDens = (double *)malloc( DMnd * pSPARC->Nspdentd * sizeof(double) );
        assert(pSPARC->electronDens != NULL);
        if (pSPARC->usefock > 0)
            pSPARC->electronDens_pbe = (double *)malloc( DMnd * pSPARC->Nspdentd * sizeof(double) );
        // allocate memory for magnetization
        if (pSPARC->spin_typ > 0) {
            pSPARC->mag = (double *)malloc( DMnd * pSPARC->Nmag * sizeof(double) );
            assert(pSPARC->mag != NULL);
            int ncol = (pSPARC->spin_typ > 1) + pSPARC->spin_typ; // 0 0 1 3
            pSPARC->mag_at = (double *)malloc( DMnd * ncol * sizeof(double) );
            assert(pSPARC->mag_at != NULL);
            pSPARC->AtomMag = (double *)malloc( (pSPARC->spin_typ == 2 ? 3 : 1) * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->AtomMag != NULL);
        }
        // allocate memory for charge extrapolation arrays
        if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1 || pSPARC->RelaxFlag == 3){
            pSPARC->delectronDens = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens != NULL);
            pSPARC->delectronDens_0dt = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens_0dt != NULL);
            pSPARC->delectronDens_1dt = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens_1dt != NULL);
            pSPARC->delectronDens_2dt = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens_2dt != NULL);
            pSPARC->atom_pos_nm = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_nm != NULL);
            pSPARC->atom_pos_0dt = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_0dt != NULL);
            pSPARC->atom_pos_1dt = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_1dt != NULL);
            pSPARC->atom_pos_2dt = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_2dt != NULL);
        }
        // allocate memory for electrostatic potential
        pSPARC->elecstPotential = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->elecstPotential != NULL);

        // allocate memory for XC potential
        pSPARC->XCPotential = (double *)malloc( DMnd * pSPARC->Nspdend * sizeof(double) );
        assert(pSPARC->XCPotential != NULL);

        // allocate memory for exchange-correlation energy density
        pSPARC->e_xc = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->e_xc != NULL);

        // if GGA then allocate for xc energy per particle for each grid point and der. wrt. grho
        if(pSPARC->isgradient) {
            pSPARC->Dxcdgrho = (double *)malloc( DMnd * pSPARC->Nspdentd * sizeof(double) );
            assert(pSPARC->Dxcdgrho != NULL);
        }

        pSPARC->Veff_loc_dmcomm_phi = (double *)malloc(DMnd * pSPARC->Nspden * sizeof(double));        
        pSPARC->mixing_hist_xk      = (double *)malloc(DMnd * pSPARC->Nspden * sizeof(double));
        pSPARC->mixing_hist_fk      = (double *)calloc(DMnd * pSPARC->Nspden , sizeof(double));
        pSPARC->mixing_hist_fkm1    = (double *)calloc(DMnd * pSPARC->Nspden , sizeof(double));
        pSPARC->mixing_hist_xkm1    = (double *)malloc(DMnd * pSPARC->Nspden * sizeof(double));
        pSPARC->mixing_hist_Xk      = (double *)malloc(DMnd * pSPARC->Nspden * pSPARC->MixingHistory * sizeof(double));
        pSPARC->mixing_hist_Fk      = (double *)malloc(DMnd * pSPARC->Nspden * pSPARC->MixingHistory * sizeof(double));
        pSPARC->mixing_hist_Pfk     = (double *)calloc(DMnd * pSPARC->Nspden, sizeof(double));
        assert(pSPARC->Veff_loc_dmcomm_phi != NULL && pSPARC->mixing_hist_xk   != NULL &&
               pSPARC->mixing_hist_fk      != NULL && pSPARC->mixing_hist_fkm1 != NULL &&
               pSPARC->mixing_hist_xkm1    != NULL && pSPARC->mixing_hist_Xk   != NULL &&
               pSPARC->mixing_hist_Fk      != NULL && pSPARC->mixing_hist_Pfk  != NULL);

        if (pSPARC->MixingVariable == 1) {
            pSPARC->Veff_loc_dmcomm_phi_in = (double *)malloc(DMnd * pSPARC->Nspdend * sizeof(double));
            assert(pSPARC->Veff_loc_dmcomm_phi_in != NULL);
        } 

        if (pSPARC->MixingVariable == 0 && pSPARC->spin_typ) {
            pSPARC->electronDens_in = (double *)malloc(DMnd * pSPARC->Nspdentd * sizeof(double));
            assert(pSPARC->electronDens_in != NULL);
        }

        // initialize electrostatic potential as random guess vector
        if (pSPARC->FixRandSeed == 1) {
            SeededRandVec(pSPARC->elecstPotential, pSPARC->DMVertices, gridsizes, -1.0, 1.0, 0);
        } else {
            srand(rank+1);
            double rand_min = -1.0, rand_max = 1.0;
            for (i = 0; i < DMnd; i++) {
                pSPARC->elecstPotential[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX; // or 1.0
            }
        }

        if (pSPARC->spin_typ == 2) {
            // allocate memory for XC potential
            pSPARC->XCPotential_nc = (double *)malloc( DMnd * pSPARC->Nspden * sizeof(double) );
            assert(pSPARC->XCPotential_nc != NULL);
            if (pSPARC->MixingVariable == 1) {
                pSPARC->Veff_dia_loc_dmcomm_phi = (double *)malloc(DMnd * pSPARC->Nspdend * sizeof(double));
                assert(pSPARC->Veff_dia_loc_dmcomm_phi != NULL);
            } 
        }

        if (pSPARC->BC == 1) {
            pSPARC->MpExp = (MPEXP_OBJ *) malloc(sizeof(MPEXP_OBJ));
            init_multipole_expansion(pSPARC, pSPARC->MpExp, 
                pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, pSPARC->Nx_d, pSPARC->Ny_d, pSPARC->Nz_d, pSPARC->DMVertices, pSPARC->dmcomm_phi);
        }
    }

    // Set up D2D target objects between phi comm and psi comm
    // Note: Set_D2D_target require gridsizes, rdims, sdims to be global in union{send_comm, recv_comm},
    //       i.e., all processes know about these values and they are the same in all processes
    int sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    rdims[0] = pSPARC->npNdx;
    rdims[1] = pSPARC->npNdy;
    rdims[2] = pSPARC->npNdz;
    sdims[0] = pSPARC->npNdx_phi;
    sdims[1] = pSPARC->npNdy_phi;
    sdims[2] = pSPARC->npNdz_phi;

    Set_D2D_Target(&pSPARC->d2d_dmcomm_phi, &pSPARC->d2d_dmcomm, gridsizes, pSPARC->DMVertices, pSPARC->DMVertices_dmcomm, pSPARC->dmcomm_phi,
                   sdims, (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, rdims, MPI_COMM_WORLD);

    // Set up D2D target objects between psi comm and kptcomm_topo comm
    // check if kptcomm_topo is the same as dmcomm_phi
    // If found rank order in the cartesian topology is different for dmcomm_phi and 
    // kptcomm_topo, consider to add MPI_Bcast for is_phi_eq_kpt_topo
    if ((pSPARC->npNdx_phi == pSPARC->npNdx_kptcomm) && 
        (pSPARC->npNdy_phi == pSPARC->npNdy_kptcomm) && 
        (pSPARC->npNdz_phi == pSPARC->npNdz_kptcomm))
        pSPARC->is_phi_eq_kpt_topo = 1;
    else
        pSPARC->is_phi_eq_kpt_topo = 0;

    if (((pSPARC->chefsibound_flag == 0 || pSPARC->chefsibound_flag == 1) &&
            pSPARC->spincomm_index >=0 && pSPARC->kptcomm_index >= 0
            && (pSPARC->spin_typ != 0 || !pSPARC->is_phi_eq_kpt_topo || !pSPARC->isGammaPoint))
            || (pSPARC->usefock > 0) )
    {
        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;
        sdims[0] = pSPARC->npNdx;
        sdims[1] = pSPARC->npNdy;
        sdims[2] = pSPARC->npNdz;
        rdims[0] = pSPARC->npNdx_kptcomm;
        rdims[1] = pSPARC->npNdy_kptcomm;
        rdims[2] = pSPARC->npNdz_kptcomm;

        Set_D2D_Target(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, pSPARC->DMVertices_dmcomm, pSPARC->DMVertices_kptcomm,
                       pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL, sdims,
                       pSPARC->kptcomm_topo, rdims, pSPARC->kptcomm);
    }

    // parallelization summary
    #ifdef DEBUG
    if (rank == 0) {
        printf("\n");
        printf("-----------------------------------------------\n");
        printf("Parallelization summary\n");
        printf("Total number of processors: %d\n", nproc);
        printf("-----------------------------------------------\n");
        printf("== Psi domain ==\n");
        printf("Total number of processors used for Psi domain: %d\n", pSPARC->npspin*pSPARC->npkpt*pSPARC->npband*pSPARC->npNd);
        printf("npspin  : %d\n", pSPARC->npspin);
        printf("# of spin per spincomm           : %.0f\n", ceil(pSPARC->Nspin / (double)pSPARC->npspin));
        printf("npkpt   : %d\n", pSPARC->npkpt);
        printf("# of k-points per kptcomm        : %.0f\n", ceil(pSPARC->Nkpts_sym / (double)pSPARC->npkpt));
        printf("npband  : %d\n", pSPARC->npband);
        printf("# of bands per bandcomm          : %.0f\n", ceil(pSPARC->Nstates / (double)pSPARC->npband));
        printf("npdomain: %d\n", pSPARC->npNd);
        printf("Embeded Cartesian topology dims: (%d,%d,%d)\n", pSPARC->npNdx, pSPARC->npNdy, pSPARC->npNdz);
        printf("# of FD-grid points per processor: %d = (%d,%d,%d)\n", pSPARC->Nd_d_dmcomm,pSPARC->Nx_d_dmcomm,pSPARC->Ny_d_dmcomm,pSPARC->Nz_d_dmcomm);
        printf("-----------------------------------------------\n");
        printf("== Phi domain ==\n");
        printf("Total number of processors used for Phi domain: %d\n", pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi);
        printf("Embeded Cartesian topology dims: (%d,%d,%d)\n", pSPARC->npNdx_phi,pSPARC->npNdy_phi, pSPARC->npNdz_phi);
        printf("# of FD-grid points per processor: %d = (%d,%d,%d)\n", pSPARC->Nd_d,pSPARC->Nx_d,pSPARC->Ny_d,pSPARC->Nz_d);
        printf("-----------------------------------------------\n");
    }
    #endif
}



/**
 * @brief   Creates a balanced division of processors/subset of processors in a
 *          Cartesian grid according to application size.
 */
void SPARC_Dims_create(const int nproc, const int ndims, const int *gridsizes, int minsize, int *dims, int *ierr) {
#define NPSTRIDE 5
#define NPSTRIDE3D 2
    *ierr = 0;
    minsize = max(minsize,0); // minsize should be greater than or equal to 0
    if (ndims == 1) {
        *dims = min(nproc, *gridsizes / minsize);
        *dims = max(*dims,1);
    } else if (ndims == 2) {
        double r, rnormi[(NPSTRIDE*2+1)*2],  min_rnorm, tmp1, tmp2;
        int i, j, count, best_ind, max_npi, tmpint1, tmpint2, dimx[(NPSTRIDE*2+1)*2], dimy[(NPSTRIDE*2+1)*2], npi[(NPSTRIDE*2+1)*2];
        r = sqrt((double)nproc) / sqrt( (double)gridsizes[0] * (double)gridsizes[1] );
        r = min(r, 1/(double)minsize);
        double r2 = r * r;

        // initial estimate
        dims[0] = round(gridsizes[0] * r);
        dims[1] = round(gridsizes[1] * r);
        count = 0;
        for (i = 0; i < (NPSTRIDE*2+1); i++) {
            tmpint1 = dims[0] + i - NPSTRIDE;
            dimx[count] = (tmpint1 > 0) ? (tmpint1 <= nproc ? tmpint1 : (tmpint1-(NPSTRIDE*2+1))) : (tmpint1+(NPSTRIDE*2+1));
            // dimx[count] > 0 ? (dimx[count] <= nproc ? : (dimx[count] = 1)) : (dimx[count] = 1);
            if (dimx[count] > 0 && dimx[count] <= nproc) {
                SPARC_Dims_create(max(1,nproc/dimx[count]), 1, &gridsizes[1], minsize, &dimy[count], ierr);
                npi[count] = dimx[count] * dimy[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; npi[count] = 0;
                rnormi[count] = 0;
            }
            count++;
        }
        for (j = 0; j < (NPSTRIDE*2+1); j++) {
            tmpint2 = dims[1] + j - NPSTRIDE;
            dimy[count] = (tmpint2 > 0) ? (tmpint2 <= nproc ? tmpint2 : (tmpint2-(NPSTRIDE*2+1))) : (tmpint2+(NPSTRIDE*2+1));
            //dimy[count] > 0 ? (dimy[count] <= nproc ? : (dimy[count] = 1)) : (dimy[count] = 1);
            if (dimy[count] > 0 && dimy[count] <= nproc) {
                SPARC_Dims_create(max(1,nproc/dimy[count]), 1, &gridsizes[0], minsize, &dimx[count], ierr);
                npi[count] = dimx[count] * dimy[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; npi[count] = 0;
                rnormi[count] = 0;
            }
            count++;
        }

        // check which one uses largest number of processes provided
        max_npi = 0;
        count = 0; best_ind = -1; min_rnorm = 1e4;
        for (i = 0; i < (NPSTRIDE*2+1)*2; i++) {
            if (npi[count] < max_npi || npi[count] > nproc || npi[count] <= 0) {
                count++;
                continue;
            }
            if (npi[count] > max_npi && gridsizes[0]/dimx[count] >= minsize && gridsizes[1]/dimy[count] >= minsize) {
                best_ind = count;
                max_npi = npi[count];
                min_rnorm = rnormi[count];
            } else if (npi[count] == max_npi && rnormi[count] < min_rnorm) {
                best_ind = count;
                min_rnorm = rnormi[count];
            }
            count++;
        }

        // TODO: after first scan, perhaps we can allow np to be up to 3% smaller, and choose the one with smaller rnormi,
        // the idea is that by reducing total number of process we lose speed by 3% or less, but we might gain speed up in
        // communication by more than 3%
        if (best_ind != -1) {
            dims[0] = dimx[best_ind]; dims[1] = dimy[best_ind];
        } else {
            dims[0] = nproc; dims[1] = 1;;
            *ierr = 1; // cannot find any admissable distribution
        }
    } else if (ndims == 3) {
        double r, rnormi[(NPSTRIDE3D*2+1)*3], min_rnorm, tmp1, tmp2, tmp3;
        int i, j, k, count, best_ind, max_npi, tmpint1, tmpint2, tmpint3, dims_temp[2], gridsizes_temp[2];
        int dimx[(NPSTRIDE3D*2+1)*3], dimy[(NPSTRIDE3D*2+1)*3], dimz[(NPSTRIDE3D*2+1)*3], npi[(NPSTRIDE3D*2+1)*3];
        r = cbrt((double)nproc) / cbrt( (double)gridsizes[0] * (double)gridsizes[1] * (double)gridsizes[2] );
        r = min(r, 1/(double)minsize);
        double r2 = r * r;

        // initial estimate
        dims[0] = round(gridsizes[0] * r);
        dims[1] = round(gridsizes[1] * r);
        dims[2] = round(gridsizes[2] * r);
        count = 0;
        for (i = 0; i < (NPSTRIDE3D*2+1); i++) {
            tmpint1 = dims[0] + i - NPSTRIDE3D;
            dimx[count] = (tmpint1 > 0) ? (tmpint1 <= nproc ? tmpint1 : (tmpint1-(NPSTRIDE3D*2+1))) : (tmpint1+(NPSTRIDE3D*2+1));
            // dimx[count] > 0 ? (dimx[count] <= nproc ? : (dimx[count] = 1)) : (dimx[count] = 1);
            if (dimx[count] > 0 && dimx[count] <= nproc) {
                SPARC_Dims_create(max(1,nproc/dimx[count]), 2, &gridsizes[1], minsize, dims_temp, ierr);
                dimy[count] = dims_temp[0];
                dimz[count] = dims_temp[1];
                npi[count] = dimx[count] * dimy[count] * dimz[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                tmp3 = ((double) gridsizes[2]) / dimz[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2) + (tmp1 - tmp3) * (tmp1 - tmp3) + (tmp2 - tmp3) * (tmp2 - tmp3);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; dimz[count] = 0;
                npi[count] = 0;
                rnormi[count] = 0;
            }

            count++;
        }

        gridsizes_temp[0] = gridsizes[0];
        gridsizes_temp[1] = gridsizes[2];
        for (j = 0; j < (NPSTRIDE3D*2+1); j++) {
            tmpint2 = dims[1] + j - NPSTRIDE3D;
            dimy[count] = (tmpint2 > 0) ? (tmpint2 <= nproc ? tmpint2 : (tmpint2-(NPSTRIDE3D*2+1))) : (tmpint2+(NPSTRIDE3D*2+1));
           // dimy[count] > 0 ? (dimy[count] <= nproc ? : (dimy[count] = 1)) : (dimy[count] = 1);
            if (dimy[count] > 0 && dimy[count] <= nproc) {
                SPARC_Dims_create(max(1,nproc/dimy[count]), 2, gridsizes_temp, minsize, dims_temp, ierr);
                dimx[count] = dims_temp[0];
                dimz[count] = dims_temp[1];
                npi[count] = dimx[count] * dimy[count] * dimz[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                tmp3 = ((double) gridsizes[2]) / dimz[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2) + (tmp1 - tmp3) * (tmp1 - tmp3) + (tmp2 - tmp3) * (tmp2 - tmp3);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; dimz[count] = 0;
                npi[count] = 0;
                rnormi[count] = 0;
            }
            count++;
        }

        for (k = 0; k < (NPSTRIDE3D*2+1); k++) {
            tmpint3 = dims[2] + k - NPSTRIDE3D;
            dimz[count] = (tmpint3 > 0) ? (tmpint3 <= nproc ? tmpint3 : (tmpint3-(NPSTRIDE3D*2+1))) : (tmpint3+(NPSTRIDE3D*2+1));
            // for these cases, one can actually skip the calculation
            // dimz[count] > 0 ? (dimz[count] <= nproc ? : (dimz[count] = 1)) : (dimz[count] = 1);
            if (dimz[count] > 0 && dimz[count] <= nproc) {
                SPARC_Dims_create(max(1,nproc/dimz[count]), 2, &gridsizes[0], minsize, dims_temp, ierr);
                dimx[count] = dims_temp[0];
                dimy[count] = dims_temp[1];
                npi[count] = dimx[count] * dimy[count] * dimz[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                tmp3 = ((double) gridsizes[2]) / dimz[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2) + (tmp1 - tmp3) * (tmp1 - tmp3) + (tmp2 - tmp3) * (tmp2 - tmp3);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; dimz[count] = 0;
                npi[count] = 0;
                rnormi[count] = 0;
            }
            count++;
        }

        // check which one uses largest number of processes provided
        max_npi = 0;
        count = 0; best_ind = -1; min_rnorm = 1e4;
        for (i = 0; i < (NPSTRIDE3D*2+1)*3; i++) {
            if (npi[count] < max_npi || npi[count] > nproc || npi[count] <= 0) {
                count++;
                continue;
            }
            if (npi[count] > max_npi && gridsizes[0]/dimx[count] >= minsize && gridsizes[1]/dimy[count] >= minsize  && gridsizes[2]/dimz[count] >= minsize) {
                best_ind = count;
                max_npi = npi[count];
                min_rnorm = rnormi[count];
            } else if (npi[count] == max_npi && rnormi[count] < min_rnorm) {
                best_ind = count;
                min_rnorm = rnormi[count];
            }
            count++;
        }
        // TODO: after first scan, perhaps we can allow np to be up to 3% smaller, and choose the one with smaller rnormi,
        // the idea is that by reducing total number of process we lose speed by 3% or less, but we might gain speed up in
        // communication by more than 3%
        if (best_ind != -1) {
            dims[0] = dimx[best_ind]; dims[1] = dimy[best_ind]; dims[2] = dimz[best_ind];
        } else {
            dims[0] = nproc; dims[1] = 1; dims[2] = 1;
            *ierr = 1; // cannot find any admissable distribution
        }
    } else {
        *ierr = 1;
        exit(EXIT_FAILURE); // currently only works for 1d, 2d and 3d
    }
#undef NPSTRIDE
#undef NPSTRIDE3D
}



/**
 * @brief   Look for a 2D decomposition of number of processors according to application size.
 *          This function is adapted to optimizing 2D block cyclic distribution of matrices,
 *          where the number of processors are kept fixed and number of nodes/direction is not
 *          substantial regarding performance. The goal is to factorize nproc = a x b (a >= b),
 *          such that a and b are as close to each other as possible.
 *
 * @ref     This function uses algorithms copied from Rosetta Code
 *          https://rosettacode.org/wiki/Factors_of_an_integer#C
 *          to find all factors of the number of processors.
 */
void ScaLAPACK_Dims_2D_BLCYC(int nproc, int *gridsizes, int *dims)
{
    int i;
    Factors ftors = {NULL, 0};

    // factorize nproc using alg from Rosetta Code
    factor(nproc, &ftors);

    // the factors are in increasing order
    // f0 x f1 = f2 * f3 = ... = nproc
    // f0 <= f1 <= f2 <= ...
    // go over the list of factors and find which one gives most 
    // uniform ng/proc in two directions
    char isTall = gridsizes[0] > gridsizes[1] ? 'y' : 'n';
    double diff = 0.0;
    int ind1 = 0;
    int ind2 = 0;
    double diff1, diff2, last_diff;
    last_diff = 1e20;
    for (i = 0; i < ftors.count/2; i++) {
        diff1 = fabs((double)gridsizes[1] / ftors.list[2*i] - (double)gridsizes[0] / ftors.list[2*i+1]);
        diff2 = fabs((double)gridsizes[0] / ftors.list[2*i] - (double)gridsizes[1] / ftors.list[2*i+1]);
        diff = diff1 < diff2 ? diff1 : diff2;
        if (i > 0 && diff > last_diff) {
            ind1 = (i - 1) * 2;
            ind2 = ind1 + 1;
            break;
        } else {
            ind1 = i * 2;
            ind2 = ind1 + 1;
        }
        last_diff = diff; // update old diff
    }

    if (ftors.count % 2 && i == ftors.count/2) {
        diff1 = fabs((double)gridsizes[1] / ftors.list[ftors.count-1] - (double)gridsizes[0] / ftors.list[ftors.count-1]);
        if (diff1 < diff) {
            ind1 = ind2 = ftors.count-1;
        }
    }

    if (isTall == 'y') {
        dims[0] = ftors.list[ind2];
        dims[1] = ftors.list[ind1];
    } else {
        dims[0] = ftors.list[ind1];
        dims[1] = ftors.list[ind2];
    }

    free(ftors.list);
}



/**
 * @brief  For the given parallelization params, find how many count of 
 *         work load each process is assigned, we take the ceiling since
 *         at least one process would have that many columns and will
 *         dominate the cost.
 *         
 **/
int workload_assigned(const int Nk, const int Ns, 
    const int np1, const int np2)
{
    return ceil_div(Nk,np1) * ceil_div(Ns,np2);
}


/**
 * @brief  Caluclate the efficiency for a given set of parameters.
 *         This is derived based on the assumption that there are
 *         infinite grid points (Nd) so that np can not be larger
 *         than Nk*Ns*Nd in any case. In practice, Nd ~ O(1e6), so
 *         this assumption is quite reasonable.
 *         1. Ideal load: Nk*Ns*Nd/np
 *         2. Actual load: ceil(Nk/np1)*ceil(Ns/np2)*ceil(Nd/np3) 
 *                      ~= ceil(Nk/np1)*ceil(Ns/np2)* (Nd/np3) #since Nd is large
 *         3. efficiency = Ideal load / Actual load
 *                      ~= Nk*Ns*np3/ (ceil(Nk/np1)*ceil(Ns/np2)*np)
 * @param Nk   Number of kpoints (after symmetry reduction).
 * @param Ns   Number of states/bands.
 * @param np   Number of processors available.
 * @param np1  Number of kpoint groups for kpoint parallelization.
 * @param np2  Number of band groups for band parallelization.
 * @param np3  Number of domain groups for domain parallelization.
 *
 **/
double work_load_efficiency(
    const int Nk, const int Ns, const int np, 
    const int np1, const int np2, const int np3
)
{
    int ncols = workload_assigned(Nk, Ns, np1, np2);
    return Nk * Ns * np3 / (double)(ncols * np);
}


/**
 * @brief  Caluclate a division of processors in a 2D Cartesian grid.
 *
 *         For a given number of processors, choose np1 and np2 so that the maximum 
 *         work assigned to each process is the smallest, i.e., choose x * y <= np, 
 *         s.t., ceil(N1/x) * ceil(N2/y) is minimum.
 *
 *         Here we assume the parallelization of the two properties are the same,
 *         therefore there's no preference to use more or less processes in any
 *         dimension. 
 *         
 *         The objective function and the constraint are both quadratic, which 
 *         makes this a very hard problem in general.
 *         
 *         In this func, we try to find a reasonably good solution using the following 
 *         strategy: note that if x0 | N1 and y0 | N2 ( "|" means divides), then 
 *         (x0,y0) is a solution to the problem with a weaker constraint x*y <= x0*y0.
 *         We can then keep reducing the constraint from np down until we find a 
 *         solution for the subproblem, and choose the best of all the searched combo. 
 *
 * @param N1   Number of properties to parallelize on the grid in the 1st dimension.
 * @param N2   Number of properties to parallelize on the grid in the 2nd dimension.
 * @param np   Number of processors available.
 *
 **/
void dims_divide_2d(const int N1, const int N2, const int np, int *np1, int *np2)
{
    int search_count = 0;
    
    // initialize with the naive way, parallelize as much as possible for N1 first
    int cur1 = min(np, N1);
    int cur2 = min(np/cur1, N2);
    int best = workload_assigned(N1,N2,cur1,cur2); 

    int np_up = np > N1*N2 ? N1*N2 : np; // upper bound of np that can be used
    for (int p = np_up; p > 0; p--) {
        // find factors of p
        Factors facs = {NULL, 0};
        sorted_factor(p, &facs);
        int nfacs = facs.count; // total number of factors
        for (int i = 0; i < nfacs; i++) {
            int fac1 = facs.list[i]; // small to large
            int fac2 = p / fac1; // large to small
            // use larger factor for the first dim from the beginning
            int load = workload_assigned(N1,N2,fac2,fac1);
            if (load < best) {
                best = load;
                cur1 = fac2;
                cur2 = fac1;
            }
            if ((N1 % fac2 == 0 && N2 % fac1 == 0) || search_count > 1e5) {
                *np1 = cur1;
                *np2 = cur2;
                free(facs.list);
                return;
            }
        }
        free(facs.list);
    }
}



/**
 * @brief  Caluclate a division of processors in for kpoint, band, domain (KBD) 
 *         parallelization.
 *
 *         For a given number of processors, choose npkpt, npband, and npdomain so 
 *         that the maximum work assigned to each process is the smallest, i.e., 
 *         choose x * y * z <= np, s.t., 
 *         ceil(Nk/x) * ceil(Ns/y) * ceil(Nd/z) is minimum.
 *
 *         Here we assume the parallelization of the two properties are the same,
 *         therefore there's no preference to use more or less processes in any
 *         dimension. 
 *         
 *         The objective function and the constraint are both nonlinear, which 
 *         makes this a very hard problem in general.
 *         
 *         In this func, we try to find a reasonably good solution using the following 
 *         strategy: since the parallelization over kpoint and band are both very 
 *         efficient, we try to parallel over kpoint and band first, then we consider
 *         parallelization over domain. There are two cases: np <=  or > Nk * Ns.
 *         Case 1: np <= Nk * Ns. Then we try to fix npdomain = 1:10, and find the best
 *         parameters for the given npdomain, we always prefer to use less npdomain if
 *         possible.
 *         Case 2: np > Nk * Ns. Then we try to provide Nk*Ns ./ [1:10] for kpoint and
 *         band parallelization, and pick the best combination. Again we prefer to use
 *         as less npdomain (more for K & B) if work load is the similar.
 *
 * @param Nk        Number of kpoints (after symmetry reduction).
 * @param Ns        Number of states.
 * @param gridsizes Number of grid points in all three directions.
 * @param np        Number of processors available.
 * @param np1 (OUT) Number of kpoint groups.
 * @param np2 (OUT) Number of band groups.
 * @param np3 (OUT) Number of domain groups.
 **/
void dims_divide_kbd(
    const int Nk, const int Ns, const int *gridsizes,
    const int np, int *np1, int *np2, int *np3)
{
#define LEN_NPDM 8

    // SPARC_Dims_create(nproc, ndims, gridsizes, minsize, int *dims, int *ierr); 
    int cur1 = min(np,Nk);
    int cur2 = min(np/cur1,Ns);
    int cur3 = np / (cur1 * cur2);
    double best_weight = work_load_efficiency(Nk,Ns,np,cur1,cur2,cur3);
    if (best_weight > 0.97) { // prefer kpoint to band
        *np1 = cur1; *np2 = cur2; *np3 = cur3;
        return;
    }

    int npkpt, npband, npdm;
    double weight;

    int npdm_list[LEN_NPDM];
    int npdm_count = LEN_NPDM;
    if (np > Nk * Ns) {
        int npdm_base = np / (Nk * Ns);
        int count = 0;
        for (int i = 1; i <= LEN_NPDM; i++) {
            if (npdm_base * i <= np) {
                npdm_list[count] = npdm_base * i;
                count++;
            }
        }
        npdm_count = count;
    } else {
        int count = 0;
        for (int i = 1; i <= LEN_NPDM; i++) {
            npdm_list[count] = i;
            count++;
        }
        npdm_count = count;
    }

    for (int i = 0; i < npdm_count; i++) {
        // # TODO: check domain parallelization first
        // # npNx,npNy,npNz = domain_paral(Nx,Ny,Nz,npdomain)
        // # if npNx*npNy*npNz != npdomain:
        // #     continue
        npdm = npdm_list[i];
        int np_2d = np / npdm;
        if (np_2d < 1) continue;
        
        dims_divide_2d(Nk, Ns, np_2d, &npkpt, &npband);
        npdm = np / (npkpt * npband);
        weight = work_load_efficiency(Nk,Ns,np,npkpt,npband,npdm);
        if (weight > best_weight) {
            cur1 = npkpt;
            cur2 = npband;
            cur3 = npdm; 
            best_weight = weight;
        }
        if (weight > 0.95) break;
    }

    *np1 = cur1;
    *np2 = cur2;
    *np3 = cur3;
}


double skbd_band_efficiency(const int x) {
    double eff = 1.00;
    double x_table[5] = {1,   8,   27,  125, 512};
    double y_table[5] = {1.00,0.95,0.75,0.55,0.45};
    int len = 5;
    for (int i = 0; i < len - 1; i++) {
        if (x >= x_table[i] && x < x_table[i+1]) {
            double slope = (y_table[i+1]-y_table[i])/(x_table[i+1]-x_table[i]);
            eff = y_table[i] + (x - x_table[i]) * slope;
        }
    }

    if (x > x_table[len-1]) eff = 0.44;
    return eff;
}


double skbd_domain_efficiency(const int x) {
    double eff = 1.00;
    double x_table[7] = {1,   2,   8,   20,  27  ,120 ,200 };
    double y_table[7] = {1.00,0.80,0.75,0.65,0.60,0.40,0.30};
    int len = 7;
    for (int i = 0; i < len - 1; i++) {
        if (x >= x_table[i] && x < x_table[i+1]) {
            double slope = (y_table[i+1]-y_table[i])/(x_table[i+1]-x_table[i]);
            eff = y_table[i] + (x - x_table[i]) * slope;
        }
    }

    if (x > x_table[len-1]) eff = 0.29;
    return eff;
}

/**
 * @brief  The scaling factor introduced by DMndbyNband
 * 
 *         Adding a side effect for tall thin local psi where applying ACE operator is slow
 */
double scaling_DMndbyNband(const int Ns, const int *gridsizes, const int npb, const int npd)
{
    int Nd = gridsizes[0] * gridsizes[1] * gridsizes[2];
    double DMndbyNband = ceil_div(Nd,npd)/((double)ceil_div(Ns,npb));
    // empirically DMndbyNband around 40 to 80 is good
    double x = (DMndbyNband - 60)/20.0;
    double cheby4 = max(0,16*pow(x,4)-12*pow(x,2)+1);
    return pow(0.98,cheby4);
}

double skbd_weighted_efficiency(
    const int Nspin, const int Nk, const int Ns, 
    const int *gridsizes, const int np, const int nps, 
    const int npk, const int npb, const int npd, const int isfock)
{
    double eff;
    int ncols = workload_assigned(Nspin*Nk, Ns, nps*npk, npb);
    eff = Nspin * Nk * Ns * npd / (double)(ncols * np);
    // apply weight for band
    eff *= skbd_band_efficiency(npb);
    // apply weight for domain
    eff *= skbd_domain_efficiency(npd);
    
    if (isfock) {
        // scale by DMndbyNband coefficients 
        eff *= scaling_DMndbyNband(Ns, gridsizes, npb, npd);
    }
    return eff;
}

double work_load_efficiency_fock(
    const int Nk, const int Ns, const int *gridsizes,
    const int np, const int np1, const int np2, const int np3
)
{
    int ncols = workload_assigned(Nk, Ns, np1, np2);
    double eff = Nk * Ns * np3 / (double)(ncols * np);
    eff *= scaling_DMndbyNband(Ns, gridsizes, np2, np3);
    return eff;
}

void dims_divide_kbd_fock(
    const int Nk, const int Ns, const int *gridsizes,
    const int np, int *np1, int *np2, int *np3, const int minsize)
{
#define LEN_NPB 20

    // SPARC_Dims_create(nproc, ndims, gridsizes, minsize, int *dims, int *ierr); 
    int cur1 = min(np,Nk);
    int npNdx_max = gridsizes[0]/minsize;
    int npNdy_max = gridsizes[1]/minsize;
    int npNdz_max = gridsizes[2]/minsize;
    int npd_max = npNdx_max * npNdy_max * npNdz_max;
    int cur3 = min(npd_max,np/cur1);
    int cur2 = min(np/(cur1*cur3),Ns);
    double best_weight = work_load_efficiency_fock(Nk,Ns,gridsizes,np,cur1,cur2,cur3);
    
    if (best_weight > 0.9) { // prefer kpt to domain to band
        *np1 = cur1; *np2 = cur2; *np3 = cur3;
        return;
    }

    int npkpt, npband, npdm;
    double weight;

    int npb_list[LEN_NPB];
    int npb_count = LEN_NPB;
    if (np > Nk * npd_max) {
        int npb_base = np / (Nk * npd_max);
        int count = 0;
        for (int i = 1; i <= LEN_NPB; i++) {
            if (npb_base * i <= np) {
                npb_list[count] = npb_base * i;
                count++;
            }
        }
        npb_count = count;
    } else {
        int count = 0;
        for (int i = 1; i <= LEN_NPB; i++) {
            npb_list[count] = i;
            count++;
        }
        npb_count = count;
    }

    for (int i = 0; i < npb_count; i++) {
        npband = npb_list[i];
        int np_2d = np / npband;
        if (np_2d < 1) continue;
        
        // Use as much npkpt as possible for fock
        npkpt = min(np_2d,Nk);
        npdm = np_2d / npkpt;
        npband = np / (npkpt * npdm);
        weight = work_load_efficiency_fock(Nk,Ns,gridsizes,np,npkpt,npband,npdm);
        if (weight > best_weight) {
            cur1 = npkpt;
            cur2 = npband;
            cur3 = npdm; 
            best_weight = weight;
        }
        if (weight > 0.93) break;
    }

    *np1 = cur1;
    *np2 = cur2;
    *np3 = cur3;
}

/**
 * @brief  Caluclate a division of processors in for spin, kpoint, band, 
 *         domain (SKBD). 
 *         parallelization.
 *
 * @param Nspin     Number of spin, 1 or 2.
 * @param Nk        Number of kpoints (after symmetry reduction).
 * @param Ns        Number of states.
 * @param gridsizes Number of grid points in all three directions.
 * @param np        Number of processors available.
 * @param nps (OUT) Number of spin groups.
 * @param npk (OUT) Number of kpoint groups.
 * @param npp (OUT) Number of band groups.
 * @param npd (OUT) Number of domain groups.
 * @param minsize   Minimum size in domain parallelization
 * @param isfock    Flag for if it's hybrid calculation
 **/
void dims_divide_skbd(
    const int Nspin, const int Nk, const int Ns, 
    const int *gridsizes, const int np, 
    int *nps, int *npk, int *npb, int *npd, int minsize, int isfock)
{
    if (Nspin != 1 && Nspin != 2) 
        exit(1);

    int nps_cur, npk_cur, npb_cur, npd_cur;
    double weight_cur;
    // first set naive way
    if (!isfock) {
        nps_cur = min(Nspin,np);
        npk_cur = min(Nk,np/nps_cur);
        npb_cur = min(Ns,np/(nps_cur*npk_cur));
        npd_cur = np/(nps_cur*npk_cur*npb_cur);
    } else {
        nps_cur = min(Nspin,np);
        npk_cur = min(Nk,np/nps_cur);
        int npNdx_max = gridsizes[0]/minsize;
        int npNdy_max = gridsizes[1]/minsize;
        int npNdz_max = gridsizes[2]/minsize;
        int npd_max = npNdx_max * npNdy_max * npNdz_max;
        npd_cur = min(npd_max,np/(nps_cur*npk_cur));
        npb_cur = np/(nps_cur*npk_cur*npd_cur);
    }
    weight_cur = skbd_weighted_efficiency(Nspin,Nk,Ns,gridsizes,np,nps_cur,npk_cur,npb_cur,npd_cur,isfock);


    // try with both Nspin = 1 
    int nps_1, npk_1, npb_1, npd_1;
    nps_1 = 1;
    if (!isfock) {
        dims_divide_kbd(Nk,Ns,gridsizes,np/(nps_1),&npk_1,&npb_1,&npd_1);
    } else {
        dims_divide_kbd_fock(Nk,Ns,gridsizes,np/(nps_1),&npk_1,&npb_1,&npd_1,minsize);
    }
    // double weight_1 = work_load_efficiency(Nspin*Nk,Ns,np,nps_1*npk_1,npb_1,npd_1);
    double weight_1;
    weight_1 = skbd_weighted_efficiency(Nspin,Nk,Ns,gridsizes,np,nps_1,npk_1,npb_1,npd_1,isfock);
    if (weight_1 > weight_cur) {
        nps_cur = nps_1;
        npk_cur = npk_1;
        npb_cur = npb_1;
        npd_cur = npd_1;
        weight_cur = weight_1;
    }

    // if there's spin, also try npspin = 2, and see if gives a better result
    if (Nspin > 1 && np > 1) {
        int nps_2, npk_2, npb_2, npd_2;
        nps_2 = 2;
        if (!isfock) {
            dims_divide_kbd(Nk,Ns,gridsizes,np/(nps_2),&npk_2,&npb_2,&npd_2);
        } else {
            dims_divide_kbd_fock(Nk,Ns,gridsizes,np/(nps_2),&npk_2,&npb_2,&npd_2,minsize);
        }
        double weight_2;
        weight_2 = skbd_weighted_efficiency(Nspin,Nk,Ns,gridsizes,np,nps_2,npk_2,npb_2,npd_2,isfock);
        if (weight_2 >= weight_cur) {
            nps_cur = nps_2;
            npk_cur = npk_2;
            npb_cur = npb_2;
            npd_cur = npd_2;
            weight_cur = weight_2;
        }
    }

    *nps = nps_cur;
    *npk = npk_cur;
    *npb = npb_cur;
    *npd = npd_cur;
}





/**
 * @brief   This function sets up the input structures needed for the function D2D(),
 *          which transfers data from one domain decomposition to another domain
 *          decomposition. It's required that send_comm, recv_comm and union_comm overlap
 *          at the same root process.
 *
 *          One can first call this function to set up d2d_sender and d2d_recvr, which
 *          contains the number of processes to communicate with along with their ranks
 *          in the union_comm.
 *          Note: In principle, it is not required to call this function if there're other
 *          ways to find out the senders and receivers for each process.
 */
void Set_D2D_Target(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, int *gridsizes, int *sDMVert, int *rDMVert,
         MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    double t3, t4;
#endif
    int rank_recv_comm, nproc_recv_comm, rank_send_comm, nproc_send_comm,
        nproc_union_comm, rank_union_comm, coords_send_comm[3], coords_recv_comm[3],
        send_coord_start[3], send_coord_end[3], nsend[3], *send_coords,
        *rcounts_send_comm, *scounts_send_comm, *displs_send_comm, *scoords_vec,
        *srank_vec, *srank_vec_union, n, i, j, count, coord_temp[3], nsend_tot,
        nrecv_tot, recv_coord_start[3], recv_coord_end[3], nrecv[3], *recv_coords,
        *rcounts_recv_comm, *scounts_recv_comm, *displs_recv_comm, *rcoords_vec,
        *rrank_vec, *rrank_vec_union;
    MPI_Group send_group, recv_group, union_group;

    // initialize pointers to avoid "may be uninitialized" warnings
    send_coords = NULL; 
    recv_coords = NULL;
    scounts_send_comm = NULL;
    scounts_recv_comm = NULL;
    rcounts_send_comm = NULL;
    rcounts_recv_comm = NULL;
    displs_send_comm = NULL;
    displs_recv_comm = NULL;
    scoords_vec = NULL;
    rcoords_vec = NULL;
    srank_vec = NULL;
    rrank_vec = NULL;
    srank_vec_union = NULL;
    rrank_vec_union = NULL;

    // extract groups from the provided communicators, used for translating ranks
    if (send_comm != MPI_COMM_NULL) {
        MPI_Comm_group(send_comm, &send_group);
        MPI_Comm_rank(send_comm, &rank_send_comm);
        MPI_Comm_size(send_comm, &nproc_send_comm);
    } else {
        send_group = MPI_GROUP_EMPTY;
        rank_send_comm = -1;
        nproc_send_comm = 0;
    }
    if (recv_comm != MPI_COMM_NULL) {
        MPI_Comm_group(recv_comm, &recv_group);
        MPI_Comm_rank(recv_comm, &rank_recv_comm);
        MPI_Comm_size(recv_comm, &nproc_recv_comm);
    } else {
        recv_group = MPI_GROUP_EMPTY;
        rank_recv_comm = -1;
        nproc_recv_comm = 0;
    }

    MPI_Comm_group(union_comm, &union_group);
    MPI_Comm_size(union_comm, &nproc_union_comm);
    MPI_Comm_rank(union_comm, &rank_union_comm);

    // initialize nsend_tot and nrecv_tot to zero
    d2d_sender->n_target = 0;
    d2d_recvr ->n_target = 0;

    // set up send processes
    if (send_comm != MPI_COMM_NULL) {
#ifdef DEBUG
        t3 = MPI_Wtime();
#endif
        MPI_Cart_coords(send_comm, rank_send_comm, 3, coords_send_comm);
        nsend_tot = 1;
        // find out in each dimension, how many recv processes the local domain spans over
        for (n = 0; n < 3; n++) {
            send_coord_start[n] = block_decompose_rank(gridsizes[n], rdims[n], sDMVert[n*2]);
            send_coord_end[n] = block_decompose_rank(gridsizes[n], rdims[n], sDMVert[n*2+1]);
            nsend[n] = (send_coord_end[n] - send_coord_start[n] + 1);
            nsend_tot *= nsend[n];
        }
        d2d_sender->n_target = nsend_tot;
        d2d_sender->target_ranks = (int *)malloc(nsend_tot * sizeof(int));

        send_coords = (int *)malloc(nsend_tot * 3 * sizeof(int));
        // find out all the coordinates of the receiver processes in the recv_comm Cart Topology
        c_ndgrid(3, send_coord_start, send_coord_end, send_coords);

#ifdef DEBUG
        t4 = MPI_Wtime();
        if (rank == 0) printf("======Set_D2D_Target: find receivers in each process (c_ndgrid) in send_comm took %.3f ms\n", (t4-t3)*1e3);
        t3 = MPI_Wtime();
#endif

        // TODO: Gatherv the send_coords in root process in send_comm
        if (rank_send_comm == 0) { // we require both comms overlap at rank 0
            rcounts_send_comm = (int *)malloc(nproc_send_comm * sizeof(int));
            scounts_send_comm = (int *)malloc(nproc_send_comm * sizeof(int));
            displs_send_comm  = (int *)malloc((nproc_send_comm+1) * sizeof(int));
            // first gather the number of coordinates to be received from each process
            MPI_Gather(&nsend_tot, 1, MPI_INT, scounts_send_comm, 1, MPI_INT, 0, send_comm);

            // set displs_send_comm
            displs_send_comm[0] = 0;
            for (i = 0; i < nproc_send_comm; i++) {
                rcounts_send_comm[i] = 3 * scounts_send_comm[i];
                displs_send_comm[i+1] = displs_send_comm[i] + rcounts_send_comm[i];
            }

            scoords_vec = (int *)malloc(displs_send_comm[nproc_send_comm] * sizeof(int));
            srank_vec = (int *)malloc(displs_send_comm[nproc_send_comm] / 3 * sizeof(int));
            srank_vec_union = (int *)malloc(displs_send_comm[nproc_send_comm] / 3 * sizeof(int));

            // gather the coordinates
            MPI_Gatherv(send_coords, nsend_tot * 3, MPI_INT, scoords_vec, rcounts_send_comm, displs_send_comm, MPI_INT, 0, send_comm);

            // In root process, find out rank corresponding to each coord in recv_comm,
            // this works because root process is in both send_comm and recv_comm
            count = 0;
            for (n = 0; n < nproc_send_comm; n++) {
                // find srank_vec
                for (j = 0; j < scounts_send_comm[n]; j++) {
                    coord_temp[0] = scoords_vec[displs_send_comm[n]+j];
                    coord_temp[1] = scoords_vec[displs_send_comm[n]+scounts_send_comm[n]+j];
                    coord_temp[2] = scoords_vec[displs_send_comm[n]+2*scounts_send_comm[n]+j];
                    // find rank corresponding to send_coords in recv_comm
                    MPI_Cart_rank(recv_comm, coord_temp, &srank_vec[count]);
                    count++;
                }
            }
            // Translate ranks in recv_comm to ranks in union_comm using MPI_Group_translate_ranks
            MPI_Group_translate_ranks(recv_group, count, srank_vec, union_group, srank_vec_union);

            // Scatter the ranks back to each process in send_comm
            displs_send_comm[0] = 0;
            for (i = 0; i < nproc_send_comm; i++)
                displs_send_comm[i+1] = displs_send_comm[i] + scounts_send_comm[i];
            MPI_Scatterv(srank_vec_union, scounts_send_comm, displs_send_comm, MPI_INT, d2d_sender->target_ranks, nsend_tot, MPI_INT, 0, send_comm);
        } else {
            // gather the number of coordinates to be received from each process
            MPI_Gather(&nsend_tot, 1, MPI_INT, scounts_send_comm, 1, MPI_INT, 0, send_comm);
            // gather the coordinates
            MPI_Gatherv(send_coords, nsend_tot * 3, MPI_INT, scoords_vec, rcounts_send_comm, displs_send_comm, MPI_INT, 0, send_comm);
            // Scatter the ranks back to each process in send_comm
            MPI_Scatterv(srank_vec_union, scounts_send_comm, displs_send_comm, MPI_INT, d2d_sender->target_ranks, nsend_tot, MPI_INT, 0, send_comm);
        }

#ifdef DEBUG
        t4 = MPI_Wtime();
        if (rank == 0) printf("======Set_D2D_Target: Gather and Scatter receivers in send_comm took %.3f ms\n", (t4-t3)*1e3);
#endif
    }

    // set up receiver processes
    if (recv_comm != MPI_COMM_NULL) {
        MPI_Cart_coords(recv_comm, rank_recv_comm, 3, coords_recv_comm);
        nrecv_tot = 1;
        // find in each dimension, how many send processes the local domain spans over
        for (n = 0; n < 3; n++) {
            recv_coord_start[n] = block_decompose_rank(gridsizes[n], sdims[n], rDMVert[n*2]);
            recv_coord_end[n] = block_decompose_rank(gridsizes[n], sdims[n], rDMVert[n*2+1]);
            nrecv[n] = (recv_coord_end[n] - recv_coord_start[n] + 1);
            nrecv_tot *= nrecv[n];
        }
        d2d_recvr->n_target = nrecv_tot;
        d2d_recvr->target_ranks = (int *)malloc(nrecv_tot * sizeof(int));
        recv_coords = (int *)malloc(nrecv_tot * 3 * sizeof(int));
        // find out all the coords of the send process in the send_comm topology
        c_ndgrid(3, recv_coord_start, recv_coord_end, recv_coords);
        // Gather all the recv_coords in root process to find out their ranks in send_comm
        if (rank_recv_comm == 0) {
            rcounts_recv_comm = (int *)malloc(nproc_recv_comm * sizeof(int));
            scounts_recv_comm = (int *)malloc(nproc_recv_comm * sizeof(int));
            displs_recv_comm  = (int *)malloc((nproc_recv_comm+1) * sizeof(int));

            // first gather the number of coordinates to be recived from each process in recv_comm
            MPI_Gather(&nrecv_tot, 1, MPI_INT, scounts_recv_comm, 1, MPI_INT, 0, recv_comm);

            // set displs_recv_comm
            displs_recv_comm[0] = 0;
            for (i = 0; i < nproc_recv_comm; i++) {
                rcounts_recv_comm[i] = 3 * scounts_recv_comm[i];
                displs_recv_comm[i+1] = displs_recv_comm[i] + rcounts_recv_comm[i];
            }

            rcoords_vec = (int *)malloc(displs_recv_comm[nproc_recv_comm] * sizeof(int));
            rrank_vec = (int *)malloc(displs_recv_comm[nproc_recv_comm] / 3 * sizeof(int));
            rrank_vec_union = (int *)malloc(displs_recv_comm[nproc_recv_comm] / 3 * sizeof(int));

            // gather the coordinates
            MPI_Gatherv(recv_coords, nrecv_tot * 3, MPI_INT, rcoords_vec, rcounts_recv_comm, displs_recv_comm, MPI_INT, 0, recv_comm);

            // In root process, find out rank corresponding to each coord in send_comm,
            // this works because root process is in both send_comm and recv_comm
            count = 0;
            for (n = 0; n < nproc_recv_comm; n++) {
                // find rrank_vec
                for (j = 0; j < scounts_recv_comm[n]; j++) {
                    coord_temp[0] = rcoords_vec[displs_recv_comm[n]+j];
                    coord_temp[1] = rcoords_vec[displs_recv_comm[n]+scounts_recv_comm[n]+j];
                    coord_temp[2] = rcoords_vec[displs_recv_comm[n]+2*scounts_recv_comm[n]+j];
                    // find rank corresponding to recv_coords in send_comm
                    MPI_Cart_rank(send_comm, coord_temp, &rrank_vec[count++]);
                }
            }

            // Translate ranks in send_comm to ranks in union_comm using MPI_Group_translate_ranks
            MPI_Group_translate_ranks(send_group, count, rrank_vec, union_group, rrank_vec_union);

            // Scatter the ranks back to each process in send_comm
            displs_recv_comm[0] = 0;
            for (i = 0; i < nproc_recv_comm; i++)
                displs_recv_comm[i+1] = displs_recv_comm[i] + scounts_recv_comm[i];
            MPI_Scatterv(rrank_vec_union, scounts_recv_comm, displs_recv_comm, MPI_INT, d2d_recvr->target_ranks, nrecv_tot, MPI_INT, 0, recv_comm);

        } else {
            // gather the number of coordinates to be received from each process
            MPI_Gather(&nrecv_tot, 1, MPI_INT, scounts_recv_comm, 1, MPI_INT, 0, recv_comm);
            // gather the coordinates
            MPI_Gatherv(recv_coords, nrecv_tot * 3, MPI_INT, rcoords_vec, rcounts_recv_comm, displs_recv_comm, MPI_INT, 0, recv_comm);
            // Scatter the ranks back to each process in send_comm
            MPI_Scatterv(rrank_vec_union, scounts_recv_comm, displs_recv_comm, MPI_INT, d2d_recvr->target_ranks, nrecv_tot, MPI_INT, 0, recv_comm);
        }
    }

    MPI_Group_free(&union_group);

    if (send_comm != MPI_COMM_NULL) {
        MPI_Group_free(&send_group);
        free(send_coords);
        if (rank_send_comm == 0) {
            free(rcounts_send_comm);
            free(scounts_send_comm);
            free(displs_send_comm);
            free(scoords_vec);
            free(srank_vec);
            free(srank_vec_union);
        }
    }

    if (recv_comm != MPI_COMM_NULL) {
        MPI_Group_free(&recv_group);
        free(recv_coords);
        if (rank_recv_comm == 0) {
            free(rcounts_recv_comm);
            free(scounts_recv_comm);
            free(displs_recv_comm);
            free(rcoords_vec);
            free(rrank_vec);
            free(rrank_vec_union);
        }
    }
}



/**
 * @brief   Free D2D_OBJ structure created by Set_D2D_Target or otherwise created manually
 *          as input to D2D.
 */
void Free_D2D_Target(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, MPI_Comm send_comm, MPI_Comm recv_comm)
{
    if (send_comm != MPI_COMM_NULL) {
        free(d2d_sender->target_ranks);
    }
    if (recv_comm != MPI_COMM_NULL) {
        free(d2d_recvr->target_ranks);
    }
}



/**
 * @brief   Transfer data from one 3-d Domain Decomposition to another.
 *
 *          This function has the same objective as DD2DD. However, the process of finding sender
 *          and receiver ranks are seperated out from this function. The reason for this change is
 *          that 1: the ranks remain the same for every simulation, so it only has to be done once,
 *          2: finding senders and receivers can become costly for large number of processes.
 *
 *          Note: In this function we assume Domain Decomposition is quasi-uniform, i.e., each
 *          process will roughly have the same number of nodes. If the dim of the Cartesian in one
 *          direction does not divide the gridsize, the first few processes will have one more node
 *          than the others.
 *
 * @param sdata         (INPUT) Local piece of the data owned by processes in send_comm.
 * @param rdata         (OUTPUT) Re-distributed data owned by processes in recv_comm.
 * @param send_comm     Original communicator in which the data was distributed. Processors not
 *                      part of the original DD should provide MPI_COMM_NULL for this argument.
 *                      send_comm is assumed to have a Cartesian topology.
 * @param sdims         Dimensions of the send_comm.
 * @param recv_comm     New Domain Decomposition communicator, processes that are not part of this
 *                      should provide MPI_COMM_NULL. recv_comm is assumed to have a Cartesian topology.
 * @param rdims         Dimensions of the recv_comm.
 * @param union_comm    Union communicator. Union communicator can be any communicator that satisfy the
 *                      following conditions:
 *                          1. send_comm and recv_comm are both sub-comms of union_comm.
 *                          2. it is not required in this function that send_comm, recv_comm and
 *                             union_comm share root process any more. As long as senders and receivers
 *                             are provided correctly.
 * @param unit_size     either sizeof(double) or sizeof(double _Complex) for the type of sdata and rdata
 */
void D2D(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, int *gridsizes, int *sDMVert, void *sdata, int *rDMVert,
         void *rdata, MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm, int unit_size)
{
    assert(unit_size == 8 || unit_size == 16);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG_D2D
    double t1, t2, t3, t4;
#endif

    int ndims, rank_recv_comm, rank_send_comm,
        coords_send_comm[3], coords_recv_comm[3];
    ndims = 3;

#ifdef DEBUG_D2D
    t1 = MPI_Wtime();
#endif

    int i, j, k, iloc, jloc, kloc, n;
    int nsend_tot, *send_coord_start, *send_coord_end, *send_coords;
    nsend_tot = 0;
    send_coord_start = NULL;
    send_coord_end = NULL;
    send_coords = NULL;

    // buffers
    int DMnd, *DMnxi, *sDMnxi, *rDMnxi, *DMVert_temp;
    void **sendbuf, **recvbuf;
    sendbuf = NULL;
    recvbuf = NULL;

    int *coord_temp = (int *)malloc(ndims * sizeof(int));
    DMVert_temp = (int *)malloc(2 * ndims * sizeof(int));
    DMnxi = (int *)malloc(ndims * sizeof(int));
    sDMnxi = (int *)malloc(ndims * sizeof(int));
    rDMnxi = (int *)malloc(ndims * sizeof(int));

    MPI_Request *send_request, *recv_request;
    send_request = NULL;
    recv_request = NULL;

    // set up send processes
    if (send_comm != MPI_COMM_NULL) {
#ifdef DEBUG_D2D
        t3 = MPI_Wtime();
#endif
        for (n = 0; n < 3; n++) {
            sDMnxi[n] = sDMVert[2*n+1] - sDMVert[2*n] + 1;
        }
        send_coord_start = (int *)malloc(ndims * sizeof(int));
        send_coord_end = (int *)malloc(ndims * sizeof(int));
        MPI_Comm_rank(send_comm, &rank_send_comm);
        MPI_Cart_coords(send_comm, rank_send_comm, ndims, coords_send_comm);

        // find out in each dimension, how many recv processes the local domain spans over
        for (n = 0; n < 3; n++) {
            send_coord_start[n] = block_decompose_rank(gridsizes[n], rdims[n], sDMVert[n*2]);
            send_coord_end[n]   = block_decompose_rank(gridsizes[n], rdims[n], sDMVert[n*2+1]);
        }
        nsend_tot = d2d_sender->n_target;
        send_coords = (int *)malloc(nsend_tot * ndims * sizeof(int));

        // find out all the coordinates of the receiver processes in the recv_comm Cart Topology
        c_ndgrid(ndims, send_coord_start, send_coord_end, send_coords);

        #ifdef DEBUG_D2D
        t4 = MPI_Wtime();
            if (rank == 0) printf("======D2D: find receivers' coords in each process (c_ndgrid) in send_comm took %.3f ms\n", (t4-t3)*1e3);
        #endif

        if (unit_size == 8)
            sendbuf = malloc(nsend_tot * sizeof(double *));
        else
            sendbuf = malloc(nsend_tot * sizeof(double _Complex *));

        send_request = malloc(nsend_tot * sizeof(*send_request));
        // go over each receiver process send data to that process
        for (n = 0; n < nsend_tot; n++) {
            DMnd = 1;
            for (i = 0; i < 3; i++) {
                coord_temp[i] = send_coords[i*nsend_tot+n];
                // local domain in the receiver process
                DMVert_temp[2*i] = block_decompose_nstart(gridsizes[i], rdims[i], coord_temp[i]);
                DMnxi[i] = block_decompose(gridsizes[i], rdims[i], coord_temp[i]);
                DMVert_temp[2*i+1] = DMVert_temp[2*i] + DMnxi[i] - 1;
                // find intersect of local domains in send process and receiver process
                DMVert_temp[2*i] = max(DMVert_temp[2*i], sDMVert[2*i]);
                DMVert_temp[2*i+1] = min(DMVert_temp[2*i+1], sDMVert[2*i+1]);
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                DMnd *= DMnxi[i];
            }
            
            if (unit_size == 8)
                *((double **)sendbuf+n) = malloc( DMnd * sizeof(double));
            else
                *((double _Complex **)sendbuf+n) = malloc( DMnd * sizeof(double _Complex));

            for (k = 0; k < DMnxi[2]; k++) {
                kloc = k + DMVert_temp[4] - sDMVert[4];
                for (j = 0; j < DMnxi[1]; j++) {
                    jloc = j + DMVert_temp[2] - sDMVert[2];
                    for (i = 0; i < DMnxi[0]; i++) {
                        iloc = i + DMVert_temp[0] - sDMVert[0];
                        int indxsb = k*DMnxi[1]*DMnxi[0]+j*DMnxi[0]+i;
                        int indxsd = kloc*sDMnxi[1]*sDMnxi[0]+jloc*sDMnxi[0]+iloc;
                        if (unit_size == 8)
                            *(*((double **)sendbuf+n)+indxsb) = *((double *)sdata+indxsd);
                        else 
                            *(*((double _Complex **)sendbuf+n)+indxsb) = *((double _Complex *)sdata+indxsd);
                    }
                }
            }
            if (unit_size == 8)
                MPI_Isend(*((double **)sendbuf+n), DMnd, MPI_DOUBLE, d2d_sender->target_ranks[n], 111, union_comm, &send_request[n]);
            else 
                MPI_Isend(*((double _Complex **)sendbuf+n), DMnd, MPI_DOUBLE_COMPLEX, d2d_sender->target_ranks[n], 111, union_comm, &send_request[n]);
        }
    }
#ifdef DEBUG_D2D
    if (rank == 0) printf("======D2D: finished initiating send_comm! Start entering receivers comm\n");
#endif
    int nrecv_tot, *recv_coord_start, *recv_coord_end, *recv_coords;
    nrecv_tot = 0;
    recv_coord_start = NULL;
    recv_coord_end = NULL;
    recv_coords = NULL;

    // set up receiver processes
    if (recv_comm != MPI_COMM_NULL) {
#ifdef DEBUG_D2D
        t3 = MPI_Wtime();
#endif
        MPI_Comm_rank(recv_comm, &rank_recv_comm);
        MPI_Cart_coords(recv_comm, rank_recv_comm, ndims, coords_recv_comm);

        // local domain sizes
        for (n = 0; n < ndims; n++) {
            rDMnxi[n] = rDMVert[2*n+1] - rDMVert[2*n] + 1;
        }

        recv_coord_start = (int *)malloc(ndims * sizeof(int));
        recv_coord_end = (int *)malloc(ndims * sizeof(int));

        // find in each dimension, how many send processes the local domain spans over
        for (n = 0; n < ndims; n++) {
            recv_coord_start[n] = block_decompose_rank(gridsizes[n], sdims[n], rDMVert[n*2]);
            recv_coord_end[n]   = block_decompose_rank(gridsizes[n], sdims[n], rDMVert[n*2+1]);
        }

        nrecv_tot = d2d_recvr->n_target;
        recv_coords = (int *)malloc(nrecv_tot * ndims * sizeof(int));

        // find out all the coords of the send process in the send_comm topology
        c_ndgrid(ndims, recv_coord_start, recv_coord_end, recv_coords);

#ifdef DEBUG_D2D
    t4 = MPI_Wtime();
    if (rank == 0) printf("======D2D: find senders' coords in each process (c_ndgrid) in recv_comm took %.3f ms\n", (t4-t3)*1e3);
#endif

        if (unit_size == 8)
            recvbuf = malloc(nrecv_tot * sizeof(double *));
        else
            recvbuf = malloc(nrecv_tot * sizeof(double _Complex *));

        recv_request = malloc(nrecv_tot * sizeof(*recv_request));
        // go over each send process and receive data from that process
        for (n = 0; n < nrecv_tot; n++) {
            DMnd = 1;
            for (i = 0; i < ndims; i++) {
                coord_temp[i] = recv_coords[i*nrecv_tot+n];
                // local domain in the send process
                DMVert_temp[2*i] = block_decompose_nstart(gridsizes[i], sdims[i], coord_temp[i]);
                DMnxi[i] = block_decompose(gridsizes[i], sdims[i], coord_temp[i]);
                DMVert_temp[2*i+1] = DMVert_temp[2*i] + DMnxi[i] - 1;
                // find intersect of local domains in send process and receiver process
                DMVert_temp[2*i] = max(DMVert_temp[2*i], rDMVert[2*i]);
                DMVert_temp[2*i+1] = min(DMVert_temp[2*i+1], rDMVert[2*i+1]);
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                DMnd *= DMnxi[i];
            }
            if (unit_size == 8) {
                *((double **)recvbuf+n) = malloc( DMnd * sizeof(double));
                MPI_Irecv(*((double **)recvbuf+n), DMnd, MPI_DOUBLE, d2d_recvr->target_ranks[n], 111, union_comm, &recv_request[n]);
            } else {
                *((double _Complex **)recvbuf+n) = malloc( DMnd * sizeof(double _Complex));
                MPI_Irecv(*((double _Complex **)recvbuf+n), DMnd, MPI_DOUBLE_COMPLEX, d2d_recvr->target_ranks[n], 111, union_comm, &recv_request[n]);
            }
        }
    }

#ifdef DEBUG_D2D
    t2 = MPI_Wtime();
    if (rank == 0) printf("======D2D: initiated sending and receiving took %.3f ms\n", (t2-t1)*1e3);
#endif

    if (send_comm != MPI_COMM_NULL) {
        MPI_Waitall(nsend_tot, send_request, MPI_STATUS_IGNORE);
    }

    if (recv_comm != MPI_COMM_NULL) {
        MPI_Waitall(nrecv_tot, recv_request, MPI_STATUS_IGNORE);
        for (n = 0; n < nrecv_tot; n++) {
            DMnd = 1;
            for (i = 0; i < ndims; i++) {
                coord_temp[i] = recv_coords[i*nrecv_tot+n];
                // local domain in the send process
                DMVert_temp[2*i] = block_decompose_nstart(gridsizes[i], sdims[i], coord_temp[i]);
                DMnxi[i] = block_decompose(gridsizes[i], sdims[i], coord_temp[i]);
                DMVert_temp[2*i+1] = DMVert_temp[2*i] + DMnxi[i] - 1;
                // find intersect of local domains in send process and receiver process
                DMVert_temp[2*i] = max(DMVert_temp[2*i], rDMVert[2*i]);
                DMVert_temp[2*i+1] = min(DMVert_temp[2*i+1], rDMVert[2*i+1]);
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                DMnd *= DMnxi[i];
            }
            for (k = 0; k < DMnxi[2]; k++) {
                kloc = k + DMVert_temp[4] - rDMVert[4];
                for (j = 0; j < DMnxi[1]; j++) {
                    jloc = j + DMVert_temp[2] - rDMVert[2];
                    for (i = 0; i < DMnxi[0]; i++) {
                        iloc = i + DMVert_temp[0] - rDMVert[0];
                        int indxrb = k*DMnxi[1]*DMnxi[0]+j*DMnxi[0]+i;
                        int indxrd = kloc*rDMnxi[1]*rDMnxi[0]+jloc*rDMnxi[0]+iloc;
                        
                        if (unit_size == 8)
                            *((double *)rdata+indxrd) = *(*((double **)recvbuf+n)+indxrb);
                        else 
                            *((double _Complex *)rdata+indxrd) = *(*((double _Complex **)recvbuf+n)+indxrb);
                    }
                }
            }
        }
    }

    if (send_comm != MPI_COMM_NULL) {
        for (n = 0; n < nsend_tot; n++) {
            if (unit_size == 8)
                free(*((double **)sendbuf+n));
            else
                free(*((double _Complex **)sendbuf+n));
        }
        free(sendbuf);
        free(send_request);
        free(send_coord_start);
        free(send_coord_end);
        free(send_coords);
    }

    if (recv_comm != MPI_COMM_NULL) {
        for (n = 0; n < nrecv_tot; n++) {
            if (unit_size == 8)
                free(*((double **)recvbuf+n));
            else
                free(*((double _Complex **)recvbuf+n));
        }
        free(recvbuf);
        free(recv_request);
        free(recv_coord_start);
        free(recv_coord_end);
        free(recv_coords);
    }

    free(coord_temp);
    free(DMVert_temp);
    free(DMnxi);
    free(sDMnxi);
    free(rDMnxi);
}


void DD2DD(SPARC_OBJ *pSPARC, int *gridsizes, int *sDMVert, void *sdata, int *rDMVert, void *rdata,
           MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm, int unit_size)
{
    D2D_OBJ d2d_sender, d2d_recvr;
    Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, sDMVert, rDMVert, send_comm, sdims, recv_comm, rdims, union_comm);
    D2D(&d2d_sender, &d2d_recvr, gridsizes, sDMVert, sdata, rDMVert, rdata, send_comm, sdims, recv_comm, rdims, union_comm, unit_size);
    Free_D2D_Target(&d2d_sender, &d2d_recvr, send_comm, recv_comm);
}

#ifdef USE_DP_SUBEIG
void BP2DP(
    const MPI_Comm comm, const int nproc, 
    const int nrow_bp, const int ncol_bp, const int *row_dp_displs,
    const int *bp2dp_sendcnts, const int *bp2dp_sdispls, 
    const int *dp2bp_sendcnts, const int *dp2bp_sdispls, 
    const int unit_size, const void *bp_data, void *packbuf, void *dp_data
)
{
    // Pack the send data for MPI_Alltoallv
    if (unit_size == 8)
    {
        const double *bp_data_ = (const double*) bp_data;
        double *packbuf_ = (double*) packbuf;
        for (int i = 0; i < nproc; i++)
        {
            int nrow_dp_i = row_dp_displs[i + 1] - row_dp_displs[i];
            const double *src_ptr = bp_data_ + row_dp_displs[i];
            double *dst_ptr = packbuf_ + bp2dp_sdispls[i];
            copy_mat_blk(
                unit_size, src_ptr, nrow_bp, 
                nrow_dp_i, ncol_bp, dst_ptr, nrow_dp_i
            );
        }
    }
    if (unit_size == 16)
    {
        const double _Complex *bp_data_ = (const double _Complex*) bp_data;
        double _Complex *packbuf_ = (double _Complex*) packbuf;
        for (int i = 0; i < nproc; i++)
        {
            int nrow_dp_i = row_dp_displs[i + 1] - row_dp_displs[i];
            const double _Complex *src_ptr = bp_data_ + row_dp_displs[i];
            double _Complex *dst_ptr = packbuf_ + bp2dp_sdispls[i];
            copy_mat_blk(
                unit_size, src_ptr, nrow_bp, 
                nrow_dp_i, ncol_bp, dst_ptr, nrow_dp_i
            );
        }
    }
    
    // Do MPI_Alltoallv
    if (unit_size == 8)
    {
        MPI_Alltoallv(
            packbuf, bp2dp_sendcnts, bp2dp_sdispls, MPI_DOUBLE,
            dp_data, dp2bp_sendcnts, dp2bp_sdispls, MPI_DOUBLE, comm
        );
    }
    if (unit_size == 16)
    {
        MPI_Alltoallv(
            packbuf, bp2dp_sendcnts, bp2dp_sdispls, MPI_C_DOUBLE_COMPLEX,
            dp_data, dp2bp_sendcnts, dp2bp_sdispls, MPI_C_DOUBLE_COMPLEX, comm
        );
    }
}

void DP2BP(
    const MPI_Comm comm, const int nproc, 
    const int nrow_bp, const int ncol_bp, const int *row_dp_displs,
    const int *bp2dp_sendcnts, const int *bp2dp_sdispls, 
    const int *dp2bp_sendcnts, const int *dp2bp_sdispls, 
    const int unit_size, const void *dp_data, void *packbuf, void *bp_data
)
{
    // Do MPI_Alltoallv
    if (unit_size == 8)
    {
        MPI_Alltoallv(
            dp_data, dp2bp_sendcnts, dp2bp_sdispls, MPI_DOUBLE,
            packbuf, bp2dp_sendcnts, bp2dp_sdispls, MPI_DOUBLE, comm
        );
    }
    if (unit_size == 16)
    {
        MPI_Alltoallv(
            dp_data, dp2bp_sendcnts, dp2bp_sdispls, MPI_C_DOUBLE_COMPLEX,
            packbuf, bp2dp_sendcnts, bp2dp_sdispls, MPI_C_DOUBLE_COMPLEX, comm
        );
    }
    
    // Unpack the received data to the band parallelization layout
    if (unit_size == 8)
    {
        double *bp_data_ = (double*) bp_data;
        double *packbuf_ = (double*) packbuf;
        for (int i = 0; i < nproc; i++)
        {
            int nrow_dp_i = row_dp_displs[i + 1] - row_dp_displs[i];
            double *src_ptr = packbuf_ + bp2dp_sdispls[i];
            double *dst_ptr = bp_data_ + row_dp_displs[i];
            copy_mat_blk(
                unit_size, src_ptr, nrow_dp_i, 
                nrow_dp_i, ncol_bp, dst_ptr, nrow_bp
            );
        }
    }
    if (unit_size == 16)
    {
        double _Complex *bp_data_ = (double _Complex*) bp_data;
        double _Complex *packbuf_ = (double _Complex*) packbuf;
        for (int i = 0; i < nproc; i++)
        {
            int nrow_dp_i = row_dp_displs[i + 1] - row_dp_displs[i];
            double _Complex *src_ptr = packbuf_ + bp2dp_sdispls[i];
            double _Complex *dst_ptr = bp_data_ + row_dp_displs[i];
            copy_mat_blk(
                unit_size, src_ptr, nrow_dp_i, 
                nrow_dp_i, ncol_bp, dst_ptr, nrow_bp
            );
        }
    }
}

#endif // "ifdef USE_DP_SUBEIG"


void Set_D2Dext_Target(D2Dext_OBJ *d2dext_sender, D2Dext_OBJ *d2dext_recvr, 
    int DMnx, int DMny, int DMnz, int xext, int yext, int zext, 
    int gridsizes[3], int dims[3], MPI_Comm cart, MPI_Comm *comm_d2dext)
{
    if (cart == MPI_COMM_NULL) {
        *comm_d2dext = MPI_COMM_NULL;
        return;
    }
    int rank, cart_coords[3];
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 3, cart_coords);

    // Estimation of maximum layers communication for SQ comm
    int max_layer = 1;
    max_layer *= (2 * (xext / (gridsizes[0] / dims[0]) + 1) + 1);
    max_layer *= (2 * (yext / (gridsizes[1] / dims[1]) + 1) + 1);
    max_layer *= (2 * (zext / (gridsizes[2] / dims[2]) + 1) + 1);

    int *sneighs = (int*) calloc(max_layer, sizeof(int));
    int *rneighs = (int*) calloc(max_layer, sizeof(int));
    int *scounts = (int*) calloc(max_layer, sizeof(int));
    int *rcounts = (int*) calloc(max_layer, sizeof(int));

    // count sending 
    int count = 0;
    for (int i = 0; i < 3; i++) {                                                    // loop over x, y, z axis
        int sign = -1;                                                               // sign indicates direction. e.g. left is -1. right is +1         
        for (int j = 0; j < 2; j++) {                                                    // loop over 2 directions
            int neigh_coords[3] = {cart_coords[0], cart_coords[1], cart_coords[2]};
            d2dext_sender->layers[i*2+j] = 0;
            
            int ext  = (i == 0 ? xext : (i == 1 ? yext : zext));
            int N_dir  = (i == 0 ? DMnx : (i == 1 ? DMny : DMnz));
            
            while (ext > 0) {
                neigh_coords[i] = (neigh_coords[i] + sign + dims[i]) % dims[i];     // shift coordinates 
                int neigh_size = block_decompose(gridsizes[i], dims[i], neigh_coords[i]);
                d2dext_sender->layers[i*2+j]++;                                 // count the layer
                scounts[count] = min(ext, N_dir);               // sending counts
                ext -= neigh_size;                              // remaining counts
                count++;
            }
            sign *= (-1);
        }
    }

    // from left to right, bottom to top, different order from above
    count = 0;
    for (int k = -d2dext_sender->layers[4]; k <= d2dext_sender->layers[5]; k++) {
        for (int j = -d2dext_sender->layers[2]; j <= d2dext_sender->layers[3]; j++) {
            for (int i = -d2dext_sender->layers[0]; i <= d2dext_sender->layers[1]; i++) {
                int neigh_coords[3] = {cart_coords[0], cart_coords[1], cart_coords[2]};
                if(i || j || k) {
                    neigh_coords[0] = (neigh_coords[0] + i + dims[0]) % dims[0];        // shift coordinates 
                    neigh_coords[1] = (neigh_coords[1] + j + dims[1]) % dims[1];        // shift coordinates 
                    neigh_coords[2] = (neigh_coords[2] + k + dims[2]) % dims[2];        // shift coordinates 
                    MPI_Cart_rank(cart, neigh_coords, sneighs + count);   // find neighbor's rank   
                    count ++;
                }
            }
        }
    }

    int destinations = (d2dext_sender->layers[0] + d2dext_sender->layers[1] + 1) 
                     * (d2dext_sender->layers[2] + d2dext_sender->layers[3] + 1) 
                     * (d2dext_sender->layers[4] + d2dext_sender->layers[5] + 1) - 1;

    // Then receiving elements from others.
    count = 0;
    for (int i = 0; i < 3; i++) {
        int sign = -1;        

        for (int j = 0; j < 2; j++) {
            int neigh_coords[3] = {cart_coords[0], cart_coords[1], cart_coords[2]};
            d2dext_recvr->layers[i*2+j] = 0;

            int ext  = (i == 0 ? xext : (i == 1 ? yext : zext));
            while (ext > 0) {
                neigh_coords[i] = (neigh_coords[i] + sign + dims[i]) % dims[i];
                int neigh_size = block_decompose(gridsizes[i], dims[i], neigh_coords[i]);
                d2dext_recvr->layers[i*2+j]++;                              // count the layer
                rcounts[count] = min(ext, neigh_size);         // sending counts
                ext -= rcounts[count];                         // remaining counts
                count++;
            }            
            sign *= (-1);
        }
    }

    count = 0;
    for (int k = d2dext_recvr->layers[5]; k >= -d2dext_recvr->layers[4]; k--) {
        for (int j = d2dext_recvr->layers[3]; j >= -d2dext_recvr->layers[2]; j--) {
            for (int i = d2dext_recvr->layers[1]; i >= -d2dext_recvr->layers[0]; i--) {
                int neigh_coords[3] = {cart_coords[0], cart_coords[1], cart_coords[2]};
                if(i || j || k) {
                    neigh_coords[0] = (neigh_coords[0] + i + dims[0]) % dims[0];     // shift coordinates 
                    neigh_coords[1] = (neigh_coords[1] + j + dims[1]) % dims[1];     // shift coordinates 
                    neigh_coords[2] = (neigh_coords[2] + k + dims[2]) % dims[2];     // shift coordinates 
                    MPI_Cart_rank(cart, neigh_coords, rneighs + count);       // find neighbor's rank                                   
                    count ++;
                }
            }
        }
    }

    int sources = (d2dext_recvr->layers[0] + d2dext_recvr->layers[1] + 1) 
                * (d2dext_recvr->layers[2] + d2dext_recvr->layers[3] + 1) 
                * (d2dext_recvr->layers[4] + d2dext_recvr->layers[5] + 1) - 1;
    
    MPI_Dist_graph_create_adjacent(cart, sources, rneighs, (int *)MPI_UNWEIGHTED, 
        destinations, sneighs, (int *)MPI_UNWEIGHTED, MPI_INFO_NULL, 0, comm_d2dext); 

    d2dext_sender->counts = (int *) calloc(destinations, sizeof(int));
    d2dext_sender->displs = (int *) calloc(destinations, sizeof(int));
    assert(d2dext_sender->counts != NULL && d2dext_sender->displs != NULL);
    d2dext_recvr->counts = (int *) calloc(sources, sizeof(int));
    d2dext_recvr->displs = (int *) calloc(sources, sizeof(int));
    assert(d2dext_recvr->counts != NULL && d2dext_recvr->displs != NULL);

    d2dext_sender->x_counts = (int *) calloc(d2dext_sender->layers[0] + d2dext_sender->layers[1] + 1, sizeof(int));
    d2dext_sender->y_counts = (int *) calloc(d2dext_sender->layers[2] + d2dext_sender->layers[3] + 1, sizeof(int));
    d2dext_sender->z_counts = (int *) calloc(d2dext_sender->layers[4] + d2dext_sender->layers[5] + 1, sizeof(int));
    assert(d2dext_sender->x_counts != NULL && d2dext_sender->y_counts != NULL && d2dext_sender->z_counts != NULL);

    d2dext_recvr->x_counts = (int *) calloc(d2dext_recvr->layers[0] + d2dext_recvr->layers[1] + 1, sizeof(int));
    d2dext_recvr->y_counts = (int *) calloc(d2dext_recvr->layers[2] + d2dext_recvr->layers[3] + 1, sizeof(int));
    d2dext_recvr->z_counts = (int *) calloc(d2dext_recvr->layers[4] + d2dext_recvr->layers[5] + 1, sizeof(int));
    assert(d2dext_recvr->x_counts != NULL && d2dext_recvr->y_counts != NULL && d2dext_recvr->z_counts != NULL);

    // change the order of sending and receiving information into left to right
    reorder_counts(scounts, d2dext_sender->layers, DMnx, DMny, DMnz, d2dext_sender->x_counts, d2dext_sender->y_counts, d2dext_sender->z_counts);
    reorder_counts(rcounts, d2dext_recvr->layers, DMnx, DMny, DMnz, d2dext_recvr->x_counts, d2dext_recvr->y_counts, d2dext_recvr->z_counts);

    free(sneighs);
    free(rneighs);
    free(scounts);
    free(rcounts);

    d2dext_sender->n = 0;
    d2dext_recvr->n = 0;

    count = 0;
    for (int k = 0; k < d2dext_sender->layers[4] + d2dext_sender->layers[5] + 1; k++) {
        for (int j = 0; j < d2dext_sender->layers[2] + d2dext_sender->layers[3] + 1; j++) {
            for (int i = 0; i < d2dext_sender->layers[0] + d2dext_sender->layers[1] + 1; i++) {
                // skip center (self)
                if (i != d2dext_sender->layers[0] || j != d2dext_sender->layers[2] || k != d2dext_sender->layers[4]) {
                    d2dext_sender->counts[count] = d2dext_sender->x_counts[i] * d2dext_sender->y_counts[j] * d2dext_sender->z_counts[k];

                    if (count > 0)
                        d2dext_sender->displs[count] = d2dext_sender->displs[count - 1] + d2dext_sender->counts[count - 1];

                    d2dext_sender->n += d2dext_sender->counts[count];
                    count ++;
                }
            }
        }
    }

    count = 0;
    for (int k = d2dext_recvr->layers[4] + d2dext_recvr->layers[5]; k >= 0; k--) {
        for (int j = d2dext_recvr->layers[2] + d2dext_recvr->layers[3]; j >= 0; j--) {
            for (int i = d2dext_recvr->layers[0] + d2dext_recvr->layers[1]; i >= 0; i--) {
                // skip center (self)
                if (i != d2dext_recvr->layers[0] || j != d2dext_recvr->layers[2] || k != d2dext_recvr->layers[4]) {
                    d2dext_recvr->counts[count] = d2dext_recvr->x_counts[i] * d2dext_recvr->y_counts[j] * d2dext_recvr->z_counts[k];

                    if (count > 0)
                        d2dext_recvr->displs[count] = d2dext_recvr->displs[count - 1] + d2dext_recvr->counts[count - 1];

                    d2dext_recvr->n += d2dext_recvr->counts[count];
                    count ++;
                }
            }
        }
    }
}


void reorder_counts(const int *counts, const int *layers, const int DMnx, const int DMny, const int DMnz, 
                    int *x_counts, int *y_counts, int *z_counts)
{
    const int *counts_ = counts;
    // x -1 dir    
    for (int i = 0; i < layers[0]; i++) {
        x_counts[i] = counts_[layers[0] - 1 - i];        
    }
    x_counts[layers[0]] = DMnx; // center
    // x +1 dir
    for (int i = 0; i < layers[1]; i++) {
        x_counts[layers[0] + 1 + i] = counts_[layers[0] + i];
    }

    counts_ += layers[0] + layers[1];
    // y -1 dir
    for (int i = 0; i < layers[2]; i++) {
        y_counts[i] = counts_[layers[2] - 1 - i];
    }
    y_counts[layers[2]] = DMny; // center
    // y +1 dir
    for (int i = 0; i < layers[3]; i++) {
        y_counts[layers[2] + 1 + i] = counts_[layers[2] + i];
    }

    counts_ += layers[2] + layers[3];
    // z -1 dir
    for (int i = 0; i < layers[4]; i++) {
        z_counts[i] = counts_[layers[4] - 1 - i];
    }
    z_counts[layers[4]] = DMnz; // center
    // z +1 dir
    for (int i = 0; i < layers[5]; i++) {
        z_counts[layers[4] + 1 + i] = counts_[layers[4] + i];
    }
}


void free_D2Dext_Target(D2Dext_OBJ *d2dext, MPI_Comm comm_d2dext)
{
    if (comm_d2dext == MPI_COMM_NULL) return;
    free(d2dext->counts);
    free(d2dext->displs);
    free(d2dext->x_counts);
    free(d2dext->y_counts);
    free(d2dext->z_counts);    
}

void D2Dext(D2Dext_OBJ *d2dext_sender, D2Dext_OBJ *d2dext_recvr, int DMnx, int DMny, int DMnz, 
    int xext, int yext, int zext, void *sdata, void *rdata, MPI_Comm comm_d2dext, int unit_size)
{
#define sdata_(i,j,k) sdata_[(i)+(j)*DMnx+(k)*DMnx*DMny]
#define rdata_(i,j,k) rdata_[(i)+(j)*DMnx_ext+(k)*DMnx_ext*DMny_ext]

    if (comm_d2dext == MPI_COMM_NULL) return;
    void *x_in  = calloc(d2dext_recvr->n,  unit_size);
    void *x_out = calloc(d2dext_sender->n, unit_size);
    assert(x_in != NULL && x_out != NULL);

    // assemble x_out
    int count = 0;
    for (int k = 0; k < d2dext_sender->layers[4] + d2dext_sender->layers[5] + 1; k++) {
        for (int j = 0; j < d2dext_sender->layers[2] + d2dext_sender->layers[3] + 1; j++) {
            for (int i = 0; i < d2dext_sender->layers[0] + d2dext_sender->layers[1] + 1; i++) {

                if (i != d2dext_recvr->layers[0] || j != d2dext_recvr->layers[2] || k != d2dext_recvr->layers[4]) {
                    int start[3] = {0, 0 , 0};                                              // range of elements to be sent out
                    int end[3]   = {DMnx,DMny, DMnz};

                    if (i < d2dext_sender->layers[0])
                        end[0] = d2dext_sender->x_counts[i];
                    else
                        start[0] = DMnx - d2dext_sender->x_counts[i];

                    if (j < d2dext_sender->layers[2])
                        end[1] = d2dext_sender->y_counts[j];
                    else
                        start[1] = DMny - d2dext_sender->y_counts[j];

                    if (k < d2dext_sender->layers[4])
                        end[2] = d2dext_sender->z_counts[k];
                    else
                        start[2] = DMnz - d2dext_sender->z_counts[k];

                    for (int kk = start[2]; kk < end[2]; kk++) {
                        for (int jj = start[1]; jj < end[1]; jj++) {
                            for (int ii = start[0]; ii < end[0]; ii++) {
                                if (unit_size == 8) {
                                    double *sdata_ = (double *) sdata;
                                    *((double *)x_out+count++) = sdata_(ii, jj, kk);
                                } else {
                                    double _Complex *sdata_ = (double _Complex *) sdata;
                                    *((double _Complex *)x_out+count++) = sdata_(ii, jj, kk);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // communicate and assemble x_in
    int DMnx_ext = DMnx + 2*xext;
    int DMny_ext = DMny + 2*yext;    

    MPI_Request request;
    if (unit_size == 8) {
        MPI_Ineighbor_alltoallv(x_out, d2dext_sender->counts, d2dext_sender->displs, MPI_DOUBLE, 
                                x_in, d2dext_recvr->counts, d2dext_recvr->displs, MPI_DOUBLE, comm_d2dext, &request); 
        // copy local part 
        restrict_to_subgrid(sdata, rdata,
                    DMnx_ext, DMnx, DMnx_ext*DMny_ext, DMnx*DMny, 
                    xext, xext+DMnx-1, yext, yext+DMny-1, zext, zext+DMnz-1, 
                    0, 0, 0, sizeof(double));
    } else {
        MPI_Ineighbor_alltoallv(x_out, d2dext_sender->counts, d2dext_sender->displs, MPI_DOUBLE_COMPLEX, 
                                x_in, d2dext_recvr->counts, d2dext_recvr->displs, MPI_DOUBLE_COMPLEX, comm_d2dext, &request);         
        // copy local part 
        restrict_to_subgrid(sdata, rdata,
                    DMnx_ext, DMnx, DMnx_ext*DMny_ext, DMnx*DMny, 
                    xext, xext+DMnx-1, yext, yext+DMny-1, zext, zext+DMnz-1, 
                    0, 0, 0, sizeof(double _Complex));
    }

    MPI_Wait(&request, MPI_STATUS_IGNORE);

    int start[3] = {0,0,0}, end[3];

    count = 0;
    end[2] = 2 * zext + DMnz;
    for (int k = d2dext_recvr->layers[4] + d2dext_recvr->layers[5]; k >= 0; k--) {
        end[1] = 2 * yext + DMny;
        for (int j = d2dext_recvr->layers[2] + d2dext_recvr->layers[3]; j >= 0; j--) {
            end[0] = 2 * xext + DMnx;
            for (int i = d2dext_recvr->layers[0] + d2dext_recvr->layers[1]; i >= 0; i--) {

                start[0] = end[0] - d2dext_recvr->x_counts[i];
                start[1] = end[1] - d2dext_recvr->y_counts[j];
                start[2] = end[2] - d2dext_recvr->z_counts[k];

                if (i != d2dext_recvr->layers[0] || j != d2dext_recvr->layers[2] || k != d2dext_recvr->layers[4]) {
                    for (int kk = start[2]; kk < end[2]; kk++) {
                        for (int jj = start[1]; jj < end[1]; jj++) {
                            for (int ii = start[0]; ii < end[0]; ii++) {
                                // Veff_PR(ii,jj,kk) = x_in[count++];
                                if (unit_size == 8) {
                                    double *rdata_ = (double *) rdata;
                                    rdata_(ii,jj,kk) = *((double *)x_in+count++);
                                } else {
                                    double _Complex *rdata_ = (double _Complex *) rdata;
                                    rdata_(ii,jj,kk) = *((double _Complex *)x_in+count++);
                                }
                            }
                        }
                    }
                }
                end[0] = start[0];
            }
            end[1] = start[1];
        }
        end[2] = start[2];
    }

    free(x_in);
    free(x_out);
#undef sdata_
#undef rdata_
}


int MPI_Allreduce_overload(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{    
    size_t unit_size;    
    if (datatype == MPI_DOUBLE) {
        unit_size = sizeof(double);
    } else if (datatype == MPI_DOUBLE_COMPLEX) {
        unit_size = sizeof(double _Complex);
    } else if (datatype == MPI_INT) {
        unit_size = sizeof(int);
    } else {
        unit_size = 1;
        printf("Please provide the type you need\n");
        exit(EXIT_FAILURE);
    }

    size_t size = count * unit_size;
    size_t base = 100000000; // 100MB
    int reps = max((size-1) / base + 1, 1);
    if (reps == 1) {
        return MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }
    size_t *counts = (size_t *) malloc(sizeof(size_t) * reps);
    size_t *displs = (size_t *) malloc(sizeof(size_t) * reps);    
    int batch = count / reps;

    displs[0] = 0;
    for (int i = 0; i < reps; i++) {
        counts[i] = batch + (i < (count % batch) ? 1 : 0);
        if (i != reps -1) displs[i+1] = displs[i] + counts[i];        
    }

    for (int rep = 0; rep < reps; rep++) {
        const void *sendbuf_ = (MPI_IN_PLACE == sendbuf) ? MPI_IN_PLACE : (sendbuf + displs[rep] * unit_size);
        void *recvbuf_ = recvbuf + displs[rep] * unit_size;        
        int res = MPI_Allreduce(sendbuf_, recvbuf_, counts[rep], datatype, op, comm);
        if (res != MPI_SUCCESS) {
            free(counts);
            free(displs);
            return res;
        }
    }
    free(counts);
    free(displs);
    return MPI_SUCCESS;
}