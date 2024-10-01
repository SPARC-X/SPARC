/***
 * @file    sqParallelization.c
 * @brief   This file contains the functions for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <limits.h>

#include "isddft.h"
#include "sqParallelization.h"
#include "occupation.h"
#include "sqNlocVecRoutines.h"
#include "parallelization.h"
#include "tools.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)


/**
 * @brief   Set up sub-communicators for SQ method.
 * 
 * Note:    Part of the codes are copied and modified from the code in parallelization.c file
 */
void Setup_Comms_SQ(SPARC_OBJ *pSPARC) {
    pSPARC->pSQ = (SQ_OBJ *) malloc(sizeof(SQ_OBJ));
    SQ_OBJ* pSQ  = pSPARC->pSQ;
    int i, dims[3] = {0, 0, 0}, periods[3], ierr;
    int nproc, rank;
    int npNd, gridsizes[3], minsize, coord_dmcomm[3], rank_dmcomm;
    int DMnx, DMny, DMnz, DMnd;
#ifdef DEBUG
    double t1, t2;
#endif
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Set up SQ communicators.\n"); 
#endif

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
    int size_spincomm = nproc / pSPARC->npspin;
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
    } else {
        pSPARC->spin_start_indx = pSPARC->spincomm_index * pSPARC->Nspin_spincomm;        
    }
    pSPARC->spin_end_indx = pSPARC->spin_start_indx + pSPARC->Nspin_spincomm - 1;    

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // split the MPI_COMM_WORLD into spincomms using color = spincomm_index
    int color = (pSPARC->spincomm_index >= 0) ? pSPARC->spincomm_index : INT_MAX; 
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &pSPARC->spincomm);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up spincomm took %.3f ms\n",(t2-t1)*1000);
#endif
    int rank_spincomm;
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
    //              set up SQcomm_topo                //
    //------------------------------------------------//
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    minsize = 1;

    // Use all processors to build SQcomm_topo
    npNd = pSPARC->npNdx_SQ * pSPARC->npNdy_SQ * pSPARC->npNdz_SQ; // try to use all processors
    if (npNd == 0)  {
        // when user does not provide domain decomposition parameters
        npNd = size_spincomm;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        if (ierr == 1 && rank == 0) {
            printf("WARNING: error occured when calculating best domain distribution\n");
        }
        pSPARC->npNdx_SQ = dims[0];
        pSPARC->npNdy_SQ = dims[1];
        pSPARC->npNdz_SQ = dims[2];
		pSPARC->useDefaultParalFlag = 1;	
    } else if (npNd < 0 || npNd > nproc) {
        // when domain decomposition parameters are not reasonable
        npNd = size_spincomm;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        pSPARC->npNdx_SQ = dims[0];
        pSPARC->npNdy_SQ = dims[1];
        pSPARC->npNdz_SQ = dims[2];
		if (!rank) printf("WARNING: Default parallelization used because parallelization parameters are unreasonable.\n");
			pSPARC->useDefaultParalFlag = 1;
    } else {
        dims[0] = pSPARC->npNdx_SQ;
        dims[1] = pSPARC->npNdy_SQ;
        dims[2] = pSPARC->npNdz_SQ;
		if (!rank) printf("WARNING: Default parallelization not used. This could result in degradation of performance.\n");
            pSPARC->useDefaultParalFlag = 0;
    }

    // SQ only works in Periodic B.C.
    periods[0] = 1;
    periods[1] = 1;
    periods[2] = 1;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    // Processes in bandcomm with rank >= dims[0]*dims[1]*…*dims[d-1] will return MPI_COMM_NULL
    if (pSPARC->spincomm_index >= 0) {
        MPI_Cart_create(pSPARC->spincomm, 3, dims, periods, 1, &pSQ->dmcomm_SQ); // 1 is to reorder rank
    } else {
        pSQ->dmcomm_SQ = MPI_COMM_NULL;
    }    

#ifdef DEBUG
    if (rank == 0) {
        printf("========================================================================\n"
                   "SQ domain decomposition:"
                   "np total = %d, npspin %d, {Nx, Ny, Nz} = {%d, %d, %d}\n"
                   "nproc used = %d = {%d, %d, %d}, nodes/proc = {%.2f, %.2f, %.2f}\n\n",
                   nproc,pSPARC->npspin,pSPARC->Nx,pSPARC->Ny,pSPARC->Nz,dims[0]*dims[1]*dims[2],dims[0],dims[1],dims[2],pSPARC->Nx/(double)dims[0],pSPARC->Ny/(double)dims[1],pSPARC->Nz/(double)dims[2]);
    }
#endif

    // find the vertices of the domain in each processor
    // pSQ->DMVertices_SQ[6] = [xs,xe,ys,ye,zs,ze]
    // int Nx_dist, Ny_dist, Nz_dist;
    if (pSQ->dmcomm_SQ != MPI_COMM_NULL) {
        MPI_Comm_rank(pSQ->dmcomm_SQ, &rank_dmcomm);
        MPI_Cart_coords(pSQ->dmcomm_SQ, rank_dmcomm, 3, coord_dmcomm);

        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;

        // find size of distributed domain
        pSQ->DMnx_SQ = block_decompose(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSQ->DMny_SQ = block_decompose(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSQ->DMnz_SQ = block_decompose(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSQ->DMnd_SQ = pSQ->DMnx_SQ * pSQ->DMny_SQ * pSQ->DMnz_SQ;

        // find corners of the distributed domain
        pSQ->DMVertices_SQ[0] = block_decompose_nstart(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSQ->DMVertices_SQ[1] = pSQ->DMVertices_SQ[0] + pSQ->DMnx_SQ - 1;
        pSQ->DMVertices_SQ[2] = block_decompose_nstart(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSQ->DMVertices_SQ[3] = pSQ->DMVertices_SQ[2] + pSQ->DMny_SQ - 1;
        pSQ->DMVertices_SQ[4] = block_decompose_nstart(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSQ->DMVertices_SQ[5] = pSQ->DMVertices_SQ[4] + pSQ->DMnz_SQ - 1;
    } else {
        rank_dmcomm = -1;
        coord_dmcomm[0] = -1; coord_dmcomm[1] = -1; coord_dmcomm[2] = -1;
        pSQ->DMnx_SQ = 0;
        pSQ->DMny_SQ = 0;
        pSQ->DMnz_SQ = 0;
        pSQ->DMnd_SQ = 0;
        pSQ->DMVertices_SQ[0] = 0;
        pSQ->DMVertices_SQ[1] = 0;
        pSQ->DMVertices_SQ[2] = 0;
        pSQ->DMVertices_SQ[3] = 0;
        pSQ->DMVertices_SQ[4] = 0;
        pSQ->DMVertices_SQ[5] = 0;
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up dmcomm_SQ took %.3f ms\n",(t2-t1)*1000);
#endif

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
    // Only orthogonal systems is allowed in SQ method

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up dmcomm_phi took %.3f ms\n",(t2-t1)*1000);
#endif

    /* allocate memory for storing atomic forces*/
    pSPARC->forces = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
    assert(pSPARC->forces != NULL);

    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        /* allocate memory for electrostatics calculation */
        DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
        DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
        DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
        DMnd = DMnx * DMny * DMnz;
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
    }

    // parallelization summary
    #ifdef DEBUG
    if (rank == 0) {
        printf("\n");
        printf("-----------------------------------------------\n");
        printf("Parallelization summary\n");
        printf("Total number of processors: %d\n", nproc);
        printf("-----------------------------------------------\n");
        printf("== SQ domain ==\n");
        printf("npspin  : %d\n", pSPARC->npspin);
        printf("Total number of processors used for SQ domain: %d\n", pSPARC->npNdx_SQ*pSPARC->npNdy_SQ*pSPARC->npNdz_SQ);
        printf("Embeded Cartesian topology dims: (%d,%d,%d)\n", pSPARC->npNdx_SQ, pSPARC->npNdy_SQ, pSPARC->npNdz_SQ);
        printf("# of FD-grid points per processor: %d = (%d,%d,%d)\n", pSQ->DMnd_SQ,pSQ->DMnx_SQ,pSQ->DMny_SQ,pSQ->DMnz_SQ);
        printf("-----------------------------------------------\n");
        printf("== Phi domain ==\n");
        printf("Total number of processors used for Phi domain: %d\n", pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi);
        printf("Embeded Cartesian topology dims: (%d,%d,%d)\n", pSPARC->npNdx_phi,pSPARC->npNdy_phi, pSPARC->npNdz_phi);
        printf("# of FD-grid points per processor: %d = (%d,%d,%d)\n", pSPARC->Nd_d,pSPARC->Nx_d,pSPARC->Ny_d,pSPARC->Nz_d);
        printf("-----------------------------------------------\n");
    }
    #endif

    // Create D2D transfer object between phi and SQ domian.
    create_D2D_sq2phi(pSPARC);
}

/**
 * @brief   Create a D2D object for communication from SQ domain to phi domain 
 */
void create_D2D_sq2phi(SPARC_OBJ *pSPARC) {
    SQ_OBJ* pSQ  = pSPARC->pSQ;
    int gridsizes[3], sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    
    // sent by dmcomm_SQ
    sdims[0] = pSPARC->npNdx_SQ;
    sdims[1] = pSPARC->npNdy_SQ;
    sdims[2] = pSPARC->npNdz_SQ;
    // received by dmcomm_phi
    rdims[0] = pSPARC->npNdx_phi;
    rdims[1] = pSPARC->npNdy_phi;
    rdims[2] = pSPARC->npNdz_phi;
    
    // printf("pSQ->dmcomm_SQ null? %d, pSPARC->dmcomm_phi null? %d\n", pSQ->dmcomm_SQ==MPI_COMM_NULL, pSPARC->dmcomm_phi==MPI_COMM_NULL);
    // fflush(stdout);

    Set_D2D_Target(&pSPARC->d2d_s2p_sq, &pSPARC->d2d_s2p_phi, gridsizes, pSQ->DMVertices_SQ, pSPARC->DMVertices,
                    (pSPARC->spincomm_index == 0 ? pSQ->dmcomm_SQ : MPI_COMM_NULL), sdims, pSPARC->dmcomm_phi, rdims, MPI_COMM_WORLD);
}

/**
 * @brief   Transferring electron density from SQ domain to phi domain using D2D
 */
void TransferDensity_sq2phi(SPARC_OBJ *pSPARC, double *rho_send, double *rho_recv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    SQ_OBJ *pSQ  = pSPARC->pSQ;
    
    int sdims[3], rdims[3], gridsizes[3];
    sdims[0] = pSPARC->npNdx_SQ; sdims[1] = pSPARC->npNdy_SQ; sdims[2] = pSPARC->npNdz_SQ;
    rdims[0] = pSPARC->npNdx_phi; rdims[1] = pSPARC->npNdy_phi; rdims[2] = pSPARC->npNdz_phi;
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    D2D(&pSPARC->d2d_s2p_sq, &pSPARC->d2d_s2p_phi, gridsizes, pSQ->DMVertices_SQ, rho_send, 
        pSPARC->DMVertices, rho_recv, (pSPARC->spincomm_index == 0 ? pSQ->dmcomm_SQ : MPI_COMM_NULL), 
        sdims, pSPARC->dmcomm_phi, rdims, MPI_COMM_WORLD, sizeof(double));
        
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, Transfer density from SQ domain to dmcomm_phi using D2D took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}

/**
 * @brief   Transferring Veff from phi domain to SQ domain using D2D
 */
void TransferVeff_phi2sq(SPARC_OBJ *pSPARC, double *Veff_send, double *Veff_recv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    SQ_OBJ *pSQ  = pSPARC->pSQ;

    int sdims[3], rdims[3], gridsizes[3];
    sdims[0] = pSPARC->npNdx_phi; sdims[1] = pSPARC->npNdy_phi; sdims[2] = pSPARC->npNdz_phi;
    rdims[0] = pSPARC->npNdx_SQ; rdims[1] = pSPARC->npNdy_SQ; rdims[2] = pSPARC->npNdz_SQ;
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    D2D(&pSPARC->d2d_s2p_phi, &pSPARC->d2d_s2p_sq, gridsizes, pSPARC->DMVertices, Veff_send, 
        pSQ->DMVertices_SQ, Veff_recv, pSPARC->dmcomm_phi, sdims, (pSPARC->spincomm_index == 0 ? pSQ->dmcomm_SQ : MPI_COMM_NULL),
        rdims, MPI_COMM_WORLD, sizeof(double));
        
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, Transfer Veff from dmcomm_phi to SQ domain using D2D took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    if (pSPARC->npspin > 1 && pSPARC->spincomm_index >= 0) {
        MPI_Bcast(Veff_recv, pSQ->DMnd_SQ, MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, Transfer Veff btw/ spincomms took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}

void TransferVeff_sq2sqext(SPARC_OBJ *pSPARC, double *Veff_send, double *Veff_recv) 
{    
    SQ_OBJ  *pSQ = pSPARC->pSQ;
    int DMnx = pSQ->DMnx_SQ;
    int DMny = pSQ->DMny_SQ;
    int DMnz = pSQ->DMnz_SQ;    
    int DMnd = pSQ->DMnd_SQ;
    int *nloc = pSQ->nloc; 

    D2Dext(pSQ->d2dext_dmcomm_sq, pSQ->d2dext_dmcomm_sq_ext, DMnx, DMny, DMnz, 
        nloc[0], nloc[1], nloc[2], Veff_send, Veff_recv, pSQ->dmcomm_d2dext_sq, sizeof(double));
}
