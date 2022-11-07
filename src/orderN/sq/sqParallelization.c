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

#include "isddft.h"
#include "sqParallelization.h"
#include "sqtool.h"
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
        npNd = nproc;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        if (ierr == 1 && rank == 0) {
            printf("WARNING: error occured when calculating best domain distribution\n");
        }
        pSPARC->npNdx_SQ = dims[0];
        pSPARC->npNdy_SQ = dims[1];
        pSPARC->npNdz_SQ = dims[2];
    } else if (npNd < 0 || npNd > nproc) {
        // when domain decomposition parameters are not reasonable
        npNd = nproc;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        pSPARC->npNdx_SQ = dims[0];
        pSPARC->npNdy_SQ = dims[1];
        pSPARC->npNdz_SQ = dims[2];
    } else {
        dims[0] = pSPARC->npNdx_SQ;
        dims[1] = pSPARC->npNdy_SQ;
        dims[2] = pSPARC->npNdz_SQ;
    }

    // SQ only works in Periodic B.C.
    periods[0] = 1;
    periods[1] = 1;
    periods[2] = 1;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    // Processes in bandcomm with rank >= dims[0]*dims[1]*…*dims[d-1] will return MPI_COMM_NULL
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &pSQ->dmcomm_SQ); // 1 is to reorder rank

#ifdef DEBUG
    if (rank == 0) {
        printf("========================================================================\n"
                   "SQ domain decomposition:"
                   "np total = %d, {Nx, Ny, Nz} = {%d, %d, %d}\n"
                   "nproc used = %d = {%d, %d, %d}, nodes/proc = {%.2f, %.2f, %.2f}\n\n",
                   nproc,pSPARC->Nx,pSPARC->Ny,pSPARC->Nz,dims[0]*dims[1]*dims[2],dims[0],dims[1],dims[2],pSPARC->Nx/(double)dims[0],pSPARC->Ny/(double)dims[1],pSPARC->Nz/(double)dims[2]);
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
        pSQ->Nx_d_SQ = block_decompose(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSQ->Ny_d_SQ = block_decompose(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSQ->Nz_d_SQ = block_decompose(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSQ->Nd_d_SQ = pSQ->Nx_d_SQ * pSQ->Ny_d_SQ * pSQ->Nz_d_SQ;

        // find corners of the distributed domain
        pSQ->DMVertices_SQ[0] = block_decompose_nstart(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSQ->DMVertices_SQ[1] = pSQ->DMVertices_SQ[0] + pSQ->Nx_d_SQ - 1;
        pSQ->DMVertices_SQ[2] = block_decompose_nstart(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSQ->DMVertices_SQ[3] = pSQ->DMVertices_SQ[2] + pSQ->Ny_d_SQ - 1;
        pSQ->DMVertices_SQ[4] = block_decompose_nstart(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSQ->DMVertices_SQ[5] = pSQ->DMVertices_SQ[4] + pSQ->Nz_d_SQ - 1;
    } else {
        rank_dmcomm = -1;
        coord_dmcomm[0] = -1; coord_dmcomm[1] = -1; coord_dmcomm[2] = -1;
        pSQ->Nx_d_SQ = 0;
        pSQ->Ny_d_SQ = 0;
        pSQ->Nz_d_SQ = 0;
        pSQ->Nd_d_SQ = 0;
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
        pSPARC->electronDens_at = (double *)malloc( DMnd * (2*pSPARC->Nspin-1) * sizeof(double) );
        pSPARC->electronDens_core = (double *)calloc( DMnd * (2*pSPARC->Nspin-1), sizeof(double) );
        pSPARC->psdChrgDens = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->psdChrgDens_ref = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->Vc = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->electronDens_core != NULL);
        assert(pSPARC->electronDens_at != NULL && pSPARC->psdChrgDens != NULL &&
               pSPARC->psdChrgDens_ref != NULL && pSPARC->Vc != NULL);
        // allocate memory for electron density
        pSPARC->electronDens = (double *)malloc( DMnd * (2*pSPARC->Nspin-1) * sizeof(double) );
        assert(pSPARC->electronDens != NULL);
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
        pSPARC->XCPotential = (double *)malloc( DMnd * pSPARC->Nspin * sizeof(double) );
        assert(pSPARC->XCPotential != NULL);

        // allocate memory for exchange-correlation energy density
        pSPARC->e_xc = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->e_xc != NULL);

        // if GGA then allocate for xc energy per particle for each grid point and der. wrt. grho
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
         || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0){
            pSPARC->Dxcdgrho = (double *)malloc( DMnd * (2*pSPARC->Nspin - 1) * sizeof(double) );
            assert(pSPARC->Dxcdgrho != NULL);
        }

        pSPARC->Veff_loc_dmcomm_phi = (double *)malloc(DMnd * pSPARC->Nspin * sizeof(double));
        pSPARC->mixing_hist_xk      = (double *)malloc(DMnd * pSPARC->Nspin * sizeof(double));
        pSPARC->mixing_hist_fk      = (double *)calloc(DMnd * pSPARC->Nspin , sizeof(double));
        pSPARC->mixing_hist_fkm1    = (double *)calloc(DMnd * pSPARC->Nspin , sizeof(double));
        pSPARC->mixing_hist_xkm1    = (double *)malloc(DMnd * pSPARC->Nspin * sizeof(double));
        pSPARC->mixing_hist_Xk      = (double *)malloc(DMnd * pSPARC->Nspin * pSPARC->MixingHistory * sizeof(double));
        pSPARC->mixing_hist_Fk      = (double *)malloc(DMnd * pSPARC->Nspin * pSPARC->MixingHistory * sizeof(double));
        assert(pSPARC->Veff_loc_dmcomm_phi != NULL && pSPARC->mixing_hist_xk   != NULL &&
               pSPARC->mixing_hist_fk      != NULL && pSPARC->mixing_hist_fkm1 != NULL &&
               pSPARC->mixing_hist_xkm1    != NULL && pSPARC->mixing_hist_Xk   != NULL &&
               pSPARC->mixing_hist_Fk      != NULL);

        if (pSPARC->MixingVariable == 1) { // for potential mixing, the history is stored already
            pSPARC->Veff_loc_dmcomm_phi_in = pSPARC->mixing_hist_xk;
        } else {                           // for denstiy mixing, need extra memory to store potential history
            pSPARC->Veff_loc_dmcomm_phi_in = (double *)malloc(DMnd * pSPARC->Nspin * sizeof(double));
            assert(pSPARC->Veff_loc_dmcomm_phi_in != NULL);
        }

        // The following rho_in and phi_in are only used for evaluating QE scf errors
        if (pSPARC->scf_err_type == 1) {
            pSPARC->rho_dmcomm_phi_in = (double *)malloc(DMnd * sizeof(double));
            assert(pSPARC->rho_dmcomm_phi_in != NULL);
            pSPARC->phi_dmcomm_phi_in = (double *)malloc(DMnd * sizeof(double));
            assert(pSPARC->phi_dmcomm_phi_in != NULL);
        }

        if (pSPARC->MixingPrecond != 0) {
            pSPARC->mixing_hist_Pfk = (double *)calloc(DMnd * pSPARC->Nspin, sizeof(double));
            assert(pSPARC->mixing_hist_Pfk != NULL);
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
        printf("Total number of processors used for SQ domain: %d\n", pSPARC->npNdx_SQ*pSPARC->npNdy_SQ*pSPARC->npNdz_SQ);
        printf("Embeded Cartesian topology dims: (%d,%d,%d)\n", pSPARC->npNdx_SQ, pSPARC->npNdy_SQ, pSPARC->npNdz_SQ);
        printf("# of FD-grid points per processor: %d = (%d,%d,%d)\n", pSQ->Nd_d_SQ,pSQ->Nx_d_SQ,pSQ->Ny_d_SQ,pSQ->Nz_d_SQ);
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
 * @brief   Create communication topology for SQ type communication
 * 
 * @param DMnxnynz  Number of FD nodes of current processor in 3 directions
 * @param np        Number of processors in 3 directions for current cartesian topology
 * @param cart      Cartesian topology, where the SQ communicator is built on
 * @param comm_sq   The SQ communicator 
 * 
 * Note:            SQ type communication is for each process getting all values within their 
 *                  P.R. domain (Process + Rcut). It is different from Laplacian type communication 
 *                  for orthogonal systems, but similar to Lap communication for unorthogonal systems. 
 */
void Comm_topologies_sq(SPARC_OBJ *pSPARC, int DMnxnynz[3], int np[3], MPI_Comm cart, MPI_Comm *comm_sq) {
#define send_layers(i,j) send_layers[(i)*2+(j)]
#define rec_layers(i,j)  rec_layers[(i)*2+(j)]

    int i, j, k, rank, count, sum, sign, neigh_size, large, small, reorder = 0;
    int neigh_coords[3], sources = 0, destinations = 0, max_layer;
    int *nloc, *coords, *rem;
    int *send_neighs, *rec_neighs, *send_counts, *rec_counts, *send_layers, *rec_layers;
    MPI_Comm_rank(cart, &rank); 
    //////////////////////////////////////////////////////////////////////

    SQ_OBJ* pSQ  = pSPARC->pSQ;
    SQIND *SqInd = pSQ->SqInd;
    
    // Estimation of maximum layers communication for SQ comm
    max_layer = 1;
    max_layer *= (2 * (pSQ->nloc[0] / (pSPARC->Nx / np[0]) + 1) + 1);
    max_layer *= (2 * (pSQ->nloc[1] / (pSPARC->Ny / np[1]) + 1) + 1);
    max_layer *= (2 * (pSQ->nloc[2] / (pSPARC->Nz / np[2]) + 1) + 1);

    SqInd->send_neighs = (int*) calloc(max_layer, sizeof(int));
    SqInd->rec_neighs  = (int*) calloc(max_layer, sizeof(int));
    SqInd->send_counts = (int*) calloc(max_layer, sizeof(int));
    SqInd->rec_counts  = (int*) calloc(max_layer, sizeof(int));

    nloc        = pSQ->nloc;
    coords      = pSQ->coords;
    rem         = pSQ->rem;
    send_neighs = SqInd->send_neighs;
    rec_neighs  = SqInd->rec_neighs;
    send_counts = SqInd->send_counts;
    rec_counts  = SqInd->rec_counts;
    send_layers = SqInd->send_layers;
    rec_layers  = SqInd->rec_layers;
    //////////////////////////////////////////////////////////////////////

    // sending 
    count = 0;
    for (i = 0; i < 3; i++) {                                                    // loop over x, y, z axis
        sign = -1;                                                               // sign indicates direction. e.g. left is -1. right is +1 
        Find_size_dir(rem[i], coords[i], DMnxnynz[i], &small, &large);

        for (j = 0; j < 2; j++) {                                                // loop over 2 directions
            Vec_copy(neigh_coords, coords, 3);
            sum  = nloc[i];
            send_layers(i, j) = 0;
            neigh_size = 0;

            while (sum > 0) {
                neigh_coords[i] = (neigh_coords[i] + sign + np[i]) % np[i];     // shift coordinates 
                send_layers(i,j)++;
                *(send_counts + count) = min(sum, DMnxnynz[i]);
                if (neigh_coords[i] < rem[i])                                   // find neighbor's size in this direction
                    neigh_size = large;
                else
                    neigh_size = small;

                sum -= neigh_size;
                count++;
            }
            sign *= (-1);
        }
    }

    count = 0;
    for (k = -send_layers[4]; k <= send_layers[5]; k++) {
        for (j = -send_layers[2]; j <= send_layers[3]; j++) {
            for (i = -send_layers[0]; i <= send_layers[1]; i++) {
                Vec_copy(neigh_coords, coords, 3);
                if(i || j || k) {
                    neigh_coords[0] = (neigh_coords[0] + i + np[0]) % np[0];     // shift coordinates 
                    neigh_coords[1] = (neigh_coords[1] + j + np[1]) % np[1];     // shift coordinates 
                    neigh_coords[2] = (neigh_coords[2] + k + np[2]) % np[2];     // shift coordinates 
                    MPI_Cart_rank(cart, neigh_coords, send_neighs + count);          // find neighbor's rank   
                    count ++;                            
                }
            }
        }
    }

    destinations = (send_layers[0] + send_layers[1] + 1) * (send_layers[2] + send_layers[3] + 1) * (send_layers[4] + send_layers[5] + 1) - 1;

    // Then receiving elements from others.
    count = 0;
    for (i = 0; i < 3; i++) {
        sign = -1;
        Find_size_dir(rem[i], coords[i], DMnxnynz[i], &small, &large);

        for (j = 0; j < 2; j++) {
            Vec_copy(neigh_coords, coords, 3);
            sum  = nloc[i];
            rec_layers(i, j) = 0;

            while (sum > 0) {
                neigh_coords[i] = (neigh_coords[i] + sign + np[i]) % np[i];

                if (neigh_coords[i] < rem[i]) {
                    neigh_size = large;
                } else {
                    neigh_size = small;
                }
                rec_layers(i,j)++;
                sum -= neigh_size;
                *(rec_counts + count) = neigh_size;
                count++;
            }
            *(rec_counts + count - 1) = neigh_size + sum;
            sign *= (-1);
        }
    }

    count = 0;
    for (k = rec_layers[5]; k >= -rec_layers[4]; k--) {
        for (j = rec_layers[3]; j >= -rec_layers[2]; j--) {
            for (i = rec_layers[1]; i >= -rec_layers[0]; i--) {
                Vec_copy(neigh_coords, coords, 3);
                if(i || j || k) {
                    neigh_coords[0] = (neigh_coords[0] + i + np[0]) % np[0];     // shift coordinates 
                    neigh_coords[1] = (neigh_coords[1] + j + np[1]) % np[1];     // shift coordinates 
                    neigh_coords[2] = (neigh_coords[2] + k + np[2]) % np[2];     // shift coordinates 
                    MPI_Cart_rank(cart, neigh_coords, rec_neighs + count);       // find neighbor's rank                                   
                    count ++;
                }
            }
        }
    }

    sources = (rec_layers[0] + rec_layers[1] + 1) * (rec_layers[2] + rec_layers[3] + 1) * (rec_layers[4] + rec_layers[5] + 1) - 1;

#undef send_layers
#undef rec_layers

    MPI_Dist_graph_create_adjacent(cart, sources, rec_neighs, (int *)MPI_UNWEIGHTED, 
        destinations, send_neighs, (int *)MPI_UNWEIGHTED, MPI_INFO_NULL, reorder, comm_sq); 

    SqInd->scounts = (int*) calloc(destinations, sizeof(int));                                 // send counts of elements
    SqInd->rcounts = (int*) calloc(sources,      sizeof(int));                                 // receive counts of elements
    SqInd->sdispls = (int*) calloc(destinations, sizeof(int));                                 // send displacements
    SqInd->rdispls = (int*) calloc(sources,      sizeof(int));                                 // receive displacements
    assert(SqInd->scounts != NULL && SqInd->rcounts != NULL && SqInd->sdispls != NULL && SqInd->rdispls != NULL);

    SqInd->x_scounts = (int*) calloc(sizeof(int), send_layers[0] + send_layers[1] + 1);
    SqInd->y_scounts = (int*) calloc(sizeof(int), send_layers[2] + send_layers[3] + 1);
    SqInd->z_scounts = (int*) calloc(sizeof(int), send_layers[4] + send_layers[5] + 1);
    SqInd->x_rcounts = (int*) calloc(sizeof(int), rec_layers[0] + rec_layers[1] + 1);
    SqInd->y_rcounts = (int*) calloc(sizeof(int), rec_layers[2] + rec_layers[3] + 1);
    SqInd->z_rcounts = (int*) calloc(sizeof(int), rec_layers[4] + rec_layers[5] + 1);
    assert(SqInd->x_scounts != NULL && SqInd->y_scounts != NULL && SqInd->z_scounts != NULL 
        && SqInd->x_rcounts != NULL && SqInd->y_rcounts != NULL && SqInd->z_rcounts != NULL);

    Get_xyz_rs_counts(send_counts, send_layers, DMnxnynz, SqInd->x_scounts, SqInd->y_scounts, SqInd->z_scounts);

    Get_xyz_rs_counts(rec_counts, rec_layers, DMnxnynz, SqInd->x_rcounts, SqInd->y_rcounts, SqInd->z_rcounts);

    SqInd->n_in  = 0; 
    SqInd->n_out = 0;
    
    count = 0;
    for (k = 0; k < send_layers[4] + send_layers[5] + 1; k++) {
        for (j = 0; j < send_layers[2] + send_layers[3] + 1; j++) {
            for (i = 0; i < send_layers[0] + send_layers[1] + 1; i++) {
                if (i != send_layers[0] || j != send_layers[2] || k != send_layers[4]) {
                    SqInd->scounts[count] = SqInd->x_scounts[i] * SqInd->y_scounts[j] * SqInd->z_scounts[k];

                    if (count > 0)
                        SqInd->sdispls[count] = SqInd->sdispls[count - 1] + SqInd->scounts[count - 1];

                    SqInd->n_out += SqInd->scounts[count];
                    count ++;
                }
            }
        }
    }

    count = 0;
    for (k = rec_layers[4] + rec_layers[5]; k >= 0; k--) {
        for (j = rec_layers[2] + rec_layers[3]; j >= 0; j--) {
            for (i = rec_layers[0] + rec_layers[1]; i >= 0; i--) {
                if (i != rec_layers[0] || j != rec_layers[2] || k != rec_layers[4]) {
                    SqInd->rcounts[count] = SqInd->x_rcounts[i] * SqInd->y_rcounts[j] * SqInd->z_rcounts[k];

                    if (count > 0)
                        SqInd->rdispls[count] = SqInd->rdispls[count - 1] + SqInd->rcounts[count - 1];

                    SqInd->n_in += SqInd->rcounts[count];
                    count ++;
                }
            }
        }
    }

    free(SqInd->send_neighs);
    free(SqInd->rec_neighs);
    free(SqInd->send_counts);
    free(SqInd->rec_counts);
}

/**
 * @brief   Transferring Veff to PR domain use SQ type communication 
 */
void Transfer_Veff_PR(SPARC_OBJ *pSPARC, double ***Veff_PR, MPI_Comm comm_sq) {
    int i, j, k, ii, jj, kk, count, *nloc, rank;
    double *x_in, *x_out;
    MPI_Request request;
    SQ_OBJ *pSQ  = pSPARC->pSQ;
    SQIND *SqInd = pSQ->SqInd;
    int psize[3] = {pSQ->Nx_d_SQ, pSQ->Ny_d_SQ, pSQ->Nz_d_SQ};        // number of nodes in each direction
    
    MPI_Comm_rank(comm_sq, &rank);
    nloc = pSQ->nloc;
    //////////////////////////////////////////////////////////////////////

    x_in  = (double*) calloc(SqInd->n_in,  sizeof(double));                                // number of elements received from each neighbor 
    x_out = (double*) calloc(SqInd->n_out, sizeof(double));                                // number of elements sent to each neighbor
    assert(x_in != NULL && x_out != NULL);

    #ifdef DEBUG
    double tcomm1, tcomm2;
    tcomm1 = MPI_Wtime(); 
    #endif
    // assemble x_out
    count = 0;
    for (k = 0; k < SqInd->send_layers[4] + SqInd->send_layers[5] + 1; k++) {
        for (j = 0; j < SqInd->send_layers[2] + SqInd->send_layers[3] + 1; j++) {
            for (i = 0; i < SqInd->send_layers[0] + SqInd->send_layers[1] + 1; i++) {

                if (i != SqInd->rec_layers[0] || j != SqInd->rec_layers[2] || k != SqInd->rec_layers[4]) {

                    int start[3] = {0, 0 , 0};                                              // range of elements to be sent out
                    int end[3]   = {psize[0], psize[1], psize[2]};

                    if (i < SqInd->send_layers[0])
                        end[0] = SqInd->x_scounts[i];
                    else
                        start[0] = psize[0] - SqInd->x_scounts[i];

                    if (j < SqInd->send_layers[2])
                        end[1] = SqInd->y_scounts[j];
                    else
                        start[1] = psize[1] - SqInd->y_scounts[j];

                    if (k < SqInd->send_layers[4])
                        end[2] = SqInd->z_scounts[k];
                    else
                        start[2] = psize[2] - SqInd->z_scounts[k];

                    for (kk = start[2]; kk < end[2]; kk++)
                        for (jj = start[1]; jj < end[1]; jj++)
                            for (ii = start[0]; ii < end[0]; ii++)
                                x_out[count++] = Veff_PR[kk + nloc[2]][jj + nloc[1]][ii + nloc[0]];
                }
            }
        }
    }
 
    MPI_Ineighbor_alltoallv(x_out, SqInd->scounts, SqInd->sdispls, MPI_DOUBLE, x_in, SqInd->rcounts, SqInd->rdispls, MPI_DOUBLE, comm_sq, &request); 
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    // assemble x_in        
    int start[3] = {0,0,0};
    int end[3];

    count = 0;
    end[2] = 2 * nloc[2] + psize[2];
    for (k = SqInd->rec_layers[4] + SqInd->rec_layers[5]; k >= 0; k--) {
        end[1] = 2 * nloc[1] + psize[1];
        for (j = SqInd->rec_layers[2] + SqInd->rec_layers[3]; j >= 0; j--) {
            end[0] = 2 * nloc[0] + psize[0];
            for (i = SqInd->rec_layers[0] + SqInd->rec_layers[1]; i >= 0; i--) {

                start[0] = end[0] - SqInd->x_rcounts[i];
                start[1] = end[1] - SqInd->y_rcounts[j];
                start[2] = end[2] - SqInd->z_rcounts[k];

                if (i != SqInd->rec_layers[0] || j != SqInd->rec_layers[2] || k != SqInd->rec_layers[4])
                    for (kk = start[2]; kk < end[2]; kk++)
                        for (jj = start[1]; jj < end[1]; jj++)
                            for (ii = start[0]; ii < end[0]; ii++)
                                Veff_PR[kk][jj][ii] = x_in[count++];
                
                end[0] = start[0];
            }
            end[1] = start[1];
        }
        end[2] = start[2];
    }

    #ifdef DEBUG
    tcomm2 = MPI_Wtime();
    if(!rank) printf("Rank %d Transfering Veff into P.R. (Process + Rcut) domain takes %.3f ms\n",rank, (tcomm2-tcomm1)*1e3); 
    #endif  
    free(x_in);
    free(x_out);
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
    
    Set_D2D_Target(&pSPARC->d2d_s2p_sq, &pSPARC->d2d_s2p_phi, gridsizes, pSQ->DMVertices_SQ, pSPARC->DMVertices,
                    pSQ->dmcomm_SQ, sdims, pSPARC->dmcomm_phi, rdims, MPI_COMM_WORLD);
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
        pSPARC->DMVertices, rho_recv, pSQ->dmcomm_SQ, sdims, pSPARC->dmcomm_phi, rdims, MPI_COMM_WORLD);
        
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
        pSQ->DMVertices_SQ, Veff_recv, pSPARC->dmcomm_phi, sdims, pSQ->dmcomm_SQ, rdims, MPI_COMM_WORLD);
        
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, Transfer Veff from dmcomm_phi to SQ domain using D2D took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}


