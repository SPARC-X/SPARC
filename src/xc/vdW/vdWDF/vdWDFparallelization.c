/**
 * @file    vdWDFparallelization.c
 * @brief   This file contains functions generating a special domain communicator for FFT and iFFT in vdW.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * Dion, Max, Henrik Rydberg, Elsebeth Schröder, David C. Langreth, and Bengt I. Lundqvist. 
 * "Van der Waals density functional for general geometries." 
 * Physical review letters 92, no. 24 (2004): 246401.
 * Román-Pérez, Guillermo, and José M. Soler. 
 * "Efficient implementation of a van der Waals density functional: application to double-wall carbon nanotubes." 
 * Physical review letters 103, no. 9 (2009): 096102.
 * Lee, Kyuho, Éamonn D. Murray, Lingzhu Kong, Bengt I. Lundqvist, and David C. Langreth. 
 * "Higher-accuracy van der Waals density functional." Physical Review B 82, no. 8 (2010): 081101.
 * Thonhauser, T., S. Zuluaga, C. A. Arter, K. Berland, E. Schröder, and P. Hyldgaard. 
 * "Spin signature of nonlocal correlation binding in metal-organic frameworks." 
 * Physical review letters 115, no. 13 (2015): 136402.
 * vdW-DF feature developed in Quantum Espresso:
 * Thonhauser, Timo, Valentino R. Cooper, Shen Li, Aaron Puzder, Per Hyldgaard, and David C. Langreth. 
 * "Van der Waals density functional: Self-consistent potential and the nature of the van der Waals bond." 
 * Physical Review B 76, no. 12 (2007): 125112.
 * Sabatini, Riccardo, Emine Küçükbenli, Brian Kolb, Timo Thonhauser, and Stefano De Gironcoli. 
 * "Structural evolution of amino acid crystals under stress from a non-empirical density functional." 
 * Journal of Physics: Condensed Matter 24, no. 42 (2012): 424209.
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <complex.h>
#include <errno.h> 
#include <time.h>

#include "isddft.h"
#include "tools.h"
#include "parallelization.h"
#include "vdWDFinitialization.h"
#include "vdWDFparallelization.h"
#include "gradVecRoutines.h"
#include "exchangeCorrelation.h"

/** FFT routines **/
#ifdef USE_MKL
#include "mkl_cdft.h"
#endif
#ifdef USE_FFTW
#include <fftw3-mpi.h>
#endif

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

int judgeUnavailableLengthK(int FFTsize, int *allLengthK) {
    // if there is 0 (FFTW) or negative number (MKL) in array allLengthK, then return 1; otherwise return 0
    for (int i = 0; i < FFTsize; i++) {
        // printf("%d, ", allLengthK[i]);
        if (allLengthK[i] <= 0) return 1;
    }
    return 0;
}

void vdWDF_Setup_Comms(SPARC_OBJ *pSPARC, int *gridsizes, int *phiDims) {
    //------------------------------------------------//
    //      set up poisson domain for FFT in vdWDF    //
    //------------------------------------------------//
    int rank, nproc;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    MPI_Comm_size(pSPARC->dmcomm_phi, &nproc);
    int *newzAxisDims = pSPARC->newzAxisDims;
    newzAxisDims[0] = 1;
    newzAxisDims[1] = 1;
    int flagUnavailableLengthK = 1;
    newzAxisDims[2] = (gridsizes[2] > 2) ? (gridsizes[2] / 2) : 1; // initial number of processors is set as 16
    if (gridsizes[1] < newzAxisDims[2]) { // length of y and z dim should not be less than nproc
        newzAxisDims[2] = gridsizes[1];
    }
    if (nproc < newzAxisDims[2]) {
        newzAxisDims[2] = nproc;
    }
    pSPARC->zAxisComm = MPI_COMM_NULL;
    MPI_Barrier(pSPARC->dmcomm_phi);
    int periods[3] = {1, 1, 1};
    while (flagUnavailableLengthK) {
        MPI_Cart_create(pSPARC->dmcomm_phi, 3, newzAxisDims, periods, 1, &(pSPARC->zAxisComm));
        int *allLengthK;
        int FFTrank = -1;
        int FFTsize = -1;
        if (pSPARC->zAxisComm != MPI_COMM_NULL) {
            MPI_Comm_rank(pSPARC->zAxisComm, &FFTrank);
            MPI_Comm_size(pSPARC->zAxisComm, &FFTsize);
            pSPARC->zAxisVertices[0] = 0; 
            pSPARC->zAxisVertices[1] = gridsizes[0] - 1;
            pSPARC->zAxisVertices[2] = 0; 
            pSPARC->zAxisVertices[3] = gridsizes[1] - 1;
            int lengthKint, startKint;
            allLengthK = (int *)malloc(sizeof(int) * FFTsize);
            #if defined(USE_MKL) // use MKL CDFT
                // initializa parallel FFT
                DFTI_DESCRIPTOR_DM_HANDLE desc = NULL;
                MKL_LONG localArrayLength, lengthK, startK;
                MKL_LONG dim_sizes[3] = {gridsizes[2], gridsizes[1], gridsizes[0]};
                DftiCreateDescriptorDM(pSPARC->zAxisComm, &desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
                DftiGetValueDM(desc, CDFT_LOCAL_SIZE, &localArrayLength);
                DftiGetValueDM(desc, CDFT_LOCAL_NX, &lengthK);
                DftiGetValueDM(desc, CDFT_LOCAL_X_START, &startK);
                lengthKint = lengthK;
                startKint = startK;
                pSPARC->zAxisVertices[4] = startKint;
                pSPARC->zAxisVertices[5] = startKint + lengthKint - 1;
                DftiFreeDescriptorDM(&desc);
            #elif defined(USE_FFTW) // use FFTW if MKL is not used
                ptrdiff_t localArrayLength, lengthK, startK;
                fftw_mpi_init();
                const ptrdiff_t N0 = gridsizes[2], N1 = gridsizes[1], N2 = gridsizes[0]; // N0: z; N1:Y; N2:x
                /* get local data size and allocate */
                localArrayLength = fftw_mpi_local_size_3d(N0, N1, N2, pSPARC->zAxisComm, &lengthK, &startK);
                lengthKint = lengthK;
                startKint = startK;
                pSPARC->zAxisVertices[4] = startKint;
                pSPARC->zAxisVertices[5] = startKint + lengthKint - 1;
                fftw_mpi_cleanup();
            #endif
            MPI_Allgather(&lengthKint, 1, MPI_INT, allLengthK, 1, MPI_INT, pSPARC->zAxisComm);
            if (FFTrank == 0) {
                flagUnavailableLengthK = judgeUnavailableLengthK(FFTsize, allLengthK);
            }
            free(allLengthK);
        }
        // MPI_Request req[2];
        // MPI_Status sta[2];
        // if (FFTrank == 0) MPI_Isend(&flagUnavailableLengthK, 1, MPI_INT, 0, 0, pSPARC->dmcomm_phi, req);
        // if (rank == 0) {
        //     MPI_Irecv(&flagUnavailableLengthK, 1, MPI_INT, MPI_ANY_SOURCE, 0, pSPARC->dmcomm_phi, &req[1]);
        //     MPI_Wait(&req[1], &sta[1]);
        // }
        MPI_Barrier(pSPARC->dmcomm_phi);
        MPI_Bcast(&flagUnavailableLengthK, 1, MPI_INT, 0, pSPARC->dmcomm_phi); // Since zAxisComm comes from dmcomm_phi by MPI_Cart_Create, the two comms have
        // similar root processor
        MPI_Barrier(pSPARC->dmcomm_phi);
        if (flagUnavailableLengthK) { // if there is empty processor in zAxisComm, then we remove a processor, and reset the zAxisComm
            newzAxisDims[2] --;
            if (pSPARC->zAxisComm != MPI_COMM_NULL) {
                MPI_Comm_free(&pSPARC->zAxisComm);
                pSPARC->zAxisComm = MPI_COMM_NULL;
            }
        }
    }
    if (pSPARC->zAxisComm != MPI_COMM_NULL) {
        int FFTrank, FFTsize;
        MPI_Comm_rank(pSPARC->zAxisComm, &FFTrank);
        MPI_Comm_size(pSPARC->zAxisComm, &FFTsize);
        int coord_newZAxisComm[3];
        MPI_Cart_coords(pSPARC->zAxisComm, FFTrank, 3, coord_newZAxisComm);
#ifdef DEBUG
        printf("I am rank %d in zAxisComm, my coord is %d %d %d, my ZAxisVertices is %d %d, there are %d processors in it.\n",
            FFTrank, coord_newZAxisComm[0], coord_newZAxisComm[1], coord_newZAxisComm[2], pSPARC->zAxisVertices[4], pSPARC->zAxisVertices[5], FFTsize);
#endif
    }

    // compose the sender and receiver for gathering thetas to do parallel FFT
    // D2D_OBJ gatherThetasSender, gatherThetasRecvr;
    Set_D2D_Target_AnyDMVert(&pSPARC->gatherThetasSender, &pSPARC->gatherThetasRecvr, gridsizes, 
                    pSPARC->DMVertices, pSPARC->zAxisVertices,
                   pSPARC->dmcomm_phi, phiDims,
                   pSPARC->zAxisComm, newzAxisDims, pSPARC->dmcomm_phi);
    // Free_D2D_Target(&gatherThetasSender, &gatherThetasRecvr, pSPARC->dmcomm_phi, newZAxisComm);
    // compose the sender and receiver for scattering thetasFT after parallel FFT
    // D2D_OBJ scatterThetasSender, scatterThetasRecvr;
    Set_D2D_Target_AnyDMVert(&pSPARC->scatterThetasSender, &pSPARC->scatterThetasRecvr, gridsizes,
                    pSPARC->zAxisVertices, pSPARC->DMVertices,
                   pSPARC->zAxisComm, newzAxisDims,
                   pSPARC->dmcomm_phi, phiDims, pSPARC->dmcomm_phi);
    // Free_D2D_Target(&scatterThetasSender, &scatterThetasRecvr, newZAxisComm, pSPARC->dmcomm_phi);
    // if (newZAxisComm != MPI_COMM_NULL)
    //     MPI_Comm_free(&newZAxisComm);
}


/**
 * @brief Input the vertices on a direction and the dimensions (number of processors) on that direction,
 *        return which dimension (or interval) the node is in
**/ 
int block_decompose_rank_AnyDMVert(const int *vertices, const int p, int node_indx) {
    int index = 0;
    for (index = 0; index < p - 1; index++) {
        if ((node_indx >= vertices[index]) && (node_indx < vertices[index + 1]))
            return index;
    }
    return index;
}

/**
 * @brief let all processors in send_comm know how the space in recv_comm is decomposed by giving them recv_vertices;
 *        and let all processors in recv_comm know how the space in send_comm is decomposed by giving them send_vertices.
 *        It required that send_comm, recv_comm overlap at the same root process.
**/
void Exchange_send_recv_vertices(int *sDMVert, int *rDMVert, int *all_send_vertices, int *all_recv_vertices,
    MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims) {
    // MPI_Request req[2];
    MPI_Status sta[2];
    int myRecvRank = -1;
    int mySendRank = -1;
    // let all processors in send_comm know vertices in recv_comm
    if (recv_comm != MPI_COMM_NULL) {
        int count = 0;
        MPI_Comm_rank(recv_comm, & myRecvRank);
        for (int dir = 0; dir < 3; dir++) {
            all_recv_vertices[count] = 0; // all decomposition starts from 0th grid
            count++;
            for (int index = 1; index < rdims[dir]; index++) {
                int recv_coord_temp[3] = {0, 0, 0};
                recv_coord_temp[dir] = index;
                int theRecvRank = 0;
                MPI_Cart_rank(recv_comm, recv_coord_temp, &theRecvRank);
                if (myRecvRank == theRecvRank) 
                    MPI_Send(&rDMVert[2*dir], 1, MPI_INT, 0, 20, recv_comm);
                if (myRecvRank == 0) {
                    MPI_Recv(&all_recv_vertices[count], 1, MPI_INT, theRecvRank, 20, recv_comm, &sta[1]);
                    // MPI_Wait(&req[1], &sta[1]);
                }
                count++;
            }
        }
    }
    if (send_comm != MPI_COMM_NULL) {
        MPI_Bcast(all_recv_vertices, (rdims[0] + rdims[1] + rdims[2]), MPI_INT, 0, send_comm);
    }
    // let all processors in recv_comm know vertices in send_comm
    if (send_comm != MPI_COMM_NULL) {
        int count = 0;
        MPI_Comm_rank(send_comm, & mySendRank);
        for (int dir = 0; dir < 3; dir++) {
            all_send_vertices[count] = 0;
            count++;
            for (int index = 1; index < sdims[dir]; index++) {
                int send_coord_temp[3] = {0, 0, 0};
                send_coord_temp[dir] = index;
                int theSendRank = 0;
                MPI_Cart_rank(send_comm, send_coord_temp, &theSendRank);
                if (mySendRank == theSendRank) 
                    MPI_Send(&sDMVert[2*dir], 1, MPI_INT, 0, 21, send_comm);
                if (mySendRank == 0) {
                    MPI_Recv(&all_send_vertices[count], 1, MPI_INT, theSendRank, 21, send_comm, &sta[1]);
                    // MPI_Wait(&req[1], &sta[1]);
                }
                count++;
            }
        }
    }
    if (recv_comm != MPI_COMM_NULL) {
        MPI_Bcast(all_send_vertices, (sdims[0] + sdims[1] + sdims[2]), MPI_INT, 0, recv_comm);
    }

}

/**
 * @brief   This function sets up the input structures needed for the function D2D(),
 *          which transfers data from one domain decomposition to another domain
 *          decomposition. It's required that send_comm, recv_comm and union_comm overlap
 *          at the same root process.
 *          The difference between D2D and D2D_AnyDMVert is D2D_AnyDMVert accept any possible 
 *          decomposition of the 3-d domain.
 *          One can first call this function to set up d2d_sender and d2d_recvr, which
 *          contains the number of processes to communicate with along with their ranks
 *          in the union_comm.
 */
void Set_D2D_Target_AnyDMVert(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, int *gridsizes, int *sDMVert, int *rDMVert,
         MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t3, t4;

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

    int *all_send_vertices = (int*)malloc(sizeof(int) * (sdims[0] + sdims[1] + sdims[2])); // how the 3D spaces are divided in send_comm? x y z dir
    int *all_recv_vertices = (int*)malloc(sizeof(int) * (rdims[0] + rdims[1] + rdims[2])); // how the 3D spaces are divided in recv_comm? x y z dir
    
    Exchange_send_recv_vertices(sDMVert, rDMVert, all_send_vertices, all_recv_vertices,
        send_comm, sdims, recv_comm, rdims);
    
    // initialize nsend_tot and nrecv_tot to zero
    d2d_sender->n_target = 0;
    d2d_recvr ->n_target = 0;

    // set up send processes
    if (send_comm != MPI_COMM_NULL) {
        t3 = MPI_Wtime();
        MPI_Cart_coords(send_comm, rank_send_comm, 3, coords_send_comm);
        nsend_tot = 1;
        int all_recv_vertices_start = 0;
        // find out in each dimension, how many recv processes the local domain spans over
        for (n = 0; n < 3; n++) {
            send_coord_start[n] = block_decompose_rank_AnyDMVert((all_recv_vertices + all_recv_vertices_start), rdims[n], sDMVert[n*2]);
            send_coord_end[n] = block_decompose_rank_AnyDMVert((all_recv_vertices + all_recv_vertices_start), rdims[n], sDMVert[n*2+1]);
            nsend[n] = (send_coord_end[n] - send_coord_start[n] + 1);
            nsend_tot *= nsend[n];
            all_recv_vertices_start  += rdims[n];
        }
        d2d_sender->n_target = nsend_tot;
        d2d_sender->target_ranks = (int *)malloc(nsend_tot * sizeof(int));

        send_coords = (int *)malloc(nsend_tot * 3 * sizeof(int));
        // find out all the coordinates of the receiver processes in the recv_comm Cart Topology
        c_ndgrid(3, send_coord_start, send_coord_end, send_coords);

        t4 = MPI_Wtime();
#ifdef DEBUG
        if (rank == 0) printf("======Set_D2D_Target: find receivers in each process (c_ndgrid) in send_comm took %.3f ms\n", (t4-t3)*1e3);
#endif
        t3 = MPI_Wtime();

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

        t4 = MPI_Wtime();
#ifdef DEBUG
        if (rank == 0) printf("======Set_D2D_Target: Gather and Scatter receivers in send_comm took %.3f ms\n", (t4-t3)*1e3);
#endif
    }

    // set up receiver processes
    if (recv_comm != MPI_COMM_NULL) {
        MPI_Cart_coords(recv_comm, rank_recv_comm, 3, coords_recv_comm);
        nrecv_tot = 1;
        // find in each dimension, how many send processes the local domain spans over
        int all_send_vertices_start = 0;
        for (n = 0; n < 3; n++) {
            recv_coord_start[n] = block_decompose_rank_AnyDMVert((all_send_vertices + all_send_vertices_start), sdims[n], rDMVert[n*2]);
            recv_coord_end[n] = block_decompose_rank_AnyDMVert((all_send_vertices + all_send_vertices_start), sdims[n], rDMVert[n*2+1]);
            nrecv[n] = (recv_coord_end[n] - recv_coord_start[n] + 1);
            nrecv_tot *= nrecv[n];
            all_send_vertices_start += sdims[n];
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

#ifdef DEBUG
    if (rank_send_comm == 0) {
        printf("I am send %d, my send vertices are %d %d %d %d %d %d, nsend %d %d %d\n", rank_send_comm,
            sDMVert[0], sDMVert[1], sDMVert[2], sDMVert[3], sDMVert[4], sDMVert[5],
            nsend[0], nsend[1], nsend[2]);
        printf("I am send %d, recv vertices ", rank_send_comm);
        for (int index = 0; index < (rdims[0] + rdims[1] + rdims[2]); index++)
            printf("%d ", all_recv_vertices[index]);
        printf("\n");
    }
    if (rank_recv_comm == 0) {
        printf("I am recv %d, my recv vertices are %d %d %d %d %d %d, nrecv %d %d %d\n", rank_recv_comm,
            rDMVert[0], rDMVert[1], rDMVert[2], rDMVert[3], rDMVert[4], rDMVert[5],
            nrecv[0], nrecv[1], nrecv[2]);
        printf("I am recv %d, send vertices ", rank_recv_comm);
        for(int index = 0; index < (sdims[0] + sdims[1] + sdims[2]); index++)
            printf("%d ", all_send_vertices[index]);
        printf("\n");
    }
#endif

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

    free(all_send_vertices);
    free(all_recv_vertices);
}



/**
 * @brief   Free D2D_OBJ structure created by Set_D2D_Target or otherwise created manually
 *          as input to D2D.
 */
void Free_D2D_Target_AnyDMVert(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, MPI_Comm send_comm, MPI_Comm recv_comm)
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
 *          The difference between D2D and D2D_AnyDMVert is D2D_AnyDMVert accept any possible 
 *          decomposition of the 3-d domain.
 */
void D2D_AnyDMVert(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, int *gridsizes, int *sDMVert, double *sdata, int *rDMVert,
         double *rdata, MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2, t3, t4;

    int ndims, rank_recv_comm, rank_send_comm,
        coords_send_comm[3], coords_recv_comm[3];
    ndims = 3;

    t1 = MPI_Wtime();

    int i, j, k, iloc, jloc, kloc, n;
    int nsend_tot, *send_coord_start, *send_coord_end, *send_coords;
    nsend_tot = 0;
    send_coord_start = NULL;
    send_coord_end = NULL;
    send_coords = NULL;

    // buffers
    int DMnd, *DMnxi, *sDMnxi, *rDMnxi, *DMVert_temp;
    double **sendbuf, **recvbuf;
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

    int *all_send_vertices = (int*)malloc(sizeof(int) * (sdims[0] + sdims[1] + sdims[2])); // how the 3D spaces are divided in send_comm? x y z dir
    int *all_recv_vertices = (int*)malloc(sizeof(int) * (rdims[0] + rdims[1] + rdims[2])); // how the 3D spaces are divided in recv_comm? x y z dir

    Exchange_send_recv_vertices(sDMVert, rDMVert, all_send_vertices, all_recv_vertices,
        send_comm, sdims, recv_comm, rdims);

    // set up send processes
    if (send_comm != MPI_COMM_NULL) {

        t3 = MPI_Wtime();

        for (n = 0; n < 3; n++) {
            sDMnxi[n] = sDMVert[2*n+1] - sDMVert[2*n] + 1;
        }
        send_coord_start = (int *)malloc(ndims * sizeof(int));
        send_coord_end = (int *)malloc(ndims * sizeof(int));
        MPI_Comm_rank(send_comm, &rank_send_comm);
        MPI_Cart_coords(send_comm, rank_send_comm, ndims, coords_send_comm);

        // find out in each dimension, how many recv processes the local domain spans over
        int all_recv_vertices_start = 0;
        for (n = 0; n < 3; n++) {
            send_coord_start[n] = block_decompose_rank_AnyDMVert((all_recv_vertices + all_recv_vertices_start), rdims[n], sDMVert[n*2]);
            send_coord_end[n] = block_decompose_rank_AnyDMVert((all_recv_vertices + all_recv_vertices_start), rdims[n], sDMVert[n*2+1]);
            all_recv_vertices_start  += rdims[n];
        }
        nsend_tot = d2d_sender->n_target;
        send_coords = (int *)malloc(nsend_tot * ndims * sizeof(int));

        // find out all the coordinates of the receiver processes in the recv_comm Cart Topology
        c_ndgrid(ndims, send_coord_start, send_coord_end, send_coords);

        t4 = MPI_Wtime();
        #ifdef DEBUG
            if (rank == 0) printf("======D2D: find receivers' coords in each process (c_ndgrid) in send_comm took %.3f ms\n", (t4-t3)*1e3);
        #endif

        sendbuf = (double **)malloc(nsend_tot * sizeof(double *));
        send_request = malloc(nsend_tot * sizeof(*send_request));
        // go over each receiver process send data to that process
        for (n = 0; n < nsend_tot; n++) {
            DMnd = 1;
            all_recv_vertices_start = 0;
            for (i = 0; i < 3; i++) {
                coord_temp[i] = send_coords[i*nsend_tot+n];
                // local domain in the receiver process
                DMVert_temp[2*i] = *(all_recv_vertices + all_recv_vertices_start + coord_temp[i]);
                if (coord_temp[i] < rdims[i] - 1)
                    DMVert_temp[2*i+1] = *(all_recv_vertices + all_recv_vertices_start + coord_temp[i] + 1) - 1; // the start vertice of the next processor -1
                else // if the last processor on ith direction, its end vertice is gridsize on that direction
                    DMVert_temp[2*i+1] = gridsizes[i] - 1;
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                // find intersect of local domains in send process and receiver process
                DMVert_temp[2*i] = max(DMVert_temp[2*i], sDMVert[2*i]);
                DMVert_temp[2*i+1] = min(DMVert_temp[2*i+1], sDMVert[2*i+1]);
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                DMnd *= DMnxi[i];
                all_recv_vertices_start += rdims[i];
            }
            sendbuf[n] = (double *)malloc( DMnd * sizeof(double));
            for (k = 0; k < DMnxi[2]; k++) {
                kloc = k + DMVert_temp[4] - sDMVert[4];
                for (j = 0; j < DMnxi[1]; j++) {
                    jloc = j + DMVert_temp[2] - sDMVert[2];
                    for (i = 0; i < DMnxi[0]; i++) {
                        iloc = i + DMVert_temp[0] - sDMVert[0];
                        sendbuf[n][k*DMnxi[1]*DMnxi[0]+j*DMnxi[0]+i] =
                        sdata[kloc*sDMnxi[1]*sDMnxi[0]+jloc*sDMnxi[0]+iloc];
                    }
                }
            }
            MPI_Isend(sendbuf[n], DMnd, MPI_DOUBLE, d2d_sender->target_ranks[n], 111, union_comm, &send_request[n]);
        }
    }
#ifdef DEBUG
    if (rank == 0) printf("======D2D: finished initiating send_comm! Start entering receivers comm\n");
#endif
    int nrecv_tot, *recv_coord_start, *recv_coord_end, *recv_coords;
    nrecv_tot = 0;
    recv_coord_start = NULL;
    recv_coord_end = NULL;
    recv_coords = NULL;

    // set up receiver processes
    if (recv_comm != MPI_COMM_NULL) {
        t3 = MPI_Wtime();
        MPI_Comm_rank(recv_comm, &rank_recv_comm);
        MPI_Cart_coords(recv_comm, rank_recv_comm, ndims, coords_recv_comm);

        // local domain sizes
        for (n = 0; n < ndims; n++) {
            rDMnxi[n] = rDMVert[2*n+1] - rDMVert[2*n] + 1;
        }

        recv_coord_start = (int *)malloc(ndims * sizeof(int));
        recv_coord_end = (int *)malloc(ndims * sizeof(int));

        // find in each dimension, how many send processes the local domain spans over
        int all_send_vertices_start = 0;
        for (n = 0; n < ndims; n++) {
            recv_coord_start[n] = block_decompose_rank_AnyDMVert((all_send_vertices + all_send_vertices_start), sdims[n], rDMVert[n*2]);
            recv_coord_end[n] = block_decompose_rank_AnyDMVert((all_send_vertices + all_send_vertices_start), sdims[n], rDMVert[n*2+1]);
            all_send_vertices_start += sdims[n];
        }

        nrecv_tot = d2d_recvr->n_target;
        recv_coords = (int *)malloc(nrecv_tot * ndims * sizeof(int));

        // find out all the coords of the send process in the send_comm topology
        c_ndgrid(ndims, recv_coord_start, recv_coord_end, recv_coords);

    t4 = MPI_Wtime();
#ifdef DEBUG
    if (rank == 0) printf("======D2D: find senders' coords in each process (c_ndgrid) in recv_comm took %.3f ms\n", (t4-t3)*1e3);
#endif

        recvbuf = (double **)malloc(nrecv_tot * sizeof(double *));
        recv_request = malloc(nrecv_tot * sizeof(*recv_request));
        // go over each send process and receive data from that process
        for (n = 0; n < nrecv_tot; n++) {
            DMnd = 1;
            all_send_vertices_start = 0;
            for (i = 0; i < ndims; i++) {
                coord_temp[i] = recv_coords[i*nrecv_tot+n];
                // local domain in the send process
                DMVert_temp[2*i] = *(all_send_vertices + all_send_vertices_start + coord_temp[i]);
                if (coord_temp[i] < sdims[i] - 1)
                    DMVert_temp[2*i+1] = *(all_send_vertices + all_send_vertices_start + coord_temp[i] + 1) - 1; // the start vertice of the next processor -1
                else // if the last processor on ith direction, its end vertice is gridsize on that direction
                    DMVert_temp[2*i+1] = gridsizes[i] - 1;
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                // find intersect of local domains in send process and receiver process
                DMVert_temp[2*i] = max(DMVert_temp[2*i], rDMVert[2*i]);
                DMVert_temp[2*i+1] = min(DMVert_temp[2*i+1], rDMVert[2*i+1]);
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                DMnd *= DMnxi[i];
                all_send_vertices_start += sdims[i];
            }
            recvbuf[n] = (double *)malloc( DMnd * sizeof(double));
            MPI_Irecv(recvbuf[n], DMnd, MPI_DOUBLE, d2d_recvr->target_ranks[n], 111, union_comm, &recv_request[n]);
        }
    }

    t2 = MPI_Wtime();
#ifdef DEBUG
    if (rank == 0) printf("======D2D: initiated sending and receiving took %.3f ms\n", (t2-t1)*1e3);
#endif

    if (send_comm != MPI_COMM_NULL) {
        MPI_Waitall(nsend_tot, send_request, MPI_STATUS_IGNORE);
    }

    if (recv_comm != MPI_COMM_NULL) {
        MPI_Waitall(nrecv_tot, recv_request, MPI_STATUS_IGNORE);
        int all_send_vertices_start = 0;
        for (n = 0; n < nrecv_tot; n++) {
            DMnd = 1;
            all_send_vertices_start = 0;
            for (i = 0; i < ndims; i++) {
                coord_temp[i] = recv_coords[i*nrecv_tot+n];
                // local domain in the send process
                DMVert_temp[2*i] = *(all_send_vertices + all_send_vertices_start + coord_temp[i]);
                if (coord_temp[i] < sdims[i] - 1)
                    DMVert_temp[2*i+1] = *(all_send_vertices + all_send_vertices_start + coord_temp[i] + 1) - 1; // the start vertice of the next processor -1
                else // if the last processor on ith direction, its end vertice is gridsize on that direction
                    DMVert_temp[2*i+1] = gridsizes[i] - 1;
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                // find intersect of local domains in send process and receiver process
                DMVert_temp[2*i] = max(DMVert_temp[2*i], rDMVert[2*i]);
                DMVert_temp[2*i+1] = min(DMVert_temp[2*i+1], rDMVert[2*i+1]);
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                DMnd *= DMnxi[i];
                all_send_vertices_start += sdims[i];
            }
            for (k = 0; k < DMnxi[2]; k++) {
                kloc = k + DMVert_temp[4] - rDMVert[4];
                for (j = 0; j < DMnxi[1]; j++) {
                    jloc = j + DMVert_temp[2] - rDMVert[2];
                    for (i = 0; i < DMnxi[0]; i++) {
                        iloc = i + DMVert_temp[0] - rDMVert[0];
                        rdata[kloc*rDMnxi[1]*rDMnxi[0]+jloc*rDMnxi[0]+iloc] =
                        recvbuf[n][k*DMnxi[1]*DMnxi[0]+j*DMnxi[0]+i];
                    }
                }
            }
        }
    }

    if (send_comm != MPI_COMM_NULL) {
        for (n = 0; n < nsend_tot; n++) {
            free(sendbuf[n]);
        }
        free(sendbuf);
        free(send_request);
        free(send_coord_start);
        free(send_coord_end);
        free(send_coords);
    }

    if (recv_comm != MPI_COMM_NULL) {
        for (n = 0; n < nrecv_tot; n++) {
            free(recvbuf[n]);
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

    free(all_send_vertices);
    free(all_recv_vertices);
}