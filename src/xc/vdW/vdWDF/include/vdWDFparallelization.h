/**
 * @file    vdWDFparallelization.h
 * @brief   This file contains the function declarations for setting FFT topologies in vdWDF.
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
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef vdWDF_PARALLEL
#define vdWDF_PARALLEL

#include "isddft.h"
#include "exchangeCorrelation.h"

/**
 * @brief To detect whether there is weird space decomposition in FFT communicators
 *        if there is 0 (FFTW) or negative number (MKL) in array allLengthK, then return 1; otherwise return 0
 * @param FFTsize    the number of processors in zAxisComm
 * @param allLengthK the array collecting local length on z axis of all processors
**/ 
int judgeUnavailableLengthK(int FFTsize, int *allLengthK);

void vdWDF_Setup_Comms(SPARC_OBJ *pSPARC, int *gridsizes, int *phiDims);

/**
 * @brief Input the vertices on a direction and the dimensions (number of processors) on that direction,
 *        return which dimension (or interval) the node is in
 * @param vertices  The array of start vertices of the decomposed space along the direction (x, y or z).
 * @param p         The dimension of the 3-D topology along the direction (x, y or z).
 * @param node_indx The node index to be found which dimension it belongs to.
**/ 
int block_decompose_rank_AnyDMVert(const int *vertices, const int p, int node_indx);

/**
 * @brief let all processors in send_comm know how the space in recv_comm is decomposed by giving them recv_vertices;
 *        and let all processors in recv_comm know how the space in send_comm is decomposed by giving them send_vertices.
 *        It required that send_comm, recv_comm overlap at the same root process.
 * @param sDMVert       (INPUT) The vertices of the processor in send communicator.
 * @param rDMVert       (INPUT) The vertices of the processor in receive communicator.
 * @param all_send_vertices (OUTPUT) The start vertices of decomposed send communicator along all three directions, the length is sdims[0]+sdims[1]+sdims[2].
 * @param all_recv_vertices (OUTPUT) The start vertices of decomposed receive communicator along all three directions, the length is rdims[0]+rdims[1]+rdims[2].
 * @param send_comm     Original communicator in which the data was distributed. Processors not
 *                      part of the original DD should provide MPI_COMM_NULL for this argument.
 *                      send_comm is assumed to have a Cartesian topology.
 * @param sdims         (INPUT) Dimensions of the send_comm.
 * @param recv_comm     New Domain Decomposition communicator, processes that are not part of this
 *                      should provide MPI_COMM_NULL. recv_comm is assumed to have a Cartesian topology.
 * @param rdims         (INPUT) Dimensions of the recv_comm.
**/
void Exchange_send_recv_vertices(int *sDMVert, int *rDMVert, int *all_send_vertices, int *all_recv_vertices,
    MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims);

/**
 * @brief   This function sets up the input structures needed for the function D2D(), 
 *          which transfers data from one domain decomposition to another domain 
 *          decomposition. It required that send_comm, recv_comm and union_comm overlap
 *          at the same root process.
 *          The difference between D2D and D2D_AnyDMVert is D2D_AnyDMVert accept any possible 
 *          decomposition of the 3-d domain.
 *          One can first call this function to set up d2d_sender and d2d_recvr, which
 *          contains the number of processes to communicate with along with their ranks
 *          in the union_comm.
 */
void Set_D2D_Target_AnyDMVert(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, int *gridsizes, int *sDMVert, int *rDMVert,
         MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm);

void Free_D2D_Target_AnyDMVert(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, MPI_Comm send_comm, MPI_Comm recv_comm);

/**
 * @brief   Transfer data from one 3-d Domain Decomposition to another.
 *          The difference between D2D and D2D_AnyDMVert is D2D_AnyDMVert accept any possible 
 *          decomposition of the 3-d domain. It required that send_comm, recv_comm overlap
 *          at the same root process.
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
 */
void D2D_AnyDMVert(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, int *gridsizes, int *sDMVert, double *sdata, int *rDMVert,
         double *rdata, MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm);

#endif