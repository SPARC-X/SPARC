/**
 * @file    parallelization.h
 * @brief   This file contains the function declarations for parallelization.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef PARALLELIZATION_H
#define PARALLELIZATION_H 

#include "isddft.h"

void bind_proc_to_phys_core();
void bind_kptcomm_rank0_to_all_cores();
int  get_kptcomm_core_num();


/**
 * @brief   Set up sub-communicators.
 */
void Setup_Comms(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculates numbero of nodes of a distributed domain owned by  
 *          the process (in one direction).
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
//inline int block_decompose(const int n, const int p, const int rank);
static inline int block_decompose(const int n, const int p, const int rank)
{
    return n / p + ((rank < n % p) ? 1 : 0);
}


/**
 * @brief   Calculates start node of a distributed domain owned by  
 *          the process (in one direction).
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
//inline int block_decompose_nstart(const int n, const int p, const int rank);
static inline int block_decompose_nstart(const int n, const int p, const int rank)
{
    return n / p * rank + ((rank < n % p) ? rank : (n % p));
}


/**
 * @brief   Calculates which process owns the provided node of a 
 *          distributed domain (in one direction).
 */
static inline int block_decompose_rank(const int n, const int p, const int node_indx)
{
    return node_indx < (n % p) * ((n - 1) / p + 1) ? 
           node_indx / ((n - 1) / p + 1) : 
           (node_indx - (n % p)) / (n / p);
}


/**
 * @brief   Creates a balanced division of processors/subset of processors in a 
 *          Cartesian grid according to application size.
 *
 *          SPARC_Dims_create helps the user select a balanced distribution of processes per coordinate 
 *          direction, depending on the gridsize per process in each direction in the group to be 
 *          balanced and optional constraint that can be specified by the user. This function is a more
 *          adapted version of MPI_Dims_create in the sense that user can specify gridsize in each
 *          direction which is useful for domain decomposition. One can try to make the data distributed
 *          in each process to be as close to a cube as possible. In the case where the total number of 
 *          processors is not preferable, this function will try to find a better distribution in a subgroup
 *          of the processors and chose the biggest subset that are reasonable.
 * 
 * @param  nproc        Total number of processes.
 * @param  ndims        Number of Cartesian dimensions, for now it's either 2 or 3.
 * @param  gridsizes    Integer array of size ndims specifying the problem size in each dimension.
 * @param  minsize      The smallest gridsize per process in each direction. (e.g. finite-difference order)
 * @param  dims         OUTPUT. Integer array of size ndims specifying the number of nodes in each direction.
 * @param  ierr         Returns 1 if error occurs, otherwise return 0
 *
 */
void SPARC_Dims_create(int nproc, int ndims, int *gridsizes, int minsize, int *dims, int *ierr);


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

void ScaLAPACK_Dims_2D_BLCYC(int nproc, int *gridsizes, int *dims);


/**
 * @brief   Transfer data from one 3-d Domain Decomposition to another.
 *
 *          In this function we assume Domain Decomposition is quasi-uniform, i.e., each
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
 */
void DD2DD(SPARC_OBJ *pSPARC, int *gridsizes, int *sDMVert, double *sdata, int *rDMVert, double *rdata, 
           MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm);


/**
 * @brief   This function sets up the input structures needed for the function D2D(), 
 *          which transfers data from one domain decomposition to another domain 
 *          decomposition. It required that send_comm, recv_comm and union_comm overlap
 *          at the same root process.
 *          
 *          One can first call this function to set up d2d_sender and d2d_recvr, which
 *          contains the number of processes to communicate with along with their ranks
 *          in the union_comm.
 *          Note: In principle, it is not required to call this function if there're other
 *          ways to find out the senders and receivers for each process.
 */
void Set_D2D_Target(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, int *gridsizes, int *sDMVert, int *rDMVert,
         MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm);


/**
 * @brief   Free D2D_OBJ structure created by Set_D2D_Target or otherwise created manually
 *          as input to D2D.
 */
void Free_D2D_Target(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, MPI_Comm send_comm, MPI_Comm recv_comm);


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
 */
void D2D(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, int *gridsizes, int *sDMVert, double *sdata, int *rDMVert, 
         double *rdata, MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm);

#ifdef USE_DP_SUBEIG

/** @ brief   Copy row-major matrix block
 *
 *  @param unit_size  Size of data element in bytes (double == 8, double _Complex == 16)
 *  @param src_       Pointer to the top-left element of the source matrix 
 *  @param lds        Leading dimension of the source matrix
 *  @param nrow       Number of rows to copy
 *  @param ncol       Number of columns to copy
 *  @param dst_       Pointer to the top-left element of the destination matrix
 *  @param ldd        Leading dimension of the destination matrix
 */
void copy_mat_blk(
    const size_t unit_size, const void *src_, const int lds, 
    const int nrow, const int ncol, void *dst_, const int ldd
);

/** 
 *  Parameters for BP2DP and DP2BP
 *  comm           : MPI communicator 
 *  nproc          : Number of processes in comm
 *  {nrow,ncol}_bp : Number of rows/columns in local BP matrix block
 *  row_dp_displs  : Displacements of row indices in DP
 *  bp2dp_sendcnts : BP2DP send counts for MPI_Alltoallv
 *  bp2dp_sdispls  : BP2DP send displacements for MPI_Alltoallv
 *  dp2bp_sendcnts : DP2BP send counts for MPI_Alltoallv
 *  dp2bp_sdispls  : DP2BP send displacements for MPI_Alltoallv
 *  unit_size      : Size of data element in bytes (double == 8, double _Complex == 16)
 *  {bp,dp}_data_  : Pointer to local BP/DP column-major matrix block
 *  packbuf_       : Packing buffer, size >= nrow_bp * ncol_bp
 */

/**
 * @brief   Transfer data from band parallelization to domain parallelization.
 */
void BP2DP(
    const MPI_Comm comm, const int nproc, 
    const int nrow_bp, const int ncol_bp, const int *row_dp_displs,
    const int *bp2dp_sendcnts, const int *bp2dp_sdispls, 
    const int *dp2bp_sendcnts, const int *dp2bp_sdispls, 
    const int unit_size, const void *bp_data_, void *packbuf_, void *dp_data_
);

/**
 * @brief   Transfer data from domain parallelization to band parallelization.
 */
void DP2BP(
    const MPI_Comm comm, const int nproc, 
    const int nrow_bp, const int ncol_bp, const int *row_dp_displs, 
    const int *bp2dp_sendcnts, const int *bp2dp_sdispls, 
    const int *dp2bp_sendcnts, const int *dp2bp_sdispls, 
    const int unit_size, const void *dp_data_, void *packbuf_, void *bp_data_
);

#endif // "ifdef USE_DP_SUBEIG"

#endif // PARALLELIZATION_H 
