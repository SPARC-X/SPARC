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
void SPARC_Dims_create(const int nproc, const int ndims, const int *gridsizes, int minsize, int *dims, int *ierr);


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
void dims_divide_2d(const int N1, const int N2, const int np, int *np1, int *np2);


/**
 * @brief  For the given parallelization params, find how many count of 
 *         work load each process is assigned, we take the ceiling since
 *         at least one process would have that many columns and will
 *         dominate the cost.
 *         
 **/
int workload_assigned(const int Nk, const int Ns, 
    const int np1, const int np2);


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
);



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
    const int np, int *np1, int *np2, int *np3);


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
    int *nps, int *npk, int *npb, int *npd, int minsize, int isfock);


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
 * @param unit_size     either sizeof(double) or sizeof(double _Complex) for the type of sdata and rdata
 */
void DD2DD(SPARC_OBJ *pSPARC, int *gridsizes, int *sDMVert, void *sdata, int *rDMVert, void *rdata, 
           MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm, int unit_size);


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
 * @param unit_size     either sizeof(double) or sizeof(double _Complex) for the type of sdata and rdata
 */
void D2D(D2D_OBJ *d2d_sender, D2D_OBJ *d2d_recvr, int *gridsizes, int *sDMVert, void *sdata, int *rDMVert,
         void *rdata, MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims, MPI_Comm union_comm, int unit_size);
         
#ifdef USE_DP_SUBEIG
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


void Set_D2Dext_Target(D2Dext_OBJ *d2dext_sender, D2Dext_OBJ *d2dext_recvr, 
    int DMnx, int DMny, int DMnz, int xext, int yext, int zext, 
    int gridsizes[3], int dims[3], MPI_Comm cart, MPI_Comm *comm_d2dext);

    
void reorder_counts(const int *counts, const int *layers, const int DMnx, const int DMny, const int DMnz, 
                    int *x_counts, int *y_counts, int *z_counts);

void free_D2Dext_Target(D2Dext_OBJ *d2dext, MPI_Comm comm_d2dext);

void D2Dext(D2Dext_OBJ *d2dext_sender, D2Dext_OBJ *d2dext_recvr, int DMnx, int DMny, int DMnz, 
    int xext, int yext, int zext, void *sdata, void *rdata, MPI_Comm comm_d2dext, int unit_size);


#ifdef __cplusplus
extern "C" {
#endif

int MPI_Allreduce_overload(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif // PARALLELIZATION_H 
