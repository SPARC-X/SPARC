/**
 * @file    mpi_nbr_shim.c
 * @brief   Implementation of SPARC_Ineighbor_alltoallv.
 *
 * See include/mpi_nbr_shim.h for the two matching rules, the measured
 * evidence, and why this exists.
 *
 * @author  Alfredo Metere <alfredo.metere@metereconsulting.com>
 */

#include <stdlib.h>
#include <mpi.h>

#include "mpi_nbr_shim.h"

/* Opt-out build. Correct ONLY on a library verified to deliver the ordering of
 * MPI-4.0 section 8.6 for duplicate neighbour edges. Measured counter-examples
 * exist in BOTH directions -- see the header. Do not define this unless
 * tests/mpi_halo/halo_convention_test passes on the target machine. */
#ifdef SPARC_MPI_NBR_NO_SHIM

int SPARC_Ineighbor_alltoallv(
        const void *sendbuf, const int sendcounts[], const int sdispls[],
        MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
        const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
        MPI_Request *request)
{
    return MPI_Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                                   recvbuf, recvcounts, rdispls, recvtype,
                                   comm, request);
}

#else /* !SPARC_MPI_NBR_NO_SHIM -- the default */

/* Base for the per-edge point-to-point tags. These topology communicators
 * carry collectives and no user point-to-point, and collectives live in a
 * separate context, so 0 is safe; kept as a knob in case that ever changes. */
#define SPARC_NBR_TAG_BASE 0

/**
 * @brief Fetch the neighbour lists and note whether the topology is Cartesian.
 *
 * On success *srcs and *dsts are owned by the caller. They alias one another
 * for MPI_CART and MPI_GRAPH, where one ordered list serves both directions,
 * so the caller frees *srcs and frees *dsts only if it differs.
 *
 * @return 1 on success, 0 if comm carries no topology this routine handles.
 */
static int sparc_nbr_lists(MPI_Comm comm, int *indeg, int *outdeg,
                           int **srcs, int **dsts, int *is_cart)
{
    int status;
    MPI_Topo_test(comm, &status);

    if (status == MPI_CART) {
        int ndims;
        MPI_Cartdim_get(comm, &ndims);
        int n = 2 * ndims;
        int *nbrs = (int *) malloc((n > 0 ? n : 1) * sizeof(int));
        if (nbrs == NULL) return 0;
        /* MPI orders the Cartesian neighbours as, for each dimension, the one
         * in the negative direction then the one in the positive direction --
         * exactly the (rank_source, rank_dest) pair MPI_Cart_shift returns. */
        for (int d = 0; d < ndims; d++)
            MPI_Cart_shift(comm, d, 1, &nbrs[2 * d], &nbrs[2 * d + 1]);
        *indeg = *outdeg = n;
        *srcs = *dsts = nbrs;
        *is_cart = 1;
        return 1;
    }

    if (status == MPI_DIST_GRAPH) {
        int weighted;
        MPI_Dist_graph_neighbors_count(comm, indeg, outdeg, &weighted);
        int *s = (int *) malloc((*indeg  > 0 ? *indeg  : 1) * sizeof(int));
        int *d = (int *) malloc((*outdeg > 0 ? *outdeg : 1) * sizeof(int));
        if (s == NULL || d == NULL) { free(s); free(d); return 0; }
        MPI_Dist_graph_neighbors(comm, *indeg, s, MPI_UNWEIGHTED,
                                 *outdeg, d, MPI_UNWEIGHTED);
        *srcs = s;
        *dsts = d;
        *is_cart = 0;
        return 1;
    }

    if (status == MPI_GRAPH) {
        int rank, n;
        MPI_Comm_rank(comm, &rank);
        MPI_Graph_neighbors_count(comm, rank, &n);
        int *nbrs = (int *) malloc((n > 0 ? n : 1) * sizeof(int));
        if (nbrs == NULL) return 0;
        MPI_Graph_neighbors(comm, rank, n, nbrs);
        *indeg = *outdeg = n;
        *srcs = *dsts = nbrs;
        *is_cart = 0;
        return 1;
    }

    return 0;
}

/**
 * @brief Does any rank appear twice in list[0..n-1]?
 *
 * MPI_PROC_NULL is skipped: sends to and receives from it are no-ops, so their
 * relative order can never change what is delivered. SPARC's 26-neighbour
 * lists are full of them along non-periodic directions.
 */
static int sparc_nbr_has_duplicate(const int *list, int n)
{
    for (int i = 0; i < n; i++) {
        if (list[i] == MPI_PROC_NULL) continue;
        for (int j = 0; j < i; j++)
            if (list[j] == list[i]) return 1;
    }
    return 0;
}

/**
 * @brief How many earlier entries of list[0..i-1] equal list[i].
 *
 * The edge's multiplicity index, which is what the standard uses to pair
 * repeated edges on a graph topology: the k-th edge to a process matches the
 * k-th edge from it.
 */
static int sparc_nbr_multiplicity(const int *list, int i)
{
    int m = 0;
    for (int j = 0; j < i; j++)
        if (list[j] == list[i]) m++;
    return m;
}

int SPARC_Ineighbor_alltoallv(
        const void *sendbuf, const int sendcounts[], const int sdispls[],
        MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
        const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
        MPI_Request *request)
{
    int indeg, outdeg, is_cart, *srcs, *dsts;
    if (!sparc_nbr_lists(comm, &indeg, &outdeg, &srcs, &dsts, &is_cart)) {
        /* No topology we recognise -- leave it to the library. */
        return MPI_Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                                       recvbuf, recvcounts, rdispls, recvtype,
                                       comm, request);
    }

    /* Fast path. With no repeated rank the pairing is forced by rank matching,
     * every implementation agrees, and there is nothing to work around -- so
     * hand it back to the library and keep its (possibly offloaded) path and
     * the caller's compute/communication overlap. This is the ordinary
     * all-dims>=3 decomposition. */
    if (!sparc_nbr_has_duplicate(srcs, indeg) &&
        !sparc_nbr_has_duplicate(dsts, outdeg)) {
        free(srcs);
        if (dsts != srcs) free(dsts);
        return MPI_Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                                       recvbuf, recvcounts, rdispls, recvtype,
                                       comm, request);
    }

    MPI_Aint slb, sextent, rlb, rextent;
    MPI_Type_get_extent(sendtype, &slb, &sextent);
    MPI_Type_get_extent(recvtype, &rlb, &rextent);

    int nreq = indeg + outdeg;
    MPI_Request *reqs = (MPI_Request *) malloc((nreq > 0 ? nreq : 1) * sizeof(MPI_Request));
    if (reqs == NULL) {
        free(srcs);
        if (dsts != srcs) free(dsts);
        return MPI_ERR_NO_MEM;
    }

    /* Receives first, so the sends always have a matching post to hit. */
    for (int l = 0; l < indeg; l++) {
        /* MPI_CART: our slot l holds the neighbour in direction l, and we are
         * that neighbour's direction l^1, so it sends us its block l^1. These
         * are the tags of Example 8.10.
         * Graph: pair by multiplicity index. */
        int tag = is_cart ? (l ^ 1) : sparc_nbr_multiplicity(srcs, l);
        MPI_Irecv((char *) recvbuf + (MPI_Aint) rdispls[l] * rextent,
                  recvcounts[l], recvtype, srcs[l],
                  SPARC_NBR_TAG_BASE + tag, comm, &reqs[l]);
    }

    for (int k = 0; k < outdeg; k++) {
        int tag = is_cart ? k : sparc_nbr_multiplicity(dsts, k);
        MPI_Isend((const char *) sendbuf + (MPI_Aint) sdispls[k] * sextent,
                  sendcounts[k], sendtype, dsts[k],
                  SPARC_NBR_TAG_BASE + tag, comm, &reqs[indeg + k]);
    }

    int err = MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    free(reqs);
    free(srcs);
    if (dsts != srcs) free(dsts);

    /* The transfer is already complete. A null request makes the caller's
     * MPI_Wait() a no-op, which keeps the call sites unchanged. */
    *request = MPI_REQUEST_NULL;
    return err;
}

#endif /* SPARC_MPI_NBR_NO_SHIM */
