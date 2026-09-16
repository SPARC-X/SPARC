/**
 * @file    mpi_nbr_shim.h
 * @brief   Spec-conforming replacement for MPI_Ineighbor_alltoallv, for MPI
 *          stacks that mismatch duplicate neighbour edges.
 *
 * The rules SPARC relies on
 * -------------------------
 * A neighbour list contains the same rank twice whenever
 *
 *     dims[d] <= 2 && periods[d] == 1
 *
 * because the -1 and +1 neighbours along d are then the same process (or self).
 * Rank matching no longer decides which of the peer's blocks lands in which of
 * our slots, so the pairing falls to the convention the standard lays down:
 *
 *   MPI-4.0 / 4.1 section 8.6, Example 8.10 -- MPI_CART:
 *       block s of the sender lands in block s^1 of the receiver, for every
 *       dimension, degenerate or not. The accompanying advice to implementors
 *       names the periods[d]==1 && dims[d]==1-or-2 case and requires matched
 *       posting order or per-direction tags.
 *   MPI-4.0 / 4.1 section 8.6 -- MPI_GRAPH / MPI_DIST_GRAPH:
 *       the k-th edge to a process matches the k-th edge from it
 *       (multiplicity-index order).
 *
 * Both were clarified as errata against MPI-3.1; see MPI-4.0 Annex B.1.1
 * item 1, which applies them retroactively to MPI-3.1 sections 7.6.1 and 7.6.2.
 * SPARC's halo pack/unpack is written against exactly these two rules and
 * needs no compensation of its own on a conforming stack.
 *
 * Why the shim exists
 * -------------------
 * MPICH reverses the whole receive list unconditionally, which coincides with
 * the required rule only while at most one duplicate pair shares a partner
 * rank, and is wrong for graph topologies outright. Affects MPICH 4.x/5.x and
 * derivatives including Cray MPICH; upstream fix in pmodels/mpich#7945.
 * Measured on Tuolumne (Cray MPICH 9.0.1.498) with utils/mpicheck:
 *
 *   3x3x3  no duplicate edge           -> s^1        conforming
 *   2x3x3  one duplicate pair          -> s^1        conforming
 *   1x3x3  one duplicate pair (self)   -> s^1        conforming
 *   1x2x2  one pair per partner        -> s^1        conforming
 *   1x1x2  two unit periodic dims      -> s -> 3-s   WRONG, cross-dimension
 *   1x1x1  three unit periodic dims    -> s -> 5-s   WRONG, cross-dimension
 *
 * and the dist-graph row is wrong wherever an edge repeats.
 *
 * What this provides
 * ------------------
 * SPARC_Ineighbor_alltoallv() has the same prototype as the MPI call and is
 * used by every halo call site. It is ALWAYS ACTIVE. Measured with
 * tests/mpi_halo/halo_convention_test, three implementations give two
 * different answers for duplicate edges and one of them is not even
 * self-consistent, so the ordering cannot be left to the library:
 *
 *   Cray MPICH 9.0.1.498  swaps      (s -> s^1), and matches NEITHER
 *                         convention with two or more unit dimensions
 *   MVAPICH2 2.3.7        list order (s -> s), consistently
 *   Open MPI 4.1.2        list order (s -> s), consistently
 *
 * tests/mpi_halo/halo_convention_test reports PASS on one and FAIL on the
 * other for the same source, in both directions. So SPARC does the pairing
 * itself, and the answer stops depending on which machine it runs on.
 *
 * When the neighbour list holds no repeated rank the pairing is forced by rank
 * matching, every implementation agrees, and the wrapper forwards verbatim to
 * MPI_Ineighbor_alltoallv -- so the ordinary all-dims>=3 decomposition keeps
 * the library's own path and its compute/communication overlap. Only a
 * degenerate decomposition takes the point-to-point path, which implements the
 * two rules above, for MPI_CART with the very tags Example 8.10 uses.
 *
 * Caveat on that path: it completes the transfer before returning and sets
 * *request = MPI_REQUEST_NULL, because one MPI_Request cannot stand for the
 * 2*n point-to-point operations. The caller's MPI_Wait() is then a no-op
 * (waiting on a null request returns immediately). Correctness is unaffected;
 * the overlap is lost for degenerate decompositions only.
 *
 * Building with -DSPARC_MPI_NBR_NO_SHIM restores the bare library call. That
 * is correct only where halo_convention_test passes on the target machine, and
 * it fails silently -- with wrong energies, not an error -- where it does not.
 *
 * @author  Alfredo Metere <alfredo.metere@metereconsulting.com>
 */

#ifndef MPI_NBR_SHIM_H
#define MPI_NBR_SHIM_H

#include <mpi.h>

/**
 * @brief Same prototype and semantics as MPI_Ineighbor_alltoallv.
 */
int SPARC_Ineighbor_alltoallv(
        const void *sendbuf, const int sendcounts[], const int sdispls[],
        MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
        const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
        MPI_Request *request);

#endif // MPI_NBR_SHIM_H
