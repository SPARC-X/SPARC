/**
 * @file    halo_convention_test.c
 * @brief   Regression test for the Cartesian halo-exchange block convention.
 *
 * What it pins
 * ------------
 * SPARC's orthogonal-cell Laplacian and gradient exchange their six face slabs
 * with a single MPI_Ineighbor_alltoallv on the Cartesian domain communicator.
 * Send block k carries the face touching neighbour k; receive slot l is written
 * into the halo slab on side l. For that to produce the right ghost values, the
 * neighbour's block that lands in our slot l must be its block l^1 -- its face
 * on the side that touches us.
 *
 * The MPI standard guarantees exactly that for a Cartesian communicator, in
 * every dimension, degenerate or not: MPI-4.0 and MPI-4.1 section 8.6,
 * Example 8.10, whose advice to implementors names the
 * periods[d]==1 && dims[d]==1-or-2 case explicitly. (Clarified as errata
 * against MPI-3.1; MPI-4.0 Annex B.1.1 item 1.)
 *
 * Where dims[d] >= 3, the pairing is forced by rank matching and every
 * implementation agrees, so the convention is invisible. Where dims[d] <= 2 and
 * direction d is periodic, the -1 and +1 neighbours are the same rank, rank
 * matching no longer decides, and only the convention above does. SPARC used to
 * mirror the packed face in that case, which inverts the halo; this test fails
 * on that packing and passes on the current one.
 *
 * How it checks
 * -------------
 * A periodic global field with a distinct value per cell is distributed over
 * the process grid, the exchange is run, and every ghost cell is compared with
 * the field value at its wrapped global coordinate. That invariant is
 * independent of how the exchange is implemented, so the test cannot drift with
 * the code it guards.
 *
 * Three exchanges run per invocation. The first is SPARC's real path, through
 * SPARC_Ineighbor_alltoallv, and is the only one the exit status depends on: it
 * must pass on every machine. The other two use the bare library call with the
 * current and the historical packing, and report which convention this library
 * follows. Those two are expected to disagree between machines -- Cray MPICH
 * applies the swap, MVAPICH2 2.3.7 uses list order -- and that disagreement is
 * precisely why SPARC does the pairing itself rather than delegating it.
 *
 * Build and run
 * -------------
 *   make
 *   mpirun -n 8  ./halo_convention_test          # MPI_Dims_create picks a shape
 *   mpirun -n 18 ./halo_convention_test 2 3 3    # explicit shape
 *
 * Exit status 0 if the current packing reproduces the periodic field exactly.
 *
 * @author  Alfredo Metere <alfredo.metere@metereconsulting.com>
 *          Metere Consulting, LLC
 *
 * Copyright (c) 2026 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi_nbr_shim.h"

#define NDIM   3
#define NNBR   6      /* -x +x -y +y -z +z, the MPI_CART neighbour order */

/* SPARC's block decomposition: the first (n % p) blocks get one extra cell. */
static int block_len(int n, int p, int coord)
{
    return n / p + (coord < n % p ? 1 : 0);
}

static int block_start(int n, int p, int coord)
{
    return (n / p) * coord + (coord < n % p ? coord : n % p);
}

/* Distinct value per global cell, so a misplaced slab can never alias. */
static double field(int gi, int gj, int gk, const int *N)
{
    return 1.0 + gi + (double) N[0] * (gj + (double) N[1] * gk);
}

static int wrap(int i, int n)
{
    return ((i % n) + n) % n;
}

/**
 * @brief One exchange, then verify every ghost cell against the global field.
 *
 * @param mirror  non-zero to reproduce the historical mirrored packing
 * @return number of ghost cells that came out wrong on this rank
 */
static long run_exchange(MPI_Comm cart, const int *dims, const int *coords,
                         const int *N, int FDn, int mirror, int use_wrapper)
{
    const int DMnx = block_len(N[0], dims[0], coords[0]);
    const int DMny = block_len(N[1], dims[1], coords[1]);
    const int DMnz = block_len(N[2], dims[2], coords[2]);
    const int x0   = block_start(N[0], dims[0], coords[0]);
    const int y0   = block_start(N[1], dims[1], coords[1]);
    const int z0   = block_start(N[2], dims[2], coords[2]);

    const int DMnx_in = DMnx - FDn, DMny_in = DMny - FDn, DMnz_in = DMnz - FDn;
    const int DMnx_out = DMnx + FDn, DMny_out = DMny + FDn, DMnz_out = DMnz + FDn;
    const int DMnx_ex = DMnx_out + FDn, DMny_ex = DMny_out + FDn, DMnz_ex = DMnz_out + FDn;

    /* Send regions: block k is the face touching neighbour k. */
    const int istart[NNBR] = {0,    DMnx_in, 0,    0,       0,    0};
    const int   iend[NNBR] = {FDn,  DMnx,    DMnx, DMnx,    DMnx, DMnx};
    const int jstart[NNBR] = {0,    0,       0,    DMny_in, 0,    0};
    const int   jend[NNBR] = {DMny, DMny,    FDn,  DMny,    DMny, DMny};
    const int kstart[NNBR] = {0,    0,       0,    0,       0,    DMnz_in};
    const int   kend[NNBR] = {DMnz, DMnz,    DMnz, DMnz,    FDn,  DMnz};

    /* Receive regions: slot l is the halo slab on side l. */
    const int istart_in[NNBR] = {0,        DMnx_out, FDn,      FDn,      FDn,      FDn};
    const int   iend_in[NNBR] = {FDn,      DMnx_ex,  DMnx_out, DMnx_out, DMnx_out, DMnx_out};
    const int jstart_in[NNBR] = {FDn,      FDn,      0,        DMny_out, FDn,      FDn};
    const int   jend_in[NNBR] = {DMny_out, DMny_out, FDn,      DMny_ex,  DMny_out, DMny_out};
    const int kstart_in[NNBR] = {FDn,      FDn,      FDn,      FDn,      0,        DMnz_out};
    const int   kend_in[NNBR] = {DMnz_out, DMnz_out, DMnz_out, DMnz_out, FDn,      DMnz_ex};

    const int DMnd_ex = DMnx_ex * DMny_ex * DMnz_ex;
    double *x    = (double *) malloc((size_t) DMnx * DMny * DMnz * sizeof(double));
    double *x_ex = (double *) calloc((size_t) DMnd_ex, sizeof(double));
    if (x == NULL || x_ex == NULL) { free(x); free(x_ex); return -1; }

    for (int k = 0; k < DMnz; k++)
        for (int j = 0; j < DMny; j++)
            for (int i = 0; i < DMnx; i++)
                x[k * DMnx * DMny + j * DMnx + i] = field(x0 + i, y0 + j, z0 + k, N);

    int sendcounts[NNBR], sdispls[NNBR], recvcounts[NNBR], rdispls[NNBR];
    sendcounts[0] = sendcounts[1] = recvcounts[0] = recvcounts[1] = FDn * DMny * DMnz;
    sendcounts[2] = sendcounts[3] = recvcounts[2] = recvcounts[3] = FDn * DMnx * DMnz;
    sendcounts[4] = sendcounts[5] = recvcounts[4] = recvcounts[5] = FDn * DMnx * DMny;
    sdispls[0] = rdispls[0] = 0;
    for (int n = 1; n < NNBR; n++)
        sdispls[n] = rdispls[n] = sdispls[n - 1] + sendcounts[n - 1];

    const int nd = sdispls[NNBR - 1] + sendcounts[NNBR - 1];
    double *sbuf = (double *) malloc((size_t) nd * sizeof(double));
    double *rbuf = (double *) calloc((size_t) nd, sizeof(double));
    if (sbuf == NULL || rbuf == NULL) { free(x); free(x_ex); free(sbuf); free(rbuf); return -1; }

    int periods_get[NDIM], dims_get[NDIM], coords_get[NDIM];
    MPI_Cart_get(cart, NDIM, dims_get, periods_get, coords_get);

    long count = 0;
    for (int nbr = 0; nbr < NNBR; nbr++) {
        /* The historical packing mirrored the face whenever the dimension was
         * periodic with fewer than three processes. */
        const int b = mirror
            ? nbr + (1 - 2 * (nbr % 2))
                  * (int) (dims_get[nbr / 2] < 3 && periods_get[nbr / 2])
            : nbr;
        for (int k = kstart[b]; k < kend[b]; k++)
            for (int j = jstart[b]; j < jend[b]; j++)
                for (int i = istart[b]; i < iend[b]; i++)
                    sbuf[count++] = x[k * DMnx * DMny + j * DMnx + i];
    }

    MPI_Request req;
    if (use_wrapper)
        SPARC_Ineighbor_alltoallv(sbuf, sendcounts, sdispls, MPI_DOUBLE,
                                  rbuf, recvcounts, rdispls, MPI_DOUBLE, cart, &req);
    else
        MPI_Ineighbor_alltoallv(sbuf, sendcounts, sdispls, MPI_DOUBLE,
                                rbuf, recvcounts, rdispls, MPI_DOUBLE, cart, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    count = 0;
    for (int slot = 0; slot < NNBR; slot++)
        for (int k = kstart_in[slot]; k < kend_in[slot]; k++)
            for (int j = jstart_in[slot]; j < jend_in[slot]; j++)
                for (int i = istart_in[slot]; i < iend_in[slot]; i++)
                    x_ex[k * DMnx_ex * DMny_ex + j * DMnx_ex + i] = rbuf[count++];

    /* A ghost cell is just the periodic continuation of the global field. */
    long bad = 0;
    for (int slot = 0; slot < NNBR; slot++) {
        for (int kp = kstart_in[slot]; kp < kend_in[slot]; kp++) {
            for (int jp = jstart_in[slot]; jp < jend_in[slot]; jp++) {
                for (int ip = istart_in[slot]; ip < iend_in[slot]; ip++) {
                    const double want = field(wrap(x0 + ip - FDn, N[0]),
                                              wrap(y0 + jp - FDn, N[1]),
                                              wrap(z0 + kp - FDn, N[2]), N);
                    const double got = x_ex[kp * DMnx_ex * DMny_ex + jp * DMnx_ex + ip];
                    if (got != want) bad++;
                }
            }
        }
    }

    free(x); free(x_ex); free(sbuf); free(rbuf);
    return bad;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int FDn = 6;   /* half order; SPARC's default FD_ORDER is 12 */

    int dims[NDIM] = {0, 0, 0};
    if (argc >= 1 + NDIM) {
        for (int d = 0; d < NDIM; d++) dims[d] = atoi(argv[d + 1]);
        if (dims[0] * dims[1] * dims[2] != size) {
            if (rank == 0)
                fprintf(stderr, "dims %d x %d x %d does not match %d ranks\n",
                        dims[0], dims[1], dims[2], size);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    } else {
        MPI_Dims_create(size, NDIM, dims);
    }

    /* Every local block needs at least FDn cells per direction for the face
     * slabs to lie inside it, which is also what SPARC requires. */
    int N[NDIM];
    for (int d = 0; d < NDIM; d++) N[d] = 2 * FDn * dims[d];

    int periods[NDIM] = {1, 1, 1};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, NDIM, dims, periods, 0, &cart);
    int coords[NDIM];
    MPI_Cart_coords(cart, rank, NDIM, coords);

    long bad_sparc = run_exchange(cart, dims, coords, N, FDn, 0, 1);
    long bad_cur   = run_exchange(cart, dims, coords, N, FDn, 0, 0);
    long bad_mir   = run_exchange(cart, dims, coords, N, FDn, 1, 0);
    if (bad_sparc < 0 || bad_cur < 0 || bad_mir < 0) {
        if (rank == 0) fprintf(stderr, "allocation failure\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    long tot_sparc = 0, tot_cur = 0, tot_mir = 0;
    MPI_Reduce(&bad_sparc, &tot_sparc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bad_cur,   &tot_cur,   1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bad_mir,   &tot_mir,   1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        int ndegen = 0, nunit = 0;
        for (int d = 0; d < NDIM; d++) {
            if (dims[d] < 3) ndegen++;
            if (dims[d] == 1) nunit++;
        }
        char ver[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int vlen = 0;
        MPI_Get_library_version(ver, &vlen);
        for (char *q = ver; *q; q++) if (*q == '\n') { *q = 0; break; }

        printf("MPI: %s\n", ver);
        printf("grid %d x %d x %d over %d rank(s), global %d x %d x %d, FDn %d\n",
               dims[0], dims[1], dims[2], size, N[0], N[1], N[2], FDn);
        printf("  directions whose two neighbours are the same rank : %d\n\n", ndegen);

        printf("  SPARC halo path (SPARC_Ineighbor_alltoallv) : %s (%ld wrong ghost cells)\n",
               tot_sparc == 0 ? "PASS" : "FAIL", tot_sparc);
        printf("\n  Diagnostic -- the bare library call, which SPARC no longer relies on:\n");
        printf("    unmirrored packing : %s (%ld)\n", tot_cur == 0 ? "pass" : "fail", tot_cur);
        printf("    mirrored packing   : %s (%ld)\n", tot_mir == 0 ? "pass" : "fail", tot_mir);

        if (ndegen == 0) {
            printf("\n  No direction has duplicate neighbours, so rank matching alone fixes\n"
                   "  the pairing and every library agrees. Run a shape with a periodic\n"
                   "  dimension of size 1 or 2 to exercise the convention.\n");
        } else if (tot_cur == 0 && tot_mir != 0) {
            printf("\n  This library applies the pairwise swap of MPI-4.0 section 8.6,\n"
                   "  Example 8.10, for duplicate neighbour edges.\n");
        } else if (tot_mir == 0 && tot_cur != 0) {
            printf("\n  This library pairs duplicate neighbour edges in list order, the\n"
                   "  pre-errata convention. It disagrees with libraries that follow\n"
                   "  Example 8.10 -- which is why SPARC does the pairing itself.\n");
        } else if (tot_cur != 0 && tot_mir != 0) {
            printf("\n  Neither packing reproduces the field, so this library matches no\n"
                   "  self-consistent convention on this shape. Expected of MPICH 4.x/5.x\n"
                   "  and derivatives with %d periodic unit dimensions; see\n"
                   "  pmodels/mpich#7945 and utils/mpicheck/nbr_match_probe.\n", nunit);
        }

        rc = (tot_sparc == 0) ? 0 : 1;
        printf("\n%s\n", rc == 0 ? "No Errors" : "FAILED");
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return rc;
}
