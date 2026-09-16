/*
 * nbr_match_probe.c -- probe how the local MPI matches neighbour edges in
 * MPI_Ineighbor_alltoallv, for the two topology families SPARC uses.
 *
 * For every receive slot l this reports which of the peer's send blocks b
 * actually landed there, for EVERY partner (not just self-edges), and
 * classifies the resulting map l <- b(l) against the three candidates:
 *
 *   b(l) == l^1      "xor1"      MPI-4.1 Ex. 8.10, required for MPI_CART
 *   b(l) == l        "identity"  what SPARC's halo packing assumes
 *   b(l) == n-1-l    "reversed"  MPICH's current behaviour, pmodels/mpich#7945
 *
 * Duplicate edges - the case the MPICH bug is about - appear whenever
 * dims[d] <= 2 and periods[d] == 1, because the -1 and +1 neighbours along d
 * are then the same rank. With distinct ranks the pairing is forced by rank
 * matching and every implementation agrees.
 *
 * Build: mpicc -std=gnu11 -O0 -o nbr_match_probe nbr_match_probe.c
 * Run:   srun -n 8 ./nbr_match_probe 2 2 2     # explicit grid
 *        srun -n 2 ./nbr_match_probe           # let MPI_Dims_create choose
 * The three optional args are dims[0] dims[1] dims[2]; their product must
 * equal the number of ranks.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NNBR 6   /* -x +x -y +y -z +z */

/* Encode "rank r, send block b" in one int. */
static int  enc(int r, int b) { return r * 1000 + b; }
static int  dec_rank(int v)   { return v / 1000; }
static int  dec_blk(int v)    { return v % 1000; }

static const char *nbrname(int m)
{
    static const char *n[NNBR] = {"-x", "+x", "-y", "+y", "-z", "+z"};
    return n[m];
}

static void probe(MPI_Comm nbrcomm, const int *nbrs, const char *label, int myrank)
{
    int sbuf[NNBR], rbuf[NNBR], sc[NNBR], sd[NNBR], rc[NNBR], rd[NNBR];
    for (int i = 0; i < NNBR; i++) {
        sbuf[i] = enc(myrank, i); sc[i] = 1; sd[i] = i;
        rbuf[i] = -1;             rc[i] = 1; rd[i] = i;
    }

    MPI_Request req;
    MPI_Ineighbor_alltoallv(sbuf, sc, sd, MPI_INT, rbuf, rc, rd, MPI_INT, nbrcomm, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    if (myrank != 0) return;

    int is_id = 1, is_xor = 1, is_rev = 1, ndup = 0;
    for (int l = 0; l < NNBR; l++)
        for (int m = 0; m < l; m++)
            if (nbrs[l] == nbrs[m]) { ndup++; break; }

    printf("  %-11s slot  expect-peer  got(peer,block)  needed(xor1)\n", label);
    for (int l = 0; l < NNBR; l++) {
        int b = dec_blk(rbuf[l]), r = dec_rank(rbuf[l]);
        printf("  %-11s %2d %s   rank %-4d   (%d,%d)%*s      %d\n",
               "", l, nbrname(l), nbrs[l], r, b, 6, "", l ^ 1);
        if (b != l)            is_id  = 0;
        if (b != (l ^ 1))      is_xor = 0;
        if (b != NNBR - 1 - l) is_rev = 0;
        if (r != nbrs[l])      { is_id = is_xor = is_rev = 0; }
    }
    printf("  %-11s => %s%s%s%s   (%d slot(s) share a partner)\n\n", "",
           is_id  ? "identity " : "",
           is_xor ? "xor1 "     : "",
           is_rev ? "reversed " : "",
           (!is_id && !is_xor && !is_rev) ? "OTHER (cross-mixed)" : "",
           ndup);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[3] = {0, 0, 0};
    if (argc >= 4) {
        for (int d = 0; d < 3; d++) dims[d] = atoi(argv[d + 1]);
        if (dims[0] * dims[1] * dims[2] != size) {
            if (rank == 0)
                fprintf(stderr, "dims %d x %d x %d != %d ranks\n",
                        dims[0], dims[1], dims[2], size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        MPI_Dims_create(size, 3, dims);
    }
    int periods[3] = {1, 1, 1};

    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);

    int coords[3], nbrs[NNBR];
    MPI_Cart_coords(cart, rank, 3, coords);
    int off[NNBR][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
    for (int m = 0; m < NNBR; m++) {
        int c[3];
        for (int d = 0; d < 3; d++) c[d] = (coords[d] + off[m][d] + dims[d]) % dims[d];
        MPI_Cart_rank(cart, c, &nbrs[m]);
    }

    if (rank == 0) {
        char ver[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int vlen = 0;
        MPI_Get_library_version(ver, &vlen);
        for (char *p = ver; *p; p++) if (*p == '\n') { *p = 0; break; }
        printf("MPI: %s\n", ver);
        printf("grid: %d x %d x %d, periods 1 1 1, %d rank(s)\n",
               dims[0], dims[1], dims[2], size);
        printf("degenerate (duplicate-edge) dims:");
        int any = 0;
        for (int d = 0; d < 3; d++) if (dims[d] < 3) { printf(" %d(size=%d)", d, dims[d]); any = 1; }
        printf("%s\n\n", any ? "" : " none");
    }

    /* 1. Cartesian: MPI orders the neighbours -x +x -y +y -z +z itself. */
    probe(cart, nbrs, "cartesian", rank);

    /* 2. Distributed graph over the same offsets, one array for srcs and dsts
     *    -- exactly how SPARC builds comm_dist_graph_psi/phi. */
    MPI_Comm dg;
    MPI_Dist_graph_create_adjacent(cart, NNBR, nbrs, (int *)MPI_UNWEIGHTED,
                                   NNBR, nbrs, (int *)MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, 0, &dg);
    probe(dg, nbrs, "dist-graph", rank);

    if (rank == 0)
        printf("SPARC needs 'xor1' where it does NOT swap (non-degenerate dims)\n"
               "and 'identity' where it DOES swap (dims[d]<3 && periods[d]).\n");

    MPI_Comm_free(&dg);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
