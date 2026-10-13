/***
 * @file    sq.c
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
#include <time.h>
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "sq.h"
#include "occupation.h"
#include "parallelization.h"
#include "tools.h"
#include "sqNlocVecRoutines.h"
#include "sqParallelization.h"
#include "sqEnergy.h"
#include "tools.h"
#include "electrostatics.h"
#include "sqHighTExactExchange.h"
#include "kroneckerLaplacian.h"

#ifdef SPARCX_ACCEL
#include "accel.h"
#endif

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)

static void LanczosAlgorithm_fused(SPARC_OBJ *pSPARC, int nd, int spn_i, double *lambda_min, double *lambda_max);
static void LanczosAlgorithm_classic(SPARC_OBJ *pSPARC, int nd, int spn_i, double *lambda_min, double *lambda_max);

/**
 * SQ TODO list:
 * 1. Newtown Raphson algorithm for fermi level
 * 2. Energy and forces correction
 * 3. Always storage of nolocal projectors
 * 4. Always use spherical Rcut region
 * 5. Function for cell replication
 */


/**
 * @brief   Gauss Quadrature method
 *
 * @param scf_iter  The scf iteration counter
 */
void GaussQuadrature(SPARC_OBJ *pSPARC, int spn_i) {
    if (pSPARC->pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int nd, rank;
    int DMnx, DMny, DMnz, DMnd;
    double lambda_min, lambda_max;
#ifdef DEBUG
    double time1, time2;
#endif
    SQ_OBJ  *pSQ = pSPARC->pSQ;

    DMnx = pSQ->DMnx_SQ;
    DMny = pSQ->DMny_SQ;
    DMnz = pSQ->DMnz_SQ;
    DMnd = pSQ->DMnd_SQ;

    (void)DMnx;
    (void)DMny;
    (void)DMnz;

    MPI_Comm_rank(pSQ->dmcomm_SQ, & rank);
    //////////////////////////////////////////////////////////////////////

    #ifdef DEBUG
        time2 = MPI_Wtime();
    #endif

    #ifdef SPARCX_ACCEL
	if (pSPARC->useACCEL == 1 && pSPARC->cell_typ == 0 && pSPARC->usefock <=1)
	{
		ACCEL_SQ_LanczosAlgorithm_gauss(pSPARC, DMnx, DMny, DMnz, &lambda_min, &lambda_max);
	}
	else
	#endif // SPARCX_ACCEL
	{
    	for (nd = 0; nd < DMnd; nd ++) {
            LanczosAlgorithm_gauss(pSPARC, nd, spn_i, &lambda_min, &lambda_max);
        	pSQ->mineig[nd+spn_i*DMnd]  = lambda_min;
        	pSQ->maxeig[nd+spn_i*DMnd]  = lambda_max;
    	}
	}

    // This barrier fixed the severe slowdown of MPI communication on hive
    MPI_Barrier(pSQ->dmcomm_SQ);

    #ifdef DEBUG
        time1 = MPI_Wtime();
        if(!rank) printf("Rank %d finished Lanczos taking %.3f ms\n",rank, (time1-time2)*1e3);
    #endif
}


/**
 * @brief   Lanczos algorithm for Gauss method on grid point nd (start vector: the unit vector at the point). The
 *          point's Veff (spin spn_i) is gathered once into pSQ->Veff_box and its nonlocal projectors are built
 *          from the PR-domain set (GetNonlocalProjectorsForOnePoint); then the fused sweep (static
 *          LanczosAlgorithm_fused) or, for hybrid functionals, the classical steps (LanczosAlgorithm_classic).
 *
 * @param nd            FD node index in current process domain
 * @param spn_i         local spin index
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 */
void LanczosAlgorithm_gauss(SPARC_OBJ *pSPARC, int nd, int spn_i, double *lambda_min, double *lambda_max) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int DMnx = pSQ->DMnx_SQ, DMnxny = DMnx * pSQ->DMny_SQ;
    int k = nd / DMnxny, j = (nd - k * DMnxny) / DMnx, i = nd % DMnx;    // the point's Rcut box starts at (i, j, k) of the PR domain

    // this spin's Veff on the point's Rcut box, contiguous
    restrict_to_subgrid(pSQ->Veff_PR + (size_t)spn_i * pSQ->DMnd_PR, pSQ->Veff_box, pSQ->Nx_loc, pSQ->DMnx_PR,
        pSQ->Nx_loc * pSQ->Ny_loc, pSQ->DMnx_PR * pSQ->DMny_PR,
        0, pSQ->Nx_loc - 1, 0, pSQ->Ny_loc - 1, 0, pSQ->Nz_loc - 1, i, j, k, sizeof(double));
    // the point's nonlocal projectors
    GetNonlocalProjectorsForOnePoint(pSPARC, nd);

    if (!pSPARC->usefock)
        LanczosAlgorithm_fused(pSPARC, nd, spn_i, lambda_min, lambda_max);
    else
        LanczosAlgorithm_classic(pSPARC, nd, spn_i, lambda_min, lambda_max);
}


/**
 * @brief   Classical Lanczos steps (hybrid functionals: the exact exchange potential is part of HsubTimesVec):
 *          the normalized vectors v_0 .. v_{npl-1} are written one after another into pSQ->lanczos_vec
 *          (lanczos_vec_all[nd] at high temperature), v_{k+1} = (H v_k - a_k v_k - b_{k-1} v_{k-1}) / b_k.
 */
static void LanczosAlgorithm_classic(SPARC_OBJ *pSPARC, int nd, int spn_i, double *lambda_min, double *lambda_max) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int k, Nd = pSQ->Nd_loc, npl = pSPARC->SQ_npl_g, *nloc = pSQ->nloc;
    int center = nloc[0] + nloc[1]*pSQ->Nx_loc + nloc[2]*pSQ->Nx_loc*pSQ->Ny_loc;
    double *v = (pSPARC->sqHighTFlag == 1) ? pSQ->lanczos_vec_all[nd + spn_i*pSQ->DMnd_SQ] : pSQ->lanczos_vec;
    double *aa = (double *) malloc(npl * sizeof(double)), *bb = (double *) malloc(npl * sizeof(double));
    assert(aa != NULL && bb != NULL);

    memset(v, 0, (size_t)Nd * sizeof(double));                                     // v_0: unit vector at the point
    v[center] = 1.0;
    for (k = 0; k < npl; k++) {
        double *vk = v + (size_t)k*Nd, *y = (k < npl - 1) ? vk + Nd : pSQ->lanczos_tmp;
        HsubTimesVec(pSPARC, vk, nd, y);                                            // y = H v_k
        aa[k] = cblas_ddot(Nd, vk, 1, y, 1);
        if (k == npl - 1) break;                                                    // last step: a_k only
        cblas_daxpy(Nd, -aa[k], vk, 1, y, 1);
        if (k) cblas_daxpy(Nd, -bb[k-1], vk - Nd, 1, y, 1);
        bb[k] = cblas_dnrm2(Nd, y, 1);
        cblas_dscal(Nd, 1.0 / bb[k], y, 1);                                         // v_{k+1}
    }
    TridiagEigenSolve_gauss(pSPARC, aa, bb, nd, spn_i, lambda_min, lambda_max);
    free(aa);
    free(bb);
}


/**
 * @brief   Tridiagonal eigenvalue solver for Gauss method: T = Z diag(d) Z' by dstedc (workspace preallocated in
 *          init_SQ), the eigenvalues are the nodes of point nd (ascending), the squared first components of the
 *          eigenvectors its weights; the eigenvectors are kept in pSQ->w (w_all[nd] at high temperature) for the
 *          density matrix column, and in the force step (pSQ->forceFlag) f(T) e1 = Z f(d) Z(0,:)' is formed in
 *          pSQ->eig_F.
 *
 * @param diag          Diagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain
 * @param subdiag       Subdiagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain
 * @param nd            FD node index in current process domain
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 */
void TridiagEigenSolve_gauss(SPARC_OBJ *pSPARC, double *diag, double *subdiag, int nd, int spn_i, double *lambda_min, double *lambda_max) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int j, n = pSPARC->SQ_npl_g, DMnd_SQ = pSQ->DMnd_SQ;
    double *d = pSQ->eig_d, *e = pSQ->eig_e;
    double *w = (pSPARC->sqHighTFlag == 1) ? pSQ->w_all[nd+spn_i*DMnd_SQ] : pSQ->w;
    double *gnd = pSQ->gnd[nd+spn_i*DMnd_SQ], *gwt = pSQ->gwt[nd+spn_i*DMnd_SQ];

    if (n <= 0) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) fprintf(stderr, "WARNING: TridiagEigenSolve_gauss called with n=%d. Skipping calculation.\n", n);
        *lambda_min = 0.0;
        *lambda_max = 0.0;
        return;
    }

    for (j = 0; j < n; j++) {                                                       // copies: dstedc overwrites them
        d[j] = diag[j];
        e[j] = (j < n - 1) ? subdiag[j] : 0.0;
    }
    if (LAPACKE_dstedc_work(LAPACK_COL_MAJOR, 'I', n, d, e, w, n, pSQ->eig_work, (lapack_int)pSQ->eig_lwork,
                            pSQ->eig_iwork, (lapack_int)pSQ->eig_liwork)) {
        printf(RED "ERROR: eigenvalue solver (dstedc) failed at grid point %d.\n" RESET, nd);
        exit(EXIT_FAILURE);
    }
    // eigenvector j in column j of w (w[j*n + i] = component i): nodes and weights
    for (j = 0; j < n; j++) {
        gnd[j] = d[j];
        gwt[j] = w[j*n] * w[j*n];
    }
    if (pSQ->forceFlag) {                                                           // f(T) e1 = Z f(d) Z(0,:)'
        double *C = pSQ->eig_C, *F = pSQ->eig_F;
        for (j = 0; j < n; j++)
            C[j] = w[j*n] * smearing_function(pSPARC->Beta, d[j], pSPARC->Efermi, pSPARC->elec_T_type);
        cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, w, n, C, 1, 0.0, F, 1);
    }
    *lambda_min = d[0];
    *lambda_max = d[n-1];
}


/**
 * @brief Calculate nodal Hamiltonian times vector routines (Veff: pSQ->Veff_box, nonlocal projectors:
 *        pSQ->nlocProj_pt, both of the point and spin of the current Lanczos run)
 *
 * @param pSPARC    SPARC object pointer
 * @param x         vector
 * @param nd        index of the grid point in the process domain
 * @param Hx        nodal Hamiltonian times vecotr
 */
void HsubTimesVec(SPARC_OBJ *pSPARC, const double *x, const int nd, double *Hx)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;
    // Apply Laplacian and Veff
    Lap_vec_mult_orth_SQ_kron(pSPARC, x, -0.5, Hx);
    // Apply nonlocal projector
    Vnl_vec_mult_SQ(pSPARC, pSQ->Atom_Influence_nloc_pt, pSQ->nlocProj_pt, x, Hx);

    if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0)) {
        exact_exchange_potential_SQ(pSPARC, x, nd, Hx);
    }
}


#ifdef USE_MKL
/**
 * @brief   MKL: the JIT dgemm kernels of the Laplacian on the Rcut box, created on first use and kept in pSQ:
 *          kx: H_k += a Lx X_k, ky: H_k += a X_k Ly, slabs X_k, H_k of Nx x Ny. Returns 0 if MKL cannot create
 *          them (the caller then uses the batched stages); MKL_NO_JIT (shape too large for JIT) still gives a
 *          working kernel (standard dgemm).
 */
static int SQ_jit_kernels(SQ_OBJ *pSQ, double a)
{
    if (pSQ->jit_state == 0) {
        int Nx = pSQ->Nx_loc, Ny = pSQ->Ny_loc;
        mkl_jit_status_t sx = mkl_jit_create_dgemm(&pSQ->jit_jx, MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, Nx, Ny, Nx, a, Nx, Nx, 1.0, Nx);
        mkl_jit_status_t sy = mkl_jit_create_dgemm(&pSQ->jit_jy, MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, Nx, Ny, Ny, a, Nx, Ny, 1.0, Nx);
        if (sx == MKL_JIT_ERROR || sy == MKL_JIT_ERROR) {
            if (sx != MKL_JIT_ERROR) mkl_jit_destroy(pSQ->jit_jx);
            if (sy != MKL_JIT_ERROR) mkl_jit_destroy(pSQ->jit_jy);
            pSQ->jit_state = -1;
        } else {
            pSQ->jit_kx = mkl_jit_get_dgemm_ptr(pSQ->jit_jx);
            pSQ->jit_ky = mkl_jit_get_dgemm_ptr(pSQ->jit_jy);
            pSQ->jit_a = a;
            pSQ->jit_state = 1;
        }
    }
    return pSQ->jit_state == 1 && pSQ->jit_a == a;
}
#endif


/**
 * @brief   Destroy the JIT kernels of the Laplacian, the fused Lanczos sweep and the nonlocal pass (no-op without MKL)
 */
void SQ_jit_free(SPARC_OBJ *pSPARC)
{
#ifdef USE_MKL
    SQ_OBJ *pSQ = pSPARC->pSQ;
    if (pSQ->jit_state == 1) { mkl_jit_destroy(pSQ->jit_jx); mkl_jit_destroy(pSQ->jit_jy); }
    if (pSQ->jits_state == 1) for (int q = 0; q < SQ_JITS_N; q++) mkl_jit_destroy(pSQ->jits_j[q]);
    for (int i = 0; i < pSQ->jitnl_n; i++) { mkl_jit_destroy(pSQ->jitnl[i].j[0]); mkl_jit_destroy(pSQ->jitnl[i].j[1]); }
    free(pSQ->jitnl); pSQ->jitnl = NULL;
    pSQ->jit_state = pSQ->jits_state = pSQ->jitnl_n = pSQ->jitnl_cap = 0;
#else
    (void)pSPARC;
#endif
}


/**
 * @brief   (a * Laplacian + diag(Veff)) times x on the Rcut box, in Kronecker sum form,
 *          a (Lz (x) I (x) I + I (x) Ly (x) I + I (x) I (x) Lx) x, with the dense 1D Dirichlet Laplacians
 *          (pSQ->Lap_1D_*) and the point's Veff (pSQ->Veff_box).
 *          MKL: blocked form, per z-slab the Veff product followed by the x stage (Lx X_k) and the y stage (X_k Ly)
 *          as the JIT dgemm kernels of the box, then the z stage in row chunks of the Nx Ny x Nz column (lda = Nx Ny)
 *          sized for L2. Otherwise (or if MKL cannot create the kernels): the Veff product, one dgemm along x, then
 *          dgemm per slab along y and along z (batched with MKL).
 */
void Lap_vec_mult_orth_SQ_kron(SPARC_OBJ *pSPARC, const double *x, const double a, double *Hx)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int Nx = pSQ->Nx_loc, Ny = pSQ->Ny_loc, Nz = pSQ->Nz_loc, NxNy = Nx * Ny, Nd = pSQ->Nd_loc;
    const double *vb = pSQ->Veff_box;
#ifdef USE_MKL
    if (SQ_jit_kernels(pSQ, a)) {
        size_t Rc = 16384 / Nz > 0 ? 16384 / Nz : 1;                           // z-stage row chunk: 4 arrays of Rc x Nz doubles ~ 512 KB
        for (int k = 0; k < Nz; k++) {                                          // one z-slab: Veff, x stage, y stage
            const double *xk = x + (size_t)k*NxNy, *vk = vb + (size_t)k*NxNy;
            double *hk = Hx + (size_t)k*NxNy;
            for (int i = 0; i < NxNy; i++) hk[i] = vk[i] * xk[i];
            pSQ->jit_kx(pSQ->jit_jx, pSQ->Lap_1D_x, (double *)xk, hk);          // += a Lx X_k
            pSQ->jit_ky(pSQ->jit_jy, (double *)xk, pSQ->Lap_1D_y, hk);          // += a X_k Ly
        }
        for (size_t r0 = 0; r0 < (size_t)NxNy; r0 += Rc) {                       // z stage, row chunks
            int R = (int)(r0 + Rc < (size_t)NxNy ? Rc : (size_t)NxNy - r0);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, Nz, Nz, a, x + r0, NxNy, pSQ->Lap_1D_z, Nz, 1.0, Hx + r0, NxNy);
        }
        return;
    }
#endif
    for (int i = 0; i < Nd; i++) Hx[i] = vb[i] * x[i];                                                                          // Veff x first
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny*Nz, Nx, a, pSQ->Lap_1D_x, Nx, x, Nx, 1.0, Hx, Nx);              // + along x
#ifdef USE_MKL
    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, a, x, Nx, pSQ->Lap_1D_y, Ny, 1.0, Hx, Nx, 0, 1, Nz);     // along y
    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, a, x, NxNy, pSQ->Lap_1D_z, Nz, 1.0, Hx, NxNy, 0, 1, 1); // along z
#else
    for (int s = 0; s < Nz; s++)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, a, x + (size_t)s*NxNy, Nx, pSQ->Lap_1D_y, Ny, 1.0, Hx + (size_t)s*NxNy, Nx);   // along y
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, a, x, NxNy, pSQ->Lap_1D_z, Nz, 1.0, Hx, NxNy);            // along z
#endif
}


/* ---------------------------------------------------------------------------------------------------------------------
 * Fused Lanczos (all but hybrid functionals): the Lanczos step and the normalization fused into one row-chunked sweep
 * per step, the normalized vectors never formed (implicit normalization); the nonlocal product as a pass over the atom
 * images in last-slab order (the pass walks x and Hx slab by slab) with gather/scatter by runs of consecutive points,
 * workspace and JIT kernels per image resolved once per Lanczos run (SQ_nl_plan); in the SCF a 3-slot ring buffer for
 * the Lanczos vectors. The Laplacian is Lap_vec_mult_orth_SQ_kron.
 * ------------------------------------------------------------------------------------------------------------------ */

typedef void (*SQ_jit_fn)(void *, double *, double *, double *);
enum { SQJ_UPD, SQJ_GRAM, SQJ_UPD1 };                                           // the kernels of SQ_sweep_kernels (SQ_JITS_N of them)
typedef struct { void *j[SQ_JITS_N]; SQ_jit_fn k[SQ_JITS_N]; int R; } SQ_SWEEP_JIT;   // kernels (NULL: cblas) and chunk rows

// an atom image of the point, a run of its consecutive points, and the nonlocal plan of one Lanczos run
typedef struct {
    int ndc, nproj, z1, run0, nrun;                // points, projectors, last z-slab, runs run0 .. run0 + nrun - 1
    const double *chi, *gam;                       // Chi (ndc x nproj, ld ndc), Gamma per projector
    void *jp, *je; SQ_jit_fn kp, ke;               // JIT kernels: projection, expansion (NULL: cblas)
} SQ_NL_IMG;
typedef struct { int start, len, dst; } SQ_NL_RUN; // box points start .. start + len - 1 = rows dst .. dst + len - 1 of the image
typedef struct {
    int nimg, Nz, *by_z1, *z1_off;                 // images with last slab k: by_z1[z1_off[k] .. z1_off[k+1] - 1]
    SQ_NL_IMG *img; SQ_NL_RUN *run;
    double *gam, *xrc, *alpha, *ag;                // Gamma of all types; workspace ndc_max, nproj_max (two)
} SQ_NL_PLAN;
#define SQ_JITNL_MAX 4096


/**
 * @brief   The JIT dgemm kernels of the fused sweep (MKL), created on first use and kept in pSQ: update
 *          y[chunk] -= [z1 z2][chunk] c (R x 1 x 2, lda Nd), its one-vector form (R x 1 x 1, for non-adjacent z's)
 *          and Gram g += y[chunk]' y[chunk] (1 x 1 x R). Chunk rows R = 16384 (three R-vectors in L2), at most Nd.
 *          The kernel pointers stay NULL (cblas is used) without MKL or when MKL cannot create them; MKL_NO_JIT (too
 *          large for JIT) still gives a working kernel (standard dgemm).
 */
static void SQ_sweep_kernels(SQ_OBJ *pSQ, SQ_SWEEP_JIT *jit)
{
    int Nd = pSQ->Nd_loc;
    memset(jit, 0, sizeof(*jit));
    jit->R = min(16384, Nd);
#ifdef USE_MKL
    int q;
    if (pSQ->jits_state == 0) {
        void *j[SQ_JITS_N]; mkl_jit_status_t st[SQ_JITS_N];
        pSQ->jits_R = jit->R;
        st[SQJ_UPD]  = mkl_jit_create_dgemm(&j[SQJ_UPD],  MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, jit->R, 1, 2, -1.0, Nd, 2, 1.0, Nd);
        st[SQJ_GRAM] = mkl_jit_create_dgemm(&j[SQJ_GRAM], MKL_COL_MAJOR, MKL_TRANS,   MKL_NOTRANS, 1, 1, jit->R, 1.0, Nd, Nd, 1.0, 1);
        st[SQJ_UPD1] = mkl_jit_create_dgemm(&j[SQJ_UPD1], MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, jit->R, 1, 1, -1.0, Nd, 2, 1.0, Nd);
        pSQ->jits_state = 1;
        for (q = 0; q < SQ_JITS_N; q++)
            if (st[q] == MKL_JIT_ERROR) {
                for (int u = 0; u < SQ_JITS_N; u++) if (st[u] != MKL_JIT_ERROR) mkl_jit_destroy(j[u]);
                pSQ->jits_state = -1;
                return;
            }
        for (q = 0; q < SQ_JITS_N; q++) { pSQ->jits_j[q] = j[q]; pSQ->jits_k[q] = mkl_jit_get_dgemm_ptr(j[q]); }
    }
    if (pSQ->jits_state != 1) return;
    jit->R = pSQ->jits_R;
    for (q = 0; q < SQ_JITS_N; q++) { jit->j[q] = pSQ->jits_j[q]; jit->k[q] = pSQ->jits_k[q]; }
#else
    (void)pSQ;
#endif
}


/**
 * @brief   The JIT dgemm kernels of an atom image with ndc points and nproj projectors (MKL), one entry per
 *          (ndc, nproj, dV) in pSQ->jitnl (grown on demand, at most SQ_JITNL_MAX entries): projection
 *          alpha = dV Chi' xrc (nproj x 1 x ndc), expansion xrc = Chi ag (ndc x 1 x nproj). The image's kernel
 *          pointers stay NULL (cblas is used) without MKL, when the table is full or when MKL cannot create them.
 */
static void SQ_nl_kernels(SQ_OBJ *pSQ, int ndc, int nproj, double dV, SQ_NL_IMG *im)
{
    im->jp = im->je = NULL; im->kp = im->ke = NULL;
#ifdef USE_MKL
    int i;
    for (i = 0; i < pSQ->jitnl_n; i++)
        if (pSQ->jitnl[i].ndc == ndc && pSQ->jitnl[i].nproj == nproj && pSQ->jitnl[i].dV == dV) break;
    if (i == pSQ->jitnl_n) {
        SQ_JIT_NL_ENTRY e; mkl_jit_status_t s0, s1;
        if (i >= SQ_JITNL_MAX) return;
        if (i >= pSQ->jitnl_cap) {
            int cap = pSQ->jitnl_cap ? 2 * pSQ->jitnl_cap : 256;
            SQ_JIT_NL_ENTRY *t = (SQ_JIT_NL_ENTRY *) realloc(pSQ->jitnl, (size_t)cap * sizeof(SQ_JIT_NL_ENTRY));
            if (t == NULL) return;
            pSQ->jitnl = t; pSQ->jitnl_cap = cap;
        }
        s0 = mkl_jit_create_dgemm(&e.j[0], MKL_COL_MAJOR, MKL_TRANS,   MKL_NOTRANS, nproj, 1, ndc, dV, ndc, ndc, 0.0, nproj);
        s1 = mkl_jit_create_dgemm(&e.j[1], MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, ndc, 1, nproj, 1.0, ndc, nproj, 0.0, ndc);
        if (s0 == MKL_JIT_ERROR || s1 == MKL_JIT_ERROR) {
            if (s0 != MKL_JIT_ERROR) mkl_jit_destroy(e.j[0]);
            if (s1 != MKL_JIT_ERROR) mkl_jit_destroy(e.j[1]);
            return;
        }
        e.ndc = ndc; e.nproj = nproj; e.dV = dV;
        e.k[0] = mkl_jit_get_dgemm_ptr(e.j[0]); e.k[1] = mkl_jit_get_dgemm_ptr(e.j[1]);
        pSQ->jitnl[i] = e; pSQ->jitnl_n++;
    }
    im->jp = pSQ->jitnl[i].j[0]; im->je = pSQ->jitnl[i].j[1];
    im->kp = pSQ->jitnl[i].k[0]; im->ke = pSQ->jitnl[i].k[1];
#else
    (void)pSQ; (void)ndc; (void)nproj; (void)dV;
#endif
}


/**
 * @brief   The nonlocal plan of the current point (pSQ->Atom_Influence_nloc_pt, pSQ->nlocProj_pt) for one Lanczos
 *          run: its atom images (types with projectors) with their runs of consecutive points, Gamma per projector
 *          (the l, np, m loop of Vnl_vec_mult_SQ once per type), JIT kernels and last z-slab, the images sorted by
 *          last slab, and the workspace.
 */
static void SQ_nl_plan(SPARC_OBJ *pSPARC, SQ_NL_PLAN *pl)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;
    ATOM_NLOC_INFLUENCE_OBJ *ai = pSQ->Atom_Influence_nloc_pt;
    NLOC_PROJ_OBJ *nl = pSQ->nlocProj_pt;
    int NxNy = pSQ->Nx_loc * pSQ->Ny_loc, Nz = pSQ->Nz_loc, ityp, iat, i, k, l, m, np, nimg = 0, nrun = 0, ndc_max = 1, nproj_sum = 0, nproj_max = 1, *fill;
    double *gam;

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {                                // sizes
        if (!nl[ityp].nproj) continue;
        nproj_sum += nl[ityp].nproj; nproj_max = max(nproj_max, nl[ityp].nproj);
        nimg += ai[ityp].n_atom;
        for (iat = 0; iat < ai[ityp].n_atom; iat++) {
            int ndc = ai[ityp].ndc[iat], *pos = ai[ityp].grid_pos[iat];
            ndc_max = max(ndc_max, ndc);
            for (i = 0; i < ndc; i++) nrun += (i == 0 || pos[i] != pos[i-1] + 1);
        }
    }
    pl->nimg = nimg; pl->Nz = Nz;
    pl->img = (SQ_NL_IMG *) malloc((size_t)(nimg + 1) * sizeof(SQ_NL_IMG));
    pl->run = (SQ_NL_RUN *) malloc((size_t)(nrun + 1) * sizeof(SQ_NL_RUN));
    pl->by_z1 = (int *) malloc((size_t)(nimg + 1) * sizeof(int));
    pl->z1_off = (int *) calloc(Nz + 1, sizeof(int));
    pl->gam = (double *) malloc((size_t)(nproj_sum + 1) * sizeof(double));
    pl->xrc = (double *) malloc((size_t)ndc_max * sizeof(double));
    pl->alpha = (double *) malloc((size_t)nproj_max * sizeof(double));
    pl->ag = (double *) malloc((size_t)nproj_max * sizeof(double));
    assert(pl->img != NULL && pl->run != NULL && pl->by_z1 != NULL && pl->z1_off != NULL
        && pl->gam != NULL && pl->xrc != NULL && pl->alpha != NULL && pl->ag != NULL);

    nimg = nrun = 0; gam = pl->gam;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int nproj = nl[ityp].nproj, count = 0, ldispl = 0;
        if (!nproj) continue;
        for (l = 0; l <= pSPARC->psd[ityp].lmax; l++) {                             // Gamma per projector, as in Vnl_vec_mult_SQ
            if (l == pSPARC->localPsd[ityp]) { ldispl += pSPARC->psd[ityp].ppl[l]; continue; }
            for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++)
                for (m = -l; m <= l; m++) gam[count++] = pSPARC->psd[ityp].Gamma[ldispl + np];
            ldispl += pSPARC->psd[ityp].ppl[l];
        }
        if (count != nproj) { printf(RED "ERROR: nonlocal plan: %d projectors of type %d, expected %d.\n" RESET, count, ityp, nproj); exit(EXIT_FAILURE); }
        for (iat = 0; iat < ai[ityp].n_atom; iat++) {
            SQ_NL_IMG *im = pl->img + nimg++;
            int ndc = ai[ityp].ndc[iat], *pos = ai[ityp].grid_pos[iat];
            im->ndc = ndc; im->nproj = nproj; im->chi = nl[ityp].Chi[iat]; im->gam = gam; im->run0 = nrun; im->z1 = -1;
            for (i = 0; i < ndc; i++) {
                int z = pos[i] / NxNy;
                if (z > im->z1) im->z1 = z;
                if (i == 0 || pos[i] != pos[i-1] + 1) { pl->run[nrun].start = pos[i]; pl->run[nrun].dst = i; pl->run[nrun].len = 1; nrun++; }
                else pl->run[nrun-1].len++;
            }
            im->nrun = nrun - im->run0;
            SQ_nl_kernels(pSQ, ndc, nproj, pSPARC->dV, im);
        }
        gam += nproj;
    }
    for (i = 0; i < nimg; i++) pl->z1_off[pl->img[i].z1 + 1]++;                  // images by last slab (counting sort)
    for (k = 0; k < Nz; k++) pl->z1_off[k+1] += pl->z1_off[k];
    fill = (int *) calloc((size_t)(Nz > 0 ? Nz : 1), sizeof(int));
    for (i = 0; i < nimg; i++) { int z1 = pl->img[i].z1; pl->by_z1[pl->z1_off[z1] + fill[z1]++] = i; }
    free(fill);
}


static void SQ_nl_plan_free(SQ_NL_PLAN *pl)
{
    free(pl->img); free(pl->run); free(pl->by_z1); free(pl->z1_off);
    free(pl->gam); free(pl->xrc); free(pl->alpha); free(pl->ag);
}


/**
 * @brief   Hx += Vnl_image x: gather of the image's points by runs into xrc (ndc), alpha = dV Chi' xrc,
 *          ag = Gamma alpha, xrc = Chi ag, scatter-add by runs. JIT kernels of the image when it has them,
 *          cblas otherwise.
 */
static void SQ_nl_apply(const SQ_NL_PLAN *pl, const SQ_NL_IMG *im, const double *x, double *Hx, double dV)
{
    int ndc = im->ndc, nproj = im->nproj, j;
    const SQ_NL_RUN *run = pl->run + im->run0, *rend = run + im->nrun, *ru;
    double *xrc = pl->xrc, *alpha = pl->alpha, *ag = pl->ag;
    for (ru = run; ru < rend; ru++) {                                               // gather, run by run
        const double *s = x + ru->start;
        double *d = xrc + ru->dst;
        for (j = 0; j < ru->len; j++) d[j] = s[j];
    }
    if (im->kp) im->kp(im->jp, (double *)im->chi, xrc, alpha);                     // alpha = dV Chi' xrc
    else cblas_dgemv(CblasColMajor, CblasTrans, ndc, nproj, dV, im->chi, ndc, xrc, 1, 0.0, alpha, 1);
    for (j = 0; j < nproj; j++) ag[j] = im->gam[j] * alpha[j];                      // Gamma alpha
    if (im->ke) im->ke(im->je, (double *)im->chi, ag, xrc);                         // xrc = Chi Gamma alpha
    else cblas_dgemv(CblasColMajor, CblasNoTrans, ndc, nproj, 1.0, im->chi, ndc, ag, 1, 0.0, xrc, 1);
    for (ru = run; ru < rend; ru++) {                                               // scatter-add, run by run
        const double *s = xrc + ru->dst;
        double *d = Hx + ru->start;
        for (j = 0; j < ru->len; j++) d[j] += s[j];
    }
}


/**
 * @brief   Nodal Hamiltonian times a vector with the point's nonlocal plan: the Laplacian (+ Veff), then the
 *          nonlocal product as a pass over the plan's atom images in last-slab order (SQ_nl_apply)
 */
static void HsubTimesVec_plan(SPARC_OBJ *pSPARC, const double *x, double *Hx, const SQ_NL_PLAN *pl)
{
    Lap_vec_mult_orth_SQ_kron(pSPARC, x, -0.5, Hx);                                 // (-1/2 Laplacian + Veff) x
    for (int q = 0; q < pl->nimg; q++) SQ_nl_apply(pl, pl->img + pl->by_z1[q], x, Hx, pSPARC->dV);   // + Vnl x
}


/**
 * @brief   The fused Lanczos step, row chunk by row chunk (R rows, kept in L2):
 *          y[chunk] <- y[chunk] / s (skipped when s = 0), y[chunk] -= c1 z1[chunk] + c2 z2[chunk] (c = [c1; c2];
 *          z1 = NULL: the z2 term only, z2 = NULL: no update), g += y[chunk]' y[chunk].
 *          z1 and z2 adjacent (z2 = z1 + Nd): one R x 1 x 2 product, else one per vector. JIT kernels (jit->k, when
 *          set) for the full chunks, cblas otherwise.
 */
static void LanczosSweep(int Nd, const double *z1, const double *z2, double s, const double *cc, double *y, double *g, const SQ_SWEEP_JIT *jit)
{
    int R = jit->R, merged = z1 != NULL && z2 == z1 + Nd;
    double gg = 0.0;
    for (int r0 = 0; r0 < Nd; r0 += R) {
        int m = r0 + R <= Nd ? R : Nd - r0;
        double *yc = y + r0;
        if (s != 0.0) cblas_dscal(m, 1.0 / s, yc, 1);
        if (merged) {
            if (jit->k[SQJ_UPD] && m == R) jit->k[SQJ_UPD](jit->j[SQJ_UPD], (double *)(z1 + r0), (double *)cc, yc);
            else cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, 1, 2, -1.0, z1 + r0, Nd, cc, 2, 1.0, yc, Nd);
        } else if (z2) {
            if (z1) {
                if (jit->k[SQJ_UPD1] && m == R) jit->k[SQJ_UPD1](jit->j[SQJ_UPD1], (double *)(z1 + r0), (double *)cc, yc);
                else cblas_daxpy(m, -cc[0], z1 + r0, 1, yc, 1);
            }
            if (jit->k[SQJ_UPD1] && m == R) jit->k[SQJ_UPD1](jit->j[SQJ_UPD1], (double *)(z2 + r0), (double *)(cc + 1), yc);
            else cblas_daxpy(m, -cc[1], z2 + r0, 1, yc, 1);
        }
        if (jit->k[SQJ_GRAM] && m == R) jit->k[SQJ_GRAM](jit->j[SQJ_GRAM], yc, yc, &gg);
        else gg += cblas_ddot(m, yc, 1, yc, 1);
    }
    *g = gg;
}


/**
 * @brief   Fused Lanczos on grid point nd (all but hybrid functionals). The stored vectors are V_0 (slot 0, the
 *          unit vector at the point) and Z_k = W_k (slot k+1), with V_{k+1} = Z_k / beta_k never formed. Step k:
 *            Y = H Z_{k-1} (HsubTimesVec_plan), alpha_k = Z_{k-1}' Y / beta_{k-1}^2;
 *            the sweep (LanczosSweep): W_k = Y / beta_{k-1} - Z_{k-2} beta_{k-1}/beta_{k-2} - Z_{k-1} alpha_k/beta_{k-1}
 *            in the slot of Y, with g = W_k' W_k accumulated chunk by chunk; beta_k = sqrt(g).
 *          The last step forms alpha_{npl-1} only. The Nd-vectors are streamed four times per step (read Y, Z_{k-2},
 *          Z_{k-1}, write W_k) plus the two of the alpha_k product, instead of nine. The point's nonlocal plan
 *          (SQ_nl_plan) is built once per run.
 *          SCF (ambient): the stored vectors are not used, so only Z_{k-2}, Z_{k-1} and Y are kept, in 3 slots at the
 *          start of pSQ->lanczos_vec (slot k % 3), so that the slot written by H and streamed by the sweep is still in
 *          cache. Force step (pSQ->forceFlag): all vectors kept and eig_F[k] = (f(T) e1)[k] / beta_{k-1}, so that the
 *          density matrix column is [V_0 Z_0 ... Z_{npl-2}] eig_F / dV (sqProperties.c). High temperature: all vectors
 *          kept in lanczos_vec_all[nd] and normalized once at the end (V_k = Z_{k-1} / beta_{k-1}), the eigenvectors in
 *          w_all[nd], for the density matrix columns of the force step.
 */
static void LanczosAlgorithm_fused(SPARC_OBJ *pSPARC, int nd, int spn_i, double *lambda_min, double *lambda_max)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int k, Nd = pSQ->Nd_loc, npl = pSPARC->SQ_npl_g, highT = pSPARC->sqHighTFlag == 1, *nloc = pSQ->nloc;
    int center = nloc[0] + nloc[1]*pSQ->Nx_loc + nloc[2]*pSQ->Nx_loc*pSQ->Ny_loc;
    double *z = highT ? pSQ->lanczos_vec_all[nd + spn_i*pSQ->DMnd_SQ] : pSQ->lanczos_vec;   // slot 0: V_0, slot k+1: Z_k
    double *aa = (double *) malloc(npl * sizeof(double)), *bb = (double *) malloc(npl * sizeof(double)), cc[2], g;
    SQ_SWEEP_JIT jit;
    SQ_NL_PLAN plan;
    int ring = !pSQ->forceFlag && !highT && npl >= 3;                               // SCF: 3 slots, slot k % 3
#define SQ_SLOT(k) ((ring) ? (k) % 3 : (k))
    assert(aa != NULL && bb != NULL);
    SQ_sweep_kernels(pSQ, &jit);                                                    // sweep kernels (MKL) and chunk rows
    SQ_nl_plan(pSPARC, &plan);                                                      // the point's nonlocal plan

    memset(z, 0, (size_t)Nd * sizeof(double));                                      // V_0
    z[center] = 1.0;
    for (k = 0; k < npl; k++) {
        double *zk = z + (size_t)SQ_SLOT(k)*Nd, *y = (k < npl - 1 || ring) ? z + (size_t)SQ_SLOT(k+1)*Nd : pSQ->lanczos_tmp;   // slot k (V_0 or Z_{k-1}); Y in slot k+1
        double skm1 = k ? bb[k-1] : 1.0;                                            // |Z_{k-1}| = beta_{k-1} (V_0: 1)
        HsubTimesVec_plan(pSPARC, zk, y, &plan);                                    // Y = H Z_{k-1}
        aa[k] = cblas_ddot(Nd, zk, 1, y, 1) / (skm1 * skm1);                       // alpha_k = Z_{k-1}' Y / beta_{k-1}^2
        if (k == npl - 1) break;                                                    // last step: alpha_k only
        cc[0] = k ? bb[k-1] / (k > 1 ? bb[k-2] : 1.0) : 0.0;                        // c = [beta_{k-1}/beta_{k-2}; alpha_k/beta_{k-1}] (k = 0: [0; alpha_0])
        cc[1] = aa[k] / skm1;
        LanczosSweep(Nd, k ? z + (size_t)SQ_SLOT(k-1)*Nd : NULL, zk, k ? skm1 : 0.0, cc, y, &g, &jit);   // W_k = Y/beta_{k-1} - c1 Z_{k-2} - c2 Z_{k-1}, g = W_k' W_k
        if (!(g > 0.0)) {
            printf(RED "ERROR: Lanczos breakdown at grid point %d, step %d. Use a smaller SQ_NPL_G.\n" RESET, nd, k);
            exit(EXIT_FAILURE);
        }
        bb[k] = sqrt(g);                                                            // beta_k
    }
    TridiagEigenSolve_gauss(pSPARC, aa, bb, nd, spn_i, lambda_min, lambda_max);
    if (pSQ->forceFlag)                                                             // force step: (f(T) e1)[k] / beta_{k-1}
        for (k = 1; k < npl; k++) pSQ->eig_F[k] /= bb[k-1];
    if (highT)                                                                      // the normalized basis for the density matrix columns
        for (k = 1; k < npl; k++) cblas_dscal(Nd, 1.0 / bb[k-1], z + (size_t)k*Nd, 1);
    SQ_nl_plan_free(&plan);
    free(aa); free(bb);
#undef SQ_SLOT
}
