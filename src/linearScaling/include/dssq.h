/**
 * @file    dssq.h
 * @brief   This file contains the data structure declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef DSSQ_H
#define DSSQ_H 

#include <mpi.h>

// declare datatypes (defined elsewhere)
typedef struct _D2DEXT_OBJ D2Dext_OBJ;
typedef struct _KRON_LAP KRON_LAP;
typedef struct _MPEXP_OBJ MPEXP_OBJ;
typedef struct _NLOC_PROJ_OBJ NLOC_PROJ_OBJ;
typedef struct _ATOM_NLOC_INFLUENCE_OBJ ATOM_NLOC_INFLUENCE_OBJ;
// MKL: the JIT dgemm kernels of one atom-image shape of the nonlocal pass (ndc points, nproj projectors, dV):
// 0 projection alpha = dV Chi' x, 1 expansion x = Chi (Gamma alpha)
typedef struct { int ndc, nproj; double dV; void *j[2]; void (*k[2])(void *, double *, double *, double *); } SQ_JIT_NL_ENTRY;

/**
 * @brief   This structure type is for SQ method.
 */
typedef struct _SQ_OBJ
{
    int nloc[3];                      // Number of finite difference intervals on each direction within Rcut distance

    MPI_Comm dmcomm_SQ;             // SQ Communicator topology
    int DMnx_SQ;
    int DMny_SQ;
    int DMnz_SQ;
    int DMnd_SQ;
    int DMVertices_SQ[6];

    int Nx_loc;
    int Ny_loc;
    int Nz_loc;
    int Nd_loc;                       // Total number of finite difference nodes within Rcut domain

    int DMnx_PR;
    int DMny_PR;
    int DMnz_PR;
    int DMnd_PR;
    int DMVertices_PR[6];             // Domain vertices of PR (process+Rcut) domain

    int Nx_ex;
    int Ny_ex;
    int Nz_ex;
    int Nd_ex;

    MPI_Comm dmcomm_d2dext_sq;      // SQ Communicator for transferring data in PR domain
    D2Dext_OBJ *d2dext_dmcomm_sq;
    D2Dext_OBJ *d2dext_dmcomm_sq_ext;

    double **lanczos_vec_all;
    double **w_all;
    double *lanczos_vec;
    double *w;

    double *mineig;
    double *maxeig;
    double *lambda_max;
    double *Veff_loc_SQ;
    double **gnd;
    double **gwt;
    double *Veff_PR;                // Veff operator within PR (process+Rcut) domain
    double *x_ex;
    int forceFlag;

    // the nonlocal projectors of the current grid point: rows of the PR-domain projectors (nlocProj_kptcomm) within
    // its Rcut sphere and box, built per Lanczos run (GetNonlocalProjectorsForOnePoint) into grow-only buffers
    NLOC_PROJ_OBJ *nlocProj_pt;                      // Ntypes entries: nproj, Chi[iat] (into nlpt_chi)
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc_pt; // Ntypes entries: n_atom, coords, atom_index, ndc, grid_pos[iat] (into nlpt_pos)
    int *nlpt_cap;                                   // per type: images its coords/atom_index/ndc/grid_pos/Chi arrays can hold
    int *nlpt_pos;                                   // flat grid_pos storage of all images of the point (nlpt_pos_cap ints)
    double *nlpt_chi;                                // flat Chi storage of all images of the point (nlpt_chi_cap doubles)
    size_t nlpt_pos_cap, nlpt_chi_cap;
    double *Lap_1D_x;                 // 1D Dirichlet Laplacians of the Rcut box (Kronecker form of the Laplacian)
    double *Lap_1D_y;
    double *Lap_1D_z;
    double *Veff_box;                 // Veff on the current point's Rcut box, contiguous (Nd_loc)
    double *lanczos_tmp;              // the last H v of a Lanczos run, not stored (Nd_loc)
    // eigensolve of the tridiagonal T by dstedc, workspace preallocated (N = SQ_npl_g)
    double *eig_d;                    // diagonal, then the eigenvalues (N)
    double *eig_e;                    // subdiagonal (N)
    double *eig_work;                 // dstedc workspace, eig_lwork doubles
    int *eig_iwork;                   // dstedc integer workspace, eig_liwork ints
    long eig_lwork, eig_liwork;
    double *eig_C;                    // force step: Z(0,:)' .* f(theta) (N)
    double *eig_F;                    // force step: f(T) e1 (N), the density matrix column is V eig_F / dV
    // MKL: JIT dgemm kernels of the Laplacian on the Rcut box (x stage Lx X_k, y stage X_k Ly), created on first use
    int jit_state;                    // 0 not created, 1 created, -1 MKL could not create them (the batched stages are used)
    double jit_a;
    void *jit_jx, *jit_jy;
    void (*jit_kx)(void *, double *, double *, double *);
    void (*jit_ky)(void *, double *, double *, double *);
    // MKL: JIT dgemm kernels of the fused Lanczos sweep: update (K = 2 and K = 1), Gram; chunk rows jits_R
#define SQ_JITS_N 3
    int jits_state, jits_R;
    void *jits_j[SQ_JITS_N];
    void (*jits_k[SQ_JITS_N])(void *, double *, double *, double *);
    // MKL: JIT dgemm kernels of the nonlocal pass per (points, projectors): projection, expansion
    SQ_JIT_NL_ENTRY *jitnl;
    int jitnl_n, jitnl_cap;

    // exact exchange
    KRON_LAP *kron_lap;
    MPEXP_OBJ *MpExp_exx;
    double **Dn;
    double **Dn_exx;
    double *basis;
    double **exxPot;
    double *erfcR;
} SQ_OBJ;


#endif // DSSQ_H