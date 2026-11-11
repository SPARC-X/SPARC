"""
Kubo-Greenwood Optical Spectra Calculator for SPARC Output Files

    Reference: SPARC (https://github.com/SPARC-X/SPARC), the routines each operator mirrors:
               - gradient   : FDweights_D1, lapcT (initialization.c); fract (tools.c); Gradient_vec_dir_kpt,
                              Calc_DX_kpt (gradVecRoutinesKpt.c); [D, x] from the FD Laplacian of
                              Lap_plus_diag_vec_mult_nonorth_kpt (lapVecRoutinesKpt.c)
               - nonlocal   : GetInfluencingAtoms_nloc, CalculateNonlocalProjectors_kpt, Vnl_vec_mult_kpt
                              (nlocVecRoutines.c); Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt (pressure.c);
                              RealSphericalHarmonic, getYD_gen (tools.c)
               - spin-orbit : CalculateNonlocalProjectors_SOC, CreateChiSOMatrix, Vnl_vec_mult_SOC1/SOC2
                              (spinOrbitCoupling.c); ComplexSphericalHarmonic (tools.c)
               - meta-GGA   : compute_mGGA_term_hamil(_kpt) (xc/mgga/mGGAhamiltonianTerm.c)

    Discrete operators of SPARC's real-space Hamiltonian that enter the velocity operator
    v = i[H, r]. They reproduce SPARC's stencils, projectors and conventions, so that v is the
    exact commutator of the Hamiltonian whose eigenstates are stored in the .psi file:
    - kinetic: finite-difference first derivative along each lattice direction, with Bloch
      phases (periodic) or zero ghost points (Dirichlet), and its commutator with the lattice
      coordinate; together these give i[T, r] for SPARC's FD Laplacian, including the mixed
      derivatives of non-orthogonal cells
    - nonlocal: Kleinman-Bylander projectors with periodic images and their first moments,
      giving i[V_NL, r]
    - spin-orbit: projectors of fully relativistic psp8 files and the L.S coupling matrix,
      giving i[V_SO, r] for spinor orbitals
    - meta-GGA: i[H_tau, r] for H_tau = -1/2 sum_ij D_i V_tau g_ij D_j

    @file    operators.py
    @brief   Discrete operators of SPARC's Hamiltonian for the velocity i[H, r]
    @authors Shubhang K Trivedi <strivedi44@gatech.edu>
             Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

    Copyright (c) 2026 Material Physics & Mechanics Group, Georgia Tech.
"""

# *** Imports ****************************************************************
import os
import itertools
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix, bmat, block_diag
from scipy.linalg.blas import get_blas_funcs

TEMP_TOL = 1e-12                  # TEMP_TOL of SPARC's nlocVecRoutines.c (image search)

# Real spherical harmonics exactly as SPARC's RealSphericalHarmonic (tools.c:1231), l = 0..3, m = -l..l
_YLM = {
    (0,  0): lambda x, y, z, r: 0.282094791773878 * np.ones_like(r),
    (1, -1): lambda x, y, z, r: 0.488602511902920 * y / r,
    (1,  0): lambda x, y, z, r: 0.488602511902920 * z / r,
    (1,  1): lambda x, y, z, r: 0.488602511902920 * x / r,
    (2, -2): lambda x, y, z, r: 1.092548430592079 * x * y / r**2,
    (2, -1): lambda x, y, z, r: 1.092548430592079 * y * z / r**2,
    (2,  0): lambda x, y, z, r: 0.315391565252520 * (-x*x - y*y + 2*z*z) / r**2,
    (2,  1): lambda x, y, z, r: 1.092548430592079 * z * x / r**2,
    (2,  2): lambda x, y, z, r: 0.546274215296040 * (x*x - y*y) / r**2,
    (3, -3): lambda x, y, z, r: 0.590043589926644 * (3*x*x - y*y) * y / r**3,
    (3, -2): lambda x, y, z, r: 2.890611442640554 * x * y * z / r**3,
    (3, -1): lambda x, y, z, r: 0.457045799464466 * y * (4*z*z - x*x - y*y) / r**3,
    (3,  0): lambda x, y, z, r: 0.373176332590115 * z * (2*z*z - 3*x*x - 3*y*y) / r**3,
    (3,  1): lambda x, y, z, r: 0.457045799464466 * x * (4*z*z - x*x - y*y) / r**3,
    (3,  2): lambda x, y, z, r: 1.445305721320277 * z * (x*x - y*y) / r**3,
    (3,  3): lambda x, y, z, r: 0.590043589926644 * x * (x*x - 3*y*y) / r**3,
}


# *** Functions **************************************************************
def gradient(S, psi, kred, d, comm=False):
    """
    **********************************************************************
     Purpose:
       - D_d psi : central FD first derivative of order FD_ORDER along lattice
         direction d (non-Cartesian coordinate, spacing delta_d)
             D psi(i) = sum_p w_p (psi(i+p) - psi(i-p)) / delta_d,
             w_p = (2(p%2)-1) fract(FDn,p) / p
       - comm=True: the commutator [D_d, x_d] psi with the coordinate x_d along d,
             [D, x] psi(i) = sum_p p w_p (psi(i+p) + psi(i-p))
         a neighbour average whose weights sum to 1 (= 1 in the continuum,
         1 + O((k delta)^FD_ORDER) on the grid); it enters the exact velocity
         i[T, r] of the cross terms D_i D_j of SPARC's non-orthogonal Laplacian
       - ghost points: periodic -> wrapped values times e^{-ikL} (left) and
         e^{+ikL} (right), kL = 2 pi kred_d; Dirichlet -> zero
       - applied as an (N_d x N_d) matrix along grid axis d with matmul (batched BLAS-3 gemm):
         stencil weights, Bloch phase per cell crossing or Dirichlet truncation built in

     Reference:
       - SPARC: FDweights_D1 (initialization.c:2301), fract (tools.c:725),
                Gradient_vec_dir_kpt / Calc_DX_kpt (gradVecRoutinesKpt.c:64-390)

     Input Parameters:
       S    : SimpleNamespace -- grid (N, delta, BC, fd_order).
       psi  : (Nd, nb) array  -- orbitals, x-fastest grid ordering.
       kred : (3,) array      -- reduced k-point.
       d    : int             -- lattice direction (0, 1, 2).
       comm : bool            -- False: D_d psi;  True: [D_d, x_d] psi.

     Returns:
       (Nd, nb) array -- D_d psi or [D_d, x_d] psi.
    **********************************************************************
    """
    FDn   = S.fd_order // 2
    fract = lambda n, k: np.prod(np.arange(n-k+1, n+1.)) / np.prod(np.arange(n+1, n+k+1.))
    w     = [(2*(p % 2) - 1) * fract(FDn, p) / p for p in range(1, FDn+1)]
    c, sg = ([p * w[p-1] for p in range(1, FDn+1)], 1.0) if comm else ([x / S.delta[d] for x in w], -1.0)
    per   = S.BC[d] == 'P'
    ph    = np.exp(2j*np.pi*kred[d]) if per and kred[d] else 1.0                   # Bloch phase per cell crossing

    # 1D matrix D_d (n x n): row i, columns i +- p, weights c_p and sg c_p, Bloch phase ph^q for q cell crossings
    n, p  = S.N[d], np.arange(1, FDn+1)
    i     = np.arange(n)[:, None]
    j     = np.hstack([i + p, i - p])                                            # (n, 2 FDn) stencil columns
    cp    = np.broadcast_to(np.concatenate([c, sg * np.asarray(c)]), j.shape)
    q     = j // n                                                               # cell crossings
    ok    = np.ones(j.shape, bool) if per else q == 0                            # Dirichlet: ghosts are zero
    D     = np.zeros((n, n), complex if np.iscomplexobj(ph) else float)
    np.add.at(D, (np.broadcast_to(i, j.shape)[ok], (j % n)[ok]), (cp * ph ** q)[ok])
    # 3D operator = 1D matrix along axis d (Kronecker form, never built): x fastest, psi (Nd, ncol) viewed as
    # (slower axes, axis d, faster axes x ncol); one batched gemm along axis d
    return np.matmul(D, psi.reshape(-1, n, int(np.prod(S.N[:d])) * psi.shape[1])).reshape(S.Nd, -1)


def axpy_sum(terms, out=None):
    """ sum_i c_i A_i for terms [(c_i, A_i)] by BLAS-1 axpy, in place (no temporaries); added to out if given """
    if out is None:
        (c, A), *terms = terms
        out = A * c                                                              # new C-contiguous array
    axpy = get_blas_funcs('axpy', (out,))
    for c, A in terms:
        axpy(A.ravel(), out.ravel(), a=c)                                          # out += c A (out.ravel() is a view)
    return out


def mgga_operator(S, psi, kred, vtau, D1=None):
    """
    **********************************************************************
     Purpose:
       - exact commutator of SPARC's meta-GGA term H_tau psi = -1/2 sum_ij D_i (V_tau g_ij D_j psi)
         (g = lapcT) with the lattice coordinate x_k ([D_j, x_k] = delta_jk S_k, S_k = [D_k, x_k]):
             i[H_tau, x_k] psi = -i Y_k,
             Y_k = 1/2 [ sum_i g_ik D_i (V_tau S_k psi) + S_k (V_tau sum_j g_kj D_j psi) ]
         continuum (S = 1): -i/2 (grad V_tau + V_tau grad); Cartesian: sum_k U_ka (...)

     Reference:
       - SPARC: compute_mGGA_term_hamil(_kpt) (xc/mgga/mGGAhamiltonianTerm.c:46-202)

     Input Parameters:
       S    : SimpleNamespace -- grid (N, delta, BC, fd_order, U).
       psi  : (Nd, nb) array  -- orbitals of one spin channel.
       kred : (3,) array      -- reduced k-point.
       vtau : (Nd,) array     -- V_tau of this spin channel (mgga.vtau).
       D1   : list of 3 (Nd, nb) arrays -- D_d psi if already computed (velocity_matrix), else None.

     Returns:
       list of 3 (Nd, nb) arrays -- Y_k, k = lattice directions.
    **********************************************************************
    """
    Ui = np.linalg.inv(S.U)
    g  = Ui.T @ Ui
    V  = vtau[:, None]
    D1 = D1 if D1 is not None else [gradient(S, psi, kred, d) for d in range(3)]
    nz = lambda k: [j for j in range(3) if abs(g[k, j]) > 1e-12]                  # nonzero metric entries
    Y  = []
    for k in range(3):
        VS = gradient(S, psi, kred, k, comm=True)
        VS *= V                                                                    # V_tau S_k psi
        y  = axpy_sum([(g[i, k], gradient(S, VS, kred, i)) for i in nz(k)])
        VG = axpy_sum([(g[k, j], D1[j]) for j in nz(k)])
        VG *= V                                                                    # V_tau sum_j g_kj D_j psi
        y += gradient(S, VG, kred, k, comm=True)
        y *= 0.5
        Y.append(y)
    return Y


def nonlocal_projectors(S):
    """
    **********************************************************************
     Purpose:
       - k-independent setup of the projectors chi_Jlpm(r - R_J') of every atom
         image J' whose rc-sphere reaches the grid, and of (r - R_J')_a chi
         (Cartesian a), stored as sparse (grid point, projector) entries
       - chi = UdV_lp(|r-R|) * Y_lm(r-R); UdV by the not-a-knot cubic spline
         (same slopes as SPARC's getYD_gen); Y_lm = 0 at r < 1e-10 for l > 0
       - search box along lattice direction i: rc*sqrt(lapcT_ii), the exact
         extent of the rc-sphere; SPARC's larger box (CUTOFF_x for non-orth
         cells) selects the same points, since only r <= rc is kept

     Reference:
       - SPARC: GetInfluencingAtoms_nloc (nlocVecRoutines.c:43),
                CalculateNonlocalProjectors_kpt (nlocVecRoutines.c:578),
                Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt (pressure.c:1166),
                nonCart2Cart_coord (initialization.c:3922)

     Input Parameters:
       S : SimpleNamespace -- grid, cell (U, L), atoms (atom_pos, atom_typ), S.psds.

     Returns:
       S : SimpleNamespace -- S.nl (rows, cols, chi, X (3,nnz), img (nnz,3), gamma, nproj, and blocks: one
                              k-independent (4 nproj x Nd) CSR matrix [chi; (r-R)_a chi]^T per periodic image).
    **********************************************************************
    """
    gradT = np.linalg.inv(S.U).T
    rows, cols, chi, X, img, gamma = [], [], [], [], [], []
    for x0, t in zip(S.atom_pos, S.atom_typ):
        psd   = S.psds[t]
        lpm   = [(l, p, m) for l in range(psd.lmax+1) if l != psd.lloc for p in range(psd.ppl[l]) for m in range(-l, l+1)]
        udv   = {(l, p): CubicSpline(psd.r, psd.udv[l][:, p], bc_type='not-a-knot') for l, p, _ in lpm}
        rc    = max(psd.rc)
        rcbox = rc * np.sqrt(np.diag(gradT @ gradT.T)) + TEMP_TOL
        rng   = [range(-int(np.floor((rcbox[i] + x0[i]) / S.L[i] + TEMP_TOL)),
                       int(np.floor((rcbox[i] + S.L[i] - x0[i]) / S.L[i] + TEMP_TOL)) + 1) if S.BC[i] == 'P' else range(1)
                 for i in range(3)]
        for n in itertools.product(*rng):
            xi = x0 + np.array(n) * S.L                                    # image position (non-Cartesian)
            lo = np.maximum(np.ceil((xi - rcbox) / S.delta), 0).astype(int)
            hi = np.minimum(np.floor((xi + rcbox) / S.delta), S.N - 1).astype(int)
            if np.any(hi < lo): continue
            ijk = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]+1) for i in range(3)], indexing='ij'), -1).reshape(-1, 3)
            dc  = (ijk * S.delta - xi) @ S.U                               # Cartesian r - R_J'
            r   = np.linalg.norm(dc, axis=1)
            ok  = r**2 <= rc**2
            if not ok.any(): continue
            ijk, dc, r = ijk[ok], dc[ok], r[ok]
            g   = ijk[:, 0] + ijk[:, 1] * S.N[0] + ijk[:, 2] * S.N[0] * S.N[1]
            rs  = np.where(r < 1e-10, 1.0, r)
            for j, (l, p, m) in enumerate(lpm):
                c = udv[l, p](r) * _YLM[l, m](*dc.T, rs) * (1.0 if l == 0 else (r >= 1e-10))
                rows.append(g); cols.append(np.full(g.size, len(gamma) + j)); chi.append(c)
                X.append(dc.T * c); img.append(np.tile(np.floor(xi / S.L), (g.size, 1)))
        gamma += [psd.gamma[l][p] for l, p, _ in lpm]
    S.nl = SimpleNamespace(rows=np.concatenate(rows), cols=np.concatenate(cols), chi=np.concatenate(chi),
                           X=np.concatenate(X, axis=1), img=np.concatenate(img), gamma=np.array(gamma), nproj=len(gamma))
    # k-independent sparse blocks, one per periodic image n: [chi; (r-R)_a chi]^T as a (4 nproj x Nd) CSR matrix;
    # the k-dependence is only the Bloch phase of each block (nonlocal_operator)
    nl, n = S.nl, S.nl.nproj
    imgs, which = np.unique(nl.img, axis=0, return_inverse=True)
    nl.blocks = [(u, csr_matrix((np.concatenate([nl.chi[m], *nl.X[:, m]]),
                                 (np.concatenate([nl.cols[m] + a * n for a in range(4)]), np.tile(nl.rows[m], 4))),
                                shape=(4 * n, S.Nd)))
                 for u, m in ((imgs[b], which.ravel() == b) for b in range(len(imgs)))]
    return S


def nonlocal_operator(S, psi, kred):
    """
    **********************************************************************
     Purpose:
       - Bloch projectors chi~(r) = sum_J' e^{-i theta_J'} chi(r - R_J'),
         theta_J' = -k . floor(R_J'/L) L  (i.e. e^{+2 pi i kred . n_J'})
       - C   = dV chi~^H psi            (nproj, nb)   -- SPARC's alpha / dV-weighted
       - D_a = dV [(r-R)_a chi]~^H psi  (3, nproj, nb)
       - both at once: sum over periodic images n of e^{-2 pi i kred . n} (P_n @ psi), with the
         k-independent sparse blocks P_n of nonlocal_projectors (one sparse-dense product per image)

     Reference:
       - SPARC: Vnl_vec_mult_kpt (nlocVecRoutines.c:893-1003),
                Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt (pressure.c:1166)

     Input Parameters:
       S    : SimpleNamespace -- S.nl from nonlocal_projectors, S.Nd, S.dV.
       psi  : (Nd, nb) array  -- orbitals.
       kred : (3,) array      -- reduced k-point.

     Returns:
       C : (nproj, nb), D : (3, nproj, nb)
    **********************************************************************
    """
    n  = S.nl.nproj
    CD = 0.0
    for u, P in S.nl.blocks:                                                         # sum over periodic images
        CD = CD + (np.exp(-2j * np.pi * (u @ kred)) if kred.any() else 1.0) * (P @ psi)   # conjugate Bloch phase
    CD = CD * S.dV
    return CD[:n], CD[n:].reshape(3, n, -1)


def soc_projectors(S):
    """
    **********************************************************************
     Purpose:
       - k-independent setup of the spin-orbit term of fully relativistic
         pseudopotentials, V_SO = sum_{J,l,p} |chi_Jlp> gamma_so L.S <chi_Jlp|
       - projectors: nonlocal_projectors applied to the SO radial functions
         (psd.soc), i.e. the same atom images, grid points (scalar rc), splines
         and (r - R) as the scalar projectors, here with real Y_lm
       - coupling matrix M (2 nproj x 2 nproj, spinor blocks [up, dn]) with
         V_SO = |chi~> M <chi~|, built per (atom, l, p) in SPARC's complex Y_lm
         basis (Condon-Shortley phase) exactly as SPARC applies it:
             up-up : +m/2 gamma                             (Vnl_vec_mult_SOC1, spinorfac +1)
             dn-dn : -m/2 gamma                             (Vnl_vec_mult_SOC1, spinorfac -1)
             up-dn : |m> sqrt(l(l+1)-m(m+1))/2 gamma <m+1|  (Vnl_vec_mult_SOC2, spinor 0)
             dn-up : the transpose                          (Vnl_vec_mult_SOC2, spinor 1)
         (= gamma (L_z S_z + (L_- S_+ + L_+ S_-)/2) = gamma L.S), then rotated to
         the real Y_lm basis: M_real = U^T M U*, where Y_l^m = sum_m' U_mm' Y_lm'

     Reference:
       - SPARC: CalculateNonlocalProjectors_SOC, CreateChiSOMatrix, Vnl_vec_mult_SOC1,
                Vnl_vec_mult_SOC2 (spinOrbitCoupling.c:44-632), Hamiltonian_vectors_mult_kpt
                (hamiltonianVecRoutines.c:190-221), ComplexSphericalHarmonic (tools.c:1664)

     Input Parameters:
       S : SimpleNamespace -- as for nonlocal_projectors, with S.psds[t].soc (read_psp8).

     Returns:
       S : SimpleNamespace -- S.so (rows, cols, chi, X, img, gamma, nproj as S.nl, and M (sparse)).
    **********************************************************************
    """
    for f, p in zip(S.psp_files, S.psds):
        if not p.pspsoc:
            raise ValueError(f'{os.path.basename(f)} has no spin-orbit projectors, but the run used fully relativistic '
                             f'pseudopotentials: PSP_DIR points to a different file than the SPARC run used.')
    S.so = nonlocal_projectors(SimpleNamespace(**{**vars(S), 'psds': [p.soc for p in S.psds]})).nl
    blk  = [(l, g) for t in S.atom_typ for l in range(1, S.psds[t].lmax + 1) if l != S.psds[t].lloc for g in S.psds[t].soc.gamma[l]]
    U    = {l: np.zeros((2*l+1, 2*l+1), complex) for l, _ in blk}                    # rows m (complex), columns m' (real)
    for l in U:
        U[l][l, l] = 1.0
        for m in range(1, l + 1):
            U[l][l+m, [l+m, l-m]] = (-1)**m * np.array([1, 1j]) / np.sqrt(2)          # Y_l^m  = (-1)^m (Y_lm + i Y_l,-m)/sqrt2
            U[l][l-m, [l+m, l-m]] = np.array([1, -1j]) / np.sqrt(2)                   # Y_l^-m = (Y_lm - i Y_l,-m)/sqrt2
    Lz = lambda l: np.diag(np.arange(-l, l + 1) / 2.0)                                   # L_z S_z (up)
    Lm = lambda l: np.diag(np.sqrt(l*(l+1) - np.arange(-l, l) * np.arange(-l+1, l+1)) / 2.0, 1)   # L_- S_+ / 2
    Mc = lambda l, g: [[g * Lz(l), g * Lm(l)], [g * Lm(l).T, -g * Lz(l)]]
    S.so.M = bmat([[block_diag([U[l].T @ Mc(l, g)[s][t] @ U[l].conj() for l, g in blk]) for t in range(2)]
                   for s in range(2)], format='csr')
    return S


def soc_operator(S, psi, kred):
    """
    **********************************************************************
     Purpose:
       - C and D of nonlocal_operator for the spin-orbit projectors (S.so), for
         both spinor components, stacked [up; dn] to match the blocks of S.so.M

     Input Parameters:
       S    : SimpleNamespace -- S.so from soc_projectors, S.Nd, S.dV.
       psi  : (2 Nd, nb) array -- spinor orbitals [psi_up; psi_dn].
       kred : (3,) array       -- reduced k-point.

     Returns:
       C : (2 nproj, nb), D : (3, 2 nproj, nb)
    **********************************************************************
    """
    CD = [nonlocal_operator(SimpleNamespace(Nd=S.Nd, dV=S.dV, nl=S.so), x, kred) for x in psi.reshape(2, S.Nd, -1)]
    return np.concatenate([C for C, _ in CD]), np.concatenate([D for _, D in CD], axis=1)
