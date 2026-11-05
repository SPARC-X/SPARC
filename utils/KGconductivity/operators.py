"""
*************************************************************************************
*     Program  :  KGconductivity  (Kubo-Greenwood conductivity from SPARC output)
*     Module   :  operators
*     Authors  :  <placeholder>
*
*
*     Created  :  2026-11-04
*     Modified :  2026-11-04
*
*     Purpose  :  The two pieces of SPARC's discrete Hamiltonian that enter the
*                 velocity operator v = -i grad + i[V_NL, r]:
*                   - gradient : FD first derivative along a lattice direction
*                                with Bloch phases (periodic) or zero ghosts (Dirichlet)
*                   - nonlocal : projections C = <chi|psi> and D = <(r-R) chi|psi>
*                                of the Kleinman-Bylander projectors (with images)
*
*     Routines :  gradient()              (public)
*                 nonlocal_projectors()   (public, k-independent setup)
*                 nonlocal_operator()     (public)
*
*     Reference:  SPARC  --  https://github.com/SPARC-X/SPARC
*************************************************************************************
"""

# *** Imports ****************************************************************
import itertools
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix

TEMP_TOL = 1e-12                  # TEMP_TOL in SPARC isddft.h (image search)

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
def gradient(S, psi, kred, d):
    """
    **********************************************************************
     Purpose:
       - D_d psi : central FD first derivative of order FD_ORDER along lattice
         direction d (non-Cartesian coordinate, spacing delta_d)
             D psi(i) = sum_p w_p (psi(i+p) - psi(i-p)) / delta_d,
             w_p = (2(p%2)-1) fract(FDn,p) / p
       - ghost points: periodic -> wrapped values times e^{-ikL} (left) and
         e^{+ikL} (right), kL = 2 pi kred_d; Dirichlet -> zero

     Reference:
       - SPARC: FDweights_D1 (initialization.c:2301), fract (tools.c:725),
                Gradient_vec_dir_kpt / Calc_DX_kpt (gradVecRoutinesKpt.c:64-390)

     Input Parameters:
       S    : SimpleNamespace -- grid (N, delta, BC, fd_order).
       psi  : (Nd, nb) array  -- orbitals, x-fastest grid ordering.
       kred : (3,) array      -- reduced k-point.
       d    : int             -- lattice direction (0, 1, 2).

     Returns:
       (Nd, nb) array -- D_d psi.
    **********************************************************************
    """
    FDn   = S.fd_order // 2
    fract = lambda n, k: np.prod(np.arange(n-k+1, n+1.)) / np.prod(np.arange(n+1, n+k+1.))
    w     = [(2*(p % 2) - 1) * fract(FDn, p) / p / S.delta[d] for p in range(1, FDn+1)]
    ph    = (np.exp(2j*np.pi*kred[d]) if kred[d] else 1.0) if S.BC[d] == 'P' else 0.0

    Y  = np.moveaxis(psi.reshape(S.N[2], S.N[1], S.N[0], -1), 2-d, 0)   # direction d first
    n  = Y.shape[0]
    Yp = np.concatenate([Y[n-FDn:] * np.conj(ph), Y, Y[:FDn] * ph])      # left / right ghost blocks
    DY = sum(w[p-1] * (Yp[FDn+p:FDn+p+n] - Yp[FDn-p:FDn-p+n]) for p in range(1, FDn+1))
    return np.moveaxis(DY, 0, 2-d).reshape(S.Nd, -1)


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
       S : SimpleNamespace -- S.nl (rows, cols, chi, X (3,nnz), img (nnz,3), gamma, nproj).
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
    return S


def nonlocal_operator(S, psi, kred):
    """
    **********************************************************************
     Purpose:
       - Bloch projectors chi~(r) = sum_J' e^{-i theta_J'} chi(r - R_J'),
         theta_J' = -k . floor(R_J'/L) L  (i.e. e^{+2 pi i kred . n_J'})
       - C   = dV chi~^H psi            (nproj, nb)   -- SPARC's alpha / dV-weighted
       - D_a = dV [(r-R)_a chi]~^H psi  (3, nproj, nb)

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
    nl  = S.nl
    ph  = np.exp(2j * np.pi * (nl.img @ kred))
    prj = lambda v: csr_matrix((v * ph, (nl.rows, nl.cols)), shape=(S.Nd, nl.nproj)).conj().T @ psi * S.dV
    return prj(nl.chi), np.stack([prj(nl.X[a]) for a in range(3)])
