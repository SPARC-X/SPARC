"""
Kubo-Greenwood Optical Spectra Calculator for SPARC Output Files

    Reference: L. Calderin, V. V. Karasiev, S. B. Trickey, Comput. Phys. Commun. 221, 118 (2017)

    Kubo-Greenwood evaluation of the frequency-dependent conductivity and dielectric tensors
    (Calderin et al., Eqs. 8-11 and 21-25), in Hartree atomic units for the time dependence
    exp(-i omega t):
    - velocity matrix elements v_nm = <n|i[H, r]|m> of SPARC's discrete Hamiltonian, including
      the nonlocal and spin-orbit contributions
    - transition weights in the exact (Delta f / Delta epsilon) or approximated (Delta f / omega)
      form, with the intraband and degenerate-state (Drude) contribution -df/d epsilon
    - Lorentzian or Gaussian broadening, the complex conductivity sigma = sigma1 + i sigma2 and
      the dielectric tensor epsilon_ab = delta_ab + 4 pi i sigma_ab / omega
    - DC conductivity, f-sum rule, orthonormality and Hermiticity checks, output files and plots
    All large products use np.matmul, NumPy ufuncs and SciPy sparse products, which release the
    GIL, so (k-point, spin) pairs run concurrently on threads (SciPy's f2py BLAS wrappers hold it).
    Time-reversal symmetry (SPIN_TYP 0 with or without spin-orbit coupling, and SPIN_TYP 1)
    cancels the imaginary part of v^a_nm v^b_mn between k and -k, so only the real part enters,
    consistent with SPARC's time-reversal-reduced k-point set.

    @file    dielectrics.py
    @brief   Velocity matrix, conductivity and dielectric tensors, output files
    @authors Shubhang K Trivedi <strivedi44@gatech.edu>
             Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

    Copyright (c) 2026 Material Physics & Mechanics Group, Georgia Tech.
"""

# *** Imports ****************************************************************
import os
import time

import numpy as np
from scipy.special import dawsn

from operators import gradient, axpy_sum, nonlocal_operator, soc_operator

HA2EV     = 27.211386245988       # Ha -> eV (CONST_EH in SPARC isddft.h)
AU2SI     = 4.599848e6            # 1 a.u. of conductivity, e^2/(hbar a0), in S/m
DEGEN_TOL = 1e-6                  # |e_n - e_m| (Ha) below which a pair is treated as degenerate
F_TOL     = 1e-12                 # pairs with |F_nm| below this are skipped
BAND_BLOCK = 32                   # bands per block in velocity_matrix: peak memory ~ 1x instead of 7x the orbital block

# -df/de for SPARC's two smearing functions (smearing_function, occupation.c:33)
MDFDE = {
    'fermi-dirac': lambda S: S.beta * S.f * (1.0 - S.f),
    'gaussian'   : lambda S: S.beta / np.sqrt(np.pi) * np.exp(-(S.beta * (S.eig - S.E_F))**2),
}

# Broadening of delta(De - w): (absorptive, dispersive) kernels of x = De - w, both = pi * (shape, its Hilbert transform).
# Same full width at half maximum 2*eta for both shapes: Gaussian width sigma_g = eta/sqrt(ln 2) (Calderin App. F,
# Eq. F.4 with delta = 2 eta). The Gaussian's dispersive kernel is its exact Hilbert (Kramers-Kronig) transform,
# 2 D(x/sigma_g)/sigma_g with D the Dawson function; -> 1/x far from resonance, like the Lorentzian.
LN2 = np.log(2.0)
KERNELS = {
    'lorentzian': (lambda x, eta: eta / (x**2 + eta**2),
                   lambda x, eta: x / (x**2 + eta**2)),
    'gaussian'  : (lambda x, eta: np.sqrt(np.pi * LN2) / eta * np.exp(-LN2 * (x / eta)**2),
                   lambda x, eta: 2 * np.sqrt(LN2) / eta * dawsn(np.sqrt(LN2) * x / eta)),
}

# Pair kernel for a broadening kernel K, pair energy x0 = De_nm and frequency w:
#   'exact'        (Calderin Eq. 8, D-d)        : weight Df/De times K(x0 - w)
#   'approximated' (Calderin Eq. 9, ABINIT conducti): weight Df    times (K(x0 - w) - K(x0)) / w
# Summing all ordered pairs already includes the anti-resonant term (even parity, Eq. 20); the K(x0) term cancels
# in sigma1 and removes the spurious 1/w pole in sigma2, leaving the exact Kramers-Kronig partner of sigma1
# (the paper gives no sigma2 for Eq. 9). w = 0 uses the limit -K'(x0) (cf. Eqs. 82-83).
FORMULA = {
    'exact'       : lambda K, x0, w, h: K(x0 - w),
    'approximated': lambda K, x0, w, h: np.where(w != 0, (K(x0 - w) - K(x0)) / np.where(w != 0, w, 1.0),
                                                 (K(x0 - h) - K(x0 + h)) / (2 * h)),
}


# *** Functions **************************************************************
def format_bytes(b):
    """ bytes as '%.2f <unit>' with 1024-based units, as SPARC's formatBytes (tools.c:1205) """
    units = ['B', 'kB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while b >= 1024.0 and i < len(units) - 1:
        b, i = b / 1024.0, i + 1
    return f'{b:.2f} {units[i]}'


def occupations(S):
    """
    **********************************************************************
     Purpose:
       - g_s = electrons per orbital: 2 (SPIN_TYP 0), 1 (spinor orbital with
         spin-orbit, or one spin channel of SPIN_TYP 1) -- SPARC's occfac
       - f from the printed occupations: SPARC prints (2/Nspin/Nspinor) f
         (printing.c:230); the factor is fixed by sum_k w_k sum g_s f = N_e
       - f is an occupation probability, so it is clipped to [0, 1]: the
         rescaling (Fermi-level tolerance, 12-digit print) can give f = 1 + 1e-9,
         which would make -df/de = beta f (1 - f) slightly negative
       - -df/de from f and beta (Fermi-Dirac) or from e, E_F, beta (Gaussian)

     Input Parameters:
       S : SimpleNamespace -- occ_printed, wk, N_e, spin_typ, nspinor, smearing data.

     Returns:
       S : SimpleNamespace -- S.g_s, S.occ_factor, S.f, S.mdfde (nspin, nkpt, nstates).
    **********************************************************************
    """
    S.g_s        = 2.0 / S.nspinor if S.spin_typ == 0 else 1.0
    S.occ_factor = S.g_s * np.einsum('k,skn->', S.wk, S.occ_printed) / S.N_e
    S.f          = np.clip(S.occ_printed / S.occ_factor, 0.0, 1.0)
    S.mdfde      = MDFDE[S.smearing_type](S)
    return S


def velocity_matrix(S, psi, kred):
    """
    **********************************************************************
     Purpose:
       - v^a_nm = <psi_n| i[T, r_a] + i[V_NL, r_a] |psi_m>   (Cartesian a)
           P^a  = <psi_n| i[T, r_a] |psi_m>, the exact velocity of SPARC's FD kinetic operator
                  T = -1/2 (sum_k g_kk D2_k + sum_{j!=k} g_kj D_j D_k),  g = lapcT (cross terms: non-orth cells).
                  Along lattice coordinate x_k ([D2, x] = 2 D exactly for SPARC's weights):
                    i[T, x_k] = -i (g_kk D_k + [D_k, x_k] sum_{j!=k} g_kj D_j),   P^a = sum_k U_ka i[T, x_k]
                  Orthogonal cells: -i D_a; continuum ([D, x] = 1): -i grad
           NL^a = i (C^H Gamma D^a - D^a^H Gamma C)
         -i grad and the scalar V_NL are spin-diagonal: for spinor orbitals
         (spin-orbit) P and NL are summed over the two components, and
           SO^a = i (C^H M D^a - D^a^H M C)    (C, D of both components, M of soc_projectors)
         is added (V_NL = scalar part + V_SO, as hamiltonianVecRoutines.c:190-221).
         With one component (no spin-orbit) this is P + NL, unchanged.
       - S.velocity: 'gradient' -> P only (momentum, -i grad);
                     'gradient+nonlocal' -> P + NL (+ SO) (full velocity i[H, r])

     Reference:
       - SPARC: lapcT (initialization.c:3862), D1_stencil_coeffs_xy... (initialization.c:2357),
                Lap_plus_diag_vec_mult_nonorth_kpt, cell_typ 17 (lapVecRoutinesKpt.c:573)

     Input Parameters:
       S    : SimpleNamespace -- grid, cell, S.nl (and S.so for spinors).
       psi  : (Nd * nspinor, nb) array -- orbitals at this k-point and spin.
       kred : (3,) array      -- reduced k-point.

     Returns:
       v : (3, nb, nb) complex array.
    **********************************************************************
    """
    Ui = np.linalg.inv(S.U)
    g  = Ui.T @ Ui                                                                 # lapcT = gradT gradT^T, gradT = U^{-T}
    nb = psi.shape[1]
    v  = 0.0
    for x in psi.reshape(S.nspinor, S.Nd, -1):                                     # spinor components (one without SOC)
        P    = np.empty((3, nb, nb), complex)
        for b in range(0, nb, BAND_BLOCK):                                          # band blocks (all ops are per column)
            xb = np.ascontiguousarray(x[:, b:b+BAND_BLOCK])
            D1 = [gradient(S, xb, kred, d) for d in range(3)]
            Y  = []
            for k in range(3):                                                      # i[T, x_k] = -i Y_k
                J = [j for j in range(3) if j != k and abs(g[k, j]) > 1e-12]       # cross terms (non-orthogonal cells)
                if J:
                    y = gradient(S, axpy_sum([(g[k, j], D1[j]) for j in J]), kred, k, comm=True)
                    Y.append(axpy_sum([(g[k, k], D1[k])], out=y))                  # += g_kk D_k psi, in place
                else:
                    Y.append(g[k, k] * D1[k])
            for k, y in enumerate(Y):                                               # BLAS-3: x^H y = conj(x^T conj(y)), no copy of x
                P[k, :, b:b+BAND_BLOCK] = (-1j * S.dV) * np.matmul(x.T, np.conjugate(y, out=y)).conj()
            del xb, D1, Y                                                           # free before the next block
        v  = v + np.einsum('ka,knm->anm', S.U, P)
        if S.velocity == 'gradient+nonlocal':
            C, D = nonlocal_operator(S, x, kred)
            G = S.nl.gamma[:, None]
            v = v + np.array([1j * (C.conj().T @ (G * D[a]) - D[a].conj().T @ (G * C)) for a in range(3)])
    if S.nspinor == 2 and S.velocity == 'gradient+nonlocal':                     # spin-orbit term i[V_SO, r]
        C, D = soc_operator(S, psi, kred)
        v = v + np.array([1j * (C.conj().T @ (S.so.M @ D[a]) - D[a].conj().T @ (S.so.M @ C)) for a in range(3)])
    return v


def weights_F(eps, f, mdfde, drude=True, formula='exact'):
    """
    **********************************************************************
     Purpose:
       - 'exact'       : F_nm = (f_m - f_n) / (e_n - e_m)   (Calderin Eq. 10: Df_{n'n}/De_{nn'})
                         n = m and degenerate pairs (|e_n - e_m| <= DEGEN_TOL): F -> -df/de
                         (Drude term, Calderin Eqs. 21-25); drude=False sets them to 0
                         (interband only, as ABINIT optic / conducti)
       - 'approximated': F_nm = f_m - f_n (the 1/w is in the kernel, Eq. 9); intraband and
                         degenerate pairs have Df = 0, so this form has no Drude term

     Input Parameters:
       eps, f, mdfde : (nb,) arrays -- eigenvalues, occupations, -df/de.
       drude         : bool         -- keep the intraband/degenerate (Drude) term.
       formula       : str          -- 'exact' or 'approximated'.

     Returns:
       F : (nb, nb) array.
    **********************************************************************
    """
    if formula == 'approximated':
        return f[None, :] - f[:, None]
    de  = eps[:, None] - eps[None, :]
    deg = np.abs(de) <= DEGEN_TOL
    return np.where(deg, 0.5 * (mdfde[:, None] + mdfde[None, :]) * drude, (f[None, :] - f[:, None]) / np.where(deg, 1.0, de))


def _pair_sum(S, F, Z, de, omega, K):
    """ g_s/Omega * sum over pairs with |F| > F_TOL of F_nm Z_nm FORMULA(K, de_nm, omega), omega in chunks """
    m = np.abs(F) > F_TOL
    W, x = (F[m][:, None, None] * Z[m]).reshape(-1, 9), de[m]
    out = [FORMULA[S.formula](K, x[None, :], w[:, None], 1e-4 * S.eta) @ W
           for w in np.array_split(omega, max(1, omega.size // 64))]
    return S.g_s / S.Omega * np.concatenate(out).reshape(-1, 3, 3)


def sigma1_tilde(S, F, Z, de, omega):
    """ Calderin Eq. (10): (g_s/Omega) sum F_nm Z_nm K1(De - w); K1 = eta/(x^2+eta^2) (Lorentzian) or pi*Gaussian """
    return _pair_sum(S, F, Z, de, omega, lambda x: KERNELS[S.broadening][0](x, S.eta))


def sigma2_tilde(S, F, Z, de, omega):
    """ Calderin Eq. (11): (g_s/Omega) sum F_nm Z_nm K2(De - w); K2 = x/(x^2+eta^2) (Lorentzian) or Dawson (Gaussian) """
    return _pair_sum(S, F, Z, de, omega, lambda x: KERNELS[S.broadening][1](x, S.eta))


def sigma1(st1, st2):
    """ relation above Calderin Eq. (13): sigma1 = Re sigma~1 - Im sigma~2 """
    return st1.real - st2.imag


def sigma2(st1, st2):
    """ relation above Calderin Eq. (15), with the sign of the e^{-i omega t} convention """
    return -(st1.imag + st2.real)


def sigma(s1, s2):
    """ complex conductivity tensor (nomega, 3, 3) """
    return s1 + 1j * s2


def epsilon(sig, omega):
    """ dielectric tensor eps = delta + 4 pi i sigma / omega, for omega > 0 """
    w = omega > 0
    return np.eye(3) + 4j * np.pi * sig[w] / omega[w, None, None]


def checks(S, psi, v):
    """
    **********************************************************************
     Purpose:
       - max |psi^H psi dV - I| (normalization) and max |v - v^H| (Hermiticity)
       - Gram matrix on and above the diagonal band blocks (np.matmul): the cost of herk, one block copy

     Returns:
       (float, float)
    **********************************************************************
    """
    nb = psi.shape[1]
    G  = np.zeros((nb, nb), psi.dtype)
    for b in range(0, nb, BAND_BLOCK):                         # conj(psi^H psi), columns b:b+B, rows 0:b+B
        G[:b+BAND_BLOCK, b:b+BAND_BLOCK] = np.matmul(psi[:, :b+BAND_BLOCK].T, psi[:, b:b+BAND_BLOCK].conj())
    return (np.abs(S.dV * G - np.eye(nb)).max(), np.abs(v - v.conj().transpose(0, 2, 1)).max())


def write_output(S):
    """
    **********************************************************************
     Purpose:
       - <name>.kgout, .kglog, .sigma, .epsilon, <name> = SPARC run name; numbered as SPARC's .out_NN:
         if <name>.kgout exists, the first free .kgout_01 ... _100 and the same suffix for all files;
         the .kgout is created exclusively, so concurrent runs never get the same suffix
       - .kgout   : results, laid out as SPARC's .out (input parameters, system, DC conductivity,
                    sum rules, output files, timing); 'Label : value (unit)' lines
       - .kglog   : messages (warnings, notes, progress) and numerical checks
       - .sigma   : omega (Ha, eV), sigma1/sigma2 of xx yy zz xy xz yz, Tr/3 (a.u.), Tr sigma1/3 (S/m)
       - .epsilon : omega > 0 (Ha, eV), eps1/eps2 of the same components and Tr/3

     Input Parameters:
       S : SimpleNamespace -- results of kubo_greenwood.
    **********************************************************************
    """
    os.makedirs(S.out_dir, exist_ok=True)
    base = os.path.join(S.out_dir, S.prefix)
    for S.suffix in [''] + [f'_{i:02d}' for i in range(1, 101)]:                 # SPARC's .out_NN rule
        try:
            fp = open(base + '.kgout' + S.suffix, 'x')                               # atomic: no two runs share a suffix
            break
        except FileExistsError:
            continue
    else:
        S.suffix, fp = '', open(base + '.kgout', 'w')                                # all taken: overwrite, as SPARC
        S.warnings.append(f'{S.prefix}.kgout_01 to _100 all exist in {S.out_dir}: overwrote {S.prefix}.kgout, '
                          f'{S.prefix}.kglog, {S.prefix}.sigma and {S.prefix}.epsilon.')
        S.log.append('WARNING: ' + S.warnings[-1])
        print(' ' + S.log[-1])
    comp = [('xx', 0, 0), ('yy', 1, 1), ('zz', 2, 2), ('xy', 0, 1), ('xz', 0, 2), ('yz', 1, 2)]
    tr3  = lambda a: np.trace(a, axis1=-2, axis2=-1).real / 3
    soc  = ' (spin-orbit: spinor orbitals, V_SO in the velocity)' if S.nspinor == 2 else ''
    run  = (f'{S.prefix}: formula {S.formula}, broadening {S.broadening} (eta {S.eta} Ha), Drude {S.drude}, '
            f'velocity {S.velocity}{soc}, {S.nb} bands')

    # spectra
    cols = [S.omega, S.omega * HA2EV]
    for _, i, j in comp:
        cols += [S.sigma1[:, i, j], S.sigma2[:, i, j]]
    cols += [tr3(S.sigma1), tr3(S.sigma2), tr3(S.sigma1) * AU2SI]
    head = 'omega_Ha omega_eV ' + ' '.join(f'sigma1_{c} sigma2_{c}' for c, _, _ in comp) + ' sigma1_avg sigma2_avg sigma1_avg_S/m'
    np.savetxt(base + '.sigma' + S.suffix, np.column_stack(cols), fmt='%.10e',
               header=f'{run}\nsigma in atomic units (1 a.u. = {AU2SI:.6e} S/m)\n{head}')

    w    = S.omega > 0
    cols = [S.omega[w], S.omega[w] * HA2EV]
    for _, i, j in comp:
        cols += [S.epsilon[:, i, j].real, S.epsilon[:, i, j].imag]
    cols += [tr3(S.epsilon.real), tr3(S.epsilon.imag)]
    head = 'omega_Ha omega_eV ' + ' '.join(f'eps1_{c} eps2_{c}' for c, _, _ in comp) + ' eps1_avg eps2_avg'
    np.savetxt(base + '.epsilon' + S.suffix, np.column_stack(cols), fmt='%.10e', header=f'{run}\n{head}')

    # .kgout (results, SPARC .out layout) and .kglog (messages and checks)
    dc    = S.dc_parts.sum(0)
    avg   = np.trace(dc) / 3
    bar   = '*' * 75
    box   = lambda *rows, close=True: bar + '\n' + ''.join('*' + r.center(73) + '*\n' for r in rows) + (bar + '\n') * close
    sec   = lambda title: f'{bar}\n{title.center(75).rstrip()}\n{bar}\n'
    kv    = lambda label, val: f'{label:<35}:  {val}\n'                          # 'Label : value' (SPARC .out)
    num   = lambda label, val, unit='': f'{label:<35}:{val:18.10E}' + (f' ({unit})' if unit else '') + '\n'
    head  = box('opticalspectra: Kubo-Greenwood optical spectra of a SPARC run',
                'Copyright (c) 2026 Material Physics & Mechanics Group, Georgia Tech',
                f'Start time: {time.ctime(S.t0)}', close=False)                    # closed by the first section bar
    terms = ['kinetic'] + (['nonlocal'] + (['spin-orbit'] if S.nspinor == 2 else []) if S.velocity == 'gradient+nonlocal' else [])
    inp   = {'OUT_FILE': S.out_file, 'EIGEN_FILE': S.eigen_file, 'PSI_FILE': S.psi_file, 'STATIC_FILE': S.static_file,
             'PSP_DIR': S.psp_dir, 'OMEGA_MIN': S.omega_min, 'OMEGA_MAX': S.omega_max, 'N_OMEGA': S.n_omega,
             'ETA': S.eta, 'FORMULA': S.formula, 'BROADENING': S.broadening, 'DRUDE': S.drude, 'VELOCITY': S.velocity,
             'N_BANDS': S.nb, 'OUT_DIR': S.out_dir, 'PLOT': S.plot, 'REF_FILES': ' '.join(S.ref_files),
             'N_THREADS': S.n_threads}
    name  = lambda ext: f'{S.prefix}{ext}{S.suffix}'
    with fp:
        fp.write(head + sec('Input parameters') + ''.join(f'{k}: {v}\n' for k, v in inp.items()))
        fp.write(sec('System') + kv('SPARC run', S.prefix) + kv('Exchange correlation', S.xc)
                 + kv('Spin type', f'SPIN_TYP {S.spin_typ}' + (', spin-orbit' if S.nspinor == 2 else ''))
                 + kv('Number of k-points', len(S.wk)) + kv('Number of bands used', S.nb)
                 + kv('FD grid', ' '.join(map(str, S.N))) + kv('FD order', S.fd_order) + kv('Boundary conditions', ' '.join(S.BC))
                 + num('Volume', S.Omega, 'Bohr^3') + kv('Total number of electrons', f'{S.N_e:g}')
                 + num(f'Smearing ({S.smearing_type})', 1.0 / S.beta, 'Ha') + num('Fermi level', S.E_F, 'Ha')
                 + kv('Pseudopotentials', ' '.join(S.psp_files)) + kv('Velocity terms', ', '.join(terms))
                 + kv('K-point-spin threads', S.n_pool) + kv('BLAS threads', S.blas)
                 + kv('Estimated peak memory usage', format_bytes(S.mem['total'])))
        fp.write(sec('DC conductivity (omega = 0)') + 'Conductivity tensor (a.u.):\n'
                 + ''.join(f'{r[0]: .10E} {r[1]: .10E} {r[2]: .10E}\n' for r in dc)
                 + num('Conductivity, Tr/3', avg, 'a.u.') + num('Conductivity, Tr/3', avg * AU2SI, 'S/m')
                 + (num('Resistivity', 1e8 / (avg * AU2SI), 'micro-Ohm cm') if avg > 0 else kv('Resistivity', 'inf'))
                 + ''.join(num(f'{lab} part, Tr/3', np.trace(p) / 3, 'a.u.')
                           for lab, p in zip(['Intraband', 'Degenerate-pair', 'Interband'], S.dc_parts)))
        fp.write(sec('Sum rules') + num('f-sum rule, all pairs', S.fsum[0]) + num('f-sum rule, interband pairs', S.fsum[1]))
        fp.write(sec('Output files') + kv('Conductivity printed to', name('.sigma')) + kv('Dielectric function printed to', name('.epsilon'))
                 + (kv('Plots printed to', f'{S.prefix}_sigma{S.suffix}.png, {S.prefix}_epsilon{S.suffix}.png') if S.plot else '')
                 + kv('Messages and checks printed to', f'{name(".kglog")} (warnings: {len(S.warnings)})'))
        fp.write(sec('Timing info') + kv('Setup', f'{S.t_setup:.3f} sec') + kv('k-point and spin sums', f'{S.t_loop:.3f} sec')
                 + kv('CPU time, k-point and spin sums', f'{S.c_loop:.3f} sec (all threads)')
                 + kv('Total walltime', f'{time.time() - S.t0:.3f} sec') + '_' * 75 + '\n\n'
                 + box('Material Physics & Mechanics Group, Georgia Tech',
                       'Method: L. Calderin et al., Comput. Phys. Commun. 221, 118 (2017)'))
    with open(base + '.kglog' + S.suffix, 'w') as fl:
        fl.write(head + sec('Messages') + ''.join(m + '\n' for m in S.log)
                 + ('' if S.warnings else 'NOTE: exact Kubo-Greenwood options, no approximations.\n')
                 + 'NOTE: the f-sum rule equals 1 only for complete bands and a local potential (Calderin Eqs. 41-44).\n')
        fl.write(sec('Checks') + num('Max |psi^H psi dV - I|', S.err_norm) + num('Max |v - v^H|', S.err_herm)
                 + kv('Printed occupation factor', f'{S.occ_factor:.6f} (expected 2/Nspin/Nspinor)'))


def plot_spectra(S):
    """
    **********************************************************************
     Purpose:
       - <name>_sigma[_NN].png   : Tr sigma1/3 and Tr sigma2/3 (10^6 S/m) vs h-bar omega (eV)
       - <name>_epsilon[_NN].png : Tr eps1/3 and Tr eps2/3 vs h-bar omega (eV)
       - reference spectra (2-column files: eV, S/m; S.ref_files, at most 3)
         are overlaid on the sigma1 panel, as in Calderin et al. Figs. 5-11

     Input Parameters:
       S : SimpleNamespace -- results of kubo_greenwood.
    **********************************************************************
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#fcfcfb'
    SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']      # categorical slots 1-4, fixed order
    tr3  = lambda a: np.trace(a, axis1=-2, axis2=-1).real / 3
    refs = [(os.path.basename(f), np.loadtxt(f)) for f in S.ref_files[:3]]
    w    = S.omega > 0
    figs = {'sigma'  : (S.omega * HA2EV,    [tr3(S.sigma1) * AU2SI * 1e-6, tr3(S.sigma2) * AU2SI * 1e-6],
                        ['Tr σ₁/3 (10⁶ S/m)', 'Tr σ₂/3 (10⁶ S/m)']),
            'epsilon': (S.omega[w] * HA2EV, [tr3(S.epsilon.real),  tr3(S.epsilon.imag)],   ['Tr ε₁/3', 'Tr ε₂/3'])}
    for name, (x, ys, labels) in figs.items():
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.0), facecolor=SURF)
        for i, (ax, y, lab) in enumerate(zip(axs, ys, labels)):
            ax.set_facecolor(SURF)
            ax.axhline(0.0, color=AXIS, lw=0.8)
            ax.plot(x, y, color=SERIES[0], lw=1.2, solid_joinstyle='round', label=f'{S.prefix} (this work)')
            if name == 'sigma' and i == 0 and refs:
                for (rname, r), c in zip(refs, SERIES[1:]):
                    ax.plot(r[:, 0], r[:, 1] * 1e-6, color=c, lw=1.2, solid_joinstyle='round', label=rname)
                ax.legend(frameon=False, labelcolor=INK)
            ax.set_ylabel(lab, color=INK)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color(AXIS)
            ax.tick_params(colors=AXIS, labelcolor=MUTED)
        axs[0].set_title(f'{S.prefix}: Kubo-Greenwood {"conductivity" if name == "sigma" else "dielectric function"} '
                         f'\n{S.formula}, {S.broadening}, η = {S.eta*HA2EV:.3g} eV, Drude {"on" if S.drude else "off"}, {S.velocity}',
                         color=INK, loc='left', fontsize=10)
        axs[-1].set_xlabel('ħω (eV)', color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(S.out_dir, f'{S.prefix}_{name}{S.suffix}.png'), dpi=150, facecolor=SURF)
        plt.close(fig)
