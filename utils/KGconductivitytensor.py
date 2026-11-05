"""
*************************************************************************************
*     Program  :  KGconductivity  (Kubo-Greenwood conductivity from SPARC output)
*     Module   :  KGconductivitytensor
*     Authors  :  <placeholder>
*
*
*     Created  :  2026-11-04
*     Modified :  2026-11-04
*
*     Purpose  :  Kubo-Greenwood conductivity tensor and dielectric tensor
*                 (Hartree atomic units, time dependence e^{-i omega t}):
*                   sigma~1, sigma~2 : Eqs. (10)-(11) of Calderin et al. (2017) with
*                                      Z_nm = v^a_nm v^b_mn, v = -i grad + i[V_NL, r]
*                   sigma1 = Re sigma~1 - Im sigma~2,  sigma2 = -(Im sigma~1 + Re sigma~2)
*                   epsilon = delta + 4 pi i sigma / omega
*                 Intraband and degenerate pairs use F -> -df/de (Eqs. 21-22).
*                 Non-magnetic scope (v1): Im Z cancels between k and -k, so Re Z is
*                 used, which is also what SPARC's time-reversal folded k-mesh needs.
*
*     Routines :  occupations()      velocity_matrix()   weights_F()
*                 sigma1_tilde()     sigma2_tilde()      sigma1()
*                 sigma2()           sigma()             epsilon()
*                 checks()           write_output()      plot_spectra()
*
*     Reference:  L. Calderin, V. V. Karasiev, S. B. Trickey, Comput. Phys. Commun.
*                 221, 118 (2017);  SPARC  --  https://github.com/SPARC-X/SPARC
*************************************************************************************
"""

# *** Imports ****************************************************************
import os
import time

import numpy as np
from scipy.special import dawsn

from operators import gradient, nonlocal_operator

HA2EV     = 27.211386245988       # Ha -> eV (CONST_EH in SPARC isddft.h)
AU2SI     = 4.599848e6            # 1 a.u. of conductivity, e^2/(hbar a0), in S/m
DEGEN_TOL = 1e-6                  # |e_n - e_m| (Ha) below which a pair is treated as degenerate
F_TOL     = 1e-12                 # pairs with |F_nm| below this are skipped

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
# (my construction: the paper gives no sigma2 for Eq. 9). w = 0 uses the limit -K'(x0) (cf. Eqs. 82-83).
FORMULA = {
    'exact'       : lambda K, x0, w, h: K(x0 - w),
    'approximated': lambda K, x0, w, h: np.where(w != 0, (K(x0 - w) - K(x0)) / np.where(w != 0, w, 1.0),
                                                 (K(x0 - h) - K(x0 + h)) / (2 * h)),
}


# *** Functions **************************************************************
def occupations(S):
    """
    **********************************************************************
     Purpose:
       - f from the printed occupations: SPARC prints (2/Nspin/Nspinor) f
         (printing.c:230); the factor is fixed by sum_k w_k sum g_s f = N_e
       - -df/de from f and beta (Fermi-Dirac) or from e, E_F, beta (Gaussian)

     Input Parameters:
       S : SimpleNamespace -- occ_printed, wk, N_e, spin_typ, smearing data.

     Returns:
       S : SimpleNamespace -- S.g_s, S.occ_factor, S.f, S.mdfde (nspin, nkpt, nstates).
    **********************************************************************
    """
    S.g_s        = 2.0 if S.spin_typ == 0 else 1.0
    S.occ_factor = S.g_s * np.einsum('k,skn->', S.wk, S.occ_printed) / S.N_e
    S.f          = S.occ_printed / S.occ_factor
    S.mdfde      = MDFDE[S.smearing_type](S)
    return S


def velocity_matrix(S, psi, kred):
    """
    **********************************************************************
     Purpose:
       - v^a_nm = <psi_n| -i grad_a + i[V_NL, r_a] |psi_m>   (Cartesian a)
           P^a  = -i dV psi^H grad_a psi,  grad_cart = U^{-1} grad_lattice
           NL^a = i (C^H Gamma D^a - D^a^H Gamma C)
       - S.velocity: 'gradient' -> P only (momentum, -i grad);
                     'gradient+nonlocal' -> P + NL (full velocity i[H, r])

     Reference:
       - SPARC: nonCart2Cart_grad (initialization.c:3953)

     Input Parameters:
       S    : SimpleNamespace -- grid, cell, S.nl.
       psi  : (Nd, nb) array  -- orbitals at this k-point and spin.
       kred : (3,) array      -- reduced k-point.

     Returns:
       v : (3, nb, nb) complex array.
    **********************************************************************
    """
    P = np.einsum('ad,dnm->anm', np.linalg.inv(S.U),
                  np.array([-1j * S.dV * (psi.conj().T @ gradient(S, psi, kred, d)) for d in range(3)]))
    if S.velocity == 'gradient':
        return P
    C, D = nonlocal_operator(S, psi, kred)
    G = S.nl.gamma[:, None]
    return P + np.array([1j * (C.conj().T @ (G * D[a]) - D[a].conj().T @ (G * C)) for a in range(3)])


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

     Returns:
       (float, float)
    **********************************************************************
    """
    return (np.abs(psi.conj().T @ psi * S.dV - np.eye(psi.shape[1])).max(),
            np.abs(v - v.conj().transpose(0, 2, 1)).max())


def write_output(S):
    """
    **********************************************************************
     Purpose:
       - <prefix>.kgout    : log (inputs, system, checks, DC tensor, sum rule, timing)
       - <prefix>.sigma    : omega (Ha, eV), sigma1/sigma2 of xx yy zz xy xz yz, Tr/3 (a.u.),
                             Tr sigma1/3 (S/m)
       - <prefix>.epsilon  : omega > 0 (Ha, eV), eps1/eps2 of the same components and Tr/3
       - <prefix>.kg.npz   : omega and the complex sigma and epsilon tensors

     Input Parameters:
       S : SimpleNamespace -- results of KG_driver.
    **********************************************************************
    """
    os.makedirs(S.out_dir, exist_ok=True)
    base = os.path.join(S.out_dir, S.prefix)
    comp = [('xx', 0, 0), ('yy', 1, 1), ('zz', 2, 2), ('xy', 0, 1), ('xz', 0, 2), ('yz', 1, 2)]
    tr3  = lambda a: np.trace(a, axis1=-2, axis2=-1).real / 3
    run  = (f'{S.prefix}: formula {S.formula}, broadening {S.broadening} (eta {S.eta} Ha), Drude {S.drude}, '
            f'velocity {S.velocity}, {S.nb} bands')

    # spectra
    cols = [S.omega, S.omega * HA2EV]
    for _, i, j in comp:
        cols += [S.sigma1[:, i, j], S.sigma2[:, i, j]]
    cols += [tr3(S.sigma1), tr3(S.sigma2), tr3(S.sigma1) * AU2SI]
    head = 'omega_Ha omega_eV ' + ' '.join(f'sigma1_{c} sigma2_{c}' for c, _, _ in comp) + ' sigma1_avg sigma2_avg sigma1_avg_S/m'
    np.savetxt(base + '.sigma', np.column_stack(cols), fmt='%.10e',
               header=f'{run}\nsigma in atomic units (1 a.u. = {AU2SI:.6e} S/m)\n{head}')

    w    = S.omega > 0
    cols = [S.omega[w], S.omega[w] * HA2EV]
    for _, i, j in comp:
        cols += [S.epsilon[:, i, j].real, S.epsilon[:, i, j].imag]
    cols += [tr3(S.epsilon.real), tr3(S.epsilon.imag)]
    head = 'omega_Ha omega_eV ' + ' '.join(f'eps1_{c} eps2_{c}' for c, _, _ in comp) + ' eps1_avg eps2_avg'
    np.savetxt(base + '.epsilon', np.column_stack(cols), fmt='%.10e', header=f'{run}\n{head}')

    np.savez(base + '.kg.npz', omega=S.omega, sigma=S.sigma, epsilon=S.epsilon, omega_eps=S.omega[w],
             sigma_dc_parts=S.dc_parts, run=run)

    # log
    dc  = S.dc_parts.sum(0)
    avg = np.trace(dc) / 3
    with open(base + '.kgout', 'w') as fp:
        fp.write('*' * 75 + '\n*' + 'KGconductivity: Kubo-Greenwood conductivity from SPARC output'.center(73) + '*\n' + '*' * 75 + '\n')
        fp.write('Input\n')
        for key in ['out_file', 'eigen_file', 'psi_file', 'static_file', 'psp_dir']:
            fp.write(f'  {key:<12}: {getattr(S, key)}\n')
        fp.write(f'  omega grid  : {S.omega_min} .. {S.omega_max} Ha, {S.n_omega} points\n')
        fp.write(f'  broadening  : {S.broadening}, eta = {S.eta} Ha (half width; FWHM = {2*S.eta} Ha)\n')
        fp.write(f'  formula     : {S.formula}   Drude term: {S.drude}   velocity: {S.velocity}   bands used: {S.nb}\n')
        fp.write(''.join(f'  WARNING: {m}\n' for m in S.warnings) or '  (exact Kubo-Greenwood options, no approximations)\n')
        fp.write('System\n')
        fp.write(f'  Volume      : {S.Omega:.10E} Bohr^3\n  Grid        : {S.N.tolist()}  FD_ORDER {S.fd_order}  BC {" ".join(S.BC)}\n')
        fp.write(f'  SPIN_TYP    : {S.spin_typ}   k-points: {len(S.wk)}   N_e: {S.N_e}\n')
        fp.write(f'  Smearing    : {S.smearing_type}, beta = {S.beta:.6g} 1/Ha, Fermi level = {S.E_F:.10E} Ha\n')
        fp.write('Checks\n')
        fp.write(f'  max |psi^H psi dV - I|           : {S.err_norm:.3e}\n')
        fp.write(f'  max |v - v^H|                    : {S.err_herm:.3e}\n')
        fp.write(f'  printed occupation = {S.occ_factor:.6f} * f (expected 2/Nspin/Nspinor)\n')
        fp.write('DC conductivity sigma1(omega = 0)\n')
        fp.write('  tensor (a.u.):\n' + ''.join(f'    {r[0]: .10E} {r[1]: .10E} {r[2]: .10E}\n' for r in dc))
        fp.write(f'  Tr/3        : {avg:.10E} a.u. = {avg*AU2SI:.10E} S/m = {avg*AU2SI*1e-8:.10E} (micro-Ohm cm)^-1\n')
        fp.write(f'  resistivity : {1e8/(avg*AU2SI) if avg > 0 else float("inf"):.10E} micro-Ohm cm\n')
        for name, part in zip(['intraband', 'degenerate', 'interband'], S.dc_parts):
            fp.write(f'  {name:<11} : {np.trace(part)/3:.10E} a.u. (Tr/3)\n')
        fp.write('f-sum rule (Calderin Eqs. 41-44; = 1 for complete bands and a local potential)\n')
        fp.write(f'  interband only : {S.fsum[1]:.6f}\n  all pairs      : {S.fsum[0]:.6f}\n')
        fp.write(f'Timing\n  total walltime : {time.time() - S.t0:.3f} sec\n')


def plot_spectra(S):
    """
    **********************************************************************
     Purpose:
       - <prefix>_sigma.png   : Tr sigma1/3 and Tr sigma2/3 (10^6 S/m) vs h-bar omega (eV)
       - <prefix>_epsilon.png : Tr eps1/3 and Tr eps2/3 vs h-bar omega (eV)
       - reference spectra (2-column files: eV, S/m; S.ref_files, at most 3)
         are overlaid on the sigma1 panel, as in Calderin et al. Figs. 5-11

     Input Parameters:
       S : SimpleNamespace -- results of KG_driver.
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
        fig.savefig(os.path.join(S.out_dir, f'{S.prefix}_{name}.png'), dpi=150, facecolor=SURF)
        plt.close(fig)
