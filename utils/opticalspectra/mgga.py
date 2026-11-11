"""
Kubo-Greenwood Optical Spectra Calculator for SPARC Output Files

    Reference: SPARC xc/mgga/mGGAscan.c, mGGArscan.c, mGGAr2scan.c; exchangeCorrelation.c
               (Calculate_Vxc, add_rho_core:1290, calculate_square_norm_of_gradient:1198);
               xc/mgga/mGGAtauTransferTauVxc.c (compute_Kinetic_Density_Tau:32)

    Meta-GGA potential V_tau = d(rho eps_xc)/d tau of the SCAN, rSCAN and r2SCAN functionals
    (SPIN_TYP 0 and 1), which enters the velocity through i[H_tau, r] (operators.mgga_operator).
    SPARC does not write V_tau; it is reconstructed from the orbitals of the run by a port of
    SPARC's functional routines, with the same inputs and numerical thresholds:
    - electron density rho from the orbitals, floored at 1e-14; no model core density, since
      SPARC does not allow NLCC pseudopotentials with meta-GGA functionals
    - squared density gradient sigma = |grad rho|^2 from SPARC's FD gradient and metric,
      floored at 1e-10
    - kinetic-energy density tau = 1/2 sum_k w_k sum_n g_s f_n |grad psi_n|^2
    - spin-polarized runs: exchange from the spin-scaling relation, correlation from the total
      density and the spin polarization zeta; V_tau,s = v3x,s + v3c

    @file    mgga.py
    @brief   Meta-GGA V_tau (SCAN, rSCAN, r2SCAN) for the velocity term i[H_tau, r]
    @authors Shubhang K Trivedi <strivedi44@gatech.edu>
             Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

    Copyright (c) 2026 Material Physics & Mechanics Group, Georgia Tech.
"""

# *** Imports ****************************************************************
import numpy as np

from operators import gradient

C13 = (3.0 * np.pi**2) ** (1.0 / 3.0)
CX0 = -3.0 / (4.0 * np.pi) * (9.0 * np.pi / 4.0) ** (1.0 / 3.0)
B1C, B2C, B3C, R_H, F0 = 0.0285764, 0.0889, 0.125541, 0.031091, -0.9
BETA0 = 0.06672455060314922
CHI0  = (3.0 * np.pi**2 / 16.0) ** (2.0 / 3.0) * (BETA0 * 0.1 / 0.1778) / (CX0 - F0)
PW92  = {'0': (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),          # (A, alpha1, beta1..beta4)
         'ac': (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
         '1': (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)}
POLY  = {'x': (1.0, -0.667, -0.4445555, -0.663086601049, 1.451297044490, -0.887998041597, 0.234528941479, -0.023185843322),
         'c': (1.0, -0.64, -0.4352, -1.535685604549, 3.061560252175, -1.915710236206, 0.516884468372, -0.051848879792)}
SWITCH = {'x': (0.667, 0.8, 1.24), 'c': (0.64, 1.5, 0.7)}                      # (c1, c2, d)
ETA_R2, DP2 = 0.001, 0.361

# alpha used by each functional and d alpha / d tau  (basic_*_variables);  ds = 1 for unpolarized
ALPHA = {
    'SCAN'  : lambda tau, tw, tu: ((tau - tw) / tu, 1.0 / tu),
    'RSCAN' : lambda tau, tw, tu: (lambda a: (a**3 / (a**2 + 1e-3),
                                              (3*a**2*(a**2 + 1e-3) - 2*a**4) / (a**2 + 1e-3)**2 / (tu + 1e-4)))((tau - tw) / (tu + 1e-4)),
    'R2SCAN': lambda tau, tw, tu: ((tau - tw) / (tu + ETA_R2 * tw), 1.0 / (tu + ETA_R2 * tw)),
}


# *** Functions **************************************************************
def _switch(name, k, a):
    """ f_x (k='x') or f_c (k='c') and df/dalpha: SCAN exponential pair; rSCAN/r2SCAN with the 7th-order polynomial """
    c1, c2, d = SWITCH[k]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        lo, hi = np.exp(-c1 * a / (1.0 - a)), -d * np.exp(c2 / (1.0 - a))
        dlo, dhi = lo * (-c1 / (1.0 - a)**2), hi * (c2 / (1.0 - a)**2)
    if name == 'SCAN':
        return np.where(a > 1.0, hi, lo), np.where(a > 1.0, dhi, dlo)
    c = POLY[k]
    poly = sum(c[j] * a**j for j in range(8))
    dpoly = sum(j * c[j] * a**(j - 1) for j in range(1, 8))
    return (np.where(a > 2.5, hi, np.where(a > 0.0, poly, lo)),
            np.where(a > 2.5, dhi, np.where(a > 0.0, dpoly, dlo)))


def _exchange(name, rho, sigma, tau):
    """
    **********************************************************************
     Purpose:
       - unpolarized exchange: eps_x and v3x = rho d(eps_x)/d tau
     Reference:
       - SPARC: Calculate_scanx (mGGAscan.c:87), Calculate_rscanx (mGGArscan.c:92),
                Calculate_r2scanx (mGGAr2scan.c:86)
    **********************************************************************
    """
    s   = np.sqrt(sigma) / (2.0 * C13 * rho**(4.0/3.0))
    tw  = sigma / (8.0 * rho)
    a, dadt = ALPHA[name](tau, tw, 0.3 * C13**2 * rho**(5.0/3.0))
    fx, dfx = _switch(name, 'x', a)
    exu = -3.0 / (4.0 * np.pi) * np.cbrt(3.0 * np.pi**2 * rho)               # eps_x of the uniform gas
    if name == 'R2SCAN':
        k1, mu = 0.065, 10.0 / 81.0
        p   = s * s
        x   = ((20.0/27.0 + ETA_R2*5.0/3.0) * (-0.162742) * np.exp(-p*p / DP2**4) + mu) * p
        hx1 = 1.0 + k1 - k1 / (1.0 + x / k1)
        hx0 = 1.174
        gx  = 1.0 - np.exp(-4.9479 / p**0.25)
        dF  = gx * (hx0 - hx1) * dfx * dadt
    else:
        k1, mu = 0.065, 10.0 / 81.0
        b2 = np.sqrt(5913.0 / 405000.0); b1 = 511.0 / 13500.0 / (2.0 * b2); b3 = 0.5
        b4 = mu * mu / k1 - 1606.0 / 18225.0 - b1 * b1
        s2 = s * s
        term1 = 1.0 + b4 * s2 / mu * np.exp(-abs(b4) * s2 / mu)
        e3    = np.exp(-b3 * (1.0 - a)**2)
        term3 = 2.0 * (b1 * s2 + b2 * (1.0 - a) * e3)
        x     = mu * s2 * term1 + (term3 / 2.0)**2
        hx1   = 1.0 + k1 - k1 / (1.0 + x / k1)
        hx0   = 1.174
        gx    = 1.0 - np.exp(-4.9479 / np.sqrt(s))
        term4 = b2 * (-e3 + (1.0 - a) * e3 * (2.0 * b3 * (1.0 - a)))
        dF    = gx * (1.0 - fx) * (term3 * term4 / (1.0 + x / k1)**2) * dadt + gx * (hx0 - hx1) * dfx * dadt
    Fx = (hx1 + fx * (hx0 - hx1)) * gx
    return exu * Fx, rho * exu * dF


def _correlation(name, rho, sigma, tau, zeta):
    """
    **********************************************************************
     Purpose:
       - correlation for total rho, sigma, tau and spin polarization zeta (zeta = 0:
         the unpolarized routines): eps_c and v3c = rho d(eps_c)/d tau
     Reference:
       - SPARC: Calculate_scanc(_spin) (mGGAscan.c:206, 579), Calculate_rscanc(_spin)
                (mGGArscan.c:225, 508), Calculate_r2scanc(_spin) (mGGAr2scan.c:244, 553)
    **********************************************************************
    """
    s   = np.sqrt(sigma) / (2.0 * C13 * rho**(4.0/3.0))
    ds  = ((1.0 + zeta)**(5.0/3.0) + (1.0 - zeta)**(5.0/3.0)) / 2.0
    a, dadt = ALPHA[name](tau, sigma / (8.0 * rho), 0.3 * C13**2 * rho**(5.0/3.0) * ds)
    fc, dfc = _switch(name, 'c', a)
    phi = ((1.0 + zeta)**(2.0/3.0) + (1.0 - zeta)**(2.0/3.0)) / 2.0
    dxz = ((1.0 + zeta)**(4.0/3.0) + (1.0 - zeta)**(4.0/3.0)) / 2.0
    rs  = (0.75 / (np.pi * rho)) ** (1.0 / 3.0)
    srs = np.sqrt(rs)
    # eps_c^0
    ecLDA0 = -B1C / (1.0 + B2C * srs + B3C * rs)
    Gc  = (1.0 - 2.3631 * (dxz - 1.0)) * (1.0 - zeta**12)
    w0  = np.exp(-ecLDA0 / B1C) - 1.0
    H0  = B1C * np.log(1.0 + w0 * (1.0 - (1.0 + 4.0 * CHI0 * s * s) ** -0.25))
    ec0 = (ecLDA0 + H0) * Gc
    # eps_c^1: PW92 LSDA (spin interpolation) + H1
    pw  = {}
    for key, (A, a1, b1, b2, b3, b4) in PW92.items():
        q0, q1 = -2.0 * A * (1.0 + a1 * rs), 2.0 * A * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs)
        q1p, den = A * (b1 / srs + 2.0 * b2 + 3.0 * b3 * srs + 4.0 * b4 * rs), 1.0 / (q1 * q1 + q1)
        lg = -np.log(q1 * q1 * den)
        pw[key] = (q0 * lg, -2.0 * A * a1 * lg - q0 * q1p * den)                 # G(rs), dG/drs
    fz  = ((1.0 + zeta)**(4.0/3.0) + (1.0 - zeta)**(4.0/3.0) - 2.0) / (2.0**(4.0/3.0) - 2.0)
    z4  = zeta**4
    ec_lsda1 = pw['0'][0] + fz * (z4 * (pw['1'][0] - pw['0'][0] + pw['ac'][0] / 1.709921) - pw['ac'][0] / 1.709921)
    beta = BETA0 * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs)
    rphi3 = R_H * phi**3
    w1  = np.exp(-ec_lsda1 / rphi3) - 1.0
    t   = (3.0 * np.pi**2 / 16.0) ** (1.0 / 3.0) * s / (phi * srs)
    y   = beta / (R_H * w1) * t * t
    if name == 'R2SCAN':
        dec_lsda1 = pw['0'][1] + fz * (z4 * (pw['1'][1] - pw['0'][1] + pw['ac'][1] / 1.709921) - pw['ac'][1] / 1.709921)
        dfc2 = sum(j * POLY['c'][j] for j in range(1, 8))
        dec_lsda0 = B1C * (0.5 * B2C / srs + B3C) / (1.0 + B2C * srs + B3C * rs)**2 * Gc
        p   = s * s
        y   = y - (dfc2 / (27.0 * R_H * ds * phi**3 * w1) * (20.0 * rs * (dec_lsda0 - dec_lsda1)
                                                           - 45.0 * ETA_R2 * (ecLDA0 * Gc - ec_lsda1)) * p * np.exp(-p * p / DP2**4))
    g   = (1.0 + 4.0 * y) ** -0.25
    ec1 = ec_lsda1 + rphi3 * np.log(1.0 + w1 * (1.0 - g))
    return ec1 + fc * (ec0 - ec1), rho * dfc * dadt * (ec0 - ec1)


def _density_grids(S):
    """
    **********************************************************************
     Purpose:
       - rho (floored) and tau per spin column, and sigma, as SPARC passes them to its
         meta-GGA routines: columns [total] (SPIN_TYP 0) or [total, up, dn]
     Reference:
       - SPARC: add_rho_core (exchangeCorrelation.c:1290), calculate_square_norm_of_gradient
                (exchangeCorrelation.c:1198), compute_Kinetic_Density_Tau (mGGAtauTransferTauVxc.c:32)
    **********************************************************************
    """
    Ui = np.linalg.inv(S.U)
    g  = Ui.T @ Ui                                                            # lapcT
    ns = S.psi.shape[2]
    rv = np.zeros((ns, S.Nd)); tv = np.zeros((ns, S.Nd))
    for k in range(len(S.wk)):
        for s_ in range(ns):
            f  = S.f[s_, k]
            on = np.nonzero(f > 0.0)[0]
            x  = np.ascontiguousarray(S.psi[k, on, s_]['psi'].T)             # (Nd, n_occ), C order
            wf = S.wk[k] * S.g_s * f[on]
            rv[s_] += (np.abs(x)**2) @ wf
            D  = [gradient(S, x, S.kred[k], d) for d in range(3)]
            tv[s_] += 0.5 * sum(g[i, j] * (D[i].conj() * D[j]).real for i in range(3) for j in range(3)
                                if abs(g[i, j]) > 1e-12) @ wf
    # SPARC's columns, floor 1e-14; spin: total = up + dn
    rho, tau = np.maximum(rv, 1e-14), tv
    if ns == 2:
        rho, tau = np.vstack([rho.sum(0), rho]), np.vstack([tv.sum(0), tv])
    Dr  = [gradient(S, rho.T, np.zeros(3), d) for d in range(3)]
    sig = np.maximum(sum(g[i, j] * Dr[i] * Dr[j] for i in range(3) for j in range(3)).real.T, 1e-10)
    return rho, sig, tau


def vtau(S):
    """
    **********************************************************************
     Purpose:
       - V_tau per spin channel (nspin, Nd) of the run's meta-GGA functional (S.xc),
         as SPARC's vxcMGGA3 = v3x + v3c
     Reference:
       - SPARC: Calculate_Vxc (exchangeCorrelation.c:65-199 unpolarized, 270-364 spin)
     Input Parameters:
       S : SimpleNamespace -- run data (psi, f, g_s, wk, kred, grid, U), S.xc.
     Returns:
       (nspin, Nd) array.
    **********************************************************************
    """
    name = S.xc.upper()
    rho, sig, tau = _density_grids(S)
    if rho.shape[0] == 1:
        return (_exchange(name, rho[0], sig[0], tau[0])[1] + _correlation(name, rho[0], sig[0], tau[0], 0.0)[1])[None, :]
    v3c = _correlation(name, rho[0], sig[0], tau[0], (rho[1] - rho[2]) / rho[0])[1]
    return np.array([_exchange(name, 2.0 * rho[s], 4.0 * sig[s], 2.0 * tau[s])[1] + v3c for s in (1, 2)])


def exc(S):
    """ E_xc = sum rho eps_xc dV, for comparison with SPARC's printed value """
    name = S.xc.upper()
    rho, sig, tau = _density_grids(S)
    if rho.shape[0] == 1:
        e = _exchange(name, rho[0], sig[0], tau[0])[0] + _correlation(name, rho[0], sig[0], tau[0], 0.0)[0]
        return (rho[0] * e).sum() * S.dV
    ex = sum(rho[s] * _exchange(name, 2.0 * rho[s], 4.0 * sig[s], 2.0 * tau[s])[0] for s in (1, 2))
    return (ex + rho[0] * _correlation(name, rho[0], sig[0], tau[0], (rho[1] - rho[2]) / rho[0])[0]).sum() * S.dV
