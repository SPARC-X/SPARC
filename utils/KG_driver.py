"""
*************************************************************************************
*     Program  :  KGconductivity  (Kubo-Greenwood conductivity from SPARC output)
*     Module   :  KG_driver
*     Authors  :  <placeholder>
*
*
*     Created  :  2026-11-04
*     Modified :  2026-11-04
*
*     Purpose  :  Top-level entry point and CLI:
*                   - collect inputs (command line, else inputparams.py) into S
*                   - read the SPARC output files and build the projectors
*                   - loop over spin and k-points: velocity matrix, weights,
*                     sigma~1 and sigma~2 (summed with the k-point weights)
*                   - sigma, epsilon, DC conductivity, f-sum rule, output files
*
*     Routines :  KG_driver()   (public)
*
*     Reference:  L. Calderin, V. V. Karasiev, S. B. Trickey, Comput. Phys. Commun.
*                 221, 118 (2017);  SPARC  --  https://github.com/SPARC-X/SPARC
*************************************************************************************
"""

# *** Imports ****************************************************************
import os
import time
import argparse
from types import SimpleNamespace

import numpy as np

import inputparams
from readfiles            import read_out, read_eigen, read_psi, read_static, read_psp8
from operators            import nonlocal_projectors
from KGconductivitytensor import (DEGEN_TOL, occupations, velocity_matrix, weights_F, sigma1_tilde, sigma2_tilde,
                                  sigma1, sigma2, sigma, epsilon, checks, write_output, plot_spectra)

# Warnings for non-default method options: each is an approximation to the exact Kubo-Greenwood result
APPROXIMATIONS = {
    ('formula',    'approximated'): 'formula = approximated: Df/omega form (Calderin Eq. 9); differs from the exact result '
                                    'off resonance and at low omega, and contains no intraband (Drude) term.',
    ('broadening', 'gaussian')    : 'broadening = gaussian: Gaussian representation of the delta function; wrong '
                                    'low-frequency / DC limit (Calderin Sec. 6.1).',
    ('drude',      False)         : 'drude = False: intraband and degenerate (Drude) term omitted; the low-frequency / DC '
                                    'conductivity of metals is missing.',
    ('velocity',   'gradient')    : 'velocity = gradient: momentum -i grad only; the nonlocal pseudopotential commutator '
                                    'i[V_NL, r] is neglected.',
}


# *** Functions **************************************************************
def KG_driver(formula='exact', broadening='lorentzian', drude=True, velocity='gradient+nonlocal', **cli):
    """
    **********************************************************************
     Purpose:
       - full calculation. Workflow (function -> file):
           1. read_out, read_eigen, read_psi, read_static, read_psp8  (readfiles.py)
           2. occupations                                             (KGconductivitytensor.py)
           3. nonlocal_projectors                                     (operators.py)
           4. per spin, k: velocity_matrix, weights_F,
              sigma1_tilde, sigma2_tilde                              (KGconductivitytensor.py)
           5. sigma1, sigma2, sigma, epsilon, write_output,
              plot_spectra                                            (KGconductivitytensor.py)
       - parallelization over k-points: placeholder (S.np_kpt), serial for now

     Input Parameters (method options: change the defaults here, not inputs of the user;
                       the defaults are the exact result, any other choice prints a warning):
       formula    : 'exact' (Df/De, Calderin Eq. 8) | 'approximated' (Df/omega, Eq. 9, as ABINIT conducti)
       broadening : 'lorentzian' | 'gaussian' (same full width 2*ETA)
       drude      : True (intraband/degenerate term) | False (interband only)
       velocity   : 'gradient+nonlocal' (-i grad + i[V_NL, r]) | 'gradient' (-i grad)
       cli        : dict -- command-line values (None = take the value from inputparams.py).

     Returns:
       S : SimpleNamespace -- inputs, data and results.
    **********************************************************************
    """
    S = SimpleNamespace(**{k.lower(): v for k, v in vars(inputparams).items() if k.isupper()})
    S.__dict__.update({k: v for k, v in cli.items() if v is not None})
    S.formula, S.broadening, S.drude, S.velocity = formula, broadening, drude, velocity
    S.warnings = [msg for (key, val), msg in APPROXIMATIONS.items() if getattr(S, key) == val]
    for msg in S.warnings:
        print(' WARNING: ' + msg)
    S.t0      = time.time()
    S.psp_dir = S.psp_dir or os.path.dirname(os.path.abspath(S.out_file))
    S.prefix  = (os.path.basename(S.out_file).replace('.out', '')                                  # e.g. Si2_exact_L_drude_grad
                 + ('_exact' if S.formula == 'exact' else '_approx') + f'_{S.broadening[0].upper()}'
                 + ('_drude' if S.drude else '_nodrude') + ('_grad' if S.velocity == 'gradient' else '_gradnl'))

    ''' Read the SPARC run and set up the k-independent parts '''
    read_out(S); read_eigen(S); read_psi(S); read_static(S)
    assert S.psi.shape[0] == len(S.wk) and S.psi.shape[2] == S.eig.shape[0], '.psi must hold all k-points and spins (PRINT_ORBITAL)'
    S.psds  = [read_psp8(f if os.path.isabs(f) else os.path.join(S.psp_dir, f)) for f in S.psp]
    occupations(S)
    if S.velocity == 'gradient+nonlocal':
        nonlocal_projectors(S)
    S.nb    = S.psi.shape[1] if S.n_bands is None else min(S.n_bands, S.psi.shape[1])
    S.omega = np.linspace(S.omega_min, S.omega_max, S.n_omega)
    print(f' {S.prefix}: SPIN_TYP {S.spin_typ}, {len(S.wk)} k-points, {S.nb} bands, grid {S.N.tolist()} '
          f'({time.time() - S.t0:.2f} s)')

    ''' Sum over spin and k-points '''
    st1 = st2 = 0.0
    S.dc_parts, S.fsum, S.err_norm, S.err_herm = np.zeros((3, 3, 3)), np.zeros(2), 0.0, 0.0
    for k in range(len(S.wk)):
        for s in range(S.psi.shape[2]):
            t1  = time.time()
            psi = np.array(S.psi[k, :S.nb, s]['psi']).T                          # (Nd, nb)
            v   = velocity_matrix(S, psi, S.kred[k])
            eps = S.eig[s, k, :S.nb]
            F   = weights_F(eps, S.f[s, k, :S.nb], S.mdfde[s, k, :S.nb], S.drude, S.formula)
            Fx  = weights_F(eps, S.f[s, k, :S.nb], S.mdfde[s, k, :S.nb], S.drude)          # Df/De, for the f-sum rule
            de  = eps[:, None] - eps[None, :]
            Z   = np.einsum('anm,bmn->nmab', v, v).real                           # non-magnetic: Re Z
            st1 = st1 + S.wk[k] * sigma1_tilde(S, F, Z, de, S.omega)
            st2 = st2 + S.wk[k] * sigma2_tilde(S, F, Z, de, S.omega)

            # DC split (intraband / degenerate / interband) and f-sum rule (Calderin Eqs. 41-44)
            same = np.eye(S.nb, dtype=bool)
            deg  = np.abs(de) <= DEGEN_TOL
            S.dc_parts += S.wk[k] * np.array([sigma1_tilde(S, F * m, Z, de, np.zeros(1))[0] for m in [same, deg & ~same, ~deg]])
            FtrZ = Fx * np.einsum('nmaa->nm', Z)
            S.fsum += S.wk[k] * S.g_s / (3 * S.N_e) * np.array([FtrZ.sum(), FtrZ[~deg].sum()])
            e_n, e_h = checks(S, psi, v)
            S.err_norm, S.err_herm = max(S.err_norm, e_n), max(S.err_herm, e_h)
            print(f'   k-point {k+1}/{len(S.wk)}, spin {s}: {time.time() - t1:.2f} s')

    ''' Conductivity and dielectric tensors, output '''
    S.sigma1, S.sigma2 = sigma1(st1, st2), sigma2(st1, st2)
    S.sigma   = sigma(S.sigma1, S.sigma2)
    S.epsilon = epsilon(S.sigma, S.omega)
    write_output(S)
    if S.plot:
        plot_spectra(S)
    print(f' Done. Results in {S.out_dir}/{S.prefix}.* ({time.time() - S.t0:.2f} s)')
    return S


# CLI: python KG_driver.py [--out_file X.out ...]  -- anything not given is taken from inputparams.py
parser = argparse.ArgumentParser(prog='KG_driver', description='Kubo-Greenwood conductivity from SPARC output.')
for key, typ in [('out_file', str), ('eigen_file', str), ('psi_file', str), ('static_file', str), ('psp_dir', str),
                 ('omega_min', float), ('omega_max', float), ('n_omega', int), ('eta', float), ('n_bands', int),
                 ('out_dir', str)]:
    parser.add_argument('--' + key, type=typ)
parser.add_argument('--ref_files', nargs='*')
parser.add_argument('--no_plot', dest='plot', action='store_false', default=None)
parser.add_argument('--np_kpt', type=int, help='k-point parallel groups (placeholder, not used yet)')

if __name__ == '__main__':
    KG_driver(**vars(parser.parse_args()))
