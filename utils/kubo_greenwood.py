"""
Kubo-Greenwood Optical Spectra Calculator for SPARC Output Files

    Reference: L. Calderin, V. V. Karasiev, S. B. Trickey, Comput. Phys. Commun. 221, 118 (2017)

    Driver and command-line interface of the Kubo-Greenwood calculator. From the output of a
    single SPARC run it evaluates the linear-response optical conductivity tensor sigma(omega)
    and the dielectric tensor epsilon(omega) in four stages:
    - input: parameters from inputparams.py, overridden by command-line or keyword arguments
    - setup: SPARC output files, nonlocal and spin-orbit projectors
    - accumulation: for each spin channel and k-point, the velocity matrix and the transition
      weights, summed into the k-weighted response functions (Calderin et al., Eqs. 10-11); the
      (k-point, spin) pairs run on N_THREADS threads and are summed in (k-point, spin) order, so
      the results do not depend on N_THREADS
    - results: sigma, epsilon, the DC conductivity with its intraband, degenerate and interband
      contributions, the f-sum rule, consistency checks, output files and optional plots
    Method options other than the defaults approximate the exact Kubo-Greenwood expression and
    are reported as warnings.

    @file    kubo_greenwood.py
    @brief   Entry point and CLI of the Kubo-Greenwood optical spectra calculator
    @authors Shubhang K Trivedi <strivedi44@gatech.edu>
             Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

    Copyright (c) 2026 Material Physics & Mechanics Group, Georgia Tech.
"""

# *** Imports ****************************************************************
import os
import re
import time
import argparse
import contextlib
import importlib.util
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import inputparams
from readfiles   import read_out, read_eigen, read_psi, read_static, read_psp8
from operators   import nonlocal_projectors, soc_projectors
from dielectrics import (DEGEN_TOL, F_TOL, BAND_BLOCK, occupations, velocity_matrix, weights_F, sigma1_tilde,
                         sigma2_tilde, sigma1, sigma2, sigma, epsilon, checks, write_output, plot_spectra, format_bytes)

# Warnings for non-default method options: each is an approximation to the exact Kubo-Greenwood result
REF = 'L. Calderin et al., Comput. Phys. Commun. 221, 118 (2017)'
APPROXIMATIONS = {
    ('formula',    'approximated'): f'formula = approximated: Df/omega form; it omits the intraband (Drude) term and has the '
                                    f'wrong low-frequency / DC limit, so the low-frequency / DC conductivity of metals is '
                                    f'incorrect ({REF}, Eq. 9 and Sec. 6.1).',
    ('broadening', 'gaussian')    : f'broadening = gaussian: Gaussian representation of the delta function; wrong '
                                    f'low-frequency / DC limit ({REF}, Sec. 6.1).',
    ('drude',      False)         : 'drude = False: intraband and degenerate (Drude) term omitted; the low-frequency / DC '
                                    'conductivity of metals is missing.',
    ('velocity',   'gradient')    : 'velocity = gradient: momentum -i grad only; the nonlocal (i[V_NL, r]) and spin-orbit '
                                    '(i[V_SO, r]) commutators are neglected.',
}

# Expected type of each input (inputparams.py names in lower case) and allowed values of the method options
NUM   = (int, float)
TYPES = {'out_file': (str,), 'eigen_file': (str,), 'psi_file': (str,), 'static_file': (str,), 'psp_dir': (str, type(None)),
         'omega_min': NUM, 'omega_max': NUM, 'n_omega': (int,), 'eta': NUM, 'formula': (str,), 'broadening': (str,),
         'drude': (bool,), 'velocity': (str,), 'n_bands': (int, type(None)), 'out_dir': (str, type(None)),
         'plot': (bool,), 'ref_files': (list, tuple), 'n_threads': (int,)}
OPTIONS = {'formula': ('exact', 'approximated'), 'broadening': ('lorentzian', 'gaussian'),
           'velocity': ('gradient+nonlocal', 'gradient')}
BLAS_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')   # BLAS threads, read when NumPy loads


# *** Functions **************************************************************
def estimate_memory(S):
    """
    **********************************************************************
     Purpose:
       - estimated peak memory of the run's arrays, as SPARC's estimate_memory (initialization.c:3276):
         the large arrays alive at the same time, counted per stage, + 10% for all smaller ones.
         The .psi is memory-mapped (OS page cache) and not counted.
       - stages: setup (projectors); then per (k-point, spin) pair the orbital block with either the
         velocity-matrix workspace or the omega-sum workspace, once per thread (S.n_pool pairs at a time)
       - counts follow the code (checked with tracemalloc): velocity_matrix keeps 6 arrays of Nd x BAND_BLOCK
         per band block (+1 block copy when nb > BAND_BLOCK, +2 for the cross terms of non-orthogonal cells);
         the omega sums hold a few chunk x pairs kernel matrices (more for 'approximated'); projectors take
         ~120 bytes per nonzero

     Input Parameters:
       S : SimpleNamespace -- after reading the run, occupations(), S.nb, S.omega and S.n_pool.

     Returns:
       dict -- bytes per part ('orbitals', 'velocity', 'omega sums': all threads; 'projectors', 'other') and 'total'.
    **********************************************************************
    """
    isz, Nd, nb, nk, ns, T = (8 if S.isGamma else 16), S.Nd, S.nb, len(S.wk), S.psi.shape[2], S.n_pool
    Ui    = np.linalg.inv(S.U)
    g     = Ui.T @ Ui
    cross = np.abs(g - np.diag(np.diag(g))).max() > 1e-12                    # non-orthogonal cell
    full  = S.velocity == 'gradient+nonlocal'
    B     = min(BAND_BLOCK, nb)

    # projector nonzeros: grid points within rc of every atom (all images) times its projectors (and SO projectors)
    nlpm  = lambda p, l0: sum(p.ppl[l] * (2*l + 1) for l in range(l0, p.lmax + 1) if l != p.lloc)
    nnz   = sum(n * 4/3 * np.pi * max(p.rc)**3 / S.dV * (nlpm(p, 0) + (nlpm(p.soc, 1) if S.nspinor == 2 and p.pspsoc else 0))
                for p, n in zip(S.psds, S.n_atoms)) if full else 0

    # omega sums: pairs with |F| > F_TOL (largest over k-points and spins), omega chunks of _pair_sum
    npair = max(np.count_nonzero(np.abs(weights_F(S.eig[s, k, :nb], S.f[s, k, :nb], S.mdfde[s, k, :nb], S.drude,
                                                  S.formula)) > F_TOL) for k in range(nk) for s in range(ns))
    chunk = -(-S.n_omega // max(1, S.n_omega // 64))
    nker  = 5 if S.formula == 'approximated' else 3

    mem = {'orbitals'  : T * Nd * S.nspinor * nb * isz,                           # one (k, s) block per thread
           'velocity'  : T * ((6 + (nb > B) + 2 * cross) * Nd * B * isz + 6 * nb * nb * 16),
           'omega sums': T * (218 * nb * nb + 160 * npair + nker * chunk * npair * 8),  # v, Z, F, ... + pair kernels
           'projectors': 120 * nnz,
           'other'     : 4 * ns * nk * S.nstates * 8 + 3 * S.n_omega * 144}
    pair  = max(mem['orbitals'] + mem['velocity'], mem['omega sums'])             # orbitals freed before omega sums
    setup = 2.2 * mem['projectors']                                                # projector build
    keep  = mem['projectors'] + mem['other']                                       # persistent during the k-loop
    mem['total'] = 1.10 * max(keep + pair, setup + mem['other'])
    return mem


def kpoint_spin(S, k, s):
    """
    **********************************************************************
     Purpose:
       - contributions of one (k-point, spin) pair: velocity matrix, checks, the k-weighted sigma~1 and
         sigma~2, the DC split and the f-sum terms (Calderin Eqs. 10-11, 41-44)
       - runs on a worker thread of kubo_greenwood: reads only this pair's orbitals and leaves S unchanged

     Input Parameters:
       S    : SimpleNamespace -- run data after the setup of kubo_greenwood.
       k, s : int             -- k-point and spin index.

     Returns:
       (st1, st2, dc_parts, fsum, err_norm, err_herm, seconds)
    **********************************************************************
    """
    t1  = time.perf_counter()
    psi = np.ascontiguousarray(S.psi[k, :S.nb, s]['psi'].T)                  # (Nd, nb), C order for the gemms
    v   = velocity_matrix(S, psi, S.kred[k])
    e_n, e_h = checks(S, psi, v)
    del psi                                                                  # orbitals not needed beyond this point
    eps = S.eig[s, k, :S.nb]
    F   = weights_F(eps, S.f[s, k, :S.nb], S.mdfde[s, k, :S.nb], S.drude, S.formula)
    Fx  = weights_F(eps, S.f[s, k, :S.nb], S.mdfde[s, k, :S.nb], S.drude)              # Df/De, for the f-sum rule
    de  = eps[:, None] - eps[None, :]
    Z   = np.einsum('anm,bmn->nmab', v, v).real                               # non-magnetic: Re Z
    del v
    st1 = S.wk[k] * sigma1_tilde(S, F, Z, de, S.omega)
    st2 = S.wk[k] * sigma2_tilde(S, F, Z, de, S.omega)

    # DC split (intraband / degenerate / interband) and f-sum rule (Calderin Eqs. 41-44)
    same = np.eye(S.nb, dtype=bool)
    deg  = np.abs(de) <= DEGEN_TOL
    dc   = S.wk[k] * np.array([sigma1_tilde(S, F * m, Z, de, np.zeros(1))[0] for m in [same, deg & ~same, ~deg]])
    FtrZ = Fx * np.einsum('nmaa->nm', Z)
    fsum = S.wk[k] * S.g_s / (3 * S.N_e) * np.array([FtrZ.sum(), FtrZ[~deg].sum()])
    return st1, st2, dc, fsum, e_n, e_h, time.perf_counter() - t1


def kubo_greenwood(**cli):
    """
    **********************************************************************
     Purpose:
       - full calculation. Workflow (function -> file):
           1. read_out, read_eigen, read_psi, read_static, read_psp8  (readfiles.py)
           2. occupations                                             (dielectrics.py)
           3. nonlocal_projectors (+ soc_projectors for spinor runs)  (operators.py)
           4. per spin, k (kpoint_spin, on N_THREADS threads): velocity_matrix,
              checks, weights_F, sigma1_tilde, sigma2_tilde           (dielectrics.py)
           5. sigma1, sigma2, sigma, epsilon, write_output,
              plot_spectra                                            (dielectrics.py)
       - threads: N_THREADS (k-point, spin) pairs at a time (ThreadPoolExecutor), summed in (k, s) order;
         BLAS threads per pair from OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS (environment);
         no oversubscription: N_THREADS <= cores, and BLAS threads are set at run time (threadpoolctl) to 1 if
         not set with N_THREADS > 1, and to cores // N_THREADS if N_THREADS x BLAS threads > cores

     Input Parameters:
       cli : dict -- values overriding inputparams.py (keys = lower-case input names, e.g. out_file, formula,
                     drude); None = take the value from inputparams.py. Method options FORMULA, BROADENING,
                     DRUDE, VELOCITY default to the exact result; any other choice prints a warning.

     Returns:
       S : SimpleNamespace -- inputs, data and results.
    **********************************************************************
    """
    S = SimpleNamespace(**{k.lower(): v for k, v in vars(inputparams).items() if k.isupper()})
    S.__dict__.update({k: v for k, v in cli.items() if v is not None})

    ''' Input checks '''
    for key, typ in TYPES.items():
        val = getattr(S, key)
        if not isinstance(val, typ) or (isinstance(val, bool) and bool not in typ):
            raise TypeError(f'{key.upper()} must be {" or ".join("None" if t is type(None) else t.__name__ for t in typ)} '
                            f'(got {type(val).__name__}: {val!r}).')
    for key, opts in OPTIONS.items():
        if getattr(S, key) not in opts:
            raise ValueError(f'{key.upper()} must be {" or ".join(map(repr, opts))} (got {getattr(S, key)!r}).')
    for key in ['out_file', 'eigen_file', 'psi_file', 'static_file']:
        if not os.path.isfile(getattr(S, key)):
            raise FileNotFoundError(f'{key.upper()} not found: {getattr(S, key)}')
    if S.eta <= 0:
        raise ValueError('ETA must be > 0 (broadening half width, Ha).')
    if not (0 <= S.omega_min < S.omega_max and S.n_omega >= 2):
        raise ValueError('need 0 <= OMEGA_MIN < OMEGA_MAX and N_OMEGA >= 2.')
    if S.n_bands is not None and S.n_bands < 1:
        raise ValueError('N_BANDS must be >= 1 or None (which uses all the bands in the output).')
    if S.n_threads < 1:
        raise ValueError('N_THREADS must be >= 1.')
    if S.plot and importlib.util.find_spec('matplotlib') is None:
        raise ModuleNotFoundError('PLOT = True needs Matplotlib.')
    for f in (S.ref_files if S.plot else []):
        if not os.path.isfile(f):
            raise FileNotFoundError(f'REF_FILES entry not found: {f}')

    S.out_dir = S.out_dir or os.path.join(os.path.dirname(os.path.abspath(S.out_file)), 'KG_output')      # default as PDOS
    S.warnings = [msg for (key, val), msg in APPROXIMATIONS.items() if getattr(S, key) == val]
    step = (S.omega_max - S.omega_min) / (S.n_omega - 1)
    if step > S.eta / 4:
        S.warnings.append(f'omega step {step:.3g} Ha exceeds ETA/4 = {S.eta/4:.3g} Ha: peaks are under-resolved '
                          f'(increase N_OMEGA).')
    S.log = []                                                               # screen messages, also written to .kglog
    say   = lambda msg: (print(' ' + msg), S.log.append(msg))
    for msg in S.warnings:
        say('WARNING: ' + msg)
    S.t0      = time.time()
    S.psp_dir = S.psp_dir or os.path.dirname(os.path.abspath(S.out_file))
    S.prefix  = re.sub(r'\.out(_\d+)?$', '', os.path.basename(S.out_file))                         # SPARC run name

    ''' Read the SPARC run and set up the k-independent parts '''
    read_out(S); read_eigen(S); read_psi(S); read_static(S)
    if S.psi.shape[0] != len(S.wk) or S.psi.shape[2] != S.eig.shape[0]:
        raise ValueError(f'k-point or spin count differs between the .psi file (k-points: {S.psi.shape[0]}, '
                         f'spins: {S.psi.shape[2]}) and the .eigen file (k-points: {len(S.wk)}, spins: {S.eig.shape[0]}): '
                         f'the two files are inconsistent (e.g. from different SPARC runs).')
    full    = lambda f: f if os.path.isabs(f) else os.path.join(S.psp_dir, f)                   # path as SPARC opened it
    S.psp_files = [full(f) if os.path.exists(full(f)) else os.path.join(S.psp_dir, os.path.basename(f)) for f in S.psp]
    for f, p in zip(S.psp, S.psp_files):                                                      # else: same name in PSP_DIR
        if not os.path.isfile(p):
            raise FileNotFoundError(f'{os.path.basename(f)} (listed in the .out) not found in PSP_DIR = {S.psp_dir}.')
    S.psds  = [read_psp8(f) for f in S.psp_files]
    occupations(S)
    if S.n_bands is not None and S.n_bands > S.psi.shape[1]:
        raise ValueError(f'N_BANDS = {S.n_bands} exceeds the bands available in the .psi file (maximum {S.psi.shape[1]}).')
    S.nb    = S.psi.shape[1] if S.n_bands is None else S.n_bands
    S.omega = np.linspace(S.omega_min, S.omega_max, S.n_omega)
    ks      = [(k, s) for k in range(len(S.wk)) for s in range(S.psi.shape[2])]          # (k-point, spin) pairs
    cores   = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    S.n_pool = min(S.n_threads, len(ks), cores)
    env     = {v: os.environ[v] for v in BLAS_ENV if v in os.environ}
    nblas   = max((int(n) for n in env.values() if n.isdigit()), default=cores)   # not set: BLAS uses all cores
    S.blas_limit = (1 if S.n_pool > 1 and not env else                    # no oversubscription: not set -> 1,
                    max(1, cores // S.n_pool) if S.n_pool * nblas > cores else None)   # too many -> cores // N_THREADS
    S.blas  = (f'{S.blas_limit} (set for {S.n_pool} k-point-spin thread{"s" if S.n_pool > 1 else ""} on {cores} cores)'
               if S.blas_limit else
               ', '.join(f'{v}={n}' for v, n in env.items()) or 'not set (library default)')
    if S.blas_limit:                                                     # BLAS reads its thread count when NumPy loads
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            raise ImportError(f'Setting {S.blas_limit} BLAS thread(s) for {S.n_pool} k-point-spin thread(s) on {cores} '
                              f'cores needs threadpoolctl (pip install threadpoolctl); or start Python with '
                              f'OMP_NUM_THREADS={S.blas_limit}.') from None
    S.mem   = estimate_memory(S)                                             # before the large allocations
    say(f'Estimated peak memory usage: {format_bytes(S.mem["total"])} ('
        + ', '.join(f'{k} {format_bytes(v)}' for k, v in S.mem.items() if k != 'total' and v)
        + (f'; orbitals, velocity and omega sums for {S.n_pool} threads' if S.n_pool > 1 else '') + ')')
    if S.velocity == 'gradient+nonlocal':
        nonlocal_projectors(S)
        if S.nspinor == 2:                                                   # spin-orbit (spinor orbitals, SPIN_TYP 0)
            soc_projectors(S)
    S.t_setup = time.time() - S.t0
    say(f'{S.prefix}: SPIN_TYP {S.spin_typ}{", spin-orbit" if S.nspinor == 2 else ""}, {len(S.wk)} k-points, '
        f'{S.nb} bands, grid {S.N.tolist()}, {S.n_pool} k-point-spin thread{"s" if S.n_pool > 1 else ""} '
        f'(setup {S.t_setup:.2f} s)')

    ''' Sum over spin and k-points: N_THREADS pairs at a time, summed in (k, s) order (results independent of N_THREADS) '''
    t, c = time.perf_counter(), time.process_time()                          # wall and CPU time (all threads)
    st1 = st2 = 0.0
    S.dc_parts, S.fsum, S.err_norm, S.err_herm = np.zeros((3, 3, 3)), np.zeros(2), 0.0, 0.0
    with (threadpool_limits(S.blas_limit, user_api='blas') if S.blas_limit else contextlib.nullcontext()), \
            ThreadPoolExecutor(S.n_pool) as pool:
        for (k, s), (s1, s2, dc, fs, e_n, e_h, dt) in zip(ks, pool.map(lambda p: kpoint_spin(S, *p), ks)):
            st1, st2    = st1 + s1, st2 + s2
            S.dc_parts += dc
            S.fsum     += fs
            S.err_norm, S.err_herm = max(S.err_norm, e_n), max(S.err_herm, e_h)
            say(f'  k-point {k+1}/{len(S.wk)}, spin {s}: {dt:.2f} s')
    S.t_loop, S.c_loop = time.perf_counter() - t, time.process_time() - c
    say(f'k-point and spin sums: {S.t_loop:.2f} s, CPU time {S.c_loop:.2f} s (cores busy on average: '
        f'{S.c_loop / max(S.t_loop, 1e-9):.1f}; {S.n_pool} k-point-spin thread{"s" if S.n_pool > 1 else ""}, '
        f'BLAS threads {S.blas})')

    ''' Conductivity and dielectric tensors, output '''
    S.sigma1, S.sigma2 = sigma1(st1, st2), sigma2(st1, st2)
    S.sigma   = sigma(S.sigma1, S.sigma2)
    S.epsilon = epsilon(S.sigma, S.omega)
    write_output(S)
    if S.plot:
        plot_spectra(S)
    print(f' Done. Results in {os.path.join(S.out_dir, S.prefix)}.kgout{S.suffix}, messages and checks in '
          f'.kglog{S.suffix} ({time.time() - S.t0:.2f} s)')
    return S


# CLI: python kubo_greenwood.py [--out_file X.out ...]  -- anything not given is taken from inputparams.py
parser = argparse.ArgumentParser(prog='kubo_greenwood', description='Linear-response optical spectra (Kubo-Greenwood) from SPARC output.')
for key, typ in [('out_file', str), ('eigen_file', str), ('psi_file', str), ('static_file', str), ('psp_dir', str),
                 ('omega_min', float), ('omega_max', float), ('n_omega', int), ('eta', float), ('n_bands', int),
                 ('out_dir', str), ('n_threads', int)]:
    parser.add_argument('--' + key, type=typ)
parser.add_argument('--formula', choices=['exact', 'approximated'])
parser.add_argument('--broadening', choices=['lorentzian', 'gaussian'])
parser.add_argument('--no_drude', dest='drude', action='store_false', default=None, help='interband only')
parser.add_argument('--velocity', choices=['gradient+nonlocal', 'gradient'])
parser.add_argument('--ref_files', nargs='*')
parser.add_argument('--plot', action='store_true', default=None, help='also save the sigma and epsilon plots')

if __name__ == '__main__':
    kubo_greenwood(**vars(parser.parse_args()))
