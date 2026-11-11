"""
Kubo-Greenwood Optical Spectra Calculator for SPARC Output Files

    Reference: SPARC (https://github.com/SPARC-X/SPARC), the routines that write or read each file:
               - .out    : write_output_init (initialization.c); SCF results incl. Fermi level
                           (electronicGroundState.c)
               - .eigen  : printEigen (printing.c)
               - .psi    : print_orbitals (printing.c)
               - .static : write_output_init (initialization.c)
               - .psp8   : read_pseudopotential_PSP (readfiles.c)

    Parsers for the output of a single SPARC run and its psp8 pseudopotentials, consistent with
    the SPARC routines listed above:
    - .out    : lattice, FD order, boundary conditions, spin type, exchange-correlation
                functional, smearing, electron count, Fermi level and pseudopotential files
    - .eigen  : eigenvalues, occupations, k-points and k-point weights
    - .psi    : real-space grid and Kohn-Sham orbitals, memory-mapped and accessed one k-point
                at a time
    - .static : atomic positions
    - .psp8   : nonlocal and spin-orbit projectors, projector energies and cutoff radii
    Runs outside the supported scope (cyclix cells, SPIN_TYP 2, hybrid functionals, DFT+U) are
    rejected.

    @file    readfiles.py
    @brief   Readers for the SPARC output files and psp8 pseudopotentials
    @authors Shubhang K Trivedi <strivedi44@gatech.edu>
             Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

    Copyright (c) 2026 Material Physics & Mechanics Group, Georgia Tech.
"""

# *** Imports ****************************************************************
import os
import re
from types import SimpleNamespace

import numpy as np

KB_HA = 3.1668115634556e-6        # Boltzmann constant (Ha/K), CONST_KB in SPARC isddft.h


# *** Functions **************************************************************
def read_out(S):
    """
    **********************************************************************
     Purpose:
       - parse the SPARC .out file (last occurrence of every key wins)
       - stop for runs outside the supported scope (cyclix cells, SPIN_TYP 2, hybrid, DFT+U)

     Reference:
       - SPARC: write_output_init (initialization.c:3983), Fermi level
         (electronicGroundState.c:160)

     Input Parameters:
       S : SimpleNamespace -- needs S.out_file.

     Returns:
       S : SimpleNamespace -- with cell, grid, spin, smearing and atom data.
    **********************************************************************
    """
    txt  = open(S.out_file).read()

    def last(key, pattern=None):                                  # tokens after the last 'key' (or the pattern's group)
        hits = re.findall(pattern or rf'^{re.escape(key)}\s*(.*)$', txt, re.M)
        if not hits:
            raise ValueError(f"'{key.rstrip(':')}' not found in {S.out_file}; unusable SPARC .out file.")
        return hits[-1].split()
    nums = lambda key: [float(x) for x in last(key)]

    # unsupported runs
    if re.search(r'^TWIST_ANGLE:', txt, re.M) or set(last('BC:')) - {'P', 'D'}:
        raise ValueError('Cyclix/helical cells are not supported.')
    S.spin_typ = int(nums('SPIN_TYP:')[0])
    if S.spin_typ == 2:
        raise ValueError('SPIN_TYP 2 (noncollinear spin) runs are not supported.')
    S.xc = last('EXCHANGE_CORRELATION:')[0]
    if S.xc.upper() in ('HF', 'PBE0', 'HSE'):                         # usefock (initialization.c:5160-5200)
        raise ValueError(f'{S.xc}: hybrid runs are not supported.')
    if re.search(r'^HUBBARD_FLAG:', txt, re.M) and int(nums('HUBBARD_FLAG:')[0]):
        raise ValueError('DFT+U runs are not supported.')

    # cell: full Cartesian lattice vectors (rows) as printed in the 'Cell' section
    last('Lattice vectors (Bohr):')
    lines = txt.splitlines()
    i = [j for j, l in enumerate(lines) if l.startswith('Lattice vectors (Bohr):')][-1]
    S.latvec = np.array([[float(x) for x in l.split()] for l in lines[i+1:i+4]])
    S.L      = np.linalg.norm(S.latvec, axis=1)                  # range_x, range_y, range_z
    S.U      = S.latvec / S.L[:, None]                           # LatUVec (rows = unit lattice vectors)
    S.Omega  = abs(np.linalg.det(S.latvec))

    S.fd_order = int(nums('FD_ORDER:')[0])
    S.BC       = last('BC:')                                     # ['P'|'D'] x 3
    S.kpt_grid = [int(x) for x in nums('KPOINT_GRID:')] if re.search(r'^KPOINT_GRID:', txt, re.M) else [1, 1, 1]
    S.nstates  = int(nums('NSTATES:')[0])

    # smearing: beta = 1/SMEARING (Ha) or 1/(kB*ELEC_TEMP) (readfiles.c:344, initialization.c:4049)
    S.smearing_type = 'gaussian' if last('ELEC_TEMP_TYPE:')[0].lower().startswith('gauss') else 'fermi-dirac'
    S.beta = 1.0 / nums('SMEARING:')[0] if re.search(r'^SMEARING:', txt, re.M) else 1.0 / (KB_HA * nums('ELEC_TEMP:')[0])

    # electrons, Fermi level, atom types and pseudopotentials
    S.N_e     = float(last('Total number of electrons', r'Total number of electrons\s*:\s*(\S+)')[0])
    S.E_F     = float(last('Fermi level', r'Fermi level\s*:\s*(\S+)')[0])
    S.types   = re.findall(r'Atom type \d+\s+\(valence electrons\)\s*:\s*(\S+)', txt)
    S.psp     = re.findall(r'^Pseudopotential\s*:\s*(\S+)', txt, re.M)
    S.n_atoms = [int(n) for n in re.findall(r'Number of atoms of type \d+\s*:\s*(\d+)', txt)]
    return S


def read_eigen(S):
    """
    **********************************************************************
     Purpose:
       - parse the SPARC .eigen file: per k-point the reduced k-vector,
         the weight and (eigenvalue, printed occupation) per band and spin
       - weights are printed with 6 decimals; they are restored exactly as
         (integer TR weight)/(Kx*Ky*Kz)

     Reference:
       - SPARC: printEigen (printing.c:219-490)

     Input Parameters:
       S : SimpleNamespace -- needs S.eigen_file, S.kpt_grid, S.nstates.

     Returns:
       S : SimpleNamespace -- S.eig, S.occ_printed (nspin, nkpt, nstates), S.wk (nkpt).
    **********************************************************************
    """
    blocks = open(S.eigen_file).read().split('kred #')[1:]
    nk_full = np.prod(S.kpt_grid)
    S.wk, eig = [], []
    for b in blocks:
        w = re.search(r'weight\s*=\s*(\S+)', b)
        S.wk.append(round(float(w.group(1)) * nk_full) / nk_full if w else 1.0 / len(blocks))
        rows = [l.split() for l in b.splitlines()[1:] if re.match(r'^\d+\s', l)][:S.nstates]   # line 0 is '<k> = (k1,k2,k3)'
        eig.append(np.array(rows, dtype=float)[:, 1:])           # (nstates, 2) or (nstates, 4)
    eig = np.array(eig)                                           # (nkpt, nstates, 2*nspin)
    S.wk = np.array(S.wk)
    S.eig, S.occ_printed = eig[:, :, 0::2].transpose(2, 0, 1), eig[:, :, 1::2].transpose(2, 0, 1)
    return S


def read_psi(S):
    """
    **********************************************************************
     Purpose:
       - memory-map the binary SPARC .psi file. Records are written in the
         order kpt -> band -> spin; each holds (spin, kpt, kvec[3], band, psi).
         psi is already scaled by 1/sqrt(dV), i.e. sum |psi|^2 dV = 1.
       - spin-orbit runs (Nspinor_eig = 2, SPIN_TYP 0): psi = [psi_up (Nd), psi_dn (Nd)],
         always complex (isGammaPoint = 0 with SOC, initialization.c:2843)
       - nothing is loaded here; S.psi[k, n, s]['psi'] reads one orbital
       - checks the file size and the (kpt, band, spin) index of every record: with
         NP_SPIN_PARAL > 1, SPARC's writer can reorder records or overwrite the first one

     Reference:
       - SPARC: print_orbitals / print_orbital_real / print_orbital_complex
         (printing.c:516-794)

     Input Parameters:
       S : SimpleNamespace -- needs S.psi_file.

     Returns:
       S : SimpleNamespace -- S.psi (memmap, shape (nkpt, nband, nspin)), grid data.
    **********************************************************************
    """
    hdr = np.fromfile(S.psi_file, dtype=[('N', '<i4', 4), ('d', '<f8', 4), ('i', '<i4', 5)], count=1)[0]
    S.N, S.Nd = hdr['N'][:3], hdr['N'][3]                        # Nx, Ny, Nz (x fastest), Nd
    S.delta, S.dV = hdr['d'][:3], hdr['d'][3]                    # dx, dy, dz, dV (includes Jacobian)
    S.nspinor, S.isGamma, nspin, nkpt, nband = hdr['i']               # Nspinor_eig = 2: spin-orbit (SPIN_TYP 2 stopped in read_out)
    rec   = np.dtype([('spin', '<i4'), ('kpt', '<i4'), ('kvec', '<f8', 3), ('band', '<i4'),
                      ('psi', '<f8' if S.isGamma else '<c16', S.Nd * S.nspinor)])
    shape = (nkpt, nband, nspin)
    bad   = (f'{S.psi_file}: records missing, out of order or overwritten. Print all orbitals (PRINT_ORBITAL: 1) '
             'and, for SPIN_TYP 1, run SPARC with NP_SPIN_PARAL: 1.')
    if os.path.getsize(S.psi_file) != hdr.nbytes + np.prod(shape) * rec.itemsize:
        raise ValueError(bad)
    S.psi  = np.memmap(S.psi_file, dtype=rec, mode='r', offset=hdr.nbytes, shape=shape)
    S.kred = np.array(S.psi[:, 0, 0]['kvec'])                    # full-precision reduced k-points
    if not all(np.array_equal(S.psi[f], i) for f, i in zip(('kpt', 'band', 'spin'), np.indices(shape))):
        raise ValueError(bad)                                     # SPARC's spin-parallel writer can reorder records
    return S


def read_static(S):
    """
    **********************************************************************
     Purpose:
       - read atom positions ('Fractional coordinates of X:' blocks, one per atom
         type in type order, printed with PRINT_ATOMS: 1) and convert them to SPARC's
         non-Cartesian coordinates x_i = frac_i * L_i (atom_pos in SPARC)

     Reference:
       - SPARC: write_output_init, .static block (initialization.c:4628-4643)

     Input Parameters:
       S : SimpleNamespace -- needs S.static_file, S.types, S.n_atoms, S.L.

     Returns:
       S : SimpleNamespace -- S.atom_pos (n_atom, 3), S.atom_typ (n_atom).
    **********************************************************************
    """
    lines = open(S.static_file).read().splitlines()
    heads = [i for i, l in enumerate(lines) if l.startswith('Fractional coordinates of ')][:len(S.types)]
    if len(heads) < len(S.types):
        raise ValueError(f'{S.static_file}: atom positions missing. Run SPARC with PRINT_ATOMS: 1.')
    pos   = [[float(x) for x in l.split()] for i, n in zip(heads, S.n_atoms) for l in lines[i+1:i+1+n]]
    S.atom_pos = np.array(pos) * S.L
    S.atom_typ = np.repeat(np.arange(len(S.types)), S.n_atoms)
    return S


def read_psp8(fname):
    """
    **********************************************************************
     Purpose:
       - read the nonlocal part of an ONCV .psp8 file: lmax, lloc,
         projectors per l, projector energies Gamma (ekb), UdV = proj/r on
         the radial grid (UdV[0] = UdV[1]) and the cutoff radius per l,
         extended to the last r with |UdV| > 1e-8 (as SPARC does)
       - fully relativistic files (extension_switch 2 or 3): also the
         spin-orbit projectors, l = 1..lmax, which follow the local potential
         block. They are returned as psd.soc, a psd of their own with the
         same radial grid and cutoff radii (SPARC uses the scalar projectors'
         grid points for them). As in SPARC, UdV_soc at r = 0 is copied from
         r[1] of the projector at index lpos (not lpos_soc); the two agree
         when every l has as many SO as scalar projectors before it.

     Reference:
       - SPARC: read_pseudopotential_PSP (readfiles.c:2182-2549; SO part 2355-2447)

     Input Parameters:
       fname : str -- path to the .psp8 file.

     Returns:
       psd : SimpleNamespace -- lmax, lloc, ppl, gamma[l], udv[l] (mmax, ppl[l]), r, rc[l], pspsoc,
                                and soc (lmax, lloc, ppl, gamma[l], udv[l], r, rc) if pspsoc.
    **********************************************************************
    """
    lines = open(fname).read().splitlines()
    tok   = open(fname).read().split()
    _, _, lmax, lloc, mmax = [int(x) for x in lines[2].split()[:5]]
    ppl    = [int(x) for x in lines[4].split()[:lmax+1]]
    pspsoc = int(lines[5].split()[0]) in (2, 3)
    i      = tok.index('r_core=')
    rc     = [float(x) for x in tok[i+1:i+2+lmax]]

    psd, pos = SimpleNamespace(lmax=lmax, lloc=lloc, ppl=ppl, pspsoc=pspsoc, gamma={}, udv={}), 6 + pspsoc
    for l in range(lmax + 1):
        if l != lloc:
            psd.gamma[l] = np.array(lines[pos].split()[1:1+ppl[l]], dtype=float)
            data = np.array([x.split()[:2+ppl[l]] for x in lines[pos+1:pos+1+mmax]], dtype=float)
            psd.r, psd.udv[l] = data[:, 1], data[:, 2:] / np.where(data[:, 1:2] > 0, data[:, 1:2], 1.0)
            psd.udv[l][0] = psd.udv[l][1]
            far = psd.r[(psd.r > rc[l]) & np.any(np.abs(psd.udv[l]) > 1e-8, axis=1)]
            rc[l] = max(rc[l], far.max()) if far.size else rc[l]
        pos += 1 + mmax
    psd.rc = rc

    # spin-orbit projectors l = 1..lmax ('nprojso' line), after the local potential block ('4') if lloc > lmax
    if pspsoc:
        ppl_so    = [0] + [int(x) for x in lines[6].split()[:lmax]]
        lpos, lps = np.cumsum([0] + ppl), np.cumsum(ppl_so)              # SPARC's lpos and lpos_soc (lps[l-1] = lpos_soc[l-1])
        flat      = np.zeros((lps[-1], mmax))                            # UdV_soc of all l, in SPARC's order
        psd.soc   = SimpleNamespace(lmax=lmax, lloc=lloc, ppl=ppl_so, gamma={}, r=psd.r, rc=rc)
        pos      += (1 + mmax) * (lloc > lmax)
        for l in range(1, lmax + 1):
            if l != lloc:
                psd.soc.gamma[l] = np.array(lines[pos].split()[1:1+ppl_so[l]], dtype=float)
                data = np.array([x.split()[:2+ppl_so[l]] for x in lines[pos+1:pos+1+mmax]], dtype=float)
                flat[lps[l-1]:lps[l]]    = (data[:, 2:] / np.where(data[:, 1:2] > 0, data[:, 1:2], 1.0)).T
                flat[lps[l-1]:lps[l], 0] = flat[lpos[l-1]:lpos[l-1]+ppl_so[l], 1]     # readfiles.c:2444 (index lpos, as SPARC)
            pos += 1 + mmax
        psd.soc.udv = {l: flat[lps[l-1]:lps[l]].T for l in range(1, lmax + 1) if l != lloc}
    return psd
