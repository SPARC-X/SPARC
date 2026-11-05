"""
*************************************************************************************
*     Program  :  KGconductivity  (Kubo-Greenwood conductivity from SPARC output)
*     Module   :  readfiles
*     Authors  :  <placeholder>
*
*
*     Created  :  2026-11-04
*     Modified :  2026-11-04
*
*     Purpose  :  Read the output files of one SPARC run into the state S:
*                   - .out    : cell, FD order, BC, spin, smearing, Fermi level, atoms
*                   - .eigen  : eigenvalues, printed occupations, k-points, weights
*                   - .psi    : Kohn-Sham orbitals (memory-mapped, one k-point at a time)
*                   - .static : atom positions
*                   - .psp8   : nonlocal projectors, their energies and cutoff radii
*                 File formats follow the SPARC routines that write/read them.
*
*     Routines :  read_out()      (public)
*                 read_eigen()    (public)
*                 read_psi()      (public)
*                 read_static()   (public)
*                 read_psp8()     (public)
*
*     Reference:  SPARC  --  https://github.com/SPARC-X/SPARC
*************************************************************************************
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
       - stop for runs outside the v1 scope (cyclix cells, SPIN_TYP 2, hybrid, meta-GGA, DFT+U)

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
    last = lambda key: re.findall(rf'^{key}\s*(.*)$', txt, re.M)[-1].split()      # tokens after the last 'key'
    nums = lambda key: [float(x) for x in last(key)]

    # stop for cases not supported in v1
    if re.search(r'^TWIST_ANGLE:', txt, re.M) or set(last('BC:')) - {'P', 'D'}:
        raise ValueError('Cyclix/helical cells are not supported.')
    S.spin_typ = int(nums('SPIN_TYP:')[0])
    if S.spin_typ == 2:
        raise ValueError('SPIN_TYP 2 (non-collinear/magnetization) runs are not supported in v1.')
    S.xc = last('EXCHANGE_CORRELATION:')[0]
    if S.xc.upper() in ('HF', 'PBE0', 'HSE', 'SCAN', 'RSCAN', 'R2SCAN'):   # usefock / meta-GGA (initialization.c:5160-5200)
        raise ValueError(f'{S.xc}: hybrid and meta-GGA runs are not supported (velocity lacks the Fock / tau terms).')
    if re.search(r'^HUBBARD_FLAG:', txt, re.M) and int(nums('HUBBARD_FLAG:')[0]):
        raise ValueError('DFT+U runs are not supported (velocity lacks the Hubbard term).')

    # cell: full Cartesian lattice vectors (rows) as printed in the 'Cell' section
    lines = txt.splitlines()
    i = [j for j, l in enumerate(lines) if l.startswith('Lattice vectors (Bohr):')][-1]
    S.latvec = np.array([[float(x) for x in l.split()] for l in lines[i+1:i+4]])
    S.L      = np.linalg.norm(S.latvec, axis=1)                  # range_x, range_y, range_z
    S.U      = S.latvec / S.L[:, None]                           # LatUVec (rows = unit lattice vectors)
    S.Omega  = abs(np.linalg.det(S.latvec))
    assert abs(S.Omega - float(last('Volume:')[0])) < 1e-6 * S.Omega, 'Volume mismatch in .out'

    S.fd_order = int(nums('FD_ORDER:')[0])
    S.BC       = last('BC:')                                     # ['P'|'D'] x 3
    S.kpt_grid = [int(x) for x in nums('KPOINT_GRID:')] if re.search(r'^KPOINT_GRID:', txt, re.M) else [1, 1, 1]
    S.nstates  = int(nums('NSTATES:')[0])

    # smearing: beta = 1/SMEARING (Ha) or 1/(kB*ELEC_TEMP) (readfiles.c:344, initialization.c:4049)
    S.smearing_type = 'gaussian' if last('ELEC_TEMP_TYPE:')[0].lower().startswith('gauss') else 'fermi-dirac'
    S.beta = 1.0 / nums('SMEARING:')[0] if re.search(r'^SMEARING:', txt, re.M) else 1.0 / (KB_HA * nums('ELEC_TEMP:')[0])

    # electrons, Fermi level, atom types and pseudopotentials
    S.N_e     = float(re.findall(r'Total number of electrons\s*:\s*(\S+)', txt)[-1])
    S.E_F     = float(re.findall(r'Fermi level\s*:\s*(\S+)', txt)[-1])
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
       - nothing is loaded here; S.psi[k, n, s]['psi'] reads one orbital

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
    S.nspinor, S.isGamma, nspin, nkpt, nband = hdr['i']
    if S.nspinor != 1:
        raise ValueError('Spin-orbit (spinor) orbitals are not supported in v1.')
    rec = np.dtype([('spin', '<i4'), ('kpt', '<i4'), ('kvec', '<f8', 3), ('band', '<i4'),
                    ('psi', '<f8' if S.isGamma else '<c16', S.Nd * S.nspinor)])
    S.psi = np.memmap(S.psi_file, dtype=rec, mode='r', offset=hdr.nbytes, shape=(nkpt, nband, nspin))
    S.kred = np.array(S.psi[:, 0, 0]['kvec'])                    # full-precision reduced k-points
    assert S.psi[0, 0, 0]['band'] == 0 and S.psi[0, 0, 0]['kpt'] == 0, '.psi must start at band 0 and k-point 0'
    return S


def read_static(S):
    """
    **********************************************************************
     Purpose:
       - read atom positions ('Fractional coordinates of X:' blocks, printed
         with PRINT_ATOMS: 1) and convert them to SPARC's non-Cartesian
         coordinates x_i = frac_i * L_i (atom_pos in SPARC)

     Reference:
       - SPARC: Write_output_init .static block (initialization.c:4628-4643)

     Input Parameters:
       S : SimpleNamespace -- needs S.static_file, S.types, S.n_atoms, S.L.

     Returns:
       S : SimpleNamespace -- S.atom_pos (n_atom, 3), S.atom_typ (n_atom).
    **********************************************************************
    """
    lines = open(S.static_file).read().splitlines()
    pos = []
    for t, (el, n) in enumerate(zip(S.types, S.n_atoms)):
        i = lines.index(f'Fractional coordinates of {el}:')
        pos += [[float(x) for x in l.split()] for l in lines[i+1:i+1+n]]
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

     Reference:
       - SPARC: read_pseudopotential_PSP (readfiles.c:2182-2549)

     Input Parameters:
       fname : str -- path to the .psp8 file.

     Returns:
       psd : SimpleNamespace -- lmax, lloc, ppl, gamma[l], udv[l] (mmax, ppl[l]), r, rc[l], pspsoc.
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
    return psd
