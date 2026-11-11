"""
Kubo-Greenwood Optical Spectra Calculator for SPARC Output Files

    User input file of the Kubo-Greenwood calculator: SPARC output files, frequency grid and
    broadening, method options, number of bands and output settings. Run with
        python kubo_greenwood.py
    Each entry can be overridden on the command line (python kubo_greenwood.py -h) or as a
    keyword argument of kubo_greenwood().

    @file    inputparams.py
    @brief   User inputs of the Kubo-Greenwood optical spectra calculator
    @authors Shubhang K Trivedi <strivedi44@gatech.edu>
             Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

    Copyright (c) 2026 Material Physics & Mechanics Group, Georgia Tech.
"""

# --- SPARC output files of ONE run (SPARC flags: PRINT_ORBITAL: 1, PRINT_EIGEN: 1, PRINT_ATOMS: 1) ---
OUT_FILE    = '../pdos/examples/Si2_nmag/Si2_nmag.out'
EIGEN_FILE  = '../pdos/examples/Si2_nmag/Si2_nmag.eigen'
PSI_FILE    = '../pdos/examples/Si2_nmag/Si2_nmag.psi'
STATIC_FILE = '../pdos/examples/Si2_nmag/Si2_nmag.static'
PSP_DIR     = None              # .psp8 path in the .out: absolute as is, relative from PSP_DIR (SPARC's run folder);
                                # if no file there, the file of that name in PSP_DIR (None: folder of OUT_FILE)

# --- frequency grid and broadening (Hartree) ---
OMEGA_MIN   = 0.0
OMEGA_MAX   = 1.0
N_OMEGA     = 1001              # uniform grid; keep the step (OMEGA_MAX - OMEGA_MIN)/(N_OMEGA - 1) below ETA/4
ETA         = 0.005             # half width (Ha); full width at half maximum = 2*ETA for both broadenings

# --- method (defaults: exact Kubo-Greenwood; any other choice is an approximation and prints a warning) ---
# ABINIT optic   : FORMULA 'exact', BROADENING 'lorentzian', DRUDE False   (optic broadening = ETA)
# ABINIT conducti: FORMULA 'approximated', BROADENING 'gaussian', DRUDE False   (conducti dom = ETA/sqrt(ln 2))
FORMULA     = 'exact'           # 'exact' (Df/De) | 'approximated' (Df/omega, as ABINIT conducti)
BROADENING  = 'lorentzian'      # 'lorentzian' | 'gaussian' (same FWHM 2*ETA)
DRUDE       = True              # intraband and degenerate-pair term; False: interband only
VELOCITY    = 'gradient+nonlocal'   # full i[H, r] (kinetic, nonlocal, spin-orbit, meta-GGA) | 'gradient' (-i grad only)

# --- options ---
N_BANDS     = None              # use only the lowest N_BANDS bands (None: all bands in the .psi file = SPARC NSTATES)
OUT_DIR     = None              # None: <folder of OUT_FILE>/KG_output
PLOT        = False
REF_FILES   = []                # reference spectra overlaid on the Tr sigma1/3 plot (PLOT = True), at most 3:
                                # two-column text files, hbar*omega (eV) and sigma1 (S/m)
