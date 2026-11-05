"""
*************************************************************************************
*     Program  :  KGconductivity  (Kubo-Greenwood conductivity from SPARC output)
*     Module   :  inputparams  --  user inputs. Edit, then run:  python KG_driver.py
*                 (any value can also be given on the command line: python KG_driver.py -h)
*************************************************************************************
"""

# --- SPARC output files of ONE run (SPARC flags: PRINT_ORBITAL: 1, PRINT_EIGEN: 1, PRINT_ATOMS: 1) ---
OUT_FILE    = '../pdos/examples/Si2_nmag/Si2_nmag.out'
EIGEN_FILE  = '../pdos/examples/Si2_nmag/Si2_nmag.eigen'
PSI_FILE    = '../pdos/examples/Si2_nmag/Si2_nmag.psi'
STATIC_FILE = '../pdos/examples/Si2_nmag/Si2_nmag.static'
PSP_DIR     = None              # folder of the .psp8 files named in the .out file (None: folder of OUT_FILE)

# --- frequency grid and broadening (Hartree) ---
OMEGA_MIN   = 0.0
OMEGA_MAX   = 1.0
N_OMEGA     = 1001
ETA         = 0.005             # half width (Ha); full width at half maximum = 2*ETA for both broadenings

# --- options ---
N_BANDS     = None              # use only the lowest N_BANDS bands (None: all bands in the .psi file)
OUT_DIR     = 'KG_output'
PLOT        = True
REF_FILES   = []                # spectra overlaid on the sigma1 plot: text files with columns (eV, S/m)
