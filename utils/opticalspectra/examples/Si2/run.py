"""
Kubo-Greenwood optical spectra of the Si2 example (outputs in KG_output/).
    python run.py                       serial
    python run.py --n_threads 4         4 k-point-spin threads
    OMP_NUM_THREADS=4 python run.py     4 BLAS threads
Any other input of kubo_greenwood.py can be added the same way (python run.py -h).
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))           # paths below are relative to this folder
sys.path.insert(0, '../..')                                     # utils/opticalspectra
from kubo_greenwood import kubo_greenwood, parser

inputs = dict(out_file='Si2.out', eigen_file='Si2.eigen', psi_file='Si2.psi', static_file='Si2.static',
              psp_dir='../../../../psps',                       # SPARC pseudopotential folder
              out_dir='KG_output')
inputs.update({k: v for k, v in vars(parser.parse_args()).items() if v is not None})   # command-line inputs
S = kubo_greenwood(**inputs)
