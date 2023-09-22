import sys
sys.path.append('..')
from band_structure_plot import *

nkpt_per_line = 10
fname = 'CdS.eigen_01'
highsym_points = 'G-X-W-K-G-L-U-W-L-K'
Efermi = 0.42712 # in Ha
outfname = 'CdS_bandstruct.png'
plot_band_structure(fname, nkpt_per_line, highsym_points,
                    unit='eV', Efermi_Ha=Efermi, ylim=(-10,30),
                    outfname=outfname, dpi=200)

