
################### Modules declaration and constant variables ################################################
from __future__ import print_function
import os
import subprocess
import re
import sys
from datetime import datetime
import time
import glob
from shutil import copyfile
import math

# Other parameters to run the test (can be changed by the user)
nprocs_tests = 24  # In default tests are run with 24 processors per node
nnodes_tests = 2  # In default tests are run with 1 node
npbs = 6  # By default (number of script files the tests are distributed to)
launch_cluster_extension = ".sbatch"   # extension of the file used to launch the jobs on the cluster by default it is .sbatch
command_launch_extension = "sbatch"   # Command to launch the script to ask for resources on the cluster (example: qsub launch.pbs)
MPI_command = "srun"  # MPI command to run the executable on the given cluster



# Default tolerance
tols = {"F_tol": 1e-5, # Ha/Bohr
	"E_tol": 1e-6,   # Ha/atom
	"stress_tol": 0.1, # in percent
	"KEN_tol": 1e-6, # Ha/atom
	"wall_tol": 10, # in percent
	"CELL_tol": 0.01, # Bohr
	"scfpos_tol": 0.01, # Bohr
	"scfno_tol": 3,
	"spin_tol": 0.001, # a.u.
	"memused_tol": 10}# percent}



# -----------------   SYSTEMS INFO   ------------------------#
################################################################################################################
SYSTEMS = { "systemname": ['BaTiO3_valgrind'],
		"directory": ["./"],
	    "Tags": [['bulk', 'gga', 'denmix', 'kerker', 'gamma', 'memcheck', 'gamma', 'orth', 'smear_gauss']],
	    "Tols": [[5e-5, 1e-4, 1e-1]], # E_tol(Ha/atom), F_tol, stress_tol(%)
	    }

################################################################################################################
SYSTEMS["systemname"].append('CuSi7')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'denmix', 'kerker', 'gamma', 'orth', 'smear_gauss','ECUT'])
SYSTEMS["Tols"].append([tols["E_tol"], 3e-5, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('BaTiO3')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'denmix', 'kerker',  'gamma', 'orth', 'smear_gauss'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Fe_spin')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'denmix', 'kerker', 'kpt', 'spin','orth','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
################################################################################################################
SYSTEMS["systemname"].append('H2O_sheet')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['surface', 'gga', 'potmix','orth','smear_fd','orient'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('H2O_wire')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['wire', 'gga', 'denmix', 'kerker', 'orth','smear_fd','orient'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('O2_spin')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'spin', 'gga', 'denmix', 'kerker', 'orth','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si8_atom_geopt')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'denmix', 'kerker', 'relax_atom_lbfgs','gamma','smear_gauss'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si8_cell_geopt')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'potmix', 'relax_cell','gamma','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si8_full_geopt')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'potmix', 'relax_total_lbfgs','gamma','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si8_kpt_valgrind')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'kpt', 'lda', 'potmix', 'memcheck','nonorth','smear_fd'])
SYSTEMS["Tols"].append([5e-5, 1e-4, 5.0]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si8_kpt')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'kpt', 'gga', 'potmix','nonorth','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si8')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'potmix', 'nonorth','gamma','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('SiH4')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['molecule', 'gga', 'denmix', 'kerker', 'orth','smear_gauss','bandgap'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Au_fcc211')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'denmix', 'kerker', 'nonorth','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Cu_FCC')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'denmix', 'kerker', 'orth','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], 1e-4, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Mg_hcp')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'denmix', 'kerker', 'nonorth','smear_fd', 'kpt'])
SYSTEMS["Tols"].append([tols["E_tol"], 1e-4, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('MnAlCu2')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'denmix', 'kerker', 'orth','smear_fd', 'gamma'])
SYSTEMS["Tols"].append([tols["E_tol"], 1e-4, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('MgO')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk','gga','potmix','nonorth','smear_gauss','nlcc','orient'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('MoS2')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['surface','gga','potmix','nonorth','smear_fd','orient'])
SYSTEMS["Tols"].append([tols["E_tol"], 5e-6, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('He16_NVKG')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk','gga','potmix','orth','smear_fd','md_nvkg','gamma'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('He16_NVTNH')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk','lda','potmix','orth','smear_fd','md_nvtnh','gamma'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('LiF_NVKG')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk','gga','potmix','orth','smear_fd','md_nvkg','gamma'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('O2_spin_spinparal_NVKG')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'spin', 'gga', 'denmix', 'kerker', 'orth','smear_fd','paral','md_nvkg'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si2_kpt_paral')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'potmix', 'nonorth','kpt','smear_fd', 'paral'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si2_domain_paral')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'potmix', 'nonorth','kpt','smear_fd', 'paral'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('TiNi_monoclinic')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'potmix', 'nonorth','gamma','smear_fd','nlcc'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('P_triclinic')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'potmix', 'nonorth','gamma','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('BaTiO3_quick')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'lda', 'denmix', 'orth','gamma','smear_gauss'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 1]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('H2O_sheet_quick')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['surface', 'gga', 'potmix', 'orth','gamma','smear_fd'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 1]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################						
SYSTEMS["systemname"].append('H2O_wire_quick')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['wire', 'gga', 'denmix', 'orth','gamma','smear_fd'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 1]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################### 
SYSTEMS["systemname"].append('SiH4_quick')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['molecule', 'gga', 'denmix', 'orth','gamma','smear_gauss'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 1]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Al18Si18_NPTNH')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'nonorth', 'md_npt'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Al16Si16_NPTNH_restart')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'md_npt'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Al18Si18_NPTNH_lat23')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'nonorth', 'md_npt'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Al18Si18_NPTNP')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'nonorth', 'md_npt'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Al16Si16_NPTNP_restart')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'md_npt'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Al18C2_NPTNP_aeqb_c')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'md_npt'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Al18C2_NPTNP_onlyc')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'md_npt'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Au_wire_d3')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['wire', 'gga','d3'])
SYSTEMS["Tols"].append([tols["E_tol"], 3e-4, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('BaTiO3_vdWDF1')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'gamma','vdWDF'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('BaTiO3_vdWDF2')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'gamma','vdWDF'])
SYSTEMS["Tols"].append([tols["E_tol"], 1e-4, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('C_HSE_aux')
SYSTEMS["directory"].append("./xc_tests/exx_tests/")
SYSTEMS["Tags"].append(['bulk', 'HSE','gamma' 'nonorth','smear_fd','potmix'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Fe2_spin_gamma_ortho_vdWDF1')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['bulk', 'spin', 'gga', 'orth', 'gamma','vdWDF'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Fe2_spin_kpt_nonOrtho_vdWDF2')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['bulk', 'spin', 'gga', 'nonorth', 'kpt','vdWDF'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('MoS2_surface_d3')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['surface', 'gga','d3','nonorth'])
SYSTEMS["Tols"].append([tols["E_tol"], 3e-4, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('NaCl_PBE0')
SYSTEMS["directory"].append("./xc_tests/exx_tests/")
SYSTEMS["Tags"].append(['bulk', 'PBE0','gamma' 'nonorth','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('O2_spin_HSE')
SYSTEMS["directory"].append("./xc_tests/exx_tests/")
SYSTEMS["Tags"].append(['molecule', 'spin', 'HSE', 'denmix', 'kerker', 'orth','smear_gauss'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('PtAu_SOC')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'SOC','kpt' 'nonorth','smear_gauss'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Si2_kpt_PBE0')
SYSTEMS["directory"].append("./xc_tests/exx_tests/")
SYSTEMS["Tags"].append(['bulk', 'PBE0','kpt' 'nonorth','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Si4_kpt_vdWDF1')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'gamma','vdWDF'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Si4_kpt_vdWDF2')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'gamma','vdWDF'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si8_atom_geopt_d3')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'denmix', 'kerker', 'relax_atom_lbfgs','gamma','smear_gauss','d3'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si8_cell_geopt_d3')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'potmix', 'relax_cell','gamma','smear_fd','d3'])
SYSTEMS["Tols"].append([tols["E_tol"], 3e-4, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('SnO_bulk_d3')
SYSTEMS["directory"].append("./xc_tests/vdW_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga','d3'])
SYSTEMS["Tols"].append([tols["E_tol"], 3e-4, 5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('AlSi_orthogonal_quick_scf')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga','orth','fast'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('AlSi_primitive_quick_relax')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga','nonorth','relax_atom_lbfgs','kpt','fast'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('BN_primitive_quick_md')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'lda','nonorth','md_nve','kpt','fast'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('LiNbO2_primitive_quick_scf')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'lda','nonorth','kpt','fast'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('SiC_orthogonal_quick_relax')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'lda','orth','relax_atom_lbfgs','gamma','fast'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('TiO2_orthogonal_quick_md')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga','orth','md_nve','gamma','fast'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
################################################################################################################
SYSTEMS["systemname"].append('BaTiO3_scan')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'orth', 'gamma','scan'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('BaTiO3_rscan')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'orth', 'gamma','scan','fast'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('BaTiO3_r2scan')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'orth', 'gamma','scan','fast'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si4_kpt_scan')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'nonorth', 'kpt','scan'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Si4_kpt_rscan')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'nonorth', 'kpt','scan'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Si4_kpt_r2scan')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'nonorth', 'kpt','scan'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
################################################################################################################
SYSTEMS["systemname"].append('Fe2_spin_scan_gamma')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'spin', 'nonorth', 'gamma','scan'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Fe2_spin_scan_kpt')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'spin', 'orth', 'kpt','scan'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Fe2_spin_rscan_kpt')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'spin', 'orth', 'kpt','scan'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Fe2_spin_r2scan_kpt')
SYSTEMS["directory"].append("./xc_tests/mgga_tests/")
SYSTEMS["Tags"].append(['bulk', 'spin', 'orth', 'kpt','scan'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('highT_Al')
SYSTEMS["directory"].append("./highT_tests/")
SYSTEMS["Tags"].append(['bulk', 'highT', 'orth', 'lda'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('highT_B4C_MD')
SYSTEMS["directory"].append("./highT_tests/")
SYSTEMS["Tags"].append(['bulk', 'highT', 'orth', 'lda','md_nvkg'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('highT_BN')
SYSTEMS["directory"].append("./highT_tests/")
SYSTEMS["Tags"].append(['bulk', 'highT', 'orth', 'gga'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('highT_H')
SYSTEMS["directory"].append("./highT_tests/")
SYSTEMS["Tags"].append(['bulk', 'highT', 'orth', 'lda'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('highT_MgSiO3_valgrind')
SYSTEMS["directory"].append("./highT_tests/")
SYSTEMS["Tags"].append(['bulk', 'highT', 'orth', 'lda','memcheck'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('highT_O8')
SYSTEMS["directory"].append("./highT_tests/")
SYSTEMS["Tags"].append(['bulk', 'highT', 'orth', 'lda'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('highT_Si8')
SYSTEMS["directory"].append("./highT_tests/")
SYSTEMS["Tags"].append(['bulk', 'highT', 'orth', 'lda'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('WSe2_cyclix')
SYSTEMS["directory"].append("./cyclix_tests/")
SYSTEMS["Tags"].append(['bulk', 'cyclix','kpt','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('HfSe2_cyclix')
SYSTEMS["directory"].append("./cyclix_tests/")
SYSTEMS["Tags"].append(['bulk', 'cyclix','gamma','smear_fd'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('FeCl2_cyclix_spin')
SYSTEMS["directory"].append("./cyclix_tests/")
SYSTEMS["Tags"].append(['cyclix','spin'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('NiCl2_cyclix_spin')
SYSTEMS["directory"].append("./cyclix_tests/")
SYSTEMS["Tags"].append(['cyclix','spin'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('WS2_cyclix_SOC')
SYSTEMS["directory"].append("./cyclix_tests/")
SYSTEMS["Tags"].append(['cyclix','SOC'])
SYSTEMS["Tols"].append([tols["E_tol"], 2e-5, tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('MoS2_cyclix_SOC')
SYSTEMS["directory"].append("./cyclix_tests/")
SYSTEMS["Tags"].append(['cyclix','SOC'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('Fe3_noncollinear')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['molecule', 'gga','noncollinear','spin'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('FePt_noncollinear')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga','noncollinear','spin'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('MnAu_noncollinear')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga','noncollinear','spin'])
SYSTEMS["Tols"].append([tols["E_tol"], tols["F_tol"], tols["stress_tol"]]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('CdS_bandstruct')
SYSTEMS["directory"].append("./")
SYSTEMS["Tags"].append(['bulk', 'gga', 'denmix', 'kerker', 'orth', 'smear_fd', 'bandstruct'])
SYSTEMS["Tols"].append([1e-5, 1e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
##################################################################################################################
SYSTEMS["systemname"].append('AlSi_mlffflag1')
SYSTEMS["directory"].append("./mlff_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'mlff1', 'md_nvkg'])
SYSTEMS["Tols"].append([1e-5, 5e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%) 
##################################################################################################################
SYSTEMS["systemname"].append('AlSi_mlffflag21')
SYSTEMS["directory"].append("./mlff_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'mlff21', 'md_nvkg'])
SYSTEMS["Tols"].append([1e-5, 5e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%) 
##################################################################################################################
SYSTEMS["systemname"].append('AlSi_mlffflag22')
SYSTEMS["directory"].append("./mlff_tests/")
SYSTEMS["Tags"].append(['bulk',  'gga', 'orth', 'mlff22', 'md_nvkg'])
SYSTEMS["Tols"].append([5e-5, 5e-4, 1]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('C_sqmlff')
SYSTEMS["directory"].append("./mlff_tests/")
SYSTEMS["Tags"].append(['bulk', 'highT', 'lda', 'orth', 'mlff1', 'md_nvkg'])
SYSTEMS["Tols"].append([5e-5, 5e-4, 1]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('AlSi_NPTNH_mlff')
SYSTEMS["directory"].append("./mlff_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'orth', 'mlff1', 'md_npt'])
SYSTEMS["Tols"].append([1e-5, 5e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
SYSTEMS["systemname"].append('AlSi_mlff_nonorth')
SYSTEMS["directory"].append("./mlff_tests/")
SYSTEMS["Tags"].append(['bulk', 'gga', 'nonorth', 'mlff1', 'md_nvkg'])
SYSTEMS["Tols"].append([1e-5, 5e-4, 0.5]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
##################################################################################################################
# < Uncomment 3 lines below and fill in the details for the new systems>
# SYSTEMS["systemname"].append('??type the system name??')
# SYSTEMS["Tols"].append([??type the E_tol, F_tol and stress_tol separated by comma??]) # E_tol(Ha/atom), F_tol(Ha/Bohr), stress_tol(%)
# SYSTEMS["Tags"].append([??type the tags for the system as strings separated by comma??])


####################################################################################################################
######################################     DO NOT CHANGE ANYTHING BELOW       ######################################
####################################################################################################################


###################################################################################################################
######################### Functions and main script (Don't change below this) #####################################
###################################################################################################################
inplace_file_content = """
{
   <insert_a_suppression_name_here>
   Memcheck:User
   fun:check_mem_is_defined_untyped
   fun:walk_type
   fun:walk_type_array
   fun:check_mem_is_defined
   fun:PMPI_Allreduce
   ...
}
{
   <insert_a_suppression_name_here>
   Memcheck:User
   fun:check_mem_is_defined_untyped
   fun:walk_type
   fun:walk_type_array
   fun:check_mem_is_defined
   fun:PMPI_Reduce
   ...
}
"""


home_directory=subprocess.check_output("pwd").strip()

if os.path.exists('./../lib/sparc'):
	os.system('cp ./../lib/sparc ./')
	os.system('chmod +x sparc')

def range_with_status(total):
    #""" iterate from 0 to total and show progress in console """
    import sys
    n = 0
    while n < total:
        done = '#' * (n + 1)
        todo = '-' * (total - n - 1)
        s = '<{0}>'.format(done + todo)
        if not todo:
            s += '\n'
        if n >= 0:
            s = 'Test Status: ' + s
        print(s, end='\r')
        sys.stdout.flush()
        yield n
        n += 1

#def findsystems(tags, folder_address, filename_systeminfo):
def findsystems(tags_systems):
	#""" Returns all the systems from SYSTEMS dictionary with tags matching with tags_systems """
	systems=[]
	tags_export=[]
	tols_export = []
	dirs_export = []

	s_all = SYSTEMS["systemname"]
	tag_all = SYSTEMS["Tags"]
	tol_all = SYSTEMS["Tols"]
	dirs_all = SYSTEMS["directory"]
	for i in range(len(s_all)):
		sys_name = s_all[i]
		tags_sys = tag_all[i]
		tol_sys = tol_all[i]
		dirs_sys = dirs_all[i]
		iftagsmatch = True
		for tag_temp in tags_systems:
			iftagsmatch = iftagsmatch and (tag_temp in tags_sys)
		if iftagsmatch == True:
			systems.append(sys_name)
			tags_export.append(tags_sys)
			tols_export.append(tol_sys)
			dirs_export.append(dirs_sys)

	data = [systems, tags_export, tols_export, dirs_export]
	return(data)


def launchsystems(systems, dirs_sys, memcheck, procs_nodes_cluster, ismempbs, ifVHQ, isorient, isserial, ismlff_copy):

	home_tests_directory = os.getcwd()
	#""" Launches the systems with memcheck, specified number of processors and with valgrid """
	with open("samplescript_cluster",'r') as f_samplePBS:
		samplePBS_content_orj = [ line.strip() for line in f_samplePBS]


	jobID=[]

	countx=0 
	for syst in systems:
		os.chdir(dirs_sys[countx])
		os.chdir(syst)
		if isorient[countx] == False:
			if os.path.isdir("temp_run"):
				files = glob.glob("temp_run/*")
				for f in files:
					os.remove(f)

				if ifVHQ == True:
					os.system("cp ./high_accuracy/*.inpt ./temp_run")
					os.system("cp ./high_accuracy/*.ion ./temp_run")
					# os.system("cp *.psp8 temp_run")
					if ismlff_copy[countx] == True:
						os.system("cp ./high_accuracy/MLFF* ./temp_run")
					if syst == "Al16Si16_NPTNH_restart" or syst == "Al16Si16_NPTNP_restart":
						os.system("cp ./standard/*.restart ./temp_run")
				if ifVHQ == False:
					os.system("cp ./standard/*.inpt ./temp_run")
					os.system("cp ./standard/*.ion ./temp_run")
					# os.system("cp *.psp8 temp_run")
					if ismlff_copy[countx] == True:
						os.system("cp ./standard/MLFF* ./temp_run")
					if syst == "Al16Si16_NPTNH_restart" or syst == "Al16Si16_NPTNP_restart":
						os.system("cp ./standard/*.restart ./temp_run")
			else:
				os.mkdir("temp_run")
				
				if ifVHQ == True:
					os.system("cp ./high_accuracy/*.inpt ./temp_run")
					os.system("cp ./high_accuracy/*.ion ./temp_run")
					# os.system("cp *.psp8 temp_run")
					if ismlff_copy[countx] == True:
						os.system("cp ./high_accuracy/MLFF* ./temp_run")
					if syst == "Al16Si16_NPTNH_restart" or syst == "Al16Si16_NPTNP_restart":
						os.system("cp ./standard/*.restart ./temp_run")
				if ifVHQ == False:
					os.system("cp ./standard/*.inpt ./temp_run")
					os.system("cp ./standard/*.ion ./temp_run")
					# os.system("cp *.psp8 temp_run")
					if ismlff_copy[countx] == True:
						os.system("cp ./standard/MLFF* ./temp_run")
					if syst == "Al16Si16_NPTNH_restart" or syst == "Al16Si16_NPTNP_restart":
						os.system("cp ./standard/*.restart ./temp_run")
		else:
			if os.path.isdir("temp_run1"):
				files = glob.glob("temp_run1/*")
				for f in files:
					os.remove(f)
				if ifVHQ == True:
					os.system("cp ./high_accuracy_orientation1/*.inpt ./temp_run1")
					os.system("cp ./high_accuracy_orientation1/*.ion ./temp_run1")
					# os.system("cp *.psp8 temp_run1")
				else:
					os.system("cp ./standard_orientation1/*.inpt ./temp_run1")
					os.system("cp ./standard_orientation1/*.ion ./temp_run1")
					# os.system("cp *.psp8 temp_run1")
			else:
				os.mkdir("temp_run1")
				if ifVHQ == True:
					os.system("cp ./high_accuracy_orientation1/*.inpt ./temp_run1")
					os.system("cp ./high_accuracy_orientation1/*.ion ./temp_run1")
					# os.system("cp *.psp8 temp_run1")
				else:
					os.system("cp ./standard_orientation1/*.inpt ./temp_run1")
					os.system("cp ./standard_orientation1/*.ion ./temp_run1")
					# os.system("cp *.psp8 temp_run1")

			if os.path.isdir("temp_run2"):
				files = glob.glob("temp_run2/*")
				for f in files:
					os.remove(f)
				if ifVHQ == True:
					os.system("cp ./high_accuracy_orientation2/*.inpt ./temp_run2")
					os.system("cp ./high_accuracy_orientation2/*.ion ./temp_run2")
					# os.system("cp *.psp8 temp_run2")
				else:
					os.system("cp ./standard_orientation2/*.inpt ./temp_run2")
					os.system("cp ./standard_orientation2/*.ion ./temp_run2")
					# os.system("cp *.psp8 temp_run2")
			else:
				os.mkdir("temp_run2")
				if ifVHQ == True:
					os.system("cp ./high_accuracy_orientation2/*.inpt ./temp_run2")
					os.system("cp ./high_accuracy_orientation2/*.ion ./temp_run2")
					# os.system("cp *.psp8 temp_run2")
				else:
					os.system("cp ./standard_orientation2/*.inpt ./temp_run2")
					os.system("cp ./standard_orientation2/*.ion ./temp_run2")
					# os.system("cp *.psp8 temp_run2")

			if os.path.isdir("temp_run3"):
				files = glob.glob("temp_run3/*")
				for f in files:
					os.remove(f)
				if ifVHQ == True:
					os.system("cp ./high_accuracy_orientation3/*.inpt ./temp_run3")
					os.system("cp ./high_accuracy_orientation3/*.ion ./temp_run3")
					# os.system("cp *.psp8 temp_run3")
				else:
					os.system("cp ./standard_orientation3/*.inpt ./temp_run3")
					os.system("cp ./standard_orientation3/*.ion ./temp_run3")
					# os.system("cp *.psp8 temp_run3")
			else:
				os.mkdir("temp_run3")
				if ifVHQ == True:
					os.system("cp ./high_accuracy_orientation3/*.inpt ./temp_run3")
					os.system("cp ./high_accuracy_orientation3/*.ion ./temp_run3")
					# os.system("cp *.psp8 temp_run3")
				else:
					os.system("cp ./standard_orientation3/*.inpt ./temp_run3")
					os.system("cp ./standard_orientation3/*.ion ./temp_run3")
					# os.system("cp *.psp8 temp_run3")
		countx=countx+1
		os.chdir(home_tests_directory)

	if ismempbs == True:
		print('This option is no longer supported!\n')
		exit()
	else:
		count = 0
		countpbs = 1
		nprocs_grp = []
		nnodes_grp= []
		sys_grp =[]
		memcheck_grp = []
		orient_grp=[]
		if ifVHQ == True:
			# count_sys_pbs = 2
			count_sys_pbs = int(len(systems)/npbs) + 1
		else:
			count_sys_pbs = int(len(systems)/npbs) + 1
			# count_sys_pbs = 5
		if isserial:
			count_sys_pbs = 1
		while count < len(systems):
			nprocs_grp = []
			nnodes_grp= []
			sys_grp =[]
			dirs_grp=[]
			memcheck_grp = []
			samplePBS_content = []
			orient_grp=[]
			for lines in samplePBS_content_orj:
				samplePBS_content.append(lines)
			if len(systems)-count > count_sys_pbs:
				for cc in range(count_sys_pbs):
					nprocs_grp.append(procs_nodes_cluster[0] * procs_nodes_cluster[1])
					nnodes_grp.append(procs_nodes_cluster[1])
					sys_grp.append(systems[count+cc])
					dirs_grp.append(dirs_sys[count+cc])
					memcheck_grp.append(memcheck[count+cc])
					orient_grp.append(isorient[count+cc])
				count = count+count_sys_pbs
			else:
				for cc in range(len(systems) - count):
					nprocs_grp.append(procs_nodes_cluster[0] * procs_nodes_cluster[1])
					nnodes_grp.append(procs_nodes_cluster[1])
					sys_grp.append(systems[count+cc])
					dirs_grp.append(dirs_sys[count+cc])
					memcheck_grp.append(memcheck[count+cc])
					orient_grp.append(isorient[count+cc])
				count = count+count_sys_pbs

			index1=0
			for lines in samplePBS_content:
				if re.findall(r'mpirun',lines) == ['mpirun']:
					samplePBS_content.remove(lines)
				if re.findall(r'srun',lines) == ['srun']:
					samplePBS_content.remove(lines)
				index1 = index1+1

			for ll in range(len(sys_grp)):
				if memcheck_grp[ll] == False:
					#text_temp = "mpirun -env MV2_ENABLE_AFFINITY=1 -env MV2_CPU_BINDING_POLICY=bunch -np "+str(nprocs_grp[ll])+" ./sparc"+ " -name ./"+sys_grp[ll]+"/temp_run/"+sys_grp[ll]+" -log_summary > "+sys_grp[ll]+".log"+"\n"
					if orient_grp[ll] == False:
						samplePBS_content.append(MPI_command+" "+" ./sparc"+ " -name ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run/"+sys_grp[ll]+" -log_summary > ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run/"+sys_grp[ll]+".log")
						samplePBS_content.append("\n")
					else:
						samplePBS_content.append(MPI_command+" "+" ./sparc"+ " -name ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run1/"+sys_grp[ll]+" -log_summary > ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run1/"+sys_grp[ll]+".log")
						samplePBS_content.append("\n")
						samplePBS_content.append(MPI_command+" "+" ./sparc"+ " -name ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run2/"+sys_grp[ll]+" -log_summary > ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run2/"+sys_grp[ll]+".log")
						samplePBS_content.append("\n")
						samplePBS_content.append(MPI_command+" "+" ./sparc"+ " -name ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run3/"+sys_grp[ll]+" -log_summary > ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run3/"+sys_grp[ll]+".log")
						samplePBS_content.append("\n")
				else:
					if orient_grp[ll] == False:
						#text_temp = "mpirun -env MV2_ENABLE_AFFINITY=1 -env MV2_CPU_BINDING_POLICY=bunch -np "+str(nprocs_grp[ll])+" valgrind --leak-check=full --track-origins=yes --suppressions=./"+sys_grp[ll]+"/temp_run/inplace_reduce.supp --log-file=valgrind_out ./sparc -name ./"+sys_grp[ll]+"/temp_run/"+sys_grp[ll]+" -log_summary > "+sys_grp[ll]+".log"+"\n"
						samplePBS_content.append(MPI_command+" "+" valgrind --leak-check=full --show-reachable=yes --error-limit=no  --suppressions=./minimal.supp --gen-suppressions=all --log-file="+dirs_grp[ll]+sys_grp[ll]+"/temp_run/valgrind_out.log.%p ./sparc -name ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run/"+sys_grp[ll]+" -log_summary > ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run/"+sys_grp[ll]+".log")
						samplePBS_content.append("\n")
					else:
						samplePBS_content.append(MPI_command+" "+" valgrind --leak-check=full --show-reachable=yes --error-limit=no  --suppressions=./minimal.supp --gen-suppressions=all --log-file="+dirs_grp[ll]+sys_grp[ll]+"/temp_run1/valgrind_out.log.%p ./sparc -name ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run1/"+sys_grp[ll]+" -log_summary > ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run1/"+sys_grp[ll]+".log")
						samplePBS_content.append("\n")
						samplePBS_content.append(MPI_command+" "+" valgrind --leak-check=full --show-reachable=yes --error-limit=no  --suppressions=./minimal.supp --gen-suppressions=all --log-file="+dirs_grp[ll]+sys_grp[ll]+"/temp_run2/valgrind_out.log.%p ./sparc -name ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run2/"+sys_grp[ll]+" -log_summary > ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run2/"+sys_grp[ll]+".log")
						samplePBS_content.append("\n")
						samplePBS_content.append(MPI_command+" "+" valgrind --leak-check=full --show-reachable=yes --error-limit=no  --suppressions=./minimal.supp --gen-suppressions=all --log-file="+dirs_grp[ll]+sys_grp[ll]+"/temp_run3/valgrind_out.log.%p ./sparc -name ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run3/"+sys_grp[ll]+" -log_summary > ./"+dirs_grp[ll]+sys_grp[ll]+"/temp_run3/"+sys_grp[ll]+".log")
						samplePBS_content.append("\n")
			f_pbs = open("launch_"+str(countpbs)+launch_cluster_extension,"w")
			for lines in samplePBS_content:
					f_pbs.write(lines+"\n")
			f_pbs.close()
			temp = "launch_"+str(countpbs)+launch_cluster_extension
			p = subprocess.check_output([command_launch_extension, temp])
			countpbs = countpbs+1

def isfinished(syst, dirs_sys, isorientsys):
	home_tests_directory = os.getcwd()
	#""" Returns true if the "syst" has finished running """
	if isorientsys == False:
		if os.path.isfile(dirs_sys+syst+"/temp_run/"+syst+".out"):
			with open(dirs_sys+"./"+syst+"/temp_run/"+syst+".out",'r') as f_out:
				f_out_content = [ line.strip() for line in f_out ]
			if "Timing info" in f_out_content:
				return True
			else:
				return False
			f_out.close()
		else:
			return False
	else:
		if os.path.isfile(dirs_sys+"./"+syst+"/temp_run1/"+syst+".out") and os.path.isfile(dirs_sys+"./"+syst+"/temp_run2/"+syst+".out") and os.path.isfile(dirs_sys+"./"+syst+"/temp_run3/"+syst+".out"):
			with open(dirs_sys+"./"+syst+"/temp_run1/"+syst+".out",'r') as f_out1:
				f_out_content1 = [ line.strip() for line in f_out1 ]
			with open(dirs_sys+"./"+syst+"/temp_run2/"+syst+".out",'r') as f_out2:
				f_out_content2 = [ line.strip() for line in f_out2 ]
			with open(dirs_sys+"./"+syst+"/temp_run3/"+syst+".out",'r') as f_out3:
				f_out_content3 = [ line.strip() for line in f_out3 ]
			if ("Timing info" in f_out_content1) and ("Timing info" in f_out_content2) and ("Timing info" in f_out_content3):
				return True
			else:
				return False
			f_out.close()
		else:
			return False




def ReadOutFile(filepath, isMD, geopt_typ, isSpin):
	#""" Reads .out file from SPARC runs and reference """
	with open(filepath,'r') as f_out:
		f_out_content = [ line.strip() for line in f_out ]
	isPrintF = True
	isPrintStress = False
	isPrintPres = False
	isPrintAtoms = True
	isPrintCell = False
	no_atoms = 0
	stressDim = 3

	E = []
	walltime = []
	magnetization = []
	pressure = []
	index=0
	isbandgap = False
	nstates=0


	for lines in f_out_content:
		if re.findall(r"PRINT_FORCES",lines) == ['PRINT_FORCES']:
			val_temp = re.findall(r'\d',lines)
			val_temp = int(val_temp[0])
			if val_temp == 1:
				isPrintF = True
			elif val_temp == 0:
				isPrintF = False
		if re.findall(r"NSTATES",lines) == ['NSTATES']:
			nstates_temp = re.findall(r'\d+',lines)
			nstates = int(nstates_temp[0])
		if re.findall(r"PRINT_EIGEN",lines) == ['PRINT_EIGEN']:
			prteigen_temp =  re.findall(r'\d',lines)
			if int(prteigen_temp[0]) == 1:
				isbandgap = True

		if re.findall(r"PRINT_ATOMS",lines) == ['PRINT_ATOMS']:
			val_temp = re.findall(r'\d',lines)
			val_temp = int(val_temp[0])
			if val_temp == 1:
				isPrintAtoms = True
			elif val_temp == 0:
				isPrintAtoms = False
		if re.findall(r"CALC_STRESS",lines) == ['CALC_STRESS']:
			val_temp = re.findall(r'\d',lines)
			val_temp = int(val_temp[0])
			if val_temp == 1:
				isPrintStress = True
			elif val_temp == 0:
				isPrintStress = False
		if re.findall(r"CALC_PRES",lines) == ['CALC_PRES']:
			val_temp = re.findall(r'\d',lines)
			val_temp = int(val_temp[0])
			if val_temp == 1:
				isPrintPres = True
			elif val_temp == 0:
				isPrintPres = False
		if re.findall(r"Total number of atoms",lines) == ['Total number of atoms']:
			atom_temp =  re.findall(r'\d+',lines)
			no_atoms = int(atom_temp[0])
		if re.findall(r"Free energy per atom",lines) == ['Free energy per atom']:
			E_temp = re.findall(r'[+-]?\d+\.\d+[E][+-]\d+',lines)
			E.append(float(E_temp[0]))
		if re.findall(r"Total walltime",lines) == ['Total walltime']:
			wall_temp = re.findall(r'\d+\.\d+',lines)
			walltime.append(float(wall_temp[0]))
		if isPrintPres ==  True:
			if re.findall(r'Pressure',lines) == ['Pressure']:
				pres_temp = re.findall(r'\b[+-]?\d+\.\d+E[+-]\d+\b', lines)
				pressure.append(float(pres_temp[0]))
		if re.findall(r"BC",lines) == ['BC']:
			if lines == ['BC: P P P']:
				stressDim = 3
			if lines == ['BC: P P D'] or lines == ['BC: D P P'] or lines == ['BC: P D P']:
				stressDim = 2
			if lines == ['BC: P D D'] or lines == ['BC: D D P'] or lines == ['BC: D P D']:
				stressDim = 1
			if lines == ['BC: D D D']:
				stressDim = 0
		if isSpin == True:
			if isMD ==  True:
				if re.findall(r'Total number of SCF',lines) == ['Total number of SCF']:
					temp_spin = re.findall(r'\b[+-]?\d+\.\d+E[+-]\d+\b',f_out_content[index-1])
					magnetization.append(float(temp_spin[1]))
			else:
				if re.findall(r'Total number of SCF',lines) == ['Total number of SCF']:
					temp_spin = re.findall(r'\b[+-]?\d+\.\d+E[+-]\d+\b',f_out_content[index-1])
					magnetization=float(temp_spin[1])
		index=index+1
	if isMD == None and geopt_typ ==  None:
		SCF_no = 0
		for lines in f_out_content:
			if re.findall("Total number of SCF",lines):
				SCF_no = float(re.findall("\d+",lines)[0])
	else:
		MD_iter = len(E)
		SCF_no=[]
		for n_md in range(MD_iter):
			SCF_no.append(0)
		count1 = 0
		for lines in f_out_content:
			if re.findall("Total number of SCF",lines):
				# SCF_no.append(float(re.findall("\d+",lines)[0]))
				SCF_no[count1] = float(re.findall("\d+",lines)[0])
				count1=count1+1


	if geopt_typ ==  "cell_relax":
		isPrintF = False
		isPrintCell = True
		isPrintAtoms  = False
		isPrintStress = True
	if geopt_typ == "atom_relax":
		isPrintF = True
		isPrintCell = False
		isPrintAtoms  = True
	if geopt_typ == "full_relax":
		isPrintF = True
		isPrintCell = True
		isPrintAtoms  = True
		isPrintStress = True
	assert (no_atoms>0 and E != [] and walltime != []),"Problem in out file for system "+filepath

	Info = {"isPrintF": isPrintF,
		"isPrintStress": isPrintStress,
		"isPrintPres": isPrintPres,
		"isPrintAtoms": isPrintAtoms,
		"isbandgap": isbandgap,
		"no_atoms": no_atoms,
		"nstates": nstates,
		"stressDim": stressDim,
		"magnetization": magnetization,
		"E": E,
		"pressure": pressure,
		"walltime": walltime,
		"isPrintCell": isPrintCell,
		"SCF_no": SCF_no}
	return(Info)

def ReadStaticFile(filepath, info_out):

	#""" Reads .static file from SPARC runs and reference """

	with open(filepath,'r') as f_static:
		f_static_content = [ line.strip() for line in f_static ]
	force = []
	stress = []
	index=0
	
	for lines in f_static_content:
		if info_out["isPrintF"] == True:
			if lines == 'Atomic forces (Ha/Bohr):':
				F_tempscf =[]
				for i in range(info_out["no_atoms"]):
					line_temp = f_static_content[index+i+1]
					F_atom_temp =  re.findall(r'\b[+-]?[0-9]+\.[0-9]+E[+-]?[0-9]+\b',line_temp)
					for j in range(len(F_atom_temp)):
						F_atom_temp[j] = float(F_atom_temp[j])
					F_tempscf.append(F_atom_temp)
				force=F_tempscf
		if info_out["isPrintStress"] == True:
			if lines == 'Stress (GPa):' or lines=='Stress (Ha/Bohr**2):' or lines=='Atomic forces (Ha/Bohr):':
				St_tempscf =[]
				for i in range(info_out["stressDim"]):
					line_temp = f_static_content[index+i+1]
					St_atom_temp =  re.findall(r'\b[+-]?[0-9]+\.[0-9]+E[+-]?[0-9]+\b',line_temp)
					for j in range(len(St_atom_temp)):
						St_atom_temp[j] = float(St_atom_temp[j])
					St_tempscf.append(St_atom_temp)
				stress=St_tempscf
		index=index+1
	### Error Handling ###
	truth1 = True
	truth2=True
	if info_out["isPrintF"] and force !=[]:
		truth1=True
	elif info_out["isPrintF"]==False and force ==[]:
		truth1=True
	else:
		truth1=False
	if info_out["isPrintStress"] and stress !=[]:
		truth2=True
	elif info_out["isPrintStress"]==False and stress ==[]:
		truth2=True
	else:
		truth2=False
	assert (truth1 and truth2),"Problem in static file for system "+filepath
	### Error Handling ###
	Info_static = {"stress": stress,
				   "force": force,
				   }
	return(Info_static)



def ReadGeoptFile(filepath, info_out):

	#""" Reads .geopt file from SPARC runs and reference """
	with open(filepath,'r') as f_geopt:
		f_geopt_content = [ line.strip() for line in f_geopt ]
	force = []
	stress = []
	scfpos = []
	cell = []

	index = 0
	for lines in f_geopt_content:
		if info_out["isPrintF"] == True:
			if lines == ':F(Ha/Bohr):':
				F_tempscf =[]
				for i in range(info_out["no_atoms"]):
					line_temp = f_geopt_content[index+i+1]
					F_atom_temp =  re.findall(r'\b[+-]?[0-9]+\.[0-9]+\b',line_temp)
					for j in range(len(F_atom_temp)):
						F_atom_temp[j] = float(F_atom_temp[j])
					F_tempscf.append(F_atom_temp)
				force.append(F_tempscf)
		if info_out["isPrintAtoms"] == True:
			if lines == ':R(Bohr):':
				pos_tempscf =[]
				for i in range(info_out["no_atoms"]):
					line_temp = f_geopt_content[index+i+1]
					pos_atom_temp =  re.findall(r'\b[+-]?[0-9]+\.[0-9]+\b',line_temp)
					for j in range(len(pos_atom_temp)):
						pos_atom_temp[j] = float(pos_atom_temp[j])
					pos_tempscf.append(pos_atom_temp)
				scfpos.append(pos_tempscf)
		if info_out["isPrintStress"] == True:
			if lines == ':STRESS:':
				St_tempscf =[]
				for i in range(info_out["stressDim"]):
					line_temp = f_geopt_content[index+i+1]
					St_atom_temp =  re.findall(r'\b[+-]?[0-9]+\.[0-9]+E[+-]?[0-9]+\b',line_temp)
					for j in range(len(St_atom_temp)):
						St_atom_temp[j] = float(St_atom_temp[j])
					St_tempscf.append(St_atom_temp)
				stress.append(St_tempscf)
		if info_out["isPrintCell"] == True:
			if re.findall(r'CELL', lines) == ['CELL']:
				cell_temp = re.findall(r'\b[+-]?[0-9]+\.[0-9]+E[+-]?[0-9]+\b',lines)
				for k in range(len(cell_temp)):
					cell_temp[k] = float(cell_temp[k])
				cell.append(cell_temp)
		index=index+1
	### Error Handling ###
	truth1=True
	truth2=True
	truth3=True
	truth4=True
	if info_out["isPrintF"] and force !=[]:
		truth1=True
	elif info_out["isPrintF"]==False and force ==[]:
		truth1=True
	else:
		truth1=False
	if info_out["isPrintStress"] and stress !=[]:
		truth2=True
	elif info_out["isPrintStress"]==False and stress ==[]:
		truth2=True
	else:
		truth2=False
	if info_out["isPrintAtoms"] and scfpos !=[]:
		truth3=True
	elif info_out["isPrintAtoms"]==False and scfpos ==[]:
		truth3=True
	else:
		truth3=False
	if info_out["isPrintCell"] and cell !=[]:
		truth4=True
	elif info_out["isPrintCell"]==False and cell ==[]:
		truth4=True
	else:
		truth4=False

	assert (truth1 and truth2 and truth3 and truth4),"Problem in geopt file for system "+filepath
	### Error Handling ###
	Info_geopt = {"stress": stress,
				   "force": force,
				   "scfpos": scfpos,
				   "cell": cell}
	return(Info_geopt)


def ReadAimdFile(filepath, info_out):

	#""" Reads .aimd file from SPARC runs and reference """
	with open(filepath,'r') as f_aimd:
		f_aimd_content = [ line.strip() for line in f_aimd ]
	force = []
	stress = []
	scfpos = []
	KEN = []
	ionic_stress = []
	velocity = []

	index = 0
	for lines in f_aimd_content:
		if re.findall(r':KEN:',lines) == [':KEN:']:
			m = re.search(r'(\b:?KEN:?\s+)(\d\.\d+E[+-]\d+)\b',lines)
			ken_temp = float(m.group(2))
			KEN.append(ken_temp)
		if info_out["isPrintF"] == True:
			if lines == ':F:':
				F_tempMD = []
				for aa in range(info_out["no_atoms"]):
					line_temp = f_aimd_content[index+aa+1]
					F_atom_temp =  re.findall(r'\b[+-]?\d+\.\d+E[+-]\d+\b',line_temp)
					for j in range(len(F_atom_temp)):
						F_atom_temp[j] = float(F_atom_temp[j])
					F_tempMD.append(F_atom_temp)
				force.append(F_tempMD)
		if True:
			if lines == ':V:':
				V_tempMD = []
				for aa in range(info_out["no_atoms"]):
					line_temp = f_aimd_content[index+aa+1]
					V_atom_temp =  re.findall(r'\b[+-]?\d+\.\d+E[+-]\d+\b',line_temp)
					for j in range(len(V_atom_temp)):
						V_atom_temp[j] = float(V_atom_temp[j])
					V_tempMD.append(V_atom_temp)
				velocity.append(V_tempMD)

		if info_out["isPrintStress"]:
			if lines == ':STRIO:':
				st_tempMD = []
				for bb in range(3):
					line_temp = f_aimd_content[index+bb+1]
					st_atom_temp =  re.findall(r'\b[+-]?\d+\.\d+E[+-]\d+\b',line_temp)
					for j in range(len(st_atom_temp)):
						st_atom_temp[j] = float(st_atom_temp[j])
					st_tempMD.append(st_atom_temp)
				ionic_stress.append(st_tempMD)
		if info_out["isPrintStress"]:
			if lines == ':STRESS:':
				st_tempMD = []
				for bb in range(3):
					line_temp = f_aimd_content[index+bb+1]
					st_atom_temp =  re.findall(r'\b[+-]?\d+\.\d+E[+-]\d+\b',line_temp)
					for j in range(len(st_atom_temp)):
						st_atom_temp[j] = float(st_atom_temp[j])
					st_tempMD.append(st_atom_temp)
				stress.append(st_tempMD)
		if info_out["isPrintAtoms"]:
			if lines == ':R:':
				pos_tempscf =[]
				for i in range(info_out["no_atoms"]):
					line_temp = f_aimd_content[index+i+1]
					pos_atom_temp =  re.findall(r'\b[+-]?[0-9]+\.[0-9]+\b',line_temp)
					for j in range(len(pos_atom_temp)):
						pos_atom_temp[j] = float(pos_atom_temp[j])
					pos_tempscf.append(pos_atom_temp)
				scfpos.append(pos_tempscf)
		index = index+1
	### Error Handling ###
	truth1=True
	truth2=True
	truth3=True
	truth4=True
	if info_out["isPrintF"] and force !=[]:
		truth1=True
	elif info_out["isPrintF"]==False and force ==[]:
		truth1=True
	else:
		truth1=False
	if info_out["isPrintStress"] and stress !=[] and ionic_stress != []:
		truth2=True
	elif info_out["isPrintStress"]==False and stress ==[] and ionic_stress == []:
		truth2=True
	else:
		truth2=False
	if info_out["isPrintAtoms"] and scfpos !=[]:
		truth3=True
	elif info_out["isPrintAtoms"]==False and scfpos ==[]:
		truth3=True
	else:
		truth3=False
	if KEN == []:
		truth4 = False
	assert (truth1 and truth2 and truth3 and truth4),"Problem in aimd file for system "+filepath
	### Error Handling ###
	Info_aimd = {"stress": stress,
				   "ionic_stress": ionic_stress,
				   "velocity": velocity,
				   "force": force,
				   "scfpos": scfpos,
				   "KEN": KEN}
	return(Info_aimd)

def ReadEigenFile_molecule(filepath, info_out):

	if info_out["isbandgap"] == False:
		bandgap = 0
	else:
		with open(filepath,'r') as f_eigen:
			f_eigen_content = [ line.strip() for line in f_eigen ]
		index = 0
		for lines in f_eigen_content:
			if re.findall(r'eigval',lines) == ['eigval']:
				nstates = info_out["nstates"]
				eigval =[]
				occ =[]
				n = []
				for ltemp in range(nstates):
					band_info_temp = re.findall(r'\b[+-]?\d+\.\d+E[+-]\d+\b',f_eigen_content[index+1+ltemp])
					eigval.append(float(band_info_temp[0]))
					band_info_temp = re.findall(r'\b[+-]?[0-9]+\.[0-9]+\b',f_eigen_content[index+1+ltemp])
					occ.append(float(band_info_temp[0]))
					band_info_temp = re.findall(r'\b\d\b',f_eigen_content[index+1+ltemp])
					n.append(int(band_info_temp[0]))
				for ltemp in range(nstates):
					if occ[ltemp] < 0.01:
						bandgap = eigval[ltemp] - eigval[ltemp-1]
				break
			index=index+1
				
	return(bandgap)


# def ReadmemoutputFile(isorientsys, ismempbs, ifref, ifVHQ):
# 	memused =0
# 	if isorientsys == False:
# 		memused =0
# 		ismemused=False
# 		if ismempbs == True:
# 			ismemused=True
# 			if ifref == False:
# 				with open("./temp_run/output.sparc",'r') as f_sparc:
# 					f_sparc_content = [ line.strip() for line in f_sparc ]
# 			else:
# 				if ifVHQ == True:
# 					with open("./high_accuracy/output.sparc",'r') as f_sparc:
# 						f_sparc_content = [ line.strip() for line in f_sparc ]
# 				else:
# 					with open("./standard/output.sparc",'r') as f_sparc:
# 						f_sparc_content = [ line.strip() for line in f_sparc ]
# 			for lines in f_sparc_content:
# 				line_str=re.findall(r'Rsrc Used:',lines)
# 				if line_str == ['Rsrc Used:']:
# 					temp1=re.findall(r'\d+',lines)
# 					memused = float(temp1[-2])
# 					break
# 	else:
# 		memused =[]
# 		ismemused=False
# 		if ismempbs == True:
# 			ismemused=True
# 			if ifref == False:
# 				with open("./temp_run1/output.sparc",'r') as f_sparc1:
# 					f_sparc_content1 = [ line.strip() for line in f_sparc1 ]
# 				with open("./temp_run2/output.sparc",'r') as f_sparc2:
# 					f_sparc_content2 = [ line.strip() for line in f_sparc2 ]
# 				with open("./temp_run3/output.sparc",'r') as f_sparc3:
# 					f_sparc_content3 = [ line.strip() for line in f_sparc3 ]
# 			else:
# 				if ifVHQ == True:
# 					with open("./high_accuracy_orientation1/output.sparc",'r') as f_sparc1:
# 						f_sparc_content1 = [ line.strip() for line in f_sparc1 ]
# 					with open("./high_accuracy_orientation2/output.sparc",'r') as f_sparc2:
# 						f_sparc_content2 = [ line.strip() for line in f_sparc2 ]
# 					with open("./high_accuracy_orientation3/output.sparc",'r') as f_sparc3:
# 						f_sparc_content3 = [ line.strip() for line in f_sparc3 ]
# 				else:
# 					with open("./standard_orientation1/output.sparc",'r') as f_sparc1:
# 						f_sparc_content1 = [ line.strip() for line in f_sparc1 ]
# 					with open("./standard_orientation1/output.sparc",'r') as f_sparc2:
# 						f_sparc_content2 = [ line.strip() for line in f_sparc2 ]
# 					with open("./standard_orientation1/output.sparc",'r') as f_sparc3:
# 						f_sparc_content3 = [ line.strip() for line in f_sparc3 ]
# 			for lines in f_sparc_content1:
# 				line_str=re.findall(r'Rsrc Used:',lines)
# 				if line_str == ['Rsrc Used:']:
# 					temp1=re.findall(r'\d+',lines)
# 					memused.append(float(temp1[-2]))
# 					break
# 			for lines in f_sparc_content2:
# 				line_str=re.findall(r'Rsrc Used:',lines)
# 				if line_str == ['Rsrc Used:']:
# 					temp1=re.findall(r'\d+',lines)
# 					memused.append(float(temp1[-2]))
# 					break
# 			for lines in f_sparc_content3:
# 				line_str=re.findall(r'Rsrc Used:',lines)
# 				if line_str == ['Rsrc Used:']:
# 					temp1=re.findall(r'\d+',lines)
# 					memused.append(float(temp1[-2]))
# 					break
# 			if ifref == True:
# 				memused=memused[0]
# 	return ismemused,memused


def Readvalgridout(isorientsys, ismemch, ifref, ifVHQ):
	memlost=0


	if isorientsys == False:
		all_files = os.listdir('./temp_run')
		valgrind_files = []
		for files in all_files:
			if re.findall('valgrind',files) == ['valgrind']:
				valgrind_files.append(files)
		memlost=0
		# ismemch = False
		if ((ismemch==True) and (ifref == False)):
			ismemch = True
			for files in valgrind_files:
				with open("./temp_run/"+files,'r') as f_valg:
					f_valg_content = [ line.strip() for line in f_valg ]
				line_count = 0
				for lines in f_valg_content:
					lost_str = re.findall(r'\bLEAK SUMMARY\b',lines)
					if lost_str ==['LEAK SUMMARY']:
						m = re.findall(r'\d+[,]?[\d+]*',f_valg_content[line_count+1])
						memlost = memlost+float(m[1].replace(',',''))
						break
					line_count = line_count+1
	else:
		all_files = os.listdir('./temp_run1')
		valgrind_files = []
		for files in all_files:
			if re.findall('valgrind',files) == ['valgrind']:
				valgrind_files.append(files)
		memlost=0
		# ismemch = False
		if ((ismemch==True) and (ifref == False)):
			ismemch = True
			for files in valgrind_files:
				with open("./temp_run1/"+files,'r') as f_valg:
					f_valg_content = [ line.strip() for line in f_valg ]
				line_count = 0
				for lines in f_valg_content:
					lost_str = re.findall(r'\bLEAK SUMMARY\b',lines)
					if lost_str ==['LEAK SUMMARY']:
						m = re.findall(r'\d+[,]?[\d+]*',f_valg_content[line_count+1])
						memlost = memlost+float(m[1].replace(',',''))
						break
					line_count = line_count+1

	return ismemch, memlost


def getInfo(syst, sys_dir, singlept, Type, ifref,memcheck, ismempbs, isspin, ifVHQ, isorientsys, tolerance):
	#""" Reads from the output files (.out, .static, .aimd, .geopt, valgrind_out) of SPARC and returns the E, F, Stress, positions in a dictionary """
	home_tests_directory = os.getcwd()
	os.chdir(sys_dir+syst)


	

	if (singlept == True):
		# Extract energy, forces, stress, no of scf iteration, walltime, 
		#------------------------ Memecheck from valgrind ----------------------------#
		ismemch, memlost=Readvalgridout(isorientsys, memcheck, ifref, ifVHQ)

		#------------------------ Memecheck from valgrind ----------------------------#

		#------------------------ Memory from output.sparc ----------------------------#
		# ismemused,memused = ReadmemoutputFile(isorientsys, ismempbs, ifref, ifVHQ)
		
		
		if ifref == False:
			if isorientsys == False:
				infout = ReadOutFile("./temp_run/"+syst+".out", None, None, isspin)
				infstatic = ReadStaticFile("./temp_run/"+syst+".static", infout)
				#------------------------ Bandgap ----------------------------#
				bandgap = ReadEigenFile_molecule("./temp_run/"+syst+".eigen", infout)
				#------------------------ Bandgap ----------------------------#
				
			else:
				infout1 = ReadOutFile("./temp_run1/"+syst+".out", None, None, isspin)
				infstatic1 = ReadStaticFile("./temp_run1/"+syst+".static", infout1)
				#------------------------ Bandgap ----------------------------#
				bandgap = ReadEigenFile_molecule("./temp_run1/"+syst+".eigen", infout1)
				#------------------------ Bandgap ----------------------------#
				infout2 = ReadOutFile("./temp_run2/"+syst+".out", None, None, isspin)
				infstatic2 = ReadStaticFile("./temp_run2/"+syst+".static", infout2)
				infout3 = ReadOutFile("./temp_run3/"+syst+".out", None, None, isspin)
				infstatic3 = ReadStaticFile("./temp_run3/"+syst+".static", infout3)
		else:
			if isorientsys == False:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy/"+syst+".refout", None, None, isspin)
					infstatic = ReadStaticFile("./high_accuracy/"+syst+".refstatic", infout)
					#------------------------ Bandgap ----------------------------#
					bandgap = ReadEigenFile_molecule("./high_accuracy/"+syst+".refeigen", infout)
					
					#------------------------ Bandgap ----------------------------#
				else:

					infout = ReadOutFile("./standard/"+syst+".refout", None, None, isspin)
					infstatic = ReadStaticFile("./standard/"+syst+".refstatic", infout)
					#------------------------ Bandgap ----------------------------#
					bandgap = ReadEigenFile_molecule("./standard/"+syst+".refeigen", infout)

					#------------------------ Bandgap ----------------------------#
			else:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy_orientation1/"+syst+".refout", None, None, isspin)
					infstatic = ReadStaticFile("./high_accuracy_orientation1/"+syst+".refstatic", infout)
					#------------------------ Bandgap ----------------------------#
					bandgap = ReadEigenFile_molecule("./high_accuracy_orientation1/"+syst+".refeigen", infout)
					#------------------------ Bandgap ----------------------------#
				else:
					infout = ReadOutFile("./standard_orientation1/"+syst+".refout", None, None, isspin)
					infstatic = ReadStaticFile("./standard_orientation1/"+syst+".refstatic", infout)
					#------------------------ Bandgap ----------------------------#
					bandgap = ReadEigenFile_molecule("./standard_orientation1/"+syst+".refeigen", infout)
					#------------------------ Bandgap ----------------------------#

		if isorientsys == False or ifref == True:

			E = infout["E"]
			walltime = infout["walltime"]
			SCF_no = infout["SCF_no"]
			force = []
			pressure = []
			stress=[]
			magnetization = infout["magnetization"]
			if infout["isPrintF"] == True:
				force = infstatic["force"]
			if infout["isPrintStress"] == True:
				stress = infstatic["stress"]
			if infout["isPrintPres"] == True:
				pressure = infout["pressure"]
			no_atoms = infout["no_atoms"]
			isbandgap = infout["isbandgap"]

		else:
			E=[ infout1["E"], infout2["E"], infout3["E"]]
			SCF_no = infout1["SCF_no"]
			walltime = infout1["walltime"]#[infout1["walltime"],infout2["walltime"],infout3["walltime"]]
			force = []
			pressure = []
			stress=[]
			magnetization = infout1["magnetization"]#[infout1["magnetization"],infout2["magnetization"],infout3["magnetization"]]
			if infout1["isPrintF"] == True:
				force = infstatic1["force"]#[infstatic1["force"],infstatic2["force"],infstatic3["force"]]
			if infout1["isPrintStress"] == True:
				stress = infstatic1["stress"]#[infstatic1["stress"],infstatic2["stress"],infstatic3["stress"]]
			if infout1["isPrintPres"] == True:
				pressure = infout1["pressure"]#[infout1["pressure"],infout1["pressure"],infout1["pressure"]]
			no_atoms = infout1["no_atoms"]
			isbandgap = infout1["isbandgap"]

		Info = {"Type": "singlept",
			"isspin": isspin,
			"ismemcheck": ismemch,
			"isbandgap": isbandgap,
			"bandgap": bandgap,
			"energy": E,
			"force": force,
			"stress": stress,
			"walltime": walltime,
			"memlost": memlost,
			"magnetization": magnetization,
			"pressure": pressure,
			"no_atoms": no_atoms,
			"isorient": isorientsys,
			"tolerance": tolerance,
			"SCF_no": SCF_no,
			"bandgap": bandgap}
		os.chdir(home_tests_directory)
		return(Info)

	elif ((singlept == False) and (Type == "relax_atom")):
		
		#------------------------ Memecheck from valgrind ----------------------------#
		ismemch, memlost=Readvalgridout(isorientsys, memcheck, ifref, ifVHQ)
		
		#------------------------ Memecheck from valgrind ----------------------------#

		#------------------------ Memory from output.sparc ----------------------------#
		# ismemused,memused = ReadmemoutputFile(isorientsys, ismempbs, ifref, ifVHQ)
		
		#------------------------ Memory from output.sparc ----------------------------#
		if ifref == False:
			if isorientsys == False:
				infout = ReadOutFile("./temp_run/"+syst+".out", False, "atom_relax", isspin)
				infgeopt = ReadGeoptFile("./temp_run/"+syst+".geopt", infout)
			else:
				infout1 = ReadOutFile("./temp_run1/"+syst+".out", False, "atom_relax", isspin)
				infgeopt1 = ReadGeoptFile("./temp_run1/"+syst+".geopt", infout1)
				infout2 = ReadOutFile("./temp_run2/"+syst+".out", False, "atom_relax", isspin)
				infgeopt2 = ReadGeoptFile("./temp_run2/"+syst+".geopt", infout2)
				infout3 = ReadOutFile("./temp_run3/"+syst+".out", False, "atom_relax", isspin)
				infgeopt3 = ReadGeoptFile("./temp_run3/"+syst+".geopt", infout3)
		else:
			if isorientsys == False:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy/"+syst+".refout", False, "atom_relax", isspin)
					infgeopt = ReadGeoptFile("./high_accuracy/"+syst+".refgeopt", infout)
				else:
					infout = ReadOutFile("./standard/"+syst+".refout", False, "atom_relax", isspin)
					infgeopt = ReadGeoptFile("./standard/"+syst+".refgeopt", infout)
			else:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy_orientation1/"+syst+".refout", False, "atom_relax", isspin)
					infgeopt = ReadGeoptFile("./high_accuracy_orientation1/"+syst+".refgeopt", infout)
				else:
					infout = ReadOutFile("./standard_orientation1/"+syst+".refout", False, "atom_relax", isspin)
					infgeopt = ReadGeoptFile("./standard_orientation1/"+syst+".refgeopt", infout)
		if isorientsys == False or ifref == True:
			E = infout["E"]
			SCF_no = infout["SCF_no"]
			walltime = infout["walltime"]
			scfpos = infgeopt["scfpos"]
			force = []
			pressure = infout["pressure"]
			magnetization = infout["magnetization"]
			if infout["isPrintF"] == True:
				force = infgeopt["force"]
			no_atoms = infout["no_atoms"]
		else:
			E = [infout1["E"],infout2["E"],infout3["E"]]
			SCF_no = infout1["SCF_no"]
			walltime = infout1["walltime"]#[infout1["walltime"],infout2["walltime"],infout3["walltime"]]
			scfpos = infgeopt1["scfpos"]#[infgeopt1["scfpos"],infgeopt2["scfpos"],infgeopt3["scfpos"]]
			force = []
			pressure = infout1["pressure"]#[infout1["pressure"],infout2["pressure"],infout3["pressure"]]
			magnetization = infout1["magnetization"]#[infout1["magnetization"],infout2["magnetization"],infout3["magnetization"]]
			if infout1["isPrintF"] == True:
				force = infgeopt1["force"]#[infgeopt1["force"],infgeopt2["force"],infgeopt3["force"]]
			no_atoms = infout1["no_atoms"]

		Info = {"Type": "relax_atom",
			"isspin": isspin,
			"ismemcheck": ismemch,
			"energy": E,
			"walltime": walltime,
			"force": force,
			"scfpos": scfpos,
			"memlost": memlost,
			# "memused": memused,
			"magnetization": magnetization,
			"pressure": pressure,
			"no_atoms": no_atoms,
			"isorient": isorientsys,
			"tolerance": tolerance,
			"SCF_no": SCF_no}

		os.chdir(home_tests_directory)
		return(Info)

	elif ((singlept == False) and (Type == "relax_cell")):
		#------------------------ Memecheck from valgrind ----------------------------#
		ismemch, memlost=Readvalgridout(isorientsys, memcheck, ifref, ifVHQ)
		
		#------------------------ Memecheck from valgrind ----------------------------#

		#------------------------ Memory from output.sparc ----------------------------#
		# ismemused,memused = ReadmemoutputFile(isorientsys, ismempbs, ifref, ifVHQ)
		
		#------------------------ Memory from output.sparc ----------------------------#
		if ifref == False:
			if isorientsys == False:
				infout = ReadOutFile("./temp_run/"+syst+".out", False, "cell_relax", isspin)
				infgeopt = ReadGeoptFile("./temp_run/"+syst+".geopt", infout)
			else:
				infout1 = ReadOutFile("./temp_run1/"+syst+".out", False, "cell_relax", isspin)
				infgeopt1 = ReadGeoptFile("./temp_run1/"+syst+".geopt", infout1)
				infout2 = ReadOutFile("./temp_run2/"+syst+".out", False, "cell_relax", isspin)
				infgeopt2 = ReadGeoptFile("./temp_run2/"+syst+".geopt", infout2)
				infout3 = ReadOutFile("./temp_run3/"+syst+".out", False, "cell_relax", isspin)
				infgeopt3 = ReadGeoptFile("./temp_run3/"+syst+".geopt", infout3)
		else:
			if isorientsys == False:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy/"+syst+".refout", False, "cell_relax", isspin)
					infgeopt = ReadGeoptFile("./high_accuracy/"+syst+".refgeopt", infout)
				else:
					infout = ReadOutFile("./standard/"+syst+".refout", False, "cell_relax", isspin)
					infgeopt = ReadGeoptFile("./standard/"+syst+".refgeopt", infout)
			else:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy_orientation1/"+syst+".refout", False, "cell_relax", isspin)
					infgeopt = ReadGeoptFile("./high_accuracy_orientation1/"+syst+".refgeopt", infout)
				else:
					infout = ReadOutFile("./standard_orientation1/"+syst+".refout", False, "cell_relax", isspin)
					infgeopt = ReadGeoptFile("./standard_orientation1/"+syst+".refgeopt", infout)

		if isorientsys == False or ifref == False:
			E = infout["E"]
			SCF_no = infout["SCF_no"]
			walltime = infout["walltime"]
			scfpos = infgeopt["scfpos"]
			cell = infgeopt["cell"]
			stress = []
			magnetization = infout["magnetization"]
			pressure=[]
			if infout["isPrintPres"] == True:
				pressure = infout["pressure"]
			if infout["isPrintStress"] == True:
				stress = infgeopt["stress"]
			no_atoms = infout["no_atoms"]
		else:
			E = [infout1["E"],infout2["E"],infout3["E"]]
			SCF_no = infout1["SCF_no"]
			walltime = infout1["walltime"]#[infout1["walltime"],infout2["walltime"],infout3["walltime"]]
			scfpos = infgeopt1["scfpos"]#[infgeopt1["scfpos"],infgeopt2["scfpos"],infgeopt3["scfpos"]]
			cell = infgeopt1["cell"]#[infgeopt1["cell"],infgeopt2["cell"],infgeopt3["cell"]]
			stress = []
			magnetization = infout1["magnetization"]#[infout1["magnetization"],infout2["magnetization"],infout3["magnetization"]]
			pressure=[]
			if infout1["isPrintPres"] == True:
				pressure = infout1["pressure"]#[infout1["pressure"],infout2["pressure"],infout3["pressure"]]
			if infout1["isPrintStress"] == True:
				stress = infgeopt1["stress"]#[infgeopt1["stress"],infgeopt2["stress"],infgeopt3["stress"]]
			no_atoms = infout1["no_atoms"]

		Info = {"Type": "relax_cell",
			"isspin": isspin,
			"ismemcheck": ismemch,
			# "ismemused": ismemused,
			"energy": E,
			"walltime": walltime,
			"cell": cell,
			"memlost": memlost,
			# "memused": memused,
			"magnetization": magnetization,
			"pressure": pressure,
			"no_atoms": no_atoms,
			"isorient": isorientsys,
			"tolerance": tolerance,
			"SCF_no": SCF_no}




		os.chdir(home_tests_directory)
		return(Info)

	elif ((singlept == False) and (Type == "relax_total")):
		#------------------------ Memecheck from valgrind ----------------------------#
		ismemch, memlost=Readvalgridout(isorientsys, memcheck, ifref, ifVHQ)
		
		#------------------------ Memecheck from valgrind ----------------------------#

		#------------------------ Memory from output.sparc ----------------------------#
		# ismemused,memused = ReadmemoutputFile(isorientsys, ismempbs, ifref, ifVHQ)
		
		#------------------------ Memory from output.sparc ----------------------------#
		if ifref == False:
			if isorientsys == False:
				infout = ReadOutFile("./temp_run/"+syst+".out", False, "full_relax", isspin)
				infgeopt = ReadGeoptFile("./temp_run/"+syst+".geopt", infout)
			else:
				infout1 = ReadOutFile("./temp_run1/"+syst+".out", False, "cell_relax", isspin)
				infgeopt1 = ReadGeoptFile("./temp_run1/"+syst+".geopt", infout1)
				infout2 = ReadOutFile("./temp_run2/"+syst+".out", False, "cell_relax", isspin)
				infgeopt2 = ReadGeoptFile("./temp_run2/"+syst+".geopt", infout2)
				infout3 = ReadOutFile("./temp_run3/"+syst+".out", False, "cell_relax", isspin)
				infgeopt3 = ReadGeoptFile("./temp_run3/"+syst+".geopt", infout3)

		else:
			if isorientsys == False:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy/"+syst+".refout", False, "full_relax", isspin)
					infgeopt = ReadGeoptFile("./high_accuracy/"+syst+".refgeopt", infout)
				else:
					infout = ReadOutFile("./standard/"+syst+".refout", False, "full_relax", isspin)
					infgeopt = ReadGeoptFile("./standard/"+syst+".refgeopt", infout)
			else:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy_orientation1/"+syst+".refout", False, "full_relax", isspin)
					infgeopt = ReadGeoptFile("./high_accuracy_orientation1/"+syst+".refgeopt", infout)
				else:
					infout = ReadOutFile("./standard_orientation1/"+syst+".refout", False, "full_relax", isspin)
					infgeopt = ReadGeoptFile("./standard_orientation1/"+syst+".refgeopt", infout)
		if isorientsys == False or ifref == True:
			E = infout["E"]
			SCF_no = infout["SCF_no"]
			walltime = infout["walltime"]
			scfpos = infgeopt["scfpos"]
			cell = infgeopt["cell"]
			stress = []
			force = []
			pressure = infout["pressure"]
			magnetization = infout["magnetization"]
			if infout["isPrintStress"] == True:
				stress = infgeopt["stress"]
			if infout["isPrintF"] == True:
				force = infgeopt["force"]
			no_atoms = infout["no_atoms"]
		else:
			E = [infout1["E"],infout2["E"],infout3["E"]]
			SCF_no = infout1["SCF_no"]
			walltime = infout1["walltime"]#[infout1["walltime"],infout2["walltime"],infout3["walltime"]]
			scfpos = infgeopt1["scfpos"]#[infgeopt1["scfpos"],infgeopt2["scfpos"],infgeopt3["scfpos"]]
			cell = infgeopt1["cell"]#[infgeopt1["cell"],infgeopt2["cell"],infgeopt3["cell"]]
			stress = []
			force = []
			pressure = infout1["pressure"]#[infout1["pressure"],infout2["pressure"],infout3["pressure"]]
			magnetization = infout["magnetization"]
			if infout1["isPrintStress"] == True:
				stress = infgeopt1["stress"]#[infgeopt1["stress"],infgeopt2["stress"],infgeopt3["stress"]]
			if infout1["isPrintF"] == True:
				force = infgeopt1["force"]#[infgeopt1["force"],infgeopt2["force"],infgeopt3["force"]]
			no_atoms = infout1["no_atoms"]

		Info = {"Type": "relax_total",
			"isspin": isspin,
			"ismemcheck": ismemch,
			# "ismemused": ismemused,
			"energy": E,
			"stress": stress,
			"walltime": walltime,
			"cell": cell,
			"memlost": memlost,
			# "memused": memused,
			"magnetization": magnetization,
			"pressure": pressure,
			"no_atoms": no_atoms,
			"scfpos": scfpos,
			"isorient": isorientsys,
			"tolerance": tolerance,
			"SCF_no": SCF_no}


		os.chdir(home_tests_directory)
		return(Info)

	elif ((singlept == False) and (Type == "MD")):
		#------------------------ Memecheck from valgrind ----------------------------#
		ismemch, memlost=Readvalgridout(isorientsys, memcheck, ifref, ifVHQ)
		
		#------------------------ Memecheck from valgrind ----------------------------#

		#------------------------ Memory from output.sparc ----------------------------#
		# ismemused,memused = ReadmemoutputFile(isorientsys, ismempbs, ifref, ifVHQ)
		
		#------------------------ Memory from output.sparc ----------------------------#
		if ifref == False:
			if isorientsys == False:
				infout = ReadOutFile("./temp_run/"+syst+".out", True, "None", isspin)
				infaimd = ReadAimdFile("./temp_run/"+syst+".aimd", infout)
			else:
				infout1 = ReadOutFile("./temp_run1/"+syst+".out", True, "None", isspin)
				infaimd1 = ReadAimdFile("./temp_run1/"+syst+".aimd", infout1)
				infout2 = ReadOutFile("./temp_run2/"+syst+".out", True, "None", isspin)
				infaimd2 = ReadAimdFile("./temp_run2/"+syst+".aimd", infout2)
				infout3 = ReadOutFile("./temp_run3/"+syst+".out", True, "None", isspin)
				infaimd3 = ReadAimdFile("./temp_run3/"+syst+".aimd", infout3)
		else:
			if isorientsys == False:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy/"+syst+".refout", True, "None", isspin)
					infaimd = ReadAimdFile("./high_accuracy/"+syst+".refaimd", infout)
				else:
					infout = ReadOutFile("./standard/"+syst+".refout", True, "None", isspin)
					infaimd = ReadAimdFile("./standard/"+syst+".refaimd", infout)
			else:
				if ifVHQ == True:
					infout = ReadOutFile("./high_accuracy_orientation1/"+syst+".refout", True, "None", isspin)
					infaimd = ReadAimdFile("./high_accuracy_orientation1/"+syst+".refaimd", infout)
				else:
					infout = ReadOutFile("./standard_orientation1/"+syst+".refout", True, "None", isspin)
					infaimd = ReadAimdFile("./standard_orientation1/"+syst+".refaimd", infout)
		if isorientsys == False or ifref == True:
			E = infout["E"]
			SCF_no = infout["SCF_no"]
			walltime = infout["walltime"]
			KEN = infaimd["KEN"]
			pressure=infout["pressure"]
			velocity = infaimd["velocity"]
			
			scfpos = []
			stress = []
			force = []
			ionic_stress = []
			magnetization = infout["magnetization"]
			if infout["isPrintStress"] == True:
				stress = infaimd["stress"]
				ionic_stress = infaimd["ionic_stress"]
			if infout["isPrintF"] == True:
				force = infaimd["force"]
			if infout["isPrintAtoms"] ==  True:
				scfpos = infaimd["scfpos"]
			no_atoms = infout["no_atoms"]
		else:
			E = [infout1["E"],infout2["E"],infout3["E"]]
			SCF_no = infout1["SCF_no"]
			walltime = infout1["walltime"]#[infout1["walltime"],infout2["walltime"],infout3["walltime"]]
			KEN = infaimd1["KEN"]#[infaimd1["KEN"],infaimd2["KEN"],infaimd3["KEN"]]
			pressure=infout1["pressure"]
			velocity = infaimd1["velocity"]
			scfpos = []
			stress = []
			force = []
			magnetization = infout1["magnetization"]#[infout1["magnetization"],infout2["magnetization"],infout3["magnetization"]]
			if infout1["isPrintStress"] == True:
				stress = infaimd1["stress"]#[infaimd1["stress"],infaimd2["stress"],infaimd3["stress"]]
				ionic_stress = infaimd1["ionic_stress"]
			if infout1["isPrintF"] == True:
				force = infaimd1["force"]#[infaimd1["force"],infaimd2["force"],infaimd3["force"]]
			if infout1["isPrintAtoms"] ==  True:
				scfpos = infaimd1["scfpos"]#[infaimd1["scfpos"],infaimd2["scfpos"],infaimd3["scfpos"]]
			no_atoms = infout1["no_atoms"]
		if True:
			Info = {"Type": "MD",
			"isspin": isspin,
			"ismemcheck": ismemch,
			"force": force,
			"stress": stress,
			"ionic_stress": ionic_stress,
			"velocity": velocity,
			# "ismemused": ismemused,
			"energy": E,
			"walltime": walltime,
			"scfpos": scfpos,
			"KEN": KEN,
			"memlost": memlost,
			# "memused": memused,
			"magnetization": magnetization,
			"no_atoms": no_atoms,
			"isorient": isorientsys,
			"tolerance": tolerance,
			"SCF_no": SCF_no}
			

		os.chdir(home_tests_directory)
		return(Info)
def WriteReport(data_info, systems, isparallel, ifVHQ, isorient):

	now = datetime.now() # current date and time

	year = now.strftime("%Y")
	month = now.strftime("%m")
	day = now.strftime("%d")
	time = now.strftime("%H:%M:%S")
	date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

	# E_tol = tols["E_tol"]
	# F_tol = tols["F_tol"]
	CELL_tol = tols["CELL_tol"]
	wall_tol = tols["wall_tol"]
	scfno_tol = tols["scfno_tol"]
	scfpos_tol = tols["scfpos_tol"]
	KEN_tol = tols["KEN_tol"]
	# stress_tol = tols["stress_tol"]
	spin_tol = tols["spin_tol"]
	memused_tol = tols["memused_tol"]
	########## Error calculation ######################
	Ener_error = []
	test_status=[]
	texttoprint=[]
	Error_message_global  = []
	Warning_message_global = []
	Wall_error = []

	for i in range(len(systems)):
		info_temp = data_info[i]
		info_run = info_temp['a']
		info_ref = info_temp['b']
		E_tol = info_run["tolerance"][0]
		F_tol = info_run["tolerance"][1]
		stress_tol = info_run["tolerance"][2]
		if len(info_temp) == 3:
			isabinit = True
			info_abinit = info_temp['c']
		else:
			isabinit = False
		if info_run["Type"]=="singlept":
			memlost=0
			text1=''
			err_memused = 0
			text2=''
			errspin = 0
			text3=''
			warning_message = ""
			no_atoms = info_run["no_atoms"]
			if info_run["isbandgap"] == True:
				err_bandgap = abs(info_run["bandgap"] - info_ref["bandgap"])
			else:
				err_bandgap = 0
			if info_run["isspin"] == True:
				if info_run["isorient"] == False:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					errspin = abs(magnetization_run - magnetization_ref)
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"

				else:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					errspin = max([abs(magnetization_run[0] - magnetization_ref),abs(magnetization_run[1] - magnetization_ref),abs(magnetization_run[2] - magnetization_ref)])
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"

			if info_run["ismemcheck"]==True:
				if info_run["isorient"] == False:
					memlost = info_run["memlost"]
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
				else:
					memlost = info_run["memlost"]
					memlost = max(memlost)
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
			if info_run["isorient"] == False:
				E_sys_err = abs(info_run["energy"][0]-info_ref["energy"][0])
			else:
				E_sys_err = max([abs(info_run["energy"][0][0]-info_ref["energy"][0]),abs(info_run["energy"][1][0]-info_ref["energy"][0]),abs(info_run["energy"][2][0]-info_ref["energy"][0])])
			Ener_error.append(E_sys_err)
			F_ref = info_ref["force"]
			F_run = info_run["force"]

			SCF_no_ref = info_ref["SCF_no"]
			SCF_no_run = info_run["SCF_no"]

			Error_SCF_no = SCF_no_run - SCF_no_ref;
			if Error_SCF_no < 0:
				warning_message=warning_message+ " Number of SCF iterations are smaller (" +str(Error_SCF_no)+"/"+str(SCF_no_ref)+") than the reference"
			elif Error_SCF_no > 0:
				warning_message=warning_message+ " Number of SCF iterations are larger (" +str(Error_SCF_no)+"/"+str(SCF_no_ref)+") than the reference"
				


			force_error=[]
			stress_error=[]
			for j in range(len(F_ref)):
				force_error.append([abs(F_ref[j][0]-F_run[j][0]),abs(F_ref[j][1]-F_run[j][1]),abs(F_ref[j][2]-F_run[j][2])])
			if len(sum(force_error,[]))>0:
				force_error = max(sum(force_error,[]))
			else:
				force_error = 0


			stress_run = info_run["stress"]
			stress_ref = info_ref["stress"]


			for j in range(len(stress_run)):
				temp =[]
				for jj in range(len(stress_run[j])):
					if abs(stress_ref[j][jj]) > 0.01:
						temp.append((abs(stress_ref[j][jj]-stress_run[j][jj]))*100/abs(stress_ref[j][jj]))
					else:
						temp.append(0)
				stress_error.append(temp)
				#stress_error.append([(abs(stress_ref[j][0]-stress_run[j][0]))*100/abs(stress_ref[j][0]),(abs(stress_ref[j][1]-stress_run[j][1]))*100/abs(stress_ref[j][1]),(abs(stress_ref[j][2]-stress_run[j][2]))*100/abs(stress_ref[j][2])])
			if len(sum(stress_error,[])) >0:
				stress_error = max(sum(stress_error,[]))
			else:
				stress_error = 0

			walltime_error = (info_run["walltime"][0]-info_ref["walltime"][0])/info_ref["walltime"][0] *100

			if isparallel == False or info_run["ismemcheck"] == True:
				walltime_error = 0

			#scfno_error = abs(info_run["scfno"][0]-info_ref["scfno"][0])
			Wall_error.append(walltime_error)
			if walltime_error < 0:
				warning_message=warning_message+" Walltime is smaller than the reference"
			if walltime_error > wall_tol:
				warning_message=warning_message+" Walltime exceeded by "+ str(walltime_error)+"%"

			text="System name: "+systems[i]+"\n"+"Single Point Calculation \nEnergy error (Ha/atom): "+ str(E_sys_err)+"\nForce error (Ha/Bohr): "+'{0:1.2e}'.format(force_error)+"\n"
			#for j in range(no_atoms):
				#text = text+'{0:1.2e}'.format(force_error[j][0])+" "+'{0:1.2e}'.format(force_error[j][1])+" "+'{0:1.2e}'.format(force_error[j][2])+"\n"
			text = text+"Stress (%) error: "+ '{0:1.2e}'.format(stress_error)+"\n"
			text = text+"Number of SCF iteration error: "+ str(Error_SCF_no)+"\n"
			#for j in range(3):
				#text = text+'{0:1.2e}'.format(stress_error[j][0])+" "+'{0:1.2e}'.format(stress_error[j][1])+" "+'{0:1.2e}'.format(stress_error[j][2])+"\n"
			if isparallel == True and info_run["ismemcheck"] == False:
				text = text+"walltime error (%): "+'{0:1.2e}'.format(walltime_error)+"\n"
			if info_run["isbandgap"] == True:
				text = text+"Bandgap error (Ha): "+'{0:1.2e}'.format(err_bandgap)+"\n"
			#text = text+"Error in number of SCF iterations for convergence: "+str(scfno_error)+"\n"

			text=text+text1+text2+text3
			Failure_text=""
			if (err_bandgap <= 0.001 and Error_SCF_no <=  scfno_tol and errspin <= spin_tol  and E_sys_err <= E_tol and force_error <= F_tol and stress_error <= stress_tol  and memlost == 0):
				test_status.append("passed")
				text="Test Result: Passed \n"+text
			else:
				Failure_text = Failure_text+"Test for this system "+" failed in: "
				if (errspin > spin_tol):
					Failure_text =  Failure_text + "Spin polarization, "
				if (E_sys_err > E_tol):
					Failure_text =  Failure_text + "Energy, "
				if (force_error > F_tol):
					Failure_text =  Failure_text + "Force, "
				if (stress_error > stress_tol):
					Failure_text =  Failure_text + "Stress, "
				if (memlost > 0):
					Failure_text =  Failure_text + "Memory leak, "
				if (Error_SCF_no >  scfno_tol):
					Failure_text =  Failure_text + "Number of SCF iterations, "
				Error_message_global.append(Failure_text)

				test_status.append("failed")
				text="Test Result: Failed \n"+text
				#print(len(texttoprint))
			if walltime_error > wall_tol:
				text = text + "Warning: walltime exceeded"
			if err_memused > memused_tol:
				text = text + "Warning: Memory used exceeded"

			texttoprint.append(text)
			Warning_message_global.append(warning_message)


		elif info_run["Type"]=="relax_atom":
			memlost=0
			text1=''
			err_memused = 0
			text2=''
			errspin = 0
			warning_message = ""
			text3=''
			E_run = info_run["energy"]
			no_atoms = info_run["no_atoms"]
			relax_steps = len(E_run)
			if info_run["isspin"] == True:
				if info_run["isorient"] == False:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					errspin = abs(magnetization_run - magnetization_ref)
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"

				else:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					errspin = max([abs(magnetization_run[0] - magnetization_ref),abs(magnetization_run[1] - magnetization_ref),abs(magnetization_run[2] - magnetization_ref)])
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"

			
			if info_run["ismemcheck"]==True:
				if info_run["isorient"] == False:
					memlost = info_run["memlost"]
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
				else:
					memlost = info_run["memlost"]
					memlost = max(memlost)
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
			if len(info_run["energy"]) != len(info_ref["energy"]):
				# test_status.append("failed")
				text = "System name: "+systems[i]+"\n"+"Warning: different relaxation iterations for the convergence!"
			E_ref = info_ref["energy"]
			E_run = info_run["energy"]

			SCF_no_ref = info_ref["SCF_no"]
			SCF_no_run = info_run["SCF_no"]
			Error_SCF_no=0
			if len(SCF_no_ref)!=len(SCF_no_run):
				warning_message = "Number of electronic steps for atom position relaxation for system "+systems[i]+" is different from the reference"
			else:
				Error_SCF_no = []
				for scfno in range(len(SCF_no_run)):
					Error_SCF_no.append(SCF_no_run[scfno] - SCF_no_ref[scfno])
				Error_SCF_no1=Error_SCF_no
				Error_SCF_no = max(Error_SCF_no)
				if Error_SCF_no < 0:
					warning_message=warning_message+ " Number of SCF iterations are smaller (" +str(Error_SCF_no)+"/"+str(SCF_no_ref[Error_SCF_no1.index(Error_SCF_no)])+") than the reference"
				elif Error_SCF_no > 0:
					warning_message=warning_message+ " Number of SCF iterations are larger (" +str(Error_SCF_no)+"/"+str(SCF_no_ref[Error_SCF_no1.index(Error_SCF_no)])+") than the reference"
				

			if info_run["isorient"] == False:
				E_err=abs(E_ref[-1]-E_run[-1])
			else:
				E_err=max([abs(E_ref[-1]-E_run[0][-1]),abs(E_ref[-1]-E_run[1][-1]),abs(E_ref[-1]-E_run[2][-1])])
			E_sys_err = E_err
			Ener_error.append(E_sys_err)
			scfpos_run = info_run["scfpos"]
			scfpos_ref = info_ref["scfpos"]
			
			#relax_steps = len(F_run)
			#F_error = []
			#F_error_relax=[]
			temp_scfpos=[]

			#temp= []
			for k in range(len(scfpos_run[0])):
				temp_scfpos.append([abs(scfpos_run[-1][k][0]-scfpos_ref[-1][k][0]), abs(scfpos_run[-1][k][1]-scfpos_ref[-1][k][1]), abs(scfpos_run[-1][k][2]-scfpos_ref[-1][k][2])])

			temp_scfpos = sum(temp_scfpos,[])
			scfpos_err = max(temp_scfpos)


			# scfno_run = info_run["scfno"]
			# scfno_ref = info_ref["scfno"]
			# scfno_error = []
			# for j in range(len(scfno_run)):
			# 	scfno_error.append(abs(scfno_run[j]-scfno_ref[j]))
			# scfno_error = scfno_error[-1]

			walltime_error = (info_run["walltime"][0]-info_ref["walltime"][0])/info_ref["walltime"][0] *100

			if isparallel == False or info_run["ismemcheck"] == True:
				walltime_error = 0
			Wall_error.append(walltime_error)
			if walltime_error < 0:
				warning_message=warning_message+" Walltime is smaller than the reference"
			if walltime_error > wall_tol:
				warning_message=warning_message+" Walltime exceeded by "+ str(walltime_error)+"%"

			text = "System name: "+systems[i]+"\n"+"Atom position relaxation\n"
			text = text+ "Error in energy in the final relaxed position (Ha/atom): "+ '{0:1.2e}'.format(E_err) +"  \n"
			text = text+ "Error in the final relaxed atom position (Bohr): "+ '{0:1.2e}'.format(scfpos_err) +"  \n"
			text = text+"Number of SCF iteration error: "+ str(Error_SCF_no)+"\n"
			if isparallel == True and info_run["ismemcheck"] == False:
				text = text+"walltime error (%): "+'{0:1.2e}'.format(walltime_error)+"\n"

			text = text+text1+text2+text3
			Failure_text=""
			if (Error_SCF_no <=  scfno_tol and errspin <= spin_tol  and E_sys_err <= E_tol and scfpos_err <= scfpos_tol  and memlost == 0):
				test_status.append("passed")
				text="Test Result: Passed \n"+text
			else:
				Failure_text = Failure_text+"Test for this system "+" failed in: "
				if (errspin > spin_tol):
					Failure_text =  Failure_text + "Spin polarization, "
				if (E_sys_err > E_tol):
					Failure_text =  Failure_text + "Energy, "
				if (scfpos_err > scfpos_tol):
					Failure_text =  Failure_text + "Relaxed position, "
				if (Error_SCF_no >  scfno_tol):
					Failure_text =  Failure_text + "Number of SCF iterations, "
				if (memlost > 0):
					Failure_text =  Failure_text + "Memory leak, "
				Error_message_global.append(Failure_text)

				test_status.append("failed")
				text="Test Result: Failed\n"+text
			if walltime_error > wall_tol:
				text = text + "Warning: walltime exceeded"
			texttoprint.append(text)
			Warning_message_global.append(warning_message)


		elif info_run["Type"]=="relax_cell":
			memlost=0
			text1=''
			err_memused = 0
			text2=''
			errspin = 0
			warning_message = ""
			text3=''
			if info_run["isspin"] == True:
				if info_run["isorient"] == False:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					errspin = abs(magnetization_run - magnetization_ref)
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"

				else:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					errspin = max([abs(magnetization_run[0] - magnetization_ref),abs(magnetization_run[1] - magnetization_ref),abs(magnetization_run[2] - magnetization_ref)])
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"

			
			if info_run["ismemcheck"]==True:
				if info_run["isorient"] == False:
					memlost = info_run["memlost"]
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
				else:
					memlost = info_run["memlost"]
					memlost = max(memlost)
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
			if len(info_run["energy"]) != len(info_ref["energy"]):
				# test_status.append("failed")
				text = "System name: "+systems[i]+"\n"+"different relaxation iterations for the convergence hence failed!"
			E_ref = info_ref["energy"]
			E_run = info_run["energy"]
			SCF_no_ref = info_ref["SCF_no"]
			SCF_no_run = info_run["SCF_no"]
			Error_SCF_no=0
			if len(SCF_no_ref)!=len(SCF_no_run):
				warning_message = warning_message+"Number of electronic steps for atom position relaxation for system "+systems[i]+" is different from the reference"
			else:
				Error_SCF_no = []
				for scfno in range(len(SCF_no_run)):
					Error_SCF_no.append(SCF_no_run[scfno] - SCF_no_ref[scfno])	
				Error_SCF_no1=Error_SCF_no
				Error_SCF_no = max(Error_SCF_no)
				if Error_SCF_no < 0:
					warning_message=warning_message+ " Number of SCF iterations are smaller (" +str(Error_SCF_no)+"/"+str(SCF_no_ref[Error_SCF_no1.index(Error_SCF_no)])+") than the reference"
				elif Error_SCF_no > 0:
					warning_message=warning_message+ " Number of SCF iterations are larger (" +str(Error_SCF_no)+"/"+str(SCF_no_ref[Error_SCF_no1.index(Error_SCF_no)])+") than the reference"
				
			if Error_SCF_no < 0:
				warning_message=warning_message+" Number of SCF iterations are smaller than the reference"

			# E_err_relax=[]
			# for j in range(len(info_run["energy"])):
			# 	E_err_relax.append(abs(E_ref[j]-E_run[j]))
			if info_run["isorient"] == False:
				E_sys_err = abs(E_ref[-1]-E_run[-1])
			else:
				E_sys_err = max([abs(E_ref[-1]-E_run[0][-1]),abs(E_ref[-1]-E_run[1][-1]),abs(E_ref[-1]-E_run[2][-1])])
			Ener_error.append(E_sys_err)

			cell_run = info_run["cell"]
			cell_ref = info_ref["cell"]
			cell_error= []
			for k in range(len(cell_run[0])):
				cell_error.append(abs(cell_run[-1][k]-cell_ref[-1][k]))
			cell_error=max(cell_error)
			

			walltime_error = (info_run["walltime"][0]-info_ref["walltime"][0])/info_ref["walltime"][0] *100
			if isparallel == False or info_run["ismemcheck"] == True:
				walltime_error = 0
			Wall_error.append(walltime_error)
			if walltime_error < 0:
				warning_message=warning_message+" Walltime is smaller than the reference"
			if walltime_error > wall_tol:
				warning_message=warning_message+" Walltime exceeded by "+ str(walltime_error)+"%"


			text = "System name: "+systems[i]+"\n"+"CELL relaxation\n"#+"Relaxation step    "+"Energy Error (Ha/atom)    "+"Stress Error (GPa)    "+"Error in cell dimesions (Bohr)\n"
			text = text + "Error in energy in the final relaxed position (Ha/atom): "+ '{0:1.2e}'.format(E_sys_err) +"  \n"
			text = text+ "Error in the final relaxed Cell (Bohr): "+ '{0:1.2e}'.format(cell_error) +"  \n"
			text = text+"Number of SCF iteration error: "+ str(Error_SCF_no)+"\n"
			if isparallel == True and info_run["ismemcheck"] == False :
				text = text+"walltime error (%): "+'{0:1.2e}'.format(walltime_error)+"\n"
			#text = text+"Error in stress "
			#text = text+"Error in number of SCF iterations for convergence: "+'{0:1.2e}'.format(cell_error)+"\n"
			text = text+text1+text2+text3
			Failure_text=""
			if (Error_SCF_no <=  scfno_tol and errspin <= spin_tol  and E_sys_err <= E_tol  and cell_error <= CELL_tol  and  memlost == 0):
				test_status.append("passed")
				text="Test Result: Passed \n"+text
			else:
				Failure_text = Failure_text+"Test for this system "+" failed in: "
				if (errspin > spin_tol):
					Failure_text =  Failure_text + "Spin polarization, "
				if (E_sys_err > E_tol):
					Failure_text =  Failure_text + "Energy, "
				if (cell_error > CELL_tol):
					Failure_text =  Failure_text + "Relaxed Cell length, "
				if (Error_SCF_no >  scfno_tol):
					Failure_text =  Failure_text + "Number of SCF iterations, "
				if (memlost > 0):
					Failure_text =  Failure_text + "Memory leak, "
				Error_message_global.append(Failure_text)

				test_status.append("failed")
				text="Test Result: Failed \n"+text
			if walltime_error > wall_tol:
				text = text + "Warning: walltime exceeded"
			texttoprint.append(text)
			Warning_message_global.append(warning_message)


		elif info_run["Type"]=="relax_total":
			memlost=0
			text1=''
			err_memused = 0
			text2=''
			errspin = 0
			warning_message = ""
			text3=''
			E_run = info_run["energy"]
			no_atoms = info_run["no_atoms"]
			relax_steps = len(E_run)
			if info_run["isspin"] == True:
				if info_run["isorient"] == False:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					errspin = abs(magnetization_run - magnetization_ref)
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"
				else:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					errspin = max([abs(magnetization_run[0] - magnetization_ref),abs(magnetization_run[1] - magnetization_ref),abs(magnetization_run[2] - magnetization_ref)])
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"
			
			if info_run["ismemcheck"]==True:
				if info_run["isorient"] == False:
					memlost = info_run["memlost"]
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
				else:
					memlost = info_run["memlost"]
					memlost = max(memlost)
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
			if len(info_run["energy"]) != len(info_ref["energy"]):
				# test_status.append("failed")
				text = "System name: "+systems[i]+"\n"+"different relaxation iterations for the convergence hence failed!"
			E_ref = info_ref["energy"]
			E_run = info_run["energy"]
			if info_run["isorient"] == False:
				E_err= abs(E_ref[-1]-E_run[-1])
			else:
				E_err= max([abs(E_ref[-1]-E_run[0][-1]),abs(E_ref[-1]-E_run[1][-1]),abs(E_ref[-1]-E_run[2][-1])])

			E_sys_err = E_err
			Ener_error.append(E_sys_err)
			SCF_no_ref = info_ref["SCF_no"]
			SCF_no_run = info_run["SCF_no"]

			if len(SCF_no_ref)!=len(SCF_no_run):
				warning_message = warning_message+"Number of electronic steps for atom position relaxation for system "+systems[i]+" is different from the reference"
			else:
				Error_SCF_no = []
				for scfno in range(len(SCF_no_run)):
					Error_SCF_no.append(SCF_no_run[scfno] - SCF_no_ref[scfno])
				Error_SCF_no1=Error_SCF_no
				Error_SCF_no = max(Error_SCF_no)
				if Error_SCF_no < 0:
					warning_message=warning_message+ " Number of SCF iterations are smaller (" +str(Error_SCF_no)+"/"+str(SCF_no_ref[Error_SCF_no1.index(Error_SCF_no)])+") than the reference"
				elif Error_SCF_no > 0:
					warning_message=warning_message+ " Number of SCF iterations are larger (" +str(Error_SCF_no)+"/"+str(SCF_no_ref[Error_SCF_no1.index(Error_SCF_no)])+") than the reference"
				
			scfpos_run = info_run["scfpos"]
			scfpos_ref = info_ref["scfpos"]
			scfpos_err = []


			#for j in range(relax_steps):
			for k in range(len(scfpos_run[0])):
				scfpos_err.append([abs(scfpos_run[-1][k][0]-scfpos_ref[-1][k][0]), abs(scfpos_run[-1][k][1]-scfpos_ref[-1][k][1]), abs(scfpos_run[-1][k][2]-scfpos_ref[-1][k][2])])
			scfpos_err=max(sum(scfpos_err,[]))

			cell_run = info_run["cell"]
			cell_ref = info_ref["cell"]
			cell_error = []

			for k in range(len(cell_run[0])):
				cell_error.append(abs(cell_run[-1][k]-cell_ref[-1][k]))

			cell_error =max(cell_error)


			walltime_error = (info_run["walltime"][0]-info_ref["walltime"][0])/info_ref["walltime"][0] *100
			if isparallel == False or info_run["ismemcheck"] == True:
				walltime_error = 0
			Wall_error.append(walltime_error)

			if walltime_error < 0:
				warning_message=warning_message+" Walltime is smaller than the reference"
			if walltime_error > wall_tol:
				warning_message=warning_message+" Walltime exceeded by "+ str(walltime_error)+"%"

			text = "System name: "+systems[i]+"\n"+"Total relaxation\n"
			text = text+"Error in energy in the final relaxed structure (Ha/atom): "+'{0:1.2e}'.format(E_err)+"\n"
			text = text+ "Error in the final relaxed Cell (Bohr): "+ '{0:1.2e}'.format(cell_error) +"  \n"
			text = text+ "Error in the final relaxed atom position (Bohr): "+ '{0:1.2e}'.format(scfpos_err) +"  \n"
			text = text+"Number of SCF iteration) error: "+ str(Error_SCF_no)+"\n"

			if isparallel == True and info_run["ismemcheck"] == False:
				text = text+"walltime error (%): "+'{0:1.2e}'.format(walltime_error)+"\n"
			#text = text+"Error in number of SCF iterations for convergence: "+str(scfno_error)+"\n"
			text = text+text1+text2+text3
			Failure_text = ""
			if (Error_SCF_no <=  scfno_tol and errspin <= spin_tol  and E_err <= E_tol and cell_error <= CELL_tol  and scfpos_err <= scfpos_tol and memlost == 0):
				test_status.append("passed")
				text="Test Result: Passed \n"+text
			else:
				Failure_text = Failure_text+"Test for this system "+" failed in: "
				if (errspin > spin_tol):
					Failure_text =  Failure_text + "Spin polarization, "
				if (E_sys_err > E_tol):
					Failure_text =  Failure_text + "Energy, "
				if (cell_error > CELL_tol):
					Failure_text =  Failure_text + "Relaxed Cell length, "
				if (scfpos_err > scfpos_tol):
					Failure_text =  Failure_text + "Relaxed position, "
				if (Error_SCF_no >  scfno_tol):
					Failure_text =  Failure_text + "Number of SCF iterations, "
				if (memlost > 0):
					Failure_text =  Failure_text + "Memory leak, "
				Error_message_global.append(Failure_text)

				test_status.append("failed")
				text="Test Result: Failed \n"+text
			if walltime_error > wall_tol:
				text = text + "Warning: walltime exceeded"
			texttoprint.append(text)
			Warning_message_global.append(warning_message)


		elif info_run["Type"]=="MD":
			memlost=0
			warning_message = ""
			text1=''
			err_memused = 0
			text2=''
			errspin = 0
			text3=''
			no_atoms = info_run["no_atoms"]
			
			if info_run["isspin"] == True:
				if info_run["isorient"] == False:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					if (type(magnetization_ref) == list):
						errspin = 0.0;
						for mm in range(len(magnetization_run)):
							if (abs(magnetization_run[mm] - magnetization_ref[mm])>errspin):
								errspin = abs(magnetization_run[mm] - magnetization_ref[mm])
					else:
						errspin = abs(magnetization_run - magnetization_ref)
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"

				else:
					magnetization_ref = info_ref["magnetization"]
					magnetization_run = info_run["magnetization"]
					errspin = max([abs(magnetization_run[0] - magnetization_ref),abs(magnetization_run[1] - magnetization_ref),abs(magnetization_run[2] - magnetization_ref)])
					text3 = "Spin polarized calculation: \n"+"Error in net magnetization: " + str(errspin)+"\n"

			if info_run["ismemcheck"]==True:
				if info_run["isorient"] == False:
					memlost = info_run["memlost"]
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
				else:
					memlost = info_run["memlost"]
					memlost = max(memlost)
					text1="Memory leak check valgrind: "+"\n"+"Total memory lost: "+str(memlost)+" Bytes \n"
			if len(info_run["energy"]) != len(info_ref["energy"]):
				test_status.append("failed")
				text = "System name: "+systems[i]+"\n"+"different number of MD iterations from the hence failed!"
			else:
				E_ref = info_ref["energy"]
				E_run = info_run["energy"]

				SCF_no_ref = info_ref["SCF_no"]
				SCF_no_run = info_run["SCF_no"]
				Error_SCF_no = []
				for scfno in range(len(SCF_no_run)):
					Error_SCF_no.append(SCF_no_run[scfno] - SCF_no_ref[scfno])
				Error_SCF_no1=Error_SCF_no
				Error_SCF_no = max(Error_SCF_no)
				if Error_SCF_no < 0:
					warning_message=warning_message+ " Number of SCF iterations are smaller (" +str(Error_SCF_no)+"/"+str(SCF_no_ref[Error_SCF_no1.index(Error_SCF_no)])+") than the reference"
				elif Error_SCF_no > 0:
					warning_message=warning_message+ " Number of SCF iterations are larger (" +str(Error_SCF_no)+"/"+str(SCF_no_ref[Error_SCF_no1.index(Error_SCF_no)])+") than the reference"

				E_err_relax=[]
				

				for j in range(len(info_run["energy"])):
					if info_run["isorient"] == False:
						E_err_relax.append(abs(E_ref[j]-E_run[j]))
					else:
						E_err_relax.append(max([abs(E_ref[j]-E_run[0][j]),abs(E_ref[j]-E_run[1][j]),abs(E_ref[j]-E_run[2][j])]))

				E_sys_err = max(E_err_relax)
				Ener_error.append(E_sys_err)
				
				ken_ref = info_ref["KEN"]
				ken_run = info_run["KEN"]


				MD_iter = len(ken_run)

				ken_error = []
				
				for j in range(MD_iter):
					ken_error.append(abs(ken_ref[j]-ken_run[j]))

				max_KENerror = max(ken_error)

				velocity_run = info_run["velocity"]
				velocity_ref = info_ref["velocity"]

				velocity_error = []
				velocity_error_relax=[]
				#no_atoms = len(F_run[0])
				if len(sum(velocity_run,[])) > 0:
					for j in range(MD_iter):
						temp= []
						for k in range(len(velocity_run[0])):
							temp.append([abs(velocity_run[j][k][0]-velocity_ref[j][k][0]), abs(velocity_run[j][k][1]-velocity_ref[j][k][1]), abs(velocity_run[j][k][2]-velocity_ref[j][k][2])])
						velocity_error.append(temp)

					for j in range(MD_iter):
						temp = velocity_error[j]
						temp = sum(temp,[])
						velocity_error_relax.append(max(temp))
					maxvelocity_err = max(velocity_error_relax)
				else:
					velocity_error_relax = [0 for md in range(MD_iter)] 
					maxvelocity_err = 0
					# F_error_relax_abinit = [0 for md in range(MD_iter)] 


				F_run = info_run["force"]
				F_ref = info_ref["force"]

				F_error = []
				F_error_relax=[]
				#no_atoms = len(F_run[0])
				if len(sum(F_run,[])) > 0:
					for j in range(MD_iter):
						temp= []
						for k in range(len(F_run[0])):
							temp.append([abs(F_run[j][k][0]-F_ref[j][k][0]), abs(F_run[j][k][1]-F_ref[j][k][1]), abs(F_run[j][k][2]-F_ref[j][k][2])])
						F_error.append(temp)

					for j in range(MD_iter):
						temp = F_error[j]
						temp = sum(temp,[])
						F_error_relax.append(max(temp))
					maxF_err = max(F_error_relax)

				else:
					F_error_relax = [0 for md in range(MD_iter)] 
					maxF_err = 0




				ionic_stress_run = info_run["ionic_stress"]
				ionic_stress_ref = info_ref["ionic_stress"]
				ionic_stress_error = []
				ionic_stress_error_relax=[]
				if len(sum(ionic_stress_run,[]))>0:
					for j in range(MD_iter):
						temp= []
						for k in range(len(ionic_stress_run[0])):
							temp1 =[]
							for jj in range(len(ionic_stress_run[0][k])):
								if abs(ionic_stress_run[j][k][jj]) > 0.01:
									temp1.append(100*(abs(ionic_stress_run[j][k][jj]-ionic_stress_ref[j][k][jj]))/abs(ionic_stress_ref[j][k][jj]))
								else:
									temp1.append(0)
							temp.append(temp1)
							#temp.append([100*(abs(stress_run[j][k][0]-stress_ref[j][k][0]))/abs(stress_ref[j][k][0]), 100*(abs(stress_run[j][k][1]-stress_ref[j][k][1]))/abs(stress_ref[j][k][1]), 100*(abs(stress_run[j][k][2]-stress_ref[j][k][2]))/abs(stress_ref[j][k][2])])
						ionic_stress_error.append(temp)
					for j in range(MD_iter):
						temp = ionic_stress_error[j]
						temp = sum(temp,[])
						ionic_stress_error_relax.append(max(temp))
					max_ionic_stress_error = max(ionic_stress_error_relax)
				else:
					ionic_stress_error_relax = [0 for md in range(MD_iter)] 
					# stress_error_relax_abinit = [0 for md in range(MD_iter)]
					max_ionic_stress_error= 0 


				stress_run = info_run["stress"]
				stress_ref = info_ref["stress"]
				stress_error = []
				stress_error_relax=[]
				if len(sum(stress_run,[]))>0:
					for j in range(MD_iter):
						temp= []
						for k in range(len(stress_run[0])):
							temp1 =[]
							for jj in range(len(stress_run[0][k])):
								if abs(stress_run[j][k][jj]) > 0.01:
									temp1.append(100*(abs(stress_run[j][k][jj]-stress_ref[j][k][jj]))/abs(stress_ref[j][k][jj]))
								else:
									temp1.append(0)
							temp.append(temp1)
							#temp.append([100*(abs(stress_run[j][k][0]-stress_ref[j][k][0]))/abs(stress_ref[j][k][0]), 100*(abs(stress_run[j][k][1]-stress_ref[j][k][1]))/abs(stress_ref[j][k][1]), 100*(abs(stress_run[j][k][2]-stress_ref[j][k][2]))/abs(stress_ref[j][k][2])])
						stress_error.append(temp)
					for j in range(MD_iter):
						temp = stress_error[j]
						temp = sum(temp,[])
						stress_error_relax.append(max(temp))
					max_stress_error = max(stress_error_relax)
				else:
					stress_error_relax = [0 for md in range(MD_iter)] 
					max_stress_error= 0 


				walltime_error = (info_run["walltime"][0]-info_ref["walltime"][0])/info_ref["walltime"][0] *100
				if isparallel == False or info_run["ismemcheck"] == True:
					walltime_error = 0
				Wall_error.append(walltime_error)
				if walltime_error < 0:
					warning_message=warning_message+" Walltime is smaller than the reference"
				if walltime_error > wall_tol:
					warning_message=warning_message+" Walltime exceeded by "+ str(walltime_error)+"%"

				text = "System name: "+systems[i]+"\n"+"MD Simulation\n"+"MD step    "+"Energy Error (Ha/atom)   "+"Ionic KE error (Ha/atom)     Force Error (Ha/Bohr)      Stress error (%)      Ionic Stress error (%)      velocity error (A.U.)\n"
				
				for j in range(MD_iter):
					text = text+str(j)+"   	     "+'{0:1.2e}'.format(E_err_relax[j])+"   			     "+'{0:1.2e}'.format(ken_error[j])+ "   			     " + '{0:1.2e}'.format(F_error_relax[j])+ "	   	 		   "+'{0:1.2e}'.format(stress_error_relax[j])+"	   	 		   "+'{0:1.2e}'.format(ionic_stress_error_relax[j])+"	   	 		   "+'{0:1.2e}'.format(velocity_error_relax[j])+"\n"
				text = text+"Number of SCF iteration) error: "+ str(Error_SCF_no)+"\n"
				if isparallel == True and info_run["ismemcheck"] == False:
					text = text+"walltime error (%): "+str(walltime_error)+"\n"
				#text = text+"Error in number of SCF iterations for convergence: "+str(scfno_error)+"\n"
				text = text+text1+text2+text3
				Failure_text=""
				if (Error_SCF_no <=  scfno_tol and errspin <= spin_tol  and E_sys_err <= E_tol   and max_KENerror <= KEN_tol and memlost == 0 and maxF_err <= F_tol and maxvelocity_err <= F_tol and max_ionic_stress_error <= stress_tol and max_stress_error <= stress_tol):
					test_status.append("passed")
					text="Test Result: Passed \n"+text
				else:
					Failure_text = Failure_text+"Test for this system "+" failed in: "
					if (errspin > spin_tol):
						Failure_text =  Failure_text + "Spin polarization, "
					if (E_sys_err > E_tol):
						Failure_text =  Failure_text + "Energy, "
					if (max_KENerror > KEN_tol):
						Failure_text =  Failure_text + "Ionic KE, "
					if (maxF_err > F_tol):
						Failure_text =  Failure_text + "Force, "
					if (Error_SCF_no >  scfno_tol):
						Failure_text =  Failure_text + "Number of SCF iterations, "
					if (memlost > 0):
						Failure_text =  Failure_text + "Memory leak, "
					Error_message_global.append(Failure_text)
					test_status.append("failed")
					text="Test Result: Failed \n"+text
				if walltime_error > wall_tol:
					text = text + "Warning: walltime exceeded"
				texttoprint.append(text)
				Warning_message_global.append(warning_message)



	passtests = 0;
	failtests = 0;

	for pp in range(len(test_status)):
		if test_status[pp]=="passed":
			passtests=passtests+1
		else:
			failtests=failtests+1
	########## End Error calculation ######################
  ################### Printing #############################################################
	f_report = open("Report.txt",'w')
	f_report.write("*************************************************************************** \n")
	f_report.write("*                   TEST REPORT (Version 11 August 2024)                    *\n*                      Date:  "+date_time+"                        * \n")
	f_report.write("*************************************************************************** \n")
	f_report.write("Tests Passed: "+str(passtests)+"/"+str(passtests+failtests)+"\n")
	f_report.write("Tests Failed: "+str(failtests)+"/"+str(passtests+failtests)+"\n")
	f_report.write("Average error in energy (Ha/atom): "+str(sum(Ener_error)/len(Ener_error))+"\n")
	f_report.write("*************************************************************************** \n")
	f_report.write("*************************************************************************** \n")		
	f_report.write("                    Details for the Passed systems               \n")
	#f_report.write("*************************************************************************** \n")
	for ii in range(len(systems)):
		if test_status[ii] == "passed":
			f_report.write("-------------------------- \n")
			f_report.write(texttoprint[ii])
			f_report.write("-------------------------- \n")
			f_report.write("\n")
	#f_report.write("*************************************************************************** \n")		
	f_report.write("                    End for the Passed systems               \n")
	f_report.write("*************************************************************************** \n")

	f_report.write("\n")
	f_report.write("\n")

	f_report.write("*************************************************************************** \n")
	f_report.write("                    Details for the Failed systems               \n")
	#f_report.write("*************************************************************************** \n")
	for ii in range(len(systems)):
		if test_status[ii] == "failed":
			f_report.write("-------------------------- \n")
			f_report.write(texttoprint[ii])
			f_report.write("-------------------------- \n")
			f_report.write("\n")
	#f_report.write("*************************************************************************** \n")		
	f_report.write("                    End for the Failed systems               \n")
	f_report.write("*************************************************************************** \n")
	f_report.close()
	return(test_status, Warning_message_global, Error_message_global)

# Main python file for the testing framework
# written by Shashikant Kumar, PhD

#############################################################################################################################################################################
#############################################################################################################################################################################

if __name__ == '__main__':
	args = sys.argv[1:]
	# finding systems and corresponding tags
	isparallel = True
	ismempbs =False
	ifVHQ = False
	isAuto = False
	is_valgrind_all = False
	is_update_reference = False
	temp_result =  False
	no_concurrency=6 # number of jobs running concurrently on github server

	systemstags = findsystems(['memcheck'])
	systems_valgrind = systemstags[0]
	tags_sys_valgrind = systemstags[1]
	tols_sys_valgrind = systemstags[2]
	dirs_sys_valgrind = systemstags[3]

	systemstags1 = findsystems(['ddbp'])
	systems_dddp = systemstags1[0]
	tags_sys_dddp = systemstags1[1]
	tols_sys_dddp = systemstags1[2]
	dirs_sys_dddp = systemstags1[3]




	systems_all  = SYSTEMS['systemname']
	tags_sys_all = SYSTEMS['Tags']
	tols_sys_all = SYSTEMS['Tols']
	dirs_sys_all = SYSTEMS['directory']

	home_tests_directory = os.getcwd()

	index_memcheck_systems = []
	for i in range(len(systems_valgrind)):
		index_temp = systems_all.index(systems_valgrind[i])
		index_memcheck_systems.append(index_temp)
		del systems_all[index_temp]
		del tags_sys_all[index_temp]
		del tols_sys_all[index_temp]
		del dirs_sys_all[index_temp]

	if 'ddbp' in args:
		systems = systems_dddp
		tags_sys = tags_sys_dddp
		tols_sys = tols_sys_dddp
		dirs_sys = dirs_sys_dddp
		args.remove('ddbp')
	else:
		for i in range(len(systems_dddp)):
			index_temp = systems_all.index(systems_dddp[i])
			del systems_all[index_temp]
			del tags_sys_all[index_temp]
			del tols_sys_all[index_temp]
			del dirs_sys_all[index_temp]




	if 'only_compare' in args:
		temp_result =  True 
		args.remove('only_compare')

	if 'update_reference' in args:
		is_update_reference =  True 
		args.remove('update_reference')


	if len(args) == 1 and re.findall(r'run_local',args[0]) == ['run_local']:
		# systems=SYSTEMS['systemname']
		# tags_sys=SYSTEMS['Tags']
		# tols_sys=SYSTEMS['Tols']
		systems = systems_all
		tags_sys = tags_sys_all
		tols_sys = tols_sys_all
		dirs_sys = dirs_sys_all
		isAuto =  True
		ifVHQ = False
		isparallel = False

	if len(args) == 1 and re.findall(r'clean_temp',args[0]) == ['clean_temp']:
		# systems=SYSTEMS['systemname']
		# tags_sys=SYSTEMS['Tags']
		# tols_sys=SYSTEMS['Tols']
		systems = systems_all
		tags_sys = tags_sys_all
		tols_sys = tols_sys_all
		dirs_sys = dirs_sys_all
		count=0
		for s in systems:
			os.chdir(dirs_sys[count])
			os.chdir(s)
			if 'orient' in tags_sys[count]:
				os.system("rm -r temp_run1 temp_run2 temp_run3")
			else:
				os.system("rm -r temp_run")
			count=count+1
			os.chdir(home_tests_directory)
		sys.exit("Deleted the temp files")

	if len(args) == 1 and re.findall(r'quick_run',args[0]) == ['quick_run']:
		systems=['BaTiO3_quick','H2O_sheet_quick','H2O_wire_quick','SiH4_quick']
		tags_sys = []
		tols_sys = []
		dirs_sys = []
		for i in range(len(systems)):
			for j in range(len(SYSTEMS["systemname"])):
				if systems[i] == SYSTEMS["systemname"][j]:
					tags_sys.append(SYSTEMS["Tags"][j])
					tols_sys.append(SYSTEMS["Tols"][j])
					dirs_sys.append(SYSTEMS["directory"][j])
		isAuto =  True
		ifVHQ = False
		isparallel = False

	if len(args) == 1 and re.findall(r'autosys',args[0]) == ['autosys']:
		indx_test_temp = re.findall(r'\d+',args[0])
		indx_test = int(indx_test_temp[0])
		if True:
			isAuto =  True
			ifVHQ = False
			isparallel = False
			# systems1=SYSTEMS['systemname']
			# tags_sys1=SYSTEMS['Tags']
			# tols_sys1=SYSTEMS['Tols']
			systems1 = systems_all
			tags_sys1 = tags_sys_all
			tols_sys1 = tols_sys_all
			dirs_sys1 = dirs_sys_all

			tags_sys2 = [ tags_sys1[i] for i in range(len(systems1)) if systems1[i] not in ['Fe_spin','He16_NVKG','MgO','Si8_kpt_valgrind','MoS2','SiH4','BaTiO3_valgrind']]
			tols_sys2 = [ tols_sys1[i] for i in range(len(systems1)) if systems1[i] not in ['Fe_spin','He16_NVKG','MgO','Si8_kpt_valgrind','MoS2','SiH4','BaTiO3_valgrind']]
			dirs_sys2 = [ dirs_sys1[i] for i in range(len(systems1)) if systems1[i] not in ['Fe_spin','He16_NVKG','MgO','Si8_kpt_valgrind','MoS2','SiH4','BaTiO3_valgrind']]
			systems2 = [ systems1[i] for i in range(len(systems1)) if systems1[i] not in ['Fe_spin','He16_NVKG','MgO','Si8_kpt_valgrind','MoS2','SiH4','BaTiO3_valgrind']]
			no_systems = len(systems2)

			systems = systems2[(indx_test-1)*int(no_systems/no_concurrency):(indx_test-1)*int(no_systems/no_concurrency)+int(no_systems/no_concurrency)]
			tols_sys = tols_sys2[(indx_test-1)*int(no_systems/no_concurrency):(indx_test-1)*int(no_systems/no_concurrency)+int(no_systems/no_concurrency)]
			tags_sys = tags_sys2[(indx_test-1)*int(no_systems/no_concurrency):(indx_test-1)*int(no_systems/no_concurrency)+int(no_systems/no_concurrency)]
			dirs_sys = dirs_sys2[(indx_test-1)*int(no_systems/no_concurrency):(indx_test-1)*int(no_systems/no_concurrency)+int(no_systems/no_concurrency)]
			remain_systems = no_systems - no_concurrency * int(no_systems/no_concurrency);

			if indx_test < remain_systems:
				systems.append(systems2[indx_test+no_concurrency * int(no_systems/no_concurrency)])
				tols_sys.append(tols_sys2[indx_test+no_concurrency * int(no_systems/no_concurrency)])
				tags_sys.append(tags_sys2[indx_test+no_concurrency * int(no_systems/no_concurrency)])
				dirs_sys.append(dirs_sys2[indx_test+no_concurrency * int(no_systems/no_concurrency)])

	# if len(args) == 1:
	# 	if args[0] == "autosys":
	# 		ifVHQ = False
	# 		isparallel = False
	# 		systems_temp=SYSTEMS['systemname']
	# 		tags_sys_temp=SYSTEMS['Tags']
	# 		tols_sys_temp=SYSTEMS['Tols']
	# 		systems = []
	# 		tags_sys = []
	# 		tags_sys = []
	# 		for i in range(len(systems_temp)):
	# 			if systems_temp[i] not in ['He16_NVTNH','He16_NVKG','MgO','Si8_kpt','CuSi7','MoS2']:
	# 				systems.append(systems_temp[i])
	# 				tags_sys.append(tags_sys_temp[i])
	# 				tags_sys.append(tags_sys_temp[i])
	if len(args) >= 2:
		assert (args[0]=="-tags" or args[0] == "-systems" ), "first argument of the the code is either '-tags' or '-systems'"

		if args[0] == "-tags":
			tags = args[1:]
			if tags == ['VHQ']:
				ifVHQ = True
				# systems=SYSTEMS['systemname']
				# tags_sys=SYSTEMS['Tags']
				# tols_sys=SYSTEMS['Tols']
				systems = systems_all
				tags_sys = tags_sys_all
				tols_sys = tols_sys_all
				dirs_sys = dirs_sys_all
			elif tags == ['valgrind_include']:
				is_valgrind_include = True
				systems = systems_valgrind
				tags_sys = tags_sys_valgrind
				tols_sys = tols_sys_valgrind
				dirs_sys = dirs_sys_valgrind
				# systems=SYSTEMS['systemname']
				# tags_sys=SYSTEMS['Tags']
				# tols_sys=SYSTEMS['Tols']
			elif ((tags == ['valgrind_include', 'VHQ']) or (tags == ['VHQ','valgrind_include'])):
				is_valgrind_include = True
				systems = systems_valgrind
				tags_sys = tags_sys_valgrind
				tols_sys = tols_sys_valgrind
				dirs_sys = dirs_sys_valgrind
				ifVHQ = True
			elif tags == ['valgrind_all']:
				is_valgrind_all = True
				# systems=SYSTEMS['systemname']
				# tags_sys=SYSTEMS['Tags']
				# tols_sys=SYSTEMS['Tols']
				systems = systems_all
				tags_sys = tags_sys_all
				tols_sys = tols_sys_all
				dirs_sys = dirs_sys_all

			elif tags == ['serial','memused']:
				isparallel = False
				ismempbs = True
				tags.remove('memused')
				tags.remove('serial')
				# systems=SYSTEMS['systemname']
				# tags_sys=SYSTEMS['Tags']
				# tols_sys=SYSTEMS['Tols']
				systems = systems_all
				tags_sys = tags_sys_all
				tols_sys = tols_sys_all
				dirs_sys = dirs_sys_all
			elif tags ==['serial']:
				isparallel = False
				tags.remove('serial')
				# systems=SYSTEMS['systemname']
				# tags_sys=SYSTEMS['Tags']
				# tols_sys=SYSTEMS['Tols']
				systems = systems_all
				tags_sys = tags_sys_all
				tols_sys = tols_sys_all
				dirs_sys = dirs_sys_all
			elif tags == ['memused']:
				ismempbs = True
				tags.remove('memused')
				# systems=SYSTEMS['systemname']
				# tags_sys=SYSTEMS['Tags']
				# tols_sys=SYSTEMS['Tols']
				systems = systems_all
				tags_sys = tags_sys_all
				tols_sys = tols_sys_all
				dirs_sys = dirs_sys_all
			else:
				if "serial" in tags:
					isparallel = False
					tags.remove('serial')
				if "valgrind_all" in tags:
					is_valgrind_all = True;
					tags.remove('valgrind_all')
				if "memused" in tags:
					ismempbs = True
					tags.remove('memused')
				if "VHQ" in tags:
					ifVHQ = True
					tags.remove('VHQ')
				if "run_local" in tags:
					isAuto =  True
					ifVHQ = False
					isparallel = False
					tags.remove('run_local')
				if tags == []:
					# tags_sys=SYSTEMS['Tags']
					# systems=SYSTEMS['systemname']
					# tols_sys=SYSTEMS['Tols']
					systems = systems_all
					tags_sys = tags_sys_all
					tols_sys = tols_sys_all
					dirs_sys = dirs_sys_all
				else:
					systemstags = findsystems(tags)
					systems = systemstags[0]
					tags_sys = systemstags[1]
					tols_sys = systemstags[2]
					dirs_sys = systemstags[3]
		if args[0] == "-systems":
			if ('memused' in  args[1:]):
				ismempbs = True
				args.remove('memused')

			if 'VHQ' in  args[1:]:
				ifVHQ = True
				systems = args[1:]
				systems.remove('VHQ')
				tags_sys = []
				tols_sys = []
				dirs_sys = []
				for i in range(len(systems)):
					for j in range(len(SYSTEMS["systemname"])):
						if systems[i] == SYSTEMS["systemname"][j]:
							tags_sys.append(SYSTEMS["Tags"][j])
							tols_sys.append(SYSTEMS["Tols"][j])
							dirs_sys.append(SYSTEMS["directory"][j])

			elif ('serial' in  args[1:]):
				isparallel = False
				ismempbs = True
				systems = args[1:]
				systems.remove('serial')
				tags_sys = []
				tols_sys = []
				dirs_sys = []
				for i in range(len(systems)):
					for j in range(len(SYSTEMS["systemname"])):
						if systems[i] == SYSTEMS["systemname"][j]:
							tags_sys.append(SYSTEMS["Tags"][j])
							tols_sys.append(SYSTEMS["Tols"][j])
							dirs_sys.append(SYSTEMS["directory"][j])

			elif ('valgrind_all' in  args[1:]):
				is_valgrind_all = True;
				systems = args[1:]
				systems.remove('valgrind_all')
				tags_sys = []
				tols_sys = []
				dirs_sys = []
				for i in range(len(systems)):
					for j in range(len(SYSTEMS["systemname"])):
						if systems[i] == SYSTEMS["systemname"][j]:
							tags_sys.append(SYSTEMS["Tags"][j])
							tols_sys.append(SYSTEMS["Tols"][j])
							dirs_sys.append(SYSTEMS["directory"][j])

			elif 'run_local' in  args[1:]:
				isAuto =  True
				ifVHQ = False
				isparallel = False
				systems = args[1:]
				systems.remove('run_local')
				tags_sys = []
				tols_sys = []
				dirs_sys = []
				for i in range(len(systems)):
					for j in range(len(SYSTEMS["systemname"])):
						if systems[i] == SYSTEMS["systemname"][j]:
							tags_sys.append(SYSTEMS["Tags"][j])
							tols_sys.append(SYSTEMS["Tols"][j])
							dirs_sys.append(SYSTEMS["directory"][j])

			else:
				systems = args[1:]
				tags_sys = []
				tols_sys = []
				dirs_sys = []
				for i in range(len(systems)):
					for j in range(len(SYSTEMS["systemname"])):
						if systems[i] == SYSTEMS["systemname"][j]:
							tags_sys.append(SYSTEMS["Tags"][j])
							tols_sys.append(SYSTEMS["Tols"][j])
							dirs_sys.append(SYSTEMS["directory"][j])
				
	if len(args) == 0:
		# systems=SYSTEMS['systemname']
		# tags_sys=SYSTEMS['Tags']
		# tols_sys=SYSTEMS['Tols']
		systems = systems_all
		tags_sys = tags_sys_all
		tols_sys = tols_sys_all
		dirs_sys = dirs_sys_all

	######################## Classifying further for memcheck, MD, relax ###########################################

	singlept = []
	Type=[]
	memcheck=[]
	isspin=[]
	isorient=[]
	ismlff_copy = []
	for i in range(len(systems)):
		if ("orient" in tags_sys[i]):
			isorient.append(True)
		else:
			isorient.append(False)
		if ("mlff21" in tags_sys[i]) or ("mlff22" in tags_sys[i]):
			ismlff_copy.append(True)
		else:
			ismlff_copy.append(False)
		if ("spin" in tags_sys[i]):
			isspin.append(True)
		else:
			isspin.append(False)
		if ("memcheck" in tags_sys[i]) or (is_valgrind_all == True):
			memcheck.append(True)
		else:
			memcheck.append(False)

		if ("relax_cell" in tags_sys[i]):
			singlept.append(False)
			Type.append("relax_cell")
		elif ("relax_atom_nlcg" in tags_sys[i]) or ("relax_atom_lbfgs" in tags_sys[i]) or ("relax_atom_fire" in tags_sys[i])  :
			singlept.append(False)
			Type.append("relax_atom")
		elif ("relax_total_nlcg" in tags_sys[i]) or ("relax_total_lbfgs" in tags_sys[i]) or ("relax_total_fire" in tags_sys[i]):
			singlept.append(False)
			Type.append("relax_total")
		elif ("md_nve" in tags_sys[i]) or ("md_nvtnh" in tags_sys[i]) or ("md_nvkg" in tags_sys[i]) or ("md_npt" in tags_sys[i]):
			singlept.append(False)
			Type.append("MD")
		else:
			singlept.append(True)
			Type.append("None")
	
	# if True:
	# 	index_count=0
	# 	for systs in systems:
	# 		os.chdir(systs)
	# 		if 'orient' in tags_sys[index_count]:
	# 			os.system("mv low_accuracy_orientation1 standard_orientation1")
	# 			os.system("mv low_accuracy_orientation2 standard_orientation2")
	# 			os.system("mv low_accuracy_orientation3 standard_orientation3")
	# 		else:
	# 			os.system("mv low_accuracy standard")
	# 		index_count = index_count + 1
	# 		os.chdir("./..")
	# 	sys.exit("Renamed low-accuracy folders")
	# ewoo

	if is_update_reference:
		if ifVHQ:
			accuracy_text = 'high_accuracy'
		else:
			accuracy_text = 'standard'
		
		index_count=0
		for systs in systems:
			os.chdir(dirs_sys[index_count])
			os.chdir(systs)
			if 'orient' in tags_sys[index_count]:
				if Type[index_count] == "None":
					os.system("cp temp_run1/"+systs+".out "+accuracy_text+"_orientation1/"+systs+".refout")
					os.system("cp temp_run1/"+systs+".static "+accuracy_text+"_orientation1/"+systs+".refstatic")
					os.system("cp temp_run2/"+systs+".out "+accuracy_text+"_orientation2/"+systs+".refout")
					os.system("cp temp_run2/"+systs+".static "+accuracy_text+"_orientation2/"+systs+".refstatic")
					os.system("cp temp_run3/"+systs+".out "+accuracy_text+"_orientation3/"+systs+".refout")
					os.system("cp temp_run3/"+systs+".static "+accuracy_text+"_orientation3/"+systs+".refstatic")
				elif Type[index_count] == "MD":
					os.system("cp temp_run1/"+systs+".out "+accuracy_text+"_orientation1/"+systs+".refout")
					os.system("cp temp_run1/"+systs+".aimd "+accuracy_text+"_orientation1/"+systs+".refaimd")
					os.system("cp temp_run2/"+systs+".out "+accuracy_text+"_orientation2/"+systs+".refout")
					os.system("cp temp_run2/"+systs+".aimd "+accuracy_text+"_orientation2/"+systs+".refaimd")
					os.system("cp temp_run3/"+systs+".out "+accuracy_text+"_orientation3/"+systs+".refout")
					os.system("cp temp_run3/"+systs+".aimd "+accuracy_text+"_orientation3/"+systs+".refaimd")
				elif ((Type[index_count] == "relax_atom") or (Type[index_count] == "relax_cell") or (Type[index_count] == "relax_total")):
					os.system("cp temp_run1/"+systs+".out "+accuracy_text+"_orientation1/"+systs+".refout")
					os.system("cp temp_run1/"+systs+".geopt "+accuracy_text+"_orientation1/"+systs+".refgeopt")
					os.system("cp temp_run2/"+systs+".out "+accuracy_text+"_orientation2/"+systs+".refout")
					os.system("cp temp_run2/"+systs+".geopt "+accuracy_text+"_orientation2/"+systs+".refgeopt")
					os.system("cp temp_run3/"+systs+".out "+accuracy_text+"_orientation3/"+systs+".refout")
					os.system("cp temp_run3/"+systs+".geopt "+accuracy_text+"_orientation3/"+systs+".refgeopt")
			else:
				if Type[index_count] == "None":
					os.system("cp temp_run/"+systs+".out "+accuracy_text+"/"+systs+".refout")
					os.system("cp temp_run/"+systs+".static "+accuracy_text+"/"+systs+".refstatic")
				elif Type[index_count] == "MD":
					os.system("cp temp_run/"+systs+".out "+accuracy_text+"/"+systs+".refout")
					os.system("cp temp_run/"+systs+".aimd "+accuracy_text+"/"+systs+".refaimd")
				elif ((Type[index_count] == "relax_atom") or (Type[index_count] == "relax_cell") or (Type[index_count] == "relax_total")):
					os.system("cp temp_run/"+systs+".out "+accuracy_text+"/"+systs+".refout")
					os.system("cp temp_run/"+systs+".geopt "+accuracy_text+"/"+systs+".refgeopt")
			index_count = index_count + 1
			os.chdir(home_tests_directory)
		sys.exit("Reference files have been updated\n")

	### Reading number of processors from the input file if isparallel == True
	indexy=0
	if isparallel == True:
		procs_nodes_cluster = [nprocs_tests, nnodes_tests]
	else:
		procs_nodes_cluster = [1, 1]

	######################### Launching the jobs ######################################################################
	# launch in a batch of 5 systems in a single pbs file in case of "mempbscheck == False" and in a batch of 1 otherwise
	# Input to the launch function should be  - (i) systems (ii) ifmempbs (iii) numberofprocs
	if isAuto == False and temp_result == False:
		jobID = launchsystems(systems, dirs_sys, memcheck, procs_nodes_cluster, ismempbs, ifVHQ, isorient, not isparallel, ismlff_copy)

	############################### Monitoring #########################################################################
		syst_temp = []
		dirs_temp =[]
		isorient_temp=[]
		for i in range(len(systems)):
			syst_temp.append(systems[i])
			isorient_temp.append(isorient[i])
			dirs_temp.append(dirs_sys[i])

		for i in range_with_status(len(systems)):
			temp = True
			while temp:
				# print(syst_temp, "\n")
				for j in range(len(syst_temp)):
					# if isfinishedJobsID(jobID) == True:
					# 	del syst_temp[j]
					# 	del isorient_temp[j]
					# 	temp = False
					# 	break
					if isfinished(syst_temp[j], dirs_temp[j], isorient_temp[j]) == True:
						del syst_temp[j]
						del isorient_temp[j]
						del dirs_temp[j]
						# syst_temp.remove(syst_temp[j])
						# isorient_temp.remove(isorient_temp[j])
						temp = False
						break
						time.sleep(0.1)
				time.sleep(0.1)

		print('\n')
	elif isAuto == True and temp_result == False:
		countrun=0
		for systs in systems:
			print(str(countrun)+": "+systs+" started running")
			os.chdir(systs)
			if isorient[countrun] == False:
				if os.path.exists("temp_run"):
					os.system("rm -r temp_run")
					os.system("mkdir temp_run")
					os.system("cp standard/*.inpt ./temp_run/")
					os.system("cp standard/*.ion ./temp_run/")
					# os.system("cp ./*.psp8 ./temp_run/")
				else:
					os.system("mkdir temp_run")
					os.system("cp standard/*.inpt ./temp_run/")
					os.system("cp standard/*.ion ./temp_run/")
					# os.system("cp ./*.psp8 ./temp_run/")
				os.chdir("temp_run")
				os.system("./../../sparc -name "+systs+" > log")
			else:
				if os.path.exists("temp_run1"):
					os.system("rm -r temp_run1")
					os.system("mkdir temp_run1")
					os.system("cp standard_orientation1/*.inpt ./temp_run1/")
					os.system("cp standard_orientation1/*.ion ./temp_run1/")
					# os.system("cp ./*.psp8 ./temp_run1/")
				else:
					os.system("mkdir temp_run1")
					os.system("cp standard_orientation1/*.inpt ./temp_run1/")
					os.system("cp standard_orientation1/*.ion ./temp_run1/")
					# os.system("cp ./*.psp8 ./temp_run1/")
				os.chdir("temp_run1")
				os.system("./../../sparc -name "+systs+" > log")

				os.chdir("./..")
				if os.path.exists("temp_run2"):
					os.system("rm -r temp_run2")
					os.system("mkdir temp_run2")
					os.system("cp standard_orientation2/*.inpt ./temp_run2/")
					os.system("cp standard_orientation2/*.ion ./temp_run2/")
					# os.system("cp ./*.psp8 ./temp_run2/")
				else:
					os.system("mkdir temp_run2")
					os.system("cp standard_orientation2/*.inpt ./temp_run2/")
					os.system("cp standard_orientation2/*.ion ./temp_run2/")
					# os.system("cp ./*.psp8 ./temp_run2/")

				os.chdir("temp_run2")
				os.system("./../../sparc -name "+systs+" > log")
				os.chdir("./..")
				if os.path.exists("temp_run3"):
					os.system("rm -r temp_run3")
					os.system("mkdir temp_run3")
					os.system("cp standard_orientation3/*.inpt ./temp_run3/")
					os.system("cp standard_orientation3/*.ion ./temp_run3/")
					# os.system("cp ./*.psp8 ./temp_run3/")
				else:
					os.system("mkdir temp_run3")
					os.system("cp standard_orientation3/*.inpt ./temp_run3/")
					os.system("cp standard_orientation3/*.ion ./temp_run3/")
					# os.system("cp ./*.psp8 ./temp_run3/")
				os.chdir("temp_run3")
				os.system("./../../sparc -name "+systs+" > log")
			countrun=countrun+1
			print(str(countrun)+": "+systs+" has finished running")
			os.chdir("./../..")


	#######################################################################################################################

	count_run=0
	data_info={}
	sys_which_ran_idx=[]
	try:
		os.chdir(home_directory)
		temp=getInfo(systems[0], dirs_sys[0],  singlept[0],Type[0],False,memcheck[0],ismempbs,isspin[0],ifVHQ,isorient[0],tols_sys[0])
		temp1=getInfo(systems[0], dirs_sys[0] ,singlept[0],Type[0],True,memcheck[0],ismempbs,isspin[0],ifVHQ,isorient[0],tols_sys[0])
		data_info[count_run] = {'a': temp, 'b': temp1}
		sys_which_ran_idx.append(count_run)
		count_run=count_run+1
	except:
		print("Warning: "+systems[0]+" did not run or some other issue: please check that \n")

	#temp2 = getInfo(systems[0],singlept[0],Type[0],True,memcheck[0],ismempbs,isspin[0])
	# if os.path.exists('./'+systems[0]+"/"+systems[0]+".refabinitout"):
	# 	temp2 = getInfoAbinit(systems[0],singlept[0],Type[0],isspin[0],ifVHQ)
	# 	data_info = {0: {'a': temp, 'b': temp1, 'c': temp2}}
	# else:

	for i in range(len(systems)):
		if i>0:
			try:
				os.chdir(home_directory)
				temp=getInfo(systems[i], dirs_sys[i],singlept[i],Type[i],False,memcheck[i],ismempbs,isspin[i],ifVHQ,isorient[i],tols_sys[i])				
				temp1=getInfo(systems[i], dirs_sys[i],singlept[i],Type[i],True,memcheck[i],ismempbs,isspin[i],ifVHQ,isorient[i],tols_sys[i])
				temp_dict = {'a': temp, 'b': temp1}
				data_info[count_run] = temp_dict
				sys_which_ran_idx.append(i)
				count_run=count_run+1

			except:
				print("Warning: system named '"+systems[i]+"' did not run or some other issue: please check and rerun this system again \n")


	#tols = readtol(tolfilname)
	sys_which_ran=[]
	isparallel_which_ran=[]
	ifVHQ_which_ran=[]
	isorient_which_ran=[]

	for i in range(len(systems)):
		if i in sys_which_ran_idx:
			sys_which_ran.append(systems[i])
			isorient_which_ran.append(isorient[i])


	os.chdir(home_directory)
	test_status, Warning_message_global, Error_message_global = WriteReport(data_info, sys_which_ran, isparallel, ifVHQ, isorient_which_ran)
	passtests = 0;
	failtests = 0;
	for pp in range(len(test_status)):
		if test_status[pp]=="passed":
			passtests=passtests+1
		else:
			failtests=failtests+1

	#print("out of "+str(passtests+failtests)+"tests, "+str(passtests)+" tests have passed, and "+str(failtests)+" have failed \n")
	CGREEN='\033[92m'
	CRED = '\033[91m'
	CWHITE='\33[0m'
	CBLUE='\033[94m'
	print('--------------------------------------------------------------\n')
	print("Total systems: "+str(passtests+failtests)+"\n")
	print(CGREEN+"Tests passed: "+str(passtests)+CWHITE+"\n")
	print(CRED+"Tests failed: "+str(failtests)+CWHITE+"\n")
	print("Detailed report available in Report.txt file \n")

	count_fail=0
	print('--------------------------------------------------------------\n')
	if failtests > 0:
		print(CRED+'\033[1m'+'Failed test summary: '+CWHITE+ '\033[0m'+'\n')
		for pp in range(len(test_status)):
			if test_status[pp]!="passed":
				print(CRED+str(count_fail+1)+". "+sys_which_ran[pp]+": "+Error_message_global[count_fail]+CWHITE+"\n")
				count_fail=count_fail+1
	print('--------------------------------------------------------------\n')

	print('--------------------------------------------------------------\n')
	count_warn=0;
	print(CBLUE+'\033[1m'+'Warning summary: '+CWHITE+'\033[0m'+'\n')
	for pp in range(len(Warning_message_global)):
		if Warning_message_global[pp]!="":
			print(CBLUE+str(count_warn+1)+". "+sys_which_ran[pp]+": "+Warning_message_global[pp]+CWHITE+"\n")
			count_warn=count_warn+1
	print('--------------------------------------------------------------\n')
	os.chdir(home_directory)
	if os.path.exists("launch_1.pbs"):
		os.system("rm *.pbs")
		os.system("rm *.sparc")

	if isAuto == True:
		if failtests > 0:
			raise Exception(str(failtests) + " out of "+str(passtests+failtests) +" failed")

