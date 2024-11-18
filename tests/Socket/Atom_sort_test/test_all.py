"""Testing single point calculations between pure SPARC and socket mode
"""

import os
import shutil
import time
from pathlib import Path
from subprocess import PIPE, Popen

import ase
import numpy as np
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.socketio import SocketIOCalculator
from ase.io import read, write

# from ase.optimize.lbfgs import LBFGS
from ase.optimize.bfgs import BFGS
from sparc import SPARC

os.environ["SPARC_PSP_PATH"] = "../../../psps/"

sparc_params = {
    "h": 0.16,
    "PRECOND_KERKER_THRESH": 0,
    "ELEC_TEMP_TYPE": "Fermi-Dirac",
    "ELEC_TEMP": 300,
    "MIXING_PARAMETER": 1.0,
    "TOL_SCF": 1e-3,
    # "RELAX_FLAG": 1,
    # "CALC_STRESS": 1,
    "PRINT_ATOMS": 1,
    "PRINT_FORCES": 1,
}

# The atoms are sorted
mol = molecule("H2O", vacuum=6, pbc=False)[[1, 2, 0]]


def simple_sparc():
    atoms = mol.copy()
    calc = SPARC(directory="pbc_sp", **sparc_params)
    atoms.calc = calc
    opt = BFGS(atoms, trajectory="sparc-sp.traj")
    opt.run(fmax=0.02)
    return


def sparc_socket():
    atoms = mol.copy()
    calc = SPARC(directory="h2o_test", **sparc_params)
    calc.write_input(atoms)

    with SocketIOCalculator(port=12345) as calc:
        time.sleep(
            1.0
        )  # In some emulated systems the SocketIOCalculator may delay binding
        p_ = Popen(
            "mpirun -n 2 ../../../../lib/sparc -socket :12345 -name SPARC > sparc.log",
            shell=True,
            cwd="h2o_test",
        )
        atoms.calc = calc
        opt = BFGS(atoms, trajectory="sparc-socket.traj")
        opt.run(fmax=0.03)
    assert opt.get_number_of_steps() <= 10 # For the correct mesh spacing, opt should be very close
    return


def main():
    sparc_socket()


if __name__ == "__main__":
    main()
