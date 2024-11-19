"""Testing single point calculations between pure SPARC and socket mode
"""

import os
import shutil
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from subprocess import PIPE, Popen

import ase
import numpy as np
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.socketio import SocketIOCalculator
from ase.io import read, write

# from ase.optimize.lbfgs import LBFGS
from ase.optimize.bfgs import BFGS
from sparc import SPARC

os.environ["SPARC_PSP_PATH"] = "../../../psps/"

sparc_params = {
    "h": 0.28,
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


class TimeoutException(Exception):
    """Simple class for timeout"""

    pass


@contextmanager
def time_limit(seconds):
    """Usage:
    try:
        with time_limit(60):
            do_something()
    except TimeoutException:
        raise
    """

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out closing sparc process.")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


mol_orign = molecule("H2", vacuum=3, pbc=True)


def sparc_socket(mol, calc_stress=False):
    atoms = mol.copy()
    params = sparc_params.copy()
    if calc_stress:
        params["CALC_STRESS"] = 1
    else:
        params["CALC_STRESS"] = 0
    calc_origin = SPARC(directory="pbc_test", **params)
    calc_origin.write_input(atoms)

    with SocketIOCalculator(port=12345) as calc:
        time.sleep(
            1.0
        )  # In some emulated systems the SocketIOCalculator may delay binding
        p_ = Popen(
            "mpirun -n 2 ../../../../lib/sparc -socket :12345 -name SPARC > sparc.log",
            shell=True,
            cwd="pbc_test",
        )
        atoms.calc = calc
        calc.calculate(atoms)
        results = calc.results
    return results


def main():
    mol = mol_orign.copy()
    mol.pbc = True
    # CASE 1: pbc = True, there is a stress (although meaningless)
    res1 = sparc_socket(mol, calc_stress=True)
    print(res1)
    assert "stress" in res1.keys()
    for i in (0, 1, 2):
        assert abs(res1["stress"][i]) > 1.0e-6  # Should see a real stress component
        assert abs(res1["stress"][i + 3]) < 1.0e-6  # Should see a real stress component

    # CASE 2: pbc = True but no stress calculated
    mol = mol_orign.copy()
    mol.pbc = True
    res2 = sparc_socket(mol, calc_stress=False)
    print(res2)
    assert "stress" in res2.keys()
    assert np.isclose(res2["stress"], 0).all()

    # CASE 2-1: pbc = False with CALC_STRESS=1 --> cannot continue SPARC calculation
    # mol = mol_orign.copy()
    # mol.pbc = False

    # CASE 2-2: pbc = False with CALC_STRESS=0. SocketServer will not return stress
    # because pbc = False in all directions
    mol = mol_orign.copy()
    mol.pbc = False
    res2_2 = sparc_socket(mol, calc_stress=False)
    print(res2_2)
    assert "stress" not in res2_2.keys()

    # CASE 3: pbc=True, CALC_STRESS=1
    al = bulk("Al", cubic=True)
    res3 = sparc_socket(al, calc_stress=True)
    assert "stress" in res3.keys()
    for i in (0, 1, 2):
        assert abs(res3["stress"][i]) > 1.0e-6  # Should see a real stress component
        assert abs(res3["stress"][i + 3]) < 1.0e-6  # Should see a real stress component

    # CASE 4: pbc=[T, T, F], CALC_STRESS=1, and all other 2D BCs
    al = bulk("Al", cubic=True)
    al.pbc = [True, True, False]
    res4 = sparc_socket(al, calc_stress=True)
    print(res4)
    assert "stress" in res4.keys()
    for i in (0, 1):
        assert abs(res4["stress"][i]) > 1.0e-6  # Should see a real stress component
    assert np.isclose(res4["stress"][2:], 0, atol=1.0e-6).all()

    al = bulk("Al", cubic=True)
    al.pbc = [True, False, True]
    res4_1 = sparc_socket(al, calc_stress=True)
    print(res4_1)
    assert "stress" in res4_1.keys()
    for i in (0, 2):
        assert abs(res4_1["stress"][i]) > 1.0e-6  # Should see a real stress component
    assert np.isclose(res4_1["stress"][[1, 3, 4, 5]], 0, atol=1.0e-6).all()

    al = bulk("Al", cubic=True)
    al.pbc = [False, True, True]
    res4_2 = sparc_socket(al, calc_stress=True)
    print(res4_2)
    assert "stress" in res4_2.keys()
    for i in (1, 2):
        assert abs(res4_2["stress"][i]) > 1.0e-6  # Should see a real stress component
    assert np.isclose(res4_2["stress"][[0, 3, 4, 5]], 0, atol=1.0e-6).all()

    # CASE 5: pbc=[T, F, F], CALC_STRESS=1,
    al = bulk("Al", cubic=True)
    al.pbc = [True, False, False]
    res5 = sparc_socket(al, calc_stress=True)
    print(res5)
    assert "stress" in res5.keys()
    i = 0
    assert abs(res5["stress"][i]) > 1.0e-6  # Should see a real stress component
    assert np.isclose(res5["stress"][1:], 0, atol=1.0e-6).all()

    al = bulk("Al", cubic=True)
    al.pbc = [False, True, False]
    res5_1 = sparc_socket(al, calc_stress=True)
    print(res5_1)
    assert "stress" in res5_1.keys()
    i = 1
    assert abs(res5_1["stress"][i]) > 1.0e-6  # Should see a real stress component
    assert np.isclose(res5_1["stress"][[0, 2, 3, 4, 5]], 0, atol=1.0e-6).all()

    al = bulk("Al", cubic=True)
    al.pbc = [False, False, True]
    res5_2 = sparc_socket(al, calc_stress=True)
    print(res5_2)
    assert "stress" in res5_2.keys()
    i = 2
    assert abs(res5_2["stress"][i]) > 1.0e-6  # Should see a real stress component
    assert np.isclose(res5_2["stress"][[0, 1, 3, 4, 5]], 0, atol=1.0e-6).all()


if __name__ == "__main__":
    main()
