"""Testing single point calculations between pure SPARC and socket mode
"""

import os
import shutil
import time
from pathlib import Path
from subprocess import PIPE, Popen

import ase
import numpy as np
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.socketio import SocketIOCalculator
from ase.io import read, write
from ase.optimize.lbfgs import LBFGS
from sparc import SPARC

os.environ["SPARC_PSP_PATH"] = "../../../psps/"

sparc_params = {
    "h": 0.28,
    "PRECOND_KERKER_THRESH": 0,
    "ELEC_TEMP_TYPE": "Fermi-Dirac",
    "ELEC_TEMP": 300,
    "MIXING_PARAMETER": 1.0,
    "TOL_SCF": 1e-3,
    "RELAX_FLAG": 1,
    "CALC_STRESS": 1,
    "PRINT_ATOMS": 1,
    "PRINT_FORCES": 1,
}


al = bulk("Al", cubic=True)
al.rattle(0.2, seed=42)


def sparc_singlepoint():
    shutil.rmtree("sparc_geopt", ignore_errors=True)
    try:
        os.remove("sparc_geopt.extxyz")
    except Exception:
        pass
    atoms = al.copy()
    calc = SPARC(
        directory=f"sparc_geopt",
        command="mpirun -n 2 ../../../../lib/sparc",
        **sparc_params,
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    out_images = read("sparc_geopt", format="sparc", index=":")
    write("sparc_geopt.extxyz", out_images)
    return out_images


def sparc_socket():
    atoms = al.copy()

    inputs = Path("./sparc_geopt")
    copy_to = Path("./socket_test")
    shutil.rmtree(copy_to, ignore_errors=True)
    os.makedirs(copy_to, exist_ok=True)
    shutil.copy(inputs / "SPARC.ion", copy_to)
    shutil.copy(inputs / "SPARC.inpt", copy_to)
    shutil.copy(inputs / "13_Al_3_1.9_1.9_pbe_n_v1.0.psp8", copy_to)

    with SocketIOCalculator(port=12345) as calc:
        time.sleep(
            1.0
        )  # In some emulated systems the SocketIOCalculator may delay binding
        p_ = Popen(
            "mpirun -n 2 ../../../../lib/sparc -socket :12345 -name SPARC > sparc.log",
            shell=True,
            cwd=copy_to,
        )
        atoms.calc = calc
        opt = LBFGS(atoms, trajectory="sparc-socket.traj")
        opt.run(fmax=0.02)
        p_.kill()
    return atoms.copy()


def main():
    images_sparc = sparc_singlepoint()
    final_socket = sparc_socket()
    final_sparc = images_sparc[-1]
    final_socket.wrap()
    final_sparc.wrap()
    positions_change = final_socket.positions - final_sparc.positions
    max_change = np.linalg.norm(positions_change)
    print("Max shift: ", max_change)
    assert max_change < 0.02


if __name__ == "__main__":
    main()
