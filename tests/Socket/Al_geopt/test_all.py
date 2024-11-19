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
from sparc import SPARC

os.environ["SPARC_PP_PATH"] = "../../../psps/"

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
        command="mpirun -n 2 --oversubscribe ../../../../lib/sparc",
        **sparc_params,
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    out_images = read("sparc_geopt", format="sparc", index=":")
    write("sparc_geopt.extxyz", out_images)
    return out_images


def sparc_socket():
    imgfile = Path("sparc_geopt.extxyz")
    images = read(imgfile, ":")

    inputs = Path("./sparc_geopt")
    copy_to = Path("./socket_test")
    shutil.rmtree(copy_to, ignore_errors=True)
    os.makedirs(copy_to, exist_ok=True)
    shutil.copy(inputs / "SPARC.ion", copy_to)
    shutil.copy(inputs / "SPARC.inpt", copy_to)
    shutil.copy(inputs / "13_Al_3_1.9_1.9_pbe_n_v1.0.psp8", copy_to)

    out_images = []
    with SocketIOCalculator(port=12345) as calc:
        time.sleep(1.0)
        p_ = Popen(
            "mpirun -n 2 --oversubscribe ../../../../lib/sparc -socket :12345 -name SPARC > sparc.log",
            shell=True,
            cwd=copy_to,
        )
        for i, atoms in enumerate(images):
            atoms.calc = calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            s = atoms.get_stress()
            atoms_cp = atoms.copy()
            atoms_cp.calc = SinglePointCalculator(
                atoms_cp, energy=e, forces=f, stress=s
            )
            out_images.append(atoms_cp)
        p_.kill()
    return out_images


def main():
    images_sparc = sparc_singlepoint()
    images_socket = sparc_socket()
    ediff = np.array(
        [
            img2.get_potential_energy() - img1.get_potential_energy()
            for img1, img2 in zip(images_sparc, images_socket)
        ]
    )
    fdiff = np.array(
        [
            img2.get_forces() - img1.get_forces()
            for img1, img2 in zip(images_sparc, images_socket)
        ]
    )
    sdiff = np.array(
        [
            img2.get_stress() - img1.get_stress()
            for img1, img2 in zip(images_sparc, images_socket)
        ]
    )

    print("Ediff, ", np.max(np.abs(ediff)))
    print("Fdiff, ", np.max(np.abs(fdiff)))
    print("Sdiff, ", np.max(np.abs(sdiff)))
    assert np.max(np.abs(ediff)) < 1.0e-3
    assert np.max(np.abs(fdiff)) < 0.01
    assert np.max(np.abs(sdiff)) < 0.01


if __name__ == "__main__":
    main()
