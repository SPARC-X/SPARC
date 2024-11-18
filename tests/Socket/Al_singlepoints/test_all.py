"""Testing single point calculations between pure SPARC and socket mode
"""

import os
import shutil
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
    "RELAX_FLAG": 0,
    "CALC_STRESS": 1,
    "PRINT_ATOMS": 1,
    "PRINT_FORCES": 1,
}


def make_images():
    atoms = bulk("Al", cubic=True)

    imgfile = Path("al_images.extxyz")

    if not imgfile.is_file():
        images = []
        for i in range(5):
            at = atoms.copy()
            at.rattle(0.1, seed=i)
            images.append(at)
        write(imgfile, images)

    images = read(imgfile, ":")
    return images


def sparc_singlepoint():
    for d in Path(".").glob("sp_image*"):
        shutil.rmtree(d, ignore_errors=True)
    images = make_images()
    out_images = []
    for i, img in enumerate(images):
        # relateive to the path
        calc = SPARC(
            directory=f"sp_image{i:02d}",
            command="mpirun -n 2 --oversubscribe ../../../../lib/sparc",
            **sparc_params,
        )
        img.calc = calc
        e = img.get_potential_energy()
        f = img.get_forces()
        s = img.get_stress()
        img_cp = img.copy()
        img_cp.calc = SinglePointCalculator(img_cp, energy=e, forces=f, stress=s)
        out_images.append(img_cp)
        print(f"image {i}, energy {e}")
    return out_images


def sparc_socket():
    images = make_images()
    # Copy inputs from single point --> socket
    inputs = Path("./sp_image00")
    copy_to = Path("./socket_test")
    shutil.rmtree(copy_to, ignore_errors=True)
    os.makedirs(copy_to, exist_ok=True)
    shutil.copy(inputs / "SPARC.ion", copy_to)
    shutil.copy(inputs / "SPARC.inpt", copy_to)
    shutil.copy(inputs / "13_Al_3_1.9_1.9_pbe_n_v1.0.psp8", copy_to)

    calc = SocketIOCalculator(port=12345)
    p_ = Popen(
        "mpirun -n 2 --oversubscribe ../../../../lib/sparc -socket :12345 -name SPARC > sparc.log 2>&1",
        shell=True,
        cwd=copy_to,
    )
    out_images = []
    with calc:
        for i, atoms in enumerate(images):
            atoms.calc = calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            s = atoms.get_stress()
            e_old = read(
                f"sp_image{i:02d}", format="sparc", index=-1
            ).get_potential_energy()
            #             forces = atoms.get_forces()
            #             stress = atoms.get_stress()
            print("Cycle: ", i)
            print("Energy: ", e)
            print("Energy SPARC SP: ", e_old)
            atoms_cp = atoms.copy()
            atoms_cp.calc = SinglePointCalculator(
                atoms_cp, energy=e, forces=f, stress=s
            )
            out_images.append(atoms_cp)
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


if __name__ == "__main__":
    main()
