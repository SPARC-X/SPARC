"""Testing single point calculations with volume and position changes
between pure SPARC and socket mode
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

os.environ["SPARC_PSP_PATH"] = "../../../psps/"

sparc_params = {
    "h": 0.22,
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

rats = [1.0, 1.01, 1.02, 1.03, 1.04, 1.0, 0.99, 0.98, 0.97, 0.96]

def make_images():
    atoms = bulk("Al", cubic=True)

    images = []
    for i in range(len(rats)):
        at = atoms.copy()
        # at.rattle(0.1, seed=i)
        cell_origin = at.cell.copy()
        rat = rats[i]
        cell = cell_origin * [rat, rat, rat]
        at.set_cell(cell, scale_atoms=True)
        images.append(at)

    return images


def sparc_singlepoint():
    for d in Path(".").glob("sp_image*"):
        shutil.rmtree(d, ignore_errors=True)
    images = make_images()
    out_images = []
    for i, img in enumerate(images):
        # relateive to the path
        rat = rats[i]
        params = sparc_params.copy()
        params["h"] *= rat
        calc = SPARC(
            directory=f"sp_image{i:02d}",
            command="mpirun -n 2 --oversubscribe ../../../../lib/sparc",
            **params,
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

    out_images = []
    with SocketIOCalculator(port=12345) as calc:
        time.sleep(1.0)
        p_ = Popen(
            "mpirun -n 2 --oversubscribe ../../../../lib/sparc -socket :12345 -name SPARC > sparc.log 2>&1",
            shell=True,
            cwd=copy_to,
        )
        for i, atoms in enumerate(images):
            atoms.calc = calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            s = atoms.get_stress()
            e_old = read(
                f"sp_image{i:02d}", format="sparc", index=-1
            ).get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            print("Cycle: ", i)
            print("Energy: ", e)
            print("Energy SPARC SP: ", e_old)
            print("Ediff: ", np.abs(e - e_old))
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
    assert np.max(np.abs(ediff)) < 1.e-3
    assert np.max(np.abs(fdiff)) < 5.e-3
    assert np.max(np.abs(ediff)) < 1.e-3

if __name__ == "__main__":
    main()
