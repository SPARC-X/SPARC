"""PLUMED bias and the ion stepper that sits on top of a SPARC force call.

Arrays passed to ``plumed.Plumed()`` are eV, Å, and fs. ``plumed.dat`` literals
follow the UNITS line in that file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from .socket import SparcSocketSession, apply_runtime_env
from .system import ANG_TO_NM, EV_AMU_TO_ANG_FS2, EV_TO_KJMOL, FS_TO_PS, KB_EV, Structure


class PlumedHandle:
    """One ``plumed.Plumed()`` object. Bias forces are added to the DFT forces."""

    def __init__(
        self,
        natoms: int,
        timestep_fs: float,
        temperature_K: float = 300.0,
        plumed_dat: Optional[Union[str, Path]] = None,
        input_lines: Optional[Sequence[str]] = None,
        log_file: str = "plumed.log",
        engine: str = "python-SPARC-socket",
        restart: bool = False,
    ) -> None:
        apply_runtime_env()
        try:
            import plumed
        except ImportError as exc:
            raise ImportError(
                "Python module `plumed` is missing. "
                "Install py-plumed and set PLUMED_KERNEL to libplumedKernel.so."
            ) from exc

        self._pl = plumed.Plumed()
        self.natoms = int(natoms)
        self._pl.cmd("setRealPrecision", 8)
        self._pl.cmd("setMDEnergyUnits", EV_TO_KJMOL)
        self._pl.cmd("setMDLengthUnits", ANG_TO_NM)
        self._pl.cmd("setMDTimeUnits", FS_TO_PS)
        self._pl.cmd("setMDChargeUnits", 1.0)
        self._pl.cmd("setMDMassUnits", 1.0)
        self._pl.cmd("setNatoms", self.natoms)
        self._pl.cmd("setMDEngine", engine)
        self._pl.cmd("setLogFile", str(log_file))
        self._pl.cmd("setTimestep", float(timestep_fs))
        self._pl.cmd("setKbT", float(temperature_K) * KB_EV)
        self._pl.cmd("setRestart", bool(restart))
        if plumed_dat is not None and input_lines:
            raise ValueError("pass either plumed_dat or input_lines, not both")
        if plumed_dat is not None:
            self._pl.cmd("setPlumedDat", str(plumed_dat))
            self._pl.cmd("init")
        else:
            self._pl.cmd("init")
            for line in input_lines or []:
                self._pl.cmd("readInputLine", line)

        n = self.natoms
        self._pos = np.zeros((n, 3), dtype=np.float64)
        self._masses = np.zeros(n, dtype=np.float64)
        self._charges = np.zeros(n, dtype=np.float64)
        self._box = np.zeros((3, 3), dtype=np.float64)
        self._forces_plumed = np.zeros((n, 3), dtype=np.float64)
        self._virial = np.zeros((3, 3), dtype=np.float64)

    def evaluate(
        self,
        step: int,
        atoms: Structure,
        energy_ev: float,
        forces_ev_a: np.ndarray,
        charges: Optional[np.ndarray] = None,
        virial_ev: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Return the bias energy (eV) and the DFT forces plus the PLUMED forces."""
        self._pos[:, :] = np.asarray(atoms.get_positions(), dtype=np.float64)
        self._masses[:] = np.asarray(atoms.get_masses(), dtype=np.float64)
        self._box[:, :] = np.asarray(atoms.get_cell(), dtype=np.float64)
        self._forces_plumed.fill(0.0)
        if virial_ev is None:
            self._virial.fill(0.0)
        else:
            self._virial[:, :] = np.asarray(virial_ev, dtype=np.float64).reshape(3, 3)
        if charges is None:
            self._charges.fill(0.0)
        else:
            self._charges[:] = np.asarray(charges, dtype=np.float64)

        self._pl.cmd("setStep", int(step))
        self._pl.cmd("setPositions", self._pos)
        self._pl.cmd("setMasses", self._masses)
        self._pl.cmd("setCharges", self._charges)
        self._pl.cmd("setBox", self._box)
        self._pl.cmd("setEnergy", float(energy_ev))
        self._pl.cmd("setForces", self._forces_plumed)
        self._pl.cmd("setVirial", self._virial)
        try:
            self._pl.cmd("prepareCalc")
            self._pl.cmd("performCalc")
        except Exception:
            self._pl.cmd("calc")

        bias = np.zeros(1, dtype=np.float64)
        try:
            self._pl.cmd("getBias", bias)
            energy_bias = float(bias[0])
        except Exception:
            energy_bias = 0.0
        total_forces = np.ascontiguousarray(forces_ev_a, dtype=np.float64) + self._forces_plumed
        self.last_virial = np.array(self._virial, dtype=np.float64, copy=True)
        return energy_bias, total_forces

    def finalize(self) -> None:
        """Close the PLUMED object and flush its files."""
        self._pl.finalize()

    def __enter__(self) -> "PlumedHandle":
        return self

    def __exit__(self, *args) -> None:
        self.finalize()


class SparcPlumedEngine:
    """One step: SPARC DFT, then an optional PLUMED bias, recorded in ``history``."""

    def __init__(self, sparc: SparcSocketSession, plumed: Optional[PlumedHandle] = None) -> None:
        self.sparc = sparc
        self.plumed = plumed
        self.istep = 0
        self.history = {
            "step": [],
            "energy_dft": [],
            "energy_bias": [],
            "energy": [],
            "volume_a3": [],
        }
        self.last_virial = np.zeros((3, 3))

    def evaluate(self, atoms: Structure) -> Tuple[float, np.ndarray]:
        """Return the total energy (eV) and the total forces (eV/Å)."""
        energy_dft, forces_dft = self.sparc.get_energy_forces(atoms)
        virial = np.array(self.sparc.last_virial_ev)
        energy_bias = 0.0
        forces = forces_dft
        if self.plumed is not None:
            energy_bias, forces = self.plumed.evaluate(
                self.istep, atoms, energy_dft, forces_dft, virial_ev=virial
            )
            virial = np.array(self.plumed.last_virial)
        self.last_virial = virial
        self.history["step"].append(self.istep)
        self.history["energy_dft"].append(energy_dft)
        self.history["energy_bias"].append(energy_bias)
        self.history["energy"].append(energy_dft + energy_bias)
        self.history["volume_a3"].append(float(abs(np.linalg.det(atoms.cell))))
        self.istep += 1
        return energy_dft + energy_bias, forces


class Verlet:
    """Velocity Verlet in Å, eV, fs, and amu.

    A positive temperature draws Maxwellian velocities at the start and then
    removes the center-of-mass velocity. Later steps are microcanonical.
    """

    def __init__(self, timestep_fs: float = 1.0, temperature_K: float = 0.0, seed: int = 1) -> None:
        self.timestep_fs = float(timestep_fs)
        self.temperature_K = float(temperature_K)
        self.seed = int(seed)

    def run(
        self,
        atoms: Structure,
        forces_fn: Callable[[Structure], np.ndarray],
        nsteps: int,
        on_step: Optional[Callable[[int, Structure], None]] = None,
    ) -> None:
        """Integrate ``nsteps`` steps. ``forces_fn`` returns forces in eV/Å."""
        dt = self.timestep_fs
        mass = atoms.masses.reshape(-1, 1)
        if self.temperature_K > 0.0:
            rng = np.random.default_rng(self.seed)
            sigma = np.sqrt(np.maximum(self.temperature_K * KB_EV * EV_AMU_TO_ANG_FS2 / mass, 0.0))
            atoms.velocities = rng.normal(0.0, 1.0, atoms.positions.shape) * sigma
            atoms.velocities -= atoms.velocities.mean(axis=0, keepdims=True)
        else:
            atoms.velocities[:] = 0.0

        forces = forces_fn(atoms)
        acc = forces / mass * EV_AMU_TO_ANG_FS2
        for step in range(nsteps):
            atoms.velocities += 0.5 * dt * acc
            atoms.positions += dt * atoms.velocities
            forces = forces_fn(atoms)
            acc = forces / mass * EV_AMU_TO_ANG_FS2
            atoms.velocities += 0.5 * dt * acc
            if on_step is not None:
                on_step(step + 1, atoms)


def run_verlet(
    atoms: Structure,
    forces_fn: Callable[[Structure], np.ndarray],
    nsteps: int,
    timestep_fs: float = 1.0,
    temperature_K: float = 0.0,
    seed: int = 1,
    on_step: Optional[Callable[[int, Structure], None]] = None,
) -> None:
    """Velocity Verlet. ``temperature_K=0`` keeps the velocities at zero."""
    Verlet(timestep_fs, temperature_K, seed).run(atoms, forces_fn, nsteps, on_step)


def run_nvt(
    atoms: Structure,
    forces_fn: Callable[[Structure], np.ndarray],
    nsteps: int,
    timestep_fs: float = 1.0,
    temperature_K: float = 300.0,
    tau_fs: float = 10.0,
    seed: int = 1,
    on_step: Optional[Callable[[int, Structure], None]] = None,
) -> None:
    """Draw velocities at ``temperature_K``, then run velocity Verlet.

    ``tau_fs`` is accepted so older calls still parse. This stepper does not
    apply a Langevin friction.
    """
    del tau_fs
    run_verlet(
        atoms,
        forces_fn,
        nsteps=nsteps,
        timestep_fs=timestep_fs,
        temperature_K=temperature_K,
        seed=seed,
        on_step=on_step,
    )
