"""SPARC as an i-PI force client.

This process is the socket server. SPARC, built with ``USE_SOCKET=1``, connects
with ``-socket localhost:PORT`` and returns energy, forces, and the virial.
The wire uses Bohr and Hartree. Callers see Å and eV.
"""

from __future__ import annotations

import os
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as _np

from .system import (
    ANG_TO_BOHR,
    HARTREE_BOHR_TO_EV_ANG,
    HARTREE_TO_EV,
    Structure,
)

_HDRLEN = 12
_AL_NAME = "Al"

_DRIVER_DIR = Path(__file__).resolve().parent
_PLUMED_DIR = _DRIVER_DIR.parent
COMMON_DIR = _PLUMED_DIR / "examples" / "_common"
SPARC_REPO = Path(os.environ.get("SPARC_REPO", _PLUMED_DIR.parents[2]))
SPARC_BIN = Path(os.environ.get("SPARC", SPARC_REPO / "lib" / "sparc"))
PSP_FILE = Path(
    os.environ.get(
        "SPARC_PSP",
        SPARC_REPO / "psps" / "13_Al_3_1.9_1.9_pbe_n_v1.0.psp8",
    )
)


def plumed_libdir() -> Optional[Path]:
    """Directory of ``libplumedKernel.so``, or None when PLUMED is not located.

    ``PLUMED_KERNEL`` and ``PLUMED_ROOT`` win. ``$HOME/opt/plumed`` is used
    only when that directory exists, so a module-provided install is left alone.
    """
    kernel = os.environ.get("PLUMED_KERNEL")
    if kernel and Path(kernel).is_file():
        return Path(kernel).resolve().parent
    root = os.environ.get("PLUMED_ROOT")
    if root and (Path(root) / "lib").is_dir():
        return Path(root) / "lib"
    home_lib = Path.home() / "opt" / "plumed" / "lib"
    if home_lib.is_dir():
        return home_lib
    return None


def apply_runtime_env() -> None:
    """Prepend a discovered PLUMED ``lib`` directory to ``LD_LIBRARY_PATH``."""
    libdir = plumed_libdir()
    if libdir is None:
        return
    lib = str(libdir)
    parts = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    if lib not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([lib] + parts)
    kernel = libdir / "libplumedKernel.so"
    if kernel.is_file():
        os.environ.setdefault("PLUMED_KERNEL", str(kernel))


def mpi_exec(nrank: int) -> list:
    """MPI launcher plus the rank count. ``MPIEXEC`` defaults to ``mpirun -np``."""
    raw = os.environ.get("MPIEXEC", "mpirun -np")
    return [tok for tok in raw.split() if tok] + [str(nrank)]


def _header(message: str) -> bytes:
    return message.upper().ljust(_HDRLEN).encode("ascii")


class _IpiServer:
    """Server end of the i-PI force socket. One SPARC client connects."""

    def __init__(self, port: int, timeout: float = 600.0) -> None:
        self.port = int(port)
        self.timeout = timeout
        self._listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen.bind(("127.0.0.1", self.port))
        self._listen.listen(1)
        self._listen.settimeout(timeout)
        self.conn: Optional[socket.socket] = None

    def accept_client(self) -> None:
        """Block until SPARC connects."""
        conn, _addr = self._listen.accept()
        conn.settimeout(self.timeout)
        self.conn = conn

    def close(self) -> None:
        """Ask the client to exit, then close both sockets."""
        if self.conn is not None:
            try:
                self._send_header("EXIT")
            except OSError:
                pass
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None
        try:
            self._listen.close()
        except OSError:
            pass

    def get_energy_forces(
        self, positions_bohr: _np.ndarray, cell_bohr: _np.ndarray
    ) -> Tuple[float, _np.ndarray, _np.ndarray]:
        """One force evaluation. Returns energy (Ha), forces (Ha/Bohr), virial (Ha)."""
        pos = _np.ascontiguousarray(positions_bohr, dtype=_np.float64).reshape(-1)
        cell = _np.ascontiguousarray(cell_bohr, dtype=_np.float64).reshape(3, 3)
        natoms = pos.size // 3
        self._wait_status("READY")
        payload = (
            _header("POSDATA")
            + cell.astype(_np.float64).tobytes()
            + _np.linalg.inv(cell).astype(_np.float64).tobytes()
            + _np.int32(natoms).tobytes()
            + pos.tobytes()
        )
        self._sendall(payload)
        self._wait_status("HAVEDATA")
        self._send_header("GETFORCE")
        return self._recv_forces(natoms)

    def _send_header(self, message: str) -> None:
        self._sendall(_header(message))

    def _sendall(self, data: bytes) -> None:
        assert self.conn is not None
        self.conn.sendall(data)

    def _recvall(self, nbytes: int) -> bytes:
        assert self.conn is not None
        buf = bytearray()
        while len(buf) < nbytes:
            chunk = self.conn.recv(nbytes - len(buf))
            if not chunk:
                raise ConnectionError("SPARC socket closed")
            buf.extend(chunk)
        return bytes(buf)

    def _wait_status(self, expect: str) -> None:
        self._send_header("STATUS")
        raw = self._recvall(_HDRLEN).decode("ascii", errors="replace").strip()
        token = raw.split()[0].upper() if raw else ""
        if token != expect.upper():
            raise RuntimeError(f"expected {expect}, got {raw!r}")

    def _recv_forces(self, natoms: int) -> Tuple[float, _np.ndarray, _np.ndarray]:
        header = self._recvall(_HDRLEN).decode("ascii", errors="replace").strip().split()[0].upper()
        if header != "FORCEREADY":
            raise RuntimeError(f"expected FORCEREADY, got {header!r}")
        energy = struct.unpack("d", self._recvall(8))[0]
        nrecv = struct.unpack("i", self._recvall(4))[0]
        if nrecv != natoms:
            raise RuntimeError(f"natoms mismatch: server {natoms} client {nrecv}")
        forces = _np.frombuffer(self._recvall(8 * 3 * natoms), dtype=_np.float64).reshape(natoms, 3).copy()
        virial = _np.frombuffer(self._recvall(8 * 9), dtype=_np.float64).reshape(3, 3).copy()
        extra = struct.unpack("i", self._recvall(4))[0]
        if extra:
            self._recvall(extra)
        return float(energy), forces, virial


def load_al_fcc(xyz: Optional[Path] = None) -> Structure:
    """4-atom cubic FCC Al with lattice constant 4.05 Å (7.653391 Bohr).

    When ``xyz`` exists and has four positions, those coordinates replace the
    ideal lattice.
    """
    cell = _np.eye(3) * 4.05
    frac = _np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    positions = frac @ cell
    if xyz is not None and Path(xyz).is_file():
        positions = _read_xyz_positions(Path(xyz), fallback=positions)
    return Structure(["Al", "Al", "Al", "Al"], positions, cell)


def _read_xyz_positions(path: Path, fallback: _np.ndarray) -> _np.ndarray:
    lines = path.read_text().splitlines()
    try:
        natoms = int(lines[0].split()[0])
        rows = []
        for line in lines[2 : 2 + natoms]:
            parts = line.split()
            rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
        arr = _np.array(rows, dtype=_np.float64)
        if arr.shape == fallback.shape:
            return arr
    except (ValueError, IndexError):
        pass
    return fallback


def _prepare_inputs(workdir: Path) -> None:
    """Copy Al.inpt, Al.ion, and the Al pseudopotential into the run directory."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for name in ("Al.inpt", "Al.ion"):
        src = COMMON_DIR / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy(src, workdir / name)
    psp_dst = workdir / PSP_FILE.name
    if psp_dst.is_file():
        return
    if not PSP_FILE.is_file():
        raise FileNotFoundError(f"pseudopotential not found: {PSP_FILE}")
    shutil.copy(PSP_FILE, psp_dst)


class SparcSocketSession:
    """Start SPARC and evaluate energy and forces for one structure at a time."""

    def __init__(
        self,
        workdir: Path,
        port: int,
        np: int = 4,
        sparc_bin: Path = SPARC_BIN,
        timeout: float = 600.0,
        name: str = _AL_NAME,
    ) -> None:
        self.workdir = Path(workdir)
        self.port = int(port)
        self.np = int(os.environ.get("NP", np))
        self.sparc_bin = Path(os.environ.get("SPARC", sparc_bin))
        self.timeout = timeout
        self.name = name
        extra = os.environ.get("MPIRUN_FLAGS", "").split()
        self.mpirun_extra = [item for item in extra if item]
        self.server: Optional[_IpiServer] = None
        self.proc: Optional[subprocess.Popen] = None
        self.last_energy_ev = 0.0
        self.last_forces_ev = _np.zeros((0, 3))
        self.last_virial_ev = _np.zeros((3, 3))

    def __enter__(self) -> "SparcSocketSession":
        """Bind the port, launch SPARC, and wait until it connects."""
        apply_runtime_env()
        _prepare_inputs(self.workdir)
        if not self.sparc_bin.is_file():
            raise FileNotFoundError(
                f"SPARC binary not found: {self.sparc_bin}\n"
                "Set SPARC= to a binary built with USE_SOCKET=1."
            )
        self.log_fp = (self.workdir / "sparc.log").open("w")
        self.server = _IpiServer(self.port, timeout=self.timeout)
        cmd = (
            mpi_exec(self.np)
            + self.mpirun_extra
            + [str(self.sparc_bin), "-socket", f"localhost:{self.port}", "-name", self.name]
        )
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.workdir),
            stdout=self.log_fp,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        try:
            self.server.accept_client()
        except OSError as exc:
            if self.proc.poll() is not None:
                self.log_fp.flush()
                tail = (self.workdir / "sparc.log").read_text(errors="replace")[-2000:]
                raise RuntimeError(
                    f"SPARC client exited before connecting (code {self.proc.returncode}).\n"
                    f"Command: {' '.join(cmd)}\n--- sparc.log ---\n{tail}"
                ) from exc
            raise
        time.sleep(0.05)
        if self.proc.poll() is not None:
            tail = (self.workdir / "sparc.log").read_text(errors="replace")[-2000:]
            raise RuntimeError(f"SPARC died after connect.\n{tail}")
        return self

    def get_energy_forces(self, atoms: Structure) -> Tuple[float, _np.ndarray]:
        """Return the DFT energy (eV) and forces (eV/Å). The virial is stored on the session."""
        assert self.server is not None
        energy_ha, forces_ha, virial_ha = self.server.get_energy_forces(
            atoms.positions * ANG_TO_BOHR,
            atoms.cell * ANG_TO_BOHR,
        )
        self.last_energy_ev = float(energy_ha * HARTREE_TO_EV)
        self.last_forces_ev = _np.array(forces_ha * HARTREE_BOHR_TO_EV_ANG, dtype=_np.float64, copy=True)
        self.last_virial_ev = _np.array(virial_ha * HARTREE_TO_EV, dtype=_np.float64, copy=True)
        return self.last_energy_ev, self.last_forces_ev

    def __exit__(self, exc_type, exc, tb) -> None:
        """Send EXIT and wait for the SPARC process to finish."""
        try:
            if self.server is not None:
                self.server.close()
        finally:
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
            if hasattr(self, "log_fp"):
                self.log_fp.close()
