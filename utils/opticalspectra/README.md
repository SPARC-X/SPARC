# opticalspectra

Calculates the linear-response optical spectra of one [SPARC](https://github.com/SPARC-X/SPARC) run with the Kubo–Greenwood method: the optical conductivity σ(ω) and the dielectric function ε(ω), as full tensors.

## Method

- **Reference:** L. Calderín, V. V. Karasiev, S. B. Trickey, *Comput. Phys. Commun.* **221**, 118 (2017), [doi:10.1016/j.cpc.2017.08.008](https://doi.org/10.1016/j.cpc.2017.08.008)
- σ̃₁ and σ̃₂ follow Eqs. (10)–(11):
  - σ₁ = Re σ̃₁ − Im σ̃₂
  - σ₂ = −(Im σ̃₁ + Re σ̃₂)
  - ε = 1 + 4πiσ/ω
- Pair weights:
  - exact: Δf/Δε (Eq. 8);
  - approximated: Δf/ω (Eq. 9).
- Drude term: intraband and degenerate pairs (|Δε| ≤ 10⁻⁶ Ha) use −∂f/∂ε (Eqs. 21–25).
- Velocity: v = i[H, r] = i[T, r] + i[V_NL, r], plus i[V_SO, r] with spin–orbit and i[H_τ, r] with meta-GGA.
  - Built from SPARC's own discrete operators: its finite-difference Laplacian, including the cross terms of non-orthogonal cells, and its real-space projectors.
  - This makes it the exact velocity of SPARC's Hamiltonian.
  - Meta-GGA: V_τ = ∂(ρε_xc)/∂τ, which SPARC does not print, is rebuilt with SPARC's own SCAN/rSCAN/r2SCAN formulas from ρ and τ of the `.psi`. SPARC runs meta-GGA without NLCC, so no core density enters.
- Checks printed with every run: f-sum rule (Eqs. 41–44), orthonormality of ψ, Hermiticity of v.
- Units: Hartree atomic units, time dependence e^{−iωt}.

## Scope

- **Supported:**
  - `SPIN_TYP` 0 (with or without spin–orbit) and 1;
  - orthogonal and non-orthogonal cells;
  - periodic and Dirichlet boundaries;
  - Γ-point and k-point runs;
  - LDA, GGA and meta-GGA (SCAN, rSCAN, r2SCAN) functionals.
- **Not supported (the run stops with an error):**
  - `SPIN_TYP` 2;
  - cyclix and helical cells;
  - hybrid functionals;
  - DFT+U.
- **TODO:**
  - parallel calculations (k-points and spins are processed serially).

## Approximations (method options)

The defaults give the exact Kubo–Greenwood result. Any other choice prints a warning to the screen and the `.kglog`.

| Option | Default (exact) | Alternative |
|---|---|---|
| `FORMULA` | `'exact'`: Δf/Δε | `'approximated'`: Δf/ω |
| `BROADENING` | `'lorentzian'`, half width `ETA` | `'gaussian'`, same FWHM 2·`ETA` |
| `DRUDE` | `True`: intraband and degenerate-pair term | `False`: interband only |
| `VELOCITY` | `'gradient+nonlocal'`: full i[H, r] | `'gradient'`: −i∇ only |

Settings that mimic ABINIT:
- **[ABINIT optic](https://docs.abinit.org/guide/optic/):** `'exact'`, `'lorentzian'`, `DRUDE False`; optic `broadening` = `ETA`.
- **[ABINIT conducti](https://docs.abinit.org/topics/Optic/):** `'approximated'`, `'gaussian'`, `DRUDE False`; conducti `dom` = `ETA/√ln2`.

Inherent to any Kubo–Greenwood calculation:
- finite broadening `ETA`;
- a finite band count (`N_BANDS`, the run's `NSTATES`);
- the k-point sampling and mesh of the SPARC run.

## How to Run

**Prerequisites**
- Software:
  - [Python 3](https://www.python.org)
  - [NumPy](https://numpy.org)
  - [SciPy](https://scipy.org)
  - [Matplotlib](https://matplotlib.org) (only for plots)
- Output files of a finished SPARC run:
  - `<name>.out`
  - `<name>.eigen`
  - `<name>.psi`
  - `<name>.static`
  - the `.psp8` files it used (in the `.out` folder, or set `PSP_DIR`)
- The SPARC run needs these flags in its `.inpt` file ([SPARC manual](../../doc/Manual.pdf)):
  ```
  PRINT_ORBITAL: 1
  PRINT_EIGEN: 1
  PRINT_ATOMS: 1
  NP_SPIN_PARAL: 1      # SPIN_TYP 1 only
  ```

**Ways to run** (any input not given is taken from [`inputparams.py`](inputparams.py))
1. **Input file:** set the inputs in [`inputparams.py`](inputparams.py), then
   ```bash
   python kubo_greenwood.py
   ```
2. **Command line:** pass inputs as `--<name in lower case> <value>`:
   ```bash
   python kubo_greenwood.py --out_file Si.out --eigen_file Si.eigen --psi_file Si.psi --static_file Si.static
   ```
   Flags: `--plot` saves plots, `--no_drude` drops the Drude term, `-h` lists all options.
3. **Python:** with this folder on `sys.path`,
   ```python
   from kubo_greenwood import kubo_greenwood
   S = kubo_greenwood(out_file='Si.out', eigen_file='Si.eigen', psi_file='Si.psi', static_file='Si.static')
   ```
   `S` holds all results.

**Results** are written to `KG_output/` next to the `.out` file (see [Outputs](#outputs)). Relative paths are taken from the folder you run in.

## Inputs

**[`inputparams.py`](inputparams.py).** Each entry can also be given on the command line (see [How to Run](#how-to-run)).

| Input | Default | Meaning |
|---|---|---|
| `OUT_FILE` | required | `.out`: cell, grid and FD order, boundary conditions, spin type, smearing, Fermi level, electron count, `.psp8` names |
| `EIGEN_FILE` | required | `.eigen`: eigenvalues ε_n, occupations f_n, k-points and weights (pair energies and weights) |
| `PSI_FILE` | required | `.psi`: Kohn–Sham orbitals ψ_n (velocity matrix elements ⟨n\|v\|m⟩) |
| `STATIC_FILE` | required | `.static`: atom positions (nonlocal projector centres) |
| `PSP_DIR` | folder of `OUT_FILE` | folder of the `.psp8` files: nonlocal (and spin–orbit) projectors for i[V_NL, r] |
| `OMEGA_MIN`, `OMEGA_MAX` | 0.0, 1.0 Ha | ω range |
| `N_OMEGA` | 1001 | uniform grid; keep the step below `ETA/4` |
| `ETA` | 0.005 Ha | broadening half width |
| `FORMULA`, `BROADENING`, `DRUDE`, `VELOCITY` | exact | see [Approximations](#approximations-method-options) |
| `N_BANDS` | `None` | number of lowest bands used; `None`: all bands in the `.psi` (= SPARC `NSTATES`) |
| `OUT_DIR` | `<folder of OUT_FILE>/KG_output` | output folder |
| `PLOT` | `False` | also save PNG plots (`--plot`) |
| `REF_FILES` | `[]` | up to 3 reference spectra (eV, S/m) overlaid on the σ₁ plot |

## Outputs

Files are named after the SPARC run. They are numbered like SPARC's `.out_NN`: if `Si.kgout` exists, the next run writes `Si.kgout_01`, `.kglog_01`, `.sigma_01`, `.epsilon_01`. Concurrent runs never share a suffix.

| File | Contents |
|---|---|
| `<name>.kgout` | results, laid out like SPARC's `.out` (`Label : value (unit)` lines): input parameters, system (with the estimated peak memory, as SPARC's `.out`), DC σ₁ tensor (a.u., S/m, resistivity) with its intraband/degenerate/interband split, f-sum rules, output files, timing |
| `<name>.kglog` | messages: warnings, notes, progress, and the checks (orthonormality of ψ, Hermiticity of v, occupation factor) |
| `<name>.sigma` | conductivity spectrum per ω (Ha, eV): σ₁ (absorptive) and σ₂ (reactive) of xx yy zz xy xz yz, Tr/3 (a.u.), Tr σ₁/3 (S/m) |
| `<name>.epsilon` | dielectric function per ω > 0: ε₁ and ε₂ of the same components, Tr/3 (from which refractive index, reflectivity and absorption follow) |
| `<name>_sigma.png`, `<name>_epsilon.png` | with `PLOT` only: Tr/3 spectra vs ħω (eV), with `REF_FILES` overlaid |

- 1 a.u. of conductivity = 4.599848 × 10⁶ S/m; ε is dimensionless.
- Only Tr/3 is plotted; all six tensor components are in the text files.

## Notes

- Converge the spectrum in `N_BANDS`/`NSTATES`, mesh and k-points.
- The DC conductivity of metals needs `DRUDE True`. Gaussian broadening gives the wrong low-ω/DC limit.
- `.psi` files are memory-mapped and read one k-point at a time.
