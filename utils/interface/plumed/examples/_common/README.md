# Shared 4-atom FCC Al client inputs (same cell as i-PI tests)

- `Al.inpt` — DFT only, `MD_FLAG: 0`. No `PLUMED_FLAG`.
- `Al.ion` — 4 Al atoms, cubic cell 7.653391 Bohr = 4.05 Å.
- `init.xyz` — same geometry in Å for the Python driver.

The Python session copies these plus the psp8 file from `psps/` in the SPARC
repo root into each test directory.
