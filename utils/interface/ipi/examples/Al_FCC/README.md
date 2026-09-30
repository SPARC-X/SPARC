# Minimal i-PI + SPARC example (FCC Al, 4 atoms, 5 NVE steps)

Two processes:

1. **i-PI** = MD server (`input.xml`)
2. **SPARC** = DFT force client (`mpirun … sparc -socket …`)

## Setup (once)

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/Al_FCC
cp ../../../../../psps/13_Al_3_1.9_1.9_pbe_n_v1.0.psp8 .
```

## Run (two terminals)

**Terminal A — start i-PI first:**

Use unbuffered Python output. A plain `i-pi … | tee …` often shows a blank terminal until SPARC connects (stdout buffering).

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/Al_FCC
PYTHONUNBUFFERED=1 i-pi input.xml 2>&1 | tee i-pi.log
```

You should see the i-PI banner and messages about initializing / binding forces. It will then wait for a client on port `31415` — that wait is expected.

**Terminal B — start SPARC client (second window):**

```bash
cd utils/interface/ipi/examples/Al_FCC
# MPIEXEC="srun -n" on Slurm. Default is "mpirun -np".
./B_run_sparc.sh
```

## Success checks

- `i-pi.log` advances through steps without hanging
- `simulation.out` appears and gains rows
- `Al.out` / `Al.static` show SCF cycles
- Both processes exit cleanly after 5 steps

## Notes

- Port `31415` must match in `input.xml` and the `-socket` argument
- SPARC `.inpt` keeps `MD_FLAG: 0`; dynamics live in i-PI
- Cell length 7.653391 Bohr ≈ 4.05 Å matches `init.xyz`
