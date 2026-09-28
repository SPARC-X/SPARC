# E3 — Restart from i-PI `RESTART`

Copies `b2_nvt/RESTART` (and SPARC inputs) and continues with `i-pi RESTART`.

## Pass criteria

- Restart run completes
- `simulation.out` has additional MD rows
- i-PI log does not report a failed restart

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/e3_restart
./run_all.sh
```

Uses the port stored in the B2 RESTART file (31432).
