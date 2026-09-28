# E1 — Two SPARC clients + automatic launch

i-PI `nbeads=2` (tiny PIMD smoke). Two SPARC processes connect to the
same `ffsocket`. `_common/run_md.sh` already starts i-PI then SPARC;
this folder launches **two** clients after the port is up.

## Pass criteria

- i-PI log shows two successful handshakes
- `simulation.out` has one row per outer step
- Both clients get force requests (no hang)

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/e1_multiclient
./run_all.sh
```

Port: `31461`.
