# D1 / D2 — MTS: SPARC slow, PLUMED restraint fast

Outer / slow: SPARC `ffsocket` (`mts_weights [1,0]`)
Inner / fast: `ffplumed` restraint (`mts_weights [0,1]`, `nmts [1,4]`)

## Pass criteria

- SPARC socket evaluations ≈ number of outer MD steps
- PLUMED `Calculating (forward loop)` count in `i-pi.log` ≈ 4× SPARC
  (`COLVAR` may not flush for a constant restraint)
- Conserved quantity stays finite (\(|\Delta| < 0.2\) Ha on this short run)

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/d1_mts
./run_all.sh
```

Port: `31451`.
