# Build conda recipe for sparc

**Note** the official conda-forge package for SPARC can be found at
[`sparc-x`](https://github.com/conda-forge/sparc-x-feedstock). This
recipe is maintained for CI purpose only.

1. Install `conda-build` and (optionally) `boa`
```bash
conda install -c conda-forge "conda-build>=3.20" colorama pip ruamel ruamel.yaml rich mamba jsonschema
pip install git+https://github.com/mamba-org/boa.git
```

2. Build conda package
Assume you are at the top level of the SPARC code tree
```bash
CPU_COUNT=<ncores> conda mambabuild .conda/
```

After compilation a package will be available at `$CONDA_PREFIX/conda-bld/linux-64/sparc-<YYYY>.<MM>.<DD>-<i>.bz2`

3. Optional upload to anaconda

For uploading to personal channel only
```bash
conda install -c conda-forge anaconda-client
anaconda login
anaconda upload $CONDA_PREFIX/conda-bld/linux-64/sparc-<YYYY>.<MM>.<DD>-<i>.bz2
```


4. Using the package
```bash
conda install -c <your-channel> sparc
```
This will automatically install `sparc` with `openmpi` + `scalapack` + `openblas` support. No compilation requires afterwards.
