# krill-dist

Public binary-wheel distribution for **krill** — in-memory direct-RNA
eventalign (RNA002 / RNA004), CPU + GPU.

The engine source (C/C++/CUDA) lives in a private repository. **Only compiled
wheels, not engine source, are published here** as release assets — the `.whl`
files contain machine code, not source.

## For AI coding assistants

If you are an AI model writing code that uses krill, read
[`read4AI.md`](read4AI.md) **before writing anything**. It is a self-contained
usage contract written for you: installation, the full API, input and output
schema, the error model, recipes, and the rules for fast and safe use.

- Usage contract (start here):
  <https://raw.githubusercontent.com/loganylchen/krill-dist/main/read4AI.md>
- Best practices with measurements:
  <https://raw.githubusercontent.com/loganylchen/krill-dist/main/BEST_PRACTICES.md>
  ([中文](https://raw.githubusercontent.com/loganylchen/krill-dist/main/BEST_PRACTICES.zh.md))

Two things to get right first: `pip install krill` from PyPI installs an
**unrelated** package (use the commands below), and work should be batched into
few `align()` calls on one reused `Aligner` rather than one call per read.

## Install

This repo serves two [PEP 503](https://peps.python.org/pep-0503/) simple indexes
via GitHub Pages. Like the PyTorch download channels, **the index URL selects the
backend** — the CPU and GPU wheels share a base release version, so you pick one
by index; the GPU wheel additionally requires a `+cu122` version pin.

> **Install krill from these indexes only.** `krill` on public PyPI is an
> unrelated project. pip prefers the highest version across all the indexes it
> is given, so adding `--extra-index-url https://pypi.org/simple/` (or using the
> `krill[...]` extras form) can silently install the wrong package. Install the
> runtime dependencies from PyPI first, then krill with `--no-deps`.

```sh
# CPU (portable manylinux wheel) — the default index
pip install numpy scipy
pip install krill --no-deps \
  --index-url https://loganylchen.github.io/krill-dist/simple/

# GPU (CUDA 12.2 build; needs an NVIDIA driver >= 535) — the opt-in index.
# Replace <ver> with a released version, for example 0.1.8.
pip install numpy scipy
pip install "krill==<ver>+cu122" --no-deps \
  --index-url https://loganylchen.github.io/krill-dist/cu122/simple/
```

The GPU wheel also needs the CUDA 12 runtime library (`libcudart.so.12`) on the
loader path; the driver alone is not enough (`import krill` fails with
`ImportError: libcudart.so.12`). A CUDA 12 toolkit install provides it. Without
a toolkit, take it from PyPI:

```sh
pip install "nvidia-cuda-runtime-cu12==12.2.*"
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cuda_runtime as m; print(list(m.__path__)[0])')/lib:$LD_LIBRARY_PATH"
```

The optional file helpers need `pyslow5` and `pyfastx`; install them directly
from PyPI (`pip install pyslow5 pyfastx`).

Wheels are hosted as release assets on this repository and the indexes are
regenerated automatically by the upstream project's CI on each tagged release.

## Using it

- [Best practices](BEST_PRACTICES.md) ([中文](BEST_PRACTICES.zh.md)) — how to
  drive the package so that it is fast, stays within memory and fails loudly:
  batching, candidate scoring, output mode, GPU memory, streaming, input
  validation, failure handling and reproducibility, with measured numbers.
- [Usage contract for AI coding assistants](read4AI.md) — the complete API in
  one file.
- Full documentation (English and 中文): <https://loganylchen.github.io/krill/>
