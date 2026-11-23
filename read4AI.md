# read4AI.md — krill, an agent-readable usage spec

<!-- Generated from the krill docs (docs/read4AI.md) by .github/scripts/export_dist_docs.py. Do not edit here. -->

> Audience: an AI agent that must use the `krill` Python package correctly with
> no other context. This file is normative and self-contained. Every signature,
> key, dtype, and semantic below is verified against the source. If something is
> not stated here, do not assume it exists.
>
> Describes krill **0.1.8**. Public raw copy:
> <https://raw.githubusercontent.com/loganylchen/krill-dist/main/read4AI.md>.
> Read §10 (rules for fast and safe use) before writing code; the measurements
> behind those rules are in
> [BEST_PRACTICES.md](https://github.com/loganylchen/krill-dist/blob/main/BEST_PRACTICES.md).

---

## 0. TL;DR (the one thing to know)

krill turns a nanopore direct-RNA read's **own basecalled sequence + raw signal**
into **eventalign-format per-column numpy arrays**, fully in memory. No reference
genome, no BAM, no FAST5/HDF5. It is "align-to-self" / resquiggle semantics.

```python
import krill
aligner = krill.Aligner(pore="rna004", use_gpu=False)
out = aligner.align({
    "read_id": "r1",
    "sequence": "ACGUACGUACGU...",   # the read's OWN basecalls (RNA, U ok)
    "signal": raw_adc_float32,        # 1-D np.float32, raw ADC samples
    "digitisation": 8192.0,
    "offset": 5.0,
    "range": 1437.0,
    "sample_rate": 4000.0,
})
# out is list[dict]; out[0]["event_level_mean"] etc. are numpy arrays.
```

---

## 1. What krill is / is NOT

| It IS | It is NOT |
|---|---|
| In-memory direct-RNA eventalign (RNA002 / RNA004) | A reference-genome aligner |
| CPU and optional CUDA GPU | A basecaller |
| Emits the read's **own** k-mers (self-alignment) | A methylation caller (that's the separate f5c CLI) |
| Pure numpy in / numpy out | A FAST5/HDF5/BAM tool |

Because it aligns to the read's own sequence, the `reference_kmer`,
`model_kmer`, and `position` columns are about the read itself — they are NOT
genome coordinates and will NOT match `f5c eventalign` against a reference.
Everything reference-independent (event means, scaling, signal indices) matches
the f5c engine exactly.

There is also a **separate, additive DTW primitive** (`krill.dtw_*`) for
read-vs-read signal distances; it is unrelated to the aligner. See §6.

---

## 2. Install / import

Package name on the index is **`krill`**. It is distributed as prebuilt wheels
from its own GitHub-Pages PEP 503 index (NOT public PyPI — `krill` on PyPI is
an **unrelated** project, currently at a *higher* version than this one).

> Critical: do NOT use `--extra-index-url`. pip aggregates all indexes and
> prefers the highest version, so the unrelated PyPI `krill` would win. Install
> krill from ONLY its own index with `--index-url --no-deps`, and install its
> runtime deps (`numpy`, `scipy`) separately from PyPI first.

```bash
# runtime deps from PyPI (krill itself must NOT be resolved from PyPI)
pip install numpy scipy

# CPU wheel — from the krill index only
pip install krill --no-deps --index-url https://loganylchen.github.io/krill-dist/simple/

# GPU wheel (CUDA 12.2). The +cu122 local-version label is REQUIRED to get the
# GPU build (PyPI has no +cu122, so this pin is self-protecting).
pip install "krill==0.1.8+cu122" --no-deps \
    --index-url https://loganylchen.github.io/krill-dist/cu122/simple/

# The GPU wheel needs the CUDA 12 runtime library (libcudart.so.12) on the loader
# path; an NVIDIA driver alone is not enough (import fails with
# "ImportError: libcudart.so.12"). Without a CUDA toolkit, take it from PyPI:
pip install "nvidia-cuda-runtime-cu12==12.2.*"
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cuda_runtime as m; print(list(m.__path__)[0])')/lib:$LD_LIBRARY_PATH"
```

```python
import krill
krill.built_with_cuda      # bool ATTRIBUTE (not a call): True if compiled with CUDA
krill.gpu_device_count()   # CALL: >0 device count, 0 none, -1 CPU-only build

from importlib.metadata import version
version("krill")           # "0.1.8" or "0.1.8+cu122"; there is no krill.__version__
```

Trap: `built_with_cuda` is a bool attribute; `gpu_device_count` is a function.
Calling `built_with_cuda()` raises `TypeError`.

Runtime deps: `numpy`, `scipy` (scipy only used by the DTW >2047 resample path).
Optional file helpers (§7) need `pyslow5` + `pyfastx`. Install them **directly
from PyPI** — do NOT use `pip install "krill[slow5]"`, because the `krill[...]`
form re-resolves `krill` itself against PyPI and could pull the unrelated PyPI
package:

```bash
pip install pyslow5 pyfastx
```

---

## 3. `krill.Aligner` — the core object

```python
Aligner(pore="rna004",
        use_gpu=None,        # None/"auto": GPU iff a device exists; True/False to force
        batch_size=0,        # 0 = auto (GPU 2048, CPU 512)
        num_thread=0,        # 0 = auto (the CPUs this process may run on)
        cuda_dev_id=0,
        hmm_confidence=True, # re-decode events w/ profile HMM; flag bad events 'B'
        keep_kmer_skips=True,# also emit marked 'K' k-mer-skip rows (only if hmm_confidence)
        polya=False,         # poly(A) tail-length estimate, RNA002 + RNA004 (see below)
        cuda_max_avg_events_per_kmer=0.0, # GPU split gate; 0 = engine default 5.0
        cuda_max_readlen=0.0,             # GPU split gate; 0 = engine default 3.0
        cuda_mem_frac=0.0)   # fraction of FREE GPU memory to pre-allocate; 0 = default (0.9)
```

Pass these by keyword.

- `pore`: `"rna002"` (alias `"r9"`, `"RNA002"`) or `"rna004"` (`"RNA004"`).
  Any other value → `ValueError`. RNA002 = 5-mer model, RNA004 = 9-mer model.
- `use_gpu`:
    - `None` or `"auto"` → GPU iff `gpu_device_count() > 0`, else CPU.
    - `True` → GPU; if no device / CPU-only build, it **warns and falls back to CPU**
      (does NOT raise).
    - `False` → CPU.
    - A string other than `"auto"` → `ValueError`.
- `cuda_dev_id`: validated only when GPU is actually selected; out of
  `[0, device_count)` → `ValueError` (this guard prevents a host-interpreter
  crash — see §5).
- `use_gpu` is a **runtime** choice; it is independent of whether the wheel was
  built with CUDA. A CPU-only wheel simply ignores GPU requests (warns).
- `cuda_max_avg_events_per_kmer` / `cuda_max_readlen`: the GPU load-balancer
  split gates. A read is processed on the GPU only if its events/base is below
  `cuda_max_avg_events_per_kmer` **and** its length is below `cuda_max_readlen ×
  (batch mean length)`. Both default to `0.0`, which keeps the engine defaults
  (`5.0` / `3.0`). Direct-RNA reads run ~1.6 events/base — well under `5.0` — so
  with `use_gpu=True` they are routed to the GPU automatically; these knobs only
  exist to tune the CPU/GPU split for benchmarking or unusual batches. (`<=0`
  always means "keep the engine default".)
- `num_thread`: `0` = the number of CPUs the process is allowed to run on (its
  affinity mask: a SLURM binding, `taskset`, a container cpuset). A CPU *quota*
  without a cpuset (`docker --cpus=4`) is not visible; pass `num_thread` there.
- `cuda_mem_frac`: a GPU `Aligner` pre-allocates this fraction of the device's
  **free** memory at construction and holds it until it is destroyed. `0`
  (default) = 0.9. Use a value in `(0, 1)` to share a GPU, e.g. `0.3`; any other
  value → `ValueError`. Reads that do not fit in a batch's GPU capacity are
  aligned on the CPU instead (nothing fails). Ignored on the CPU.
- `polya`: `False` (default). If `True`, each result also carries a per-read
  poly(A) tail-length estimate (the nanopolish segmentation-HMM estimator; see
  §4.1). Both pores run: **RNA002** is the faithful nanopolish port;
  **RNA004** reuses the same pore-agnostic segmentation engine with a separate
  constant pack learned offline — emission gaussians bootstrapped by unsupervised
  EM on RNA004 signal, and an affine length conversion calibrated against dorado
  `pt:i` (held-out MAE ≈ 4 nt, r ≈ 0.99; dorado is the ceiling, short tails < 10
  nt remain hard). `polya=True` adds output keys but never changes the event rows.

Read-only properties after construction:
`aligner.use_gpu` (bool), `aligner.batch_size` (int), `aligner.num_thread` (int),
`aligner.device_count` (int; 0 none, -1 CPU-only build), `aligner.kmer_center`
(int; RNA002→2, RNA004→4 — the default position offset to the k-mer's central
base), `aligner.cuda_max_avg_events_per_kmer` (float; the requested value, `0.0`
= engine default), `aligner.cuda_max_readlen` (float; same convention),
`aligner.cuda_mem_frac` (float; same convention).

### 3.1 `aligner.align(reads) -> list[dict]`

Accepts **one dict** or a **list of dicts**. ALWAYS returns a `list[dict]` (a
single dict in → a 1-element list out). Batch many reads in one call for
throughput (the engine fans them across `num_thread` workers with the GIL
released).

`align()` is **not reentrant on one `Aligner`**: a second concurrent call on the
same object raises `RuntimeError`. Use one `Aligner` per thread; distinct
`Aligner`s run concurrently.

Input dict schema (per read):

| key | type | required | notes |
|---|---|---|---|
| `sequence` | str | **yes** | the read's own basecalls. A/C/G/T/U, any case. U→T internally. Any other char → `ValueError`. Shorter than the pore k-mer (5 for RNA002, 9 for RNA004) → `ValueError`. |
| `signal` | np.ndarray | **yes** | **1-D**, cast to float32. Raw ADC samples (NOT pA). 2-D → `ValueError`; empty → `ValueError`; any NaN/inf → `ValueError`. |
| `digitisation` | float | **yes** | must be finite and > 0 |
| `offset` | float | **yes** | must be finite |
| `range` | float | **yes** | must be finite and > 0 |
| `sample_rate` | float | **yes** | must be finite and > 0 (Hz, e.g. 4000) |
| `read_id` | str | no | default `""`; echoed nowhere except your own use |
| `label` | any | no | if present (and not `None`), echoed back verbatim as `result["label"]` |
| `start` | int | no | added to every `position` in the output (transcript-coord offset); default 0 |
| `init_scale`, `init_shift` | float | no | give **both or neither**. Fix the initial per-read scaling instead of estimating it from this read's events; used when aligning a slice of a read (pass the full-read result's `init_scale` / `init_shift`). `init_scale` must be finite and > 0, `init_shift` finite. |

One invalid read makes the **whole call** raise (§5), so validate before batching.

The four scalars (`digitisation`/`offset`/`range`/`sample_rate`) come from the
signal file's read metadata (e.g. slow5/blow5 `digitisation`, `offset`, `range`,
`sampling_rate`). They define ADC→pA conversion; wrong values → garbage scaling.

---

## 4. Output schema (each result dict)

Per-read scalars:

| key | type | meaning |
|---|---|---|
| `read_index` | int | index of this read in the input list |
| `status` | int | 0 = ok; else bitwise OR of FAILED_* flags (see §5) |
| `scale` | float | calibrated scaling |
| `shift` | float | calibrated shift |
| `var` | float | calibrated variance |
| `init_scale`, `init_shift` | float | the initial scaling that drove the alignment (estimated, or the values you passed) |
| `label` | (echo) | present only if you passed `label` in the input |

Poly(A) per-read scalars — **present only when the aligner was built with
`polya=True`** (both pores supported; RNA004 emissions learned by EM on RNA004
signal, length calibrated to dorado `pt:i` — see §3 and §4.1). Omitted entirely
otherwise:

| key | type | meaning |
|---|---|---|
| `polya_length` | float | estimated poly(A) tail length in **nucleotides** (clamped ≥0; `-1.0` if not computed) |
| `read_rate` | float | nt/sec used in the estimate (`0.0` if not computed) |
| `polya_sample_start` | int | poly(A) start sample index into the read's signal, inclusive (`-1` if n/a) |
| `polya_sample_end` | int | poly(A) end sample index, exclusive (`-1` if n/a) |
| `polya_qc` | str | `"PASS"` / `"NOREGION"` / `"ADAPTER"` / `"READ_FAILED"` — see §4.1 |

See §4.1 for semantics. `polya_length` is per-READ (not per-row); trust it only
when `polya_qc == "PASS"`.

Per-row arrays (all length `n_rows`; one row = one event aligned to one k-mer).
`N = len(d["position"])`:

| key | dtype | meaning |
|---|---|---|
| `position` | int32 | k-mer position along the read (+ your `start` offset). Points at the k-mer's CENTRAL base. |
| `reference_kmer` | list[str] | the read's OWN k-mer (DNA alphabet, T not U) |
| `event_index` | int32 | event number |
| `event_level_mean` | float32 | event mean in **pA** (raw ADC→pA; the fitted `scale`/`shift` are reported separately and are NOT applied here, so this is not model-normalized) |
| `event_stdv` | float32 | event std dev |
| `event_length` | float32 | event duration in **seconds** |
| `model_kmer` | list[str] | model k-mer; all-`N` (length = pore k-mer: 5 for RNA002, 9 for RNA004) for HMM 'B' bad events |
| `model_mean` | float32 | pore-model mean for `model_kmer`, **already scaled to this read** (`scale * level + shift`); do not apply `scale`/`shift` again. **0.0 (unset) on 'B' rows** — not a real model value |
| `model_stdv` | float32 | pore-model std dev, **already scaled to this read** (`level_stdv * var`). **0.0 (unset) on 'B' rows** |
| `standardized_level` | float32 | `(event_level_mean - model_mean) / (sqrt(var) * model_stdv)`; **NaN for HMM background 'B' rows** |
| `start_idx` | int64 | signal sample start (inclusive) |
| `end_idx` | int64 | signal sample end (non-inclusive) |
| `hmm_state` | list[str] | see below; `''` only if the underlying char is `\0` |

Notes for consumers:
- `reference_kmer`, `model_kmer`, `hmm_state` are **Python lists of str**, not
  numpy arrays. Everything else is a numpy array of the dtype above.
- `hmm_state` values depend on construction flags:
    - `hmm_confidence=True` (default, and a successful HMM re-decode): you get
      `'M'` (match) / `'B'` (bad/background event; `model_kmer` is all-`N` of the
      pore's k-mer length — `NNNNN` for RNA002, `NNNNNNNNN` for RNA004, so filter
      on `hmm_state == 'B'`, never on a literal N-string), plus `'K'` (k-mer skip)
      rows iff `keep_kmer_skips=True`.
    - `hmm_confidence=False`, OR the HMM re-decode failed and fell back to the raw
      ABEA band: rows come from `postalign`, which sets `'M'` (this k-mer differs
      from the previous row's k-mer = a move) or `'E'` (same k-mer as the previous
      row = an extra/stay event). There are no `'B'`/`'K'` rows in this mode, and
      `standardized_level` is always a real number (no NaN, since there are no 'B'
      rows).
    - The two modes are **different alignments**, not the same rows with and
      without flags. Use one mode per analysis.
- Signal slice for a row is `signal[start_idx:end_idx]`. **Caveat:** `'K'`
  (k-mer-skip) rows consume no event — they share `event_index` / `start_idx` /
  `end_idx` with a neighbouring row. If you slice signal per row, filter out or
  special-case `hmm_state == 'K'` first, or you will double-count those samples.
- `'B'` rows have no pore-model state: `model_mean` and `model_stdv` are `0.0`
  (placeholders, not real values) and `standardized_level` is `NaN`.
- A read that produced no alignment comes back with `n_rows == 0` (all arrays
  length 0) and a non-zero `status`; the rest of the batch still completes.

### 4.1 Poly(A) estimate (`polya=True`; both pores, RNA004 EM emissions + dorado-calibrated length)

A port of nanopolish's `polya` estimator: a 6-state left-to-right
segmentation HMM (`START → LEADER → ADAPTER → [POLYA ↔ CLIFF] → TRANSCRIPT`)
Viterbi-decoded over the read's raw **pA** samples, then
`polya_length = gain * ((polya_samples / sample_rate) * read_rate) + offset`,
clamped ≥ 0, where `read_rate = 1 / median(per-k-mer summed event durations)`
(`gain = 1.0` recovers the nanopolish offset-only form).

The segmentation engine (Viterbi / state machine / region extraction / read-rate
/ QC) is **pore-agnostic**; only the emission/transition constants, the
sample-unit thresholds and the length `gain`/`offset` differ per chemistry.
**RNA002** uses nanopolish's verbatim empirical R9 constants, `gain = 1.0` and the
`−5` offset. **RNA004** uses a separate constant pack
learned offline: the emission gaussians are
bootstrapped by unsupervised EM on RNA004 signal, and the affine `gain`/`offset`
are fit against dorado `pt:i` labels (held-out MAE ≈ 4 nt, r ≈ 0.99). dorado is
the ceiling — krill learns to agree with it, not to beat it — and short tails
< 10 nt remain the known hard case.

- The estimate is **per-read**, derived from this read's signal + calibration.
  Because krill aligns to the read's own sequence, the `scale`/`shift`/`var` and
  `read_rate` are sequence-dependent, so `polya_length` can vary slightly across
  alternative candidate sequences for the same signal (e.g. in
  `align_read_variants`). Treat it as a per-alignment estimate.
- `polya_qc` priority: `"NOREGION"` (adapter or poly(A) region < 200 samples) →
  `"ADAPTER"` (estimated adapter > 300 nt) → `"PASS"`. `"READ_FAILED"` means the
  estimate was not computed (read failed alignment, or no usable k-mer
  durations). **Only trust `polya_length` when `polya_qc == "PASS"`.**
- `polya_sample_start`/`polya_sample_end` index into the read's own signal
  (native order, as passed in), half-open `[start, end)`.
- nanopolish's BAM/reference-only QC tag `SUFFCLIP` is not portable to krill's
  align-to-self model and is dropped.

---

## 5. Error model (important — read before catching)

krill validates at the Python boundary and raises normal Python exceptions; it
does **not** crash the interpreter on bad input or on a GPU fault. Two layers.

**Per-call exceptions (the whole `align()` call fails, no results):**

| exception | cause | what to do |
|---|---|---|
| `ValueError` | invalid value: bad `pore` / `use_gpu` string / `cuda_dev_id` / `cuda_mem_frac`; unsupported base; sequence shorter than the k-mer; empty, non-1-D or non-finite `signal`; non-finite or non-positive scalar; `init_scale` without `init_shift`. Per-read messages name the read (`read 3: ...`). | fix the constructor argument, or fix or drop that read; retrying fails again |
| `KeyError` | a required key is missing from a read dict | fix the input |
| `RuntimeError: Unable to cast ...` | a field has the wrong type (e.g. a `str` where a number is expected) | fix the input |
| `RuntimeError: Aligner.align() is already running ...` | two threads used one `Aligner` | one `Aligner` per thread; the object is still usable |
| `RuntimeError: GPU runtime failure during align() ...` | a CUDA error or GPU allocation failure during the run | this call produced **no results** and this `Aligner` refuses further work: build a new `Aligner`, or continue with `use_gpu=False` (recipe in §8). Its GPU memory is released at the moment of failure, so a new GPU `Aligner` can be built at once (on 0.1.7 only: `del` the failed one first, or the new one can fail on its next call). If building the new GPU `Aligner` raises (`GPU initialisation error`; reason on stderr), continue on the CPU; the GPU may be unusable for this process, and if it stays so, restart the process |
| `RuntimeError` (other) | construction failed (e.g. `"GPU initialisation error; see stderr"`) or out of memory building a batch | not retryable as is |
| `TypeError` | calling `built_with_cuda` as a function; a malformed `start` spec in the file helpers | fix the call |

**Per-read soft failures (the batch survives):** the read is returned with a
non-zero `status` and empty arrays. `status` is a bitwise OR of:

| flag | value | meaning |
|---|---|---|
| `FAILED_CALIBRATION` | `0x001` | calibration failed |
| `FAILED_ALIGNMENT` | `0x002` | alignment failed |
| `FAILED_QUALITY_CHK` | `0x004` | quality check failed |
| `FAILED_ALLOC` | `0x008` | a per-read allocation failed (library path; does not exit) |

Always check `d["status"] == 0` before trusting a read's rows or scalars. A few
percent of real reads fail alignment; that is normal.

On the GPU the engine writes one progress line per batch to stderr
(`[align_cuda] Load ...`); those lines are informational, not errors.

---

## 6. DTW primitive (`krill.dtw_*`) — separate facility

Computes DTW **distances between raw signal segments** (read-vs-read), e.g. for
clustering per-position signals in modification detection. It does NOT align
signal to a reference and is unrelated to `Aligner`.

Backends: GPU (vendored cuDTW++ warp-shuffle kernel `krill._dtw`, only on CUDA
builds) or CPU (always present, multithreaded). Same local-cost definition
(squared-euclidean, 3-neighbour min, final sqrt). Divergence point: signals
longer than **2047** samples are resampled to 2047 on GPU (scipy, lossy) but run
full-length on CPU, so CPU/GPU results differ above 2047.

Availability / selection helpers:

```python
krill.dtw_backend()             # "gpu" iff extension built AND a device present, else "cpu"
krill.dtw_available()           # bool: GPU DTW usable right now
krill.dtw_extension_available() # bool: cuDTW++ compiled in (ignores runtime device)
krill.dtw_cleanup()             # effectively a no-op (each DTW call frees its own resources); does NOT reset CUDA
```

`use_gpu` contract for all DTW functions: `None` = auto (GPU iff device present),
`True` = **require** GPU (raises `RuntimeError` if extension missing or no
device — note this DIFFERS from `Aligner`, which falls back to CPU), `False` =
CPU. `num_threads<=0` = auto core count (CPU only; ignored on GPU).

Functions (all return float64; matrices are symmetric with zero diagonal):

```python
krill.dtw_distance(seq1, seq2, use_gpu=None) -> float
    # two 1-D signals.

krill.dtw_pairwise(sequences, use_gpu=None, num_threads=0) -> ndarray (N, N)
    # sequences: (N, L) array, all EQUAL length L.

krill.dtw_pairwise_varlen(signals, use_gpu=None, num_threads=0) -> ndarray (N, N)
    # signals: list of N 1-D arrays of possibly DIFFERENT lengths.

krill.dtw_multi_position_pairwise(position_signals, use_gpu=None,
                                  num_streams=16, device_id=0, num_threads=0)
    -> list[ndarray]   # one (n_p, n_p) matrix per position
    # position_signals[p][r] = signal for position p, read r.
```

Low-level CPU kernels are also exposed but the wrappers above are preferred:
`krill._krill.dtw_distance_cpu(seq1, seq2)`,
`krill._krill.dtw_pairwise_cpu(seqs_2d, num_threads=0)`.

---

## 7. Convenience file helpers (optional, need `pyslow5` + `pyfastx`)

These read slow5/blow5 + fastq for you, then call `Aligner.align`. The core
engine needs no files; use these only when you have on-disk inputs.

```python
# all reads in a fastq that exist in the blow5
krill.align_blow5(blow5_path, fastq_path, pore="rna004", use_gpu=None,
                  batch_size=0, num_thread=0, cuda_dev_id=0) -> list[dict]

# ONE read's signal vs MANY candidate sequences
krill.align_read_variants(blow5_path, read_id, sequences, pore="rna004",
                          use_gpu=None, num_thread=0, cuda_dev_id=0,
                          aligner=None, start=None) -> list[dict]
    # sequences: str | list[str] | {label: seq}. Each result gets "variant_label".
    # start: int | {label: offset} | list[int]  (added to position).

# MANY reads x MANY candidates, flattened into ONE align() call (fastest)
krill.align_reads_variants(blow5_path, reads_variants, pore="rna004",
                           use_gpu=None, num_thread=0, cuda_dev_id=0,
                           aligner=None, start=None, batch_size=0)
    -> {read_id: list[dict]}
    # reads_variants: {read_id: sequences} or iterable of (read_id, sequences).
    # start: {read_id: start_spec}. Reads whose signal is missing are dropped.
```

Pass `aligner=<existing Aligner>` to `align_read_variants` /
`align_reads_variants` whenever you call them more than once; without it every
call builds a new `Aligner` (and on the GPU re-allocates its memory). Results of
`align_reads_variants` also carry `read_id`.

A read whose signal is missing from the file: `align_read_variants` raises
`KeyError`; `align_reads_variants` and `align_stream` leave it out, so compare
the returned read ids with the ones you asked for. `align_blow5` leaves it out
too, but its results carry no read id.

Streaming driver, for workloads too large to hold in memory (many clusters ×
many reads × several candidates). It reads the workload lazily, keeps a bounded
number of chunks in flight and yields results as chunks finish:

```python
krill.align_stream(blow5_path, workload, *, pore="rna004", lanes="auto",
                   chunk_pairs=2048, queue_depth=3,
                   gpu_num_thread=None, cpu_num_thread=None,
                   aligner_kwargs=None, gpu_aligner=None, cpu_aligner=None)
    -> iterator of (read_id, list[dict])
    # workload: any iterable/generator of (read_id, sequences) or
    #           (read_id, sequences, start); `sequences` / `start` as above.
    # "variant_label" is always a str: str(dict_key), or "0", "1", ... for a list.
    # Each result dict carries "variant_label", "read_id" and a "read_index" in
    # global input order. Groups are per read id: if the same read id appears in
    # several workload items, its results may arrive merged into one group or as
    # several groups. Identify results by "variant_label", not by group.
    # lanes: "auto" (ONE lane: GPU if usable, else CPU) | "gpu" | "cpu" | "gpu+cpu".
    # aligner_kwargs: forwarded to the Aligners it builds (hmm_confidence, polya,
    #                 batch_size, cuda_mem_frac, ...); not pore/use_gpu/num_thread.
    # gpu_aligner / cpu_aligner: use your own Aligner for that lane instead (it can
    #                 be reused across calls; aligner_kwargs and *_num_thread do not
    #                 apply to it). Pass lanes="gpu" or lanes="cpu" explicitly with it.
    # chunk_pairs: (read, candidate) pairs per align() call; a read's candidates
    #              are never split, so one huge item makes one huge chunk.

krill.align_all(blow5_path, workload, **same_keywords) -> {read_id: list[dict]}
    # collects align_stream; only for workloads whose results fit in memory.
```

If you stop iterating early, close the generator (`contextlib.closing(...)`) so
its threads and the file handle are released. An exception from `align()` (§5,
including `RuntimeError: GPU runtime failure ...`) is re-raised unchanged by the
iterator. If it raises, results already
yielded are complete and correct, but work it had read ahead is lost: restart
with every item you have not saved, not by continuing the same iterator.

---

## 8. Recipes

Single read, check success:
```python
a = krill.Aligner(pore="rna004", use_gpu=None)   # auto GPU/CPU
d = a.align({"read_id": rid, "sequence": seq, "signal": sig.astype("float32"),
             "digitisation": dig, "offset": off, "range": rng, "sample_rate": sr})[0]
if d["status"] == 0 and d["position"].size:
    means = d["event_level_mean"]          # numpy float32
    kmers = d["model_kmer"]                # list[str]
```

Batch (preferred for throughput — one call, many reads):
```python
results = a.align([read_dict_1, read_dict_2, ...])  # list aligned to input order
ok = [r for r in results if r["status"] == 0]
```

One read vs candidate transcripts (variant scoring):
```python
res = krill.align_read_variants("reads.blow5", "read-uuid",
        {"refA": seqA, "refB": seqB}, pore="rna004", start={"refA": 1, "refB": 5})
for r in res:
    if r["status"] == 0:                   # a failed candidate has empty arrays
        print(r["variant_label"], r["position"][0])
```

Many reads × several candidate sequences each, across many clusters (bounded
memory, one process, no competition for the GPU):
```python
import contextlib

def workload(clusters):          # {cluster_id: {read_id: {name: sequence}}}
    for cluster_id, reads in clusters.items():
        for read_id, candidates in reads.items():
            # carry the cluster in the label; it comes back as variant_label
            yield read_id, {f"{cluster_id}|{name}": seq
                            for name, seq in candidates.items()}

stream = krill.align_stream("reads.blow5", workload(clusters), pore="rna004",
                            aligner_kwargs=dict(hmm_confidence=False))
with contextlib.closing(stream):
    for read_id, results in stream:
        for r in results:
            if r["status"] == 0:
                cluster_id, name = r["variant_label"].split("|", 1)
                save(cluster_id, read_id, name, r)       # persist as you go
```

Long run in chunks, surviving a GPU failure:
```python
aligner = krill.Aligner(pore="rna004")
for chunk in chunks:                         # each chunk: a list of read dicts
    try:
        results = aligner.align(chunk)
    except RuntimeError as e:
        if "GPU runtime failure" not in str(e):
            raise
        aligner = krill.Aligner(pore="rna004", use_gpu=False)
        results = aligner.align(chunk)       # redo this chunk on the CPU
    save(results)
```

DTW clustering of per-position signals (CPU):
```python
mats = krill.dtw_multi_position_pairwise(position_signals, use_gpu=False)
```

---

## 9. Gotchas checklist (do all of these)

- [ ] `signal` must be **1-D** and float32, raw ADC (not pA), no NaN/inf.
- [ ] Provide all four scalars from the read's file metadata; they must be > 0
      (offset only finite). Garbage scalars → garbage scaling, no error if finite/positive.
- [ ] `align()` always returns a **list**, even for a single-dict input.
- [ ] Always check `result["status"] == 0` before trusting rows.
- [ ] `reference_kmer` / `model_kmer` / `hmm_state` are **lists of str**, not arrays.
- [ ] `standardized_level` is **NaN** on HMM 'B' rows (only present when
      `hmm_confidence=True`) — mask before stats.
- [ ] `hmm_state` is `'M'`/`'B'`/`'K'` with HMM on, but `'M'`/`'E'` in raw mode
      (`hmm_confidence=False` or HMM fallback) — don't assume it's empty.
- [ ] `'B'` rows: `model_mean`/`model_stdv` are 0.0 placeholders (not real).
- [ ] `'K'` rows share signal indices with a neighbour — filter them before
      slicing `signal[start_idx:end_idx]` or you double-count samples.
- [ ] `built_with_cuda` is an attribute; `gpu_device_count()` is a call.
- [ ] `Aligner(use_gpu=True)` **falls back to CPU** (warns) if no GPU; DTW
      `use_gpu=True` **raises** instead — different contracts.
- [ ] Output k-mers/positions are read-relative (self-alignment), NOT genome coords.
- [ ] `polya=True` works for **both** pores. RNA002 uses nanopolish's verbatim
      constants; RNA004 uses constants learned offline (EM bootstrap + affine
      calibration against dorado `pt:i`, held-out MAE ≈ 4 nt). The poly(A) keys
      appear only when `polya=True`, and `polya_length` is trustworthy only when
      `polya_qc == "PASS"`.
- [ ] Reuse one `Aligner` across many `align()` calls; construction loads the model.

---

## 10. Rules for fast and safe use

Measured on real RNA004 reads (RTX 3090, 16 CPU cores); numbers and method in
[BEST_PRACTICES.md](https://github.com/loganylchen/krill-dist/blob/main/BEST_PRACTICES.md)
([中文](https://github.com/loganylchen/krill-dist/blob/main/BEST_PRACTICES.zh.md)).

1. **Batch.** Pass many reads to ONE `align()` call. One call per read is about
   9× slower; a call with one read has nothing to parallelise.
2. **Candidates.** For reads × candidate sequences use `align_reads_variants`
   (fits in memory) or `align_stream` (does not). One call per read is 3.6×
   slower, one call per `(read, candidate)` 11× slower.
3. **One `Aligner`, reused; one per thread.** Never build an `Aligner` inside a
   loop. Do not start several processes that each build a GPU `Aligner` on the
   same device: each takes 0.9 of the free GPU memory. Use one process and one
   stream instead, or set `cuda_mem_frac`.
4. **Output mode.** `hmm_confidence=False` is 2.5–3.4× faster and returns the raw
   alignment (`M`/`E`). Keep the default only if you need the HMM re-decode and
   its `B`/`K` flags. Do not mix the two modes in one analysis.
5. **Threads.** Leave `num_thread=0` unless you run several `Aligner`s or
   processes at once (then make the totals add up to the core count), or run in
   Docker without `--cpuset-cpus` (then pass it explicitly).
6. **Streams.** Keep `lanes="auto"`. `"gpu+cpu"` only pays off with many spare
   cores and makes results depend on timing.
7. **Validate before batching, check after.** One invalid read fails the whole
   call (§5); every result needs `status == 0`.
8. **Save as you go** in long runs, so a failure costs one chunk (§8).
9. **Reproducibility.** CPU results do not depend on batching or thread count.
   On the GPU a read's result can depend on the batch it shares, and CPU and GPU
   differ slightly on a few percent of reads (per-read scaling by about 0.1% at
   most; some events assigned to a neighbouring k-mer). Never compare a CPU run
   with a GPU run read by read; for identical output from run to run, use the CPU.
