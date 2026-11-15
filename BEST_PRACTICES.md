<!-- Generated from the krill docs (docs/best-practices.en.md) by .github/scripts/export_dist_docs.py. Do not edit here. -->
# Best practices

中文版：[BEST_PRACTICES.zh.md](BEST_PRACTICES.zh.md) · Full documentation: <https://loganylchen.github.io/krill/>

How to drive krill so that it is fast, stays within memory, and fails loudly
instead of quietly. The numbers on this page were measured on one machine (RTX
3090, 2000 real RNA004 reads, median 716 bases) with the process limited to 16
CPU cores; the one figure taken with all 128 cores says so. Your absolute
numbers will differ; the ratios are the point.

> **Version note.** `cuda_mem_frac`, the affinity-aware automatic thread count
> and the non-fatal GPU failure handling are available from 0.1.7. On 0.1.6 and
> earlier, pass `num_thread` explicitly whenever the process runs
> under a CPU limit, and treat a GPU runtime error as fatal to the process.

## The short version

| Do this | Why (measured) |
|---------|----------------|
| Pass many reads to **one** `align()` call | one call per read is **9× slower** |
| Score candidates with `align_reads_variants` or `align_stream`, never one call per candidate | one call per read is 3.6× slower, one per candidate **11× slower** |
| Build each `Aligner` once and reuse it; one `Aligner` per thread | a GPU `Aligner` holds its GPU memory for life, and `align()` is not reentrant |
| Set `hmm_confidence=False` when the raw alignment (no HMM re-decode, no `B`/`K` flags) is what you want | **3.4× faster** on GPU, 2.5× on CPU; the output is a different alignment |
| Filter invalid reads **before** batching | one invalid read raises for the whole call |
| Check `status == 0` on every result | failed reads are returned, not dropped |
| Use `align_stream` with its default single lane for workloads that do not fit in memory | bounded memory; a second lane only pays off with many spare cores |
| Run on the CPU when you need identical output from run to run | on the GPU a read's result can depend on its batch; CPU and GPU differ on a few percent of reads |

## 1. Batch your calls

All parallelism happens inside one `align()` call, across the reads you pass in.
A call with a single read has nothing to parallelise.

```python
aligner = krill.Aligner(pore="rna004")      # build once

results = aligner.align(reads)              # GOOD: one call, the whole list

# BAD: one read per call
# results = [aligner.align([r])[0] for r in reads]
```

| 600 reads | one call | one call per read |
|-----------|----------|-------------------|
| GPU | 2.65 s | 24.93 s (9.4×) |
| CPU, 16 threads | 2.90 s | 26.75 s (9.2×) |

## 2. Reads with several candidate sequences

When each read is scored against several candidate sequences, flatten all
`(read, candidate)` pairs into one call. `align_reads_variants` does that for
you; `align_stream` does it lazily for workloads too large to hold in memory
([Streaming](https://loganylchen.github.io/krill/streaming/)).

```python
aligner = krill.Aligner(pore="rna004", hmm_confidence=False)

out = krill.align_reads_variants(
    "reads.blow5",
    {"read-A": {"iso1": seq_a1, "iso2": seq_a2},     # label -> full sequence
     "read-B": {"iso1": seq_b1, "iso3": seq_b3}},
    aligner=aligner)                        # reuse the Aligner
```

| 300 reads × 4 candidates, GPU | time |
|-------------------------------|------|
| `align_reads_variants`, one call | 4.6 s |
| `align_read_variants` once per read | 16.6 s (3.6×) |
| one call per `(read, candidate)` | 51.3 s (11×) |

Always pass `aligner=` when you call `align_read_variants` or
`align_reads_variants` more than once. Without it every call builds a new
`Aligner`, and on the GPU each one allocates its memory again.

## 3. Pick the output mode you need

`hmm_confidence=True` (the default) re-decodes every read with a profile HMM and
flags low-confidence events (`B`) and k-mer skips (`K`). That re-decode runs on
the CPU and dominates the run time. `hmm_confidence=False` skips it and returns
the raw banded alignment, with `M`/`E` states.

The two modes are different outputs, not the same rows with and without flags:
on this data set every read came back with a different set of rows. Choose one
mode per analysis and do not compare results across modes.

| 2000 reads | `hmm_confidence=True` | `hmm_confidence=False` |
|------------|-----------------------|------------------------|
| GPU | 236 reads/s | 809 reads/s |
| CPU, 16 threads | 202 reads/s | 513 reads/s |

## 4. GPU, CPU and threads

**Device.** `use_gpu=None` uses the GPU when one is available. Check what you
actually got: asking for a GPU on a CPU-only build warns and falls back.

```python
print(krill.built_with_cuda, krill.gpu_device_count())
aligner = krill.Aligner(pore="rna004")
print(aligner.use_gpu, aligner.num_thread, aligner.batch_size)
```

The GPU helps most in raw mode: 1.6× over 16 CPU threads with
`hmm_confidence=False`, 1.2× with the default. In the default mode most of the
time is CPU-side work, so threads matter more than the device.

**Threads.** `num_thread=0` uses the CPUs the process is allowed to run on. If
you run several `Aligner`s at once (or several processes), set `num_thread` on
each so that the total is about the number of cores.

**One `Aligner` per thread.** A second concurrent `align()` on the same
`Aligner` raises `RuntimeError`. Separate `Aligner`s can run concurrently.

**GPU memory.** A GPU `Aligner` pre-allocates a fraction of the device's *free*
memory when it is built (0.9 by default, about 21.8 GB on a 24 GB card) and keeps
it until it is destroyed. To share a device, set the fraction:

```python
a = krill.Aligner(pore="rna004", cuda_mem_frac=0.3)   # about 7 GB of 24 GB
b = krill.Aligner(pore="rna004", cuda_mem_frac=0.3)   # 0.3 of what is left
```

As a sizing rule, the engine needs roughly 1.6 kB of device memory per base of
capacity, and a batch needs capacity for `batch_size × mean read length` bases.
The default batch of 2048 reads of 1 kb therefore needs about 3.3 GB. Reads that
do not fit in a batch are aligned on the CPU instead; nothing fails.

## 5. Large workloads: stream them

`align_stream` takes any iterable, so you can generate work across all your
clusters without building one giant list. Only a bounded number of chunks is
in flight at any time, so memory grows with `chunk_pairs` and `queue_depth`, not
with the size of the workload.

```python
import contextlib

def workload(clusters):
    """clusters: {cluster_id: {read_id: {candidate_name: sequence}}}"""
    for cluster_id, reads in clusters.items():
        for read_id, candidates in reads.items():
            # carry the cluster in the label; it comes back as variant_label
            yield read_id, {f"{cluster_id}|{name}": seq
                            for name, seq in candidates.items()}

stream = krill.align_stream("reads.blow5", workload(clusters), pore="rna004",
                            aligner_kwargs=dict(hmm_confidence=False))
with contextlib.closing(stream):            # releases threads if you stop early
    for read_id, results in stream:
        for r in results:
            if r["status"] == 0:
                cluster_id, name = r["variant_label"].split("|", 1)
                save(cluster_id, read_id, name, r)   # persist as you go
```

- **Keep the default lane.** `lanes="auto"` runs one GPU lane (or one CPU lane
  without a GPU). On 16 cores a second lane was slower (162 reads/s for
  `"gpu+cpu"` against 179 for `"gpu"`); on 128 cores it was 1.26× faster. Opt
  into `"gpu+cpu"` only with many spare cores.
- **Lower `chunk_pairs` to save memory.** 2048 held about 8.5 GB of host memory
  on this data set, 256 about 3 GB.
- **Split reads that carry thousands of candidates** into several workload
  items. A read's candidates are never split across chunks, so one such item
  makes a chunk larger than `chunk_pairs`.

## 6. Validate the input, check the output

One invalid read makes the whole `align()` call raise, so filter before you
batch:

- the sequence contains only `A`, `C`, `G`, `T`, `U` (either case) and is at
  least as long as the k-mer (9 for RNA004, 5 for RNA002);
- the signal is a non-empty 1-D array of finite values;
- `digitisation`, `range` and `sample_rate` are finite and positive, and `offset`
  is finite;
- `start`, if given, is an integer, and the positions it leads to (`start` plus
  the offset within the read) fit in 32 bits;
- `init_scale` and `init_shift`, if given, are given together, with `init_scale`
  finite and positive and `init_shift` finite;
- every required field is present and has the expected type.

```python
import numpy as np

K = 9                                        # 5 for pore="rna002"

def usable(read):
    seq, sig = read["sequence"], read["signal"]
    return (len(seq) >= K and set(seq.upper()) <= set("ACGTU")
            and sig.ndim == 1 and sig.size > 0 and bool(np.isfinite(sig).all())
            and all(np.isfinite(read[k]) and read[k] > 0
                    for k in ("digitisation", "range", "sample_rate"))
            and bool(np.isfinite(read["offset"])))

results = aligner.align([r for r in reads if usable(r)])
```

This filter covers the sequence, the signal and the calibration values. Add
checks for `start` and `init_scale` / `init_shift` if you pass them.

Then check every result. A read that could not be aligned is returned with a
non-zero `status` and empty arrays; its scalars are not meaningful.

```python
good = [r for r in results if r["status"] == 0]
```

`align_reads_variants` and `align_stream` leave out a read whose signal is
missing from the blow5, so compare the returned read ids with the ones you asked
for. `align_blow5` leaves it out too, but its results carry no read id.
`align_read_variants` raises `KeyError` instead.

## 7. Handle failures

- **`ValueError`** means an input value is invalid; the message names the read.
  A missing field raises `KeyError`, a field of the wrong type (a string where a
  number is expected) raises `RuntimeError: Unable to cast ...`, and the helpers
  raise `TypeError` for a malformed candidate specification. In all of these,
  fix or drop the offending input; retrying the same input fails again.
- **`RuntimeError: GPU runtime failure ...`** means the GPU state of that
  `Aligner` is gone. The call returns no results and the `Aligner` refuses
  further work. Build a new one, or continue on the CPU:

```python
aligner = krill.Aligner(pore="rna004")
for chunk in chunks:
    try:
        results = aligner.align(chunk)
    except RuntimeError as e:
        if "GPU runtime failure" not in str(e):
            raise
        aligner = krill.Aligner(pore="rna004", use_gpu=False)
        results = aligner.align(chunk)
    save(results)                            # a failure costs one chunk
```

Process long runs in chunks and save as you go, as above, so a failure costs one
chunk. `align_stream` keeps what it has already returned, but work it had read
ahead from your workload is lost when it raises. Restart it with every item you
have not saved, not by continuing the same iterator.

On the GPU the engine writes a progress line per batch to stderr
(`[align_cuda] Load ...`). Those lines are informational.

## 8. Reproducibility

- On the **CPU**, a read's result does not depend on what it is batched with or
  on the thread count.
- On the **GPU**, a read runs on the GPU only if it is short enough relative to
  the mean length of its batch; otherwise it runs on the CPU. So the same read
  can be routed differently in a different batch.
- **CPU and GPU results are close but not identical.** With every output field
  compared they differed on 13 of 300 reads. All 13 differed slightly in the
  per-read scaling (`scale` by 0.02% in the median, 0.1% at most) and the values
  derived from it; on 7 of them some events were also assigned to a different
  k-mer.

If you need identical output from run to run, the simplest way is to run on the
CPU. On the GPU, keep the batches the same (the same calls, `batch_size` and
input order, a single lane) and also the GPU memory the `Aligner` gets, which
depends on `cuda_mem_frac` and on how much memory was free when it was built. If
differences of that size are acceptable, use whatever is fastest.

## 9. Other features

- **Slices of a read.** To align part of a read consistently with the full read,
  pass the full read's `init_scale` / `init_shift`
  ([Aligning a slice of a read](https://loganylchen.github.io/krill/parameters/#aligning-a-slice-of-a-read)).
- **poly(A).** Build the `Aligner` with `polya=True` and trust `polya_length`
  only where `polya_qc == "PASS"` ([Output columns](https://loganylchen.github.io/krill/output/)).
- **DTW.** Use the batched functions (`dtw_pairwise`, `dtw_pairwise_varlen`,
  `dtw_multi_position_pairwise`) rather than `dtw_distance` in a loop. Inputs
  must be finite. Above 2047 samples the GPU resamples and the CPU does not, so
  the two backends diverge there.

## 10. SLURM and containers

- The automatic thread count is the number of CPUs the process may run on. A
  SLURM job that is bound to its cores (`--cpus-per-task`) gets the right number.
- A Docker container does not inherit that binding, and a CPU quota
  (`docker --cpus=4`) is not visible to krill: in a 4-CPU job on a 128-core host
  a container saw 128 CPUs, and 4 with `--cpuset-cpus=0-3`. Use `--cpuset-cpus`
  or pass `num_thread`.
- Give each GPU to one process, and run one GPU `Aligner` on it unless you set
  `cuda_mem_frac`.
