<!-- Generated from the krill docs (docs/best-practices.zh.md) by .github/scripts/export_dist_docs.py. Do not edit here. -->
# 最佳实践

English: [BEST_PRACTICES.md](BEST_PRACTICES.md) · Full documentation: <https://loganylchen.github.io/krill/zh/>

怎样使用 krill 才能跑得快、内存可控、出错时明确报错而不是悄悄给出错误结果。本页的
数字都是在同一台机器上实测的（RTX 3090、2000 条真实 RNA004 read，中位长度 716
碱基），进程被限制在 16 个 CPU 核上；唯一一个用全部 128 核测的数字会单独注明。你的
绝对数值会不同，重点是倍数关系。

> **版本说明。** `cuda_mem_frac`、按 CPU 亲和性取自动线程数、GPU 运行期出错不再
> 终止进程，这三项从 0.1.7 起才有。在 0.1.6 及更早版本上，只要进程运行在
> CPU 限额下就请显式传 `num_thread`，并把 GPU 运行期错误当作会终止进程来处理。

## 简要版

| 这样做 | 原因（实测） |
|--------|--------------|
| 把很多 read 交给**一次** `align()` 调用 | 每条 read 调一次要慢 **9 倍** |
| 候选序列打分用 `align_reads_variants` 或 `align_stream`，不要每个候选调一次 | 每条 read 调一次慢 3.6 倍，每个候选调一次慢 **11 倍** |
| 每个 `Aligner` 只构造一次并复用；一个线程一个 `Aligner` | GPU `Aligner` 在整个生命周期内占着显存，且 `align()` 不可重入 |
| 需要的就是原始对齐（不做 HMM 重解码、没有 `B`/`K` 标记）时设 `hmm_confidence=False` | GPU 上快 **3.4 倍**，CPU 上快 2.5 倍；输出是另一种对齐结果 |
| 组 batch **之前**先过滤掉不合法的 read | 一条不合法的 read 会让整次调用报错 |
| 对每个结果检查 `status == 0` | 失败的 read 会被返回，而不是被丢掉 |
| 内存装不下的工作量用 `align_stream`，保持默认的单通道 | 内存有界；第二条通道只有在空闲核很多时才划算 |
| 需要每次运行输出完全一致时，用 CPU 跑 | GPU 上一条 read 的结果可能取决于它所在的 batch；CPU 与 GPU 在百分之几的 read 上有差异 |

## 1. 批量调用

所有并行都发生在一次 `align()` 调用内部、在你传入的 read 之间。只传一条 read 的调用
没有可并行的东西。

```python
aligner = krill.Aligner(pore="rna004")      # 只构造一次

results = aligner.align(reads)              # 推荐：一次调用，传整个列表

# 不推荐：每条 read 调一次
# results = [aligner.align([r])[0] for r in reads]
```

| 600 条 read | 一次调用 | 每条 read 调一次 |
|-------------|----------|------------------|
| GPU | 2.65 秒 | 24.93 秒（9.4 倍） |
| CPU，16 线程 | 2.90 秒 | 26.75 秒（9.2 倍） |

## 2. 一条 read 对多个候选序列

每条 read 要和多个候选序列比对时，把所有 `(read, 候选)` 对摊平成一次调用。
`align_reads_variants` 替你做这件事；工作量大到内存放不下时，`align_stream` 以惰性
方式做同样的事（见[流式处理](https://loganylchen.github.io/krill/zh/streaming/)）。

```python
aligner = krill.Aligner(pore="rna004", hmm_confidence=False)

out = krill.align_reads_variants(
    "reads.blow5",
    {"read-A": {"iso1": seq_a1, "iso2": seq_a2},     # 标签 -> 完整序列
     "read-B": {"iso1": seq_b1, "iso3": seq_b3}},
    aligner=aligner)                        # 复用 Aligner
```

| 300 条 read × 4 个候选，GPU | 耗时 |
|------------------------------|------|
| `align_reads_variants`，一次调用 | 4.6 秒 |
| 每条 read 调一次 `align_read_variants` | 16.6 秒（3.6 倍） |
| 每个 `(read, 候选)` 调一次 | 51.3 秒（11 倍） |

`align_read_variants` 或 `align_reads_variants` 只要会被调用不止一次，就一定要传
`aligner=`。不传的话每次调用都会新建一个 `Aligner`，在 GPU 上每次都要重新分配显存。

## 3. 按需选择输出模式

`hmm_confidence=True`（默认）会用 profile HMM 对每条 read 重新解码，并标出低置信事件
（`B`）和 k-mer 跳过（`K`）。这一步在 CPU 上跑，是耗时的大头。`hmm_confidence=False`
跳过这一步，直接返回原始的带状对齐结果，状态为 `M`/`E`。

两种模式是不同的输出，并不是同样的行加不加标记的区别：在这份数据上，每条 read 两种
模式返回的行都不一样。一项分析只用一种模式，不要跨模式比较结果。

| 2000 条 read | `hmm_confidence=True` | `hmm_confidence=False` |
|--------------|-----------------------|------------------------|
| GPU | 236 条/秒 | 809 条/秒 |
| CPU，16 线程 | 202 条/秒 | 513 条/秒 |

## 4. GPU、CPU 与线程

**设备。** `use_gpu=None` 在有 GPU 时使用 GPU。请确认实际用的是什么：在 CPU-only 的
构建上要求 GPU 只会给出警告并回退到 CPU。

```python
print(krill.built_with_cuda, krill.gpu_device_count())
aligner = krill.Aligner(pore="rna004")
print(aligner.use_gpu, aligner.num_thread, aligner.batch_size)
```

GPU 在原始模式下帮助最大：`hmm_confidence=False` 时比 16 个 CPU 线程快 1.6 倍，默认
模式下快 1.2 倍。默认模式的大部分时间花在 CPU 侧，所以线程数比设备更重要。

**线程。** `num_thread=0` 使用进程被允许运行的 CPU。如果同时跑多个 `Aligner`（或多个
进程），请给每个都设 `num_thread`，让总数约等于核数。

**一个线程一个 `Aligner`。** 对同一个 `Aligner` 并发调用第二个 `align()` 会抛
`RuntimeError`。不同的 `Aligner` 可以并发运行。

**显存。** GPU `Aligner` 在构造时按比例预分配设备上**当时空闲**的显存（默认 0.9，在
24 GB 的卡上约 21.8 GB），并一直占用到被销毁。要共用一张卡，就设置这个比例：

```python
a = krill.Aligner(pore="rna004", cuda_mem_frac=0.3)   # 24 GB 中约 7 GB
b = krill.Aligner(pore="rna004", cuda_mem_frac=0.3)   # 剩余部分的 0.3
```

估算规则：引擎每个碱基的容量大约需要 1.6 kB 显存，一个 batch 需要
`batch_size × 平均 read 长度` 个碱基的容量。所以默认 2048 条、每条 1 kb 的 batch 大约
需要 3.3 GB。一个 batch 里放不下的 read 会改在 CPU 上对齐，不会报错。

## 5. 大工作量：用流式

`align_stream` 接受任意可迭代对象，所以你可以跨所有 cluster 逐个产出工作，而不必先
拼出一个巨大的列表。任何时刻在途的块数都有上限，所以内存随 `chunk_pairs` 和
`queue_depth` 增长，与工作量大小无关。

```python
import contextlib

def workload(clusters):
    """clusters: {cluster_id: {read_id: {候选名: 序列}}}"""
    for cluster_id, reads in clusters.items():
        for read_id, candidates in reads.items():
            # 把 cluster 放进 label 里；它会作为 variant_label 原样返回
            yield read_id, {f"{cluster_id}|{name}": seq
                            for name, seq in candidates.items()}

stream = krill.align_stream("reads.blow5", workload(clusters), pore="rna004",
                            aligner_kwargs=dict(hmm_confidence=False))
with contextlib.closing(stream):            # 提前停止时释放线程
    for read_id, results in stream:
        for r in results:
            if r["status"] == 0:
                cluster_id, name = r["variant_label"].split("|", 1)
                save(cluster_id, read_id, name, r)   # 边算边存
```

- **保持默认通道。** `lanes="auto"` 跑一条 GPU 通道（没有 GPU 时跑一条 CPU 通道）。
  16 核上加第二条通道反而更慢（`"gpu+cpu"` 162 条/秒，`"gpu"` 179 条/秒）；128 核上
  快 1.26 倍。只有空闲核很多时才用 `"gpu+cpu"`。
- **调小 `chunk_pairs` 可以省内存。** 在这份数据上，2048 占用约 8.5 GB 主机内存，256
  约 3 GB。
- **带几千个候选的 read 要拆开**，分成多个 workload 条目。同一条 read 的候选不会被拆到
  不同的块里，所以这样的一个条目会让一个块超过 `chunk_pairs`。

## 6. 检查输入，检查输出

一条不合法的 read 会让整次 `align()` 调用报错，所以组 batch 之前先过滤：

- 序列只含 `A`、`C`、`G`、`T`、`U`（大小写均可），且长度不小于 k-mer（RNA004 为 9，
  RNA002 为 5）；
- 信号是非空的一维数组，且所有值有限；
- `digitisation`、`range`、`sample_rate` 是有限的正数，`offset` 是有限值；
- `start` 如果给了，必须是整数，并且由它得到的位置（`start` 加上 read 内的偏移）不超出
  32 位范围；
- `init_scale` 和 `init_shift` 如果给了，必须成对出现，`init_scale` 是有限的正数，
  `init_shift` 是有限值；
- 必需的字段都在，且类型正确。

```python
import numpy as np

K = 9                                        # pore="rna002" 时为 5

def usable(read):
    seq, sig = read["sequence"], read["signal"]
    return (len(seq) >= K and set(seq.upper()) <= set("ACGTU")
            and sig.ndim == 1 and sig.size > 0 and bool(np.isfinite(sig).all())
            and all(np.isfinite(read[k]) and read[k] > 0
                    for k in ("digitisation", "range", "sample_rate"))
            and bool(np.isfinite(read["offset"])))

results = aligner.align([r for r in reads if usable(r)])
```

这个过滤函数覆盖了序列、信号和标定参数。如果你传了 `start` 或
`init_scale` / `init_shift`，请自己加上对应的检查。

然后检查每个结果。对不上的 read 会以非零 `status` 和空数组返回，它的标量字段没有意义。

```python
good = [r for r in results if r["status"] == 0]
```

`align_reads_variants` 和 `align_stream` 会略过 blow5 里找不到信号的 read，所以请把
返回的 read id 和你请求的对一下。`align_blow5` 也会略过，但它的结果里不带 read id。
`align_read_variants` 遇到这种情况则抛 `KeyError`。

## 7. 处理失败

- **`ValueError`** 表示某个输入值不合法，报错信息会指出是哪条 read。缺少字段会抛
  `KeyError`，字段类型不对（该是数字的地方给了字符串）会抛
  `RuntimeError: Unable to cast ...`，helper 在候选序列的写法不对时会抛 `TypeError`。
  这几种情况都要修正或丢弃出问题的输入；用同样的输入重试只会再次失败。
- **`RuntimeError: GPU runtime failure ...`** 表示这个 `Aligner` 的 GPU 状态已经失效。
  这次调用不返回任何结果，这个 `Aligner` 也拒绝继续工作。新建一个，或者改用 CPU 继续。
  从 0.1.8 起，出故障的 `Aligner` 在故障发生的当下就归还显存，所以可以马上新建 GPU
  `Aligner`。在 0.1.7 上，请先销毁出故障的那个（`del aligner`）：它还存在时新建的 GPU
  `Aligner`，下一次调用可能会失败。如果新建 GPU `Aligner` 本身就报错
  （`GPU initialisation error`，原因见 stderr），请改用 CPU 继续。这时 GPU 对这个进程可能
  已经不可用；如果一直如此，请重启进程。

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
    save(results)                            # 一次失败只损失一个块
```

长时间的运行请像上面这样分块处理并边算边存，这样一次失败只损失一个块。`align_stream`
已经返回的结果不受影响，但它出错时，已经从你的 workload 里预读出来、还没返回的那部分
工作会丢掉。重启时请把所有还没存下结果的条目重新交给它，不要接着用原来的迭代器。

在 GPU 上，引擎每个 batch 会往 stderr 写一行进度（`[align_cuda] Load ...`），这些只是
信息，不是错误。

## 8. 可复现性

- 在 **CPU** 上，一条 read 的结果不取决于它和谁同批，也不取决于线程数。
- 在 **GPU** 上，一条 read 只有在相对所在 batch 的平均长度足够短时才走 GPU，否则走
  CPU。所以同一条 read 在不同的 batch 里可能走不同的路径。
- **CPU 与 GPU 的结果接近但不完全相同。** 逐字段比较时，300 条 read 里有 13 条不同。
  这 13 条的每条 read 缩放参数都有轻微差异（`scale` 的中位差异为 0.02%，最大 0.1%），
  由它推出的数值也随之不同；其中 7 条还有部分事件被分到了不同的 k-mer。

如果需要每次运行输出完全一致，最简单的办法是用 CPU 跑。在 GPU 上，要保持 batch 不变
（同样的调用、`batch_size` 和输入顺序，单通道），还要保持 `Aligner` 拿到的显存不变——
它取决于 `cuda_mem_frac` 以及构造时设备上有多少空闲显存。如果这种程度的差异可以接受，
就用最快的配置。

## 9. 其他功能

- **read 的切片。** 要让一条 read 的局部对齐与整条 read 的结果尽量吻合，把整条 read 的
  `init_scale` / `init_shift` 传进去
  （见[对一段 read 切片做对齐](https://loganylchen.github.io/krill/zh/parameters/#read)）。
- **poly(A)。** 用 `polya=True` 构造 `Aligner`，只在 `polya_qc == "PASS"` 时信任
  `polya_length`（见[输出列](https://loganylchen.github.io/krill/zh/output/)）。
- **DTW。** 用批量函数（`dtw_pairwise`、`dtw_pairwise_varlen`、
  `dtw_multi_position_pairwise`），不要在循环里调 `dtw_distance`。输入必须是有限值。
  超过 2047 个采样点时 GPU 会重采样而 CPU 不会，两种后端在这个范围内结果不同。

## 10. SLURM 与容器

- 自动线程数等于进程被允许运行的 CPU 数。被绑定到所分配核上的 SLURM 作业
  （`--cpus-per-task`）会得到正确的数字。
- Docker 容器不会继承这种绑定，CPU 配额（`docker --cpus=4`）krill 也看不到：在 128 核
  主机上一个 4 核的作业里，容器看到的是 128 个 CPU，加了 `--cpuset-cpus=0-3` 才是 4 个。
  请使用 `--cpuset-cpus` 或显式传 `num_thread`。
- 一块 GPU 只给一个进程用，并且在上面只跑一个 GPU `Aligner`，除非你设置了
  `cuda_mem_frac`。
