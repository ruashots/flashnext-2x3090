# Qwen3.8-Flash-Next on 2x RTX 3090s

Qwen released Qwen3.8-Flash-Next on August 26, 2026. It is a 125B MoE with a 51B n-gram lookup table, and most of the published ways to run it assume hardware well above what most people have at home. The two community builds aimed at 3090-class cards both start at four RTX 3090s.

I wanted to know what happens with two.

Turns out, it runs.

|                          |            Result |
| ------------------------ | ----------------: |
| UD-IQ4_XS decode         |       ~38-40 tok/s |
| UD-Q2_K_XL decode        |   up to ~53 tok/s |
| Prefill                  |  up to ~892 tok/s |
| Stable tested context    |              128k |
| Longest prompt completed |    119,482 tokens |
| Retrieval test           |       90/90 exact |
| GPUs                     | 2x RTX 3090 24 GB |
| RAM                      |             64 GB |

The main trick is not the quant. It is where the model lives.

The 28.8 GB n-gram table can stay on the CPU and be memory-mapped from NVMe, while the routed experts are split across the GPUs and host RAM in a way that does not leave half of one 3090 sitting empty.

The result is about 38 to 40 tok/s with UD-IQ4_XS and up to 53 tok/s with UD-Q2_K_XL. It stays stable through 32k context, and a separate long-context setup completed a 119,482-token prompt.

It is still slower than Qwen3.8-27B on the same machine. The 27B sits entirely in VRAM and runs around 85 to 100 tok/s here. I tried to measure whether Flash-Next gives enough quality back to justify that gap, but the test was too small to answer it. The speed numbers are solid. The quality comparison is not.

I first measured all of this against the pull request branch, the day before support merged. After it landed in master I rebuilt and reran everything, and the numbers here are from master. What the earlier build looked like, and what changed, is at the end.

Everything in this repo was measured on the machine below. The benchmark scripts, raw results and server logs are included.

## Hardware

| Component         | Hardware                                                               |
| ----------------- | ---------------------------------------------------------------------- |
| GPUs              | 2x RTX 3090, 24,576 MiB each                                           |
| NVLink            | None                                                                   |
| PCIe              | Gen4 x16 per GPU                                                       |
| CPU               | Ryzen 7 9800X3D, 8C / 16T                                              |
| RAM               | 64 GB DDR5-4800                                                        |
| Test memory limit | 54 GiB, no swap                                                        |
| Model disk        | ADATA XPG S40G NVMe, PCIe Gen3 x4                                      |
| Driver            | 595.71.05                                                              |
| CUDA              | 13.0.88                                                                |
| llama.cpp         | upstream `master`, after [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) merged |
| Model             | `unsloth/Qwen3.8-Flash-Next-GGUF`, UD-IQ4_XS                           |

## Working 32k configuration

```bash
/opt/llama.cpp/build/bin/llama-server \
  -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --host 0.0.0.0 --port 8080 --alias flashnext \
  -ngl 99 -sm layer -fit off -c 32768 -fa on -ctk f16 -ctv f16 \
  -b 2048 -ub 2048 -t 8 --threads-batch 8 --jinja \
  -ot "^per_layer_token_embd\.weight$=CPU" \
  -ot "blk\.([0-8]|2[5-9]|3[01])\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU"
```

At load:

```text
CPU_Mapped model buffer size = 40048.08 MiB
CPU_Mapped model buffer size =  8872.03 MiB
CUDA0 model buffer size      = 20277.75 MiB
CUDA1 model buffer size      = 21381.88 MiB
```

The two `-ot` overrides are what make this configuration work.

The first puts the n-gram table on the CPU. The second puts two bands of routed experts on the CPU, one inside each GPU's layer range, instead of offloading a single block from the start of the model.

## Why the model fits at all

The GGUF is 93.67 GB, but 28.80 GB of that is the n-gram lookup table.

Reading the tensor table directly with `gguf-py`:

| Tensor group                                |  Quant |     Size |
| ------------------------------------------- | -----: | -------: |
| `per_layer_token_embd.weight`, 51.2B params | IQ4_NL | 28.80 GB |
| Everything else                             |  mixed | 64.87 GB |
| Total                                       |        | 93.67 GB |

The table size is exactly what the packing predicts:

```text
51.2B x 4.5 bits / 8 = 28.80 GB
```

That tensor is used as a lookup table, not as a normal matrix multiply. It can sit in host memory and be memory-mapped from NVMe instead of taking up GPU memory.

That leaves 64.87 GB of the actual model to split across 48 GiB of VRAM and the usable system RAM.

One useful detail is that the n-gram table stays 28.80 GB across the quants in the Unsloth repo. Moving from UD-IQ4_XS to UD-Q2_K_XL shrinks the expert pool, not the lookup table.

## Why I am not using `--n-cpu-moe`

The obvious way to do this is `--n-cpu-moe N`. It works, but it is a bad fit for two GPUs.

| Configuration    |    GPU 0 |                                GPU 1 |
| ---------------- | -------: | -----------------------------------: |
| `--n-cpu-moe 22` |          | does not load, card 1 wants 29.7 GiB |
| `--n-cpu-moe 32` |  5.3 GiB |                             23.2 GiB |
| Two expert bands | 20.3 GiB |                             21.4 GiB |

`--n-cpu-moe` removes routed experts from the first N layers. With a normal layer split, GPU 0 then gets a bunch of layers that have already had their heavy tensors removed, while GPU 1 gets the loaded end of the model.

With `--n-cpu-moe 32`, almost 19 GiB of VRAM on GPU 0 was doing nothing useful while the host was under more pressure.

The two-band split fixes that by moving expert layers out of both halves of the model.

It was not a small difference either. With `--n-cpu-moe 32`, NVMe reads went as high as 766 MB/s and both GPUs spent a lot of time around 7 to 25 percent utilisation. With the two-band setup, mean NVMe read during the measured phase dropped to 38 MB/s and decode improved by about 25 percent.

`-ts` can also improve the balance, but it steers more than just the expert placement, and the layers are not equal in size anyway. I had better results naming the tensors directly.

`--n-cpu-ffn` was suggested to me as an alternative and it does not work here, it will not even load. llama.cpp's own help says why: it keeps *dense* FFN weights on the CPU and points you at `--n-cpu-moe` for expert weights. Every layer in this model is MoE, so it has nothing to move.

`--tensor-read-lazy on` is worth having. It cut load time from 45 seconds to 20 and changed nothing else.

## Speed

These runs use 512 output tokens, a unique nonce on every request so prefix caching cannot help, and two 512-token generations before recording anything so the CPU expert pages are warm.

The 27B comparison is the same machine, prompts and harness, but runs in vLLM with the model entirely in VRAM.

### Decode

| Model                                      | ~125 tok |    ~4.2k |   ~16.5k |
| ------------------------------------------ | -------: | -------: | -------: |
| Flash-Next UD-IQ4_XS, 16 host layers       |     39.5 |     38.0 |     33.1 |
| Flash-Next UD-Q2_K_XL, same 16 host layers |     46.8 |     44.5 |     37.8 |
| Flash-Next UD-Q2_K_XL, 8 host layers       | **53.3** | **50.8** | **42.1** |
| Qwen3.8-27B W4A16-AWQ, vLLM                |     96.2 |     99.4 |     84.7 |

### Prefill

| Model                                      |   ~4.2k |  ~16.5k |
| ------------------------------------------ | ------: | ------: |
| Flash-Next UD-IQ4_XS, 16 host layers       |     615 |     725 |
| Flash-Next UD-Q2_K_XL, same 16 host layers |     676 |     778 |
| Flash-Next UD-Q2_K_XL, 8 host layers       | **816** | **892** |
| Qwen3.8-27B W4A16-AWQ, vLLM                |    1560 |    1497 |

The Q2 build gets two rows because running both quants with the same CPU offload would not be a useful comparison.

UD-IQ4_XS has 55.43 GiB of routed experts. UD-Q2_K_XL has 42.92 GiB. The smaller one only needs eight expert layers on the host instead of sixteen, so it gets to use more of the GPUs.

Leaving it on the sixteen-layer split wastes about 8 GiB of VRAM and makes the CPU do eight extra expert passes per token for no reason.

## Long context

Long prompts need a different setup. The 32k configuration does not leave enough VRAM for the buffers built while processing something that large, so I moved 22 expert layers to the CPU and dropped `-ub` from 2048 to 1024, then launched at `-c 131072`.

That loads at 21.8 and 22.5 GB across the two cards. Walking the prompt length up:

| Prompt tokens |     Result | Prefill |
| ------------: | ---------: | ------: |
|        39,893 |         ok | 398 tok/s |
|        65,165 |         ok | 429 tok/s |
|        89,786 |         ok | 411 tok/s |
|   **119,482** |     **ok** | 378 tok/s |

Decode on that configuration is 35.8 tok/s at short prompts and 21.9 tok/s at 64k, with prefill at 64k of 413.

For comparison, the 27B on vLLM does 82.4 tok/s decode and 1225 tok/s prefill at 64k, with 54 seconds to first token against about 160 here.

## A configuration can load fine and still OOM later

The compute buffer grows with the prompt actually being processed.

That matters because a configuration can load, answer short prompts and look completely stable, then die halfway through a long one.

**This is much better on master than it was when I started.** These three failures are from the pull request branch, before the merge. On master the first of them runs its whole 32k window without complaint. I am keeping them because the lesson holds and because it is why the long-context setup above looks the way it does:

| Expert bands  | ubatch |       Died at | Allocation requested |
| ------------- | -----: | ------------: | -------------------: |
| `0-8 + 25-31` |   2048 | 17,849 tokens |             1599 MiB |
| `0-9 + 25-32` |   2048 | 26,624 tokens |             2088 MiB |
| `0-9 + 25-32` |   1024 | 59,392 tokens |             1976 MiB |

The sparse-attention inputs scale with both cached tokens and ubatch, so `-ub` is one of the useful knobs when pushing context.

Halving it bought a lot of room.

I would not tune this right to the edge though. In the last crash, GPU 1 reported 2328 MiB free and still failed a 1976 MiB allocation after repeated buffer growth. Fragmentation appears to matter.

The working configurations here leave roughly another 1 GiB per card instead of treating the load log as proof that the model fits.

## 32k retrieval

I also wanted to make sure "32k works" meant more than "the server did not crash."

This one was run on the pull request branch and I have not repeated it on master. Sparse attention did not change in the merge, so I expect it still holds, but it is the one measurement here that is not from the current build.

The retrieval test uses five maintenance records inside the same prompt. Each one gets a different sensor name and random 8-digit value, and they are placed at five different depths. The model is then asked for one by name.

The other four records are distractors, so this is a little harder than putting one weird sentence in a wall of filler and asking the model to repeat it.

Six fresh trials per cell, greedy decoding:

| Prompt |  5% | 25% | 50% | 75% | 95% |
| ------ | --: | --: | --: | --: | --: |
| 1,024  | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| 16,384 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| 32,768 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |

90 out of 90 were exact.

No corrupted digits, wrong records or missing answers.

That does not prove retrieval is perfect. Six trials in one cell is still a tiny sample. What I am comfortable saying is that retrieval was not breaking at 32k in this test.

## Sparse attention is actually running

Good retrieval does not prove sparse attention is active. If the model silently fell back to dense attention, retrieval might look just as good.

So I checked the graph instead.

With `GGML_SCHED_DEBUG=2`, the scheduler shows these nodes executing:

```text
indexer_top_k-3
indexer_top_k-7
indexer_top_k-11
indexer_top_k-15
indexer_top_k-19
indexer_top_k-23
indexer_top_k-27
indexer_top_k-31
indexer_top_k-35
indexer_top_k-39
indexer_top_k-43
indexer_top_k-47
```

That is every fourth layer across the 48-layer model, which matches the documented three Gated DeltaNet layers followed by one sparse-attention layer.

The model metadata also contains:

```text
qwen4exp.attention.indexer.top_k = 2048
```

So the selection path is being built and executed. I am not inferring it from speed.

## Prefill gets stranger at long context

A single prefill tok/s number is not very useful here because there is a fixed request cost and then a growing long-context cost on top.

These five points are from the pull request branch. Master is faster across the board, so read the shape rather than the values. On the 22-host-layer, `-ub 1024` configuration:

| Prompt tokens |     Prefill | Wall time |
| ------------: | ----------: | --------: |
|         2,003 | 162.3 tok/s |   12.34 s |
|         8,130 | 354.4 tok/s |   22.94 s |
|        16,479 | 387.2 tok/s |   42.56 s |
|        32,788 | 443.6 tok/s |   73.91 s |
|        65,470 | 386.6 tok/s |  169.35 s |

Read only as throughput, it rises and then falls.

A quadratic wall-time fit matched the points better than a simple fixed-overhead plus linear-throughput fit. At 65,470 tokens, the quadratic term accounts for about 55 of the 169 seconds.

The likely source is the indexer. Sparse attention only attends to the selected positions afterwards, but the indexer still has to score the cached positions before it can choose those 2048.

Five data points is not enough to turn that fit into some universal performance model, but it is enough to show why one prefill number at 4k tells you very little about what happens at 64k.

## UD-IQ4_XS vs UD-Q2_K_XL

Q2 is faster on this hardware because its expert pool is smaller.

|            | Routed experts |
| ---------- | -------------: |
| UD-IQ4_XS  |      55.43 GiB |
| UD-Q2_K_XL |      42.92 GiB |

That lets Q2 keep eight more expert layers on the GPUs, and it is worth about 14 tok/s: 53.3 against 39.5 at short prompts.

There is a quality tradeoff I did not measure well enough to quantify. UD-Q2_K_XL is the first tier here that starts reducing the attention projections. They are Q8_0 in UD-IQ4_XS and UD-Q3_K_XL, while 59 of 84 are Q5_K in UD-Q2_K_XL.

The sparse-attention indexer tensors stay BF16/F32 across the quants.

So 43 tok/s is a real speed result. I am not claiming it comes free.

## Flash-Next vs Qwen3.8-27B

I ran a small quality comparison because this is the obvious question after seeing the speed difference.

It did not answer it.

There were twenty prompts. Six had one defensible answer and were graded mechanically. Fourteen went through blind A/B comparison.

Both models scored 5/6 on the objective set.

The blind comparison ended at six wins each, with eight prompts where there was no useful separation.

That is nowhere near enough data to claim parity. Six wins out of twelve decisive comparisons is consistent with a very wide range of actual win rates. Detecting something like a real 60/40 difference would need roughly 194 decisive comparisons at 80 percent power.

So I am leaving the result where it belongs: inconclusive.

Two prompts did produce measurable differences. One SQL rewrite was 20x slower than the query it replaced. In another test, one model found all three dirty Git repositories in a fixture while the other found two.

Interesting, but still anecdotes.

What was more noticeable is that both models shared some of the same bad habits: confidently inventing mechanisms, shipping dead code and occasionally leaving "Wait" or "Actually" style false starts in the answer even with reasoning disabled.

## Sampling footgun

The GGUF contains:

```text
general.sampling.top_k = 20
general.sampling.top_p = 0.95
general.sampling.temp  = 1.0
```

Those match the model card's thinking-mode sampler.

If you run with reasoning disabled and do not set your sampler explicitly, llama.cpp can pull those values from the file.

For the comparison I set:

```text
temperature = 0.7
top_p = 0.8
```

There was still a difference between engines. llama.cpp used `top_k 20` from the GGUF and its default `min_p 0.05`, while vLLM used neither. Neither side used the recommended non-thinking presence penalty of 1.5.

That is another reason I would not read much into the quality comparison.

## Why not vLLM

I normally prefer vLLM, so I tried to make it work first.

On this hardware there are two separate blockers.

The first is RAM. The published W4A16 builds keep the 51.2B n-gram table in BF16 on the host:

```text
51.2B x 2 bytes = 102.4 GB
```

This machine has 64 GB.

The second is the expert weights. The routed experts are most of the 125B main model, and the W4A16 Marlin path on sm_86 does not give us the kind of CPU expert offload that makes the GGUF setup possible here. There is also no 2-bit or 3-bit MoE kernel for sm_86 that turns one of the smaller quants into a solution.

That is why the community W4A16 builds list four 3090s around the floor.

For two 3090s, llama.cpp is the thing that currently works.

## Benchmark traps I hit

A couple of these wasted enough time that they are worth writing down.

### Loading is not enough

This configuration loaded at:

```text
GPU0: 23310 MiB
GPU1: 23018 MiB
```

It answered short prompts too.

Then at 17,849 tokens:

```text
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1599.01 MiB on device 1
cudaMalloc failed: out of memory
Segmentation fault
```

Test a real long prompt before calling a configuration stable.

### llama.cpp can retry with a smaller batch

If the requested batch does not fit, llama.cpp can halve it and continue.

The request succeeds, but the number you measured is no longer for the batch size you thought you were testing.

I grep every server log for:

```text
retrying with smaller batch size
Compute error
Context size has been exceeded
failed to allocate graph
cudaMalloc failed
out of memory
```

I also confirmed `GGML_CUDA_ENABLE_UNIFIED_MEMORY` was unset. If it is enabled, an allocation that should fail can spill into managed memory and turn into slow PCIe paging instead.

### Prefix caching completely destroys prefill measurements

My first 16k prefill result was 10,917 tok/s.

The real number was 1,497.

I had repeated the same prompt and llama.cpp was serving most of it from prefix cache.

Every benchmark request here gets a unique nonce at the start of the prompt.

Putting it at the end is not enough because the prefix can still match.

### Warm the CPU experts

With part of the expert pool sitting in host page cache, short prompts do not necessarily warm all the pages decode will touch.

That originally made some short-context decode numbers look slower than longer-context ones.

Two 512-token generations before recording fixed it.

It also changed which ubatch looked best. While the machine was paging, `-ub 512` looked better than `-ub 2048`. Once the placement was fixed, `-ub 2048` almost doubled prefill and did not hurt decode.

## So, would I actually use it?

For chatting, yes. 38 to 53 tok/s is completely usable, and the model itself is very good.

I was honestly surprised by how strong it felt for something running partly out of my system RAM and SSD.

I am still keeping the 27B as my daily driver.

The issue is not whether Flash-Next runs. It does. The issue is that once you put it inside an agent loop, every call is roughly two and a half times slower at the same quant, and that stacks up quickly. Longer contexts make the gap worse.

The quality test here was not good enough to tell me how much intelligence I am getting back for that speed loss.

Using both models makes me think there is something there. Just not enough for how I use them.

If I wanted a strong local model mostly for conversation, I would have no problem running Flash-Next like this. For coding agents and repeated tool loops, I would take the 27B.

## Things that could make this much better

More VRAM is the obvious one. Every expert layer moved back to the GPU removes CPU compute, transfer and synchronisation from every token. Published results with all of the experts resident are much faster.

The llama.cpp support is still very new. It merged the day I was testing and got about 20 percent faster in the process, which is not a curve that has flattened yet.

There is also a graph-reuse patch showing large gains when all experts are resident on GPU. With CPU-offloaded experts, its author measured only about a 3.3 percent improvement. My own run-to-run spread is as high as 8 percent, so I did not bother publishing a number for something this harness could not resolve.


## What the day-one build looked like

I measured everything first against the pull request branch at commit `6c5afc8`, the day before [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) merged. Then it merged, I rebuilt from master, and reran it. Same box, same GGUF, same configs.

| decode, tok/s | day one | master |
| ------------- | ------: | -----: |
| UD-IQ4_XS, ~125 tok | 32.9 | **39.5** |
| UD-IQ4_XS, ~4.2k | 32.9 | **38.0** |
| UD-IQ4_XS, ~16.5k | 30.6 | **33.1** |
| UD-Q2_K_XL 8 host layers, ~125 tok | 43.2 | **53.3** |
| UD-Q2_K_XL 8 host layers, ~16.5k | 38.7 | **42.1** |

Two other things changed. The 32k configuration used to segfault at 17,849 tokens and now runs its whole window. And the longest prompt I could complete went from 65,682 tokens to 119,482.

Load time also dropped from 115 seconds to 45, and to 20 with `--tensor-read-lazy on`.

One OOM path still ends in a segfault rather than a handled error, because a graph allocation failure is not checked where it should be. Filed as [#27817](https://github.com/ggml-org/llama.cpp/issues/27817).

## Reproducing it

Everything used for this write-up is in the repo.

| File                   | What it does                                                                    |
| ---------------------- | ------------------------------------------------------------------------------- |
| `scripts/bench.py`     | Speed harness, nonce per request, expert warm-up, decode and prefill separately |
| `scripts/quality.py`   | Fixed 20-prompt comparison set                                                  |
| `scripts/autograde.py` | Grades the objective items                                                      |
| `scripts/qsa_probe.py` | Prefill ladder and long-context retrieval test                                  |
| `scripts/sysmon.sh`    | GPU, RAM and NVMe telemetry once per second                                     |
| `scripts/run_ot.sh`    | Starts a server with a selected expert split                                    |

Raw results and server logs are in `raw/`.

Every completed log was checked for:

```text
retrying with smaller batch size
Compute error
Context size has been exceeded
failed to allocate graph
cudaMalloc failed
out of memory
```

All six were zero on the completed configurations. The configuration that actually crashed contains its expected `cudaMalloc failed`, and its incomplete rows are not part of the reported results.

The main run also has the expected row counts: 54 speed rows and 60 quality rows.

If somebody finds a better split for two 3090s, pushes the context further, or gets vLLM working on the same hardware without turning the machine into swap soup, I would genuinely like to see it.
