# Benchmarks: Strata v0.1.40.2 on 2x RTX 3090

On Reddit I said I'd leave the machine benchmarking overnight and do retrieval and needle tests too. This is that run, from October 7, 2026: speed from a few hundred tokens up to 1M, one versus two requests at once, needle retrieval up to 1M, GSM8K and my 20-prompt set.

Everything was run three ways on the same box and the same model files:

| Name | Config | What it is |
| --- | --- | --- |
| **262K** | [`strata-orca-0402-262k-img4096.json`](setups/2026-10-strata-v0.1.40.2/strata-orca-0402-262k-img4096.json) | What I run every day. 262,144 context, two requests at once, images on |
| **512K** | [`strata-orca-0402-512k.json`](setups/2026-10-strata-v0.1.40.2/strata-orca-0402-512k.json) | The model stretched x2 with yarn, one request at a time |
| **1M** | [`strata-orca-0402-1m.json`](setups/2026-10-strata-v0.1.40.2/strata-orca-0402-1m.json) | The model stretched x4 with yarn, one request at a time |

Before the tables, what these numbers can and can't tell you:

- **Finding planted facts proves retrieval holds to 1M, not reasoning over 1M.** The needle tests ask the model to pull one fact back out of a long text. Passing them doesn't mean it can think across a whole million tokens.
- **The single runs past 128K are noisy.** Each length past 128K ran once, so one slow or fast run moves the number a lot. The 200K row writing slower than the 245K row is most likely that noise.
- **The cards run at a 225 W power cap**, not at the stock limit.
- **The 512K and 1M setups stretch the model with yarn to get past 262K, and serve one request at a time.** They're for when the length is needed, not for every day.
- **A non-streamed ~1M request once never got its reply.** The engine finished in 544 s, but the answer never reached the client in the 30 minutes after that. Streamed requests at the same length came back fine, so every 1M needle test ran streamed.
- **The server kept serving my agents during the whole day of runs.** Their requests landed in the middle of some measurements, so expect some noise in every row, and some numbers lower than a quiet machine would give.

## Speed

One request at a time. Prompt reading and writing in tokens per second, as the client sees them. Up to 128K it's the median of 6 runs (3 code, 3 reasoning), past that a single run.

| Setup | Prompt (tokens) | Reads the prompt (tok/s) | Writes (tok/s) |
| --- | ---: | ---: | ---: |
| 262K | 139 | - | 110 |
| 262K | 4,222 | 989 | 112 |
| 262K | 16,521 | 1,842 | 112 |
| 262K | 32,922 | 2,165 | 103 |
| 262K | 65,722 | 2,487 | 100 |
| 262K | 131,320 | 2,564 | 93 |
| 262K | 205,118 | 2,314 | 76 |
| 262K | 251,239 | 2,567 | 107 |
| 512K | 138 | - | 108 |
| 512K | 4,222 | 773 | 109 |
| 512K | 307,618 | 2,510 | 84 |
| 512K | 410,120 | 2,469 | 107 |
| 512K | 512,617 | 2,380 | 100 |
| 1M | 140 | - | 92 |
| 1M | 4,222 | 631 | 111 |
| 1M | 615,113 | 2,120 | 65 |
| 1M | 922,620 | 1,972 | 82 |
| 1M | 1,025,121 | 1,923 | 82 |

Reading a full 1M prompt takes about 9 minutes. Reading 250K takes about a minute and a half.

## One versus two requests at once

The 262K setup, two requests sent at the same moment, 600 tokens each. Median of 3 runs, the 32K line is one run.

| Prompt each (tokens) | First token, request 1 / 2 | Each writes | Both together |
| ---: | ---: | ---: | ---: |
| ~130 | 1.9 s / 1.0 s | 58 tok/s | 102 tok/s |
| ~4.2K | 7.0 s / 3.6 s | 56 tok/s | 74 tok/s |
| ~33K | 27.2 s / 13.6 s | 46 tok/s | 33 tok/s |

Two short chats together write about what one writes alone (102 against 110 tok/s), the gain is that the second one starts right away instead of waiting for the first to finish. With long prompts it gets worse, because Strata reads new prompts one after the other, and while it reads, the other request barely moves.

## Retrieval

**Needle:** Strata's own `tools/needle_bench.py`. One code word hidden in a long text made from Strata's docs and sources, at a set depth, and the model is asked for it.

| Setup | Length | 10% | 25% | 50% | 75% | 90% |
| --- | ---: | :-: | :-: | :-: | :-: | :-: |
| 262K | 4K to 256K, 7 lengths | found | found | found | found | found |
| 512K | 300K, 400K, 500K | found | | found | | found |
| 1M | 1M | found | | found | | found |
| 1M | 600K, 800K | | | found | | |

**Multi-key needle:** four key-and-number facts hidden in the same text, and the model is asked for one of them, so it has to pick the right one out of four look-alikes ([`scripts/needle_multikey.py`](scripts/needle_multikey.py)).

| Setup | Length | 10% | 50% | 90% |
| --- | ---: | :-: | :-: | :-: |
| 262K | 32K, 128K, 256K | found | found | found |
| 512K | 400K, 500K | | found | |
| 1M | 1M | | found | |

61 of 61 found, and in none of the multi-key tests did it answer with one of the other three numbers. The longest single test took about 9 minutes.

## Quality

| Test | 262K | 1M |
| --- | ---: | ---: |
| GSM8K, the full 1,319-problem test set | **1,271 / 1,319 = 96.4%** | first 250 only: 243 / 250 = 97.2% |
| My 20-prompt set, the 6 objectively graded prompts | 5 / 6 | 5 / 6 |

GSM8K is 0-shot, greedy, thinking off, scored on the exact final number ([`scripts/gsm8k.py`](scripts/gsm8k.py)). The 1M run only did the first 250 problems, so its score isn't comparable to the full set. Stretching the model to 1M didn't break the short answers, which is all the 1M column says.

## Method

- **Client side:** the scripts in [`scripts/`](scripts), talking to the server over its OpenAI-compatible API. Times are measured by the client, so they include the network and the server's own overhead.
- **Speed:** [`scripts/bench.py`](scripts/bench.py). Every prompt starts with a random nonce so nothing comes from a cache. Two warm-up generations first. 512 tokens max, temperature 0.7, top_p 0.95. "Writes" is the rate after the first token.
- **Two at once:** `scripts/probe.py concurrent`.
- **Needles:** the haystack is built from a plain clone of Strata at `v0.1.40.2`, so the text depends on that commit. Seeds are fixed, so the code words and numbers are the same every run. At 1M the needles went through [`scripts/needle_stream.py`](scripts/needle_stream.py), the same test with a streamed reply.
- **Versions:** Strata v0.1.40.2 (`e8ca9af`) built from source with CUDA 13.0 for sm_86, NVIDIA driver 595.71.05, OrcaRouter's uncensored IQ4_XS GGUF. The full host and container details and all three configs as they ran are in [`raw/strata-canonical-2026-10-07-env.txt`](raw/strata-canonical-2026-10-07-env.txt).

## Run it yourself

With the server up on the 262K config, and `STRATA` pointing at a plain clone of Strata checked out at `v0.1.40.2`:

```bash
URL=http://127.0.0.1:8080 STRATA=/path/to/Strata scripts/canonical_262k.sh
```

Then restart the server on the 512K config and run `scripts/canonical_long.sh 512k`, and on the 1M config run `scripts/canonical_long.sh 1m` and `scripts/canonical_1m_rest.sh`, with the same `URL` and `STRATA`. Each script appends to the files in `raw/` (set `OUT=` to write somewhere else). `python3 scripts/canonical_summary.py` rebuilds the tables.

Every command in the order it ran is in [`raw/strata-canonical-2026-10-07-commands.txt`](raw/strata-canonical-2026-10-07-commands.txt), and every result row is in `raw/strata-canonical-*-2026-10-07.jsonl`.
