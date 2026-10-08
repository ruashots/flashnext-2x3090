# What I run on my inference server

Qwen3.8-Flash-Next on two RTX 3090s. This page always shows the current setup. When it changes, the new one takes the top and the old one gets a chapter in [How we got here](#how-we-got-here).

## Now: Strata v0.1.40.2, since October 7, 2026

OrcaRouter's uncensored IQ4_XS build of Qwen3.8-Flash-Next, served by [Strata](https://github.com/Niko1221/Strata) on two RTX 3090s. Images on, two requests at once, 262K context.

It writes at 100 to 110 tok/s, reads long prompts at about 2,500 tok/s, and the machine barely hums while it does it.

| Prompt length | Writes (tok/s) | Reads the prompt (tok/s) | First token |
| ------------- | -------------: | -----------------------: | ----------: |
| ~140 tokens   |            110 |                        - |       1.2 s |
| ~4.2K tokens  |            112 |                      989 |       4.3 s |
| ~66K tokens   |            100 |                    2,487 |      26.4 s |
| ~131K tokens  |             93 |                    2,564 |      51.2 s |

Median of 6 runs (3 code answers, 3 reasoning or questions about the text), 512 tokens max, temperature 0.7, top_p 0.95, a random nonce at the start of every prompt so nothing comes from a cache. Raw rows: [`raw/strata-canonical-bench-2026-10-07.jsonl`](raw/strata-canonical-bench-2026-10-07.jsonl), label `C262`.

The full set is in **[BENCHMARKS.md](BENCHMARKS.md)**, and the short version is right below.

The box: 2x RTX 3090 24 GB capped at 225 W, Ryzen 7 9800X3D, 64 GB DDR5, NVMe, Proxmox with the server in a container.

**The full recipe, configs and build steps: [`setups/2026-10-strata-v0.1.40.2/`](setups/2026-10-strata-v0.1.40.2/).**

## Where things stand

From the [benchmark run](BENCHMARKS.md) on October 7:

- **Retrieval:** 61 of 61 planted facts found, from 4K up to 1M. That shows it finds things in a long text, not that it reasons over all of it.
- **GSM8K:** 1,271 of the 1,319 test problems, 96.4%, thinking off.
- **Two short chats at once:** about 58 tok/s each. Together that's about what one writes alone, the gain is that the second one starts in a second instead of waiting.
- **Screenshots:** 6 of 7 small-text codes read exactly from a full-screen 2560x1440 screenshot.
- **Coming back to a parked 62K conversation:** 1.1 s to the first token (measured on v0.1.39, same settings).
- **Limits:** each run past 128K ran once, so those numbers are noisy. The cards are capped at 225 W, and the server kept serving my agents during the runs.

**The options**, same box, same model files, a config swap and a restart between them:

| Setup | What it gives | What it costs |
| --- | --- | --- |
| **262K**, every day | Two requests at once, images up to 4096 tokens, 110 tok/s short, 93 at 131K | Prompts top out at 262K |
| **512K** | A 500K prompt read in about 3.5 min, then 100 tok/s | Stretched with yarn, one request at a time, smaller image cap |
| **1M** | A 1M prompt read in about 9 min, then 82 tok/s | Stretched further, one request at a time, short answers slower (92 tok/s), stream the reply at this length |

The configs are in [`setups/2026-10-strata-v0.1.40.2/`](setups/2026-10-strata-v0.1.40.2/).

**Run the benchmark on your own setup.** With the server running, from a clone of this repo:

```bash
git clone --branch v0.1.40.2 https://github.com/Niko1221/Strata.git strata-haystack   # the needle tests build their text from it
U=http://127.0.0.1:8080; M=Qwen3.8-Flash-Next-OrcaRouter
python3 scripts/bench.py --url $U/v1 --model $M --label mine --out mine-speed.jsonl --contexts 0,4000,32000,128000
python3 strata-haystack/tools/needle_bench.py --url $U --lengths 32k,128k,256k --depths 10,50,90
python3 scripts/needle_multikey.py --strata strata-haystack --url $U --lengths 32k,128k,256k --label mine --out mine-needle.jsonl
python3 scripts/gsm8k.py --url $U/v1 --model $M --label mine --out mine-gsm8k.jsonl --workers 2
```

Only Python 3 is needed, no packages. The speed run takes a while, the needles longer, and GSM8K downloads its test set on first use and runs all 1,319 problems (`--limit 250` for a quick look). `scripts/canonical_262k.sh` runs my whole set in one go, and [BENCHMARKS.md](BENCHMARKS.md#run-it-yourself) has the full method and the long-context runs.

## How we got here

Same box the whole way, same benchmark scripts, every number in [`raw/`](raw).

### August 2026: llama.cpp, about 39 tok/s

Flash-Next came out and the published ways to run it started at four 3090s. llama.cpp was the one thing that ran it on two. Unsloth's UD-IQ4_XS, with the experts split by hand between the cards and RAM, wrote at 38 to 40 tok/s. What I took from it: where the model lives matters more than the quant. It ran, but my 27B was about two and a half times faster, so the 27B stayed my daily driver. [The full write-up](setups/2026-08-llama-cpp/).

### September 2026: EXL3 on TabbyAPI, 69 to 90 tok/s

A 3.05 bpw EXL3 build was small enough that most of it fit on the two cards, with only the coldest experts on the CPU. With the model's own draft head guessing ahead, it wrote at 69 to 85 tok/s, then 78 to 90 once I set the draft to 3 tokens with dynamic drafting, about 9% faster than the default 4. Keeping the busy experts on the cards and guessing ahead roughly doubled llama.cpp, and that's the step where Flash-Next replaced the 27B as my daily driver.

### Late September 2026: DominikBucko's vLLM fork, 57 to 78 tok/s

[DominikBucko's fork](https://github.com/DominikBucko/qwen38-flash-next-2x3090) read long prompts about twice as fast, 2,100 to 3,000 tok/s at 60K against about 1,300 on EXL3, and it could take two requests at once. Writing was slower, 57 to 78 tok/s one request at a time, and it made the machine sound like it was gonna take off. It was the default for a couple of days.

### October 2026: Strata, 93 to 120 tok/s

I kept running into Strata, didn't quite believe it, then tested it. The first try, v0.1.34, already wrote at about 90 tok/s on code and up to 110 on reasoning, and the whole machine was weirdly chill doing it. I'd spent about a week trying something vaguely similar myself and didn't get results this good. v0.1.39 then made two requests at once official and added conversation parking: coming back to a 62K conversation went from 27 s to 1.1 s. [That setup](setups/2026-10-strata-v0.1.39/) ran until the next engine.

### October 7, 2026: Strata v0.1.40.2, tuned

Same setup on the newer engine. Timed by the engine itself against v0.1.40.1, writing got about 5% faster at 4K and stayed the same at 60K, and long prompts read about 8% faster ([how it was counted](raw/strata-engine-timing-2026-10-07.md)). Then I tried the tuning knobs one at a time. The one that stayed was the image cap: a full-screen screenshot with small text went from 0 of 7 codes read to 6 of 7. The loop guard fired on normal thinking, and the two draft settings were slower or within noise, so they went ([why](setups/2026-10-strata-v0.1.40.2/#tried-and-not-kept)). Then the overnight run I'd promised on Reddit: speed up to 1M, needle tests up to 1M and the full GSM8K set, all in [BENCHMARKS.md](BENCHMARKS.md). That's the setup at the top.

## Credits

Qwen3.8-Flash-Next is by the Qwen team, and the uncensored GGUF is [OrcaRouter's](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF).

Strata is by [Niko1221](https://github.com/Niko1221), and it's the reason this box runs the way it does. The proverbial hat is off to Niko1221, the guy pulled it off nicely. Strata is free. If it runs well for you too, [a coffee](https://buymeacoffee.com/strataengine) keeps the work on it going.

## What's in the repo

| Path | What it is |
| --- | --- |
| [`setups/`](setups) | One folder per setup: config, service and the steps to build it. The newest is what runs now |
| [`BENCHMARKS.md`](BENCHMARKS.md) | The full benchmark set for the current setup: speed to 1M, retrieval, GSM8K, method and limits |
| [`scripts/bench.py`](scripts/bench.py) | Speed: nonce per prompt, warm-up first, decode and prompt reading separately |
| [`scripts/quality.py`](scripts/quality.py), [`scripts/autograde.py`](scripts/autograde.py) | The fixed 20-prompt set and the grader for its 6 objective items |
| [`scripts/probe.py`](scripts/probe.py) | Images, two requests at once, coming back to a conversation |
| [`scripts/suite139.sh`](scripts/suite139.sh) | All of the above against one server: `URL=http://127.0.0.1:8080/v1 scripts/suite139.sh my-run` |
| [`scripts/canonical_262k.sh`](scripts/canonical_262k.sh), [`canonical_long.sh`](scripts/canonical_long.sh), [`canonical_1m_rest.sh`](scripts/canonical_1m_rest.sh) | The benchmark set in BENCHMARKS.md, with [`gsm8k.py`](scripts/gsm8k.py), [`needle_multikey.py`](scripts/needle_multikey.py), [`needle_stream.py`](scripts/needle_stream.py) and [`canonical_summary.py`](scripts/canonical_summary.py) for the tables |
| [`scripts/knob.sh`](scripts/knob.sh), [`scripts/quality_think.py`](scripts/quality_think.py) | One tuning knob against the base bench, and the 20-prompt set with thinking on |
| [`raw/`](raw) | Every result row and log behind the numbers |
