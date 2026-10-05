# What I run on my inference server

Qwen3.8-Flash-Next on two RTX 3090s. This page always shows the current setup. When it changes, the new one takes the top and the old one gets a chapter in [How we got here](#how-we-got-here).

## Now: Strata v0.1.39, since October 2026

OrcaRouter's uncensored IQ4_XS build of Qwen3.8-Flash-Next, served by [Strata](https://github.com/Niko1221/Strata) on two RTX 3090s. Images on, two requests at once, 262K context.

It writes at around 100 tok/s, and the machine barely hums while it does it.

| Prompt length | Writes (tok/s) | Reads the prompt (tok/s) | First token |
| ------------- | -------------: | -----------------------: | ----------: |
| ~130 tokens   |            110 |                        - |       1.0 s |
| ~4.2K tokens  |             96 |                      740 |       5.7 s |
| ~62K tokens   |             96 |                    2,290 |      26.9 s |

That is a code answer, 512 tokens out, median of 3 runs on Strata v0.1.39. The second task in the same run (a reasoning question on the short prompt, a question about the text on the long ones) wrote at 93 to 120 tok/s. Every prompt starts with a random nonce so nothing comes from a cache, temperature 0.7, top_p 0.95. Raw rows: [`raw/strata-139-bench-2026-10-04.jsonl`](raw/strata-139-bench-2026-10-04.jsonl), label `live-0.1.39-262k`.

A few more things I checked on the same setup ([`raw/strata-139-probes-2026-10-04.jsonl`](raw/strata-139-probes-2026-10-04.jsonl)):

- **Two short chats at once:** 55 and 62 tok/s each. Both 600-token answers were done in 12.3 s.
- **Coming back to a 62K conversation** after another 62K conversation ran in between: 1.1 s to the first token. Reading it the first time took 35 s. That's Strata's conversation parking. On the build I ran the day before, without it, coming back took 27 s.
- **Images:** a test picture with text and two shapes, everything read correctly in 3.6 s, also while another request was writing.
- **Two fresh 62K prompts sent at the same time** is the slow case. Strata reads new prompts one after the other, so the first answer slowed to an average of 12 tok/s while the second prompt was being read, and the second one started after 55 s.
- **Quality:** 5 of the 6 objectively graded prompts in my 20-prompt set, same as every setup before it. That set is small, it tells me nothing broke, not which one is smarter.

The box: 2x RTX 3090 24 GB, Ryzen 7 9800X3D, 64 GB DDR5, NVMe, Proxmox with the server in a container.

**The full recipe, config and build steps: [`setups/2026-10-strata-v0.1.39/`](setups/2026-10-strata-v0.1.39/).**

## How we got here

Same box the whole way, same benchmark scripts, every number in [`raw/`](raw).

### August 2026: llama.cpp, about 39 tok/s

Flash-Next came out and the published ways to run it started at four 3090s. llama.cpp was the one thing that ran it on two. Unsloth's UD-IQ4_XS, with the experts split by hand between the cards and RAM, wrote at 38 to 40 tok/s. What I took from it: where the model lives matters more than the quant. It ran, but my 27B was about two and a half times faster, so the 27B stayed my daily driver. [The full write-up](setups/2026-08-llama-cpp/).

### September 2026: EXL3 on TabbyAPI, 69 to 90 tok/s

A 3.05 bpw EXL3 build was small enough that most of it fit on the two cards, with only the coldest experts on the CPU. With the model's own draft head guessing ahead, it wrote at 69 to 85 tok/s, then 78 to 90 once I set the draft to 3 tokens with dynamic drafting, about 9% faster than the default 4. Keeping the busy experts on the cards and guessing ahead roughly doubled llama.cpp, and that's the step where Flash-Next replaced the 27B as my daily driver.

### Late September 2026: DominikBucko's vLLM fork, 57 to 78 tok/s

[DominikBucko's fork](https://github.com/DominikBucko/qwen38-flash-next-2x3090) read long prompts about twice as fast, 2,100 to 3,000 tok/s at 60K against about 1,300 on EXL3, and it could take two requests at once. Writing was slower, 57 to 78 tok/s one request at a time, and it made the machine sound like it was gonna take off. It was the default for a couple of days.

### October 2026: Strata, 93 to 120 tok/s

I kept running into Strata, didn't quite believe it, then tested it. The first try, v0.1.34, already wrote at about 90 tok/s on code and up to 110 on reasoning, and the whole machine was weirdly chill doing it. I'd spent about a week trying something vaguely similar myself and didn't get results this good. v0.1.39 then made two requests at once official and added conversation parking: coming back to a 62K conversation went from 27 s to 1.1 s. That's the setup at the top.

## Credits

Qwen3.8-Flash-Next is by the Qwen team, and the uncensored GGUF is [OrcaRouter's](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF).

Strata is by [Niko1221](https://github.com/Niko1221), and it's the reason this box runs the way it does. The proverbial hat is off to Niko1221, the guy pulled it off nicely. Strata is free. If it runs well for you too, [a coffee](https://buymeacoffee.com/strataengine) keeps the work on it going.

## What's in the repo

| Path | What it is |
| --- | --- |
| [`setups/`](setups) | One folder per setup: config, service and the steps to build it. The newest is what runs now |
| [`scripts/bench.py`](scripts/bench.py) | Speed: nonce per prompt, warm-up first, decode and prompt reading separately |
| [`scripts/quality.py`](scripts/quality.py), [`scripts/autograde.py`](scripts/autograde.py) | The fixed 20-prompt set and the grader for its 6 objective items |
| [`scripts/probe.py`](scripts/probe.py) | Images, two requests at once, coming back to a conversation |
| [`scripts/suite139.sh`](scripts/suite139.sh) | All of the above against one server: `URL=http://127.0.0.1:8080/v1 scripts/suite139.sh my-run` |
| [`raw/`](raw) | Every result row and log behind the numbers |
