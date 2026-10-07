# Setup: Strata v0.1.40.2, October 2026

Qwen3.8-Flash-Next, OrcaRouter's uncensored IQ4_XS, on Strata v0.1.40.2 across two RTX 3090s. Images on with a 4096-token cap, two requests at once, 262K context, conversation parking on. The numbers are in the [main README](../../README.md) and the full set is in [BENCHMARKS.md](../../BENCHMARKS.md).

It is the [v0.1.39 setup](../2026-10-strata-v0.1.39/) on the newer engine, with one change: pictures get up to 4096 tokens instead of 1024.

## The box

| | |
| --- | --- |
| GPUs | 2x RTX 3090 24 GB (EVGA FTW3 and Gigabyte Turbo), no NVLink, power capped at 225 W each |
| CPU | Ryzen 7 9800X3D |
| RAM | 64 GB DDR5, the container gets 58 GB of it |
| Disk | NVMe |
| OS | Proxmox, the server runs in an Ubuntu 24.04 LXC container |
| Driver / CUDA | 595.71.05 / 13.0 |
| Engine | Strata v0.1.40.2 (`e8ca9af`), built from source for sm_86 |
| Model | [`orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF), IQ4_XS, plus its F16 `mmproj` for images |

The whole server config is [`strata-orca-0402-262k-img4096.json`](strata-orca-0402-262k-img4096.json). What the less obvious bits do:

- `"vision": {"max_tokens": 4096}`: how many tokens a picture can turn into. At 1024, what I had before, and at 2048, a full-screen 2560x1440 log screenshot with small text came back with 0 of its 7 codes right. At 4096 it got 6 of 7. A small picture costs the same, the test picture with one line of text and two shapes is 300 tokens at either cap.

- `--mmap-experts`: Strata maps the 61 GiB expert file instead of reading it all into locked RAM, so the OS file cache holds what the cards don't, and can give that memory back. The experts used most stay on the cards, the CPU works on the rest.
- `"layer_split": "26"` with `--trim-stage-weights`: layers 0-25 on the first card, 26-47 on the second, and each card only loads its own layers' weights, so more experts fit in VRAM.
- `--batch 2 --batch-groups 2`: two requests at once, pipelined through the two cards.
- `--conversation-cache-mib 8192 --conversation-cache-slots 8`: conversation parking, up to 8 conversations kept so coming back to one doesn't read it again.
- `--kv-resident 20480`: the context's cache lives in RAM and only the part the attention reads stays on the cards, which leaves more VRAM for experts.
- `"sampling"`: Strata answers greedy when the client sends no temperature. Before I set this, a hard prompt from a client that sends no sampler got stuck thinking in a loop and never answered. Now clients that send nothing get temperature 1.0, top_p 0.95, top_k 20, which is what the model card asks for.

## Run the same thing

You need two 24 GB NVIDIA cards, about 64 GB of RAM, about 175 GB of free disk, an NVIDIA driver and the CUDA 13.0 toolkit at `/usr/local/cuda`, plus `git`, `build-essential` and `python3-venv`.

The model repo is gated. Before you start, accept its terms on [its Hugging Face page](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) and have a Hugging Face token ready.

The paths below are the ones the config and the service use. Mine runs as root inside its own container.

**1. Get Strata v0.1.40.2 and this repo**

```bash
mkdir -p /opt/strata && cd /opt/strata
git clone https://github.com/Niko1221/Strata.git
git clone https://github.com/ruashots/flashnext-2x3090.git
cd Strata && git checkout v0.1.40.2
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**2. Compile the engine and the image encoder (10 to 20 minutes, once)**

```bash
.venv/bin/python ../flashnext-2x3090/setups/2026-10-strata-v0.1.40.2/build_strata_engine.py .
```

Strata's own `./setup.sh` compiles it too, but it also wants to download a model from its menu, and this build is not in that menu. The script calls the same build function setup uses, for the cards it finds.

**3. Download the model (about 98 GB)**

```bash
python3 -m venv /opt/strata/hfvenv && /opt/strata/hfvenv/bin/pip install -U huggingface_hub
/opt/strata/hfvenv/bin/hf auth login
/opt/strata/hfvenv/bin/hf download orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF \
  --include "*IQ4_XS*" --include "mmproj-*F16.gguf" \
  --local-dir /opt/strata/models/orca-iq4xs
```

**4. Pack it for Strata**

```bash
.venv/bin/python tools/iq_pack.py \
  --gguf /opt/strata/models/orca-iq4xs/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf \
  --out packs/orca-iq4xs --compat-bf16 --experts-bin
```

`--compat-bf16` converts the few small weights Strata reads as BF16. `--experts-bin` writes the expert file that `--mmap-experts` needs. Strata's [OrcaRouter notes](https://github.com/Niko1221/Strata/blob/main/docs/ORCA.md) only validate IQ3_XXS. IQ4_XS is what I run and measured, it isn't something Strata lists as tested. My pack and the draft head in step 5 were made with v0.1.39's tools and carried over to v0.1.40.2 as they were.

**5. The draft head for speculative decoding**

```bash
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

This is the original Qwen draft head, the way Strata's OrcaRouter notes do it. The OrcaRouter repo also has an `MTP-draft.gguf`, and I don't use it.

**6. Config and service**

```bash
cp ../flashnext-2x3090/setups/2026-10-strata-v0.1.40.2/strata-orca-0402-262k-img4096.json .
cp ../flashnext-2x3090/setups/2026-10-strata-v0.1.40.2/strata-server.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now strata-server
```

It takes about a minute to load. Then `http://127.0.0.1:8080` has Strata's chat and monitor, and any OpenAI-compatible client works with `http://127.0.0.1:8080/v1`. To run it by hand instead, run `ulimit -l unlimited` first, then `.venv/bin/python -m serve.server --engine strata --config strata-orca-0402-262k-img4096.json`.

The published config and service listen on 127.0.0.1. Mine listens on my LAN. If you open yours up, add `--api-key <secret>` to the service command.

## The long-context configs

[`strata-orca-0402-512k.json`](strata-orca-0402-512k.json) and [`strata-orca-0402-1m.json`](strata-orca-0402-1m.json) are what the 512K and 1M rows in [BENCHMARKS.md](../../BENCHMARKS.md) ran on. They stretch the model past its 262K with yarn (x2 and x4), take one request at a time and keep the image cap at 1024. They're for when the length is needed, not what runs day to day.
