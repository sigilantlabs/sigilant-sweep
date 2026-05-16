# Running sigilant-runner on Modal

Modal runs the entire benchmark sweep on a cloud GPU in your own Modal workspace.
Your machine only submits the job and displays results — no local GPU needed, no model
download on your machine.

---

## What happens when you run on Modal

1. `sigilant-runner` lists the available GGUF files in the HuggingFace repo (locally, takes ~2s)
2. Generates the 16-config grid based on the target GPU's VRAM
3. Submits the job to your Modal workspace
4. Modal spins up a container with the target GPU
5. The container downloads the model files directly from HuggingFace onto the GPU machine
6. Runs all 16 configs sequentially, measures TPS / TTFT / ITL / PPL per config
7. Returns results to your terminal
8. `sigilant-runner` scores and prints the ranked table

The model is **never downloaded to your local machine**.

---

## Prerequisites

- Python 3.10 or higher
- A terminal (macOS, Linux, or WSL on Windows)
- Internet connection
- A Modal account (free to create, free credits included)

---

## Step 1 — Get the code

```bash
git clone https://github.com/sigilantlabs/sigilant-runner.git
cd sigilant-runner
```

Or if you already have the folder:

```bash
cd ~/path/to/sigilant-runner
```

---

## Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt after this.

> **Already have a venv with Modal installed?**
> Skip to Step 3. Activate your existing venv first:
> `source /path/to/your/.venv/bin/activate`

---

## Step 3 — Install sigilant-runner with Modal support

```bash
pip install -e '.[modal]'
```

This installs:
- `sigilant-runner` — the CLI
- `modal` — the Modal SDK (submits jobs to your workspace)
- `huggingface-hub` — lists available model files from HuggingFace (local, lightweight)
- `typer` and `rich` — CLI and terminal output

Nothing inference-related is installed locally. llama.cpp runs entirely inside the
Modal container on the remote GPU.

---

## Step 4 — Create a Modal account

Go to **https://modal.com** and sign up. A free account is sufficient to get started.
Modal gives you free compute credits when you sign up.

---

## Step 5 — Authenticate with Modal

```bash
modal token new
```

This opens a browser tab. Log in to your Modal account. Your token is saved automatically
to `~/.modal.toml`. You do not need to set any environment variables manually.

**Verify it worked:**

```bash
modal profile current
```

Expected output: your Modal workspace name, e.g. `diptanshu1`.

> If the browser does not open automatically, the command prints a URL — open it manually.

---

## Step 6 — (Optional) HuggingFace token

Only needed if you want to use **gated models** (Llama 3, Gemma, Mistral NeMo).
Public models — Mistral, Qwen, Phi, TheBloke repos — work without a token.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

To make it permanent across terminal sessions:

```bash
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx' >> ~/.zshrc
source ~/.zshrc
```

---

## Step 7 — Run

```bash
sigilant-runner run \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --backend modal \
  --hardware a10g
```

The first time you run this, Modal builds the container image (~2–3 minutes, one-time
only). Subsequent runs skip the build and start in under 30 seconds.

---

## Hardware options

Pass any of these to `--hardware`:

| Value       | GPU             | VRAM  | Good for                        |
|-------------|-----------------|-------|---------------------------------|
| `t4`        | NVIDIA T4       | 16 GB | Models up to 7B                 |
| `l4`        | NVIDIA L4       | 24 GB | Models up to 13B                |
| `a10g`      | NVIDIA A10G     | 24 GB | Models up to 13B, faster than L4|
| `a100`      | NVIDIA A100 40G | 40 GB | Models up to 30B                |
| `a100-80`   | NVIDIA A100 80G | 80 GB | Models up to 70B                |
| `h100`      | NVIDIA H100     | 80 GB | Largest models, fastest         |

If you are unsure, use `a10g`. It covers most 7B–13B models and is cost-effective.

---

## Full run command reference

```bash
sigilant-runner run \
  --model <HF_REPO_OR_LOCAL_PATH> \
  --backend modal \
  --hardware <GPU> \
  [--engine llama.cpp]   \  # default, can also be: vllm
  [--configs 16]         \  # default, reduce to 4 for a quick test
  [--json]                  # also write results to sigilant_results.json
```

**Examples:**

```bash
# Standard 7B sweep on A10G
sigilant-runner run \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --backend modal \
  --hardware a10g

# 14B model on A100
sigilant-runner run \
  --model Qwen/Qwen2.5-14B-Instruct-GGUF \
  --backend modal \
  --hardware a100

# Quick 4-config smoke test (faster, cheaper)
sigilant-runner run \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --backend modal \
  --hardware a10g \
  --configs 4

# Save results to JSON
sigilant-runner run \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --backend modal \
  --hardware a10g \
  --json
```

---

## Running with a HuggingFace repo ID

Whenever you pass a HuggingFace repo ID as `--model`, sigilant-runner automatically:
1. Lists the GGUF files available in that repo
2. Selects up to 4 quant variants (Q8_0, Q6_K, Q5_K_M, Q4_K_M in preference order)
3. On Modal: the remote container downloads those files directly from HuggingFace

You do not need to download anything locally first. The format is always
`<owner>/<repo-name>` — the same as the URL path on huggingface.co.

```bash
# HuggingFace repo ID — model downloads on the remote GPU, not your machine
sigilant-runner run \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --backend modal \
  --hardware a10g
```

**Where to find the repo ID:**
Go to the model page on huggingface.co. The repo ID is the `owner/model-name`
in the URL: `huggingface.co/mistralai/Mistral-7B-Instruct-v0.3` → `mistralai/Mistral-7B-Instruct-v0.3`.

Make sure the repo contains `.gguf` files. Repos labelled `-GGUF` in the name
(e.g. TheBloke repos, Bartowski repos) always contain them.

---

## Qwen 1.5B — local vs Modal

Qwen 1.5B is small enough to run on CPU locally. These are the two commands
depending on where you want the sweep to run.

**Run locally (on your machine, CPU):**

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local
```

The model files download to `~/.cache/sigilant/models/` on first run and are reused
on all future runs. On a CPU-only machine expect ~3–6 tokens/sec. A 16-config sweep
will take roughly 30–60 minutes.

**Run on Modal (remote GPU, faster):**

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --hardware l4
```

`l4` is sufficient for a 1.5B model (24 GB VRAM, well within budget). The sweep
completes in roughly 10–15 minutes on L4. Nothing is downloaded to your local machine.

**Side by side:**

| | Local | Modal |
|---|---|---|
| Command | `--backend local` | `--backend modal --hardware l4` |
| Model download | To your machine (`~/.cache/sigilant/`) | To the remote GPU (never touches your disk) |
| Speed | ~3–6 tok/s on CPU | ~80–120 tok/s on L4 |
| Sweep time (16 configs) | 30–60 min | 10–15 min |
| Cost | Free | Modal GPU credits |
| Requires GPU | No | No (Modal provides it) |

---

## What the output looks like

```
sigilant-runner  Mistral-7B-Instruct-v0.3  ·  A10G  ·  llama.cpp  ·  16 configs

Config                                      TPS     TTFT    ITL     PPL    Score
──────────────────────────────────────────────────────────────────────────────────
Q5_K_M · ctx:16384 · kv:f16   · b:4        53.3    612ms   19ms    8.44   91  ← best
Q5_K_M · ctx:8192  · kv:f16   · b:4        53.1    609ms   19ms    8.44   89
Q4_K_M · ctx:16384 · kv:f16   · b:4        56.2    591ms   18ms    8.71   87
Q4_K_M · ctx:8192  · kv:q8_0  · b:4        55.8    594ms   18ms    8.73   85
... 12 more configs

Best config:  Q5_K_M · ctx:16384 · kv:f16 · b:4

PPL is a quality proxy, not production validation.

! Agent safety NOT evaluated.
  Structural JSON, tool calling, hallucination resistance,
  and prompt injection are not covered by this sweep.

  → sigilantlabs.com/optimize
```

---

## Expected timing

| Phase                       | Duration       |
|-----------------------------|----------------|
| Grid generation (local)     | 2–5 seconds    |
| Modal container cold start  | 20–40 seconds  |
| Model download (first time) | 5–20 minutes (depends on model size and network) |
| Model download (cached)     | 0 — Modal caches it on a volume |
| 16-config sweep (7B, A10G)  | 20–40 minutes  |
| Results returned            | Instant        |

Model files are cached on a Modal Volume after the first download. Running the same
model again skips the download entirely.

---

## Environment variables — complete list

| Variable          | Required | What it does                                                              |
|-------------------|----------|---------------------------------------------------------------------------|
| `HF_TOKEN`        | Optional | HuggingFace access token. Only needed for gated models (Llama 3, Gemma). |
| Modal credentials | Handled  | Stored in `~/.modal.toml` by `modal token new`. No env var needed.        |

No other environment variables are needed or read for a Modal run.

> `SIGILANT_LLAMA_CLI` is only relevant for the `--backend local` path.
> It is completely ignored when running on Modal.

---

## Checking credentials at any time

```bash
sigilant-runner setup
```

This checks Modal authentication, RunPod credentials, and local hardware in one pass,
and walks you through fixing anything that is missing.

---

## Troubleshooting

**`Error: No module named 'modal'`**
```bash
pip install 'sigilant-runner[modal]'
```

**`modal token new` says credentials already exist**

Your token is already saved. Run `modal profile current` to confirm it's valid.
If it shows a workspace name, you are authenticated.

**Modal job starts but fails with `OOM` or `CUDA out of memory`**

The model is too large for the selected GPU. Use a larger `--hardware` value:
```bash
# upgrade from a10g to a100
--hardware a100
```

Or use a more aggressively quantized model (Q4_K_M instead of Q8_0 variants).

**First run takes a long time before anything happens**

This is the Modal container cold start + image build (one-time, ~2–3 minutes).
Every run after the first starts in under 30 seconds.

**`No .gguf files found in <repo>`**

The HuggingFace repo does not contain GGUF files. Use a GGUF-specific repo.
TheBloke repos are a reliable source:
```bash
--model TheBloke/Mistral-7B-Instruct-v0.2-GGUF
```

**`modal profile current` returns nothing or an error after `modal token new`**

Close and reopen your terminal, then try again. The token file was written but the
current shell session may not have picked it up.
```bash
source ~/.zshrc    # or ~/.bash_profile on bash
modal profile current
```
