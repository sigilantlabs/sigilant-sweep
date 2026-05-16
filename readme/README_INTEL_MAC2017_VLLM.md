# vLLM from Intel MacBook Air 2017 (Modal-only)

Your laptop is used as control plane only; vLLM runs on Modal GPU.

## 1) Create env (Python 3.11)

```bash
cd /Users/diptanshukumar/PycharmProjects/OSS/sigilant-runner-combined
rm -rf .venv
/usr/local/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

## 2) Install package + Modal deps

```bash
pip install -e .
pip install "sigilant-sweep[modal]"
pip install huggingface-hub
```

## 3) If Modal install fails with `cbor2`

```bash
pip uninstall -y modal cbor2
pip install --only-binary=:all: "cbor2>=5.6.0"
pip install -U "modal>=1.4,<2" "huggingface-hub>=0.23.0"
pip install -e .
```

## 4) Authenticate

```bash
modal token new
```

## 5) Set quant repos

```bash
export SIGILANT_VLLM_INT8_W8A8_REPO=anhbn/Phi-3.5-mini-instruct-quantized.w8a8
export SIGILANT_VLLM_AWQ4_MARLIN_REPO=thesven/Phi-3.5-mini-instruct-awq
export SIGILANT_VLLM_GPTQ4_MARLIN_REPO=thesven/Phi-3.5-mini-instruct-GPTQ-4bit
```

## 6) Run vLLM ranking (no depth profile)

```bash
sigilant-sweep run \
  --model microsoft/Phi-3.5-mini-instruct \
  --backend modal \
  --engine vllm \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --benchmark-mode ranking
```

## 7) Single-config debug

```bash
sigilant-sweep run \
  --model microsoft/Phi-3.5-mini-instruct \
  --backend modal \
  --engine vllm \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --only-config "INT8_W8A8,32768,k8v8,long"
```
