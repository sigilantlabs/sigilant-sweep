# llama.cpp from Intel MacBook Air 2017 (Modal-only)

Your laptop is used as control plane only; inference runs on Modal GPU.

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

## 3) If Modal latest fails on your machine with `cbor2`

```bash
pip uninstall -y modal cbor2
pip install --only-binary=:all: "cbor2>=5.6.0"
pip install -U "modal>=1.4,<2" "huggingface-hub>=0.23.0"
pip install -e .
```

## 4) Authenticate and run

```bash
modal token new
```

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --benchmark-mode ranking
```

## 5) Sanity run first (cheap)

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 1 \
  --trials 1
```
