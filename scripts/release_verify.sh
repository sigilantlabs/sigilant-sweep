#!/usr/bin/env bash
set -euo pipefail

TARGET_VERSION="${1:-}"
if [[ -z "${TARGET_VERSION}" ]]; then
  echo "Usage: bash scripts/release_verify.sh <target-version>"
  exit 2
fi

if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: run this from repo root."
  exit 1
fi

echo "==> Preflight"
bash scripts/release_preflight.sh "${TARGET_VERSION}"

echo "==> Build"
rm -rf dist build
python3 -m pip install -U build twine
python3 -m build

echo "==> Check dist contains target artifacts"
WHEEL="dist/sigilant_sweep-${TARGET_VERSION}-py3-none-any.whl"
SDIST="dist/sigilant_sweep-${TARGET_VERSION}.tar.gz"
if [[ ! -f "${WHEEL}" || ! -f "${SDIST}" ]]; then
  echo "ERROR: missing target artifacts."
  ls -la dist || true
  exit 1
fi
ls -la dist

echo "==> Twine check"
python3 -m twine check "${WHEEL}" "${SDIST}"

echo "==> Upload"
python3 -m twine upload "dist/sigilant_sweep-${TARGET_VERSION}"*

echo "==> Wait until simple index sees target version"
for i in {1..20}; do
  if python3 - "${TARGET_VERSION}" <<'PY'
import sys
import urllib.request

target = sys.argv[1]
url = "https://pypi.org/simple/sigilant-sweep/"
data = urllib.request.urlopen(url, timeout=20).read().decode("utf-8", "ignore")
needle = f"sigilant_sweep-{target}-py3-none-any.whl"
sys.exit(0 if needle in data else 1)
PY
  then
    echo "OK: ${TARGET_VERSION} visible on simple index"
    break
  fi
  if [[ $i -eq 20 ]]; then
    echo "ERROR: ${TARGET_VERSION} not visible on simple index after polling."
    exit 1
  fi
  sleep 15
done

echo "==> Fresh venv install check"
VERIFY_VENV="/tmp/sigilant-sweep-verify-${TARGET_VERSION}"
rm -rf "${VERIFY_VENV}"
python3 -m venv "${VERIFY_VENV}"
source "${VERIFY_VENV}/bin/activate"
python3 -m pip install -U pip setuptools wheel

echo "==> Fresh venv install (retry until index fully propagates)"
for i in {1..20}; do
  if python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple "sigilant-sweep==${TARGET_VERSION}"; then
    break
  fi
  if [[ $i -eq 20 ]]; then
    echo "ERROR: fresh-venv pip install failed after propagation retries."
    python3 -m pip index versions sigilant-sweep --index-url https://pypi.org/simple || true
    python3 - <<'PY'
import urllib.request
u = "https://pypi.org/simple/sigilant-sweep/"
print(urllib.request.urlopen(u, timeout=20).read().decode("utf-8", "ignore"))
PY
    exit 1
  fi
  echo "Retrying fresh install in 15s (${i}/20)..."
  sleep 15
done

python3 -m pip show sigilant-sweep | grep '^Version:'
sigilant-sweep --version

echo "OK: release verification complete for ${TARGET_VERSION}"
