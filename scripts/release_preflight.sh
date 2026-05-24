#!/usr/bin/env bash
set -euo pipefail

TARGET_VERSION="${1:-}"
if [[ -z "${TARGET_VERSION}" ]]; then
  echo "Usage: bash scripts/release_preflight.sh <target-version>"
  exit 2
fi

if [[ ! -f "pyproject.toml" || ! -f "sigilant_runner/__init__.py" ]]; then
  echo "ERROR: run this from repo root (missing pyproject.toml or sigilant_runner/__init__.py)."
  exit 1
fi

PYPROJECT_VERSION="$(sed -nE 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' pyproject.toml | head -n 1)"
INIT_VERSION="$(sed -nE 's/^__version__[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' sigilant_runner/__init__.py | head -n 1)"

if [[ -z "${PYPROJECT_VERSION}" || -z "${INIT_VERSION}" ]]; then
  echo "ERROR: could not parse version from pyproject.toml or sigilant_runner/__init__.py"
  exit 1
fi

if [[ "${PYPROJECT_VERSION}" != "${INIT_VERSION}" ]]; then
  echo "ERROR: version mismatch:"
  echo "  pyproject.toml: ${PYPROJECT_VERSION}"
  echo "  __init__.py   : ${INIT_VERSION}"
  exit 1
fi

if [[ "${PYPROJECT_VERSION}" != "${TARGET_VERSION}" ]]; then
  echo "ERROR: current version (${PYPROJECT_VERSION}) does not match target (${TARGET_VERSION})"
  exit 1
fi

echo "OK: release preflight passed for version ${TARGET_VERSION}"
