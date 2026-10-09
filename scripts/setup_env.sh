#!/usr/bin/env bash
# Creates D:\naca\venv with the exact stack CLAUDE.md pins.
# All caches/temp redirected onto D: (hard rule 3: never write to C:).
set -euo pipefail

ROOT=/d/naca
PY311="C:/Users/Asus/AppData/Local/Programs/Python/Python311/python.exe"

export PIP_CACHE_DIR="D:/naca/cache/pip"
export TMPDIR="D:/naca/cache/tmp"
export TEMP="D:\naca\cache\tmp"
export TMP="D:\naca\cache\tmp"

if [ ! -f "$ROOT/venv/Scripts/python.exe" ]; then
  echo "[setup] creating venv with $("$PY311" --version)"
  "$PY311" -m venv "$ROOT/venv"
fi

VPY="$ROOT/venv/Scripts/python.exe"
"$VPY" -m pip install --upgrade pip setuptools wheel

echo "[setup] installing torch 2.5.1+cu121 / torchvision 0.20.1+cu121"
"$VPY" -m pip install --retries 20 --timeout 120 \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[setup] installing the rest"
"$VPY" -m pip install --retries 20 --timeout 120 \
  numpy scipy pandas pyyaml matplotlib tqdm psutil \
  scikit-learn pytest ruff black

echo "[setup] done"
"$VPY" -c "import torch,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.cuda.is_available())"
