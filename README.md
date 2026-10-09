# NACA — Non-Additive Component Attribution

Component attribution explains a network's output by fitting a linear surrogate over random
ablations of its components. That surrogate is fitted at a **single ablation rate**, and it is
known to degrade when evaluated at other rates — usually read as evidence that the linear form is
not expressive enough.

This repository contains the **code** for two results:

1. **The degradation is mostly transfer, not misspecification.** Refitting the *same* linear model
   at the evaluation rate (an "oracle-at-α" control) recovers most of it. Best-achievable additive
   quality is monotone in ablation rate, so the peak at the training rate that the literature
   reports is manufactured by the protocol.
2. **Under fixed-cardinality mask sampling — the standard protocol — any feature that is a
   function of the kept-component count is exactly collinear with the intercept, hence
   unidentifiable.** This is a property of the design matrix, not of the model, so no increase in
   capacity or data recovers it.

Built on **COAR** (Shah, Ilyas & Mądry 2024, [arXiv:2404.11534](https://arxiv.org/abs/2404.11534)).

The full write-up is in **[PAPER.md](PAPER.md)**.

---

## ⚠️ This is a reproduce-from-scratch repository

This copy ships with the **code only**. It deliberately contains **no** trained models, **no**
collected counterfactuals, **no** run logs, **no** precomputed results, and **no** figures. You
are expected to regenerate every one of these yourself by running the pipeline on your own
hardware. Expect a full collection to take **hours** on a GPU.

In particular:

- There is **no figure-generation code**. The paper refers to figures by filename; **you write the
  plotting scripts yourself** from your regenerated run outputs. `src/naca/plotting.py` (shared
  style helpers used by the pipeline) is included, but the scripts that assemble the paper figures
  are not.
- There is **no `results/` folder**. All findings, tables and numbers must come out of your own
  runs.

Everything in `src/`, `scripts/` (the pipeline stages) and `tests/` is the **original, unmodified**
code — do not change it, or your numbers will not line up with the paper.

---

## Repository layout

```
COAR/
├── PAPER.md                  the paper (Markdown source of the PDF)
├── pyproject.toml
│
├── src/naca/                 the library — 18 modules
│   ├── components.py           enumerate model components -> a flat index
│   ├── ablation.py             apply/restore ablations (conv filters, ViT heads, MLP groups)
│   ├── masks.py                bit-packed mask storage and generation
│   ├── collect.py              counterfactual collection loop
│   ├── solver.py               ridge on the centered system (fp64 Cholesky / block-CG)
│   ├── coar.py                 the linear surrogate
│   ├── interaction.py          low-rank indefinite interaction model (A = UU' - PP')
│   ├── screening.py            candidate-pool selection (marginal / residual / union)
│   ├── granularity.py          filter / filter-group / layer / block unit maps
│   ├── metrics.py              rho, R^2, gauge-invariant (offset-corrected) R^2, NAF
│   ├── models.py               ResNet-18/50, ViT-Tiny, training loop
│   ├── vit.py                  ViT-Tiny written out, so ablation targets are explicit
│   ├── datasets.py             CIFAR-10/100, GPU-resident
│   ├── plotting.py             figure style/palette helpers (NOT the paper figure scripts)
│   ├── editing.py              counterfactual editing
│   ├── baselines.py            MLP / GBT surrogates
│   └── config.py               typed config loading
│
├── scripts/                  pipeline stages and experiments (unmodified)
│   ├── 01_train_models.py      train the target network
│   ├── 02_collect.py           collect counterfactuals (main + extra-alpha pools + test grid)
│   ├── 03_fit_linear.py        linear COAR, evaluated across the alpha grid
│   ├── 04_fit_naca.py          screening + interaction model
│   ├── 08_ablations.py         the 19-arm ablation grid
│   ├── wp*.py                  the work-package experiments behind the paper
│   ├── print_results.py        one-command dump of a setup's full state
│   └── run_*.ps1               orchestration for the long multi-stage runs
│
├── tests/                    synthetic-recovery and ViT ablation-correctness tests
├── configs/                  one YAML per setup (A, seeds 2-5, CIFAR-100, ViT-Tiny)
└── paper/                    build_pdf.py + preamble.tex (produces the PDF from PAPER.md)
```

---

## Setup

Python 3.11 and a CUDA GPU. The pipeline is compute-bound on forward passes — roughly 8 masks/s
for ResNet-18 on CIFAR-10 at 1,000 evaluation examples, so a full collection is hours.

```bash
python -m venv venv
venv/Scripts/activate                 # Windows;  source venv/bin/activate elsewhere
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pytest tests/                         # run the test suite
```

Paths come from the `NACA_ROOT` environment variable. Point it at your clone:

```bash
export NACA_ROOT=/path/to/COAR        # $env:NACA_ROOT = "..." on Windows PowerShell
```

CIFAR-10/100 are **downloaded automatically** by the pipeline on first run; they are not vendored.

## Running the pipeline

```bash
python scripts/01_train_models.py --config configs/setup_a.yaml
python scripts/02_collect.py      --config configs/setup_a.yaml \
                                  --extra-train-alphas 0.01,0.05,0.20 --extra-train-m 10000
python scripts/03_fit_linear.py   --config configs/setup_a.yaml --collect-dir runs/<dir> --lam 100
python scripts/wp1d_structure_and_fig13.py --config configs/setup_a.yaml --collect-dir runs/<dir>
```

Collection dominates the runtime. The analysis scripts run on stored arrays in seconds to minutes.

**Two experiments reproduce the central claim in minutes and need no GPU:**

```bash
python scripts/wpA_synthetic_identifiability.py   # the proposition, on synthetic data
python scripts/wp1c_identifiability.py            # the proposition, on the real design
```

## Producing figures and the paper

The figures are **not** provided and there is **no** script to generate them. Once you have run
the pipeline and have run outputs on disk:

1. Inspect your run directories and the summaries the analysis scripts write.
2. **Write your own plotting scripts** to produce the figures the paper refers to. You may use the
   style helpers in `src/naca/plotting.py`.
3. Build the PDF once the figures your paper references exist on disk:

   ```bash
   python paper/build_pdf.py          # needs pandoc (pip install pypandoc_binary) + XeLaTeX
   ```

---

## Conventions worth knowing before reading the code

- **`v_i = 0` ablates component *i*; `v_i = 1` keeps it.** Asserted at every mask entry point,
  because it is the single most common source of sign bugs here.
- **Ridge is fitted on the centered system**, which makes the intercept unpenalised — and is also
  what makes the degeneracy visible: a constant aggregate column centers to exactly zero.
- **Raw R² is not comparable across ablation rates.** Every metric is reported as ρ, raw R²,
  offset-corrected R² and MSE together.

## Citing

This work extends COAR:

```bibtex
@article{shah2024coar,
  title   = {Decomposing and Editing Predictions by Modeling Model Computation},
  author  = {Shah, Harshay and Ilyas, Andrew and M{\k{a}}dry, Aleksander},
  journal = {arXiv preprint arXiv:2404.11534},
  year    = {2024}
}
```
