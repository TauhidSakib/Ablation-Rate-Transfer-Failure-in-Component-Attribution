# Ablation-Rate Transfer Failure in Component Attribution: A Structurally Invisible Aggregate Term

> **Reproduction brief — results not yet produced.**
> This document is the *specification* of the study you are to reproduce. It states the questions,
> the method, and the structure of the final paper, but it contains **no measurements**: every
> number, table, and figure is something **you** must generate by running the pipeline on your own
> hardware. Where a value would appear, you will find a `[ TO MEASURE ]` placeholder. Nothing here
> has been run for you.

---

## Abstract (to be written from your own results)

Component attribution explains a network's output by fitting a linear surrogate over random
ablations of its components. The surrogate is fitted at a single **ablation rate** — the fraction
of components removed per mask — and is known to degrade when evaluated at other rates. That
degradation has been read as evidence that the linear functional form is not expressive enough.

This study investigates whether that reading is correct. The claims you are to test and quantify:

1. **Is the degradation transfer rather than misspecification?** Use an *oracle-at-alpha control*
   — refit the same linear model on masks drawn at the evaluation rate — and measure how much of
   the apparent degradation survives it. `[ TO MEASURE ]`
2. **Is best-achievable additive quality monotone in ablation rate?** `[ TO MEASURE ]`
3. **Prove and verify the identifiability result:** under fixed-cardinality mask sampling, any
   feature that is a function of the kept-component count is exactly collinear with the intercept,
   hence unidentifiable. Verify numerically on the real design and on synthetic data.
4. **Measure the aggregate (quadratic) term:** its size, whether it has depth structure, and how
   much out-of-distribution error is removed by pooling rates and adding one scalar parameter.
   `[ TO MEASURE ]`
5. **Characterise pairwise structure** at filter and block granularity, with a positive control
   on the same code path, reporting every quantity against its precision floor. `[ TO MEASURE ]`
6. **Replicate across multiple independently trained seeds**, on CIFAR-100, and on a ViT-Tiny.

Write the abstract last, from the numbers you obtain.

## 1. Introduction

Framing: attribution methods are measured on their own training distribution, and the standard
protocol may have a blind spot that is structural rather than statistical. Intended contributions:

1. **A proposition** (§3) characterising exactly what fixed-cardinality sampling makes
   unidentifiable, unifying two previously separate observations.
2. **A control** (§4.1) that separates transfer failure from misspecification.
3. **A measurement**: how large the aggregate term is, whether it is uniform across depth, and
   whether it is cheaply fixable.
4. **A null with a positive control**: whether pairwise structure is negligible, and whether the
   instrument would have found it if present.

---

## 2. Background

COAR (Shah, Ilyas & Mądry 2024) and the general subset-ablation surrogate family: datamodels
(Ilyas et al. 2022), TRAK (Park et al. 2023). Definitions: components, ablation vector `v`,
correct-class margin, α, the linear surrogate `g(v) = b + wᵀv`.

**Reproduction target.** Reproduce the base COAR fit at `α_test = α_train` and report ρ.
Select λ by a held-out sweep. `scripts/03_fit_linear.py` → `[ TO MEASURE ]`

---

## 3. Theory — fixed-cardinality blindness

**This section leads.** It is the organising result: everything measured later is a consequence.
The theory is given in full here because it is not something you measure — it is what your
measurements are meant to confirm.

### 3.1 Proposition

> Let masks be drawn so that every mask ablates exactly `k` of `N` components, i.e. every row of
> the design matrix `X ∈ {0,1}^{m×N}` satisfies `Σᵢ X[t,i] = N − k =: n`. Let `φ : {0..N} → ℝ` be
> any function of the kept-count alone, and augment the model with the column
> `q[t] = φ(Σᵢ X[t,i])`.
>
> Then `q[t] = φ(n)` for all `t`, so `q = φ(n)·1_m`. The augmented design `[X | 1 | q]` has `q`
> exactly collinear with the intercept, `rank([X | 1 | q]) = rank([X | 1])`, and the coefficient
> on `q` is **not identifiable**: only the combination `b + φ(n)·a` is determined by the data.

*Proof.* Immediate from constancy of the row sums. ∎

The content is not the algebra but its reach: it holds for *every* φ, of any order, and it is a
property of the **sampling protocol**, not of the model, the network, or the estimator.

### 3.2 Corollary 1 — two known problems are one problem

- **φ linear.** `φ(n) = n`. Shifting `w → w + c·1_N`, `b → b − c·n` leaves every training
  prediction unchanged. This is the coefficient-shift ambiguity of the additive surrogate.
- **φ quadratic.** `φ(n) = n(n−1)/2`. This is exactly the contribution of a *uniform pairwise
  interaction* `A_ij = a`, since `Σ_{i<j} a·v_i·v_j = a·n(n−1)/2`.

So the sign/level ambiguity and the aggregate interaction term are the same phenomenon at
different orders of φ.

### 3.3 Corollary 2 — what breaks the degeneracy

Sampling distinct cardinalities restores identifiability. Determine from your own synthetic study
(§3.6) the minimum number of distinct cardinalities needed for the **quadratic** aggregate term.

### 3.4 Corollary 3 — what survives at fixed α

Only the *total* is pinned. Per-group kept-counts still vary. With `G` groups, work out which
block-aggregate combinations are identifiable from single-α data and which one is not.

### 3.5 Exact numerical verification

Run `scripts/wp1c_identifiability.py` on the real masks and report the collinearity checks, the
rank with and without pooling, and the block-feature rank/count. → `[ TO MEASURE ]`

Prediction to test: fitting the block features at fixed α should produce a singular
normal-equation system — singular by construction, not numerical accident.

### 3.6 A synthetic demonstration of the proposition, with no network involved

§3.5 verifies the algebra on the real design. This isolates the *consequence* on data where the
truth is known: `y(v) = b + wᵀv + β·f(n) + ε`, with `β` known and nonzero. Sweep `β` over several
magnitudes, under protocols with different numbers of distinct cardinalities, and report
`resid_frac`, identifiability, and `β̂/β`. `scripts/wpA_synthetic_identifiability.py`
→ `[ TO MEASURE ]`

Questions to answer from the output:
- Under how many distinct cardinalities is the **linear** form identifiable?
- How many distinct cardinalities does the **quadratic** aggregate require?
- Do deliberately separated cardinalities condition better than many fluctuating ones?

*(The study's two paper figures — "two failure modes" and the K-sweep — are **not** provided.
You will write the plotting code yourself from your run outputs.)*

---

## 4. Instruments

The three measurement tools, each introduced because a naive alternative gives the wrong answer.
For each, report the naive alternative and show why the instrument is needed.

### 4.1 The oracle-at-α control

**Definition.** To test whether degradation at α_test is misspecification or transfer, fit a
*fresh* surrogate on masks drawn at α_test itself and evaluate at α_test. This removes transfer
entirely; whatever degradation survives is intrinsic to that ablation rate.

Produce the transfer-ρ vs oracle-ρ table across the α grid, with `Var(f_M)` beside it. Decide
from your own numbers whether the low-α degradation is signal starvation or transfer.
`scripts/wp1d_structure_and_fig13.py` → `[ TO MEASURE ]`

### 4.1b The transfer result across seeds

Measure the oracle curve on several independently trained ResNet-18 / CIFAR-10 networks. Report
per-seed ρ at each rate, the paired within-seed drop with its effect size, and monotonicity
counts. Keep the as-collected design as primary; any matched-budget refit is a diagnostic, not a
re-score. `scripts/wp10_seed_summary.py`, `scripts/wp10_matched_budget_oracle.py` → `[ TO MEASURE ]`

### 4.2 Gauge-invariant R²

Raw R² is not comparable across α under §3. Report R² with a per-cell offset removed alongside the
raw value, and check whether ρ is invariant to the effect. → `[ TO MEASURE ]`

### 4.3 Group-confined refits, not share-weighted averages

Show that per-example NAF weighted by pool share cannot localise non-additivity, and that
confining the pool to one group and refitting can. Validate on synthetic data with planted
structure. → `[ TO MEASURE ]`

### 4.4 Positive control

Every null must be accompanied by the same pipeline run on synthetic counterfactuals generated
from a known linear + indefinite-quadratic model. Report the NAF it recovers where structure is
planted, and confirm it localises depth correctly. This control precedes any real-data null.
→ `[ TO MEASURE ]`

---

## 5. Results

*All of §5 is to be produced by you. The subsection structure below is the template; fill each
`[ TO MEASURE ]`.*

### 5.1 Is the aggregate term large, and structurally invisible?

Measure how far the additive model's predicted mean margin tracks the true mean across the α grid,
the per-component marginal decay, and whether the term is *unidentifiable* rather than *small*
(compare `mean(w)` fitted at one α vs pooled α). Rule out the gauge-constant explanation.
→ `[ TO MEASURE ]`

### 5.2 How much transfer error does one scalar parameter remove?

Score models A (single-α linear), B (pooled-α linear), C (B + one global term) on held-out α
values. Report ρ, raw R², offset-corrected R², and level tracking. State the in- vs
out-of-distribution trade-off. → `[ TO MEASURE ]`

### 5.3 Does the aggregate term have depth structure?

Test whether block-aggregate coefficients beat a single global scalar. → `[ TO MEASURE ]`

### 5.4 Filter-level structure — report as a range across seeds, with floors

Run the K-sweep under fixed-rank and capacity-matched protocols on each seed. Report every value
against that seed's bf16 precision floor, and the cross-seed CI. **Never quote a single number
here as a null or a positive.** Decide whether any concentration claim survives across seeds.
`scripts/08_ablations.py`, `scripts/wp0_precision_floor.py` → `[ TO MEASURE ]`

### 5.4b Reconciling the two granularities — the coarsening–aliasing theorem

The theorem is given (filter→block coefficients are signed parity sums). Verify all four
corollaries numerically, and test the reconciliation: are the structure-carrying filters spread
across blocks, or concentrated in one? `scripts/wp6_aliasing.py` → `[ TO MEASURE ]`

### 5.5 PRE-REGISTERED: does apparent non-additivity track ablation severity, not granularity?

**Write your prediction, with reasoning, before computing the exhaustive spectrum** (§5.6), and
commit it. State what result would confirm it and what would refute it. Then record the outcome
separately from the pre-registration. → `[ TO MEASURE ]`

### 5.6 The exhaustive surface

Enumerate all `2^9 = 512` block-granularity masks in fp32 and decompose on the Walsh basis.
Report the variance share per interaction order, the exact NAF, and what fixed-cardinality
fitting costs against the known full surface. `scripts/wp2b_exhaustive_coarse.py` → `[ TO MEASURE ]`

### 5.7 Severity cannot be matched across granularities — a calibration result

Attempt the severity-matched granularity sweep. Report the calibration (severity ratio per level)
and whether a matched sweep is achievable at each level, or why not.
`scripts/wp2a_calibrate_granularity.py` → `[ TO MEASURE ]`

### 5.8 The full pre-registered ablation, all arms

Report the complete ablation grid (pool size, rank, parameterisation, screening, linear-layer
freeze), every value against the precision floor. Do not quote only the interesting arms.
`scripts/08_ablations.py` → `[ TO MEASURE ]`

---

## 6. Predictions that failed and claims to withdraw

Keep a running log, in one place, of every prediction you pre-registered and every result you
reported and later retracted on further testing — with the reasoning that produced each. The
discipline is part of the exercise: a well-typed answer to a question you did not ask is the
failure mode this study is about. Record errors you make and catch here too.

→ `[ TO FILL IN AS YOU GO ]`

## 6b. Generality

State, in your own words, the scope of the proposition: it applies to **any linear surrogate
fitted on fixed-cardinality subset masks** (COAR, datamodels, TRAK). Protocols that sample several
rates partially break the degeneracy by Corollary 2. Discuss where the proposition stops — dense,
continuous perturbation designs (e.g. Gaussian σ with no retained count) are outside it — and
whether the transfer prediction (§5) extends to such designs as a falsifiable test.

## 6c. Generalisation: a second dataset and a second architecture

Repeat the core measurements on two further setups, **pre-registering the numeric criteria before
either model is trained**, and run them in a fixed order so a disagreement is attributable:

- **CIFAR-100 / ResNet-18** — changes the dataset, holding architecture and component definition.
- **CIFAR-10 / ViT-Tiny** — changes the architecture and component type (attention heads + MLP
  groups), holding the dataset.

Score each against your pre-registered criteria (transfer, aggregate term, coarse non-additivity,
concentration) and report which hold and which are not measurable. `scripts/wp89_score.py`
→ `[ TO MEASURE ]`

---

## 7. Limitations

State the limitations **your** study actually has once it is run. Expect to address at least:
the number of seeds; any confound between collection budget and seed; whether oracle-ρ
monotonicity holds on every seed; which component granularities you could and could not match on
severity; which ablation method you used (and that others are untested); and the range of α you
collected. Scope every claim to what you tested — never "neural computation is additive".

---

## 8. Reproducibility

Run the full pipeline and the test suite. Every figure you produce should ship its source data.
Map of claims to the code that produces them (scripts included in this repo):

| § | claim | script(s) |
|---|---|---|
| 2 | training and collection | `01_train_models.py`, `02_collect.py` |
| 2 | component count convention | `src/naca/components.py` |
| 3.5 | proposition, exact verification | `wp1c_identifiability.py` |
| 3.6 | synthetic identifiability demo | `wpA_synthetic_identifiability.py` |
| 4.1 | oracle-at-α control | `wp1d_structure_and_fig13.py` |
| 4.1b | multi-seed transfer + effect size | `wp10_seed_summary.py` |
| 4.1b | matched-budget diagnostic | `wp10_matched_budget_oracle.py` |
| 4.2 | gauge / raw-R² pitfall | `03_fit_linear.py`, `wp1a_gauge_diagnostic.py` |
| 4.3 | depth statistic | `04_fit_naca.py` |
| 5.1–5.3 | aggregate term | `wp1a_gauge_diagnostic.py`, `wp1b_aggregate_nonadditivity.py` |
| 5.4 | K-sweep, seeds, precision floor | `04_fit_naca.py`, `wp0_precision_floor.py`, `wp7_score.py` |
| 5.4b | coarsening–aliasing theorem | `wp6_aliasing.py` |
| 5.5 | severity pre-registration | `wp2c_bootstrap_severity.py` |
| 5.6 | exhaustive block surface | `wp2b_exhaustive_coarse.py` |
| 5.7 | granularity calibration | `wp2a_calibrate_granularity.py` |
| 5.8 | full ablation grid | `08_ablations.py` |
| 6c | CIFAR-100 and ViT-Tiny | `wp89_score.py` |
| — | full Setup A dump | `print_results.py` |
| — | figures | **you write these**; style helpers in `src/naca/plotting.py` |

### 8.1 Planned work that is not wired into the main claims

- **Counterfactual editing** (`scripts/05_edit.py`). Implemented and unit-tested; no editing claim
  is part of this study.
- **Non-linear baselines** (`scripts/06_baselines.py`, MLP / GBT surrogates). The comparison this
  study makes is between **sampling designs** at a fixed estimator, not between estimator classes.

---

## Notes for reproducers

- Run the pipeline end to end (`scripts/01_train_models.py` → `02_collect.py` → the `wp*`
  analysis scripts). See `README.md` for setup and commands.
- **Write your own figure code.** No plotting scripts for the paper figures are included; the
  style helpers in `src/naca/plotting.py` are available to build on.
- Report every quantity near the bf16 floor with that floor beside it.
- Build the PDF with `python paper/build_pdf.py` once your figures exist.

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
