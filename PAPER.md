# Ablation-Rate Transfer Failure in Component Attribution: A Structurally Invisible Aggregate Term

Numbers marked ✔ are measured on Setup A (ResNet-18 / CIFAR-10, N = 3,904 conv filters,
m = 50,000, E = 1,000), replicated on an independently trained second seed where stated.
**No number in this draft is pending.** Every claim links to the script that produces it, and
every experiment that was planned but not completed is reported as such in the section where it
would have appeared (§5.7) rather than omitted.

---

## Abstract (draft)

Component attribution explains a network's output by fitting a linear surrogate over random
ablations of its components. The surrogate is fitted at a single **ablation rate** -- the fraction
of components removed per mask -- and is known to degrade when evaluated at other rates. That
degradation has been read as evidence that the linear functional form is not expressive enough.

We show it is mostly not. An **oracle-at-alpha control** -- refitting the same linear model on
masks drawn at the evaluation rate -- finds that at alpha = 0.01 an additive model fitted there
reaches rho = **0.949**, the best fit anywhere on the grid, despite the *lowest* target variance
and less than half the training data; and that only **26%** of the degradation at alpha = 0.20
survives the control. Best-achievable additive quality is **strictly monotone** in ablation rate
(corr = -0.976), so the peak at the training rate that the literature reports **does not exist in
the network** -- the protocol manufactures it.

We then explain the transfer failure. We prove that under fixed-cardinality mask sampling -- the
standard protocol -- **any feature that is a function of the kept-component count is exactly
collinear with the intercept, hence unidentifiable**. Two consequences of this one proposition
have been reported separately or not at all: the known coefficient-shift ambiguity is the linear
case, and a large *aggregate* non-additivity is the quadratic case. The latter is substantial --
the marginal effect of a component falls **6x** as ablation deepens -- and is invisible to every
method fitted at one rate, including our own interaction model. Its coefficient moves **2,000x**
when the rate is allowed to vary. Pooling rates and adding **one scalar parameter** reduces
out-of-distribution error **11.6x**.

The transfer result is measured on **five independently trained seeds**: the drop in oracle rho
from alpha = 0.01 to 0.20 is **0.1896 +/- 0.0281** (Cohen's d_z = **6.74**), present on every
seed. Monotonicity holds on **4 of 5 as collected** -- one seed breaks by 1.3e-4 at the single
step with unequal training budgets, and all five are monotone at matched budget. It also holds on
**CIFAR-100** and on a **ViT-Tiny whose components are attention heads and MLP groups**, where
pooling likewise turns a negative out-of-distribution R^2 positive.

On *pairwise* structure we now claim less. Across five seeds the non-additive fraction at filter
granularity has a 95% CI that **crosses zero**, and it is confounded with collection budget
(0.029 and 0.018 at m = 50,000 against 0.0036-0.0065 at m = 25,000), so we make no claim about
its level. What is stable is the *coarse* result, which needs no estimator at all: at block
granularity the entire 2^9 counterfactual surface is enumerated and decomposed exactly in fp32,
giving **0.5406 +/- 0.0200** across five seeds.

All nulls carry a positive control on the same code path (0.83 where interactions are planted),
and every quantity near the bf16 floor is reported with that floor beside it -- including one that
falls *below* it on the second seed and is therefore withdrawn as a measurement. Section 6 lists
**sixteen** failed predictions, withdrawn claims, and errors we caught, including four
pre-registered in writing before measurement.

## 1. Introduction

Framing: attribution methods are measured on their own training distribution, and the standard
protocol has a blind spot that is structural rather than statistical. Contributions:

1. **A proposition** (§3) characterising exactly what fixed-cardinality sampling makes
   unidentifiable, unifying two previously separate observations.
2. **A control** (§4.1) that separates transfer failure from misspecification, and which
   *refuted our own prior conclusion* — see §4.1.
3. **A measurement**: the aggregate term is large, uniform across depth, and cheaply fixable.
4. **A null with a positive control**: pairwise structure is negligible, and we show the
   instrument would have found it if present.

---

## 2. Background

COAR (Shah, Ilyas & Mądry 2024) and the general subset-ablation surrogate family: datamodels
(Ilyas et al. 2022), TRAK (Park et al. 2023). Definitions: components, ablation vector `v`,
correct-class margin, α, the linear surrogate `g(v) = b + wᵀv`.

**Reproduction.** ✔ ρ = 0.836 at α_test = α_train, against the base paper's ≈0.75, on 69% more
components fitted from the same 50,000 counterfactuals (λ = 100 by held-out sweep; ρ flat to four
decimals over λ ∈ [0.1, 100]). `scripts/03_fit_linear.py`

---

## 3. Theory — fixed-cardinality blindness

**This section leads.** It is the organising result: everything measured later is a consequence.

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
property of the **sampling protocol**, not of the model, the network, or the estimator. No
increase in model capacity, sample size, or optimisation quality can recover an unidentifiable
direction.

### 3.2 Corollary 1 — two known problems are one problem

- **φ linear.** `φ(n) = n`. Shifting `w → w + c·1_N`, `b → b − c·n` leaves every training
  prediction unchanged. This is the coefficient-shift ambiguity of the additive surrogate.
- **φ quadratic.** `φ(n) = n(n−1)/2`. This is exactly the contribution of a *uniform pairwise
  interaction* `A_ij = a`, since `Σ_{i<j} a·v_i·v_j = a·n(n−1)/2`.

So the sign/level ambiguity and the aggregate interaction term are **the same phenomenon at
different orders of φ**. We are not aware of this connection being stated.

### 3.3 Corollary 2 — what breaks the degeneracy

Sampling two or more distinct cardinalities `n₁ ≠ n₂` restores identifiability; the rank gained
is (number of distinct cardinalities sampled) − 1, capped by the number of aggregate features.

### 3.4 Corollary 3 — what survives at fixed α

Only the *total* is pinned. Per-group kept-counts still vary, because a uniform k-subset
distributes randomly across a partition. With `G` groups the block-aggregate family
`{n_g(n_g−1)/2} ∪ {n_g·n_h}` has `G(G+1)/2` members, of which exactly **one** combination (the
one summing to the total) is unidentifiable and the rest are identifiable *from single-α data*.

### 3.5 Exact numerical verification

`scripts/wp1c_identifiability.py` ✔, on the real masks (N = 3,904, α = 0.10, n = 3,514):

| check | result |
|---|---|
| unique kept-counts | `{3514}`, variance 0 |
| `max‖q − φ(n)·1‖`, φ linear | **0.000e+00** (exact) |
| `max‖q − φ(n)·1‖`, φ quadratic | **0.000e+00** (exact) |
| `rank([1 | q])` | **1** |
| rank gain from `q` at fixed α | **0** |
| rank gain from `q` pooled over 3 cardinalities | **1** |
| block features: rank / count | **14 / 15** |
| null direction is the all-ones combination | `max|F_c·1| = 9.5e-11` |

The proposition also *predicts a failure we then observed*: fitting the 15 block features at
fixed α produces a singular normal-equation system. It is singular by construction, not by
numerical accident — a falsifiable prediction that held.

---

### 3.6 A synthetic demonstration of the proposition, with no network involved

§3.5 verifies the algebra on the real design. This isolates the *consequence* on data where the
truth is known: `y(v) = b + wᵀv + β·f(n) + ε`, with `β` known and nonzero, `N` = 3,904, and `w`,
`b` and the noise scale taken from the seed-1 fit so the demo sits in this paper's regime.
`β` is swept over three magnitudes spanning 10⁴. One estimator throughout.
`scripts/wpA_synthetic_identifiability.py`

The diagnostic is a projection, not a rank count: `resid_frac = ‖q_c − P_{V_c} q_c‖ / ‖q_c‖`,
the share of the aggregate column that survives projecting out the design. Zero means exact
collinearity. (Rank via an eigenvalue cutoff is unreliable here — the Gram's largest eigenvalue
is ~10¹⁶, so a "numerically nonzero" smallest eigenvalue can still be exact collinearity.)

| f(n) | protocol | distinct cardinalities | resid_frac | identifiable | β̂/β |
|---|---|---|---|---|---|
| n | A / B / B3 / C | 1 / 2 / 3 / 135 | **0** | **no** | — |
| n(n−1)/2 | A fixed | 1 | **0** | no | **0.000 exactly** |
| n(n−1)/2 | B two | 2 | 2.8e−08 | no | 0.969 |
| n(n−1)/2 | B3 three | 3 | 1.7e−02 | yes | 0.980 |
| n(n−1)/2 | C Bernoulli | 135 | 3.0e−03 | yes | 0.996 |

**At fixed cardinality β̂ = 0.000e+00 exactly**, at every magnitude and both orders — ridge's
minimum-norm representative, chosen by the regulariser rather than the data. Non-recovery is
structural, not a small-effect problem. Out-of-distribution transfer error at the largest β is
4.55 (linear) and 8.54 (quadratic) against ~0.60 for the identifiable protocols.

Two things this showed that we had not stated correctly beforehand, and both sharpen Corollary 2:

1. **The linear form is unidentifiable under every protocol.** `q = n = V·1` is exactly a column
   combination of the design, whatever the cardinality distribution. This is precisely the
   coefficient-shift ambiguity — a reparameterisation that leaves predictions unchanged, so it is
   harmless — but "two cardinalities recovers β" is false for it.
2. **Two cardinalities do not identify a quadratic aggregate.** With two distinct `n`, any `f(n)`
   lies in span{1, n}, and `n` is itself `V·1`. **Three is the minimum.** Corollary 2's "two or
   more" is therefore too weak for the quadratic term, which is the one this paper measures. Our
   own mitigation pooled four rates, so the 11.6× result in §5.2 is unaffected; the stated advice
   was not.

The trap worth naming: at two cardinalities β̂ still lands within 3% of the truth. That is the
minimum-norm choice happening to be near-correct — one aggregate column against 3,904 weight
columns — **not identification**. It is exactly the "ridge chooses a representative, silently"
failure this section warns about, now exhibited where the truth is known.

Finally, **three deliberate, well-separated cardinalities beat 135 Bernoulli-fluctuating ones**
by 5.6× on `resid_frac` (1.7e−02 vs 3.0e−03), which quantifies the conditioning argument in
`results/SECTION_scope_of_identifiability.md` §3.

![**Two failure modes, two different fixes.** *Left:* recovery of a known aggregate coefficient
against the number of distinct mask cardinalities in the design. At one cardinality the aggregate
column centers to exactly zero and the fitted value is 0.000e+00 regardless of the truth. *Right:*
the pairwise non-additive fraction as the candidate pool widens at constant rank/K, with the
synthetic positive control (planted pairwise structure, same code path) as a reference line and
the bf16 precision floor shaded. The right-hand design is full rank -- its failure is a diluted
sparse signal, not a degeneracy.](figures/two_failure_modes.pdf)

The figure places this beside the pairwise K-sweep, because the two failures
are different and have different fixes: **aggregate terms need varying cardinality; pairwise
terms need pool capacity.** The right-hand failure is not a design degeneracy at all — that
design is full rank — it is a sparse signal diluted across a widening pool.

---

## 4. Instruments

The three measurement tools, each introduced because a naive alternative gives the wrong answer.
We report the naive alternative and its failure in each case, because that is the evidence the
instrument is needed.

### 4.1 The oracle-at-α control — **and a retraction**

**Definition.** To test whether degradation at α_test is misspecification or transfer, fit a
*fresh* surrogate on masks drawn at α_test itself and evaluate at α_test. This removes transfer
entirely; whatever degradation survives is intrinsic to that ablation rate.

**We report this control partly because it refuted our own prior conclusion, and that is the
point of having it.** Following the standard reading, we first attributed the low-α branch of the
decay to signal starvation, citing `Var(f_M)` falling from 6.13 at α_train to 1.46 at α = 0.01 —
a plausible inference from a real measurement, and one we stated as a finding. The oracle
control overturns it ✔:

| α | m_train | transfer ρ | **oracle ρ** | transfer R²_oc | **oracle R²_oc** | Var(f_M) |
|---|---|---|---|---|---|---|
| 0.010 | 20,000 | 0.6036 | **0.9488** | 0.2350 | **0.7952** | 1.46 |
| 0.050 | 20,000 | 0.7776 | **0.8723** | 0.5612 | **0.7589** | 5.52 |
| 0.100 | 50,000 | 0.8362 | 0.8362 | 0.7013 | 0.7013 | 6.13 |
| 0.200 | 20,000 | 0.5335 | **0.7588** | −2.0371 | **+0.5732** | 2.04 |

The α = 0.01 row is the decisive one, and it is the row we were missing when we made the claim
we are retracting. Target variance there is **1.46**, the lowest in the table -- the strongest
case for the signal-starvation account. Yet the oracle reaches **ρ = 0.9488, the highest value
anywhere in the table**, above the 0.8362 at α_train itself, on 40% of the training masks. The
transfer surrogate manages only 0.6036 at the same rate. **148% of the apparent drop is transfer**:
once the surrogate is fitted at the rate it is evaluated at, there is no degradation left to
explain -- there is an improvement.

That is the opposite of what low target variance predicts, and it is not a marginal reversal. The
same pattern holds at α = 0.05, where the oracle reaches 0.872 on less than half the training
data and with lower target variance. Only at α = 0.20 does anything intrinsic survive: 0.077 of
the 0.303 ρ-drop, **74% transfer, 26% intrinsic**.

Read across the table, `Var(f_M)` and oracle ρ move in *opposite* directions over three of the
four rates. Our original inference took their co-movement in the transfer numbers as causal. It
was confounding: the ablation rate drives both, and it drives the surrogate's transfer error far
harder than it drives the signal.

### 4.1b The transfer result across five seeds

The oracle curve was measured on five independently trained ResNet-18 / CIFAR-10 networks.
Budgets differ (seeds 1-2 at m = 50,000, seeds 3-5 at m = 25,000) and are carried in the table.

| seed | m | ρ@0.01 | ρ@0.05 | ρ@0.10 | ρ@0.20 | monotone |
|---|---|---|---|---|---|---|
| 1 | 50,000 | 0.9488 | 0.8723 | 0.8362 | 0.7588 | yes |
| 2 | 50,000 | 0.8986 | 0.8339 | 0.8321 | 0.6620 | yes |
| 3 | 25,000 | 0.8997 | 0.8352 | 0.8306 | 0.7258 | yes |
| 4 | 25,000 | 0.8994 | 0.8294 | **0.8295** | 0.7357 | **no** |
| 5 | 25,000 | 0.9007 | 0.8401 | 0.8272 | 0.7169 | yes |
| **mean** | | 0.9094 | 0.8422 | 0.8311 | 0.7199 | |
| **95% CI** | | [.882,.937] | [.821,.864] | [.827,.835] | [.675,.764] | |

![**The decay is mostly transfer, not misspecification.** Transfer ρ (one surrogate fitted at
α_train) against oracle ρ (a fresh surrogate refit at each evaluation rate), over five
independently trained seeds; bands span the seeds. The two curves necessarily coincide at
α_train, because there the oracle *is* the transfer model. Away from it the transfer curve falls
off on both sides -- the shape the literature reports as evidence of misspecification -- while the
oracle curve is monotone decreasing in α. The vertical gap between the bands is transfer error;
the oracle band's own slope is what survives it.](figures/oracle_five_seeds.pdf)

**Effect size.** The paired within-seed drop in oracle ρ from α = 0.01 to α = 0.20 is
**0.1896 ± 0.0281**, 95% CI **[0.155, 0.225]**, **Cohen's d_z = 6.74**. The drop is present on
every seed without exception.

**Monotonicity, reported in a fixed order.**

> **Primary (as collected): 4 of 5 seeds are strictly monotone. Seed 4 breaks by -0.000132,
> between α = 0.05 and α = 0.10.**
>
> **Secondary (all seeds refit at matched m = 10,000): 5 of 5 are monotone.**

The secondary result explains the primary break; **it does not overturn it, and we do not quote
it in place of the primary count.** The as-collected design is what the criterion was registered
against, and a post-hoc refit that happens to be favourable is a diagnostic, not a re-score.

The diagnostic is nonetheless informative. α = 0.05 is fitted on an extra pool of m = 10,000
while α = 0.10 is the main pool (m = 25,000 or 50,000); more data raises ρ, so the budget
asymmetry pushes in exactly the direction of the observed inversion. Refitting α = 0.10 on a
random 10,000-mask subsample lowers **every** seed's value substantially — 0.836 → 0.769 on
seed 1, 0.832 → 0.767 on seed 2 — and the ρ(0.05) − ρ(0.10) gap goes from +0.0018…+0.036 to
+0.050…+0.103 across the board, with seed 4 moving from −0.000132 to +0.050. The effect removed
is real and sizeable, not a rounding artifact that happens to rescue one seed.
`results/FINDING_seed_monotonicity.md`, `scripts/wp10_matched_budget_oracle.py`

An inference from a correlated diagnostic (variance falls, accuracy falls) survived until it was
tested against a control that could refute it. We recommend the control be standard in this
literature. `scripts/wp1d_structure_and_fig13.py`

### 4.2 Gauge-invariant R²

Raw R² is not comparable across α under the proposition of §3: a shift along an unidentifiable
direction moves it arbitrarily. Measured ✔: raw R² at α = 0.01 is **−56.4** for a model that
explains **23.5%** of the variance. We report R² with a per-cell offset removed alongside the raw
value, and note that **ρ is invariant to this entire effect** (0.723 → 0.732 across all our
models ✔) — so the field's headline metric is blind to it.

### 4.3 Group-confined refits, not share-weighted averages

The natural statistic for "where does non-additivity live" — per-example NAF weighted by each
example's pool share in a layer group — **cannot detect localisation at all**. On synthetic data
with interactions planted only in the deep half it returns ≈0.47 for *every* group ✔. It is a
weighted mean of a quantity that barely varies across examples. Confining the pool to one group
and refitting returns `0.000006 / 0.000001 / −0.000017 / 0.0033 / 0.208` across stem→layer4 ✔ —
exactly where planted. `results/FINDING_h3_measurement.md`

### 4.4 Positive control

Every null in this paper is accompanied by the same pipeline run on synthetic counterfactuals
generated from a known linear + indefinite-quadratic model. It reports NAF = **0.83** ✔ where the
real network gives 0.0005 at the same configuration (K = 100). That last number must never be
read on its own: it sits **2.0x its measured precision floor of 0.00025** and is therefore not an
interpretable positive measurement. The interpretable filter-level number is 0.029 at K = 25
(114x the floor), where structure is concentrated; see §5.4, where the result is reported as a
range rather than a single number ✔. It also recovers 90–95% of a planted pool ✔ and localises depth correctly ✔.
**This figure precedes any real-data null in the paper.**

---

## 5. Results

### 5.1 The aggregate term is large, and structurally invisible

The additive model is effectively blind to ablation rate ✔: across the α grid the true mean
margin spans **9.139** while the predicted mean spans **0.101**. The true response saturates —
marginal effect per component runs `0.0162 → 0.0032`, a **6× decline**; straight-line fit
R² = 0.933, quadratic R² = **0.995** ✔.

That the term is *unidentifiable* rather than *small* is shown directly ✔: `mean(w)`, the
direction it lives in, is **4.59e-06** fitted at one α and **9.99e-03** fitted on pooled α — a
**2,000× jump** from letting the rate vary.

We also rule out the alternative explanation. The gauge constant *required* to produce the
observed level error is 1.48e-02; the one actually present is 6.51e-04, a **23× shortfall** ✔, and
a single constant explains only 21.3% of it. Ridge with an unpenalised intercept gauge-fixes
itself (proof in appendix); the residual is far too small to be the cause.

### 5.2 One scalar parameter, 11.6× less transfer error

Scored only on α values held out from **all** models (0.01, 0.03, 0.075, 0.125, 0.15) ✔:

| model | ρ | raw R² | R²_oc |
|---|---|---|---|
| **A** linear, single-α (COAR) | 0.7234 | **−13.86** | 0.3664 |
| **B** linear, pooled α | 0.7288 | −1.39 | 0.4581 |
| **C** B + one global term | **0.7322** | **−0.285** | **0.4642** |

Per-cell: α = 0.01 goes −56.4 → −3.2; α = 0.03 goes −11.6 → **+0.30**. Level tracking: true
spread 9.139, model A 0.101, model C 10.083 ✔.

**Trade-off, stated:** in-distribution at α_train, single-α A remains best (R²_oc 0.701 vs C's
0.657). A is fitted on 50,000 samples at exactly that rate; B and C spread capacity across three.
The choice is in-distribution sharpness versus out-of-distribution validity.

### 5.3 The aggregate term has no depth structure

Corollary 3 makes this testable for free. Fifteen block-aggregate coefficients do no better than
one global scalar ✔ (R²_oc 0.4638 vs 0.4642; raw R² −0.287 vs −0.282), and the 14 identifiable
single-α directions carry no usable signal (model D ≈ model A; raw R² actually worsens to −30.1).
**The aggregate non-additivity is uniform across depth.**

### 5.4 Filter-level structure: concentrated on one seed, not established across five

On seed 1 the pattern is concentration, and the K-sweep below is the evidence for it. **Across
five seeds that claim does not survive**, and the sub-section is titled accordingly: the level is
confounded with collection budget and its 95% CI crosses zero. We keep the single-seed analysis
because the capacity-matched control it contains is what rules out the obvious alternative
explanation, and because it is the measurement the later seeds fail to reproduce. **Never quote a
single number here as a null, or as a positive.**

Non-additive fraction at filter granularity on **seed 1**, as a function of the candidate pool
size K, under two protocols:

| K | NAF, fixed rank 8+8 | rank/K | NAF, **capacity-matched** (r=0.3K) | rank/K | params/example |
|---|---|---|---|---|---|
| 25 | 0.02902 | 0.64 | **0.02902** | 0.64 | 400 |
| 50 | 0.00537 | 0.32 | **0.00154** | 0.60 | 1,500 |
| 100 | 0.00051 † | 0.16 | **0.00061** † | 0.60 | 6,000 |
| 200 | 0.00016 † | 0.08 | **0.00002** † | 0.60 | 24,000 |

† *At or below the bf16 precision floor of 0.00025 for this seed — not resolvable, and not to be
quoted without it. See the floor table below.*

**Replication across five independently trained seeds.** Capacity-matched arm, at the budget
each seed was collected at. **Budgets are not pooled silently:** seeds 1-2 were collected at
m = 50,000 and seeds 3-5 at m = 25,000, so the budget is carried in every row.

| seed | m | NAF(K = 25) | × floor | NAF(K = 100) | × floor | ratio | floor |
|---|---|---|---|---|---|---|---|
| 1 | 50,000 | 0.02902 | 113.9× | 0.00061 † | 2.4× | 47.3× | 0.000255 |
| 2 | 50,000 | 0.01834 | 67.9× | 0.00028 † | 1.0× | 66.3× | 0.000270 |
| 3 | 25,000 | 0.00651 | 20.1× | 0.00192 | 5.9× | 3.4× | 0.000324 |
| 4 | 25,000 | 0.00358 | 11.3× | 0.00127 | 4.0× | 2.8× | 0.000317 |
| 5 | 25,000 | 0.00470 | 15.4× | 0.00136 | 4.5× | 3.5× | 0.000304 |
| **mean ± sd** | | **0.0124 ± 0.0110** | | **0.00109 ± 0.00065** | | | **0.000294** |

† *At or below the precision floor for that seed — not resolvable, and never to be quoted without
the floor beside it.*

**The magnitude of filter-level concentration does not replicate, and the failure tracks budget.**
The 95% CI on NAF(K = 25) over five seeds is **[-0.0012, 0.0261]** — it crosses zero. The two
50,000-mask seeds give 0.029 and 0.018; the three 25,000-mask seeds give 0.0036-0.0065, a factor
of 3-8 lower. **Budget and seed are confounded in this comparison and we do not separate them**:
we did not collect a 50,000-mask run for seeds 3-5. The concentration *ratio* shows the same
split — 47× and 66× at m = 50,000 against 2.8-3.5× at m = 25,000.

We therefore make no claim about the level of filter-level non-additivity. What survives across
all five seeds is the coarse-granularity result, which is exact and needs no estimator:
**NAF_coarse = 0.5406 ± 0.0200**, 95% CI **[0.5158, 0.5654]**, with every seed in 0.517-0.567.

**The precision floor, measured independently on every seed.** All counterfactuals are collected
under bf16 autocast; the floor is the bf16 noise variance as a share of the residual variance the
interaction model competes for, obtained by recomputing 3,000 of the same masks in fp32
(`scripts/wp0_precision_floor.py`). Across five seeds it is **0.000255-0.000324**, a spread of
27%, as it should be: the floor is a property of the arithmetic, not of the network. That
stability is what licenses comparing NAF values against it at all.

![**The K = 100 configuration, against its own noise floor.** Non-additive fraction across the
evaluation grid on seed 1, with the measured bf16 precision floor shaded. Only the value at
α_train is a candidate measurement, and it sits 2.0× the floor. Away from α_train the interaction
model transfers *worse* than the linear one it is built on (NAF falls to −0.0125): off-rate, the
extra capacity costs rather than helps.](figures/naf_vs_alpha_floor.pdf)

**And the K = 100 value must never be quoted alone.** It sits at 1.0-5.9× its own floor depending
on the seed, and on seed 2 it is *beneath* it (0.00028 against 0.000270). A quantity that is
marginal on three networks and unresolvable on a fourth is not a null result, not a small effect,
and not a measurement. It is the number at which our instrument stops working. Every appearance
of it in this paper is accompanied by the floor for that seed, and no claim rests on it.

This is also why the exhaustive block-level surface (§5.6) is collected in **fp32**: 512 masks
costs about 70 seconds, so at that granularity precision is free and the result carries no
noise-floor caveat at all.

**Why this is concentration and not under-capacity.** The obvious objection is
that the fixed-rank sweep is confounded: holding r_U + r_P = 16 while K grows
spreads a rank-16 matrix over more components, so expressiveness per pair
collapses (rank/K falls 0.64 -> 0.08). We therefore ran a second sweep holding
rank/K ~ 0.6, so capacity per component is constant. **NAF still collapses,
1,450x from K = 25 to K = 200.** Two details make the argument:

* At **K = 25 the two protocols coincide exactly** (0.02902 both), as they must:
  400 parameters against 300 pairs means the fixed-rank arm was already
  near-full-rank there. That agreement is a consistency check on the sweep.
* At **K = 50 the capacity-matched arm is *lower* than the fixed-rank arm**
  (0.00154 vs 0.00537). More capacity, worse held-out performance. That is
  overfitting, and it rules out under-capacity as the explanation: if the pool
  contained structure the larger model was failing to reach, adding capacity
  would have helped.

So the additional components admitted as K grows contribute pairs that are noise.
The interaction structure that exists at filter granularity lives in a small
number of components -- on the order of tens out of 3,904 -- and is diluted, not
revealed, by widening the pool.

The estimator is not at fault: the identical code path reports NAF = 0.83 on
synthetic counterfactuals with planted interactions, recovers 90-95% of a planted
pool, and localises depth correctly.

Two arms of the pre-registered ablation cut against our own design choices and
are reported as such: PSD-only beats the indefinite parameterisation on real data
(0.00106 vs 0.00051), though indefinite wins on synthetic data with planted
indefinite structure (0.984 vs 0.953); and residual screening is marginally
*worse* than marginal or union screening (0.00051 vs 0.00083 / 0.00086).
**Every number in this paragraph lies within 4x the bf16 precision floor of 0.00025**, so these
are rankings among values at the edge of measurability, not effect-size comparisons. We report
them because they were pre-registered, and we decline to draw design conclusions from them.

### 5.4b Reconciling the two granularities — the coarsening–aliasing theorem

Block-level NAF is **0.567**, with variance spread over every order; filter-level structure is
**concentrated in tens** of 3,904 components. A reviewer will ask whether those are consistent.
They are, and the answer is a theorem rather than an argument.

**Setup.** Let `π : [N] → [G]` assign filters to blocks, `B_g = π⁻¹(g)`. A block mask `b` induces
`m_i = b_{π(i)}`, so the block surface is the *restriction* of the filter surface to the subgroup
of `{0,1}^N` on which each block is constant. In the ±1 Walsh basis,

```
χ_T(m) = ∏_{i∈T} (−1)^{m_i} = ∏_g (−1)^{b_g·|T ∩ B_g|} = χ_{S(T)}(b),
        S(T) = { g : |T ∩ B_g| is odd }
```

Only **parity** survives, so each block coefficient is a *signed* sum of filter coefficients:

> **ĝ(S) = Σ_{T : S(T)=S} f̂(T)**

**Corollary 1 — coarsening cannot manufacture interaction.** For `|T| = 1`, `S(T)` is a
singleton. A purely additive filter surface therefore coarsens to a purely additive block
surface. **NAF_block = 0.567 cannot be an artifact of aggregating additive structure**; it
requires genuine multi-filter interaction. This is the sharp answer to the consistency question,
and it is a property of the map, not of our data.

**Corollary 2 — within-block pairs are invisible.** For `T = {i,j}` with `π(i) = π(j)`, the
intersection is even, so `S(T) = ∅`: the interaction aliases into the constant term. Block
granularity is structurally blind to within-block interaction.

**Corollary 3 — downward aliasing requires a shared block.** `|S(T)| ≤ |T|`, with strict
inequality only when some block holds ≥2 members of `T`. Order-3 filter structure can surface as
order-1 block structure, but only via a shared block; high-order block coefficients cannot be
manufactured from low-order filter structure.

**Corollary 4 — aliasing can cancel.** The sum is signed over an exponentially large pre-image,
so a *small* block coefficient is not evidence of small filter-level structure. Measured: **8 of
8** supergroup pre-images contain coefficients of both signs. Cancellation is the normal case
here, not a caveat, and it bounds what may be inferred downward from coarse measurements.

**Validation.** Coarsening the 9 blocks into 3 supergroups and predicting the supergroup spectrum
from the block spectrum via the map gives **max absolute error 0.000e+00** against the spectrum
computed directly from the restricted surface — exact, as an identity over sums of the same
floats must be. All four corollaries verified on the same map (`scripts/wp6_aliasing.py`).

**The reconciliation, tested.** Corollary 2 makes it falsifiable: if the ~25 structure-carrying
filters sat inside one block, their pairwise structure would be *invisible* at block level and
the large NAF_block would need another source. Block membership of the top-25 residual-screened
filters: **6 of 9 blocks** occupied, largest single share **28%**, and only **53 of 300** pairs
(18%) within-block — so 82% of the pairs do reach the block spectrum. **Spread, as required.**
§4 and §5 are consistent with evidence, not assertion.

![**Where residual screening looks.** Share of the screened candidate pool in each depth group,
divided by that group's share of the network. The stem is enriched 3.65× and `layer4` is depleted
to 0.51× — the components that carry residual structure sit near the input, not near the
classifier. Enrichment is a ratio of counts, so unlike a non-additive fraction it has no precision
floor.](figures/depth_enrichment.pdf)

One unplanned observation from the same table: the stem is enriched **15×** over expectation
(6 observed vs 0.41) while layer4 holds 1 against 13.1. That independently reproduces, from a
different statistic, the depth inversion found by residual screening (stem 3.65×, layer4 0.51×).

### 5.5 PRE-REGISTERED: apparent non-additivity tracks ablation severity, not granularity

**Written before the exhaustive spectrum was computed.** Committed at this point in the project
so the interpretation cannot be fitted to the number afterwards; the number is reported in §5.6
whatever it turns out to be.

Three measurements already in hand point at one axis, and it is not granularity:

| measurement | result |
|---|---|
| marginal effect per component, shallow to deep ablation (filter level) | falls **6x** (0.0162 -> 0.0032) |
| oracle rho -- best achievable additive fit *at* each rate | **strictly monotone** in alpha: 0.949, 0.872, 0.836, 0.759 |
| damage vs units ablated (layer level, wp2a) | **saturates by k=2**: mean abs delta 10.57 at 2 of 17 units, 11.38 at 16 of 17 |

The third is the sharpest. Ablating two of seventeen layers does essentially as much damage as
ablating sixteen. Once the network is broken it cannot be broken further, so the response
flattens -- and a flattening response is, by definition, not additive.

**The prediction.** The exhaustive block surface will show large high-order energy. We state in
advance that this is **expected and is a severity effect, not a granularity effect**. Block units
are large (up to 2,048 filters), so even a single-unit ablation is deep, and the whole surface
sits in the saturated regime where additivity must fail. It would be a mistake -- and the mistake
this section exists to forestall -- to read large high-order energy at block level as "coarse
components interact more".

**The unifying claim:** *apparent non-additivity tracks ablation severity.* One axis, four
measurements: the 6x marginal decay, the monotone oracle curve, the wp2a saturation, and the
exhaustive spectrum.

**What would refute it.** The claim is falsifiable on the same 512-mask surface, and §5.6 reports
the test rather than only the spectrum:

* If additive-model quality, measured **stratum by stratum**, is roughly flat across mask
  cardinality while high-order energy is large, then non-additivity is a property of the
  component *type* and the severity account is wrong.
* If additive quality degrades monotonically with the severity of the stratum, the severity
  account holds and granularity is a confound rather than a cause.

This is why §5.6 reports the spectrum against **measured damage** (mean abs delta margin at each
cardinality), not against interaction order alone. Order and severity are different axes and the
literature routinely conflates them.

---

**OUTCOME (added after the measurement; everything above is the pre-registration as written).**

The block-level test **could not decide the question**, for a reason the pre-registration did not
anticipate: at block granularity **severity barely varies**. Measured damage runs 7.51 at k=1 to
11.88 at k=5 and is flat thereafter -- roughly a 12% spread across the well-populated strata,
because one block already breaks the network. A test correlating a near-constant regressor
against stratum quality has almost no leverage.

Two further problems, both found on inspection rather than assumed:

* Stratum quality is confounded with **fitting weight** -- strata contain 9/36/84/126/126/84/36/9
  masks, so OLS is dominated by the middle. A stratum-balanced refit was run to remove this.
* The resulting correlation is **not interpretable**: it is -0.167 over all eight strata but
  **+0.464** with the two thin strata dropped, so its sign is set by a single point (k=8,
  R^2 = -11.1). No value is quoted from it. See Sec 6, items 8 and 9.

**Where the question is actually settled is at filter granularity**, where severity does vary
over a real range: best-achievable additive quality against ablation rate gives
**corr = -0.976** across four measured rates. That is the evidence for the severity account. The
block-level test is a consistency check that neither confirms nor contradicts it.

**The pre-registered claim is therefore neither confirmed nor refuted by this experiment.** It is
supported where it can be measured and untestable where it cannot -- and the untestability is
itself the finding: **at coarse granularity, severity cannot be varied independently of
cardinality.**

### 5.6 The exhaustive surface

All 2^9 = 512 masks at block granularity, collected in **fp32** so no precision floor applies,
decomposed on the orthogonal Walsh basis (Parseval-exact; cumulative R^2 monotone by
construction):

![**The exhaustive block surface, decomposed exactly.** Share of coefficient mass at each
interaction order, with the cumulative share overlaid. Order 1 -- everything an additive
surrogate can represent -- carries 3.6% of the mass; the bulk sits at orders 3-6. This is the one
measurement in the paper with no estimator, no candidate pool, no low-rank assumption and no
precision floor: all 512 masks are enumerated and the decomposition is
exact.](figures/block_order_spectrum.pdf)

| order | terms | variance share | cumulative R^2 |
|---|---|---|---|
| 1 | 9 | 0.5586 | 0.5586 |
| 2 | 36 | 0.2504 | 0.8089 |
| 3 | 84 | 0.1186 | 0.9276 |
| 4 | 126 | 0.0485 | 0.9761 |
| 5-9 | 257 | 0.0239 | 1.0000 |

**Exact NAF = +0.567**, with no estimator, no candidate pool, no low-rank assumption, and no
identifiability problem -- enumerating every cardinality is precisely what breaks the
proposition of Sec 3.

**What fixed-cardinality sampling costs, against known truth.** Fitting on masks of a single
cardinality -- what the standard protocol does -- and scoring on the whole surface:

| fitted on k units | additive R^2, own stratum | additive R^2, **full surface** |
|---|---|---|
| 2 | 0.594 | **-0.600** |
| 3 | 0.606 | -0.148 |
| 4 | 0.638 | 0.202 |
| 5 | 0.639 | **0.318** |
| 6 | 0.623 | 0.178 |
| 7 | 0.664 | -0.153 |

Every arm looks comparably good on its own stratum (0.59-0.66) and ranges from -0.60 to +0.32 on
the surface it is meant to describe. In-distribution fit is uninformative about the object.

**Stratum-level analysis is reported but nothing is inferred from it.** Additive R^2 by
cardinality traces a clean inverted-U under OLS
(-0.84, 0.28, 0.51, 0.58, 0.60, 0.49, -0.01, -8.89), but the strata are unequally sized
(9/36/84/126/126/84/36/9) so the fit is dominated by mid-cardinality masks. Under a
stratum-balanced refit the curve does not survive
(0.10, -1.33, -1.93, -1.04, -0.28, 0.20, 0.05, -11.10). With one stratum two orders of magnitude
from the rest, no eight-point correlation over these values is interpretable -- see Sec 6, items
8 and 9. We report the OLS curve as a description of that fit and stop there.

**Severity cannot be varied independently of cardinality at this granularity.** Measured damage
runs 7.51 at k=1 to 11.88 at k=5 and is flat thereafter -- a ~12% spread across the
well-populated strata. One block already breaks the network. This is a finding about coarse
units, and it is why the severity question is settled at *filter* granularity
(corr(alpha, oracle rho) = -0.976, four measured rates, real leverage) rather than here.

### 5.7 Severity cannot be matched across granularities — a negative calibration result

We had planned a full granularity sweep: NAF as a function of component granularity, baselined
against model C, with severity-matched and fraction-matched protocols reported separately. **We
report the calibration that was supposed to enable it, and why it does not.**

To compare granularities fairly, ablating *k* coarse units must inflict the same damage as
ablating αN filters. We searched over integer *k* at each level for the ratio of mean absolute
margin change to the filter-granularity target, accepting a ±10% band
(`scripts/wp2a_calibrate_granularity.py`):

| level | units | best *k* | component fraction | severity ratio | matched? |
|---|---|---|---|---|---|
| filter | 3,904 | 390 | 0.0999 | 1.000 | reference |
| filter_group | 488 | 24 | 0.0492 | 0.545 | **no — far below** |
| layer | 17 | 1 | 0.0567 | 1.166 | **no — outside band** |
| block | 9 | 1 | 0.1014 | 1.045 | yes |

At layer granularity the severity ratio jumps from **1.166 at k = 1 to 1.535 at k = 2** and never
returns; there is no integer *k* inside the band. This is not a search failure. **The unit is
quantised too coarsely for the target damage to be reachable at any *k*.** Ablating a single
layer already overshoots; ablating none undershoots. `filter_group` fails in the other direction,
topping out at 0.545 of target severity.

Two consequences, and we state both:

1. **A severity-matched granularity sweep is not available in this network at these levels.** Any
   cross-granularity NAF comparison here confounds granularity with severity, which §5.5 shows is
   the dominant axis. Reporting one would mean reporting a confound. We do not.
2. **The block level, which we do use (§5.6), is the one level that matches** (ratio 1.045, inside
   the band). That is fortunate rather than designed, and it is why the exhaustive block surface
   is the granularity result we report.

The remaining question — how NAF varies with granularity at matched severity — needs either an
architecture with finer-grained coarse units or a fractional-ablation scheme that makes severity
continuous. Both are outside this paper. We flag this because the sweep was pre-registered and a
reader is entitled to know it was attempted, calibrated, and abandoned for a stated reason rather
than quietly dropped.

---

### 5.8 The full pre-registered ablation, all nineteen arms

§5.4 quotes two arms of this table. The complete grid is reported here, because quoting the two
that happen to be interesting while leaving seventeen in a CSV is exactly the selective reporting
this paper complains about elsewhere. Seed 1, NAF at α_train, with each value expressed as a
multiple of that seed's bf16 precision floor (0.000255). `scripts/08_ablations.py`,
`runs/.../ablation_table_full.csv`

| arm | K | rank | params | NAF | × floor |
|---|---|---|---|---|---|
| **pool size, fixed rank 8+8** | | | | | |
| K25 | 25 | 8+8 | 400 | **0.029023** | **113.8×** |
| K50 | 50 | 8+8 | 800 | 0.005366 | 21.0× |
| K100 | 100 | 8+8 | 1,600 | 0.000512 † | 2.0× |
| K200 | 200 | 8+8 | 3,200 | 0.000155 † | 0.6× |
| **pool size, capacity-matched (rank/K ≈ 0.6)** | | | | | |
| Kcm25 | 25 | 8+8 | 400 | **0.029023** | **113.8×** |
| Kcm50 | 50 | 15+15 | 1,500 | 0.001541 † | 6.0× |
| Kcm100 | 100 | 30+30 | 6,000 | 0.000614 † | 2.4× |
| Kcm200 | 200 | 60+60 | 24,000 | 0.000024 † | 0.1× |
| **rank, at fixed K = 100** | | | | | |
| r2 | 100 | 2+2 | 400 | 0.001205 † | 4.7× |
| r4 | 100 | 4+4 | 800 | 0.000517 † | 2.0× |
| r8 | 100 | 8+8 | 1,600 | 0.000512 † | 2.0× |
| r16 | 100 | 16+16 | 3,200 | 0.000553 † | 2.2× |
| **parameterisation** | | | | | |
| indefinite (A = UU' − PP') | 100 | 8+8 | 1,600 | 0.000512 † | 2.0× |
| PSD only (A = UU') | 100 | 16+0 | 1,600 | 0.001062 † | 4.2× |
| **screening** | | | | | |
| marginal | 100 | 8+8 | 1,600 | 0.000826 † | 3.2× |
| residual | 100 | 8+8 | 1,600 | 0.000512 † | 2.0× |
| union | 100 | 8+8 | 1,600 | 0.000862 † | 3.4× |
| **linear layer** | | | | | |
| frozen | 100 | 8+8 | 1,600 | 0.000512 † | 2.0× |
| unfrozen | 100 | 8+8 | 1,600 | **−0.001544** † | negative |

† *Within 5× the precision floor, or below it.*

**The honest summary of this table is that seventeen of nineteen arms are measuring noise.**
Only K = 25 (113.8×) and K = 50 (21.0×) sit clear of the floor. Everything else lies between
0.1× and 4.7×, a band inside which we do not defend any ordering. That is why §5.4 reports the
K-sweep and declines to draw design conclusions from the parameterisation and screening arms
even though both were pre-registered.

Three observations that survive being read at this resolution:

* **Rank does not matter at K = 100.** Going 2+2 → 16+16, an 8× increase in parameters, moves NAF
  from 0.001205 to 0.000553 — *down*, and non-monotonically. If there were pairwise structure in
  that pool the extra capacity would have found it.
* **Unfreezing the linear layer makes the model worse than linear**, NAF = −0.001544. The
  interaction model, given freedom to refit the additive part jointly, overfits and loses to the
  frozen baseline out of sample. This is why `freeze_linear: true` is the default everywhere in
  this paper, and it is a result rather than a convention.
* **The two protocols agree exactly at K = 25** (0.029023 both), as they must: 400 parameters
  against 300 pairs means the fixed-rank arm was already near-full-rank there. That agreement is
  a consistency check on the sweep, not a coincidence.

---

## 6. Predictions that failed and claims we withdrew

Collected in one place rather than scattered, because the pattern is the point.
Every item below was either pre-registered before the measurement or reported as a
result and then retracted on further testing. Dates are the order they occurred.

**Refuted pre-registrations** -- stated in writing before the number existed:

1. **The gauge degeneracy causes the OOD collapse.** *Predicted; refuted.* The
   gauge constant required to produce the observed level error is 1.48e-02; the
   one actually present is 6.51e-04, a **23x shortfall**. Ridge with an
   unpenalised intercept gauge-fixes itself, provably. The real cause is a
   saturating aggregate term.
2. **Large high-order energy at block level is a severity effect.** *Pre-committed
   in writing before computing the spectrum; not supported.* The test came back
   flat, then -- see item 8 -- turned out to have no power either way.
3. **Order-2 structure is the uniform aggregate term seen from above.**
   *Predicted quantitatively (beta_ij ~ a n_i n_j, testable against model C's
   fitted a); refuted.* R^2 = **-1.19**, CV 0.705, and a is off by 3.8x. Order-2
   carries real per-pair detail. Independently refuted at layer level (R^2 = -2.06).
4. **The indefinite parameterisation is necessary.** *Design rationale; not
   supported on real data.* PSD-only scores 0.00106 against indefinite's 0.00051
   -- both within 4x the precision floor of 0.00025, so the ordering is not a
   measurement we defend. The design is vindicated on synthetic data with planted
   indefinite structure (0.984 vs 0.953) -- the network simply does not have that
   structure.

**Claims we reported and then withdrew:**

5. **The low-alpha decay is a signal-to-noise artifact.** Reported as a finding,
   following the base paper's framing and citing Var(f_M) falling 6.13 -> 1.46.
   The oracle-at-alpha control refuted it outright: at alpha = 0.01, with the
   *lowest* target variance and less than half the training data, the at-rate fit
   is the **best on the grid** (rho = 0.949). Low variance means little to
   predict, not little predictability.
6. **Variance-by-order from a truncated Moebius expansion.** The output falsified
   itself: R^2 = -29.8, -156.5, -211.7 for orders 0, 1, 2 -- non-monotone, which
   is impossible for a variance decomposition. The Moebius basis is not
   orthogonal, so a truncated exact expansion is not the best order-<=k model.
   Rewritten on the Walsh basis (Parseval-exact, monotone by construction) with
   four tests added.
7. **The K-sweep collapse is a capacity artifact.** Reported as such, with the
   rank/K argument. The capacity-matched re-sweep refuted it: NAF still collapses
   1,450x at constant rank/K, and at K = 50 the *higher*-capacity arm scores
   lower. The correct reading is concentration, not measurement failure (Sec 5.4).
8. **The block-level severity test is informative.** Reported to a collaborator as
   "NOT SUPPORTED -- non-additivity tracks component type", then downgraded within
   the hour on noticing the instrument had no power: severity spans only ~12%
   across the well-populated strata, because damage saturates almost immediately
   at block granularity. Bootstrapping then showed the balanced correlation's
   *sign* is set by a single stratum (k=8, R^2 = -11.1): -0.167 over all eight
   strata, **+0.464** with the two thin strata dropped. The statistic is
   uninterpretable and no value is quoted from it.
9. **The inverted-U in stratum R^2 survives reweighting.** Claimed on the basis
   that corr(stratum size, R^2) fell only 0.566 -> 0.409 under stratum-balanced
   weights. Wrong twice: the balanced curve is
   `[0.10, -1.33, -1.93, -1.04, -0.28, 0.20, 0.05, -11.10]`, visibly not an
   inverted-U; and that rescuing correlation is computed over the same eight
   points with the same outlier that had just invalidated item 8. **Neither fit
   supports a claim about intrinsic stratum difficulty at block granularity.** The
   OLS curve is reported as a description of that fit and nothing is inferred from
   it.

10. **The seed-2 magnitude replicates the seed-1 magnitude.** The pre-registration
    (`results/PREREG_wp7_seed2_full.md`) fixed a threshold of 0.6 x 0.029 = 0.0174 in advance;
    the measured value was 0.01834, which clears it. **We decline the confirmation.** It clears
    by 5%, and the two seeds differ by 37% in level (0.029 vs 0.018) while agreeing on shape
    (concentration ratio 47x vs 66x). A pre-registered rule scored literally would have let us
    report replication of the magnitude; the evidence supports replication of the *phenomenon*
    only, so the number is reported as a range. Pre-registration constrains us against
    post-hoc rationalisation; it does not license claiming more than the measurement carries.

**Errors we made and caught:**

11. **A "precision floor" of 8.69.** The seed-2 precision stage was invoked without `--config`,
    so it defaulted to the development config, loaded the **seed-0 checkpoint**, and differenced
    it against seed-1 margins. The reported bf16 noise was the difference between two different
    networks. It was caught by a dimensional check -- a floor is `noise/residual`, a variance
    share, and 8.69 is not a share -- not by inspection of the value. Nothing reached the draft.
    The fixes are a hard guard that verifies the checkpoint reproduces the run's stored clean
    margins (mean deviation 0.005 for the right model, 2.00 for the wrong one, against a margin
    sd of 4.33), config defaulting to the run's own `config_resolved.yaml`, and a scorer that
    refuses a floor outside [0,1). Full record: `results/BUG_precision_floor_wrong_checkpoint.md`.

    We include this because it is the same failure shape as the paper's main result: **machinery
    returning a well-typed answer to a question it was not asked.** A fixed-cardinality design
    also does not fail loudly -- it returns a fitted model, a good in-distribution score, and one
    coefficient direction set by the regulariser rather than the data. In both cases the thing
    that caught it was a cheap check independent of the quantity of interest.

12. **A granularity sweep we planned, calibrated, and could not run.** Severity cannot be matched
    at layer or filter-group granularity in this network at any integer number of ablated units
    (§5.7). We report the calibration and the abandonment rather than dropping the section.

13. **A related-work framing built on an abstract, corrected on the full text.** Unable to obtain
    Mammeri & Gagne (2025) -- OpenReview returns HTTP 403 to every programmatic client and no
    arXiv version exists -- we inferred from Semantic Scholar's full-text-derived summary
    ("replacing the linear counterfactual estimator") that their sampling protocol was COAR's, and
    framed their setting as a worked example of our barrier. **Wrong.** Their Algorithm 1 draws
    `sigma ~ N_m(0, tau^2 I_m)`: dense and continuous, with no retained count, so Proposition 1
    does not apply to them at all. The draft had stated the dependency explicitly and specified
    the repair in advance; on obtaining the PDF we executed it, and they are now cited as
    independent evidence that perturbation design governs identifiability (§6b). The item is
    listed because a claim of ours changed on evidence -- and because the hedge only helped
    because it named, before the fact, exactly which fact it rested on.

14. **A hardcoded component taxonomy that built zero features on a new architecture.**
    `FeatureSpec("block")` in `wp1d_structure_and_fig13.py` selected depth groups from a
    literal list of ResNet names (`stem`, `layer1`..`layer4`). Run on the ViT of §6c, whose
    groups are `block0..block11`, the intersection was empty: it built **zero** aggregate
    features and returned a model with no coefficient matrix. **It crashed by luck.** The
    empty model happened to be dereferenced a few lines later; had the reporting path
    tolerated it, the run would have completed and reported the block-aggregate arms as
    uninformative -- a plausible, publishable-looking null produced by a taxonomy that did
    not match the network. Now derived from the component index itself, and the reporting
    path states explicitly when no aggregate features exist.

    This is the third instance of the same shape in this paper: a default that is correct
    for the development setup and silently wrong elsewhere (see items 11 and 12). The
    generalisation experiments were what exposed it, which is an argument for running them
    even when the expected answer is "it replicates".

15. **"Two cardinalities recovers the aggregate term."** Stated as the mitigation in Corollary 2
    and in the scope section, and *predicted in writing* in the header of the synthetic
    demonstration before it was run. **Wrong for the quadratic aggregate**, which is the term
    this paper actually measures: with two distinct cardinalities any f(n) collapses into
    span{1, n}, and n is itself V·1. Three is the minimum. Wrong for the linear aggregate too,
    in a different way — it is unidentifiable under *every* protocol, being exactly a column
    combination of the design. Caught only because the synthetic demo reported a projection
    residual rather than a rank count; the rank count called the same design full rank. The
    empirical mitigation pooled four rates, so no measured result changes (§3.6).

16. **A seed that breaks a pre-registered criterion by 1.3e-4.** Seed 4's oracle ρ is
    non-monotone at one step. We report **4 of 5 as collected** as the primary result even
    though a matched-budget refit gives 5 of 5, because the criterion was registered against the
    as-collected design and a post-hoc refit that happens to be favourable is a diagnostic, not
    a re-score. The ordering is enforced in `scripts/wp10_matched_budget_oracle.py`, which
    prints the primary count above the secondary one.

**The recurring lesson.** Items 8 and 9 are the same mistake: *an eight-point
correlation containing a two-orders-of-magnitude outlier tells you about the
outlier.* We report it twice because we made it twice.

**What survives all of this.** The claims that carry the paper were each tested
somewhere with real leverage: the proposition is exact algebra verified to
0.000e+00; the transfer result rests on corr(alpha, oracle rho) = **-0.976** over
a genuine severity range with four measured rates; the aggregate term is a 2,000x
shift in a fitted parameter; and every null is accompanied by a positive control
on the same code path.

## 6b. Generality

The proposition applies to **any linear surrogate fitted on fixed-cardinality subset masks**,
which includes datamodels and TRAK as well as COAR. Protocols that sample *several* rates
(the DistilBERT COAR replication samples ρ ∈ {0.03, 0.05, 0.08, 0.10}) partially break the
degeneracy by Corollary 2 and are correspondingly less exposed — which is itself a mitigation
worth naming, and it is free.

**Where the proposition stops, and independent evidence for the same recommendation.** The result
assumes a *binary* subset-indicator design. Methods perturbing every component continuously have
no retained count, so nothing is exactly collinear with the intercept and the proposition is
silent on them. Mammeri & Gagne (2025) fit a Kolmogorov–Arnold surrogate for high-order component
attribution under dense Gaussian perturbation, `σ ~ N_m(0, τ² I_m)`; aggregate quantities such as
`‖σ‖²` vary across their draws by construction, so what is exactly unidentifiable under
fixed-cardinality masking is identifiable in their design without deliberate stratification. They
arrive from a different direction at the conclusion this paper argues: **the perturbation design,
not the surrogate's expressive power, governs which effects are recoverable.** Their interaction
object is a *local* mixed partial at σ = 0 and ours a *global* Walsh variance share — different
quantities, and we report neither as a check on the other.

Two consequences worth stating. First, §3's conditioning caveat carries over: `‖σ‖²` concentrates
with relative spread `O(1/√m)`, so a dense design escapes the exact null space but still benefits
from varying τ deliberately rather than relying on the sampler's own fluctuation. Second, their τ
plays the role of our α and they report no sensitivity to it, so §5.3 yields a **falsifiable
prediction about their method**: a KAN fitted at τ_train and evaluated at τ_test ≠ τ_train should
degrade, and an oracle-at-τ control should recover most of that degradation, making the apparent
peak at τ_train a property of the protocol rather than of the network. One refit per τ tests it.
If it holds, the transfer result extends beyond binary masking to continuous perturbation
families; if it fails, the account is specific to binary ablation. Full discussion:
`results/SECTION_scope_of_identifiability.md` §6.

---

## 6c. Generalisation: a second dataset and a second architecture

Two setups, pre-registered in `results/WP8_WP9_prereg.md` **before either model was trained**,
with four numeric criteria and the degrees of freedom (held-out α, strictness of monotonicity,
floor handling) fixed in advance. Run in a fixed order so a disagreement is attributable:
CIFAR-100 changes the **dataset** holding architecture and component definition; ViT-Tiny changes
the **architecture and component type** holding the dataset.

| | CIFAR-10 RN18 | CIFAR-100 RN18 | CIFAR-10 ViT-Tiny |
|---|---|---|---|
| accuracy | 92.78% | 74.79% | 83.97% |
| N | 3,904 | **3,904** (asserted) | **612** |
| components | conv filters | conv filters | 36 attention heads + 576 MLP groups (width 16) |
| coarse level | 9 blocks, 2⁹ | 9 blocks, 2⁹ | **12 blocks, full 2¹² = 4,096** |
| oracle ρ monotone | yes | yes | yes |
| mean(w) ratio | 2,179× | **10,594×** | 2,020× |
| OOD raw R², single → pooled | −3.226 → **+0.439** | −3.173 → **+0.334** | −0.277 → **+0.067** |
| NAF_coarse | 0.5672 | 0.6959 | 0.3987 |
| NAF(K = 25) / floor | 113.9× | 1.6× | 65.3× |
| NAF(K = 100) / floor | 2.4× | 0.64× | negative |

The component count came out at 3,904 on CIFAR-100, so **the enumeration is not
dataset-dependent**. The ViT surface was enumerated in full at 2¹²; no fallback was needed.

**Scored against the pre-registered criteria:**

| criterion | CIFAR-100 | ViT-Tiny |
|---|---|---|
| C1 transfer (ρ monotone **and** pooled beats single on OOD raw R²) | **pass** | **pass** |
| C2 aggregate (mean(w) rises ≥ 100×) | **pass** | **pass** |
| C3 coarse non-additivity (NAF ≥ 0.30) | **pass** | **pass** |
| C4 concentration (NAF(25)/NAF(100) ≥ 10) | **fail** | **fail** |

**C4 is not measurable in either setup, and that is an instrument limit rather than a weaker
effect.** On CIFAR-100 the denominator is 0.64× the floor — beneath it — and the numerator only
1.6×. On ViT the denominator is *negative*: the K = 100 model overfits 612 components and loses
to the linear baseline out of sample, so the ratio has a noise denominator and is undefined
rather than small. Notably **ViT's NAF(K = 25) = 0.054 is the largest of the three setups**, at
65.3× its floor: concentration at small K is if anything strongest there. Only the ratio is
unresolvable.

**A pre-registered expectation, scored.** The pre-registration recorded in advance that C1/C2
should be more robust than C3/C4, because the former follow from an algebraic property of the
design while the latter are empirical properties of a particular network. **That prediction was
correct**: C1 and C2 held across a dataset change and an architecture-and-component-type change,
C3 held on both, and C4 failed exactly where the instrument runs out. It is reported as a scored
prediction, not as a caveat.

`results/WP8_WP9_prereg.md`, `scripts/wp89_score.py`

---

## 7. Limitations

- **Five seeds** on ResNet-18 / CIFAR-10 (§4.1b, §5.4), plus one further dataset and one further
  architecture (§6c). One ablation method throughout (zero-ablation).
- **Budget is confounded with seed** in the filter-level results: seeds 1-2 were collected at
  m = 50,000 and seeds 3-5 at m = 25,000, and we did not collect a 50,000-mask run for seeds 3-5.
  The 95% CI on NAF(K = 25) over five seeds crosses zero, so **we make no claim about the level
  of filter-level non-additivity.** The coarse result, which needs no estimator, is stable at
  0.5406 ± 0.0200.
- **Oracle-ρ monotonicity holds on 4 of 5 seeds as collected**; seed 4 breaks by −0.000132 at the
  one step where the two oracles are fitted on unequal budgets. At matched budget all five are
  monotone (§4.1b). We report the as-collected count as primary.
- Component granularity: conv filters and blocks. The severity-matched sweep across
  intermediate granularities is **not available in this network** (§5.7), so granularity and
  severity are not separated between those two levels. Never "neural computation is
  additive" — the claim is scoped to what was tested.
- The oracle control requires a training set at the target rate. All four rates are collected on
  every seed (α ∈ {0.01, 0.05, 0.10, 0.20}); rates outside that range are untested.
- **Corollary 2's "two or more cardinalities" is too weak for a quadratic aggregate**, which
  needs three (§3.6). Our own mitigation pooled four rates, so §5.2 is unaffected, but the
  general advice as previously stated was wrong and is corrected.
- Mean-ablation and resample-ablation are untested and may behave differently.

---

## 8. Reproducibility

Full pipeline, 85 tests including a synthetic-recovery test that validates the estimator
end-to-end and a ViT component/ablation suite that checks head ablation zeroes exactly the right
weight columns. Every figure ships its source CSV. Complete map of claims to code and findings:

| § | claim | script | finding doc |
|---|---|---|---|
| 2 | training and collection | `01_train_models.py`, `02_collect.py` | — |
| 2 | component count convention | `components.py` | `FINDING_component_count.md` |
| 3.5 | proposition, exact verification | `wp1c_identifiability.py` | `SECTION_scope_of_identifiability.md` |
| 3.6 | synthetic identifiability demo | `wpA_synthetic_identifiability.py` | — |
| 3.6 | two-failure-modes figure | `wpA_two_failure_modes_figure.py` | — |
| 4.1 | oracle-at-α control, retraction | `wp1d_structure_and_fig13.py` | `FINDING_fig13_is_transfer.md` |
| 4.1b | five-seed transfer + effect size | `wp10_seed_summary.py` | `FINDING_seed_monotonicity.md` |
| 4.1b | matched-budget diagnostic | `wp10_matched_budget_oracle.py` | `FINDING_seed_monotonicity.md` |
| 4.2 | gauge / raw-R² pitfall | `03_fit_linear.py`, `wp1a_gauge_diagnostic.py` | `FINDING_gauge_degeneracy.md` |
| 4.3 | depth statistic is blind | `04_fit_naca.py` | `FINDING_h3_measurement.md` |
| 5.1–5.3 | aggregate term, 11.6× | `wp1a_gauge_diagnostic.py`, `wp1b_aggregate_nonadditivity.py` | `FINDING_aggregate_nonadditivity.md` |
| 5.4 | K-sweep, seeds, precision floor | `04_fit_naca.py`, `wp0_precision_floor.py`, `wp7_score.py` | `PREREG_wp5_seed2.md`, `PREREG_wp7_seed2_full.md` |
| 5.4b | coarsening–aliasing theorem | `wp6_aliasing.py` | `THEOREM_aliasing.md`, `WARNING_coarse_nulls.md` |
| 5.5 | severity pre-registration | `wp2c_bootstrap_severity.py` | — |
| 5.6 | exhaustive block surface | `wp2b_exhaustive_coarse.py` | — |
| 5.7 | granularity calibration (negative) | `wp2a_calibrate_granularity.py` | — |
| 5.8 | full 19-arm ablation | `08_ablations.py` | — |
| 6c | CIFAR-100 and ViT-Tiny | `wp89_score.py` | `WP8_WP9_prereg.md` |
| 6 | precision-floor bug | `wp0_precision_floor.py` | `BUG_precision_floor_wrong_checkpoint.md` |
| — | consolidated limitations | — | `LIMITATIONS.md` |
| — | full Setup A dump | `print_results.py` | `RESULTS_setup_a.md` |
| — | figures | `07_figures.py`, `naca/plotting.py` | — |

### 8.1 Planned work that was not run

Stated so that absence is not mistaken for a null result. Neither of these was completed, and
nothing in this paper depends on either:

* **Counterfactual editing** (`05_edit.py`). Implemented and unit-tested, never run on Setup A.
  No editing claim appears anywhere in this paper.
* **Non-linear baselines** (`06_baselines.py`, MLP / GBT surrogates). The script ran only on the
  smoke dataset, where a single training rate makes all three arms numerically identical, so it
  produced no result. **We therefore make no claim that a linear surrogate beats a non-linear
  one**; the comparison this paper actually makes is between sampling designs at a fixed
  estimator, not between estimator classes.
