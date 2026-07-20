# FIXES.md — v13 → v14 (`domain_adapt_v14_8cell.ipynb`)

Consolidated the whole domain-adaptive patch-mining pipeline into **8 code cells**,
each opening with a numbered `SECTION` map so a specific fix is easy to find. This
document lists the problems that were producing catastrophic / uninterpretable
output and what changed. Every fix is tagged inline in the notebook as `[v14 FIX ...]`.

All numbers referenced below come from the v13 ablation CSVs
(`patch_ablations_v11*.csv`) and were produced under `SMOKE_TEST=True` /
`FAST_MODE=True`, i.e. a sanity run — not reportable. They still diagnose the
failure modes correctly. Clinical framing remains subject to **Dr. Nimi sign-off**.

---

## 0. The bug this branch is named for

**`RandomState.integers` → `AttributeError` (controls never ran).**

`_control_scores` created a legacy `np.random.RandomState(seed + 991)`, but
`_rand_boxes` called `rng.integers(...)`, which only exists on the new
`np.random.Generator` API (`default_rng`). `RandomState` exposes `randint`, not
`integers`, so the call raised `AttributeError` **after** all 16 grid arms had
been saved — which is exactly why the delivered CSVs contained every ablation row
but **no `CTRL:random` / `CTRL:gcam_peak` rows, no summary tables, and no plots**.

- **Fix:** `rng = np.random.default_rng(seed + 991)` in `_control_scores`
  (Cell 8, SECTION 8.3). The subset-selection RNG that only calls `.choice`
  stays on `RandomState` (valid there).
- **Verified:** the internal smoke test asserts `default_rng` produces boxes *and*
  that a legacy `RandomState` is still correctly rejected by `.integers`.

Without controls, "does the pipeline beat random boxes?" was unanswerable — the
single most important question, given the findings below.

---

## 1. Catastrophic-output root causes and fixes

| ID | Problem (v13) | Evidence | Fix (v14) | Where |
|----|---------------|----------|-----------|-------|
| **A** | **Class probability was a 5-way softmax across the target classes.** These are multi-label, co-occurring findings; a patch showing both effusion and atelectasis suppressed *both* probabilities. Per-class AUROC therefore depended on the other four classes and sat near chance (Atelectasis 0.495, Consolidation 0.568). | 5-way softmax couples classes; per-class AUROC at/near 0.5 | **Binary CheXzero prompt pairs**: `P(class present) = softmax(ls·[sim(pos), sim(neg)])`, computed **independently per class**. New `ZS_PROMPTS_NEG`, `build_binary_prompt_matrices`, `binary_class_probs_gpu`, `class_probs_gpu` dispatcher. Toggle `USE_BINARY_PROMPTS`. | Cell 1 §1.6, Cell 4 §4.2/§4.5 |
| **B** | **Patch-max only** discarded whole-image context (Cardiomegaly especially needs it). | patch-only scoring | **Global ensemble**: final class score = `α·global_image_prob + (1−α)·patch_prob`. `GLOBAL_ENSEMBLE_ALPHA=0.5`; α=1.0 is the pure-global baseline to report, α=0.0 is old behaviour. New `global_image_class_probs`; ensembled in `classify_patches_gpu`. | Cell 1 §1.4, Cell 4 §4.5, Cell 5 §5.5 |
| **C** | **`FAST_MODE` dropped the 64px scale**, hurting small/subtle findings (Atelectasis was below chance). | `PATCH_SCALES=[128,256]` under FAST_MODE | `FAST_MODE=False` by default; `_SCALE_MAP` (which keeps 64px for Atelectasis/Edema) is honoured by the discovery loop regardless. Smoke test asserts 64px is retained. | Cell 1 §1.4/§1.5, Cell 6 §6.1 |
| **D** | **Image-level AUROC can't tell a well-localized patch from a lucky one** — it never measured patch quality directly. | metric blind to mining (ablation: destroying 96% of causal mining moved AUROC by −0.001, n.s.) | **A\* localization metrics** vs NIH `BBox_List_2017.csv`: **pointing-game** accuracy and **IoU@0.1 / IoU@0.25** for the top causal patch, per class (Atelectasis, Cardiomegaly, Effusion→Pleural Effusion). Plus **AUPRC vs prevalence** (AUPRC < prevalence ⇒ worse than guessing). | Cell 6 §6.4/§6.5 |
| **E** | **The causal gate was load-bearing on its *fallback*, not on causal mining.** `G:no_fallback` was the only statistically significant arm (ΔmacroAUROC −0.036, CI excludes 0) and the Edema headline (0.854→0.651, abstain 38%) was fallback-driven. | paired bootstrap, per-class CSV | `ALLOW_FALLBACK` exposed prominently with the trade-off documented; **abstain %** reported everywhere; ablation keeps the `G:no_fallback` arm and adds `H:5way_softmax`, `I:ensemble=0.0/1.0` to isolate the v14 scoring fixes. | Cell 1 §1.5, Cell 8 §8.6 |

---

## 2. New must-have outputs

- **GradCAM maps for causal patches, all 5 classes** — `plot_causal_gradcam_grid`
  renders the `(a) X-ray / (b) heatmap / (c) overlay + causal box` layout (one row
  per class), computed from each class's causal-patch query vector. (Cell 6 §6.6)
- **Bounding boxes shown separately per bucket** — `plot_bboxes_by_bucket` draws
  **causal** (green / blue=fallback), **spurious-in-anatomy** (red), and
  **spurious-outside-anatomy** (orange) on three independent panels. (Cell 6 §6.7)
- **Internal smoke test** — `run_internal_smoke_test()` validates every fix with
  **no GPU, no CheXzero checkpoint, no dataset** (synthetic tensors only). Runs in
  ~2 s and gates the notebook. (Cell 7 §7.2)

---

## 3. Best-hyperparameter config (baked defaults, Cell 1)

```
USE_BINARY_PROMPTS   = True      # FIX A
GLOBAL_ENSEMBLE_ALPHA= 0.50      # FIX B  (report α=1.0 as global-only baseline)
FAST_MODE            = False     # FIX C
ALLOW_FALLBACK       = True      # keep to match reported AUROC; False = honest gate
CAUSAL_CONTAIN_MODE  = "peak_or_cover"
PERSISTENCE_TOP_K    = 2 ; PERSISTENCE_N_LEVELS = 32
CAUSAL_MASK_COVER_FRAC = 0.35 ; CAUSAL_MASK_DILATE_PX = 8
SEMANTIC_THRESHOLD_PER_CLASS = {Atel .16, Card .18, Cons .14, Edema .15, Eff .18}
```
The v13 sweep showed the containment/persistence/dilation/cover knobs are largely
inert (deltas at the 4th decimal, CIs spanning 0), so FULL's values are retained;
the decisive changes are the **scoring model** (A/B) and the **evaluation** (D).

---

## 4. Recommended next run

1. Run Cell 7 with real data + checkpoint (`SMOKE_TEST=False`, `FAST_MODE=False`).
   Read the **A\* localization table** first — that is the direct test of patch
   quality the old AUROC could not provide.
2. Run Cell 8 ablations with `ABL_SEEDS=[42,1,7]`, `ABL_SUBSET_N=400`. Check that
   **FULL now beats `CTRL:random`** (the question the RandomState bug hid), and
   read arms **H** (5-way softmax) and **I** (ensemble off) to confirm the v14
   scoring fixes move the metric.
3. All numbers stay diagnostic until Dr. Nimi signs off on the clinical framing.
