# MASTER_PLAN_SYN_TRAIN — Synthetic training for the component model

Living document. Update the **Results log** and **TODO** after every experiment.
Branch: `feature/synthetic-gen`. **STATUS: CONCLUDED & SHIPPED — `model_Components_20260617_023227`
(E1, v1 recipe) is the deployed model** (it is now the latest in `data/models/`; the not-adopted
coreg/E2/E3 models moved to `data/models/archive/`; baseline `005214` retained for reference).
Synthetic delivered its score win at E1 (0–29 MAE −25 %, no overall cost); v2 realism helps
component-F1 but not the score → diminishing returns, iteration stopped (see Results log).
Reopen via the TODO levers if revisiting. DB is in the v1-consistent state (500 v1 synthetic).

---

## 1. Goal / problem

The component CNN (60 OCS-Plus sub-labels, TELEFRED-only) is accurate overall (calibrated
derived-score R² ≈ 0.93) but **blind in the low-score range**: it floored at ~18–20 for any
true score ≤ 17, because real low-score drawings are scarce (~175 of 4326 train, only
**35 of 1077 val ≤ 29**; 0–9 has n=3). Fix: generate synthetic low-score images with
**exact** 60-component labels and add them to training. This works (see E1) — a GAN does not
(can't guarantee labels; too few real images; the recipe already yields unlimited data).

## 2. Methodology

**Element definitions (the recipe).** The 20 elements' geometry was extracted from the
official scoring manual (`templates/FigureCopyScoring_manual_OCS-Plus.pdf`, see
`docs/SCORING_CRITERIA.md`) by `api/data_consolidation/extract_elements_from_manual.py`
(each record-form cell highlights one element in red → mapped to the 568×274 reference) and
its identity validated to the model's ELEM01–20 order by label-correlation over 5403 images
(`map_elements_by_correlation.py`). Stored in `data/element_definitions.json`; hand-editable
via the paint tool in `webapp/component_map.html`. Recipe: **element region ∩ reference ink
= that element's true strokes.**

**Generator** — `api/ai_training/gen_synth_image.py`. Composes a drawing by choosing present
elements + degrading position (shift > 25 px → position=0) / accuracy (→ accuracy=0). Labels
are exact by construction (index `e*3+{0,1,2}` = presence/accuracy/position).
- **v2 degradations** (current): per-element label-preserving affine jitter; tremor modes
  (sinusoidal + smoothed random jitter); partial/incomplete strokes + pen-lift gaps for
  lines; shape distortion (anisotropic scale/rotation + dropped sector) for detail elements
  (circle/star/cross). Reduces synthetic-style overfit.
- CLI: `--preview N [--scores ...]` (no DB write), `--insert N --score-min A --score-max B
  --seed S`, `--purge`.

**Training integration.** `--insert` writes rows with `source_format='SYNTHETIC'`,
`patient_id='SYNTH_*'` → forced into **train** by `split_strategy.stratified_group_split`
(SYNTH_ prefix). The component loader filter is `.in_(('TELEFRED','SYNTHETIC'))`
(`data_loader.py`; `ai_training_base.py`). Launch: `start_telefred_component_training.py`
(seed 42, auto-calibrates at the end).

**Evaluation protocol (fair).** All component models share the seed-42 patient split, so the
baseline `005214` `val_image_ids` are the held-out **real** val for every model (synthetic is
train-only → val unchanged at 1077 real). `api/ai_training/eval_lowscore.py` reports per-bin
**MAE / macro-F1** on those images.
**Caveat:** never evaluate on a synthetic test set — it is biased toward the synthetic-trained
model. The real val is the honest measure; low bins have small n (read directionally, with F1).

## 3. Current models

| model | what | calibrated R² | RMSE | status |
|-------|------|--------------|------|--------|
| `model_Components_20260613_005214` | baseline (real only) | 0.9256 | 2.619 | reference |
| `model_Components_20260616_015757` | + coregistration | 0.9199 | 2.718 | abandoned (wash) |
| `model_Components_20260617_023227` | + 500 v1 synthetic (uniform 0–35) | 0.9264 | 2.607 | **current best** |
| `model_Components_20260617_162212` | + 630 v2 synthetic (low-weighted) | 0.9064 | 2.939 | not adopted (E2; overall regressed) |
| `model_Components_20260618_071805` | + 500 v2 synthetic (prior-driven) | 0.9102 | 2.879 | not adopted (E3; best component-F1 but score MAE worse) |

DB backups: `data/npsketch.precoreg.db` (pre-coreg), `data/npsketch.presynth.db`
(pre-any-synthetic), `data/npsketch.prev2.db` (pre-E2). Revert synthetic anytime:
`gen_synth_image.py --purge` or restore a backup.

## 4. Results log  (per-bin MAE / macro-F1 on the shared 1077 real val; n in parens)

| Exp | change | 0–9 (3) | 10–19 (5) | 20–29 (27) | 0–29 (35) | overall (1077) | verdict |
|-----|--------|---------|-----------|------------|-----------|----------------|---------|
| E0 | baseline `005214` | 14.39/0.20 | 8.11/0.25 | 2.74/0.62 | 4.51/0.604 | 1.55/0.974 | reference |
| E1 | +500 v1 synth (0–35) → `023227` | 9.65/0.06 | 3.31/0.29 | 2.73/0.65 | **3.40/0.635** | 1.54/0.973 | **adopted** (MAE −25 % on 0–29, no overall cost) |
| E2 | +630 **v2** synth, low-weighted (380@0–18, 250@18–35) → `162212` | **5.74**/0.10 | 3.35/0.29 | 4.18/0.63 | 4.19/0.627 | 1.69/0.974 | **not adopted** — won extreme 0–19 (n=3/8) but regressed 20–29 + overall |
| E3 | +500 **v2 prior-driven** synth (0–35) → `071805` | 10.35/0.07 | 5.69/0.36 | 3.43/0.66 | 4.35/**0.655** | 1.65/0.974 | **not adopted** — best component-F1 but score MAE worse; no readout (real-only NNLS / hard@thr) recovers it |

(0–9 F1 is degenerate at n=3 — ignore; track the 0–29 derived-score MAE + overall.)

**E2 learning:** more synthetic (630 vs 500) + heavy low-weighting **over-skewed the model
low** — it nailed 0–19 but pushed 20–29 (n=27) from 2.73 → 4.18 and overall MAE 1.54 → 1.69.
So **balanced ~500 uniform (E1) is the better recipe**; don't over-weight the tail. v1 model
`023227` remains best.

**Step 0 (isotonic calibration, no retrain): rejected.** A monotone (PAVA) correction on
`023227`'s NNLS readout left overall flat (R² 0.9264→0.9266) but made low bins **worse**
(0–29 MAE 2.73→4.32) — it pulls predictions toward the dense high range. Confirms the floor is
in the model's probabilities, not the readout → calibration can't fix it.

**E3 (done): prior-driven generation — not adopted for the score.** The generator now samples
presence/accuracy/position from real per-band conditionals (`data/element_priors.json`,
`--build-priors`) — structurally realistic (circle/frame survive at low scores). Result: E3 has
the **best component-F1** in the low range (0–29 F1 0.655) — the realism *did* teach better
component recognition — **but worse derived-score MAE** (0–29 4.35 vs v1 3.40; overall 1.65 vs
1.54). Cheap readout retries on E3 (NNLS real-train-only, hard@thr) did **not** recover the
score (real-only made 0–9 worse: 14.5).

**CONSOLIDATED LEARNING (E1–E3 + Step 0).** For the **derived Total_Score**, the **v1 recipe
(`023227`: 500 uniform-presence + gentle sinusoidal tremor) is the winner and stays deployed.**
Both v2-generator runs (E2 aggressive+low-weighted, E3 aggressive+prior-driven) **improved
component-F1 but hurt the score** — the heavier v2 stroke degradations shift the synthetic
probability signature, mis-aligning the NNLS readout on real val. Calibration tricks (isotonic,
real-only NNLS, hard@thr) don't bridge it. We've hit **diminishing returns on the score via
generator realism**: synthetic delivered its win at E1 (−25 % on 0–29); the residual 0–9 error
(~9.6, n=3 val) is bounded by real-data scarcity / measurement, not the generator.

## 5. Decisions

- **Coregistration: not adopted** — A/B was a wash / marginally worse; conflicts with the
  diversity augmentation. Code kept on `feature/coreg`.
- **GAN: not pursued** — can't guarantee exact labels (the whole point), ~175 real low-score
  images is far too few (mode collapse), and the compositional recipe already yields unlimited
  exact-label data that demonstrably moved the metric.

## 6. TODO / next experiments

- [x] **E2, E3, Step 0** (done): all not adopted; **v1 `023227` remains the best score model.**
- [x] **Confound resolved**: v2 generator (aggressive degradations) consistently hurts the
      derived score (E2, E3) even when it helps component-F1 → keep v1's gentle strokes.
- **RECOMMENDATION: ship `023227`.** The synthetic-low-score effort succeeded at E1
  (0–29 MAE −25 %, no overall cost). Further generator realism doesn't improve the *score*.
- Remaining levers (only if pushing further, managed expectations — each ~10 h):
  - [ ] **E4 — low-score oversampling** (`WeightedRandomSampler` for components) on the **v1**
        dataset (not v2). Orthogonal model-side lever; untested. Modest expected gain.
  - [ ] **Measurement** is the real ceiling: 0–9 n=3 is unmeasurable. Cross-val folds would be
        needed to trust further low-bin deltas — expensive (multiple trains).
  - [ ] **Component-F1 product angle**: E3 shows realism improves component recognition; if the
        product values per-element accuracy over the summed score, E3-style data is worth it.
  - [ ] **Beyond TELEFRED**: extend the component head / synthetic to other sources.

## 7. Risks / open questions

- Overfit to synthetic *style* if too many synthetic rows (watch overall R²/F1 + 30–49 bins).
- Real low-bin n is tiny → can't measure 0–9 reliably; rely on 0–29 aggregate + overall.
- Element-identity mapping was confident for distinctive elements; container/divider lines are
  low-variance (review in the paint tool if a bin looks off).

## 8. Run cheatsheet

```bash
# preview synthetic (no DB write) + CNN read with the latest model
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/gen_synth_image.py \
  --preview 12 --scores 4,8,12,16,20,24,28,32

# insert / purge synthetic rows
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/gen_synth_image.py \
  --insert 400 --score-min 0 --score-max 18 --seed 201
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/gen_synth_image.py --purge

# retrain (background, ~10 h) + compare
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/start_telefred_component_training.py
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/eval_lowscore.py \
  20260613_005214 20260617_023227 <new_stamp>

# DB backup / restore
docker exec npsketch-api cp /app/data/npsketch.db /app/data/npsketch.<tag>.db
docker exec npsketch-api cp /app/data/npsketch.<tag>.db /app/data/npsketch.db
```
