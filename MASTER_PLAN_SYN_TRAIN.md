# MASTER_PLAN_SYN_TRAIN — Synthetic training for the component model

Living document. Update the **Results log** and **TODO** after every experiment.
Branch: `feature/synthetic-gen`. Last updated: E2 done (not adopted; v1 `023227` remains best).

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

DB backups: `data/npsketch.precoreg.db` (pre-coreg), `data/npsketch.presynth.db`
(pre-any-synthetic), `data/npsketch.prev2.db` (pre-E2). Revert synthetic anytime:
`gen_synth_image.py --purge` or restore a backup.

## 4. Results log  (per-bin MAE / macro-F1 on the shared 1077 real val; n in parens)

| Exp | change | 0–9 (3) | 10–19 (5) | 20–29 (27) | 0–29 (35) | overall (1077) | verdict |
|-----|--------|---------|-----------|------------|-----------|----------------|---------|
| E0 | baseline `005214` | 14.39/0.20 | 8.11/0.25 | 2.74/0.62 | 4.51/0.604 | 1.55/0.974 | reference |
| E1 | +500 v1 synth (0–35) → `023227` | 9.65/0.06 | 3.31/0.29 | 2.73/0.65 | **3.40/0.635** | 1.54/0.973 | **adopted** (MAE −25 % on 0–29, no overall cost) |
| E2 | +630 **v2** synth, low-weighted (380@0–18, 250@18–35) → `162212` | **5.74**/0.10 | 3.35/0.29 | 4.18/0.63 | 4.19/0.627 | 1.69/0.974 | **not adopted** — won extreme 0–19 (n=3/8) but regressed 20–29 + overall |

(0–9 F1 is degenerate at n=3 — ignore; track the 0–29 derived-score MAE + overall.)

**E2 learning:** more synthetic (630 vs 500) + heavy low-weighting **over-skewed the model
low** — it nailed 0–19 but pushed 20–29 (n=27) from 2.73 → 4.18 and overall MAE 1.54 → 1.69.
So **balanced ~500 uniform (E1) is the better recipe**; don't over-weight the tail. v1 model
`023227` remains best.

**Step 0 (isotonic calibration, no retrain): rejected.** A monotone (PAVA) correction on
`023227`'s NNLS readout left overall flat (R² 0.9264→0.9266) but made low bins **worse**
(0–29 MAE 2.73→4.32) — it pulls predictions toward the dense high range. Confirms the floor is
in the model's probabilities, not the readout → calibration can't fix it.

**E3 (running): prior-driven generation.** Measured the real per-band presence structure: low
scores keep memorable elements (circle E17 ~0.68 present at 0–9; frame ~0.23; rare details
~0.05; mean 3.4/20), not a uniform subset. `gen_synth_image` now samples presence/accuracy/
position from `data/element_priors.json` (real per-band conditionals, `--build-priors`) →
structurally realistic synthetic. 500 rows, 0–35 (E1-safe amount). Compare to `023227`.

## 5. Decisions

- **Coregistration: not adopted** — A/B was a wash / marginally worse; conflicts with the
  diversity augmentation. Code kept on `feature/coreg`.
- **GAN: not pursued** — can't guarantee exact labels (the whole point), ~175 real low-score
  images is far too few (mode collapse), and the compositional recipe already yields unlimited
  exact-label data that demonstrably moved the metric.

## 6. TODO / next experiments

- [x] **E2** (done): v2 low-weighted 630 → regressed overall; **not adopted**. v1 `023227` best.
- [ ] **E3 — isolate the confound**: v2 generator at **500 uniform 0–35** (match E1's
      distribution, only the generator differs). Tells us whether v2 strokes alone help/hurt.
- [ ] **Better low-score eval FIRST**: the real blocker is measurement — 0–9 n=3, 0–29 n=35.
      Without more real low-score eval (cross-val folds, or holding out more real low rows) we
      can't trust low-bin deltas. Consider before more 10 h trains.
- [ ] **Amount sweep** (informed): 500 looks near the sweet spot; test 350 / 500 / 700 *uniform*
      (not low-weighted — E2 showed low-weighting hurts 20–29).
- [ ] **CNN-in-the-loop hard-mining**: best model → hardest cases → generate → retrain.
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
