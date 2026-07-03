# NPSketch — Pipeline Analysis & Improvement Ideas

*Compiled 2026-07-02. Analysis of the current preprocessing + model-training pipeline, cross-checked
against recent literature (2020–2026) and against what this team has **already tried** (see
`api/ai_training/TRAINING_PIPELINE_ANALYSIS.md` and `MASTER_PLAN_SYN_TRAIN.md`). Ideas are tiered by
impact/effort. Citations inline.*

---

## Part A — Where the pipeline stands

**Architecture (recap, see `BLUEPRINT.md`).** ResNet-18 (ImageNet, grayscale-adapted) on normalized
**284×137** line drawings. Two products: regression (`Total_Score` 0–60, linear head + MSE +
score-bin `WeightedRandomSampler`) and the deployed **components** head (60 binary sub-labels = 20
elements × Presence/Accuracy/Position, `BCEWithLogitsLoss` + per-label `pos_weight`; derived
`Total_Score` = sum). Preprocessing: auto-crop → 568×274 → binarize (175) → 2px line-normalize →
downscale to 284×137. Diversity-controlled augmentation (rot ±5°, trans ±3px, IDW warp, SSIM-filtered,
~6/img). Patient-level stratified split. Element-grounded synthetic low-score data + real LOWSCORER.
Post-hoc per-label threshold + NNLS score calibration.

**What is already solid (do not redo):**
- **Patient-level split** with hard zero-overlap assert — fixed a major val-leak; metrics are now honest.
- **Linear head** (sigmoid dropped — it saturated on the score skew).
- **AR-preserving render + true 2px lines** — the earlier 1px / aspect-distortion bugs are fixed.
- **Synthetic recipe settled:** v1 *500 uniform* synthetic won (0–29 MAE −25 %, no overall cost).
  **Rejected and not worth re-attempting:** coregistration (wash), GANs (can't guarantee labels),
  *aggressive* v2 stroke-degradation (helps component-F1 but **hurts** the derived score), isotonic
  calibration on the readout (pulls low predictions toward the dense high range).

**Known weak points / open items (from the audits):**
1. **Low-score floor & measurability.** The derived score floored near ~18 for true score ≤17;
   real low-score data is scarce (val had only 35 of 1077 ≤29; 0–9 n=3 — statistically unmeasurable).
   The **LOWSCORER import (662 + new 16-pt batch)** is the biggest *real-data* lever and is the reason
   this is now partly addressed — but a retrain + re-measurement on the enlarged low set is pending.
2. **True score-0 never seen.** TeleFred `TotalScore=0` is stored unlabeled (ambiguous "not filled in"
   vs. real 0) — the model literally never trains on a true zero. Directly relevant to the low floor.
3. **Three binarization thresholds** (127 / 175 / 250) across code paths — harmless for stored b/w
   images, matters for grayscale photo uploads; unify.
4. **Component-F1 vs derived-score tension.** Realism (E3 priors) improved per-element F1 but not the
   summed score — the readout (NNLS) mis-aligns on shifted synthetic probability signatures.
5. **Single-source evaluation.** Metrics are within-pool (patient split); cross-source/scanner
   generalization (train TELEFRED → test OXFORD/OCS_MACHINE) is not measured.
6. **No human-consensus baseline yet.** The Evaluator subsystem exists but no real multi-rater study
   has been run — so we don't know the human noise floor the model is being asked to beat.

---

## Part B — Improvement ideas (prioritized)

> **Implementation status (2026-07-03):** Tier-1 items **#1 (ASL), #2 (TTA), #3 (resolution)** are
> **implemented and config-gated** (`training.components.loss: asl` + `.asl`; `inference.tta`;
> `training.model_input: 400×193`). TTA is **live at inference now**; ASL + higher resolution take
> effect on the **next component retrain** and still need an A/B vs the current model
> (`eval_lowscore.py`) on the real held-out set. Ideally retrain on the LOWSCORER-enlarged set (the
> 16-pt batch must be validated first — it is currently `validated=False` pending review).

### Tier 1 — High impact / low risk (do first)

**1. Asymmetric Loss (ASL) for the components head.**
Replace `BCEWithLogitsLoss` + `pos_weight` with **ASL** (Ridnik et al., *Asymmetric Loss for
Multi-Label Classification*, ICCV 2021). ASL down-weights easy negatives and hard-thresholds probable
false-negatives — it strictly generalizes per-label `pos_weight` and is the standard fix for exactly
your positive/negative imbalance across 60 sparse labels. Drop-in swap, SOTA on MS-COCO/OpenImages.
*Why here:* the low-score floor is partly a per-label recall problem on rarely-present elements; ASL
targets that directly. → try in `trainer.py` components branch.

**2. Test-Time Augmentation (TTA) on the deployed model.**
Average predictions over a handful of the existing geometric augmentations at inference. The ROCF
model (Langer et al., *eLife* 2024, PMID 39607424) used 5 small rotations and got a measurable boost;
near-zero implementation cost, reduces variance, and yields **free aleatoric uncertainty** for the
clinician-facing score. → wrap `predict-single` / `run-on-test-images`.

**3. Input-resolution ablation (284×137 → higher).**
284×137 is on the low side; ROCF found **232×300 optimal**, beating both smaller and larger. You
already store 568×274, so this is a cheap A/B — try 400×195 or full 568×274 as the model input. Fine
detail (accuracy/position of small elements) may be resolution-limited. → `training_config.yaml`
`model_input`.

**4. Retrain + re-measure on the enlarged LOWSCORER set, and hold out a real low-score val fold.**
The new real low-score images make the 0–29 band *measurable* for the first time. Retrain components
with LOWSCORER (once the 16-pt batch is human-reviewed/validated) and report per-bin MAE on a real
held-out low fold — this is the honest test of whether the floor is gone. Pairs with the internal
**E4 lever** (untested): a `WeightedRandomSampler` for the *components* loader on the v1 dataset.

### Tier 2 — High impact / medium effort

**5. Reframe each element as an ordinal 0–3 head (instead of 3 independent binaries).**
Each element's true rating is an ordinal 0–3 (= presence+accuracy+position sum), and the aspects are
**hierarchical** (accuracy only meaningful if present; position only if present — confirmed in the
LOWSCORER canonical split). Model it as 20 ordinal 4-class heads with **CORN** (Shi/Raschka,
`coral-pytorch`) or **soft ordinal labels** (Díaz & Marathe, CVPR 2019), which encode "off by one is
nearly right" and reliably help the tail. The ROCF paper found a **hybrid per-element regressor/
classifier choice** beat either alone (MAE 1.11 vs 1.14/1.16). *Why here:* matches the data's real
structure, enforces the presence→accuracy→position hierarchy for free, and regularizes sparse labels.

**6. Sum-consistency / structured-constraint loss.**
Since `Total_Score = Σ sub-labels`, add a consistency term tying the summed sub-label expectation to
the (known) total, and a **hierarchical mask** (zero out accuracy/position logits when presence≈0)
— Giunchiglia & Lukasiewicz, *Multi-Label NNs with Hard Logical Constraints*, JAIR 2021. This
regularizes the sparse per-label estimates with the dense total and may fix the E3 "good F1 / bad
score" mis-alignment by construction rather than via a fragile NNLS readout.

**7. Deep Imbalanced Regression on the regression head.**
For the direct-regression product (and as an auxiliary head on the components model), add **LDS+FDS**
(Yang et al., *Delving into Deep Imbalanced Regression*, ICML 2021, arXiv:2102.09554; ~11–16 % few-shot
MAE cut) or **Balanced MSE** (Ren et al., CVPR 2022 Oral, arXiv:2203.16427). These compose with your
`WeightedRandomSampler` and specifically help scarce score ranges — a lever you have **not** yet tried
(you only have sampler-based reweighting).

**8. Run a real multi-rater Evaluator study + resolve the LOWSCORER value-2 ambiguity.**
Report **ICC + MAE vs a consensus of ≥2 raters**, not a single rater. ROCF's inter-rater SD ≈ 3.25
points is the ceiling you can't validate below (Langer 2024); OCS-Plus figure-copy human agreement was
ICC 0.83 (Webb et al., *Neuropsychology* 2021, PMID 34618514). The same rater pass can resolve the
**value-2 split** (presence+accuracy vs presence+position) that's currently a convention — turning
~155 guessed cells per low batch into ground truth.

### Tier 3 — Larger bets / research

**9. Self-supervised pretraining on all drawings (biggest untapped lever).**
Pretrain the ResNet-18 encoder with **SimCLR or a masked-autoencoder** on *all* normalized drawings
(labeled + the NULL-feature "Only Missing" TELEFRED + synthetic), then fine-tune both heads. Wolf et
al. (*Sci Rep* 2023) found **MAE self-pretraining beats ImageNet transfer on small medical datasets**,
and CNN-SimCLR beats ViT-SSL in the low-data regime. This also finally *uses* the unlabeled backlog.
*Keep ResNet-18* — ViT/ConvNeXt underperform CNNs at this data scale without heavy pretraining
(Matsoukas et al., arXiv:2108.09038).

**10. Sketch-specific backbone / cross-source generalization test.**
On limited data a **sketch-tuned CNN beat generic ResNet/VGG** for ROCF (Guerrero-Martín et al.,
*Heliyon* 2024, ROCFD528 dataset). Worth a small ablation (larger first-layer filters, stroke-aware
augmentation à la Sketch-a-Net). Independently, **hold out an entire source** (train TELEFRED → test
OXFORD/OCS_MACHINE) to measure true cross-scanner generalization — the credibility signal in the ROCF
work was an independent prospective replication (MAE 1.13 ≈ 1.11).

**11. Small deep ensemble + temperature scaling for the shipped model.**
A 3–5 model deep ensemble (Lakshminarayanan 2017) gives the most robust uncertainty and typically
1–2 % metric gains at N× train cost — feasible at this dataset size. Temperature-scale the multi-label
logits for cheap calibration. Complements TTA (#2).

**12. Continue the real-stroke collage generator (v3, already in progress).**
`feature/synth-lowscore-realism` (`gen_synth_collage.py`) pastes **real** per-element stroke crops
(exact acc/pos labels, human strokes) sampled from priors — the right response to the finding that
*template*-stroke degradation can't close the realism gap. Cap synthetic proportion: diffusion/synthetic
gains **saturate above ~10:1 synthetic:real and never beat adding real images** (Sagers et al., 2023,
arXiv:2308.12453) — real LOWSCORER first, synthetic to top up.

### Housekeeping (cheap, do alongside)
- Clarify **true score-0** with the data provider and, if real, feed true-zero drawings (removes a
  structural blind spot at the very bottom).
- **Unify the binarization threshold** across `line_normalizer` / `preprocessing` / `upload.py`.
- Label the **NULL-feature "Only Missing"** TELEFRED rows so they re-enter training.

---

## Recommended sequence

1. **Tier-1 quick wins in one retrain cycle:** ASL loss (#1) + higher input resolution (#3) +
   components `WeightedRandomSampler` (#4/E4), trained on the **LOWSCORER-enlarged** set, evaluated on
   a real low-score held-out fold. Add **TTA** (#2) at inference regardless.
2. **Then the structural change:** ordinal per-element heads + hierarchy mask + sum-consistency
   (#5, #6) — the highest-value model redesign, aligned with the data's true structure.
3. **In parallel, evaluation trust:** a real multi-rater Evaluator study (#8) and a held-out-source
   generalization test (#10) — so every model change is measured against the human ceiling and
   cross-scanner.
4. **Bigger bet when the above plateaus:** self-supervised pretraining on all drawings (#9).

**Guardrails from prior experiments:** don't re-open coreg or GANs; don't over-weight the low tail
(E2 over-skewed and regressed 20–29 + overall); don't rely on readout-only calibration to fix a
model-probability floor; always evaluate on **real** held-out data (never a synthetic test set).

### Key references
Langer 2024 ROCF (*eLife*, PMID 39607424) · Webb 2021 OCS-Plus (*Neuropsychology*, PMID 34618514) ·
Guerrero-Martín 2024 ROCFD528 (*Heliyon*, PMID 39553666) · Demeyere 2021 OCS-Plus (*Sci Rep*, PMID
33846501) · Yang 2021 LDS/FDS (arXiv:2102.09554) · Ren 2022 Balanced MSE (arXiv:2203.16427) · Cao/
Raschka CORAL (arXiv:1901.07884) / CORN (arXiv:2111.08851) · Díaz & Marathe 2019 soft ordinal labels ·
Ridnik 2021 ASL (ICCV) · Giunchiglia & Lukasiewicz 2021 constraints (JAIR) · Wolf 2023 SSL small
medical (*Sci Rep*) · Matsoukas 2021 CNN-vs-Transformer (arXiv:2108.09038) · Sagers 2023 diffusion
augmentation (arXiv:2308.12453) · Lakshminarayanan 2017 deep ensembles.
