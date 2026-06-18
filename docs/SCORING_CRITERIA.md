# OCS-Plus Figure Copy — Scoring Criteria

Source: `templates/FigureCopyScoring_manual_OCS-Plus.pdf` (Translational Neuropsychology
Research Group, www.ocs-test.org/ocsplus). This is the rubric the human raters used to
produce the `features_data.components` labels in `training_data_images`, and therefore the
ground truth the component model learns. It is also the **build recipe** for synthetic
images (see `api/ai_training/gen_synth_image.py`).

## Overview

The template figure has **20 components**, each scored on three independent binary aspects:

| Aspect | Mark (0/1) | Criterion |
|--------|-----------|-----------|
| **Presence** | 1 if a recognisable component is present **anywhere** in the drawing | location-independent |
| **Accuracy** | 1 if drawn with reasonable accuracy for typical drawing ability | straight lines, clean joins at the template angles (≈90° for the container); allowances for stylus slip and obvious self-corrections (doubling a line to straighten it) |
| **Position** | 1 if positioned correctly **relative to its nearest neighbours** | dividers must also partition in proportions similar to the template; details must match position **and orientation** |

Max score = 20 components × 3 = **60** (the model's 60 sub-labels). Total_Score = sum.

Scoring should quantify the ability to copy/retain/reproduce the elements — **not** drawing
proficiency. Score each aspect independently, but **avoid double-penalising**: if one
element's inaccuracy/misposition causes a neighbour to not line up, do not score the
neighbour down for it.

## The three component groups

1. **Container** — lines defining the edges of the enclosing rectangle.
   - *Special rule:* if **no container component** is present anywhere, assume the participant
     used the drawing-area bounds as the container and **award the full 18 container marks**
     (i.e. the container group is 6 components × 3 = 18) — favour, don't penalise.
   - Accuracy: lines reasonably straight, joining cleanly at 90°. Position: same relative
     position as template, within the blue bounds.
2. **Divider** — lines that partition the container.
   - Accuracy: reasonably straight, clean joins at the template angles. Position: correct
     relative to neighbours **and** partition the container/sub-partitions in similar
     proportions, within the blue bounds.
3. **Details** — distinct features inside partitions (the **circle**, **star**, **cross**).
   - Accuracy: shape similar to template, clearly recognisable — **even if a diagonal is
     flipped**. Position: same position **and orientation** — **no position point if a
     diagonal detail is flipped**.

## Impairment cut-offs

- **COPY**: impaired if Total_Score **< 53**
- **RECALL**: impaired if Total_Score **< 32**

## Mapping to this codebase

- 60-vector order (see `api/ai_training/dataset.py:components_to_vector`): per element
  `e` (0..19) → `[presence, accuracy, position]`, i.e. index `e*3 + {0,1,2}`.
- The canonical per-element geometry (which strokes are element *e*) was extracted from the
  manual's record-form page into `/app/data/element_definitions.json`
  (`extract_elements_from_manual.py` → `map_elements_by_correlation.py`, validated to the
  model's ELEM01–20 label order). `element region ∩ reference ink = that element's strokes`.
