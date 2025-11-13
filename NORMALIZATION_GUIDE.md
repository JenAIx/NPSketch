# Target Normalization Guide

## Problem

Du trainierst ein Model, das Scores von 0-60 vorhersagen soll.

**Aktuell:**
- Model Output: unbeschränkt (-∞ bis +∞)
- Target Range: 0-60
- Keine Garantie, dass Vorhersagen im Bereich liegen

## Lösung: Min-Max Normalisierung

### Schritt 1: Normalisierung verstehen

```python
# Vor Normalisierung
Original Score: 30 (von 0-60)

# Nach Normalisierung
Normalized: 30 / 60 = 0.5 (von 0-1)

# Model lernt auf [0, 1]
# Vorhersage: 0.52

# De-Normalisierung
Prediction: 0.52 * 60 = 31.2
```

### Schritt 2: Code-Änderungen

#### A) Model mit Sigmoid Output

**File:** `api/ai_training/model.py`

```python
# Replace final layer with Sigmoid activation
self.backbone.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_outputs),
    nn.Sigmoid()  # ← Output jetzt [0, 1]
)

# Store score range for scaling
self.score_range = (0, 60)  # Für Total_Score
```

#### B) Dataset mit Normalisierung

**File:** `api/ai_training/dataset.py`

```python
# In __getitem__ method
target_value = float(features[self.target_feature])

# Normalize to [0, 1]
target_value_normalized = target_value / 60.0

target_tensor = torch.tensor([target_value_normalized], dtype=torch.float32)
```

#### C) Predictions de-normalisieren

**File:** `api/ai_training/trainer.py`

```python
# In evaluate_metrics method
predictions = np.array(all_predictions)

# De-normalize predictions
predictions = predictions * 60.0  # [0, 1] → [0, 60]

# Calculate metrics on original scale
mse = np.mean((predictions - targets) ** 2)
```

### Schritt 3: Vorteile

| Metrik | Ohne Normalisierung | Mit Normalisierung |
|--------|---------------------|-------------------|
| Loss (MSE) | 100-400 | 0.01-0.10 |
| Training Stabilität | ⚠️ Instabil | ✅ Stabil |
| Output Range | Unbeschränkt | [0, 1] → [0, 60] |
| Konvergenz | Langsam | Schnell |

### Schritt 4: Automatische Erkennung

**File:** `api/ai_training/normalization.py` (bereits erstellt!)

```python
from ai_training.normalization import get_normalizer_for_feature

# Automatisch richtigen Bereich wählen
normalizer = get_normalizer_for_feature('Total_Score')
# → verwendet (0, 60)

normalizer = get_normalizer_for_feature('MMSE')
# → verwendet (0, 30)

normalizer = get_normalizer_for_feature('ACE')
# → verwendet (0, 100)
```

## Quick Start

### Minimalste Änderung (3 Zeilen!)

**Option A: Simple Scaling**

```python
# In dataset.py __getitem__
target_value = float(features[self.target_feature]) / 60.0  # Normalize

# In trainer.py evaluate_metrics
predictions = predictions * 60.0  # Denormalize

# In model.py forward
return torch.sigmoid(self.backbone(x))  # Constrain to [0,1]
```

**Fertig!** 🎉

## Test

```python
# Vor Normalisierung
Target: 30, Prediction: 75 ❌ (außerhalb Bereich)
MSE: 2025

# Nach Normalisierung
Target: 0.5, Prediction: 0.52
MSE: 0.0004
→ De-normalisiert: 31.2 ✅ (im Bereich)
→ Original-Scale MSE: 1.44
```

## Welche Option wählst du?

1. **Full Implementation** (Normalization.py nutzen) - Beste Lösung
2. **Quick Fix** (3 Zeilen ändern) - Schnellste Lösung
3. **Nichts ändern** (Current Status) - Funktioniert, aber suboptimal

Sag mir Bescheid, und ich helfe dir bei der Implementierung!

