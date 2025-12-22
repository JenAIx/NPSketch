# Classification Training - Implementierungsplan

**Datum:** 2025-12-22  
**Ziel:** Ermögliche Classification Training für Custom_Class Features

---

## 🎯 Anforderungen

1. **Feature-Auswahl**: Custom_Class_5 statt Total_Score
2. **Model**: num_outputs = 5 (statt 1), CrossEntropyLoss (statt MSELoss)
3. **Dataset**: Liest Custom_Class["5"]["label"] (Integer 0-4)
4. **Normalization**: DEAKTIVIERT für Classification
5. **UI**: Zeigt angepasste Model-Info, versteckt Normalization-Option

---

## 📋 Aktueller Zustand (Regression)

### Backend Flow:
```python
1. start_training(target_feature="Total_Score")
2. run_training_job():
   - normalizer = get_normalizer_for_feature("Total_Score")  # min_max [0, 60]
   - trainer = CNNTrainer(num_outputs=1)
   - criterion = nn.MSELoss()
3. DrawingDataset.__getitem__():
   - target = features["Total_Score"]  # 37.0
   - target = normalizer.transform(target)  # 0.617
   - return torch.tensor([target], dtype=float32)  # Shape: [1]
4. Model output: [predicted_score]  # Shape: [1]
5. Loss: MSELoss(output, target)
```

### Frontend:
```javascript
- Zeigt: "Output: 1 neuron (regression)"
- Zeigt: "Target Normalization: Enabled [0, 60] → [0, 1]"
```

---

## 🎯 Gewünschter Zustand (Classification)

### Backend Flow:
```python
1. start_training(target_feature="Custom_Class_5")
2. run_training_job():
   - Erkenne: target_feature.startswith("Custom_Class_")
   - num_classes = 5 (aus "Custom_Class_5")
   - normalizer = None  # KEINE Normalization!
   - trainer = CNNTrainer(num_outputs=5, training_mode="classification")
   - criterion = nn.CrossEntropyLoss()
3. DrawingDataset.__getitem__():
   - target = features["Custom_Class"]["5"]["label"]  # 1 (Integer)
   - return torch.tensor(target, dtype=torch.long)  # Shape: [] (scalar)
4. Model output: [logit_class0, logit_class1, ..., logit_class4]  # Shape: [5]
5. Loss: CrossEntropyLoss(output, target)
```

### Frontend:
```javascript
- Zeigt: "Output: 5 neurons (classification)"
- Zeigt: "Classes: Poor, Fair, Moderate, Good, Excellent"
- Versteckt: "Target Normalization" Section
```

---

## 🔧 Zu ändernde Komponenten

### 1. Backend: ai_training_base.py
**Änderungen in `run_training_job()`:**
```python
# Detect if target is Custom_Class
is_classification = config['target_feature'].startswith('Custom_Class_')

if is_classification:
    # Extract num_classes from "Custom_Class_5" -> 5
    num_classes = int(config['target_feature'].replace('Custom_Class_', ''))
    num_outputs = num_classes
    normalizer = None  # NO normalization for classification!
    training_mode = "classification"
else:
    # Regression
    num_outputs = 1
    normalizer = get_normalizer_for_feature(...) if use_normalization else None
    training_mode = "regression"

# Create trainer with mode
trainer = CNNTrainer(
    num_outputs=num_outputs,
    training_mode=training_mode,
    learning_rate=config['learning_rate'],
    normalizer=normalizer
)
```

### 2. Backend: ai_training/trainer.py
**Änderungen in `__init__()`:**
```python
def __init__(
    self,
    num_outputs: int = 1,
    learning_rate: float = 0.001,
    device: str = None,
    normalizer=None,
    training_mode: str = "regression"  # NEW!
):
    self.training_mode = training_mode
    
    # Initialize model
    self.model = DrawingClassifier(num_outputs=num_outputs, pretrained=True)
    self.model.to(self.device)
    
    # Optimizer and loss - CONDITIONAL!
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    if training_mode == "classification":
        self.criterion = nn.CrossEntropyLoss()
        print(f"   Loss function: CrossEntropyLoss (classification)")
    else:
        self.criterion = nn.MSELoss()
        print(f"   Loss function: MSELoss (regression)")
```

### 3. Backend: ai_training/dataset.py
**Änderungen in `DrawingDataset`:**
```python
def __init__(
    self,
    images_data: List[Dict],
    target_feature: str,
    transform=None,
    normalizer: Optional[TargetNormalizer] = None,
    is_classification: bool = False,  # NEW!
    num_classes: int = None  # NEW!
):
    self.is_classification = is_classification
    self.num_classes = num_classes

def __getitem__(self, idx):
    # ... (image loading bleibt gleich) ...
    
    features = json.loads(img_data['features_data'])
    
    if self.is_classification:
        # Read from Custom_Class
        custom_class = features.get("Custom_Class", {})
        class_data = custom_class.get(str(self.num_classes))
        
        if class_data:
            target_value = int(class_data["label"])
        else:
            target_value = 0  # Fallback
        
        # NO normalization for classification!
        target_tensor = torch.tensor(target_value, dtype=torch.long)
    else:
        # Regression (wie bisher)
        target_value = float(features[self.target_feature])
        
        if self.normalizer is not None:
            target_value = self.normalizer.transform(np.array([target_value]))[0]
        
        target_tensor = torch.tensor([target_value], dtype=torch.float32)
    
    return img_tensor, target_tensor
```

### 4. Frontend: ai_training_train.html (JavaScript)
**Änderungen in `selectFeature()`:**
```javascript
async function selectFeature(feature) {
    selectedFeature = feature;
    
    // Detect if Custom_Class
    const isClassification = feature.startsWith('Custom_Class_');
    
    if (isClassification) {
        // Extract num_classes
        const numClasses = parseInt(feature.replace('Custom_Class_', ''));
        
        // Update Model Info
        updateModelInfoForClassification(numClasses, feature);
        
        // HIDE normalization section
        document.getElementById('normalizationSection').style.display = 'none';
        
    } else {
        // Regression mode
        updateModelInfoForRegression();
        
        // SHOW normalization section
        document.getElementById('normalizationSection').style.display = 'block';
    }
    
    // Show config section
    document.getElementById('trainingConfig').style.display = 'block';
}

function updateModelInfoForClassification(numClasses, featureName) {
    // Fetch class names from Custom_Class
    // Show: "Output: N neurons (one per class)"
    // Show: "Classes: Poor, Fair, Moderate, Good, Excellent"
    // Show: "Loss Function: CrossEntropyLoss"
}

function updateModelInfoForRegression() {
    // Show: "Output: 1 neuron (regression)"
    // Show: "Loss Function: MSELoss"
}
```

**HTML Änderungen:**
```html
<!-- Wrap normalization section with ID -->
<div id="normalizationSection" style="...">
    <input type="checkbox" id="enableNormalization" checked>
    <label>Enable Target Normalization</label>
    ...
</div>
```

---

## 📝 Implementierungsschritte

### Phase 1: Backend (Core Logic)
1. ✅ `ai_training_base.py`: Erkenne Custom_Class, setze num_outputs
2. ✅ `trainer.py`: training_mode Parameter, conditional loss
3. ✅ `dataset.py`: Lese Custom_Class["N"]["label"] für Classification

### Phase 2: Backend (Integration)
4. ✅ Pass `is_classification` und `num_classes` an Dataset
5. ✅ Teste mit Custom_Class_5 Feature
6. ✅ Validiere: num_outputs=5, CrossEntropyLoss, keine Normalization

### Phase 3: Frontend (UI)
7. ✅ Erkenne Custom_Class in selectFeature()
8. ✅ Update Model Info für Classification
9. ✅ Hide Normalization Section
10. ✅ Show Class Names (from Custom_Class data)

### Phase 4: Testing
11. ✅ Test Regression (Total_Score) - sollte weiter funktionieren
12. ✅ Test Classification (Custom_Class_5)
13. ✅ Validiere Metriken (Accuracy statt MAE/RMSE)

---

## ⚠️ Wichtige Punkte

### Tensor Shapes:
```python
# Regression:
output: [batch_size, 1]  # z.B. [8, 1]
target: [batch_size, 1]  # z.B. [8, 1]
loss = MSELoss(output, target)

# Classification:
output: [batch_size, num_classes]  # z.B. [8, 5]
target: [batch_size]  # z.B. [8] (Integer-Indices)
loss = CrossEntropyLoss(output, target)
```

### NO Normalization für Classification:
- Classification nutzt Integer-Labels (0, 1, 2, 3, 4)
- CrossEntropyLoss erwartet raw logits + class indices
- Normalization würde Fehler verursachen

### Feature Detection:
```python
# Regression
"Total_Score" → features["Total_Score"] → float

# Classification  
"Custom_Class_5" → features["Custom_Class"]["5"]["label"] → int
```

---

## 🧪 Test-Szenarien

### Szenario 1: Regression (bestehend)
```
Feature: Total_Score
→ num_outputs=1
→ MSELoss
→ Normalization: enabled
→ Target: [0.617] (normalized)
✅ Sollte weiter funktionieren
```

### Szenario 2: Classification (neu)
```
Feature: Custom_Class_5
→ num_outputs=5
→ CrossEntropyLoss
→ Normalization: disabled
→ Target: 1 (class index)
✅ Sollte trainieren können
```

---

## 📊 Erwartete Änderungen

**Backend:**
- `ai_training_base.py`: ~30 Zeilen
- `trainer.py`: ~15 Zeilen
- `dataset.py`: ~25 Zeilen

**Frontend:**
- `ai_training_train.html`: ~80 Zeilen

**Gesamt:** ~150 Zeilen neue/geänderte Code

---

## ✅ Bereit für Implementation

Alle Komponenten identifiziert, Plan erstellt.

**Nächster Schritt:** Implementation starten

---

**Erstellt:** 2025-12-22

