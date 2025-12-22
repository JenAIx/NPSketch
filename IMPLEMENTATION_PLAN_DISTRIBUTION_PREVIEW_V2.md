# Implementation Plan: Distribution Preview (Serverseitig Optimiert)

## 🎯 Überarbeitete Architektur

**Kernidee:** Alles serverseitig berechnen, nur Graph-Daten ans Frontend senden. Klassen-Labels in DB speichern für minimale Code-Änderungen.

---

## 📊 Serverseitige API Endpoints

### 1. Neuer Endpoint: `/api/ai-training/feature-distribution/{feature_name}`

**Request:**
```
GET /api/ai-training/feature-distribution/Total_Score
```

**Response:**
```json
{
  "feature_name": "Total_Score",
  "total_samples": 919,
  "statistics": {
    "min": 2.0,
    "max": 60.0,
    "mean": 51.02,
    "median": 55.0,
    "std": 10.91,
    "range": 58.0,
    "q25": 45.0,
    "q75": 60.0
  },
  "histogram": {
    "bins": 25,
    "data": [
      {"min": 2.0, "max": 4.32, "count": 5, "percentage": 0.5},
      {"min": 4.32, "max": 6.64, "count": 8, "percentage": 0.9},
      ...
    ]
  },
  "auto_classifications": {
    "2_classes": {
      "method": "quantile",
      "classes": [
        {"id": 0, "min": 2.0, "max": 45.0, "count": 147, "percentage": 16.0, "label": "Low"},
        {"id": 1, "min": 45.0, "max": 60.0, "count": 772, "percentage": 84.0, "label": "High"}
      ]
    },
    "3_classes": {...},
    "4_classes": {...},
    "5_classes": {...}
  }
}
```

**Vorteile:**
- ✅ Keine einzelnen Werte ans Frontend
- ✅ Histogramm-Daten bereits berechnet
- ✅ Auto-Klassifikationen vorberechnet
- ✅ Schnelle Antwort (nur Aggregationen)

---

## 💾 Datenbank-Erweiterung

### Erweiterte `features_data` Struktur

**Aktuell:**
```json
{
  "Total_Score": 37.0
}
```

**Mit Klassen-Labels als ZUSÄTZLICHE Features (optional):**
```json
{
  "Total_Score": 37.0,
  "class_label_4_classes": 1,
  "class_name_4_classes": "Fair (21-40)",
  "class_label_5_classes": 2,
  "class_name_5_classes": "Good (31-45)"
}
```

**Wichtig:**
- ✅ `Total_Score` bleibt unverändert (für Regression)
- ✅ Klassen-Labels werden als zusätzliche Features gespeichert
- ✅ Mehrere Klassifikationen parallel möglich (z.B. `class_label_4_classes`, `class_label_5_classes`)
- ✅ Alles andere bleibt wie es ist!

---

## 🔄 Workflow: Klassen-Labels in DB speichern

### Schritt 1: Benutzer wählt Klassifikation im Frontend
- Wählt Feature: "Total_Score"
- Wählt Modus: "Classification"
- Wählt Anzahl Klassen: 4
- Klickt "Generate Classes"

### Schritt 2: Backend generiert Klassen
```python
# In neuem Endpoint: /api/ai-training/generate-classes
POST /api/ai-training/generate-classes
{
  "feature_name": "Total_Score",
  "num_classes": 4,
  "method": "quantile"
}
```

**Backend:**
1. Liest alle Scores aus DB
2. Berechnet Klassen-Grenzen (quantile-based)
3. **Speichert Klassen-Labels in DB** (Update `features_data`)
4. Gibt Klassen-Info zurück

### Schritt 3: DB Update (ZUSÄTZLICH, nicht ersetzend!)
```python
# Für jedes Entry:
features = json.loads(entry.features_data)
score = features["Total_Score"]  # Bleibt unverändert!

# Bestimme Klasse
class_id = determine_class(score, class_boundaries)
class_name = f"Class_{class_id} ({boundaries[class_id]}-{boundaries[class_id+1]})"

# Update features_data - NUR HINZUFÜGEN, nicht ersetzen!
features[f"class_label_{num_classes}_classes"] = class_id
features[f"class_name_{num_classes}_classes"] = class_name
# Total_Score bleibt unverändert!

entry.features_data = json.dumps(features)
db.commit()
```

### Schritt 4: Training verwendet Klassen-Labels
```python
# In dataset.py oder data_loader.py
if training_mode == "classification":
    # Lese class_label statt Total_Score
    target_value = features.get("class_label")
    num_outputs = num_classes  # 4 statt 1
else:
    # Regression wie bisher
    target_value = features.get("Total_Score")
    num_outputs = 1
```

---

## 🏗️ Code-Änderungen (Minimal!)

### 1. Neuer API Endpoint (Distribution Preview)

**File:** `api/routers/ai_training.py`

```python
@router.get("/feature-distribution/{feature_name}")
async def get_feature_distribution(
    feature_name: str,
    db: Session = Depends(get_db)
):
    """
    Get distribution data for a feature (histogram, stats, auto-classifications).
    Returns only aggregated data, not individual values.
    """
    # 1. Fetch all scores from DB
    images = db.query(TrainingDataImage).filter(
        TrainingDataImage.features_data.isnot(None)
    ).all()
    
    scores = []
    for img in images:
        features = json.loads(img.features_data)
        if feature_name in features:
            scores.append(float(features[feature_name]))
    
    # 2. Calculate statistics
    stats = calculate_stats(scores)  # min, max, mean, median, std, etc.
    
    # 3. Generate histogram (25 bins)
    histogram = generate_histogram(scores, bins=25)
    
    # 4. Pre-calculate auto-classifications (2-5 classes)
    auto_classes = {}
    for num_classes in [2, 3, 4, 5]:
        classes = generate_quantile_classes(scores, num_classes)
        auto_classes[f"{num_classes}_classes"] = classes
    
    return {
        "feature_name": feature_name,
        "total_samples": len(scores),
        "statistics": stats,
        "histogram": histogram,
        "auto_classifications": auto_classes
    }
```

### 2. Neuer Endpoint: Klassen generieren & speichern

```python
@router.post("/generate-classes")
async def generate_and_save_classes(
    config: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Generate class boundaries and save class labels to database.
    
    Request:
    {
        "feature_name": "Total_Score",
        "num_classes": 4,
        "method": "quantile"  # or "equal-width"
    }
    """
    feature_name = config["feature_name"]
    num_classes = config["num_classes"]
    method = config.get("method", "quantile")
    
    # 1. Get all scores
    images = db.query(TrainingDataImage).filter(
        TrainingDataImage.features_data.isnot(None)
    ).all()
    
    scores = []
    image_map = {}  # score -> image
    for img in images:
        features = json.loads(img.features_data)
        if feature_name in features:
            score = float(features[feature_name])
            scores.append(score)
            image_map[score] = img
    
    # 2. Calculate class boundaries
    if method == "quantile":
        boundaries = np.percentile(scores, np.linspace(0, 100, num_classes + 1))
    else:  # equal-width
        boundaries = np.linspace(min(scores), max(scores), num_classes + 1)
    
    # 3. Assign classes and update DB (ZUSÄTZLICH, nicht ersetzend!)
    updated_count = 0
    class_distribution = {i: 0 for i in range(num_classes)}
    
    for img in images:
        features = json.loads(img.features_data)
        if feature_name in features:
            score = float(features[feature_name])  # Bleibt unverändert!
            
            # Determine class
            class_id = np.digitize(score, boundaries) - 1
            class_id = max(0, min(class_id, num_classes - 1))  # Clamp
            
            # Update features_data - NUR HINZUFÜGEN!
            # Total_Score bleibt unverändert!
            features[f"class_label_{num_classes}_classes"] = int(class_id)
            features[f"class_name_{num_classes}_classes"] = f"Class_{class_id}"
            # Optional: Boundaries für Referenz
            features[f"classification_boundaries_{num_classes}_classes"] = boundaries.tolist()
            
            img.features_data = json.dumps(features)
            class_distribution[class_id] += 1
            updated_count += 1
    
    db.commit()
    
    return {
        "success": True,
        "updated_count": updated_count,
        "num_classes": num_classes,
        "boundaries": boundaries.tolist(),
        "class_distribution": class_distribution
    }
```

### 3. Minimal Änderung: Dataset verwendet Klassen-Labels (optional)

**File:** `api/ai_training/dataset.py`

**Aktuell (Zeile ~91-92):**
```python
features = json.loads(img_data['features_data'])
target_value = float(features[self.target_feature])

# Apply normalization if normalizer is provided
if self.normalizer is not None:
    target_value = self.normalizer.transform(np.array([target_value]))[0]

target_tensor = torch.tensor([target_value], dtype=torch.float32)
```

**Geändert (nur wenn classification_mode aktiv):**
```python
features = json.loads(img_data['features_data'])

# Check if classification mode
if hasattr(self, 'classification_mode') and self.classification_mode:
    # Use class_label if available (z.B. "class_label_4_classes")
    class_key = f"class_label_{self.num_classes}_classes"
    target_value = features.get(class_key, 0)
    
    # KEINE Normalisierung für Klassifikation! (Integer-Klassen)
    target_tensor = torch.tensor(target_value, dtype=torch.long)  # Long für CrossEntropyLoss
else:
    # Regression mode (wie bisher - UNVERÄNDERT!)
    target_value = float(features[self.target_feature])
    
    # Normalisierung wie bisher (nur für Regression)
    if self.normalizer is not None:
        target_value = self.normalizer.transform(np.array([target_value]))[0]
    
    target_tensor = torch.tensor([target_value], dtype=torch.float32)
```

### 4. Minimal Änderung: Trainer unterstützt Klassifikation

**File:** `api/ai_training/trainer.py`

**Aktuell (Zeile ~61, 66):**
```python
self.model = DrawingClassifier(num_outputs=num_outputs, pretrained=True)
self.criterion = nn.MSELoss()
```

**Geändert:**
```python
# Check if classification mode
if config.get('training_mode') == 'classification':
    num_classes = config.get('num_classes', 4)
    self.model = DrawingClassifier(num_outputs=num_classes, pretrained=True)
    self.criterion = nn.CrossEntropyLoss()
    
    # Optional: Class weights for imbalanced data
    class_weights = config.get('class_weights')
    if class_weights:
        weights_tensor = torch.tensor(class_weights, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # KEINE Normalisierung für Klassifikation!
    # (Normalizer wird nur für Regression verwendet)
else:
    # Regression (wie bisher - UNVERÄNDERT!)
    self.model = DrawingClassifier(num_outputs=1, pretrained=True)
    self.criterion = nn.MSELoss()
    # Normalizer wie bisher (nur für Regression)
```

**Wichtig: Normalisierung nur für Regression!**
- ✅ **Regression:** Normalisierung sinnvoll (0-60 → 0-1 für besseres Training)
- ❌ **Klassifikation:** KEINE Normalisierung nötig (Integer-Klassen 0, 1, 2, 3...)
- ✅ CrossEntropyLoss erwartet Integer-Klassen, keine normalisierten Werte

---

## 📱 Frontend: Vereinfacht!

### JavaScript Module: `ai_training_preview_target_distribution.js`

**Viel einfacher, da alles serverseitig berechnet wird:**

```javascript
const DistributionPreview = {
    /**
     * Show distribution preview modal
     */
    async showPreview(featureName, event) {
        event.stopPropagation(); // Prevent feature selection
        
        // Fetch pre-calculated data from server
        const response = await fetch(
            `/api/ai-training/feature-distribution/${featureName}`
        );
        const data = await response.json();
        
        // Render modal with data
        this.renderModal(data);
    },
    
    /**
     * Render modal content (histogram, stats, classes)
     */
    renderModal(data) {
        // 1. Render statistics
        this.renderStats(data.statistics);
        
        // 2. Render histogram (data.histogram already has bins)
        this.renderHistogram(data.histogram);
        
        // 3. Render auto-classification buttons
        this.renderClassButtons(data.auto_classifications);
    },
    
    /**
     * Handle "Add N Classes" button click
     */
    async handleGenerateClasses(featureName, numClasses) {
        // Call backend to generate and save classes
        const response = await fetch('/api/ai-training/generate-classes', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                feature_name: featureName,
                num_classes: numClasses,
                method: 'quantile'
            })
        });
        
        const result = await response.json();
        
        // Show success message
        // Classes are now saved in DB!
    }
};
```

**Viel weniger Code, da:**
- ✅ Keine Histogramm-Berechnung im Frontend
- ✅ Keine Statistik-Berechnung im Frontend
- ✅ Keine Klassen-Generierung im Frontend
- ✅ Nur Rendering!

---

## 🔄 Vergleich: Alt vs. Neu

### Alt (Frontend-Berechnung):
```
Frontend:
- Fetch all 919 individual Total_Score Werte (z.B. [37.0, 45.0, 52.0, ...])
- Calculate statistics (JavaScript)
- Generate histogram (JavaScript)
- Generate classes (JavaScript)
- Render everything

Probleme:
- 919 einzelne Werte über Netzwerk
- Langsam (JavaScript)
- Code-Duplikation
```

### Neu (Serverseitig):
```
Frontend:
- Fetch aggregated data (25 histogram bins + stats)
- Render only

Backend:
- Calculate everything (Python, numpy)
- Return only what's needed
- Save classes to DB

Vorteile:
- Nur ~25 Histogramm-Bins über Netzwerk (statt 919 einzelne Werte)
- Schnell (Python/numpy)
- Klassen in DB = minimal Code-Änderung
```

**Wichtig:** Die "~25 Werte" beziehen sich auf die **Histogramm-Visualisierung** (Preview), nicht auf das Training!

**Für das Training:**
- **Regression:** Total_Score Werte (0-60, kontinuierlich) - wie bisher
- **Klassifikation:** Klassen-Labels (0, 1, 2, 3, 4) - 2-5 verschiedene Klassen

---

## 💡 Warum Klassen in DB speichern?

### Vorteile:

1. **Minimale Code-Änderungen:**
   - Dataset liest einfach `class_label` statt `Total_Score`
   - Keine Umwandlung während Training
   - Klassen sind persistent

2. **Konsistenz:**
   - Alle Einträge haben gleiche Klassen-Grenzen
   - Keine Inkonsistenzen zwischen Trainings

3. **Flexibilität:**
   - Kann mehrere Klassifikationen parallel haben
   - z.B. `classification_4_classes` und `classification_5_classes`

4. **Performance:**
   - Keine Berechnung während Training
   - Klassen bereits zugewiesen

---

## 📝 Implementierungs-Plan

### Phase 1: API Endpoints (Backend)
1. ✅ `GET /api/ai-training/feature-distribution/{feature_name}`
   - Histogramm-Daten berechnen
   - Statistiken berechnen
   - Auto-Klassifikationen vorberechnen

2. ✅ `POST /api/ai-training/generate-classes`
   - Klassen-Grenzen berechnen
   - Klassen-Labels in DB speichern
   - Distribution zurückgeben

### Phase 2: Frontend (Vereinfacht)
1. ✅ Preview-Icon hinzufügen
2. ✅ Modal mit Histogramm (nur Rendering)
3. ✅ Auto-Klassifikation Buttons
4. ✅ Klassen-Generierung aufrufen

### Phase 3: Training Integration (Minimal)
1. ✅ Dataset liest `class_label` wenn vorhanden
2. ✅ Trainer unterstützt `training_mode: "classification"`
3. ✅ Model: `num_outputs = num_classes`
4. ✅ Loss: `CrossEntropyLoss` statt `MSELoss`

---

## 🎯 Code-Änderungen Übersicht

### Backend:
- **Neu:** `api/routers/ai_training.py` - 2 neue Endpoints (~150 Zeilen)
- **Minimal:** `api/ai_training/dataset.py` - ~10 Zeilen ändern
- **Minimal:** `api/ai_training/trainer.py` - ~15 Zeilen ändern
- **Minimal:** `api/ai_training/model.py` - Keine Änderung (nutzt `num_outputs`)

### Frontend:
- **Neu:** `webapp/js/ai_training_preview_target_distribution.js` - ~200 Zeilen (nur Rendering!)
- **Minimal:** `webapp/ai_training_train.html` - Preview-Icon + Modal HTML

### Datenbank:
- **Keine Schema-Änderung!** Nur JSON in `features_data` erweitern

---

## ✅ Zusammenfassung

**Deine Idee ist perfekt!**

1. ✅ **Serverseitige Berechnung** - Alles in Python/numpy (schnell)
2. ✅ **Nur Graph-Daten** - Keine einzelnen Werte ans Frontend
3. ✅ **Klassen als ZUSÄTZLICHE Features** - Total_Score bleibt unverändert!
4. ✅ **Optional Classification** - Netzwerk einfach umschaltbar
5. ✅ **KEINE Normalisierung für Klassifikation** - Nur für Regression

**Vorteile:**
- Schneller (Python vs JavaScript)
- **Weniger Netzwerk-Traffic für Preview:** ~25 Histogramm-Bins statt 919 einzelne Werte
- Konsistente Daten
- Minimal invasive Änderungen
- Klassen persistent gespeichert
- **Regression bleibt unverändert!**
- **Mehrere Klassifikationen parallel möglich**

**Klarstellung:**
- **Preview (Histogramm):** ~25 aggregierte Bins statt 919 einzelne Werte
- **Training Regression:** Total_Score Werte (0-60, kontinuierlich) - wie bisher
- **Training Klassifikation:** Klassen-Labels (0, 1, 2, 3, 4) - 2-5 verschiedene Klassen

**Normalisierung:**
- ✅ **Regression:** Normalisierung sinnvoll (0-60 → 0-1)
- ❌ **Klassifikation:** KEINE Normalisierung nötig (Integer-Klassen)

**Nächste Schritte:**
1. Warten bis Training fertig
2. API Endpoints implementieren
3. Frontend vereinfachen (nur Rendering)
4. Training-Code minimal anpassen (nur wenn classification_mode aktiv)

---

**Status:** Planung abgeschlossen - Bereit für Implementation  
**Letzte Aktualisierung:** 2025-12-22

