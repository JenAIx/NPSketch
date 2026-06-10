# TeleFred Import Summary — 202606 (Union mit 20260119)

Date: 2026-06-10
Dataset: `templates/training_data_telefred_202606`
Import scripts: `api/telefred_extraction/` (getracktes Modul)
Session ID in DB: `telefred_20260610`

## Hintergrund

`training_data_telefred_202606` ist eine **kuratierte Neulieferung** desselben TeleFred-Studien­
datensatzes wie `training_data_telefred_20260119`. Gegenüber der Vorlieferung:

- ~897 fc0 + ~899 fc1 Bilder **neu**,
- 559 fc0 + 553 fc1 Bilder **entfernt** (überwiegend sauber gelabelt, Kohorte/Zeitfenster
  ~2023-12 bis 2024 — kein Qualitätsfilter; nicht in der alten `quality_flags_current.csv` enthalten),
- **856 überlappende Bilder** (416 fc0 + 440 fc1) mit **korrigiertem/nachgetragenem `TotalScore`**
  (häufig `0 → echter Score`).
- **Neu:** 202606 enthält **leere Scans** (0 rote Pixel): 294 in fc0, 86 in fc1 — diese werden beim
  Import übersprungen. Die alte Lieferung hatte keine.

## Import-Strategie: Union mit Präzedenz 202606

Ziel: DB enthält die **Vereinigung** beider Lieferungen; bei überlappenden (byte-identischen) Bildern
gelten die **aktuelleren Labels aus 202606**.

Vorgehen (reproduzierbar):

```bash
# 1. Backup
docker exec npsketch-api cp /app/data/npsketch.db /app/data/npsketch.db.bak_pre_telefred202606

# 2. Alte TELEFRED-Einträge entfernen
#    DELETE FROM training_data_images WHERE source_format='TELEFRED';  (4577 rows)

# 3. Union-Import — 202606 zuerst (gewinnt bei Dedup), dann 20260119
docker exec -e PYTHONPATH=/app npsketch-api python3 \
  /app/telefred_extraction/telefred_import.py \
  --base /app/templates/training_data_telefred_202606 \
  --base /app/templates/training_data_telefred_20260119 \
  --limit 0
```

Der SHA256-Dedup (über die Original-PNG-Bytes) ist zuverlässig: alle 1756 fc0 + 1728 fc1
überlappenden Bilder sind byte-identisch zwischen den Lieferungen.

## Import-Kriterien

- Mindestens ein rotes Pixel erforderlich (sonst übersprungen).
- `TotalScore = 0` → `features_data` leer (unlabeled).
- `source_format = TELEFRED`, `task_type` aus Ordnername (`FC0` / `FC1`).
- Dedup per SHA256 der Original-PNG-Bytes; erste Quelle (202606) gewinnt.

## Ergebnis (Database)

| Quelle / Ordner | Selected (≥1 rot) | Importiert | Duplikate | Zero-red | Unreadable |
|---|---|---|---|---|---|
| 202606 / fc0 | 2359 | 2350 | 9 | 294 | 0 |
| 202606 / fc1 | 2540 | 2531 | 9 | 86 | 1 |
| 20260119 / fc0 | 2315 | **559** (nur Alt-only) | 1756 | 0 | 0 |
| 20260119 / fc1 | 2280 | **553** (nur Alt-only) | 1727 | 0 | 1 |
| **GESAMT** | | **5993** | 3501 | 380 | 2 |

- **Total TELEFRED entries: 5993** (vorher 4577, +1416)
  - FC0: 2909
  - FC1: 3084
- Labeled (features_data gesetzt): 5385
- Unlabeled (features_data leer): 608
- Alle Einträge `session_id = telefred_20260610`.

## Notes

- Unlesbares Bild (übersprungen, in beiden Lieferungen): `fc1/2024_10_28-3991-fc1.png` (PNG defekt).
- DB-Backup vor dem Rebuild: `data/npsketch.db.bak_pre_telefred202606`.
- Die 559 fc0 + 553 fc1 in 202606 entfernten Bilder bleiben über die Union erhalten (aus 20260119,
  mit ihren ursprünglichen Labels).
