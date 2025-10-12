#!/usr/bin/env python3
"""
Automatische Optimierung der Line Detection Parameter für das Reference Image.
Iteriert bis die erwarteten Linien korrekt erkannt werden.
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Tuple
from image_processing.line_detector import LineDetector
from image_processing.utils import normalize_image

def load_expected_lines(json_path: str = '/app/templates/image_description.json') -> Dict:
    """Lade erwartete Linien aus JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['expected_lines']

def test_parameters(threshold: int, min_length: int, max_gap: int, 
                   target_image: np.ndarray, expected: Dict) -> Tuple[bool, Dict, int]:
    """
    Teste Parameter-Kombination.
    
    Returns:
        (success, counts, total_diff)
    """
    detector = LineDetector(
        threshold=threshold,
        min_line_length=min_length,
        max_line_gap=max_gap
    )
    
    features = detector.extract_features(target_image)
    counts = features['line_counts']
    
    # Berechne Differenz
    h_diff = abs(counts['horizontal'] - expected['horizontal'])
    v_diff = abs(counts['vertical'] - expected['vertical'])
    d_diff = abs(counts['diagonal'] - expected['diagonal'])
    total_diff = h_diff + v_diff + d_diff
    
    # Perfect match?
    success = (h_diff == 0 and v_diff == 0 and d_diff == 0)
    
    return success, counts, total_diff

def optimize_parameters():
    """Hauptfunktion: Iterative Parameter-Optimierung."""
    
    print('╔═══════════════════════════════════════════════════════════╗')
    print('║   NPSketch Line Detection Parameter Optimization         ║')
    print('╚═══════════════════════════════════════════════════════════╝')
    print()
    
    # 1. Lade und verarbeite Bild
    print('📷 Schritt 1: Lade Reference Image')
    img_path = '/app/templates/reference_image.png'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f'❌ Fehler: Bild nicht gefunden: {img_path}')
        return
    
    print(f'   Original: {img.shape}')
    
    # 2. Normalisiere auf 256x256
    print('   Normalisiere auf 256x256...')
    img_norm = normalize_image(img)
    print(f'   Normalisiert: {img_norm.shape}')
    
    # 3. Konvertiere zu Schwarz/Weiß
    print('   Konvertiere zu Schwarz/Weiß...')
    gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    print(f'   ✓ Preprocessing abgeschlossen\n')
    
    # 4. Lade erwartete Werte
    print('📋 Schritt 2: Lade erwartete Linien aus JSON')
    expected = load_expected_lines()
    print(f'   Erwartet: H={expected["horizontal"]}, V={expected["vertical"]}, D={expected["diagonal"]}')
    print(f'   Total: {sum(expected.values())}\n')
    
    # 5. Parameter-Grid-Search
    print('🔍 Schritt 3: Parameter-Optimierung (Grid Search)')
    print('=' * 70)
    
    # Parameter-Bereiche (basierend auf vorherigen erfolgreichen Werten)
    thresholds = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    min_lengths = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]
    max_gaps = [20, 30, 40, 50, 60, 70, 80, 100, 120]
    
    best_params = None
    best_diff = float('inf')
    best_counts = None
    
    total_tests = len(thresholds) * len(min_lengths) * len(max_gaps)
    test_count = 0
    
    print(f'Testing {total_tests} parameter combinations...\n')
    
    for threshold in thresholds:
        for min_length in min_lengths:
            for max_gap in max_gaps:
                test_count += 1
                
                success, counts, diff = test_parameters(
                    threshold, min_length, max_gap, 
                    img_norm, expected
                )
                
                # Progress
                if test_count % 100 == 0:
                    progress = (test_count / total_tests) * 100
                    print(f'  Progress: {progress:.1f}% ({test_count}/{total_tests})')
                
                # Perfect match gefunden?
                if success:
                    print(f'\n✅ ✅ ✅ PERFECT MATCH GEFUNDEN! ✅ ✅ ✅')
                    print(f'\nOptimale Parameter:')
                    print(f'  threshold:       {threshold}')
                    print(f'  min_line_length: {min_length}')
                    print(f'  max_line_gap:    {max_gap}')
                    print(f'\nResultat:')
                    print(f'  H: {counts["horizontal"]}/{expected["horizontal"]} ✓')
                    print(f'  V: {counts["vertical"]}/{expected["vertical"]} ✓')
                    print(f'  D: {counts["diagonal"]}/{expected["diagonal"]} ✓')
                    print(f'  Total: {counts["total"]}/{sum(expected.values())} ✓')
                    
                    # Speichere Parameter in Datei
                    save_optimal_parameters(threshold, min_length, max_gap)
                    
                    return threshold, min_length, max_gap
                
                # Beste bisherige Kombination?
                if diff < best_diff:
                    best_diff = diff
                    best_params = (threshold, min_length, max_gap)
                    best_counts = counts
    
    # Kein Perfect Match gefunden
    print(f'\n⚠️  Kein Perfect Match gefunden')
    print(f'\nBeste Annäherung:')
    print(f'  threshold:       {best_params[0]}')
    print(f'  min_line_length: {best_params[1]}')
    print(f'  max_line_gap:    {best_params[2]}')
    print(f'\nResultat:')
    print(f'  H: {best_counts["horizontal"]}/{expected["horizontal"]} (diff: {abs(best_counts["horizontal"]-expected["horizontal"])})')
    print(f'  V: {best_counts["vertical"]}/{expected["vertical"]} (diff: {abs(best_counts["vertical"]-expected["vertical"])})')
    print(f'  D: {best_counts["diagonal"]}/{expected["diagonal"]} (diff: {abs(best_counts["diagonal"]-expected["diagonal"])})')
    print(f'  Total Difference: {best_diff}')
    
    save_optimal_parameters(best_params[0], best_params[1], best_params[2])
    
    return best_params

def save_optimal_parameters(threshold: int, min_length: int, max_gap: int):
    """Speichere optimale Parameter in Datei."""
    params = {
        'threshold': threshold,
        'min_line_length': min_length,
        'max_line_gap': max_gap,
        'optimized_at': str(np.datetime64('now'))
    }
    
    output_path = '/app/data/optimal_parameters.json'
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f'\n💾 Parameter gespeichert: {output_path}')

if __name__ == '__main__':
    optimize_parameters()

