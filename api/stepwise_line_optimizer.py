#!/usr/bin/env python3
"""
Stepwise Line Detection Optimizer for NPSketch
===============================================

This module optimizes Hough Transform parameters to match expected line counts
in test images. It uses an iterative approach to adjust parameters until the
detected line count matches the expected total (correct + extra).

Usage:
    python3 stepwise_line_optimizer.py

Author: NPSketch Team
Date: 2025-10-12
"""

import sys
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
from PIL import Image
from io import BytesIO

sys.path.insert(0, '/app')

from database import SessionLocal, TestImage, ReferenceImage
from image_processing import LineDetector, ImageRegistration


class StepwiseLineOptimizer:
    """
    Optimizes Line Detection parameters to match expected line counts.
    
    Strategy:
    --------
    For each test image with expected values (correct, missing, extra):
    1. Calculate target line count: correct + extra
    2. Detect current line count with default parameters
    3. If detected > target: Increase threshold (stricter)
    4. If detected < target: Decrease threshold (more sensitive)
    5. Iterate until detected == target or max iterations reached
    
    Parameters Optimized:
    --------------------
    - threshold: Main control parameter (impacts detection sensitivity)
    - max_line_gap: Secondary parameter (connects broken lines)
    - min_line_length: Tertiary parameter (filters short lines)
    
    Example:
    -------
    >>> optimizer = StepwiseLineOptimizer()
    >>> results = optimizer.optimize_all_test_images()
    >>> print(f"Optimal threshold: {results['optimal_parameters']['threshold']}")
    """
    
    def __init__(self, use_registration: bool = True):
        """
        Initialize the optimizer.
        
        Args:
            use_registration: Whether to apply image registration before detection
        """
        self.db = SessionLocal()
        self.use_registration = use_registration
        self.registration = ImageRegistration() if use_registration else None
        
        # Starting parameters (current optimal values)
        self.base_params = {
            'threshold': 75,
            'min_line_length': 65,
            'max_line_gap': 50
        }
        
        print("🔧 Stepwise Line Detection Optimizer initialized")
        print(f"   Registration: {'Enabled' if use_registration else 'Disabled'}")
        print(f"   Base parameters: {self.base_params}")
    
    def optimize_for_image(
        self,
        test_image: TestImage,
        reference: ReferenceImage,
        max_iterations: int = 20,
        tolerance: int = 0
    ) -> Dict:
        """
        Optimize line detection parameters for a single test image.
        
        Args:
            test_image: Test image from database
            reference: Reference image
            max_iterations: Maximum optimization iterations
            tolerance: Acceptable difference from target (default: 0 = exact match)
            
        Returns:
            Dict with optimization results:
            {
                'test_name': str,
                'target_lines': int (correct + extra),
                'detected_lines': int,
                'optimal_parameters': dict,
                'iterations': int,
                'success': bool
            }
        """
        # Calculate target line count
        target_lines = test_image.expected_correct + test_image.expected_extra
        
        print(f"\n{'='*60}")
        print(f"🎯 Optimizing: {test_image.test_name}")
        print(f"   Target lines: {target_lines} (correct: {test_image.expected_correct}, extra: {test_image.expected_extra})")
        
        # Load and prepare image
        test_img = Image.open(BytesIO(test_image.image_data))
        test_array = np.array(test_img)
        
        ref_img = Image.open(BytesIO(reference.processed_image_data))
        ref_array = np.array(ref_img)
        
        # Apply registration if enabled
        if self.use_registration:
            test_array, reg_info = self.registration.register_images(
                test_array, ref_array,
                method="ecc",
                motion_type="similarity",
                max_rotation_degrees=30.0
            )
            print(f"   Registration: {reg_info.get('method', 'N/A')}, Rotation: {reg_info.get('rotation_degrees', 0):.1f}°")
        
        # Initialize search parameters
        current_params = self.base_params.copy()
        best_params = current_params.copy()
        best_diff = float('inf')
        
        # Stepwise optimization loop
        for iteration in range(max_iterations):
            # Create detector with current parameters
            detector = LineDetector(
                threshold=current_params['threshold'],
                min_line_length=current_params['min_line_length'],
                max_line_gap=current_params['max_line_gap']
            )
            
            # Detect lines
            features = detector.extract_features(test_array)
            detected_lines = features['num_lines']
            
            diff = abs(detected_lines - target_lines)
            
            print(f"   Iteration {iteration+1:2d}: threshold={current_params['threshold']:2d}, "
                  f"detected={detected_lines:2d}, target={target_lines:2d}, diff={diff:2d}")
            
            # Check if this is the best so far
            if diff < best_diff:
                best_diff = diff
                best_params = current_params.copy()
            
            # Check if we've reached the target
            if diff <= tolerance:
                print(f"   ✅ SUCCESS! Reached target in {iteration+1} iterations")
                return {
                    'test_name': test_image.test_name,
                    'target_lines': target_lines,
                    'detected_lines': detected_lines,
                    'optimal_parameters': current_params,
                    'iterations': iteration + 1,
                    'success': True,
                    'final_diff': diff
                }
            
            # Adjust parameters based on difference
            if detected_lines > target_lines:
                # Too many lines detected - be more strict
                # Increase threshold (fewer lines will pass the vote threshold)
                current_params['threshold'] += max(1, (detected_lines - target_lines) * 2)
                
            else:  # detected_lines < target_lines
                # Too few lines detected - be more sensitive
                # Decrease threshold (more lines will be detected)
                decrease = max(1, (target_lines - detected_lines) * 2)
                current_params['threshold'] = max(20, current_params['threshold'] - decrease)
                
                # Also slightly increase max_line_gap to connect broken lines
                if current_params['max_line_gap'] < 70:
                    current_params['max_line_gap'] += 2
        
        # Max iterations reached without exact match
        print(f"   ⚠️  Max iterations reached. Best diff: {best_diff}")
        
        return {
            'test_name': test_image.test_name,
            'target_lines': target_lines,
            'detected_lines': detected_lines,
            'optimal_parameters': best_params,
            'iterations': max_iterations,
            'success': False,
            'final_diff': best_diff
        }
    
    def optimize_all_test_images(
        self,
        max_iterations: int = 20,
        tolerance: int = 0
    ) -> Dict:
        """
        Optimize line detection parameters for all test images.
        
        Finds parameters that work best across all test images by:
        1. Optimizing for each image individually
        2. Finding the parameter set that minimizes total error across all images
        3. Recommending the best global parameters
        
        Args:
            max_iterations: Maximum iterations per image
            tolerance: Acceptable difference from target
            
        Returns:
            Dict with results for all images and recommended global parameters
        """
        print("\n" + "="*60)
        print("🚀 STEPWISE LINE DETECTION OPTIMIZATION")
        print("="*60)
        
        # Get all test images and reference
        test_images = self.db.query(TestImage).all()
        reference = self.db.query(ReferenceImage).first()
        
        if not reference:
            print("❌ No reference image found!")
            return {'success': False, 'error': 'No reference image'}
        
        if len(test_images) == 0:
            print("❌ No test images found!")
            return {'success': False, 'error': 'No test images'}
        
        print(f"📊 Found {len(test_images)} test image(s)")
        print(f"🎯 Target: Optimize parameters to match expected line counts\n")
        
        # Optimize for each image
        results = []
        for test_img in test_images:
            result = self.optimize_for_image(
                test_img, reference,
                max_iterations=max_iterations,
                tolerance=tolerance
            )
            results.append(result)
        
        # Analyze results and find best global parameters
        successful = [r for r in results if r['success']]
        
        print("\n" + "="*60)
        print("📊 OPTIMIZATION RESULTS")
        print("="*60)
        
        for result in results:
            status = "✅" if result['success'] else "⚠️"
            print(f"{status} {result['test_name']:20s}: {result['detected_lines']:2d}/{result['target_lines']:2d} lines "
                  f"(diff: {result['final_diff']:+2d}, iterations: {result['iterations']:2d})")
        
        print(f"\n✨ Success rate: {len(successful)}/{len(results)} images")
        
        # Calculate recommended global parameters (average of successful optimizations)
        if successful:
            avg_threshold = int(np.mean([r['optimal_parameters']['threshold'] for r in successful]))
            avg_min_length = int(np.mean([r['optimal_parameters']['min_line_length'] for r in successful]))
            avg_max_gap = int(np.mean([r['optimal_parameters']['max_line_gap'] for r in successful]))
            
            recommended_params = {
                'threshold': avg_threshold,
                'min_line_length': avg_min_length,
                'max_line_gap': avg_max_gap
            }
            
            print(f"\n💡 RECOMMENDED GLOBAL PARAMETERS:")
            print(f"   threshold:       {recommended_params['threshold']}")
            print(f"   min_line_length: {recommended_params['min_line_length']}")
            print(f"   max_line_gap:    {recommended_params['max_line_gap']}")
            
        else:
            recommended_params = self.base_params
            print(f"\n⚠️  No successful optimizations. Using base parameters.")
        
        return {
            'success': len(successful) > 0,
            'total_images': len(test_images),
            'successful_optimizations': len(successful),
            'individual_results': results,
            'recommended_parameters': recommended_params,
            'base_parameters': self.base_params
        }
    
    def generate_report(self, results: Dict, filename: str = "line_detection_optimization.html") -> str:
        """
        Generate an HTML report of the optimization results.
        
        Args:
            results: Results from optimize_all_test_images()
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = os.path.join('/app/data/test_output', filename)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Line Detection Optimization Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .params {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .param-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 15px;
        }}
        .param-item {{
            text-align: center;
        }}
        .param-name {{
            color: #666;
            font-size: 0.9em;
        }}
        .param-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .results-table {{
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .results-table th, .results-table td {{
            padding: 15px;
            text-align: left;
        }}
        .results-table th {{
            background: #667eea;
            color: white;
        }}
        .results-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .perfect {{ background: #d4edda !important; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 Line Detection Optimization Report</h1>
        <p>Stepwise parameter optimization to match expected line counts</p>
        <p style="opacity: 0.8; font-size: 0.9em;">Generated: {timestamp}</p>
    </div>
    
    <div class="summary">
        <div class="stat-card">
            <div class="stat-value">{results['total_images']}</div>
            <div class="stat-label">Test Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{results['successful_optimizations']}</div>
            <div class="stat-label">Successful</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{results['successful_optimizations']/results['total_images']*100:.0f}%</div>
            <div class="stat-label">Success Rate</div>
        </div>
    </div>
    
    <div class="params">
        <h2>💡 Recommended Parameters</h2>
        <p style="color: #666;">These parameters should give the best overall line detection across all test images:</p>
        <div class="param-grid">
            <div class="param-item">
                <div class="param-name">Threshold</div>
                <div class="param-value">{results['recommended_parameters']['threshold']}</div>
            </div>
            <div class="param-item">
                <div class="param-name">Min Line Length</div>
                <div class="param-value">{results['recommended_parameters']['min_line_length']}</div>
            </div>
            <div class="param-item">
                <div class="param-name">Max Line Gap</div>
                <div class="param-value">{results['recommended_parameters']['max_line_gap']}</div>
            </div>
        </div>
    </div>
    
    <h2 style="margin-bottom: 20px;">📊 Individual Image Results</h2>
    <table class="results-table">
        <thead>
            <tr>
                <th>Test Image</th>
                <th>Target Lines</th>
                <th>Detected</th>
                <th>Difference</th>
                <th>Iterations</th>
                <th>Threshold</th>
                <th>Min Length</th>
                <th>Max Gap</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for result in results['individual_results']:
            row_class = 'perfect' if result['success'] else ''
            status = '<span class="success">✅ Success</span>' if result['success'] else '<span class="warning">⚠️ Partial</span>'
            diff_sign = '+' if result['final_diff'] > 0 else ''
            
            html += f"""
            <tr class="{row_class}">
                <td><strong>{result['test_name']}</strong></td>
                <td>{result['target_lines']}</td>
                <td>{result['detected_lines']}</td>
                <td>{diff_sign}{result['final_diff']}</td>
                <td>{result['iterations']}</td>
                <td>{result['optimal_parameters']['threshold']}</td>
                <td>{result['optimal_parameters']['min_line_length']}</td>
                <td>{result['optimal_parameters']['max_line_gap']}</td>
                <td>{status}</td>
            </tr>
"""
        
        html += """
        </tbody>
    </table>
    
    <div style="margin-top: 40px; padding: 20px; background: #e3f2fd; border-radius: 8px; border-left: 4px solid #2196f3;">
        <h3 style="color: #1565c0; margin-top: 0;">📘 How to Use These Parameters</h3>
        <p style="color: #333;">To apply the recommended parameters to your LineDetector:</p>
        <pre style="background: white; padding: 15px; border-radius: 4px; overflow-x: auto;"><code>from image_processing import LineDetector

detector = LineDetector(
    threshold={results['recommended_parameters']['threshold']},
    min_line_length={results['recommended_parameters']['min_line_length']},
    max_line_gap={results['recommended_parameters']['max_line_gap']}
)</code></pre>
    </div>
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"\n📄 Report generated: {report_path}")
        return report_path
    
    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'db'):
            self.db.close()


def main():
    """
    Main execution function.
    
    This will:
    1. Load all test images from the database
    2. Optimize line detection parameters for each image
    3. Find best global parameters
    4. Generate an HTML report
    """
    print("\n" + "="*60)
    print("🔧 NPSketch - Stepwise Line Detection Optimizer")
    print("="*60)
    
    # Initialize optimizer
    optimizer = StepwiseLineOptimizer(use_registration=True)
    
    # Run optimization
    results = optimizer.optimize_all_test_images(
        max_iterations=20,
        tolerance=0  # Exact match required
    )
    
    # Generate report
    if results['success']:
        optimizer.generate_report(results)
    
    print("\n" + "="*60)
    print("✅ Optimization Complete!")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    main()

