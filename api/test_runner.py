#!/usr/bin/env python3
"""
Automated Test Runner for NPSketch
==================================

This script runs automated tests with different parameter combinations
to find optimal settings for line detection and matching.

It generates HTML reports in /app/data/test_output/
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import from our modules
sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage, Base
from services.evaluation_service import EvaluationService
from services.reference_service import ReferenceService
from image_processing import LineComparator


class TestRunner:
    """
    Automated test runner that evaluates different parameter combinations.
    """
    
    def __init__(self, output_dir: str = "/app/data/test_output"):
        """Initialize test runner."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Database setup
        DATABASE_URL = "sqlite:////app/data/npsketch.db"
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        self.db = SessionLocal()
        
        print(f"📊 Test Runner initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def get_test_images(self) -> List[TestImage]:
        """Load all test images from database."""
        test_images = self.db.query(TestImage).order_by(TestImage.id).all()
        print(f"✅ Loaded {len(test_images)} test images")
        return test_images
    
    def get_reference(self) -> ReferenceImage:
        """Load reference image."""
        ref_service = ReferenceService(self.db)
        references = ref_service.list_all_references()
        if not references:
            raise ValueError("No reference image found!")
        return references[0]
    
    def run_test_suite(
        self,
        use_registration: bool = True,
        registration_motion: str = "similarity",
        max_rotation_degrees: float = 30.0,
        position_tolerance: float = 20.0,
        angle_tolerance: float = 15.0,
        length_tolerance: float = 0.3
    ) -> Dict:
        """
        Run tests with specific parameters.
        
        Returns:
            Dictionary with test results and statistics
        """
        test_images = self.get_test_images()
        reference = self.get_reference()
        
        if not test_images:
            return {
                'total_tests': 0,
                'results': [],
                'statistics': {},
                'parameters': {}
            }
        
        # Create evaluation service with parameters
        eval_service = EvaluationService(
            self.db,
            use_registration=use_registration,
            registration_motion=registration_motion,
            max_rotation_degrees=max_rotation_degrees
        )
        
        # Set custom comparator
        eval_service.comparator = LineComparator(
            position_tolerance=position_tolerance,
            angle_tolerance=angle_tolerance,
            length_tolerance=length_tolerance
        )
        
        # Run evaluations
        results = []
        for test_img in test_images:
            try:
                # Load image
                nparr = np.frombuffer(test_img.image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Evaluate
                evaluation = eval_service.evaluate_test_image(
                    image,
                    reference.id,
                    f"test_{test_img.id}"
                )
                
                # Calculate both scores
                from image_processing import LineDetector
                line_detector = LineDetector()
                ref_features = line_detector.features_from_json(reference.feature_data)
                total_ref_lines = len(ref_features['lines'])
                
                # Reference Match: How good is the image vs reference
                effective_correct = max(0, evaluation.correct_lines - evaluation.extra_lines)
                detection_score = effective_correct / total_ref_lines if total_ref_lines > 0 else 0.0
                detection_score = max(0.0, min(1.0, detection_score))
                
                # Test Rating: How well did we predict the results (Expected vs Actual)
                correct_diff = evaluation.correct_lines - test_img.expected_correct
                missing_diff = evaluation.missing_lines - test_img.expected_missing
                extra_diff = evaluation.extra_lines - test_img.expected_extra
                
                max_error_per_metric = total_ref_lines
                total_max_error = max_error_per_metric * 3
                total_actual_error = abs(correct_diff) + abs(missing_diff) + abs(extra_diff)
                
                prediction_accuracy = 1.0 - (total_actual_error / total_max_error) if total_max_error > 0 else 1.0
                prediction_accuracy = max(0.0, min(1.0, prediction_accuracy))
                
                results.append({
                    'test_id': test_img.id,
                    'test_name': test_img.test_name,
                    'expected': {
                        'correct': test_img.expected_correct,
                        'missing': test_img.expected_missing,
                        'extra': test_img.expected_extra
                    },
                    'actual': {
                        'correct': evaluation.correct_lines,
                        'missing': evaluation.missing_lines,
                        'extra': evaluation.extra_lines
                    },
                    'detection_score': detection_score,
                    'prediction_accuracy': prediction_accuracy,
                    'accuracy': prediction_accuracy,  # Use prediction as main metric
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'test_id': test_img.id,
                    'test_name': test_img.test_name,
                    'success': False,
                    'error': str(e),
                    'accuracy': 0.0
                })
        
        # Calculate statistics
        successful = [r for r in results if r.get('success', False)]
        
        if successful:
            avg_detection_score = sum(r['detection_score'] for r in successful) / len(successful)
            avg_prediction_accuracy = sum(r['prediction_accuracy'] for r in successful) / len(successful)
            perfect_detections = sum(1 for r in successful if r['detection_score'] == 1.0)
            perfect_predictions = sum(1 for r in successful if r['prediction_accuracy'] == 1.0)
            
            statistics = {
                'total_tests': len(test_images),
                'successful': len(successful),
                'failed': len(test_images) - len(successful),
                'average_detection_score': avg_detection_score,
                'average_prediction_accuracy': avg_prediction_accuracy,
                'average_accuracy': avg_prediction_accuracy,  # Use prediction as main metric
                'perfect_detections': perfect_detections,
                'perfect_predictions': perfect_predictions,
                'perfect_matches': perfect_predictions,  # Use prediction as main metric
                'pass_rate': avg_prediction_accuracy
            }
        else:
            statistics = {
                'total_tests': len(test_images),
                'successful': 0,
                'failed': len(test_images),
                'average_accuracy': 0.0,
                'perfect_matches': 0,
                'pass_rate': 0.0
            }
        
        return {
            'total_tests': len(test_images),
            'results': results,
            'statistics': statistics,
            'parameters': {
                'use_registration': use_registration,
                'registration_motion': registration_motion,
                'max_rotation_degrees': max_rotation_degrees,
                'position_tolerance': position_tolerance,
                'angle_tolerance': angle_tolerance,
                'length_tolerance': length_tolerance
            }
        }
    
    def parameter_grid_search(self) -> List[Dict]:
        """
        Run grid search over parameter space to find optimal settings.
        """
        print("\n🔍 Starting Parameter Grid Search...")
        print("=" * 60)
        
        # Define parameter grid
        # OPTIMIZED FOR TEST RATING >90% (Expected vs Actual)
        # Higher tolerances = better prediction accuracy
        param_grid = {
            'registration_motion': ['similarity', 'euclidean'],
            'max_rotation_degrees': [30.0, 45.0, 60.0],
            'position_tolerance': [80.0, 100.0, 120.0, 150.0, 180.0],
            'angle_tolerance': [40.0, 50.0, 60.0, 75.0],
            'length_tolerance': [0.6, 0.7, 0.8, 0.9]
        }
        
        all_results = []
        total_combinations = (
            len(param_grid['registration_motion']) *
            len(param_grid['max_rotation_degrees']) *
            len(param_grid['position_tolerance']) *
            len(param_grid['angle_tolerance']) *
            len(param_grid['length_tolerance'])
        )
        
        print(f"📊 Testing {total_combinations} parameter combinations...")
        print()
        
        iteration = 0
        for reg_motion in param_grid['registration_motion']:
            for max_rot in param_grid['max_rotation_degrees']:
                for pos_tol in param_grid['position_tolerance']:
                    for ang_tol in param_grid['angle_tolerance']:
                        for len_tol in param_grid['length_tolerance']:
                            iteration += 1
                            
                            print(f"[{iteration}/{total_combinations}] Testing: ", end="")
                            print(f"Motion={reg_motion}, ", end="")
                            print(f"MaxRot={max_rot}°, ", end="")
                            print(f"PosTol={pos_tol}px, ", end="")
                            print(f"AngTol={ang_tol}°, ", end="")
                            print(f"LenTol={len_tol*100}%")
                            
                            result = self.run_test_suite(
                                use_registration=True,
                                registration_motion=reg_motion,
                                max_rotation_degrees=max_rot,
                                position_tolerance=pos_tol,
                                angle_tolerance=ang_tol,
                                length_tolerance=len_tol
                            )
                            
                            prediction_acc = result['statistics']['average_prediction_accuracy']
                            detection_score = result['statistics']['average_detection_score']
                            print(f"   → Test Rating: {prediction_acc*100:.1f}% (Ref Match: {detection_score*100:.1f}%)")
                            
                            all_results.append(result)
        
        # Sort by prediction accuracy (main metric)
        all_results.sort(key=lambda x: x['statistics']['average_prediction_accuracy'], reverse=True)
        
        print("\n" + "=" * 60)
        print("✅ Grid Search Complete!")
        print(f"📊 Best Test Rating: {all_results[0]['statistics']['average_prediction_accuracy']*100:.1f}%")
        print(f"   (Reference Match: {all_results[0]['statistics']['average_detection_score']*100:.1f}%)")
        
        return all_results
    
    def generate_html_report(self, all_results: List[Dict], filename: str = "test_report.html"):
        """Generate HTML report from test results."""
        report_path = os.path.join(self.output_dir, filename)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sort results by prediction accuracy (main metric)
        sorted_results = sorted(
            all_results,
            key=lambda x: x['statistics']['average_prediction_accuracy'],
            reverse=True
        )
        
        best_result = sorted_results[0]
        best_prediction = best_result['statistics']['average_prediction_accuracy'] * 100
        best_detection = best_result['statistics']['average_detection_score'] * 100
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NPSketch Test Report - {timestamp}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .timestamp {{
            color: #666;
            margin-bottom: 30px;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        
        .stat-card.best {{
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left-color: #28a745;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }}
        
        .stat-card.best .stat-value {{
            color: #28a745;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .best-params {{
            background: #fff3cd;
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #ffc107;
            margin-bottom: 40px;
        }}
        
        .best-params h2 {{
            color: #856404;
            margin-bottom: 15px;
        }}
        
        .param-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .param-item {{
            background: white;
            padding: 10px 15px;
            border-radius: 6px;
        }}
        
        .param-label {{
            font-weight: 500;
            color: #666;
            font-size: 0.85em;
        }}
        
        .param-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }}
        
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .results-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 500;
        }}
        
        .results-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        
        .results-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .accuracy-bar {{
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }}
        
        .accuracy-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }}
        
        .accuracy-text {{
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            font-size: 0.75em;
            font-weight: bold;
            color: #333;
        }}
        
        .rank {{
            display: inline-block;
            width: 30px;
            height: 30px;
            line-height: 30px;
            text-align: center;
            border-radius: 50%;
            font-weight: bold;
            color: white;
        }}
        
        .rank-1 {{ background: #ffd700; color: #333; }}
        .rank-2 {{ background: #c0c0c0; color: #333; }}
        .rank-3 {{ background: #cd7f32; color: white; }}
        .rank-other {{ background: #667eea; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 NPSketch Test Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>
        
        <div class="summary">
            <div class="stat-card best">
                <div class="stat-value">{best_prediction:.1f}%</div>
                <div class="stat-label">Best Test Rating</div>
                <div style="font-size: 0.85em; color: #333; margin-top: 5px;">Expected vs Actual</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{best_detection:.1f}%</div>
                <div class="stat-label">Avg Reference Match</div>
                <div style="font-size: 0.85em; color: #333; margin-top: 5px;">vs Reference Image</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(all_results)}</div>
                <div class="stat-label">Configurations Tested</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{best_result['statistics']['total_tests']}</div>
                <div class="stat-label">Test Images</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{best_result['statistics']['perfect_predictions']}</div>
                <div class="stat-label">Perfect Tests</div>
                <div style="font-size: 0.85em; color: #333; margin-top: 5px;">100% Match</div>
            </div>
        </div>
        
        <div class="best-params">
            <h2>🏆 Best Parameters</h2>
            <div class="param-grid">
                <div class="param-item">
                    <div class="param-label">Registration Mode</div>
                    <div class="param-value">{best_result['parameters']['registration_motion']}</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Max Rotation</div>
                    <div class="param-value">{best_result['parameters']['max_rotation_degrees']}°</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Position Tolerance</div>
                    <div class="param-value">{best_result['parameters']['position_tolerance']}px</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Angle Tolerance</div>
                    <div class="param-value">{best_result['parameters']['angle_tolerance']}°</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Length Tolerance</div>
                    <div class="param-value">{best_result['parameters']['length_tolerance']*100:.0f}%</div>
                </div>
            </div>
        </div>
        
        <h2 style="margin-bottom: 20px;">📊 All Configurations (Ranked by Test Rating)</h2>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Test Rating</th>
                    <th>Ref Match</th>
                    <th>Registration</th>
                    <th>Max Rot</th>
                    <th>Position</th>
                    <th>Angle</th>
                    <th>Length</th>
                    <th>Perfect Matches</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for rank, result in enumerate(sorted_results, 1):
            prediction = result['statistics']['average_prediction_accuracy'] * 100
            detection = result['statistics']['average_detection_score'] * 100
            params = result['parameters']
            
            # Color gradient for prediction accuracy bar
            if prediction >= 80:
                color = '#28a745'
            elif prediction >= 60:
                color = '#ffc107'
            else:
                color = '#dc3545'
            
            # Rank badge
            if rank == 1:
                rank_class = 'rank-1'
                rank_emoji = '🥇'
            elif rank == 2:
                rank_class = 'rank-2'
                rank_emoji = '🥈'
            elif rank == 3:
                rank_class = 'rank-3'
                rank_emoji = '🥉'
            else:
                rank_class = 'rank-other'
                rank_emoji = ''
            
            html += f"""
                <tr>
                    <td><span class="rank {rank_class}">{rank_emoji or rank}</span></td>
                    <td>
                        <div class="accuracy-bar">
                            <div class="accuracy-fill" style="width: {prediction}%; background: {color};"></div>
                            <span class="accuracy-text">{prediction:.1f}%</span>
                        </div>
                    </td>
                    <td>{detection:.1f}%</td>
                    <td>{params['registration_motion']}</td>
                    <td>{params['max_rotation_degrees']}°</td>
                    <td>{params['position_tolerance']}px</td>
                    <td>{params['angle_tolerance']}°</td>
                    <td>{params['length_tolerance']*100:.0f}%</td>
                    <td>{result['statistics']['perfect_predictions']}/{result['statistics']['total_tests']}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"\n📄 HTML Report generated: {report_path}")
        return report_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("🧪 NPSketch Automated Test Runner")
    print("=" * 60)
    
    runner = TestRunner()
    
    # Run grid search
    all_results = runner.parameter_grid_search()
    
    # Generate report
    report_path = runner.generate_html_report(all_results)
    
    # Print summary
    best = all_results[0]
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETE!")
    print("=" * 60)
    print(f"🏆 Best Test Rating: {best['statistics']['average_prediction_accuracy']*100:.1f}%")
    print(f"   (Reference Match: {best['statistics']['average_detection_score']*100:.1f}%)")
    print(f"📊 Total Tests: {best['statistics']['total_tests']}")
    print(f"✨ Perfect Tests: {best['statistics']['perfect_predictions']}")
    print(f"\n📄 Full Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

