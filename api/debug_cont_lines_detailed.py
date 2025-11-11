#!/usr/bin/env python3
"""Debug script to understand the exact structure of cont_lines."""

import scipy.io
import numpy as np

mat_file = "/app/templates/bsp_ocsplus_202511/Machine_rater/matfiles/FIGURECOPY_data_OCSPlus_210426_Pro1003_PC56_German_20210426T125245.mat"

print(f"Loading: {mat_file}\n")
mat_data = scipy.io.loadmat(mat_file)

for key_prefix in ['data_complex_copy', 'data_complex_memory_copy']:
    task_type = 'COPY' if 'memory' not in key_prefix else 'RECALL'
    print(f"=== {task_type} ===")
    
    data = mat_data[key_prefix][0, 0]
    trails = data['trails'][0, 0]
    
    print(f"trails['cont_lines'] shape: {trails['cont_lines'].shape}")
    print(f"trails['cont_lines'] dtype: {trails['cont_lines'].dtype}")
    
    # Try different unwrapping levels
    print("\nLevel 1: trails['cont_lines']")
    level1 = trails['cont_lines']
    print(f"  shape: {level1.shape}, dtype: {level1.dtype}")
    
    if level1.shape == (1, 1):
        print("\nLevel 2: trails['cont_lines'][0, 0]")
        level2 = level1[0, 0]
        print(f"  type: {type(level2)}")
        print(f"  shape: {level2.shape if hasattr(level2, 'shape') else 'N/A'}")
        print(f"  dtype: {level2.dtype if hasattr(level2, 'dtype') else 'N/A'}")
        
        if hasattr(level2, 'shape') and level2.dtype == object:
            print(f"\n  This is an array of objects with shape {level2.shape}")
            print(f"  Number of lines: {level2.shape[1] if len(level2.shape) > 1 else level2.shape[0]}")
            
            # Inspect individual lines
            num_lines = level2.shape[1] if len(level2.shape) > 1 else level2.shape[0]
            print(f"\n  Inspecting each line:")
            for i in range(min(15, num_lines)):
                if len(level2.shape) > 1:
                    line = level2[0, i]
                else:
                    line = level2[i]
                
                if isinstance(line, np.ndarray):
                    print(f"    Line {i+1}: shape={line.shape}, dtype={line.dtype}, points={line.shape[0] if len(line.shape) > 0 else 0}")
                    if line.shape[0] > 0:
                        print(f"             First point: x={line[0,0]:.1f}, y={line[0,1]:.1f}")
                else:
                    print(f"    Line {i+1}: type={type(line)}, value={line}")
    
    print("\n")

