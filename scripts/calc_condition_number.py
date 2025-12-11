import torch
import sys
import numpy as np

def analyze_probe(model_path):
    print(f"Loading probe from: {model_path}")
    
    # Load the state dictionary
    try:
        params = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Extract the projection matrix 'proj'
    # In TwoWordPSDProbe, the matrix B is stored as 'proj'
    if 'proj' not in params:
        print("Error: Could not find 'proj' weight matrix. Is this a Linear Probe?")
        print(f"Keys found: {list(params.keys())}")
        return

    B = params['proj'] # Shape: [model_dim, rank]
    
    # Calculate Singular Value Decomposition (SVD)
    # The singular values (S) tell us exactly how much the matrix scales 
    # the space along each principal axis.
    U, S, V = torch.linalg.svd(B, full_matrices=False)
    
    max_scale = S.max().item()
    min_scale = S.min().item()
    
    # The Condition Number is the ratio of the largest stretch to the smallest stretch
    # Ideally, for an isometry, this should be 1.0 (Uniform scaling)
    cond_number = max_scale / (min_scale + 1e-9) # avoid div by zero

    print("-" * 30)
    print(f"Shape of Matrix B: {B.shape}")
    print(f"Max Scaling (Stretching): {max_scale:.4f}")
    print(f"Min Scaling (Squashing):  {min_scale:.4f}")
    print("-" * 30)
    print(f"CONDITION NUMBER: {cond_number:.4f}")
    print("-" * 30)
    
    if cond_number < 1.5:
        print("INTERPRETATION: Minimal distortion. The space is nearly isometric.")
    elif cond_number < 5.0:
        print("INTERPRETATION: Moderate distortion.")
    else:
        print("INTERPRETATION: High distortion. The probe is significantly reshaping the space.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.calc_condition_number <path_to_predictor.params>")
    else:
        analyze_probe(sys.argv[1])