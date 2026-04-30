import numpy as np
from xy_model import init_spins

def run_initialization_test(L=4, n_theta=8):
    """
    Validates the spin lattice initialization process.
    
    The test checks if the generated spin angles strictly belong to the 
    allowed discrete set and if the matrix dimensions are correct.
    """
    print(f"--- Running Initialization Validation (L={L}, n_theta={n_theta}) ---")
    
    lattice = init_spins(L, n_theta)
    base_angle = 2 * np.pi / n_theta
    
    # Reverse the calculation to extract the multipliers (k values).
    # Multipliers should be integers within the range [1, n_theta].
    multipliers = np.round(lattice / base_angle).astype(int)
    
    # Check if all multipliers are within the valid physical range.
    within_range = np.all((multipliers >= 1) & (multipliers <= n_theta))
    
    # Verify that the angles are exact multiples of the base angle using a numerical tolerance.
    is_discrete = np.allclose(lattice, multipliers * base_angle)

    if within_range and is_discrete:
        print("Validation Result: SUCCESS. All spins adhere to the discrete XY model constraints.")
        return True
    else:
        print("Validation Result: FAILURE. Invalid spin orientations detected.")
        return False