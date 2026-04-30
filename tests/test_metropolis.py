import numpy as np
from xy_model import init_spins, MetropolisXY

def run_metropolis_test(L=10, n_theta=8):
    """
    Validates the MetropolisXY function.
    
    Ensures that the algorithm returns a matrix of the correct shape 
    and that all spins remain within the allowed discrete set after updates.
    """
    print(f"--- Running Metropolis Validation (L={L}, iterations=1000) ---")
    
    # Setup initial state
    initial_lattice = init_spins(L, n_theta)
    beta = 1.0  # arbitrary test temperature
    J = 1.0     # interaction strength
    numIters = 1000
    
    # Run the Metropolis algorithm
    final_lattice = MetropolisXY(initial_lattice, n_theta, beta, J, numIters)
    
    # Check 1: Output shape should remain L x L
    shape_ok = final_lattice.shape == (L, L)
    
    # Check 2: All values must still be valid discrete angles
    base_angle = 2 * np.pi / n_theta
    multipliers = np.round(final_lattice / base_angle).astype(int)
    within_range = np.all((multipliers >= 1) & (multipliers <= n_theta))
    is_discrete = np.allclose(final_lattice, multipliers * base_angle)
    
    if shape_ok and within_range and is_discrete:
        print("Validation Result: SUCCESS. Metropolis step maintains physical constraints.\n")
        return True
    else:
        print("Validation Result: FAILURE. Constraints violated during Metropolis steps.\n")
        return False