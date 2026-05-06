import numpy as np
from xy_model import EnergyXY, MagXY

def run_thermodynamics_test():
    """
    Validates thermodynamic calculations using a known ground state.
    In a fully aligned lattice (all angles = 0), energy per spin should be -2J
    and normalized squared magnetization should be 1.0.
    """
    print("--- Running Thermodynamics Validation (Ground State) ---")
    
    L = 10
    J = 1.0
    # Create a lattice where all spins are perfectly aligned at angle 0
    ground_state = np.zeros((L, L))
    
    # Calculate physical properties
    energy = EnergyXY(ground_state, J)
    magnetization = MagXY(ground_state)
    
    print(f"Calculated Energy per spin: {energy}")
    print(f"Calculated Magnetization: {magnetization}")
    
    # Validations
    energy_ok = np.isclose(energy, -2.0 * J)
    mag_ok = np.isclose(magnetization, 1.0)
    
    if energy_ok and mag_ok:
        print("Validation Result: SUCCESS. Thermodynamic functions are accurate.\n")
        return True
    else:
        print("Validation Result: FAILURE. Incorrect thermodynamic values.\n")
        return False

if __name__ == "__main__":
    run_thermodynamics_test()