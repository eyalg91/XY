import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from xy_model import init_spins, MetropolisXY, PlotXY
from tests.test_initialization import run_initialization_test
from tests.test_metropolis import run_metropolis_test
from tests.test_plot import run_plot_test

def run_task_1_4():
    """
    Simulating and plotting the XY model 
    at high (10.0) and low (0.02) temperatures.
    """
    L = 100
    n_theta = 16
    J = 1.0
    numIters = 5*10**6

    print(f"\n--- Starting Task 1.4 (L={L}, Iters={numIters}) ---")
    
    # Initialize the base lattice
    print("Initializing base lattice...")
    initial_lattice = init_spins(L, n_theta)

    # 1. High Temperature Simulation
    T_high = 10.0
    beta_high = 1.0 / T_high
    print(f"Running Metropolis for High Temperature (k_B T = {T_high}). Please wait...")
    lattice_high = MetropolisXY(initial_lattice, n_theta, beta_high, J, numIters)
    PlotXY(lattice_high, title=f"XY Model: High Temperature (k_B T = {T_high})")

    # 2. Low Temperature Simulation
    T_low = 0.02
    beta_low = 1.0 / T_low
    print(f"Running Metropolis for Low Temperature (k_B T = {T_low}). Please wait...")
    lattice_low = MetropolisXY(initial_lattice, n_theta, beta_low, J, numIters)
    PlotXY(lattice_low, title=f"XY Model: Low Temperature (k_B T = {T_low})")

    print("Simulations complete. Rendering plots...")
    plt.show()

def main():
    """
    Main execution script. Runs validations first, then physical simulations.
    """
    L_SIZE_TEST = 20
    N_DISCRETE_ANGLES = 16
    
    print("Running system validations...")
    init_ok = run_initialization_test(L=L_SIZE_TEST, n_theta=N_DISCRETE_ANGLES)
    metro_ok = run_metropolis_test(L=L_SIZE_TEST, n_theta=N_DISCRETE_ANGLES)
    
    # We can skip the visual test run_plot_test here to avoid blocking 
    # the workflow, as Task 1.4 will plot the final results anyway.
    
    if init_ok and metro_ok:
        print("\nAll system validations passed. Proceeding to physical simulations.")
        run_task_1_4()
    else:
        print("\nSystem validation failed. Please check the logic.")

if __name__ == "__main__":
    main()