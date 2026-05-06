import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xy_model import init_spins, MetropolisXY, PlotXY, EnergyXY, MagXY, CvXY, CorrXY

def simulate_thermal_states():
    """
    TASK 1.4: Visualizing high and low temperature states.
    """
    L = 64
    n_theta = 16
    J = 1.0
    numIters = 10**6

    print(f"\n--- Starting Task 1.4 (L={L}, Iters={numIters}) ---")
    
    # 1. Initialization
    initial_lattice = init_spins(L, n_theta)

    # 2. High Temperature (Disordered)
    T_high = 10.0
    beta_high = 1.0 / T_high
    print(f"Running High Temp (T={T_high})...")
    lattice_high = MetropolisXY(initial_lattice, n_theta, beta_high, J, numIters)
    PlotXY(lattice_high, title=f"High Temp Disordered State (T={T_high})")

    # 3. Low Temperature (Quasi-Ordered)
    T_low = 0.02
    beta_low = 1.0 / T_low
    print(f"Running Low Temp (T={T_low})...")
    lattice_low = MetropolisXY(initial_lattice, n_theta, beta_low, J, numIters)
    PlotXY(lattice_low, title=f"Low Temp Quasi-Ordered State (T={T_low})")

    plt.show()

def run_thermodynamic_simulation():
    """
    TASK 2.5: Full thermodynamic simulation across temperatures.
    """
    L = 64
    n_theta = 16
    J = 1.0
    numPoints = 20
    
    print(f"\n=== Starting Task 2.5 (L={L}) ===")
    
    # 1. Initialization (Random state)
    lattice = init_spins(L, n_theta)
    
    # 2. Thermalization / Cooling
    T_initial = 0.02
    beta_initial = 1.0 / T_initial
    cooling_iters = 10**6
    print(f"Thermalizing at T={T_initial} ({cooling_iters} iters)...")
    lattice = MetropolisXY(lattice, n_theta, beta_initial, J, cooling_iters)
    
    # Setup arrays for data collection
    T_array = np.linspace(0.02, 2.0, numPoints)
    E_avg_list, M_avg_list = [], []
    C_r_low, C_r_high = None, None
    
    # 3. Main Heating Loop
    print("\n--- Starting Heating Loop ---")
    for i, T in enumerate(T_array):
        beta = 1.0 / T
        E_accum = 0.0
        M_accum = 0.0
        
        print(f"Sampling T = {T:.3f} ...", end="", flush=True)
        
        # 100 Independent Measurements
        for _ in range(100):
            # Decorrelation steps
            lattice = MetropolisXY(lattice, n_theta, beta, J, 10**4)
            
            # Measurement
            E_accum += EnergyXY(lattice, J)
            M_accum += MagXY(lattice)
            
        # Store thermodynamic averages
        E_avg_list.append(E_accum / 100.0)
        M_avg_list.append(M_accum / 100.0)
        print(" Done.")
        
        # Store correlation function at extremes
        if i == 0:
            C_r_low = CorrXY(lattice)
        elif i == numPoints - 1:
            C_r_high = CorrXY(lattice)
            
    # Convert to numpy arrays
    E_avg_array = np.array(E_avg_list)
    M_avg_array = np.array(M_avg_list)
    
    # 4. Calculate Heat Capacity (Derivative)
    Cv_array = CvXY(E_avg_array, T_array)
    T_array_Cv = T_array[:-1] # Match array length for plotting
    
    # 5. Plotting Results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"XY Model Thermodynamics (L={L})", fontsize=16)
    
    # Energy
    axs[0, 0].plot(T_array, E_avg_array, 'o-', color='blue')
    axs[0, 0].set(title='Average Energy vs Temp', xlabel='T', ylabel='<E>/N')
    axs[0, 0].grid(True)
    
    # Magnetization
    axs[0, 1].plot(T_array, M_avg_array, 'o-', color='red')
    axs[0, 1].set(title='Squared Magnetization vs Temp', xlabel='T', ylabel='<M^2>/N^2')
    axs[0, 1].grid(True)
    
    # Heat Capacity
    axs[1, 0].plot(T_array_Cv, Cv_array, 's-', color='green')
    axs[1, 0].set(title='Heat Capacity vs Temp', xlabel='T', ylabel='C_v')
    axs[1, 0].grid(True)
    
    # Correlation
    r_values = np.arange(1, len(C_r_low) + 1)
    axs[1, 1].plot(r_values, C_r_low, 'o-', label=f'T={T_array[0]:.2f}', color='cyan')
    axs[1, 1].plot(r_values, C_r_high, 's-', label=f'T={T_array[-1]:.2f}', color='magenta')
    axs[1, 1].set(title='Spatial Correlation C(r)', xlabel='Distance (r)', ylabel='C(r)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    

def main():
    """
    Main menu for execution.
    """
    print("=========================================")
    print("     XY Model Simulation - Main Menu     ")
    print("=========================================")
    print("1: Run Task 1.4 (Phase Extremes)")
    print("2: Run Task 2.5 (Thermodynamics)")
    print("3: Run Both Tasks")
    print("0: Exit")
    print("=========================================")
    
    choice = input("Enter choice (0/1/2/3): ")
    
    if choice == '1':
        simulate_thermal_states()
    elif choice == '2':
        run_thermodynamic_simulation()
    elif choice == '3':
        simulate_thermal_states()
        run_thermodynamic_simulation()
    elif choice == '0':
        sys.exit()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()