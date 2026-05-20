import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Ensure root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Clean and complete imports from xy_model
from xy_model import (init_spins, MetropolisXY, PlotXY, EnergyXY, 
                      MagXY, CvXY, CorrXY, VortXY, VortPlotXY, 
                      VortPlotXY_ax, VectorizedMetropolisXY, WolffXY)


def simulate_thermal_states():
    """
    TASK 1.4: Visualizing high and low temperature states.
    """
    L = 64
    n_theta = 16
    J = 1.0
    numIters = 10**6

    print(f"\n--- Starting Task 1.4 (L={L}, Iters={numIters}) ---")
    start_time = time.time()
    
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

    end_time = time.time()
    print(f"\nTask 1.4 completed in {end_time - start_time:.2f} seconds.")
    plt.show()

def run_thermodynamic_simulation():
    """
    TASK 2.5: Full thermodynamic simulation across temperatures.
    Strictly follows assignment instructions: Quench directly to T=0.02.
    """
    L = 64
    n_theta = 16
    J = 1.0
    numPoints = 20
    
    print(f"\n=== Starting Task 2.5 (L={L}) ===")
    start_time = time.time()
    
    # 1. Initialization (Random state)
    lattice = init_spins(L, n_theta)
    
    # 2. Thermalization / Cooling (Quenching directly as instructed)
    T_initial = 0.02
    beta_initial = 1.0 / T_initial
    cooling_iters = 10**6
    print(f"Quenching directly to T={T_initial} ({cooling_iters} iters)...")
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
            
    end_time = time.time()
    
    # Convert to numpy arrays
    E_avg_array = np.array(E_avg_list)
    M_avg_array = np.array(M_avg_list)
    
    # 4. Calculate Heat Capacity (Derivative)
    Cv_array = CvXY(E_avg_array, T_array)
    T_array_Cv = T_array[:-1] # Match array length for plotting
    
    print(f"\nTask 2.5 completed in {end_time - start_time:.2f} seconds.")
    
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

def visualize_vortices():
    """
    TASK 3.3: Analyzes and visualizes vortices at three temperatures 
    (Low, Transition, High) in a single row.
    Uses Metropolis to generate the expected metastable domains and trapped vortices.
    """
    L = 64
    n_theta = 16
    J = 1.0
    numIters = 10**6

    print(f"\n--- Starting Task 3.3: Ultimate Vortex Analysis (L={L}) ---")
    start_time = time.time()
    
    temps = [0.02, 0.95, 10.0]
    titles = ["Low Temp (T=0.02)\nTrapped Vortices at Boundaries", 
              "Transition (T=0.95)\nThermal Pairs Breaking", 
              "High Temp (T=10.0)\nFree Vortex Plasma"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Task 3.3: Vortex Evolution Across Temperatures", fontsize=18)
    
    initial_lattice = init_spins(L, n_theta)
    
    for ax, T, title in zip(axes, temps, titles):
        beta = 1.0 / T
        print(f"Thermalizing at T={T} with {numIters} iters...")
        
        # Using the standard Metropolis quench to recreate the trapped vortices!
        lattice = MetropolisXY(initial_lattice, n_theta, beta, J, numIters)
        
        V, NumVort = VortXY(lattice)
        print(f" -> Found {NumVort:.0f} vortices.")
        
        im = VortPlotXY_ax(lattice, V, ax, title=f"{title}\nVortices: {NumVort:.0f}")
        
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.04)
    cbar.set_label('Spin Angle (Radians)')
    
    end_time = time.time()
    print(f"\nTask 3.3 completed in {end_time - start_time:.2f} seconds.")
    plt.show()

def simulate_vortex_density():
    """
    TASK 3.4: Calculates and plots the vortex density as a function 
    of 1/T on a semilogarithmic scale.
    """
    L = 64
    N = L**2
    n_theta = 16
    J = 1.0
    numPoints = 20
    
    print(f"\n=== Starting Task 3.4: Vortex Density (L={L}) ===")
    start_time = time.time()
    
    lattice = init_spins(L, n_theta)
    
    # Thermalization
    T_initial = 0.02
    beta_initial = 1.0 / T_initial
    cooling_iters = 10**6
    print(f"Thermalizing at T={T_initial} ({cooling_iters} iters)...")
    lattice = MetropolisXY(lattice, n_theta, beta_initial, J, cooling_iters)
    
    T_array = np.linspace(0.02, 2.0, numPoints)
    beta_array = 1.0 / T_array  # This is 1 / k_B T
    density_list = []
    
    print("\n--- Starting Heating Loop ---")
    for i, T in enumerate(T_array):
        beta = 1.0 / T
        vort_accum = 0.0
        
        print(f"Sampling T = {T:.3f} (1/T = {beta:.2f}) ...", end="", flush=True)
        
        # 100 Independent Measurements
        for _ in range(100):
            lattice = MetropolisXY(lattice, n_theta, beta, J, 10**4)
            _, NumVort = VortXY(lattice)
            vort_accum += NumVort
            
        # Average number of vortices for this temperature
        avg_NumVort = vort_accum / 100.0
        
        # Density is average vortices divided by total spins (N)
        density_list.append(avg_NumVort / N)
        print(" Done.")
        
    density_array = np.array(density_list)
    end_time = time.time()
    print(f"\nTask 3.4 completed in {end_time - start_time:.2f} seconds.")
    
    # Plotting on a semilog-y axis
    plt.figure(figsize=(8, 6))
    plt.semilogy(beta_array, density_array, 'o-', color='purple')
    plt.title('Task 3.4: Vortex Density vs 1 / T')
    plt.xlabel('Inverse Temperature (1 / k_B T)')
    plt.ylabel('Vortex Density (<NumVort> / N)')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

def run_fast_thermodynamics():
    """
    BONUS TASK: High-Performance Thermodynamics using Vectorized Metropolis.
    Produces all 4 plots identical to Task 2.5 but magnitudes faster.
    """
    L = 64
    n_theta = 16
    J = 1.0
    numPoints = 20
    
    print(f"\n=== Starting BONUS TASK: FAST Thermodynamics (L={L}) ===")
    print("Using Vectorized Checkerboard Update...")
    
    start_time = time.time()
    
    lattice = init_spins(L, n_theta)
    
    T_initial = 0.02
    beta_initial = 1.0 / T_initial
    cooling_iters = 10**6
    print("Fast Thermalizing...")
    lattice = VectorizedMetropolisXY(lattice, n_theta, beta_initial, J, cooling_iters)
    
    T_array = np.linspace(0.02, 2.0, numPoints)
    E_avg_list, M_avg_list = [], []
    C_r_low, C_r_high = None, None
    
    print("Fast Heating Loop...")
    for i, T in enumerate(T_array):
        beta = 1.0 / T
        E_accum = 0.0
        M_accum = 0.0
        
        for _ in range(100):
            lattice = VectorizedMetropolisXY(lattice, n_theta, beta, J, 10**4)
            E_accum += EnergyXY(lattice, J)
            M_accum += MagXY(lattice)
            
        E_avg_list.append(E_accum / 100.0)
        M_avg_list.append(M_accum / 100.0)
        
        # Capture Correlation function
        if i == 0:
            C_r_low = CorrXY(lattice)
        elif i == numPoints - 1:
            C_r_high = CorrXY(lattice)
            
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate Heat Capacity
    E_avg_array = np.array(E_avg_list)
    M_avg_array = np.array(M_avg_list)
    Cv_array = CvXY(E_avg_array, T_array)
    T_array_Cv = T_array[:-1]
    
    print(f"\nFast Vectorized Simulation completed in {total_time:.2f} seconds.")
    
    # Plotting all 4 graphs
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"FAST Vectorized Thermodynamics (Time: {total_time:.2f}s)", fontsize=16)
    
    axs[0, 0].plot(T_array, E_avg_array, 'o-', color='blue')
    axs[0, 0].set(title='Average Energy vs Temp', xlabel='T', ylabel='<E>/N')
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(T_array, M_avg_array, 'o-', color='red')
    axs[0, 1].set(title='Squared Magnetization vs Temp', xlabel='T', ylabel='<M^2>/N^2')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(T_array_Cv, Cv_array, 's-', color='green')
    axs[1, 0].set(title='Heat Capacity vs Temp', xlabel='T', ylabel='C_v')
    axs[1, 0].grid(True)
    
    r_values = np.arange(1, len(C_r_low) + 1)
    axs[1, 1].plot(r_values, C_r_low, 'o-', label=f'T={T_array[0]:.2f}', color='cyan')
    axs[1, 1].plot(r_values, C_r_high, 's-', label=f'T={T_array[-1]:.2f}', color='magenta')
    axs[1, 1].set(title='Spatial Correlation C(r)', xlabel='Distance (r)', ylabel='C(r)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_wolff_thermodynamics():
    """
    BONUS TASK: Perfect Hybrid Thermodynamics (Checkerboard + Wolff).
    Includes physical Annealing, 3-state visualization, and 3-temp correlation.
    """
    L = 64
    n_theta = 16
    J = 1.0
    numPoints = 20
    
    print(f"\n=== Starting PERFECT HYBRID Thermodynamics (L={L}) ===")
    start_time = time.time()
    
    lattice = init_spins(L, n_theta)
    
    # --- PHASE 1: ANNEALING (Cooling down properly) ---
    print("Annealing to ground state to prevent spin-glass frustration...")
    anneal_temps = np.linspace(2.0, 0.02, 10)
    for T_ann in anneal_temps:
        beta_ann = 1.0 / T_ann
        lattice = VectorizedMetropolisXY(lattice, n_theta, beta_ann, J, sweeps=100)
        lattice = WolffXY(lattice, n_theta, beta_ann, J, num_clusters=10)
    
    # Deep freeze at the target initial temperature
    lattice = VectorizedMetropolisXY(lattice, n_theta, 1.0/0.02, J, sweeps=1000)
    
    # --- PHASE 2: MEASUREMENT (Heating up) ---
    T_array = np.linspace(0.02, 2.0, numPoints)
    mid_index = 7  # T_array[7] is approximately 0.75
    
    E_avg_list, M_avg_list = [], []
    
    # Variables to store snapshots for the 3 specific temperatures
    C_r_low, C_r_mid, C_r_high = None, None, None
    lattice_low, lattice_mid, lattice_high = None, None, None
    
    print("Heating Loop running...")
    for i, T in enumerate(T_array):
        beta = 1.0 / T
        E_accum = 0.0
        M_accum = 0.0
        
        for _ in range(100):
            # The Ultimate Hybrid: 10 Local sweeps + 5 Global cluster flips
            lattice = VectorizedMetropolisXY(lattice, n_theta, beta, J, sweeps=10)
            lattice = WolffXY(lattice, n_theta, beta, J, num_clusters=5)
            
            E_accum += EnergyXY(lattice, J)
            M_accum += MagXY(lattice)
            
        E_avg_list.append(E_accum / 100.0)
        M_avg_list.append(M_accum / 100.0)
        
        # Capture Correlation and Grid Snapshots at the 3 specific points
        if i == 0:
            C_r_low = CorrXY(lattice)
            lattice_low = lattice.copy()
        elif i == mid_index:
            C_r_mid = CorrXY(lattice)
            lattice_mid = lattice.copy()
        elif i == numPoints - 1:
            C_r_high = CorrXY(lattice)
            lattice_high = lattice.copy()
            
    end_time = time.time()
    total_time = end_time - start_time
    
    E_avg_array = np.array(E_avg_list)
    M_avg_array = np.array(M_avg_list)
    Cv_array = CvXY(E_avg_array, T_array)
    T_array_Cv = T_array[:-1]
    
    print(f"\nPerfect Hybrid Simulation completed in {total_time:.2f} seconds.")
    
    # --- PLOTTING PHASE ---
    
    # Figure 1: The 3 Spin Configurations with Vortices (using VortPlotXY_ax)
    print("Generating Visualizations...")
    vis_temps = [T_array[0], T_array[mid_index], T_array[-1]]
    vis_lattices = [lattice_low, lattice_mid, lattice_high]
    titles = [f"Low Temp (T={vis_temps[0]:.2f})\nPerfect Order", 
              f"Mid Temp (T={vis_temps[1]:.2f})\nBound Pairs Emerging", 
              f"High Temp (T={vis_temps[2]:.2f})\nFree Vortex Plasma"]
    
    fig_vis, axes_vis = plt.subplots(1, 3, figsize=(18, 6))
    fig_vis.suptitle("Wolff Algorithm: Spin Configurations & Vortices", fontsize=18)
    
    for ax, temp_lat, title in zip(axes_vis, vis_lattices, titles):
        V, NumVort = VortXY(temp_lat)
        im = VortPlotXY_ax(temp_lat, V, ax, title=f"{title}\nVortices: {NumVort:.0f}")
        
    cbar = fig_vis.colorbar(im, ax=axes_vis.ravel().tolist(), fraction=0.015, pad=0.04)
    cbar.set_label('Spin Angle (Radians)')
    
    # Figure 2: The Thermodynamic 2x2 Grid
    fig_thermo, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig_thermo.suptitle(f"HYBRID WOLFF Thermodynamics (Time: {total_time:.2f}s)", fontsize=16)
    
    axs[0, 0].plot(T_array, E_avg_array, 'o-', color='blue')
    axs[0, 0].set(title='Average Energy vs Temp', xlabel='T', ylabel='<E>/N')
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(T_array, M_avg_array, 'o-', color='red')
    axs[0, 1].set(title='Squared Magnetization vs Temp', xlabel='T', ylabel='<M^2>/N^2')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(T_array_Cv, Cv_array, 's-', color='green')
    axs[1, 0].set(title='Heat Capacity vs Temp', xlabel='T', ylabel='C_v')
    axs[1, 0].grid(True)
    
    # Plotting Correlation with all 3 temperatures
    r_values = np.arange(1, len(C_r_low) + 1)
    axs[1, 1].plot(r_values, C_r_low, 'o-', label=f'T={T_array[0]:.2f}', color='cyan')
    axs[1, 1].plot(r_values, C_r_mid, '^-', label=f'T={T_array[mid_index]:.2f}', color='orange')
    axs[1, 1].plot(r_values, C_r_high, 's-', label=f'T={T_array[-1]:.2f}', color='magenta')
    axs[1, 1].set(title='Spatial Correlation C(r)', xlabel='Distance (r)', ylabel='C(r)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display both figures simultaneously
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
    print("3: Run Both Tasks (1.4 & 2.5)")
    print("4: Run Task 3.3 (Vortex Visualization)")
    print("5: Run Task 3.4 (Vortex Density Plot)")
    print("6: Run BONUS 1 (Fast Vectorized Thermodynamics)")
    print("7: Run BONUS 2 (Wolff Cluster Thermodynamics)")
    print("0: Exit")
    print("=========================================")
    
    choice = input("Enter choice (0/1/2/3/4/5/6/7): ")
    
    if choice == '1':
        simulate_thermal_states()
    elif choice == '2':
        run_thermodynamic_simulation()
    elif choice == '3':
        simulate_thermal_states()
        run_thermodynamic_simulation()
    elif choice == '4':
        visualize_vortices()
    elif choice == '5':
        simulate_vortex_density()
    elif choice == '6':
        run_fast_thermodynamics()
    elif choice == '7':
        run_wolff_thermodynamics()
    elif choice == '0':
        sys.exit()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()