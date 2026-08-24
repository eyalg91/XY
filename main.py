import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Ensure root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xy_model import (init_spins, MetropolisXY, PlotXY, EnergyXY,
                      MagXY, CvXY, CorrXY, VortXY, VortPlotXY,
                      VortPlotXY_ax, VectorizedMetropolisXY, WolffXY)

PLOTS_DIR = "PLOTS"
SAVE_PLOTS = False  # When True (during "Run All"), plots are saved to PLOTS_DIR instead of shown.

# Larger, report-friendly font sizes for every figure (axis labels, ticks, legends, subplot titles).
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
})

def finish_plot(figs_with_names):
    """Saves figures to PLOTS_DIR (high-res PNG) if SAVE_PLOTS is on, otherwise shows them as before."""
    if SAVE_PLOTS:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        for fig, name in figs_with_names:
            path = os.path.join(PLOTS_DIR, f"{name}.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        print(f"Saved {len(figs_with_names)} plot(s) to '{PLOTS_DIR}/'")
    else:
        plt.show()

def simulate_thermal_states():
    """TASK 1.4: Visualizing high and low temperature states."""
    L = 64
    n_theta = 16
    J = 1.0
    numIters = 10**6

    print(f"\n--- Starting Task 1.4 (L={L}, Iters={numIters}) ---")
    start_time = time.time()
    
    initial_lattice = init_spins(L, n_theta)

    T_high = 10.0
    print(f"Running High Temp (T={T_high})...")
    lattice_high = MetropolisXY(initial_lattice, n_theta, 1.0/T_high, J, numIters)
    fig_high = PlotXY(lattice_high)

    T_low = 0.02
    print(f"Running Low Temp (T={T_low})...")
    lattice_low = MetropolisXY(initial_lattice, n_theta, 1.0/T_low, J, numIters)
    fig_low = PlotXY(lattice_low)

    end_time = time.time()
    print(f"\nTask 1.4 completed in {end_time - start_time:.2f} seconds.")
    finish_plot([
        (fig_high, "SpinConfiguration_Disordered_HighTemperature_T10.0"),
        (fig_low, "SpinConfiguration_QuasiOrdered_LowTemperature_T0.02"),
    ])

def run_thermodynamic_simulation():
    """TASK 2.5: Full thermodynamic simulation. Quenches directly to T=0.02."""
    L = 64
    n_theta = 16
    J = 1.0
    numPoints = 20
    
    print(f"\n=== Starting Task 2.5 (L={L}) ===")
    start_time = time.time()
    
    lattice = init_spins(L, n_theta)
    
    T_initial = 0.02
    cooling_iters = 10**6
    print(f"Quenching directly to T={T_initial} ({cooling_iters} iters)...")
    lattice = MetropolisXY(lattice, n_theta, 1.0/T_initial, J, cooling_iters)
    
    T_array = np.linspace(0.02, 2.0, numPoints)
    E_avg_list, M_avg_list = [], []
    C_r_low, C_r_high = None, None
    
    print("\n--- Starting Heating Loop ---")
    for i, T in enumerate(T_array):
        beta = 1.0 / T
        E_accum, M_accum = 0.0, 0.0
        print(f"Sampling T = {T:.3f} ...", end="", flush=True)
        for _ in range(100):
            lattice = MetropolisXY(lattice, n_theta, beta, J, 10**4)
            E_accum += EnergyXY(lattice, J)
            M_accum += MagXY(lattice)
        E_avg_list.append(E_accum / 100.0)
        M_avg_list.append(M_accum / 100.0)
        print(" Done.")
        if i == 0: C_r_low = CorrXY(lattice)
        elif i == numPoints - 1: C_r_high = CorrXY(lattice)
            
    end_time = time.time()
    E_avg_array, M_avg_array = np.array(E_avg_list), np.array(M_avg_list)
    Cv_array = CvXY(E_avg_array, T_array)
    
    print(f"\nTask 2.5 completed in {end_time - start_time:.2f} seconds.")
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].plot(T_array, E_avg_array, 'o-', color='blue')
    axs[0, 0].set(title='Average Energy vs. Temperature', xlabel='T', ylabel='<E>/N')
    axs[0, 0].grid(True)
    axs[0, 1].plot(T_array, M_avg_array, 'o-', color='red')
    axs[0, 1].set(title='Squared Magnetization vs. Temperature', xlabel='T', ylabel='<M^2>/N^2')
    axs[0, 1].grid(True)
    axs[1, 0].plot(T_array[:-1], Cv_array, 's-', color='green')
    axs[1, 0].set(title='Heat Capacity vs. Temperature', xlabel='T', ylabel='C_v')
    axs[1, 0].grid(True)
    r_values = np.arange(1, len(C_r_low) + 1)
    axs[1, 1].plot(r_values, C_r_low, 'o-', label=f'T={T_array[0]:.2f}', color='cyan')
    axs[1, 1].plot(r_values, C_r_high, 's-', label=f'T={T_array[-1]:.2f}', color='magenta')
    axs[1, 1].set(title='Spin-Spin Correlation Function', xlabel='Distance (r)', ylabel='C(r)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    plt.tight_layout()
    finish_plot([(fig, "ThermodynamicObservables_StandardMetropolis_L64")])

def visualize_vortices():
    """TASK 3.3: Analyzes and visualizes vortices at three temperatures."""
    L = 64
    n_theta = 16
    J = 1.0
    numIters = 10**6

    print(f"\n--- Starting Task 3.3: Ultimate Vortex Analysis (L={L}) ---")
    start_time = time.time()
    temps = [0.02, 0.95, 10.0]
    titles = ["Low Temperature (T=0.02)\nTrapped Vortices at Boundaries",
              "Near the Transition (T=0.95)\nThermal Pairs Breaking",
              "High Temperature (T=10.0)\nFree Vortex Plasma"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    initial_lattice = init_spins(L, n_theta)

    for ax, T, title in zip(axes, temps, titles):
        print(f"Thermalizing at T={T} with {numIters} iters...")
        lattice = MetropolisXY(initial_lattice, n_theta, 1.0/T, J, numIters)
        V, NumVort = VortXY(lattice)
        print(f" -> Found {NumVort:.0f} vortices.")
        im = VortPlotXY_ax(lattice, V, ax, title=f"{title}\nVortex Count: {NumVort:.0f}")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.04)
    cbar.set_label('Spin Angle (Radians)')
    print(f"\nTask 3.3 completed in {time.time() - start_time:.2f} seconds.")
    finish_plot([(fig, "VortexEvolution_AcrossTemperature_L64")])

def simulate_vortex_density():
    """TASK 3.4: Calculates and plots the vortex density."""
    L = 64
    N = L**2
    n_theta = 16
    J = 1.0
    numPoints = 20
    
    print(f"\n=== Starting Task 3.4: Vortex Density (L={L}) ===")
    start_time = time.time()
    lattice = init_spins(L, n_theta)
    
    cooling_iters = 10**6
    print(f"Thermalizing at T=0.02 ({cooling_iters} iters)...")
    lattice = MetropolisXY(lattice, n_theta, 1.0/0.02, J, cooling_iters)
    
    T_array = np.linspace(0.02, 2.0, numPoints)
    density_list = []
    
    print("\n--- Starting Heating Loop ---")
    for i, T in enumerate(T_array):
        vort_accum = 0.0
        print(f"Sampling T = {T:.3f} ...", end="", flush=True)
        for _ in range(100):
            lattice = MetropolisXY(lattice, n_theta, 1.0/T, J, 10**4)
            _, NumVort = VortXY(lattice)
            vort_accum += NumVort
        density_list.append((vort_accum / 100.0) / N)
        print(" Done.")
        
    fig = plt.figure(figsize=(8, 6))
    plt.semilogy(1.0/T_array, np.array(density_list), 'o-', color='purple')
    plt.xlabel('Inverse Temperature (1 / k_B T)')
    plt.ylabel('Vortex Density (<NumVort> / N)')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    print(f"\nTask 3.4 completed in {time.time() - start_time:.2f} seconds.")
    finish_plot([(fig, "VortexDensity_vs_InverseTemperature_StandardMetropolis_L64")])

def run_fast_thermodynamics():
    """BONUS 1: Fast Vectorized Thermodynamics."""
    L = 64
    N = L**2
    n_theta = 16
    J = 1.0
    numPoints = 20
    
    print(f"\n=== Starting BONUS TASK: FAST Thermodynamics (L={L}) ===")
    start_time = time.time()
    lattice = init_spins(L, n_theta)
    
    cooling_sweeps = int(10**6 / N) 
    print(f"Fast Thermalizing ({cooling_sweeps} full lattice sweeps)...")
    lattice = VectorizedMetropolisXY(lattice, n_theta, 1.0/0.02, J, cooling_sweeps)
    
    T_array = np.linspace(0.02, 2.0, numPoints)
    E_avg_list, M_avg_list = [], []
    C_r_low, C_r_high = None, None
    decorrelation_sweeps = max(1, int(10**4 / N))
    
    print("Fast Heating Loop...")
    for i, T in enumerate(T_array):
        E_accum, M_accum = 0.0, 0.0
        for _ in range(100):
            lattice = VectorizedMetropolisXY(lattice, n_theta, 1.0/T, J, decorrelation_sweeps)
            E_accum += EnergyXY(lattice, J)
            M_accum += MagXY(lattice)
        E_avg_list.append(E_accum / 100.0)
        M_avg_list.append(M_accum / 100.0)
        if i == 0: C_r_low = CorrXY(lattice)
        elif i == numPoints - 1: C_r_high = CorrXY(lattice)
            
    end_time = time.time()
    print(f"BONUS 1 completed in {end_time - start_time:.2f} seconds.")
    E_avg_array, M_avg_array = np.array(E_avg_list), np.array(M_avg_list)
    Cv_array = CvXY(E_avg_array, T_array)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(T_array, E_avg_array, 'o-', color='blue')
    axs[0, 0].set(title='Average Energy vs. Temperature', xlabel='T', ylabel='<E>/N')
    axs[0, 0].grid(True)

    axs[0, 1].plot(T_array, M_avg_array, 'o-', color='red')
    axs[0, 1].set(title='Squared Magnetization vs. Temperature', xlabel='T', ylabel='<M^2>/N^2')
    axs[0, 1].grid(True)

    axs[1, 0].plot(T_array[:-1], Cv_array, 's-', color='green')
    axs[1, 0].set(title='Heat Capacity vs. Temperature', xlabel='T', ylabel='C_v')
    axs[1, 0].grid(True)

    r_values = np.arange(1, len(C_r_low) + 1)
    axs[1, 1].plot(r_values, C_r_low, 'o-', label=f'T={T_array[0]:.2f}', color='cyan')
    axs[1, 1].plot(r_values, C_r_high, 's-', label=f'T={T_array[-1]:.2f}', color='magenta')
    axs[1, 1].set(title='Spin-Spin Correlation Function', xlabel='Distance (r)', ylabel='C(r)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    finish_plot([(fig, "ThermodynamicObservables_VectorizedMetropolis_L64")])

def clean_correlation_array(c_array, threshold=1e-3):
    """Utility to cleanly cut off noisy exponential drops."""
    cutoff_indices = np.where(c_array < threshold)[0]
    if len(cutoff_indices) > 0:
        return cutoff_indices[0]
    return len(c_array)

def run_wolff_thermodynamics():
    """BONUS 2: Perfect Hybrid Thermodynamics. Features proper layouts and zero-noise curves."""
    L = 64
    N = L**2
    n_theta = 16
    J = 1.0
    numPoints = 20
    
    print(f"\n=== Starting PERFECT HYBRID Thermodynamics (L={L}) ===")
    start_time = time.time()
    
    lattice = init_spins(L, n_theta)
    print("Annealing to ground state...")
    for T_ann in np.linspace(2.0, 0.02, 10):
        lattice = VectorizedMetropolisXY(lattice, n_theta, 1.0/T_ann, J, sweeps=100)
        lattice = WolffXY(lattice, n_theta, 1.0/T_ann, J, num_clusters=10)
    lattice = VectorizedMetropolisXY(lattice, n_theta, 1.0/0.02, J, sweeps=1000)
    
    T_array = np.linspace(0.02, 2.0, numPoints)
    mid_index = 7  # T approx 0.75
    
    E_avg_list, M_avg_list, density_list = [], [], []
    C_r_low, C_r_mid, C_r_high = None, None, None
    lattice_low, lattice_mid, lattice_high = None, None, None
    
    print("Heating Loop running...")
    for i, T in enumerate(T_array):
        beta = 1.0 / T
        E_accum, M_accum, vort_accum = 0.0, 0.0, 0.0
        
        for _ in range(100):
            lattice = VectorizedMetropolisXY(lattice, n_theta, beta, J, sweeps=10)
            lattice = WolffXY(lattice, n_theta, beta, J, num_clusters=5)
            
            E_accum += EnergyXY(lattice, J)
            M_accum += MagXY(lattice)
            _, NumVort = VortXY(lattice)
            vort_accum += NumVort
            
        E_avg_list.append(E_accum / 100.0)
        M_avg_list.append(M_accum / 100.0)
        density_list.append((vort_accum / 100.0) / N)
        
        if i == 0:
            C_r_low = CorrXY(lattice)
            lattice_low = lattice.copy()
        elif i == mid_index:
            C_r_mid = CorrXY(lattice)
            lattice_mid = lattice.copy()
        elif i == numPoints - 1:
            C_r_high = CorrXY(lattice)
            lattice_high = lattice.copy()
            
    total_time = time.time() - start_time
    print(f"PERFECT HYBRID Thermodynamics completed in {total_time:.2f} seconds.")

    # ---------------------------------------------------------
    # Figure 1: Visualizations (2x3 Grid)
    # ---------------------------------------------------------
    print("Generating Visualizations...")
    vis_temps = [T_array[0], T_array[mid_index], T_array[-1]]
    vis_lattices = [lattice_low, lattice_mid, lattice_high]
    temp_labels = [f"T = {vis_temps[0]:.2f}", f"T = {vis_temps[1]:.2f}", f"T = {vis_temps[2]:.2f}"]

    fig_vis, axes_vis = plt.subplots(2, 3, figsize=(18, 11))
    for col, (temp_lat, temp_label) in enumerate(zip(vis_lattices, temp_labels)):
        V, NumVort = VortXY(temp_lat)
        VortPlotXY_ax(temp_lat, V, axes_vis[0, col], title=f"Spin Configuration, {temp_label}", show_vortices=False)
        im = VortPlotXY_ax(temp_lat, V, axes_vis[1, col], title=f"Vortex Positions, {temp_label} (Count = {NumVort:.0f})", show_vortices=True)
    fig_vis.colorbar(im, ax=axes_vis.ravel().tolist(), fraction=0.015, pad=0.04)

    # ---------------------------------------------------------
    # Figure 2: Thermodynamics 2x2 Grid (Restored labels)
    # ---------------------------------------------------------
    fig_th, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Energy
    axs[0, 0].plot(T_array, np.array(E_avg_list), 'o-', color='blue')
    axs[0, 0].set(title='Average Energy vs. Temperature', xlabel='T', ylabel='<E>/N')
    axs[0, 0].grid(True)

    # Magnetization
    axs[0, 1].plot(T_array, np.array(M_avg_list), 'o-', color='red')
    axs[0, 1].set(title='Squared Magnetization vs. Temperature', xlabel='T', ylabel='<M^2>/N^2')
    axs[0, 1].grid(True)

    # Heat Capacity
    axs[1, 0].plot(T_array[:-1], CvXY(np.array(E_avg_list), T_array), 's-', color='green')
    axs[1, 0].set(title='Heat Capacity vs. Temperature', xlabel='T', ylabel='C_v')
    axs[1, 0].grid(True)

    # Log-Log Correlation (Cleaned noise)
    r_vals = np.arange(1, len(C_r_low) + 1)

    axs[1, 1].loglog(r_vals, C_r_low, 'o-', label=f'T={T_array[0]:.2f}', color='cyan')
    axs[1, 1].loglog(r_vals, C_r_mid, '^-', label=f'T={T_array[mid_index]:.2f}', color='orange')

    # Clean exponential drop
    cutoff = clean_correlation_array(C_r_high, threshold=1e-3)
    axs[1, 1].loglog(r_vals[:cutoff], C_r_high[:cutoff], 's-', label=f'T={T_array[-1]:.2f} (Exp Drop)', color='magenta')

    # Fit line
    fit_r = r_vals[:15]
    fit_c = C_r_mid[:15]
    coeffs = np.polyfit(np.log(fit_r), np.log(fit_c), 1)
    fit_line = np.exp(coeffs[1]) * (fit_r ** coeffs[0])
    axs[1, 1].loglog(fit_r, fit_line, 'k--', linewidth=2, label=rf'Fit R <= 15 ($\eta$={-coeffs[0]:.3f})')

    axs[1, 1].set(title='Spin-Spin Correlation Function (Log-Log)', xlabel='Distance r (log)', ylabel='C(r) (log)')
    axs[1, 1].legend()
    axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()

    # ---------------------------------------------------------
    # Figure 3: Separate Vortex Density Graph
    # ---------------------------------------------------------
    fig_vort = plt.figure(figsize=(8, 6))
    density_arr = np.array(density_list)
    # Replace exactly 0 with NaN so the semilogy plot drops the line cleanly instead of failing
    density_clean = np.where(density_arr == 0, np.nan, density_arr)

    plt.semilogy(1.0/T_array, density_clean, 'o-', color='purple')
    plt.xlabel('Inverse Temperature (1 / k_B T)')
    plt.ylabel('Vortex Density (<NumVort> / N)')
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    finish_plot([
        (fig_vis, "SpinAndVortexConfigurations_WolffCluster_L64"),
        (fig_th, "ThermodynamicObservables_WolffCluster_L64"),
        (fig_vort, "VortexDensity_vs_InverseTemperature_WolffCluster_L64"),
    ])

def run_large_scale_demo():
    """BONUS 3: Large-Scale Lattice Thermodynamics & Visualization (L=256).
    Replicates the exact same layout as Bonus 2, but on a massive grid.
    """
    L = 256
    N = L**2
    n_theta = 16
    J = 1.0
    numPoints = 12  # Reduced to keep execution fast (~45 sec)
    
    print(f"\n=== Starting LARGE-SCALE Demo (L={L}x{L} = {N} spins) ===")
    start_time = time.time()
    
    lattice = init_spins(L, n_theta)
    for T_ann in np.linspace(2.0, 0.02, 5):
        lattice = VectorizedMetropolisXY(lattice, n_theta, 1.0/T_ann, J, sweeps=20)
        lattice = WolffXY(lattice, n_theta, 1.0/T_ann, J, num_clusters=3)
    
    T_array = np.linspace(0.02, 2.0, numPoints)
    mid_index = 4  # Approx T=0.74
    
    E_avg_list, M_avg_list, density_list = [], [], []
    C_r_low, C_r_mid, C_r_high = None, None, None
    lattice_low, lattice_mid, lattice_high = None, None, None
    
    print("Heating Loop running on 65,000+ spins...")
    for i, T in enumerate(T_array):
        beta = 1.0 / T
        E_accum, M_accum, vort_accum = 0.0, 0.0, 0.0
        
        # 20 measurements per temp to keep demo extremely fast
        for _ in range(20):
            lattice = VectorizedMetropolisXY(lattice, n_theta, beta, J, sweeps=5)
            lattice = WolffXY(lattice, n_theta, beta, J, num_clusters=2)
            
            E_accum += EnergyXY(lattice, J)
            M_accum += MagXY(lattice)
            _, NumVort = VortXY(lattice)
            vort_accum += NumVort
            
        E_avg_list.append(E_accum / 20.0)
        M_avg_list.append(M_accum / 20.0)
        density_list.append((vort_accum / 20.0) / N)
        
        if i == 0:
            C_r_low = CorrXY(lattice)
            lattice_low = lattice.copy()
        elif i == mid_index:
            C_r_mid = CorrXY(lattice)
            lattice_mid = lattice.copy()
        elif i == numPoints - 1:
            C_r_high = CorrXY(lattice)
            lattice_high = lattice.copy()
            
    total_time = time.time() - start_time
    print(f"LARGE-SCALE Demo completed in {total_time:.2f} seconds.")

    # ---------------------------------------------------------
    # Figure 1: Visualizations (2x3 Grid) - Arrows disabled for performance
    # ---------------------------------------------------------
    print("Generating Large Scale Visualizations...")
    vis_temps = [T_array[0], T_array[mid_index], T_array[-1]]
    vis_lattices = [lattice_low, lattice_mid, lattice_high]
    temp_labels = [f"T = {vis_temps[0]:.2f}", f"T = {vis_temps[1]:.2f}", f"T = {vis_temps[2]:.2f}"]

    fig_vis, axes_vis = plt.subplots(2, 3, figsize=(18, 11))
    for col, (temp_lat, temp_label) in enumerate(zip(vis_lattices, temp_labels)):
        V, NumVort = VortXY(temp_lat)
        VortPlotXY_ax(temp_lat, V, axes_vis[0, col], title=f"Spin Configuration, {temp_label}", show_vortices=False, show_arrows=False)
        im = VortPlotXY_ax(temp_lat, V, axes_vis[1, col], title=f"Vortex Positions, {temp_label} (Count = {NumVort:.0f})", show_vortices=True, show_arrows=False)
    fig_vis.colorbar(im, ax=axes_vis.ravel().tolist(), fraction=0.015, pad=0.04)

    # ---------------------------------------------------------
    # Figure 2: Thermodynamics 2x2 Grid
    # ---------------------------------------------------------
    fig_th, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(T_array, np.array(E_avg_list), 'o-', color='blue')
    axs[0, 0].set(title='Average Energy vs. Temperature', xlabel='T', ylabel='<E>/N')
    axs[0, 0].grid(True)

    axs[0, 1].plot(T_array, np.array(M_avg_list), 'o-', color='red')
    axs[0, 1].set(title='Squared Magnetization vs. Temperature', xlabel='T', ylabel='<M^2>/N^2')
    axs[0, 1].grid(True)

    axs[1, 0].plot(T_array[:-1], CvXY(np.array(E_avg_list), T_array), 's-', color='green')
    axs[1, 0].set(title='Heat Capacity vs. Temperature', xlabel='T', ylabel='C_v')
    axs[1, 0].grid(True)

    r_vals = np.arange(1, len(C_r_low) + 1)
    axs[1, 1].loglog(r_vals, C_r_low, 'o-', label=f'T={T_array[0]:.2f}', color='cyan')
    axs[1, 1].loglog(r_vals, C_r_mid, '^-', label=f'T={T_array[mid_index]:.2f}', color='orange')

    cutoff = clean_correlation_array(C_r_high, threshold=1e-3)
    axs[1, 1].loglog(r_vals[:cutoff], C_r_high[:cutoff], 's-', label=f'T={T_array[-1]:.2f} (Exp Drop)', color='magenta')

    axs[1, 1].set(title='Spin-Spin Correlation Function (Log-Log)', xlabel='Distance r (log)', ylabel='C(r) (log)')
    axs[1, 1].legend()
    axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    # ---------------------------------------------------------
    # Figure 3: Separate Vortex Density Graph
    # ---------------------------------------------------------
    fig_vort = plt.figure(figsize=(8, 6))
    density_arr = np.array(density_list)
    density_clean = np.where(density_arr == 0, np.nan, density_arr)

    plt.semilogy(1.0/T_array, density_clean, 'o-', color='purple')
    plt.xlabel('Inverse Temperature (1 / k_B T)')
    plt.ylabel('Vortex Density (<NumVort> / N)')
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    finish_plot([
        (fig_vis, "SpinAndVortexConfigurations_WolffCluster_L256"),
        (fig_th, "ThermodynamicObservables_WolffCluster_L256"),
        (fig_vort, "VortexDensity_vs_InverseTemperature_WolffCluster_L256"),
    ])

def run_all_tasks():
    """Runs options 1, 2, 4, 5, 6, 7, 8 sequentially (option 3 is skipped since it
    just re-runs 1 and 2). Plots are saved to PLOTS_DIR instead of being displayed."""
    global SAVE_PLOTS
    SAVE_PLOTS = True
    os.makedirs(PLOTS_DIR, exist_ok=True)
    try:
        simulate_thermal_states()
        run_thermodynamic_simulation()
        visualize_vortices()
        simulate_vortex_density()
        run_fast_thermodynamics()
        run_wolff_thermodynamics()
        run_large_scale_demo()
    finally:
        SAVE_PLOTS = False

def main():
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
    print("8: Run BONUS 3 (Large Scale 256x256 Demo)")
    print("9: Run All (1,2,4,5,6,7,8 - saves plots to PLOTS/ folder)")
    print("0: Exit")
    print("=========================================")

    choice = input("Enter choice (0/1/2/3/4/5/6/7/8/9): ")

    if choice == '1': simulate_thermal_states()
    elif choice == '2': run_thermodynamic_simulation()
    elif choice == '3': simulate_thermal_states(); run_thermodynamic_simulation()
    elif choice == '4': visualize_vortices()
    elif choice == '5': simulate_vortex_density()
    elif choice == '6': run_fast_thermodynamics()
    elif choice == '7': run_wolff_thermodynamics()
    elif choice == '8': run_large_scale_demo()
    elif choice == '9': run_all_tasks()
    elif choice == '0': sys.exit()
    else: print("Invalid choice.")

if __name__ == "__main__":
    main()