import matplotlib.pyplot as plt
from xy_model import init_spins, PlotXY

def run_plot_test(L=20, n_theta=16):
    """
    Validates the PlotXY visualization function.
    Generates a random lattice and displays it. The user must manually close 
    the plot window for the test to complete.
    """
    print(f"--- Running Plotting Validation (L={L}) ---")
    print("A plot window should open. Close it to continue.")
    
    # Generate an initial random state
    lattice = init_spins(L, n_theta)
    
    # Generate the plot
    fig = PlotXY(lattice, title="Test Plot: Initial Random State")
    
    # Display the plot
    plt.show()
    
    print("Validation Result: Plotting executed successfully.\n")
    return True