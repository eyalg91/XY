from tests.test_initialization import run_initialization_test

def main():
    """
    Main execution script for the XY Model simulation project.
    
    Currently configured to execute system validation tests before 
    proceeding with Monte Carlo iterations.
    """
    # System parameters for testing
    L_SIZE = 64
    N_DISCRETE_ANGLES = 16
    
    # Trigger the initialization test suite
    initialization_ok = run_initialization_test(L=L_SIZE, n_theta=N_DISCRETE_ANGLES)
    
    if initialization_ok:
        print("\nSystem verified. Ready to proceed with physical simulations.")
    else:
        print("\nSystem error detected during initialization. Aborting execution.")

if __name__ == "__main__":
    main()