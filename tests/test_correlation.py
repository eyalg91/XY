import numpy as np
from xy_model import CorrXY


def run_correlation_test():
    """
    Validates CorrXY using a fully aligned ground-state lattice.
    """
    L = 10
    ground_state = np.zeros((L, L))

    C_r = CorrXY(ground_state)

    correct_length = len(C_r) == (L // 2)
    all_ones = np.allclose(C_r, 1.0)

    if correct_length and all_ones:
        print("Validation Result: SUCCESS. CorrXY matches expected ground-state behavior.")
        return True
    else:
        print("Validation Result: FAILURE. CorrXY does not match expected ground-state behavior.")
        return False


if __name__ == "__main__":
    run_correlation_test()
