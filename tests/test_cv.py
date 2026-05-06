import numpy as np
from xy_model import CvXY


def run_cv_test():
    T = np.array([0.1, 0.2, 0.3])
    E = np.array([-2.0, -1.5, -0.5])

    cv_values = CvXY(E, T)
    expected = np.array([5.0, 10.0])
    is_correct = np.allclose(cv_values, expected)

    if is_correct:
        print("Validation Result: SUCCESS. CvXY output matches expected values.")
        return True
    else:
        print("Validation Result: FAILURE. CvXY output does not match expected values.")
        return False


if __name__ == "__main__":
    run_cv_test()
