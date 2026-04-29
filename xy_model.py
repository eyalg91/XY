import numpy as np

def init_spins(L, n_theta):
    """
    Initializes a 2D lattice of spins with discrete orientations.
    
    Each spin in the L x L lattice is assigned a random angle from a set 
    of n_theta discrete values uniformly distributed between 2*pi/n_theta and 2*pi.
    This corresponds to Task 1.1 of the XY model project.

    Args:
        L (int): The number of spins along one dimension (total spins N = L^2).
        n_theta (int): The number of allowed discrete angular orientations.

    Returns:
        np.ndarray: An L x L matrix where each element represents a spin angle.
    """
    # Generate an L x L grid of random integers k in the range [1, n_theta].
    # These integers serve as indices for the discrete set of allowed angles.
    random_indices = np.random.randint(1, n_theta + 1, size=(L, L))
    
    # Map the indices to the corresponding angles: theta = (2*pi / n_theta) * k.
    # This vectorized approach ensures computational efficiency.
    spin_matrix = (2 * np.pi / n_theta) * random_indices
    
    return spin_matrix