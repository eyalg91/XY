import numpy as np
import matplotlib.pyplot as plt

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


def MetropolisXY(S, n_theta, beta, J, numIters):
    """
    Performs the Metropolis Monte Carlo algorithm to update the spin lattice.
    
    Args:
        S (np.ndarray): The initial L x L spin configuration.
        n_theta (int): The number of allowed discrete angles.
        beta (float): Inverse temperature (1 / k_B T).
        J (float): Interaction strength between nearest neighbors.
        numIters (int): Number of Monte Carlo steps to perform.
        
    Returns:
        np.ndarray: The updated spin configuration after numIters steps.
    """
    L = S.shape[0]
    S_new = S.copy()  # Create a copy to avoid modifying the original array directly
    
    # Pre-calculate the set of all allowed discrete angles
    allowed_angles = (2 * np.pi / n_theta) * np.arange(1, n_theta + 1)
    
    for _ in range(numIters):
        # 1. Select a random spin
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        # 2. Propose a new valid orientation
        theta_old = S_new[i, j]
        theta_new = np.random.choice(allowed_angles)
        
        # 3. Identify nearest neighbors using periodic boundary conditions
        top = S_new[(i - 1) % L, j]
        bottom = S_new[(i + 1) % L, j]
        left = S_new[i, (j - 1) % L]
        right = S_new[i, (j + 1) % L]
        
        # 4. Calculate energy change (Delta E)
        E_old = -J * (np.cos(theta_old - top) + np.cos(theta_old - bottom) + 
                      np.cos(theta_old - left) + np.cos(theta_old - right))
        E_new = -J * (np.cos(theta_new - top) + np.cos(theta_new - bottom) + 
                      np.cos(theta_new - left) + np.cos(theta_new - right))
        dE = E_new - E_old
        
        # 5 & 6. Accept or reject the proposed state
        if dE <= 0:
            S_new[i, j] = theta_new
        else:
            prob = np.exp(-beta * dE)
            if np.random.rand() < prob:
                S_new[i, j] = theta_new
                
    return S_new



def PlotXY(S, title="XY Model Configuration"):
    """
    Visualizes the spin lattice with colors and arrows.
    
    Args:
        S (np.ndarray): The L x L matrix containing spin angles.
        title (str): Title for the plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    L = S.shape[0]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # 1. Background colors: use 'hsv' colormap because angles are cyclic (0 and 2*pi are the same)
    # origin='lower' ensures the origin is at the bottom-left, matching standard Cartesian coordinates
    im = ax.imshow(S, cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower')
    
    # 2. Arrows (Quiver plot): prepare grid and vector components
    X, Y = np.meshgrid(np.arange(L), np.arange(L))
    U = np.cos(S)
    V = np.sin(S)
    
    # Draw the arrows. pivot='mid' centers the arrow on the pixel.
    ax.quiver(X, Y, U, V, color='black', pivot='mid', angles='xy', scale_units='xy', scale=1)
    
    # 3. Add a colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spin Angle (Radians)')
    
    # 4. Remove axis ticks as requested
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(title)
    
    return fig


def EnergyXY(S, J):
    """
    Calculates the average energy per spin of the current lattice configuration.
    Corresponds to Task 2.1.
    
    Args:
        S (np.ndarray): The L x L matrix containing spin angles.
        J (float): Interaction strength between nearest neighbors.
        
    Returns:
        float: The average energy per spin <E>.
    """
    N = S.size  # Total number of spins (L^2)
    
    # Use np.roll to efficiently get neighbors without loops.
    # shift=-1 on axis=1 shifts the matrix left, meaning we align each spin with its right neighbor.
    # shift=-1 on axis=0 shifts the matrix up, aligning each spin with its bottom neighbor.
    right_neighbors = np.roll(S, shift=-1, axis=1)
    bottom_neighbors = np.roll(S, shift=-1, axis=0)
    
    # Calculate the interaction energy for these two directions.
    # This covers all bonds in the lattice exactly once.
    energy_matrix = -J * (np.cos(S - right_neighbors) + np.cos(S - bottom_neighbors))
    
    # Calculate the total Hamiltonian and divide by N for the average per spin
    avg_energy = np.sum(energy_matrix) / N
    
    return avg_energy

def MagXY(S):
    """
    Calculates the normalized squared magnetization per spin.
    Corresponds to Task 2.2.
    
    Args:
        S (np.ndarray): The L x L matrix containing spin angles.
        
    Returns:
        float: The squared magnetization per spin <M^2> / N^2.
    """
    N = S.size
    
    # Sum the x (cosine) and y (sine) components of all spins
    sum_cos = np.sum(np.cos(S))
    sum_sin = np.sum(np.sin(S))
    
    # Calculate <M^2> / N^2 according to the formula
    mag_squared = (sum_cos**2 + sum_sin**2) / (N**2)
    
    return mag_squared