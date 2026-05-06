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

def CvXY(Energy, Temperature):
    """
    Calculates the heat capacity (Cv) as the numerical derivative of energy 
    with respect to temperature. Corresponds to Task 2.3.
    
    Args:
        Energy (np.ndarray or list): Array of average energies of length numPoints.
        Temperature (np.ndarray or list): Array of temperatures of length numPoints.
        
    Returns:
        np.ndarray: Array of heat capacities of length (numPoints - 1).
    """
    # Convert inputs to numpy arrays to ensure vectorized operations
    E_array = np.array(Energy)
    T_array = np.array(Temperature)
    
    # Calculate the discrete differences: dE = E_{i+1} - E_i and dT = T_{i+1} - T_i
    dE = np.diff(E_array)
    dT = np.diff(T_array)
    
    # Heat capacity is the ratio of the differences (the numerical derivative)
    Cv = dE / dT
    
    return Cv


def CorrXY(S):
    """
    Calculates the spatial correlation function C(r) for distances r = 1 to L/2.
    Corresponds to Task 2.4.
    
    Args:
        S (np.ndarray): The L x L matrix containing spin angles.
        
    Returns:
        np.ndarray: A 1D array of length L/2 containing the correlation C(r) 
                    for each distance r.
    """
    L = S.shape[0]
    max_r = L // 2
    
    # Initialize the correlation array
    C_r = np.zeros(max_r)
    
    # Loop over all possible distances from 1 to L/2
    for r in range(1, max_r + 1):
        # Shift the lattice by distance r along the x-axis and y-axis
        shifted_x = np.roll(S, shift=r, axis=1)
        shifted_y = np.roll(S, shift=r, axis=0)
        
        # Calculate the correlation (cosine of the angle difference)
        corr_x = np.cos(S - shifted_x)
        corr_y = np.cos(S - shifted_y)
        
        # Calculate the mean correlation across all N spins for both directions
        # and average them to get a generalized isotropic correlation for distance r.
        C_r[r - 1] = (np.mean(corr_x) + np.mean(corr_y)) / 2.0
        
    return C_r

def VortPlotXY(S, V, title="Vortices"):
    """
    Plots the spin configuration and the corresponding vortices.
    Corresponds to Task 3.2.
    
    Args:
        S (np.ndarray): Spin configurations.
        V (np.ndarray): Vorticity matrix.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 8))
    
    # 1. Plot the background vorticity map
    # We use the 'coolwarm' colormap: 0 is neutral, positive is red, negative is blue.
    plt.imshow(V, cmap='coolwarm', vmin=-2*np.pi, vmax=2*np.pi, origin='lower')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Vorticity')
    
    # 2. Plot the spin arrows
    L = S.shape[0]
    X, Y = np.meshgrid(np.arange(L), np.arange(L))
    U = np.cos(S)
    W = np.sin(S)
    
    # 'quiver' plots the arrows. We use 'mid' pivot to center them on the lattice points.
    plt.quiver(X, Y, U, W, color='black', pivot='mid', scale=L*1.2)
    
    plt.title(title, fontsize=14)
    plt.xticks([])
    plt.yticks([])


def wrap_angle(d_theta):
    """
    Wraps an angle difference to the range [-pi, pi].
    This ensures that the difference between 359 degrees and 1 degree
    is considered as -2 degrees, not 358 degrees.
    """
    return (d_theta + np.pi) % (2 * np.pi) - np.pi

def VortXY(S):
    """
    Identifies vortices in the XY model configuration.
    Corresponds to Task 3.1.
    
    Args:
        S (np.ndarray): The L x L matrix containing spin angles.
        
    Returns:
        tuple: (V, NumVort)
            - V (np.ndarray): L x L matrix of vorticities.
            - NumVort (float): Total number of vortices in the lattice.
    """
    # 1. Define the 4 corners of each plaquette using vectorized shifts.
    # We treat S[i,j] as the Bottom-Left (BL) corner.
    BL = S
    BR = np.roll(S, shift=-1, axis=1)          # Bottom-Right (shifted left)
    TR = np.roll(BR, shift=-1, axis=0)         # Top-Right (shifted up from BR)
    TL = np.roll(S, shift=-1, axis=0)          # Top-Left (shifted up from BL)
    
    # 2. Calculate the phase differences along a counter-clockwise path:
    # BL -> BR -> TR -> TL -> BL
    d1 = BR - BL  # Bottom edge (rightwards)
    d2 = TR - BR  # Right edge (upwards)
    d3 = TL - TR  # Top edge (leftwards)
    d4 = BL - TL  # Left edge (downwards)
    
    # 3. Wrap all differences to the [-pi, pi] interval
    d1_wrapped = wrap_angle(d1)
    d2_wrapped = wrap_angle(d2)
    d3_wrapped = wrap_angle(d3)
    d4_wrapped = wrap_angle(d4)
    
    # 4. Sum the wrapped differences to get the vorticity of each plaquette
    V = d1_wrapped + d2_wrapped + d3_wrapped + d4_wrapped
    
    # Clean up microscopic floating-point errors (e.g., setting 1e-15 to 0.0)
    V[np.abs(V) < 1e-5] = 0.0
    
    # 5. Calculate the total number of vortices.
    # Note: Each vortex contributes exactly 2*pi or -2*pi to V.
    # The document mentions dividing by 2, which is likely a typo for 2*pi,
    # as mathematically the sum of |V| yields 2*pi per vortex.
    NumVort = np.sum(np.abs(V)) / (2 * np.pi)
    
    return V, NumVort