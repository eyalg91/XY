import numpy as np
import matplotlib.pyplot as plt

def init_spins(L, n_theta):
    """Initializes a 2D lattice of spins with discrete orientations."""
    random_indices = np.random.randint(1, n_theta + 1, size=(L, L))
    spin_matrix = (2 * np.pi / n_theta) * random_indices
    return spin_matrix

def MetropolisXY(S, n_theta, beta, J, numIters):
    """Performs the standard local Metropolis Monte Carlo algorithm."""
    L = S.shape[0]
    S_new = S.copy()
    allowed_angles = (2 * np.pi / n_theta) * np.arange(1, n_theta + 1)
    
    for _ in range(numIters):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        theta_old = S_new[i, j]
        theta_new = np.random.choice(allowed_angles)
        
        top = S_new[(i - 1) % L, j]
        bottom = S_new[(i + 1) % L, j]
        left = S_new[i, (j - 1) % L]
        right = S_new[i, (j + 1) % L]
        
        E_old = -J * (np.cos(theta_old - top) + np.cos(theta_old - bottom) + 
                      np.cos(theta_old - left) + np.cos(theta_old - right))
        E_new = -J * (np.cos(theta_new - top) + np.cos(theta_new - bottom) + 
                      np.cos(theta_new - left) + np.cos(theta_new - right))
        dE = E_new - E_old
        
        if dE <= 0:
            S_new[i, j] = theta_new
        else:
            prob = np.exp(-beta * dE)
            if np.random.rand() < prob:
                S_new[i, j] = theta_new
    return S_new

def PlotXY(S, title="XY Model Configuration"):
    """Visualizes the spin lattice with colors and arrows."""
    L = S.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(S, cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower')
    X, Y = np.meshgrid(np.arange(L), np.arange(L))
    U = np.cos(S)
    V = np.sin(S)
    ax.quiver(X, Y, U, V, color='black', pivot='mid', angles='xy', scale_units='xy', scale=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spin Angle (Radians)')
    ax.set_xticks([])
    ax.set_yticks([])
    # No on-figure title: single-panel figures are captioned externally (e.g. in a report).
    return fig

def EnergyXY(S, J):
    """Calculates the average energy per spin."""
    N = S.size
    right_neighbors = np.roll(S, shift=-1, axis=1)
    bottom_neighbors = np.roll(S, shift=-1, axis=0)
    energy_matrix = -J * (np.cos(S - right_neighbors) + np.cos(S - bottom_neighbors))
    avg_energy = np.sum(energy_matrix) / N
    return avg_energy

def MagXY(S):
    """Calculates the normalized squared magnetization per spin."""
    N = S.size
    sum_cos = np.sum(np.cos(S))
    sum_sin = np.sum(np.sin(S))
    mag_squared = (sum_cos**2 + sum_sin**2) / (N**2)
    return mag_squared

def CvXY(Energy, Temperature):
    """Calculates the heat capacity (Cv)."""
    E_array = np.array(Energy)
    T_array = np.array(Temperature)
    dE = np.diff(E_array)
    dT = np.diff(T_array)
    Cv = dE / dT
    return Cv

def CorrXY(S):
    """Calculates the spatial correlation function C(r)."""
    L = S.shape[0]
    max_r = L // 2
    C_r = np.zeros(max_r)
    for r in range(1, max_r + 1):
        shifted_x = np.roll(S, shift=r, axis=1)
        shifted_y = np.roll(S, shift=r, axis=0)
        corr_x = np.cos(S - shifted_x)
        corr_y = np.cos(S - shifted_y)
        C_r[r - 1] = (np.mean(corr_x) + np.mean(corr_y)) / 2.0
    return C_r

def wrap_angle(d_theta):
    """Wraps an angle difference to the range [-pi, pi]."""
    return (d_theta + np.pi) % (2 * np.pi) - np.pi

def VortXY(S):
    """Identifies vortices and returns the vorticity matrix and total count."""
    BL = S
    BR = np.roll(S, shift=-1, axis=1)
    TR = np.roll(BR, shift=-1, axis=0)
    TL = np.roll(S, shift=-1, axis=0)
    
    d1 = wrap_angle(BR - BL)
    d2 = wrap_angle(TR - BR)
    d3 = wrap_angle(TL - TR)
    d4 = wrap_angle(BL - TL)
    
    V = d1 + d2 + d3 + d4
    V[np.abs(V) < 1e-5] = 0.0
    NumVort = np.sum(np.abs(V)) / (2 * np.pi)
    return V, NumVort

def VortPlotXY(S, V, title="Vortices"):
    """Original task plot for vortices."""
    plt.figure(figsize=(8, 8))
    L = S.shape[0]
    plt.imshow(V, cmap='coolwarm', vmin=-2*np.pi, vmax=2*np.pi, origin='lower', extent=[0, L, 0, L])
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Vorticity')
    X, Y = np.meshgrid(np.arange(L), np.arange(L))
    U = np.cos(S)
    W = np.sin(S)
    plt.quiver(X, Y, U, W, color='black', pivot='mid', scale=L*1.2)
    plt.title(title, fontsize=14)
    plt.xlim(-0.5, L-0.5)
    plt.ylim(-0.5, L-0.5)
    plt.xticks([])
    plt.yticks([])

def VortPlotXY_ax(S, V, ax, title="Vortices", show_vortices=True, show_arrows=True):
    """Flexible plotting on a specific matplotlib axis."""
    L = S.shape[0]
    im = ax.imshow(S, cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower')
    
    if show_arrows:
        # Scale arrows appropriately for large grids to prevent a black screen
        step = max(1, L // 64)
        X, Y = np.meshgrid(np.arange(0, L, step), np.arange(0, L, step))
        U, W = np.cos(S[::step, ::step]), np.sin(S[::step, ::step])
        ax.quiver(X, Y, U, W, color='black', pivot='mid', angles='xy', scale_units='xy', scale=1)
    
    if show_vortices:
        v_y, v_x = np.where(V > 0.1)
        ax.scatter(v_x + 0.5, v_y + 0.5, s=120, facecolors='none', edgecolors='white', linewidths=2)
        av_y, av_x = np.where(V < -0.1)
        ax.scatter(av_x + 0.5, av_y + 0.5, s=120, facecolors='none', edgecolors='black', linewidths=2)
        
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def VectorizedMetropolisXY(S, n_theta, beta, J, sweeps):
    """High-Performance Vectorized Metropolis (Checkerboard)."""
    L = S.shape[0]
    black_mask = ((np.arange(L)[:, None] + np.arange(L)) % 2 == 0)
    red_mask = ~black_mask
    S_current = S.copy()
    
    for _ in range(sweeps):
        for mask in [black_mask, red_mask]:
            random_indices = np.random.randint(0, n_theta, size=(L, L))
            S_new = random_indices * (2 * np.pi / n_theta)
            
            up = np.roll(S_current, shift=1, axis=0)
            down = np.roll(S_current, shift=-1, axis=0)
            left = np.roll(S_current, shift=1, axis=1)
            right = np.roll(S_current, shift=-1, axis=1)
            
            E_new = -J * (np.cos(S_new - up) + np.cos(S_new - down) + 
                          np.cos(S_new - left) + np.cos(S_new - right))
            E_old = -J * (np.cos(S_current - up) + np.cos(S_current - down) + 
                          np.cos(S_current - left) + np.cos(S_current - right))
            dE = E_new - E_old
            
            accept = np.random.rand(L, L) < np.exp(-beta * dE)
            update_condition = accept & mask
            S_current = np.where(update_condition, S_new, S_current)
    return S_current

def WolffXY(S, n_theta, beta, J, num_clusters):
    """Discrete Wolff Cluster Algorithm."""
    L = S.shape[0]
    S_new = S.copy()
    
    for _ in range(num_clusters):
        k = np.random.randint(0, 2 * n_theta)
        psi = k * (np.pi / n_theta)
        
        i, j = np.random.randint(0, L, size=2)
        in_cluster = np.zeros((L, L), dtype=bool)
        in_cluster[i, j] = True
        stack = [(i, j)]
        
        while stack:
            curr_i, curr_j = stack.pop()
            proj_curr = np.cos(S_new[curr_i, curr_j] - psi)
            
            neighbors = [
                ((curr_i + 1) % L, curr_j), ((curr_i - 1) % L, curr_j),
                (curr_i, (curr_j + 1) % L), (curr_i, (curr_j - 1) % L)
            ]
            
            for ni, nj in neighbors:
                if not in_cluster[ni, nj]:
                    proj_neighbor = np.cos(S_new[ni, nj] - psi)
                    dot_product = proj_curr * proj_neighbor
                    
                    if dot_product > 0:
                        p_add = 1.0 - np.exp(-2.0 * beta * J * dot_product)
                        if np.random.rand() < p_add:
                            in_cluster[ni, nj] = True
                            stack.append((ni, nj))
                            
        S_new[in_cluster] = 2 * psi + np.pi - S_new[in_cluster]
        
    S_new = np.round(S_new / (2 * np.pi / n_theta)) * (2 * np.pi / n_theta)
    return S_new % (2 * np.pi)