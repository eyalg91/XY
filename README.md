# XY Model Simulation

A Python simulation of the 2D classical XY model, a statistical-physics spin model used to
study the Berezinskii-Kosterlitz-Thouless (BKT) phase transition. Spins on a square lattice
are represented by discrete planar angles, and the simulation explores how the system's
order, thermodynamics, and topological defects (vortices) evolve with temperature.

## What it does

- **Lattice initialization** — creates an `L x L` grid of spins with `n_theta` allowed
  discrete orientations.
- **Monte Carlo sampling** — three interchangeable update algorithms:
  - `MetropolisXY`: standard single-spin-flip Metropolis algorithm.
  - `VectorizedMetropolisXY`: checkerboard-vectorized Metropolis for much faster sweeps.
  - `WolffXY`: cluster-flip Wolff algorithm to reduce critical slowing down near the
    transition temperature.
- **Thermodynamic observables** — average energy, squared magnetization, heat capacity,
  and the spatial spin-spin correlation function `C(r)`.
- **Vortex analysis** — detects topological vortices/antivortices from the lattice
  configuration and tracks vortex density as a function of temperature, characteristic of
  the BKT transition.
- **Visualization** — plots spin configurations (as color/arrow fields), vortex maps, and
  thermodynamic curves (energy, magnetization, heat capacity, correlation, vortex density)
  using Matplotlib.

## Project structure

- `xy_model.py` — core simulation functions (lattice setup, Monte Carlo algorithms,
  observables, and plotting helpers).
- `main.py` — interactive menu that runs the various simulation tasks (phase-extreme
  visualization, full thermodynamics, vortex visualization/density, and faster
  vectorized/Wolff/large-scale variants).
- `tests/` — unit tests covering initialization, Metropolis updates, correlation, heat
  capacity, thermodynamics, and plotting.

## Setup

1. Create and activate a virtual environment (if needed):
   - Windows PowerShell: `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`

## Run

`python main.py`

This opens an interactive menu where you can choose which simulation task to run, e.g.
visualizing high/low temperature states, computing thermodynamic quantities across a
temperature range, visualizing vortices, or running faster large-scale simulations.
