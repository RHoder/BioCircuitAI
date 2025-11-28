**BioCircuitAI**

- Surrogate-driven design of simple genetic circuits with simulation, learning, and optimization in one repo. Targeted at fast iteration on design parameters for phenotypic goals (e.g., target steady-state expression levels).

**Run it yourself:**
```bash
python scripts/run_dbtl.py --workers 6 --trials 40 --targets "10:8:default,1:8:flip"

**What It Does**
- Simulates canonical circuits (toggle switch; repressilator stub) with ODEs to generate data.
- Trains a PyTorch MLP surrogate to learn mapping from circuit parameters → phenotype summaries.
- Uses Bayesian optimization on the surrogate to propose parameters that achieve target phenotypes.
- Validates proposals with the true ODE model; logs artifacts and plots for review.
- Orchestrates a DBTL-style loop (Design/Build/Test/Learn) with simple registry logging.

**Why It Matters**
- Reduces expensive search over wet-lab relevant parameter spaces by learning a fast differentiable surrogate.
- Demonstrates end-to-end applied modeling: dynamical systems, ML, BO, and reproducible tooling.
- Extensible foundation for richer biophysical models, constraints, and multi-objective design.

**Highlights**
- End-to-end pipeline: simulation → dataset → training → BO → ODE validation → artifacts.
- Normalized training with log transforms for rate/affinity-like parameters and standardization.
- Windows-safe parallel sweeps; deterministic seeding; lightweight tests.
- Clear configuration and CLI scripts for quick evaluation by reviewers.

**Project Structure**
- `src/biocircuitai/simulate/` — ODE models and sweeps
  - `models.py` (toggle switch ODE), `ode_solver.py` (solve_ivp wrapper), `experiments.py` (grid sweeps)
- `src/biocircuitai/surrogate/` — data, model, training, evaluation
  - `dataset.py`, `preprocess.py`, `model.py`, `train.py`, `evaluate.py`
- `src/biocircuitai/control/` — optimization and DBTL orchestration
  - `optimizer.py` (surrogate wrapper, BO, validation, plots), `multitarget.py`, `scheduler.py`
- `src/biocircuitai/io/` — logging and run registry (`logging.py`, `storage.py`)
- `src/biocircuitai/config.py` — global config: grids, bounds, targets, paths
- `scripts/` — CLI entry points for each stage
- `tests/` — minimal tests for ODE, training, and BO
- `notebooks/` — exploratory analysis and training diagnostics (optional)

**Quick Start**
- Python 3.10+ recommended. Create an environment and install deps:
  - `python -m venv .venv && .venv\Scripts\activate`
  - `pip install -r requirements.txt`

Full DBTL-style loop in one command (sweep → train → multi-target BO → validate):
- `python scripts/run_dbtl.py --workers 1 --epochs 60 --bs 512 --trials 80 --targets "10:8:default"`

Generate a dataset (toggle-switch grid sweep):
- `python scripts/run_simulation.py --workers 1 --chunksize 1000 --out data/processed/toggle_dataset.csv`

Train a surrogate (normalized MLP):
- `python scripts/train_surrogate.py --csv data/processed/toggle_dataset.csv --epochs 60 --bs 512 --lr 1e-3 --hidden 256,256 --out data/models/surrogate.pt`

Optimize to a target phenotype and validate with the true ODE:
- `python scripts/optimize_circuit.py --model data/models/surrogate.pt --targetA 10 --targetB 8 --trials 80 --outdir data/opt/`

Multi-target batch optimization (writes per-target folders and a summary CSV):
- `python scripts/run_multitarget.py --model data/models/surrogate.pt --trials 80 --outdir data/multitarget/ --targets "10:8:A10_B8, 1:8:A1_B8, 60:20:A60_B20"`

Run tests (assumes a dataset and a trained model exist at default paths):
- `pytest -q`

**Configuration**
- Central settings live in `src/biocircuitai/config.py`:
  - Simulation defaults and parameter grid (`CONFIG.defaults`, `CONFIG.grid`).
  - BO bounds and default target phenotype (`CONFIG.bounds`, `CONFIG.target`).
  - Data/model/log directories and global seed.
- Edit here to change grids, bounds, or target A/B without touching scripts.

**Design Details**
- Models: Deterministic ODEs via `scipy.integrate.solve_ivp`; toggle switch (mutual repression) implemented; repressilator stub provided for extension.
- Features/Targets: Inputs are 8 physical-ish parameters (`alpha_A/B`, `K_A/B`, `n_A/B`, `dA/dB`). Targets are steady-state summaries (`steady_A`, `steady_B`).
- Preprocessing: Log1p transform for rate/affinity-like inputs (`K_*`, `d*`) plus `StandardScaler` on X and Y for stable training.
- Surrogate: Compact MLP in PyTorch with configurable hidden sizes and dropout.
- Optimization: Optuna TPE minimizes `|A−tA| + |B−tB|` with light L2 on parameters; designs validated on the true ODE and plotted.
- Orchestration: Simple run registry at `data/registry.csv` captures phase, status, and artifact paths for traceability.

**Outputs and Artifacts**
- Datasets: `data/processed/toggle_dataset.csv` (or per-run under `data/dbtl/<run_id>/`).
- Models/Scalers: `data/models/surrogate.pt`, `data/models/scalers.joblib` (or per-run equivalents).
- Optimization: JSON of best params, convergence plot, and true ODE trajectory in the chosen `--outdir`.
- Logs: Timestamped text logs in `logs/` and a CSV registry in `data/registry.csv`.

**What to Look For (Reviewer Guide)**
- Clean separation of concerns (simulate/surrogate/control/io) with minimal surface area.
- Reproducible runs and deterministic seeds; Windows-safe parallel sweeps.
- Extensible hooks: add new ODEs in `simulate/models.py`; expand targets or objectives in `control/optimizer.py`.
- Tests covering ODE integration, training loop viability, and BO execution.

**Extending the Project**
- Add richer phenotypes: include dynamics (rise time, overshoot) in `simulate/ode_solver.py` summaries and `TARGET_COLS`.
- Multi-objective design: switch to Pareto-based BO or scalarization in `control/optimizer.py`.
- Constraints: add feasibility/penalty terms, parameter tying, or library constraints.
- Alternative surrogates: gradient-boosted trees, GPs for uncertainty, ensembles.
- Noisy/stoichiometric models: SSA or SDEs as additional backends.

**Assumptions & Limitations**
- Current surrogate targets steady-state means for a simple toggle-switch; no measurement noise or cell-to-cell variation.
- Bounds and grids are illustrative; tune for your biology and units.

**Acknowledgements**
- Built to showcase applied modeling and ML for genetic circuit design. Inspired by classic toggle-switch/repressilator literature and standard BO workflows.