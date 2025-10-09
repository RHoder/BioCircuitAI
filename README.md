BioCircuitAI/
├─ README.md
├─ requirements.txt              # or pyproject.toml if you prefer
├─ src/
│  └─ biocircuitai/
│     ├─ __init__.py
│     ├─ config.py               # global hyperparams, paths, seeds
│     ├─ simulate/
│     │  ├─ __init__.py
│     │  ├─ models.py            # ODEs: toggle switch, repressilator
│     │  ├─ ode_solver.py        # wraps scipy.integrate.solve_ivp
│     │  └─ experiments.py       # param sweeps → dataset.csv
│     ├─ surrogate/
│     │  ├─ __init__.py
│     │  ├─ dataset.py           # load csv → tensors, splits
│     │  ├─ model.py             # small MLP/Transformer-ish surrogate
│     │  ├─ train.py             # training loop + early stopping
│     │  └─ evaluate.py          # metrics + calibration plots
│     ├─ control/
│     │  ├─ __init__.py
│     │  ├─ controllers.py       # PID / simple state feedback (optional)
│     │  ├─ optimizer.py         # Bayesian opt / RL agent
│     │  └─ scheduler.py         # asyncio DBTL orchestrator
│     ├─ io/
│     │  ├─ logging.py           # structured logs
│     │  ├─ storage.py           # SQLite/CSV artifact registry
│     │  └─ plots.py             # matplotlib/plotly figures
│     └─ app/
│        ├─ __init__.py
│        └─ dashboard.py         # Streamlit UI for demos
├─ notebooks/
│  ├─ 01_simulation_exploration.ipynb
│  └─ 02_surrogate_training.ipynb
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ models/
├─ scripts/
│  ├─ run_simulation.py          # CLI to generate dataset
│  ├─ train_surrogate.py
│  └─ optimize_circuit.py
└─ tests/
   ├─ test_ode.py
   ├─ test_surrogate.py
   └─ test_control.py
