# src/biocircuitai/control/scheduler.py
from __future__ import annotations
import asyncio, os, json, uuid
from dataclasses import dataclass
from typing import Dict, Tuple

from ..config import CONFIG
from ..io.logging import get_logger
from ..io.storage import RunRecord, register
from ..simulate.experiments import default_sweep
from ..surrogate.train import TrainConfig, train_surrogate
from ..surrogate.evaluate import load_model
from ..surrogate.dataset import PARAM_COLS
from .optimizer import (
    BOConfig, SurrogateWrapper, bo_optimize, validate_on_ode,
    plot_convergence, plot_timeseries
)

log = get_logger("biocircuitai.scheduler")

@dataclass
class DBTLConfig:
    outdir: str = "data/dbtl/"
    csv_name: str = "toggle_dataset.csv"
    model_name: str = "surrogate.pt"
    trials: int = 80
    targetA: float = CONFIG.target["A"]
    targetB: float = CONFIG.target["B"]
    hidden: Tuple[int, ...] | None = None   # None = infer

async def run_dbtl(cfg: DBTLConfig):
    run_id = uuid.uuid4().hex[:8]
    outdir = os.path.join(cfg.outdir, run_id)
    os.makedirs(outdir, exist_ok=True)
    log.info(f"DBTL run_id={run_id} → {outdir}")

    # -------- D/B/T: Sweep (Design/Build/Test → dataset) --------
    try:
        log.info("Phase: sweep")
        df = default_sweep()
        csv_path = os.path.join(outdir, cfg.csv_name)
        df.to_csv(csv_path, index=False)
        register(RunRecord(run_id, "sweep", "OK", {"csv": csv_path}, "grid sweep completed"))
        log.info(f"Saved dataset → {csv_path} ({len(df)} rows)")
    except Exception as e:
        register(RunRecord(run_id, "sweep", "FAIL", {}, f"{e}"))
        raise

    # -------- Learn: Train surrogate --------
    try:
        log.info("Phase: train")
        model_path = os.path.join(outdir, cfg.model_name)
        tcfg = TrainConfig(csv_path=csv_path, model_out=model_path, epochs=60, batch_size=512, lr=1e-3)
        model, best_val = train_surrogate(tcfg)
        register(RunRecord(run_id, "train", "OK",
                           {"model": model_path, "val_mse": round(float(best_val),6)},
                           "training completed"))
        log.info(f"Saved model → {model_path} (best val MSE={best_val:.6f})")
    except Exception as e:
        register(RunRecord(run_id, "train", "FAIL", {}, f"{e}"))
        raise

    # -------- Learn: BO on surrogate --------
    try:
        log.info("Phase: optimize (BO)")
        model, device, used_hidden = load_model(model_path, hidden=cfg.hidden)
        sur = SurrogateWrapper(model, device)
        bocfg = BOConfig(target={"A": cfg.targetA, "B": cfg.targetB},
                         bounds=CONFIG.bounds, n_trials=cfg.trials)
        best, hist = bo_optimize(sur, bocfg)
        params_vec = [best.params.get(k, None) for k in PARAM_COLS]
        # turn into a flat list (kept in PARAM_COLS order)
        x_best = [float(p) for p in params_vec]
        # save BO artifacts
        bo_json = os.path.join(outdir, "bo_best.json")
        with open(bo_json, "w") as f:
            json.dump({"loss": float(best.value),
                       "params": {k: x_best[i] for i,k in enumerate(PARAM_COLS)},
                       "hidden": used_hidden},
                      f, indent=2)
        plot_convergence(hist, os.path.join(outdir, "bo_convergence.png"))
        register(RunRecord(run_id, "bo", "OK",
                           {"bo_best": bo_json, "hidden": str(used_hidden)},
                           "bayesian optimization completed"))
        log.info(f"BO best loss={best.value:.4f}")
    except Exception as e:
        register(RunRecord(run_id, "bo", "FAIL", {}, f"{e}"))
        raise

    # -------- Validate on true ODE --------
    try:
        log.info("Phase: validate (true ODE)")
        val = validate_on_ode(x_best)
        traj_png = os.path.join(outdir, "true_trajectory.png")
        plot_timeseries(val["t"], val["Y"], traj_png)
        res_json = os.path.join(outdir, "validate_result.json")
        with open(res_json, "w") as f:
            json.dump({"true_steady_A": float(val["steady_A"]),
                       "true_steady_B": float(val["steady_B"]),
                       "target": {"A": cfg.targetA, "B": cfg.targetB}},
                      f, indent=2)
        register(RunRecord(run_id, "validate", "OK",
                           {"trajectory": traj_png, "result": res_json},
                           "validation completed"))
        log.info(f"Validation: A={val['steady_A']:.3f}, B={val['steady_B']:.3f}")
    except Exception as e:
        register(RunRecord(run_id, "validate", "FAIL", {}, f"{e}"))
        raise

    log.info(f"DBTL run complete: {outdir}")
    return outdir

# Convenience entry for synchronous callers
def run_dbtl_sync(cfg: DBTLConfig):
    return asyncio.run(run_dbtl(cfg))
