import optuna
import numpy as np

class SurrogateWrapper:
    def __init__(self, model, x_cols, scaler=None):
        self.model, self.x_cols, self.scaler = model, x_cols, scaler
    def predict(self, x_dict):
        # x_dict: {'alpha_A':..., 'K_A':..., ...}
        x = np.array([x_dict[k] for k in self.x_cols], dtype=np.float32)[None,:]
        # apply scaler if used
        import torch
        with torch.no_grad():
            y = self.model(torch.tensor(x)).numpy()[0]
        return y  # e.g., [steady_A, steady_B]

def bo_optimize(target, bounds, surrogate, n_trials=50):
    def objective(trial):
        x = {k: trial.suggest_float(k, *bounds[k]) for k in bounds}
        y = surrogate.predict(x)
        # Example: target steady_A ~ target['A']; loss = |yA - tA| + |yB - tB|
        loss = abs(y[0] - target['A']) + abs(y[1] - target['B'])
        return loss
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value
