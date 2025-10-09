import itertools, pandas as pd
from .models import toggle_switch_ode
from .ode_solver import integrate, summarize_timeseries

def sweep(param_grid, seeds=[0,1,2]):
    rows = []
    keys = sorted(param_grid.keys())
    for vals in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        for seed in seeds:
            y0 = [0.1+0.01*seed, 0.2+0.01*seed]
            t, Y = integrate(toggle_switch_ode, y0, (0, 200), params, t_eval=None)
            summary = summarize_timeseries(t, Y)
            rows.append({**params, "steady_A": summary["steady"][0],
                         "steady_B": summary["steady"][1],
                         "var_A": summary["var"][0], "var_B": summary["var"][1]})
    df = pd.DataFrame(rows)
    return df  # caller saves to CSV
