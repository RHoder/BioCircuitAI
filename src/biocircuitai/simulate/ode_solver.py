from scipy.integrate import solve_ivp

def integrate(model_fn, y0, t_span, params, t_eval=None, **solver_kw):
    # wraps solve_ivp; returns times, states
    sol = solve_ivp(lambda t, y: model_fn(t, y, params),
                    t_span=t_span, y0=y0, t_eval=t_eval,
                    method=solver_kw.get("method","RK45"),
                    rtol=1e-6, atol=1e-8)
    return sol.t, sol.y.T

def summarize_timeseries(t, Y):
    # compute steady-state or oscillation amplitude/frequency
    # Pseudocode: use last 20% of points
    tail = Y[int(0.8*len(Y)):, :]
    steady = tail.mean(axis=0)
    var = tail.var(axis=0)
    return {"steady": steady, "var": var}
