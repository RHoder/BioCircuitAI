import asyncio
# Pseudocode skeleton
async def dbtl_cycle(simulate_fn, train_fn, optimize_fn, config):
    # 1) simulate parameter sweeps → dataset
    csv_path = await simulate_fn(config['sweep'])
    # 2) train surrogate
    model = await train_fn(csv_path)
    # 3) optimize toward target phenotype
    best_x, best_loss = await optimize_fn(model, config['target'], config['bounds'])
    return best_x, best_loss
