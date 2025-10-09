import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .dataset import CircuitDataset
from .model import MLP

def train_surrogate(csv_path, x_cols, y_cols, epochs=200, bs=64, lr=1e-3):
    ds = CircuitDataset(csv_path, x_cols, y_cols)
    n_val = max(1, int(0.2*len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    tl, vl = DataLoader(train_ds, bs, shuffle=True), DataLoader(val_ds, bs)
    model = MLP(len(x_cols), len(y_cols))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = (1e9, None)

    for ep in range(epochs):
        model.train(); tr = 0
        for X,y in tl:
            opt.zero_grad(); pred = model(X); loss = loss_fn(pred,y); loss.backward(); opt.step()
            tr += loss.item()*len(X)
        model.eval(); va = 0
        with torch.no_grad():
            for X,y in vl:
                va += loss_fn(model(X), y).item()*len(X)
        tr /= len(train_ds); va /= len(val_ds)
        if va < best[0]: best = (va, {k:v.cpu().clone() for k,v in model.state_dict().items()})
        # print or log(tr, va)
    if best[1] is not None: model.load_state_dict(best[1])
    return model
