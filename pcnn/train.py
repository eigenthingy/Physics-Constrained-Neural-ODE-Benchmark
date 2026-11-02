"""Training loop -- one loop, four models, identical budget.

Fairness is enforced here rather than assumed:

* the budget is counted in **optimizer steps**, not epochs, so a model with a
  cheaper forward pass does not silently get more updates;
* every model sees the same data, the same batch order and the same LR
  schedule for a given seed;
* model selection uses the **noisy** validation targets, because at deployment
  nobody has clean ones. Selecting on clean data is a subtle but common way to
  leak the answer into a noise-robustness study.

The soft-penalty model gets one extra thing the others do not: unlabelled
collocation points. That is not an unfairness, it is the method -- the whole
selling point of a PINN residual is that it needs no labels. We report it.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import torch

from .data import DataBundle
from .losses import physics_residual
from .systems import System


@dataclass
class TrainConfig:
    steps: int = 6000
    batch_size: int = 512
    lr: float = 3e-3
    min_lr_frac: float = 0.01
    weight_decay: float = 1e-6
    lam: float = 0.0  #: weight of the physics penalty (model B only)
    n_colloc_batch: int = 512
    eval_every: int = 250
    grad_clip: float = 10.0
    seed: int = 0
    history: bool = False


@dataclass
class TrainResult:
    val_loss: float
    train_loss: float
    steps_done: int
    seconds: float
    n_params: int
    curve: list = field(default_factory=list)


def _cosine_lr(step, total, lr, min_frac):
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * step / total)))


def train(model, system: System, data: DataBundle, cfg: TrainConfig) -> TrainResult:
    torch.manual_seed(cfg.seed + 10_000)
    gen = torch.Generator().manual_seed(cfg.seed + 20_000)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    zt, dzt = data.train.z, data.train.dz
    n = zt.shape[0]
    colloc = data.colloc

    best = (float("inf"), copy.deepcopy(model.state_dict()))
    curve = []
    t0 = time.time()

    model.train()
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, cfg.steps, cfg.lr, cfg.min_lr_frac)

        idx = torch.randint(0, n, (min(cfg.batch_size, n),), generator=gen)
        pred = model(zt[idx])
        loss = torch.nn.functional.mse_loss(pred, dzt[idx])
        data_loss = loss.detach().item()

        if cfg.lam > 0.0:
            cidx = torch.randint(0, colloc.shape[0], (cfg.n_colloc_batch,), generator=gen)
            loss = loss + cfg.lam * physics_residual(model, system, colloc[cidx])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            v = _val_loss(model, data)
            if v < best[0]:
                best = (v, copy.deepcopy(model.state_dict()))
            if cfg.history:
                curve.append(dict(step=step + 1, train=data_loss, val=v))

    model.load_state_dict(best[1])
    model.eval()
    return TrainResult(
        val_loss=best[0],
        train_loss=data_loss,
        steps_done=cfg.steps,
        seconds=time.time() - t0,
        n_params=model.n_parameters(),
        curve=curve,
    )


def _val_loss(model, data: DataBundle) -> float:
    """MSE against the *noisy* validation targets -- what a practitioner can see."""
    was = model.training
    model.eval()
    with torch.no_grad():
        v = torch.nn.functional.mse_loss(model(data.val.z), data.val.dz).item()
    if was:
        model.train()
    return v
