"""Training with snapshots at an epoch ladder — the machinery for "does SAT learn
faster?".

WHY THIS EXISTS. `dynaplex.nn.train_network` returns one agent and its final
metrics; it has no callback, so there is no way to see the policy *during*
training. Measuring convergence speed needs cost(E) for several E, and the naive
way to get that is to retrain per rung: a ladder of {5,10,20,40,80,160} costs 315
epochs where one run to 160 would do. This trains ONCE and snapshots.

THE EQUIVALENCE THAT MAKES IT VALID. `train_network(epochs=E, patience=0)` returns
the BEST-VALIDATION weights within the first E epochs (`train_network` tracks
`best_state` and restores it at the end; `patience=0` is falsy so early stopping
never fires). So "the deployable policy after a budget of E epochs" is a function
of the first E epochs only, and taking a copy of `best_state` at the end of epoch
E gives exactly the same weights a dedicated E-epoch run would.

That equivalence is the load-bearing claim here, so it is asserted BITWISE against
the real `train_network` in `test_learning_curve.py` for every arm. Without that
test this module is a reimplementation of someone else's training recipe, which is
precisely the kind of thing that silently drifts.

The loop below mirrors `nn.train_network` (dynaplex 1.14) line for line: seed,
then build; one seeded shuffle then a validation split, the training set
RESHUFFLED every epoch from the same generator, masked cross-entropy over
`-1e9`-filled logits, Adam(foreach=True), checkpoints on the `val_every` cadence
plus the final epoch. Any divergence is a bug.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from dynaplex.nn import NNAgent, masked_argmax
# `_targets` is private, but it is the single source of truth for how a loss name
# becomes a target distribution; reimplementing it would be a second place to get
# soft_ce/count_ce wrong. The bitwise equivalence test covers the coupling.
from dynaplex.nn import _targets
from dynaplex.modelling.featurizers import spec_from_json


@dataclass
class CurvePoint:
    """One rung: the agent you would have deployed on a budget of `epochs`."""
    epochs: int
    agent: object
    best_val_loss: float
    best_epoch: int
    val_regret: float
    train_loss: float


@dataclass
class TrainHistory:
    """Per-epoch metrics for the whole run — free, since `train_network` already
    computes them every epoch and simply discards the history."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_regret: list[float] = field(default_factory=list)
    best_epoch_at: list[int] = field(default_factory=list)


def train_with_snapshots(samples, *, network, ladder: list[int], features=None,
                         loss: str = "ce", batch_size: int = 64, lr: float = 1e-4,
                         val_fraction: float = 0.05, val_every: int = 5, seed: int = 0,
                         device: str = "cpu", verbose: bool = False
                         ) -> tuple[list[CurvePoint], TrainHistory]:
    """Train to `max(ladder)` epochs, returning one agent per rung + the history.

    Mirrors `dynaplex.nn.train_network` exactly (see the module docstring); the
    only additions are the snapshots and the recorded history. Early stopping is
    deliberately absent: the ladder IS the budget sweep, and stopping early would
    make the upper rungs unreachable.
    """
    ladder = sorted(set(int(e) for e in ladder))
    assert ladder and ladder[0] >= 1, ladder
    epochs = ladder[-1]

    man = samples.manifest
    if features is None:
        import importlib
        fq = man["featurizer"]
        mod_name, _, attr = fq.rpartition(".")
        features_cls = getattr(importlib.import_module(mod_name), attr)
    else:
        features_cls = features
        fq = f"{features_cls.__module__}.{features_cls.__qualname__}"
    spec = spec_from_json(man["spec"])
    feats = samples.features
    num_actions = samples.num_actions

    torch.manual_seed(seed)          # BEFORE build, as train_network does (1.14):
    module = network.build(spec, num_actions)   # the initial weights are part of the seed
    targets = _targets(samples, loss)
    dev = torch.device(device)
    gen = torch.Generator().manual_seed(seed)
    if val_every < 1:
        raise ValueError(f"val_every must be >= 1, got {val_every}")
    X = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in feats.items()}
    M = torch.from_numpy(np.ascontiguousarray(samples.mask))
    T = torch.from_numpy(np.ascontiguousarray(targets.astype(np.float32)))
    n = len(samples)

    def _gather(idx):
        return {k: t[idx].to(dev) for k, t in X.items()}

    perm0 = torch.randperm(n, generator=gen)
    n_val = 0
    if val_fraction > 0.0 and n >= 2:
        n_val = min(max(1, int(round(val_fraction * n))), n - 1)
    train_idx = perm0[n_val:]
    val_idx = perm0[:n_val]
    n_train = len(train_idx)

    qbar = np.where(samples.q_count > 0,
                    samples.q_sum / np.maximum(samples.q_count, 1), np.inf)
    qbest = np.where(samples.mask & (samples.q_count > 0), qbar, np.inf).min(axis=1)

    def _validate():
        with torch.no_grad():
            xv, mv, tv = _gather(val_idx), M[val_idx].to(dev), T[val_idx].to(dev)
            logits = module(xv).masked_fill(~mv, -1e9)
            vloss = float(-(tv * torch.log_softmax(logits, dim=1)).sum(dim=1).mean())
            pred = masked_argmax(logits.cpu().numpy(), mv.cpu().numpy())
            rows = val_idx.numpy()
            regret = float(np.mean(qbar[rows, pred] - qbest[rows]))
        return vloss, regret

    module.to(dev).train()
    opt = torch.optim.Adam(module.parameters(), lr=lr, foreach=True)   # as train_network
    history = TrainHistory()
    points: list[CurvePoint] = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    best_regret = float("nan")
    mean_loss = float("nan")

    for epoch in range(1, epochs + 1):
        order = train_idx[torch.randperm(n_train, generator=gen)]
        total = 0.0
        for lo in range(0, n_train, batch_size):
            idx = order[lo:lo + batch_size]
            xb, mb, tb = _gather(idx), M[idx].to(dev), T[idx].to(dev)
            logits = module(xb).masked_fill(~mb, -1e9)
            batch_loss = -(tb * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            batch_loss.backward()
            opt.step()
            total += float(batch_loss.detach()) * len(idx)
        mean_loss = total / n_train
        # train_network (1.14) checkpoints on validation every `val_every` epochs AND
        # at its last epoch, so a run to budget E restores the best of
        # {val_every, 2*val_every, ... <= E} plus E itself. Validation itself is
        # cheap and side-effect free (eval mode, no_grad, no rng), so the history
        # still records it EVERY epoch; only the checkpoint decision follows the
        # cadence. A rung off the cadence compares its own epoch against the
        # cadence's best without feeding back into best_* (a later budget never
        # saw that epoch).
        val_loss = val_regret = float("nan")
        if n_val:
            module.eval()
            val_loss, val_regret = _validate()
            module.train()
            if epoch % val_every == 0 and val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in module.state_dict().items()}
                best_epoch = epoch
                best_regret = val_regret
        history.train_loss.append(mean_loss)
        history.val_loss.append(val_loss)
        history.val_regret.append(val_regret)
        history.best_epoch_at.append(best_epoch)

        if epoch in ladder:
            rung_val, rung_state, rung_epoch, rung_regret = best_val, best_state, best_epoch, best_regret
            if n_val and epoch % val_every != 0 and val_loss < rung_val:
                rung_val, rung_epoch, rung_regret = val_loss, epoch, val_regret
                rung_state = {k: v.detach().cpu().clone()
                              for k, v in module.state_dict().items()}
            # The deployable policy on a budget of `epoch` epochs: the best-val
            # weights SO FAR, which is what train_network(epochs=epoch, patience=0)
            # restores. Deep-copied because `best_state` is replaced later.
            #
            # The snapshot module is DEEP-COPIED, never rebuilt. `network.build`
            # draws from the GLOBAL torch rng, and the context encoder's
            # `nn.TransformerEncoderLayer` carries torch's default dropout=0.1,
            # whose masks also come from the global rng in train mode -- so
            # building a module mid-run shifts every subsequent dropout draw and
            # silently forks the training trajectory away from `train_network`'s.
            # Caught by `test_snapshot_matches_train_network_bitwise`; do not
            # replace this with a build.
            snap = copy.deepcopy(module)
            state = (copy.deepcopy(rung_state) if rung_state is not None
                     else {k: v.detach().cpu().clone()
                           for k, v in module.state_dict().items()})
            snap.load_state_dict(state)
            snap.to("cpu")
            snap.eval()
            points.append(CurvePoint(
                epochs=epoch,
                agent=NNAgent(snap, spec, num_actions, fq, network,
                              {"epochs_run": epoch, "epochs": epoch, "lr": lr,
                               "seed": seed, "batch_size": batch_size,
                               "loss": loss, "stopped_early": False,
                               "best_val_loss": rung_val, "val_regret": rung_regret,
                               "final_loss": mean_loss,
                               "generation": int(man.get("generation", 0)) + 1}),
                best_val_loss=rung_val, best_epoch=rung_epoch,
                val_regret=rung_regret, train_loss=mean_loss))
            if verbose:
                print(f"    [rung {epoch:>4}] train {mean_loss:.4f}  "
                      f"best val {rung_val:.4f} @ epoch {rung_epoch}  "
                      f"regret {rung_regret:.4f}")

    module.to("cpu").eval()
    return points, history


def epochs_to_threshold(epochs: list[int], costs: list[float],
                        tau: float) -> float | None:
    """First rung whose cost is at or below `tau`, or None if never reached.

    Deliberately NOT interpolated: the rungs are the only budgets actually
    measured, and interpolating between 40 and 80 would invent a number no run
    produced. `None` (censored) is the honest answer for an arm that never gets
    there within the ladder, and the analysis must handle censoring rather than
    dropping those seeds -- dropping them would bias the comparison towards
    whichever arm fails most often.
    """
    for e, c in zip(epochs, costs):
        if c <= tau:
            return float(e)
    return None
