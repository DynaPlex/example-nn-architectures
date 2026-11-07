"""1.14 shim: the 1.12 `make_context(mdp, seed, eval=, sample=, trajectory=)`."""
from dynaplex.modelling import new_context
def make_context(mdp, seed: int, *, eval: bool = False, sample: int = 0, trajectory: int = 0):
    ctx = new_context(mdp, seed)
    ctx.reseed(seed, 1 if eval else 0, sample, trajectory)
    return ctx
