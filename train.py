"""Train a neural-network policy for the vehicle-routing MDP.

Three networks (`--net`) and two algorithms (`algo`) behind one script:

  --net mlp        (default) the flat count-map features of featurizers.py
                   into an MLP; CELL action encoding, so the network's output
                   is a desirability map over the grid.
  --net attention  the order/vehicle/depot token bundle of featurizers_tokens.py
                   into the dot-attention network of networks.py; DIRECTIONAL
                   action encoding — the focal vehicle's 5 move logits are the
                   action logits.
  --net pointer    every grid cell as a token; a shared head scores each cell
                   token and, because tokens are written in action order, the
                   scores are the CELL action logits.

  python train.py dcl     Deep Controlled Learning, one generation by default:
                          collect states under the rollout policy, label each
                          with the best action found by rollouts, train the
                          network on those labels (~ the paper's NNPFA).
  python train.py ppo     PPO from scratch on the engine-backed vector env
                          (`--net pointer` has no critic; use dcl for it).

Both produce an NNAgent (saved under dynaplex_runs/, reloadable, e.g. by
visualize.py --agent) and finish with a PolicyComparer table against the
myopic and random benchmarks on common random numbers. Runs are idempotent
per (MDP, config, seed): DCL resumes from stored artifacts, so re-running
with the same arguments is free.

How the collection is spread over the machine (processes, engines, workers,
rollout slots, the device the network runs on) comes from dynaplex's machine
plan: `--machine auto` recognizes a laptop or a Snellius allocation and applies
the plan engineered for it (`python -m dynaplex.machine` shows what it sees).

The Snellius-scale attention runs of the README (paper budget: n=10000,
m=2000, h=100, random generation-0 rollouts) are

    python train.py dcl --net attention --rollout random --n 10000 --m 2000 --h 100 --patience 15
    python train.py dcl --net pointer   --rollout random --n 10000 --m 2000 --h 100 --patience 15
"""
from __future__ import annotations

import argparse

import dynaplex
from dynaplex.nn import Net

from featurizers import VehicleFeaturizer
from featurizers_tokens import GridTokenFeaturizer, TokenFeaturizer
from mdp import ActionEncoding, MyopicPolicy, VehicleRoutingMDP


def setup(net: str, d_model: int, layers: int):
    """The (mdp, featurizer, network, critic) quartet for a `--net` choice."""
    if net == "mlp":
        return VehicleRoutingMDP(), VehicleFeaturizer, None, None
    if net == "attention":
        return (VehicleRoutingMDP(action_encoding=ActionEncoding.DIRECTIONAL),
                TokenFeaturizer,
                Net("networks.attention_net", d_model=d_model, n_layers=layers,
                    n_heads=4, head="moves5"),
                Net("networks.attention_critic", d_model=d_model, n_layers=layers,
                    n_heads=4))
    if net == "pointer":
        return (VehicleRoutingMDP(), GridTokenFeaturizer,
                Net("networks.attention_pointer_net", d_model=d_model,
                    n_layers=layers, n_heads=4),
                None)
    raise ValueError(net)


def evaluate(mdp: VehicleRoutingMDP, agents: dict) -> None:
    comparer = dynaplex.PolicyComparer(
        mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
    policies = {"myopic": MyopicPolicy(mdp), "random": dynaplex.RandomPolicy(mdp)}
    policies.update(agents)
    print()
    print(comparer.compare(policies, benchmark="myopic"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("algo", choices=["dcl", "ppo"])
    parser.add_argument("--net", choices=["mlp", "attention", "pointer"], default="mlp",
                        help="featurizer + network pair (module docstring)")
    parser.add_argument("--d-model", type=int, default=64,
                        help="attention/pointer: token embedding width")
    parser.add_argument("--layers", type=int, default=2,
                        help="attention/pointer: transformer encoder layers")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--machine", default="auto",
                        help="dcl: execution plan (processes, engines, workers, slots, "
                             "device); 'auto' recognizes this machine — a laptop, or a "
                             "Snellius node (see `python -m dynaplex.machine`)")
    # DCL knobs
    parser.add_argument("--rollout", choices=["myopic", "random"], default="myopic",
                        help="dcl: the generation-0 rollout/labeling policy")
    parser.add_argument("--generations", type=int, default=1,
                        help="dcl: total trained generations (gen 0 is labeled "
                             "by the rollout policy, later ones by the previous net)")
    parser.add_argument("--n", type=int, default=1000,
                        help="dcl: labeled samples per generation")
    parser.add_argument("--m", type=int, default=64,
                        help="dcl: rollout replications per candidate action")
    parser.add_argument("--h", type=int, default=40,
                        help="dcl: rollout horizon (periods)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="dcl: max training epochs (default: train_network's)")
    parser.add_argument("--lr", type=float, default=None,
                        help="dcl: training learning rate (default: train_network's)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="dcl: training minibatch size (default: train_network's)")
    parser.add_argument("--patience", type=int, default=None,
                        help="dcl: early-stopping patience in epochs (default: train_network's)")
    # PPO knobs
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="ppo: total environment steps")
    args = parser.parse_args()

    mdp, features, network, critic = setup(args.net, args.d_model, args.layers)

    if args.algo == "dcl":
        rollout = (MyopicPolicy(mdp) if args.rollout == "myopic"
                   else dynaplex.RandomPolicy(mdp))
        train = {key: value for key, value in
                 (("epochs", args.epochs), ("lr", args.lr),
                  ("batch_size", args.batch_size), ("patience", args.patience))
                 if value is not None}
        dcl = dynaplex.DCL(
            mdp, rollout, features=features, network=network,
            n=args.n, m=args.m, h=args.h,
            machine=args.machine, seed=args.seed, train=train)
        agents = dcl.run(generations=args.generations)
        evaluate(mdp, {f"agent_gen{i}": a for i, a in enumerate(agents, start=1)})
    else:
        if args.net == "pointer":
            raise SystemExit("--net pointer has no PPO critic over the grid-token bundle; "
                             "train it with dcl")
        config = dynaplex.PPOConfig(
            minibatch_size=4096, total_timesteps=args.timesteps, seed=args.seed)
        ppo = dynaplex.PPO(mdp, features=features, network=network,
                           critic_network=critic, config=config)
        evaluate(mdp, {"trained": ppo.train()})


if __name__ == "__main__":
    main()
