"""Phase 1 spike: does the AGILE architecture do anything on this MDP?

Trains the reference structured-action transformer (`sat_net_ref`) with PPO over
the CPython env (`sat_env`), then scores it against the myopic benchmark.

HOW TO READ THE OUTPUT.

`PolicyComparer` refuses `backend="python"` for NN policies
(policy_comparer.py:715-717) — the engine NN harness is the only supported path
— so this script cannot use the instrument that produced the published table:

    myopic 3.096  |  flat MLP 4.07  |  order-token attention 3.324

It therefore evaluates in its own CPython loop and runs MYOPIC through that same
loop as a control, reporting both absolutes and the ratio.

The loop is CALIBRATED, not merely self-consistent: at episodes=32, warmup=100,
horizon=500 it scores myopic at 3.0815 against the engine comparer's 3.096
(within sampling noise), which confirms the period accounting and that the
CPython dynamics match the compiled ones. So the absolutes here are meaningful
at an adequate budget — but they come from a different estimator with far fewer
trajectories and no common-random-number pairing, so treat the ratio as the
headline and the absolute as indicative. Cut the budget (--quick) and the
transient dominates: the same myopic policy scores 4.39 at warmup=20/horizon=60.

The definitive comparable number still arrives in Phase 2a, from
`featurizers_sat` + DCL + the comparer.

Cost accounting: VRP is infinite-horizon average cost, so the figure is mean
cost per PERIOD. A period is one event step, and each period contains one
decision per vehicle with capacity left, so decision steps != periods — the
evaluator divides by elapsed periods (`ctx.time_elapsed`), never by steps.

Usage:
    python run_sat_spike.py              # short local run (~10 min CPU)
    python run_sat_spike.py --quick      # ~1 min, for checking the wiring
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from sat_compat import make_context
from dynaplex.modelling import StateCategory

from mdp import ActionEncoding, MyopicPolicy, VehicleRoutingMDP
from sat_schema import build_sat_observation
from sat_tokenizer import SequenceBuilder
from sat_trainer import SATTrainerConfig, train_sat_pointer_ppo


def evaluate(mdp, act, *, episodes: int, warmup: int, horizon: int,
             seed: int = 42) -> float:
    """Mean cost per period under `act`, measured in plain CPython.

    `act(state, ctx) -> action` is called at every AWAIT_ACTION. Mirrors the
    comparer's shape — a warmup discarded so the average-cost estimate is not
    dominated by the initial transient, then `horizon` periods counted. Seeds
    live in the eval space (`eval=True`), disjoint from training.

    Calibration: at episodes=32, warmup=100, horizon=500 this scores
    `MyopicPolicy` at 3.0815 where the engine comparer reports 3.096.

    Episodes ARE paired across policies (common random numbers): every policy is
    evaluated at the same `(seed, eval=True, trajectory=ep)` coordinates, so
    episode `ep` starts from the same state and draws the same event stream. The
    pairing decays once two policies diverge — the arrival redraw loop in
    `modify_state_with_event` consumes a state-dependent number of draws — which
    is equally true of the comparer. What this estimator really lacks is
    trajectory count (tens, against the comparer's 256), so treat small gaps
    between two policies as noise.
    """
    total_cost = 0.0
    total_periods = 0
    for ep in range(episodes):
        ctx = make_context(mdp, seed, eval=True, trajectory=ep)
        state = mdp.get_initial_state(ctx)

        # burn in, then zero the meter
        while ctx.time_elapsed < warmup:
            _advance_one_period(mdp, state, ctx, act)
        cost0, t0 = ctx.cumulative_cost, ctx.time_elapsed

        while ctx.time_elapsed < warmup + horizon:
            _advance_one_period(mdp, state, ctx, act)

        total_cost += ctx.cumulative_cost - cost0
        total_periods += ctx.time_elapsed - t0
    return total_cost / max(total_periods, 1)


def _advance_one_period(mdp, state, ctx, act) -> None:
    """One decision round plus its event step."""
    while state.category == StateCategory.AWAIT_ACTION:
        mdp.modify_state_with_action(state, ctx, act(state, ctx))
    while state.category == StateCategory.AWAIT_EVENT:
        mdp.modify_state_with_event(state, ctx)
        # one event step closes the period; the next decision round follows
        if state.category == StateCategory.AWAIT_ACTION:
            break


def _masked_logits(policy_net, seq_builder, mdp, state, ctx):
    """Validity-masked logits for one state.

    The mask is not optional. Action tokens span every action index (so logit
    position == action index), which means an unmasked argmax returns
    inadmissible actions — for DIRECTIONAL, an off-grid move. Measured on this
    MDP, unmasked sampling picks inadmissible actions 6% of the time under
    DIRECTIONAL and 92% under CELL.
    """
    seq = seq_builder.build(build_sat_observation(state, mdp))
    logits = policy_net.forward_logits(
        type_features=seq["type_features"],
        type_ids=seq["type_ids"],
        action_mask=seq["action_mask"],
        indices_by_type=seq["indices_by_type"],
        action_group_ids=seq.get("action_group_ids"),
        num_action_groups=seq.get("num_action_groups"),
        intra_group_pos=seq.get("intra_group_pos"),
    )
    mdp.write_action_validity(state, ctx.valid)
    m = torch.from_numpy(np.asarray(ctx.valid.arr[0], dtype=np.bool_).copy())
    return logits.masked_fill(~m[:logits.shape[0]], -1e9)


def make_net_policy(policy_net, seq_builder, mdp, *, greedy: bool,
                    seed: int = 0):
    """Policy from a trained net — argmax if `greedy`, else sampled.

    REPORT BOTH. On an undertrained net the two differ enormously, and the gap
    is the most informative diagnostic the spike produces. Measured here after
    30,720 timesteps: policy entropy 1.50 against uniform 1.61 — i.e. the logits
    are nearly flat — yet the greedy readout picked only 'left' and 'right' on
    every state it saw. A near-uniform policy read greedily becomes an arbitrary
    CONSTANT policy: the vehicle oscillates between two cells, never reaches an
    order, and cost climbs to holding_cost * max_orders. That scored 90.6 where
    sampling the same weights scores near the random baseline.

    So a bad greedy number on its own does NOT distinguish "the architecture
    failed" from "the logits are still flat". Greedy minus sampled does: a large
    gap means undertrained, a small gap with a good score means it has learned.
    """
    policy_net.eval()
    gen = torch.Generator().manual_seed(seed)

    def act(state, ctx):
        with torch.no_grad():
            logits = _masked_logits(policy_net, seq_builder, mdp, state, ctx)
            if greedy:
                return int(logits.argmax())
            probs = torch.softmax(logits, dim=0)
            return int(torch.multinomial(probs, 1, generator=gen))

    return act


def policy_diagnostics(policy_net, seq_builder, mdp, *, episodes=2,
                       warmup=20, horizon=40, seed=123):
    """Mean policy entropy and the greedy action histogram over visited states.

    Distinguishes the two failure modes that produce a bad score: entropy near
    log(num_actions) means the policy never sharpened (undertrained), entropy
    near 0 with one dominant action means it collapsed onto a degenerate action.
    """
    policy_net.eval()
    counts = np.zeros(mdp.num_actions, dtype=np.int64)
    ents: list[float] = []

    def act(state, ctx):
        with torch.no_grad():
            logits = _masked_logits(policy_net, seq_builder, mdp, state, ctx)
            p = torch.softmax(logits, dim=0)
            ents.append(float(-(p * torch.log(p + 1e-12)).sum()))
            a = int(logits.argmax())
        counts[a] += 1
        return a

    for ep in range(episodes):
        ctx = make_context(mdp, seed, eval=True, trajectory=ep)
        state = mdp.get_initial_state(ctx)
        while ctx.time_elapsed < warmup + horizon:
            _advance_one_period(mdp, state, ctx, act)
    return float(np.mean(ents)) if ents else 0.0, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget, just checks the wiring end to end")
    ap.add_argument("--encoding", choices=["directional", "cell"],
                    default="directional",
                    help="DIRECTIONAL keeps 5 action tokens; CELL writes G*G "
                         "(20x the action-side cost, see the plan's risk (d))")
    args = ap.parse_args()

    enc = (ActionEncoding.DIRECTIONAL if args.encoding == "directional"
           else ActionEncoding.CELL)
    mdp = VehicleRoutingMDP(action_encoding=enc)

    if args.quick:
        cfg = SATTrainerConfig(num_episodes=5, num_envs=4, rollout_steps=32,
                               max_steps_per_episode=128, minibatch_size=32,
                               ppo_epochs=2, log_every=1, seed=0)
        eval_kw = dict(episodes=2, warmup=20, horizon=60)
    else:
        # Budget set by CPython throughput, ~19 timesteps/s: the tokenizer walks
        # a dataclass tree per observation and the net embeds per sample in a
        # Python loop. 60 x 64 x 8 = 30,720 timesteps is ~25 min; the greedy eval
        # adds ~10 min (one batch-1 forward per decision). This is the spike's
        # real ceiling and the honest argument for the engine bridge — DCL runs
        # 97,280 slots per ROUND on 190 workers.
        cfg = SATTrainerConfig(num_episodes=60, num_envs=8, rollout_steps=64,
                               max_steps_per_episode=512, minibatch_size=64,
                               ppo_epochs=4, log_every=5, seed=0)
        eval_kw = dict(episodes=8, warmup=100, horizon=500)

    print(f"=== Phase 1 SAT spike | {enc.name} | num_actions={mdp.num_actions}")
    print(f"    eval: {eval_kw}")

    t0 = time.time()
    policy_net, _value_net, seq_builder, _rewards = train_sat_pointer_ppo(
        mdp=mdp,
        make_state=mdp.get_initial_state,
        make_action_set=lambda s: build_sat_observation(s, mdp),
        config=cfg,
    )
    train_s = time.time() - t0
    timesteps = cfg.num_episodes * cfg.rollout_steps * cfg.num_envs
    print(f"\ntrained in {train_s:.1f}s  "
          f"({timesteps:,} timesteps, {timesteps / train_s:.1f} timesteps/s)")
    # The throughput is the point, not an aside. DCL refills 97,280 slots per
    # ROUND across 190 compiled workers; at this rate the same volume of
    # experience would take days. Whatever the cost figures below say, the
    # engine bridge is what makes the architecture trainable at paper scale.

    # save the weights so a bad number can be investigated without retraining
    ckpt = f"sat_spike_{args.encoding}.pt"
    torch.save({"policy": policy_net.state_dict(),
                "raw_dims": seq_builder.raw_dims,
                "type_names": seq_builder.type_names,
                "d_model": cfg.d_model, "nhead": cfg.nhead,
                "num_layers": cfg.num_layers}, ckpt)
    print(f"saved policy weights + architecture metadata to {ckpt}")

    # --- diagnostics: is the policy sharp, flat, or collapsed? --------------
    ent, hist = policy_diagnostics(policy_net, seq_builder, mdp)
    uniform = float(np.log(mdp.num_actions))
    print(f"\npolicy entropy {ent:.4f} (uniform over {mdp.num_actions} "
          f"= {uniform:.4f}, collapsed = 0)")
    if mdp.num_actions <= 10:
        print(f"greedy action histogram {hist.tolist()}")
    print(f"greedy dominant-action share {hist.max() / max(hist.sum(), 1):.1%}")

    # --- score every policy on the SAME loop --------------------------------
    myopic = MyopicPolicy(mdp)
    rng = np.random.default_rng(0)

    def random_valid(state, ctx):
        mdp.write_action_validity(state, ctx.valid)
        return int(rng.choice(np.flatnonzero(np.asarray(ctx.valid.arr[0]))))

    runs = [
        ("myopic", lambda s, c: myopic.get_action(s, c)),
        ("random (valid actions)", random_valid),
        ("SAT/AGILE sampled", make_net_policy(policy_net, seq_builder, mdp,
                                              greedy=False)),
        ("SAT/AGILE greedy", make_net_policy(policy_net, seq_builder, mdp,
                                             greedy=True)),
    ]
    results: dict[str, float] = {}
    for name, act in runs:
        print(f"\nevaluating {name} ...", flush=True)
        t0 = time.time()
        results[name] = evaluate(mdp, act, **eval_kw)
        print(f"  {results[name]:.4f} cost/period  ({time.time() - t0:.1f}s)")

    c_myopic = results["myopic"]
    print("\n" + "=" * 62)
    print(f"{'policy':<30}{'cost/period':>14}{'/myopic':>10}")
    print("-" * 62)
    for name in results:
        print(f"{name:<30}{results[name]:>14.4f}"
              f"{results[name] / c_myopic:>10.2f}")
    print(f"{'cap (holding x max_orders)':<30}"
          f"{mdp.holding_cost * mdp.max_orders:>14.4f}"
          f"{mdp.holding_cost * mdp.max_orders / c_myopic:>10.2f}")
    print("=" * 62)
    print("How to read this. A score near the CAP means the policy is\n"
          "effectively constant — orders accumulate to max_orders and never get\n"
          "served. A large greedy-vs-sampled gap means the logits are still\n"
          "flat (undertrained), NOT that the architecture failed: greedy turns\n"
          "a near-uniform policy into an arbitrary constant one. Beating\n"
          "'random (valid actions)' is the first real milestone; beating myopic\n"
          "is the goal. Absolutes are from this CPython loop, calibrated\n"
          "against the engine on myopic (3.0815 vs 3.096) but with far fewer\n"
          "trajectories — Phase 2a produces the definitive figure.")


if __name__ == "__main__":
    main()
