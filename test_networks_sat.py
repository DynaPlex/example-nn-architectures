"""Tests for the dense AGILE net (Phase 2a).

The dense rewrite deleted most of the masking machinery `sat_net_ref` needs (the
-inf key bias, the dummy pooling bin, the NaN guards), on the argument that a
complete action set and count-derived context padding make them unnecessary.
These tests are what makes that argument checkable, and they target the two
things that argument could get wrong:

  * PADDING INDEPENDENCE — the `orders` slab is padded to `max_orders` and the
    real length lives in `orders_count`. If the derived key-padding mask is
    off by one, or built for the wrong slice of the concatenated sequence, the
    network silently attends to garbage. Shapes cannot detect this; only
    perturbing the padded region and asserting the output does not move can.
  * MASK INDEPENDENCE — the net must emit finite logits for inadmissible
    actions and must never consult the validity mask, because cropping happens
    downstream (masked CE / masked argmax). A NaN escaping here would poison the
    argmax that selects the action.

There is deliberately NO weight-copy parity test against `sat_net_ref`. Making
one possible would mean keeping the within-group action layer (dead compute at
group size 1) and folding the dropped type embeddings into projection biases —
i.e. distorting the production net so a test can exist. The featurizer parity
oracle already pins the tokens exactly; these invariants pin the network's
handling of them, which is the part the rewrite actually changed.
"""
import copy

import numpy as np
import pytest
import torch

import dynaplex.modelling as dpm
from sat_compat import make_context

if not hasattr(dpm, "VariableObjectWriter"):
    pytest.skip("this dynaplex has no object-token writers — install dynaplex==1.12.2 "
                "from Test PyPI (see README/requirements.txt), not 1.12.0 from regular PyPI",
                allow_module_level=True)

from dynaplex.modelling import (
    allocate_arrays,
    feature_spec,
    featurizer_template,
    holder_from_featurizer,
)

from featurizers_sat import SATFeaturizer
from mdp import ActionEncoding, MyopicPolicy, VehicleRoutingMDP
from networks_sat import sat_agile_critic, sat_agile_net

ENCODINGS = [ActionEncoding.DIRECTIONAL, ActionEncoding.CELL]


def _mdp(encoding=ActionEncoding.DIRECTIONAL):
    return VehicleRoutingMDP(grid_size=6, n_vehicles=3, capacity=3,
                            max_orders=12, action_encoding=encoding)


def _real_batch(mdp, n=6, seed=0):
    """A batch of REAL featurized states (not zeros — zeros hide weight grads)."""
    ctx = make_context(mdp, seed)
    policy = MyopicPolicy(mdp)
    state = mdp.get_initial_state(ctx)
    states = []
    while len(states) < n:
        states.append(copy.deepcopy(state))
        mdp.modify_state_with_action(state, ctx, policy.get_action(state, ctx))
        while state.category.name == "AWAIT_EVENT":
            mdp.modify_state_with_event(state, ctx)

    spec = feature_spec(SATFeaturizer, mdp)
    arrays = allocate_arrays(spec, len(states))
    fz = featurizer_template(SATFeaturizer, mdp)
    fz = fz.bind(holder_from_featurizer(SATFeaturizer)(**arrays))
    for i, st in enumerate(states):
        fz.reset(i)
        fz.write_features(st)
        fz.finish()
    return spec, {k: torch.as_tensor(v) for k, v in arrays.items()}, states


def _net(mdp, spec, seed=0):
    torch.manual_seed(seed)
    net = sat_agile_net(spec, mdp.num_actions, d_model=32, n_layers=1, n_heads=2)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Contract with the DCL/PPO plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_forward_returns_finite_action_logits(encoding):
    mdp = _mdp(encoding)
    spec, batch, _ = _real_batch(mdp)
    with torch.no_grad():
        out = _net(mdp, spec)(batch)
    assert out.shape == (batch["moves"].shape[0], mdp.num_actions)
    assert torch.isfinite(out).all(), "a NaN/inf here poisons the masked argmax"


def test_critic_returns_one_value_per_sample():
    """PPO builds critics as factory(spec, 1)."""
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp)
    torch.manual_seed(0)
    crit = sat_agile_critic(spec, 1, d_model=32, n_layers=1, n_heads=2)
    with torch.no_grad():
        v = crit(batch)
    assert v.shape == (batch["moves"].shape[0], 1)
    assert torch.isfinite(v).all()


def test_gradients_reach_every_parameter():
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp)
    net = _net(mdp, spec)
    net.train()
    net(batch).sum().backward()
    dead = [n for n, p in net.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    # `depot` is a constant zero vector on this MDP (depot is fixed at (0,0)), so
    # its projection WEIGHT has a structurally dead gradient — only its bias can
    # learn. Documented in test_sat_net_ref.test_depot_token_is_a_constant_zero_vector.
    assert dead == ["proj.depot.weight"], dead


def test_wrong_num_actions_is_rejected_loudly():
    """A spec/MDP mismatch must fail at construction, not silently truncate."""
    mdp = _mdp()
    spec = feature_spec(SATFeaturizer, mdp)
    with pytest.raises(AssertionError, match="1:1"):
        sat_agile_net(spec, mdp.num_actions + 1)


def test_missing_bundle_entries_are_rejected_loudly():
    mdp = _mdp()
    spec = dict(feature_spec(SATFeaturizer, mdp))
    del spec["orders_count"]
    with pytest.raises(AssertionError, match="orders_count"):
        sat_agile_net(spec, mdp.num_actions)


def test_factory_is_reachable_by_dotted_path():
    """DCL specifies networks constructibly (`dp.Net("networks_sat.sat_agile_net")`)
    so a cold cluster node can rebuild them; the path must resolve from the repo
    root and the kwargs must be JSON-round-trippable."""
    import importlib
    import json

    mod = importlib.import_module("networks_sat")
    for name in ("sat_agile_net", "sat_agile_critic"):
        assert callable(getattr(mod, name))
    kwargs = dict(d_model=32, n_layers=1, n_heads=2, ff=64, hidden_head=32)
    assert json.loads(json.dumps(kwargs)) == kwargs
    mdp = _mdp()
    spec = feature_spec(SATFeaturizer, mdp)
    assert getattr(mod, "sat_agile_net")(spec, mdp.num_actions, **kwargs) is not None


# ---------------------------------------------------------------------------
# Padding independence — the count-derived mask
# ---------------------------------------------------------------------------


def test_padded_orders_cannot_affect_the_output():
    """Scribble on the order slab PAST each row's count; the output must not move.

    This is the single most important test in this file. The engine zero-fills
    the tail, so a broken mask would go unnoticed in normal operation and only
    show up as a quiet accuracy loss once buffers were reused with longer rows.
    """
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp, n=6)
    net = _net(mdp, spec)
    with torch.no_grad():
        before = net(batch)

    counts = batch["orders_count"]
    assert (counts < spec["orders"].sample_shape[0]).any(), \
        "no row has padding — the test would be vacuous"

    poisoned = {k: v.clone() for k, v in batch.items()}
    for i, n in enumerate(counts.tolist()):
        poisoned["orders"][i, int(n):] = 7.5   # garbage past the count
    with torch.no_grad():
        after = net(poisoned)

    assert torch.equal(before, after), (
        "padded order tokens changed the output — the key-padding mask derived "
        "from orders_count is wrong")


def test_every_state_object_is_a_context_token():
    """All objects are embedded, not just the actions.

    `sat_net_ref` tokenises the whole observation tree and routes everything that
    is not part of an Action subtree through the context encoder. The dense
    rewrite must carry the same population: the summary, the depot, every
    vehicle, and one token per waiting order. Asserted against the reflective
    tokenizer's own context count, so the two cannot drift apart.
    """
    from sat_schema import build_sat_observation
    from sat_tokenizer import SequenceBuilder

    mdp = _mdp()
    spec, batch, states = _real_batch(mdp, n=4)
    net = _net(mdp, spec)
    tokens, pad = net._context(batch)

    sb = SequenceBuilder(skip_root=True)
    for i, st in enumerate(states):
        seq = sb.build(build_sat_observation(st, mdp))
        want = int((~seq["action_mask"]).sum())     # tokenizer's context tokens
        got = int((~pad[i]).sum())                  # dense real context tokens
        assert got == want, (
            f"row {i}: dense path has {got} real context tokens but the "
            f"tokenizer emits {want} — an object type is missing from _CONTEXT")
        # and the count is the sum of the per-tensor populations
        assert want == 1 + 1 + mdp.n_vehicles + int(batch["orders_count"][i])


def test_pad_mask_aligns_with_the_concatenation_order():
    """The mask must track _CONTEXT's order, not assume variable tensors come last.

    Regression guard: the offset used to be precomputed as the total width of the
    fixed tensors, which is correct only while every variable tensor follows
    every fixed one. Reordering _CONTEXT would then have masked the wrong slice —
    silently, since shapes are unaffected. Here the expected mask is rebuilt
    independently from the declared order and compared.
    """
    import networks_sat as ns

    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp, n=5)
    net = _net(mdp, spec)
    _tokens, pad = net._context(batch)

    B = batch[ns._CONTEXT[0]].shape[0]
    expected = []
    for name in ns._CONTEXT:
        width = batch[name].shape[1]
        if name in ns._COUNTS:
            cnt = batch[ns._COUNTS[name]].reshape(B, 1)
            expected.append(torch.arange(width).unsqueeze(0) >= cnt)
        else:
            expected.append(torch.zeros(B, width, dtype=torch.bool))
    assert torch.equal(pad, torch.cat(expected, dim=1))

    # the real tokens must be a CONTIGUOUS prefix of each tensor's block, in
    # declared order — i.e. padding only ever sits at the tail of a block
    offset = 0
    for name in ns._CONTEXT:
        width = batch[name].shape[1]
        block = pad[:, offset:offset + width]
        for row in range(B):
            real = (~block[row]).nonzero().flatten()
            if real.numel():
                assert real.tolist() == list(range(real.numel())), (
                    f"{name}: real tokens are not a prefix of the block")
        offset += width
    assert offset == pad.shape[1], "the mask covers exactly the concatenation"


def test_order_token_permutation_does_not_change_the_output():
    """Orders are a SET. The net has no positional encoding (position rides in
    the features), so permuting the real order tokens must be a no-op."""
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp, n=4)
    net = _net(mdp, spec)
    with torch.no_grad():
        before = net(batch)

    shuffled = {k: v.clone() for k, v in batch.items()}
    g = torch.Generator().manual_seed(0)
    for i, n in enumerate(batch["orders_count"].tolist()):
        n = int(n)
        if n > 1:
            shuffled["orders"][i, :n] = batch["orders"][i, :n][
                torch.randperm(n, generator=g)]
    with torch.no_grad():
        after = net(shuffled)
    assert torch.allclose(before, after, atol=1e-5)


def test_a_row_with_zero_orders_is_handled():
    """Empty variable block: the masked mean-pool must not divide by zero, and
    the summary/depot/vehicle tokens keep the row well-defined."""
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp, n=3)
    batch = {k: v.clone() for k, v in batch.items()}
    batch["orders_count"][0] = 0
    batch["orders"][0] = 0.0
    with torch.no_grad():
        out = _net(mdp, spec)(batch)
    assert torch.isfinite(out).all()


def test_orders_count_dtype_does_not_change_the_output():
    """DCL delivers `orders_count` as int64; PPO's rollout buffer allocates every
    spec tensor as float32 (`ppo.py:404` passes no dtype), so the same module sees
    both. It must behave identically."""
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp)
    net = _net(mdp, spec)
    as_float = {k: (v.float() if k == "orders_count" else v)
                for k, v in batch.items()}
    assert batch["orders_count"].dtype == torch.int64
    with torch.no_grad():
        assert torch.equal(net(batch), net(as_float))


# ---------------------------------------------------------------------------
# Mask independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_logits_are_finite_on_inadmissible_actions(encoding):
    """The net scores every action, admissible or not, and must never emit a
    sentinel of its own — the downstream mask supplies -1e9/-inf."""
    mdp = _mdp(encoding)
    spec, batch, states = _real_batch(mdp, n=5)
    with torch.no_grad():
        out = _net(mdp, spec)(batch)

    ctx = make_context(mdp, 0)
    checked = 0
    for i, st in enumerate(states):
        mdp.write_action_validity(st, ctx.valid)
        invalid = ~np.asarray(ctx.valid.arr[0], dtype=np.bool_)
        if not invalid.any():
            # legitimate under DIRECTIONAL: a vehicle in the grid INTERIOR has
            # all five moves admissible. (An earlier version asserted every state
            # had an inadmissible action, which only held because a state-aliasing
            # bug in the fixture made every row the same boundary state.)
            continue
        checked += 1
        vals = out[i].numpy()[invalid]
        assert np.isfinite(vals).all()
        assert (np.abs(vals) < 1e6).all(), "the net emitted a mask-like sentinel"
    assert checked, ("no state in the batch had an inadmissible action, so this "
                     "test asserted nothing — widen the batch or the seed")


def test_output_does_not_depend_on_the_validity_mask():
    """The net must not read the mask — asserted by the fact that it has no way
    to: the mask is not in its input bundle at all."""
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp)
    net = _net(mdp, spec)
    with torch.no_grad():
        a = net(batch)
        # passing an extra, mask-shaped entry must be ignored, not consumed
        b = net({**batch, "mask": torch.ones(batch["moves"].shape[0],
                                             mdp.num_actions, dtype=torch.bool)})
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# AGILE's defining property
# ---------------------------------------------------------------------------


def test_mlp_ablation_removes_cross_action_conditioning():
    """The ablation arm must score each action from its own features + the state
    ONLY — no information from the other actions.

    Asserted the strong way: perturb action 4 and require action 0's logit to be
    bitwise unchanged. If any path leaked between action nodes this would move,
    and the ablation would not be measuring what it claims to.
    """
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp, n=3)
    torch.manual_seed(0)
    net = sat_agile_net(spec, mdp.num_actions, d_model=32, n_layers=1,
                        n_heads=2, inter_action="mlp")
    net.eval()

    with torch.no_grad():
        before = net(batch)
    perturbed = {k: v.clone() for k, v in batch.items()}
    perturbed["moves"][:, 4, 1:] += 0.4
    with torch.no_grad():
        after = net(perturbed)

    assert torch.equal(before[:, 0], after[:, 0]), (
        "action 0 saw a change to action 4 — the mlp arm is still mixing across "
        "actions, so it is not a valid control")
    assert not torch.equal(before[:, 4], after[:, 4]), \
        "action 4's own logit should respond to its own features"


def test_mlp_ablation_is_capacity_matched_on_the_ffn():
    """The control keeps depth and FFN width; it differs only by the attention
    projections. If it dropped the FFNs too, a measured gap could be capacity
    rather than conditioning."""
    mdp = _mdp()
    spec = feature_spec(SATFeaturizer, mdp)
    kw = dict(d_model=32, n_layers=2, n_heads=2, ff=64)
    full = sat_agile_net(spec, mdp.num_actions, inter_action="attention", **kw)
    mlp = sat_agile_net(spec, mdp.num_actions, inter_action="mlp", **kw)

    assert len(full.inter_action_layers) == len(mlp.inter_action_layers) == 2
    ffn_params = lambda net: sum(  # noqa: E731
        p.numel() for blk in net.inter_action_layers
        for p in blk.ffn.parameters())
    assert ffn_params(full) == ffn_params(mlp)

    # the whole difference is the MultiheadAttention projections
    total = lambda net: sum(p.numel() for p in net.parameters())  # noqa: E731
    attn = sum(p.numel() for blk in full.inter_action_layers
               for p in blk.self_attn.parameters())
    assert total(full) - total(mlp) == attn


def test_none_ablation_drops_the_stack_entirely():
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp, n=2)
    net = sat_agile_net(spec, mdp.num_actions, d_model=32, n_layers=1,
                        n_heads=2, inter_action="none")
    assert len(net.inter_action_layers) == 0
    with torch.no_grad():
        out = net(batch)
    assert out.shape == (2, mdp.num_actions) and torch.isfinite(out).all()


def test_inter_action_kwarg_round_trips_as_json():
    """`dp.Net` carries factory kwargs as JSON to cold cluster nodes."""
    import json
    for mode in ("attention", "mlp", "none"):
        assert json.loads(json.dumps({"inter_action": mode}))["inter_action"] == mode
    mdp = _mdp()
    spec = feature_spec(SATFeaturizer, mdp)
    with pytest.raises(AssertionError):
        sat_agile_net(spec, mdp.num_actions, inter_action="bogus")


def test_action_representations_are_conditioned_on_the_other_actions():
    """AGILE's premise, stated as a test: an action's logit depends on which
    OTHER actions are on offer, via the complete-graph attention.

    If this fails, the inter-action stack is not wired in and the net has
    silently degenerated into a per-action scorer — i.e. it would be an
    architecture other than the one we think we are evaluating.
    """
    mdp = _mdp()
    spec, batch, _ = _real_batch(mdp, n=2)
    net = _net(mdp, spec)
    with torch.no_grad():
        before = net(batch)
    perturbed = {k: v.clone() for k, v in batch.items()}
    # change ONLY action 4's features; action 0's logit should still move
    perturbed["moves"][:, 4, 1:] += 0.4
    with torch.no_grad():
        after = net(perturbed)
    assert not torch.allclose(before[:, 0], after[:, 0], atol=1e-6), (
        "action 0's logit ignored a change to action 4 — the inter-action "
        "attention is not contributing")
