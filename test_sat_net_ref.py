"""Parity tests for the reference AGILE nets (sat_net_ref), ported from
dynaplex2_playground `tests/test_batched_forward.py`.

Why this survives into Phase 2. `sat_net_ref` is the ORACLE, not the production
net: the engine-side net consumes dense padded slabs and will delete most of the
masking these tests guard (the block-diagonal within-group layer collapses to an
MLP at group size 1, and a complete action set needs no padding bias at all).
This file is what tells you that deletion did not change the math — it pins the
batched path against the single-sample path, which is the semantics the dense
rewrite has to reproduce.

Two batch shapes are exercised deliberately:

  * REAL VRP observations, where every sample has G == mdp.num_actions. This is
    the case Phase 2 optimises for, and it is uniform — so on its own it never
    touches the padding code.
  * SYNTHETIC variable-G observations. VRP cannot produce these (its action
    count is fixed), yet the padded batched path is full of machinery that only
    runs when G differs across the batch: `max_G`, `valid_g`, the -inf key bias,
    and the `max_G + 1` dummy pooling bin that absorbs padded tokens. Testing
    parity only on uniform batches would leave all of it unverified.
"""
from dataclasses import dataclass

import pytest
import torch

from sat_compat import make_context

from mdp import MyopicPolicy, VehicleRoutingMDP
from sat_markers import Action, GlobalState
from sat_net_ref import (
    build_transformer_policy_net,
    build_transformer_value_net,
)
from sat_schema import build_sat_observation
from sat_tokenizer import SequenceBuilder

ATOL = 1e-4


# ---------------------------------------------------------------------------
# Fixtures: two kinds of batch
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Ctx(GlobalState):
    a: float
    b: float


@dataclass(slots=True)
class _Act(Action):
    qty: float


@dataclass(slots=True)
class _Obs:
    state: _Ctx
    actions: list[_Act]


def _varying_g_seqs(n=4):
    """Synthetic batch whose action counts differ per sample (3, 4, 5, ...)."""
    sb = SequenceBuilder(skip_root=True)
    return sb, [
        sb.build(_Obs(state=_Ctx(a=float(i + 1), b=float(i) * 0.5),
                      actions=[_Act(qty=float(q)) for q in range(3 + i)]))
        for i in range(n)
    ]


def _vrp_seqs(n=4):
    """Real VRP observations — uniform G, the Phase 2 case."""
    mdp = VehicleRoutingMDP(grid_size=6, n_vehicles=3, capacity=3, max_orders=12)
    ctx = make_context(mdp, 0)
    policy = MyopicPolicy(mdp)
    state = mdp.get_initial_state(ctx)
    sb = SequenceBuilder(skip_root=True)
    seqs = []
    while len(seqs) < n:
        seqs.append(sb.build(build_sat_observation(state, mdp)))
        mdp.modify_state_with_action(state, ctx, policy.get_action(state, ctx))
        while state.category.name == "AWAIT_EVENT":
            mdp.modify_state_with_event(state, ctx)
    return sb, seqs


BATCHES = {"varying_g": _varying_g_seqs, "vrp_uniform_g": _vrp_seqs}


def _policy_for(sb, seed=42):
    torch.manual_seed(seed)
    net = build_transformer_policy_net(raw_dims=sb.raw_dims,
                                       type_names=sb.type_names,
                                       d_model=32, nhead=2, num_layers=1)
    net.eval()
    return net


def _value_for(sb, seed=42):
    torch.manual_seed(seed)
    net = build_transformer_value_net(raw_dims=sb.raw_dims,
                                      type_names=sb.type_names,
                                      d_model=32, nhead=2, num_layers=1)
    net.eval()
    return net


def _single(net, seq):
    return net.forward_logits(
        type_features=seq["type_features"],
        type_ids=seq["type_ids"],
        action_mask=seq["action_mask"],
        indices_by_type=seq["indices_by_type"],
        action_group_ids=seq.get("action_group_ids"),
        num_action_groups=seq.get("num_action_groups"),
        intra_group_pos=seq.get("intra_group_pos"),
    )


# ---------------------------------------------------------------------------
# Policy net
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", BATCHES)
def test_batched_policy_matches_single_sample(batch):
    sb, seqs = BATCHES[batch]()
    net = _policy_for(sb)
    with torch.no_grad():
        expected = [_single(net, s) for s in seqs]
        flat, offsets, lengths = net.forward_logits_batched(seqs)

    assert offsets.shape[0] == len(seqs)
    for i, want in enumerate(expected):
        off, L = int(offsets[i]), int(lengths[i])
        assert L == want.shape[0]
        assert torch.allclose(want, flat[off:off + L], atol=ATOL), (
            f"sample {i}: batched != single\n want {want}\n got  {flat[off:off + L]}")


@pytest.mark.parametrize("batch", BATCHES)
def test_offsets_and_lengths_tile_the_flat_tensor(batch):
    sb, seqs = BATCHES[batch]()
    net = _policy_for(sb)
    with torch.no_grad():
        flat, offsets, lengths = net.forward_logits_batched(seqs)
    assert flat.shape[0] == int(lengths.sum())
    cum = 0
    for i in range(offsets.shape[0]):
        assert int(offsets[i]) == cum
        cum += int(lengths[i])


@pytest.mark.parametrize("batch", BATCHES)
def test_policy_logits_are_finite(batch):
    """No NaN/inf may reach the caller.

    The masked-softmax rows and the fully-blocked singleton groups inside the
    net legitimately produce NaN internally; the `nan_to_num` guards are what
    keep them out of the output. A NaN escaping here would poison the masked
    argmax that selects actions downstream.
    """
    sb, seqs = BATCHES[batch]()
    net = _policy_for(sb)
    with torch.no_grad():
        flat, _, _ = net.forward_logits_batched(seqs)
    assert torch.isfinite(flat).all()


@pytest.mark.parametrize("batch", BATCHES)
def test_policy_gradients_reach_every_projection(batch):
    """Stronger than upstream's "at least one parameter": every per-type input
    projection must receive gradient, or a token type is silently inert.

    Checks weight OR bias, not weight alone. A token type whose raw features are
    identically zero has a structurally dead weight gradient
    (dL/dW = delta (x) input) while still influencing the output through its
    bias — see test_depot_token_is_a_constant_zero_vector, which is the VRP case.
    """
    sb, seqs = BATCHES[batch]()
    net = _policy_for(sb)
    net.train()
    flat, _, _ = net.forward_logits_batched(seqs)
    flat.sum().backward()

    present = set(sb.type_names)
    for tname, proj in net.input_projs.items():
        if tname not in present:
            continue
        assert proj.weight.grad is not None, f"{tname}: no weight grad tensor"
        assert proj.bias.grad is not None, f"{tname}: no bias grad tensor"
        total = (proj.weight.grad.abs().sum() + proj.bias.grad.abs().sum())
        assert total > 0, f"{tname}: token type is inert (no gradient at all)"


def test_depot_token_is_a_constant_zero_vector():
    """Documents a real property of this MDP rather than hiding it.

    `mdp.depot` is fixed at (0, 0), so the depot token normalises to (0.0, 0.0)
    on EVERY state of EVERY sample. Consequences worth knowing before Phase 2:

      * its input projection's weight gradient is identically zero — the weight
        can never learn, only the bias can, so the token reduces to a learned
        constant (an anchor other tokens can attend to);
      * it carries no state information whatsoever, so the [1, 2] depot slab is
        pure overhead in the engine bundle.

    It is kept anyway because `featurizers_tokens.py` includes exactly the same
    token, and Phase 2 is benchmarked against those runs — dropping it would
    change the comparison. Revisit once the baseline is reproduced.
    """
    mdp = VehicleRoutingMDP(grid_size=6, n_vehicles=3, capacity=3, max_orders=12)
    ctx = make_context(mdp, 0)
    obs = build_sat_observation(mdp.get_initial_state(ctx), mdp)
    assert (obs.depot.row, obs.depot.col) == (0.0, 0.0)

    sb = SequenceBuilder(skip_root=True)
    r = sb.build(obs)
    assert torch.equal(r["type_features"]["DepotTok"], torch.zeros(1, 2))

    net = _policy_for(sb)
    net.train()
    flat, _, _ = net.forward_logits_batched([r])
    flat.sum().backward()
    proj = net.input_projs["DepotTok"]
    assert proj.weight.grad.abs().sum() == 0.0   # structurally dead
    assert proj.bias.grad.abs().sum() > 0.0      # but the token still matters


def test_policy_empty_batch():
    sb, _ = _varying_g_seqs()
    net = _policy_for(sb)
    flat, offsets, lengths = net.forward_logits_batched([])
    assert flat.numel() == 0 and offsets.numel() == 0 and lengths.numel() == 0


def test_policy_single_sample_batch():
    sb, seqs = _varying_g_seqs()
    net = _policy_for(sb)
    with torch.no_grad():
        flat, offsets, lengths = net.forward_logits_batched([seqs[0]])
        want = _single(net, seqs[0])
    assert offsets.shape[0] == 1 and int(offsets[0]) == 0
    assert torch.allclose(want, flat[:int(lengths[0])], atol=ATOL)


def test_policy_batch_with_a_zero_action_sample():
    """A sample with no actions must contribute length 0 and not disturb its
    neighbours — the trainer relies on skipping those slots by length."""
    sb, seqs = _varying_g_seqs()
    net = _policy_for(sb)
    empty = sb.build(_Obs(state=_Ctx(a=1.0, b=2.0), actions=[]))
    mixed = [seqs[0], empty, seqs[1]]

    with torch.no_grad():
        flat, offsets, lengths = net.forward_logits_batched(mixed)
        want0 = _single(net, seqs[0])
        want2 = _single(net, seqs[1])

    assert int(lengths[1]) == 0
    assert torch.allclose(want0, flat[int(offsets[0]):int(offsets[0]) + int(lengths[0])],
                          atol=ATOL)
    assert torch.allclose(want2, flat[int(offsets[2]):int(offsets[2]) + int(lengths[2])],
                          atol=ATOL)


# ---------------------------------------------------------------------------
# Value net
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", BATCHES)
def test_batched_value_matches_single_sample(batch):
    """The single-sample path must pass intra_group_pos, like forward_batched
    does — omitting it silently desynchronises the two (a real past bug, noted
    in _seq_to_value's docstring)."""
    sb, seqs = BATCHES[batch]()
    net = _value_for(sb)
    with torch.no_grad():
        expected = [
            net(type_features=s["type_features"], type_ids=s["type_ids"],
                indices_by_type=s["indices_by_type"],
                action_mask=s.get("action_mask"),
                intra_group_pos=s.get("intra_group_pos")).item()
            for s in seqs
        ]
        got = net.forward_batched(seqs)
    assert got.shape[0] == len(seqs)
    for i, want in enumerate(expected):
        assert got[i].item() == pytest.approx(want, abs=ATOL)


@pytest.mark.parametrize("batch", BATCHES)
def test_value_gradients_flow(batch):
    sb, seqs = BATCHES[batch]()
    net = _value_for(sb)
    net.train()
    net.forward_batched(seqs).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.parameters())


def test_value_empty_batch():
    sb, _ = _varying_g_seqs()
    net = _value_for(sb)
    assert net.forward_batched([]).numel() == 0
