"""Simulate one episode and write a self-contained interactive HTML player.

Edit the constants below (which policy, seed, periods, output file), then

    .venv/bin/python visualize.py

and open the HTML file in a browser. No dependencies beyond the model itself
(torch only for a trained agent): the trajectory is embedded as JSON in the
page, rendered as SVG with plain JavaScript (play/pause, step, speed, scrubber).
"""
import json

import dynaplex as dp
from dynaplex.models import RandomPolicy
from dynaplex.modelling import StateCategory

from mdp import MyopicPolicy, VehicleRoutingMDP

# which policy to watch: "myopic", "random", or the path of a trained agent
# directory such as "dynaplex_runs/VehicleRoutingMDP_dcl_.../GridGraphFeaturizer_gin_.../agent_gen1"
POLICY = "myopic"
SEED = 3
PERIODS = 200
OUT = "episode.html"


class AgentPolicy:
    """Drive a trained NNAgent in plain CPython. NNAgent.__call__ does the whole
    round trip itself — features via the featurizer class recorded in the agent
    sidecar (so this works whatever featurizer the agent was trained with), mask
    via write_action_validity, action via masked argmax."""

    def __init__(self, mdp, agent):
        self.mdp = mdp
        self.agent = agent

    def get_action(self, state, context):
        return int(self.agent([state], mdp=self.mdp)[0])


def simulate(mdp, policy, seed, periods):
    """Returns a list of per-period snapshots; index 0 is the initial state."""
    ctx = dp.modelling.new_context(mdp, seed)
    state = mdp.get_initial_state(ctx)

    def snapshot():
        return {
            "vehicles": [[v.row, v.col, v.capacity_left] for v in state.vehicles],
            "orders": [[o.row, o.col, o.weight] for o in state.orders],
            "cum_cost": ctx.cumulative_cost,
        }

    frames = [snapshot()]
    for _ in range(periods):
        while state.category == StateCategory.AWAIT_ACTION:
            mdp.modify_state_with_action(state, ctx, policy.get_action(state, ctx))
        mdp.modify_state_with_event(state, ctx)
        frames.append(snapshot())
    return frames


TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DVRP episode</title>
<style>
  body { font-family: system-ui, sans-serif; background: #fafafa; color: #222;
         display: flex; flex-direction: column; align-items: center; margin: 1.5rem; }
  #status { font-variant-numeric: tabular-nums; margin: .6rem 0; }
  #controls { display: flex; gap: .5rem; align-items: center; margin-bottom: .6rem; }
  button { font-size: 1rem; padding: .2rem .7rem; }
  #scrub { width: 480px; }
  svg { background: #fff; border: 1px solid #ccc; }
</style>
</head>
<body>
<h3>__TITLE__</h3>
<div id="controls">
  <button id="first">&#9198;</button>
  <button id="back">&#9664;</button>
  <button id="play">&#9654;</button>
  <button id="fwd">&#9654;&#9654;</button>
  <input type="range" id="scrub" min="0" value="0">
  <select id="speed">
    <option value="500">slow</option>
    <option value="250" selected>normal</option>
    <option value="100">fast</option>
  </select>
</div>
<div id="status"></div>
<svg id="board"></svg>
<script>
const META = __META__;
const FRAMES = __FRAMES__;
const CELL = 48, PAD = 10, G = META.grid_size;
const SIZE = G * CELL + 2 * PAD;
const svg = document.getElementById("board");
svg.setAttribute("width", SIZE); svg.setAttribute("height", SIZE);
const scrub = document.getElementById("scrub");
scrub.max = FRAMES.length - 1;
let frame = 0, timer = null;

const cx = c => PAD + c * CELL + CELL / 2;
const cy = r => PAD + r * CELL + CELL / 2;

function el(name, attrs, text) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}

function render() {
  svg.replaceChildren();
  for (let i = 0; i <= G; i++) {
    svg.appendChild(el("line", {x1: PAD + i * CELL, y1: PAD, x2: PAD + i * CELL,
                                y2: PAD + G * CELL, stroke: "#e3e3e3"}));
    svg.appendChild(el("line", {x1: PAD, y1: PAD + i * CELL, x2: PAD + G * CELL,
                                y2: PAD + i * CELL, stroke: "#e3e3e3"}));
  }
  svg.appendChild(el("rect", {x: cx(0) - 16, y: cy(0) - 16, width: 32, height: 32,
                              fill: "none", stroke: "#444", "stroke-width": 2}));
  svg.appendChild(el("text", {x: cx(0), y: cy(0) - 18, "text-anchor": "middle",
                              "font-size": 10, fill: "#444"}, "depot"));
  const f = FRAMES[frame];
  for (const [r, c, w] of f.orders) {
    svg.appendChild(el("circle", {cx: cx(c), cy: cy(r), r: 7, fill: "#c62828"}));
    if (w > 1)
      svg.appendChild(el("text", {x: cx(c), y: cy(r) + 3, "text-anchor": "middle",
                                  "font-size": 9, fill: "#fff"}, `${w}`));
  }
  f.vehicles.forEach(([r, c, cap], i) => {
    const off = f.vehicles.length > 1 ? (i - (f.vehicles.length - 1) / 2) * 6 : 0;
    const x = cx(c) + off, y = cy(r) - 4;
    if (frame > 0) {
      const [pr, pc] = FRAMES[frame - 1].vehicles[i];
      if (pr !== r || pc !== c)
        svg.appendChild(el("line", {x1: cx(pc) + off, y1: cy(pr) - 4, x2: x, y2: y,
                                    stroke: "#90a4ae", "stroke-width": 2,
                                    "stroke-dasharray": "3 2"}));
    }
    const color = cap === 0 ? "#ef6c00" : "#1565c0";
    svg.appendChild(el("polygon",
      {points: `${x},${y - 9} ${x - 9},${y + 7} ${x + 9},${y + 7}`, fill: color}));
    svg.appendChild(el("text", {x: x, y: y + 18, "text-anchor": "middle",
                                "font-size": 10, fill: color}, `${i}:${cap}`));
  });
  const cost = f.cum_cost, dcost = frame > 0 ? cost - FRAMES[frame - 1].cum_cost : 0;
  const avg = frame > 0 ? (cost / frame).toFixed(2) : "-";
  document.getElementById("status").textContent =
    `t=${frame}  orders=${f.orders.length}/${META.max_orders}  ` +
    `weight=${f.orders.reduce((s, o) => s + o[2], 0)}  ` +
    `cost this period=${dcost.toFixed(0)}  cumulative=${cost.toFixed(0)}  ` +
    `avg/period=${avg}`;
  scrub.value = frame;
}

function go(i) { frame = Math.max(0, Math.min(FRAMES.length - 1, i)); render(); }
function stop() { if (timer) { clearInterval(timer); timer = null;
                               document.getElementById("play").innerHTML = "&#9654;"; } }
document.getElementById("first").onclick = () => { stop(); go(0); };
document.getElementById("back").onclick = () => { stop(); go(frame - 1); };
document.getElementById("fwd").onclick = () => { stop(); go(frame + 1); };
document.getElementById("play").onclick = () => {
  if (timer) { stop(); return; }
  document.getElementById("play").innerHTML = "&#9208;";
  timer = setInterval(() => {
    if (frame >= FRAMES.length - 1) stop(); else go(frame + 1);
  }, +document.getElementById("speed").value);
};
scrub.oninput = () => { stop(); go(+scrub.value); };
document.getElementById("speed").onchange = () => { if (timer) { stop();
  document.getElementById("play").onclick(); } };
render();
</script>
</body>
</html>
"""


def main():
    mdp = VehicleRoutingMDP()
    if POLICY == "myopic":
        policy, name = MyopicPolicy(mdp), "myopic policy"
    elif POLICY == "random":
        policy, name = RandomPolicy(mdp), "random policy"
    else:
        from dynaplex.nn import NNAgent
        policy, name = AgentPolicy(mdp, NNAgent.load(POLICY)), f"agent [{POLICY}]"
    frames = simulate(mdp, policy, SEED, PERIODS)

    title = (f"{name} — {mdp.grid_size}x{mdp.grid_size}, "
             f"{mdp.n_vehicles} vehicles, seed {SEED}")
    meta = {"grid_size": mdp.grid_size, "max_orders": mdp.max_orders}
    html = (TEMPLATE.replace("__TITLE__", title)
            .replace("__META__", json.dumps(meta))
            .replace("__FRAMES__", json.dumps(frames)))
    with open(OUT, "w") as f:
        f.write(html)
    avg = frames[-1]["cum_cost"] / PERIODS
    print(f"wrote {OUT}: {PERIODS} periods, avg cost/period {avg:.2f} — open it in a browser")


if __name__ == "__main__":
    main()
