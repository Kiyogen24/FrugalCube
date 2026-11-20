"""
Le coup préféré du réseau rapproche-t-il le cube? Une étiquette juste n'est pas la seule bonne réponse: après deux
coups qui commutent, les deux inverses rapprochent, donc la précision top-1 sur l'étiquette de la marche plafonne sous
100 %. Mêmes 400 marches que walk_labels.py (même graine, même règle; leurs distances exactes sont lues dans
walk_labels.json et vérifiées à nouveau ici près du but). Pour chaque préfixe s_j à distance d, le coup préféré a
donne l'enfant s_j·a, à distance d - 1 ou d + 1 (graphe biparti):
- si a est l'étiquette de la marche, l'enfant est s_{j-1}, dont la distance est connue;
- sinon l'enfant est plus proche si et seulement si un état à au plus d - 1 - D coups de lui est dans la table exacte
  de rayon D = 8 (l'enfant lui-même quand d - 1 <= D).
Usage: python experiments/analysis/top_move.py   (résultats dans runs/horizon/top_move.json)
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
sys.path.insert(0, "experiments/cube")
import beam_torch as bt
import cube_torch as ct
import search as lab
import network_torch as nt

parser = argparse.ArgumentParser()
parser.add_argument("--labels", default="runs/horizon/walk_labels.json")
parser.add_argument("--table", type=int, default=8)
parser.add_argument("--out", default="runs/horizon/top_move.json")
args = parser.parse_args()

torch.set_num_threads(16)
CHUNK = 1 << 15
prev = json.load(open(args.labels))
pa = prev["args"]
cube = ct.TorchCube("cpu")
data = torch.load(f"data/ball{args.table}.pt")
ball = lab.Ball.__new__(lab.Ball)
ball.cube, ball.keys, ball.dist = cube, data["keys"], data["dist"]
D = args.table


def near_table(state, radius):
    """vrai si un état à au plus radius >= 1 coups de state est dans la table."""
    frontier = state[None]
    for r in range(1, radius + 1):
        keys, kept = [], []
        for s in range(0, len(frontier), CHUNK):
            kids = cube.children(frontier[s:s + CHUNK]).reshape(-1, 54)
            k = cube.state_hash(kids)
            if ball.lookup(k)[0].any():
                return True
            if r < radius:
                k, first = ball.first_of_each(k)
                keys.append(k)
                kept.append(kids[first])
        if r < radius:
            _, first = ball.first_of_each(torch.cat(keys))
            frontier = torch.cat(kept)[first]
    return False


states, actions = cube.random_walks(pa["walks"], pa["length"], torch.Generator().manual_seed(pa["seed"]))
dist = np.array(prev["dist"])
# les marches sont les mêmes: chaque préfixe à au plus D coups du but doit avoir la distance enregistrée
for j in range(1, pa["length"] + 1):
    found, exact = ball.lookup(cube.state_hash(states[j - 1]))
    f = found.numpy()
    assert (exact.numpy()[f] == dist[j][f]).all() and (dist[j][~f] > D).all(), f"walks differ at length {j}"
print("same walks and distances as walk_labels.json", flush=True)

net, _ = nt.load(pa["model"])
h = bt.NetworkHeuristic(net, cube, "ac", 1 << 14, 1.0)
with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
    top = h.outputs(states.reshape(-1, 54), "policy").argmax(1).reshape(pa["length"], pa["walks"])
label = cube.inverse_action[actions]
improves = np.zeros((pa["length"], pa["walks"]), dtype=bool)
t0 = time.time()
rows = []
for j in range(1, pa["length"] + 1):
    for w in range(pa["walks"]):
        d = dist[j, w]
        if top[j - 1, w] == label[j - 1, w]:
            improves[j - 1, w] = dist[j - 1, w] < d
            continue
        child = cube.children(states[j - 1, w][None])[0, top[j - 1, w]]
        if d - 1 <= D:
            found, exact = ball.lookup(cube.state_hash(child[None]))
            improves[j - 1, w] = bool(found[0]) and int(exact[0]) == d - 1
        else:
            improves[j - 1, w] = near_table(child, d - 1 - D)
    hit = (top[j - 1] == label[j - 1]).numpy()
    rows.append({"length": j, "walks": pa["walks"], "net_top1_label": float(hit.mean()),
                 "net_top_improves": float(improves[j - 1].mean()),
                 "improving_not_label": float((improves[j - 1] & ~hit).mean())})
    r = rows[-1]
    print(f"k={j:2} | top move = label {r['net_top1_label']:6.1%} | top move reduces the distance "
          f"{r['net_top_improves']:6.1%} (another improving move {r['improving_not_label']:5.1%}) | "
          f"{time.time() - t0:.0f} s", flush=True)
    json.dump({"args": vars(args), "labels_args": pa, "rows": rows}, open(args.out, "w"))
print("done", flush=True)
