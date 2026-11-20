"""
les étiquettes des marches sont-elles exactes?
Pour des marches tirées par la règle d'entraînement (cube.random_walks), distance exacte de chaque préfixe de
longueur <= --length, par rencontre au milieu avec la table des états à <= D coups. Le graphe est biparti, donc
d(s_j) = d(s_{j-1}) ± 1: le coup j a rapproché le cube si et seulement si un état à <= d(s_{j-1}) - 1 - D coups
de s_j est dans la table.
Par longueur k:
- part des marches qui sont des plus courts chemins (d = k), et écart moyen k - d (erreur de la cible de valeur);
- part des étiquettes de politique qui sont un coup améliorant (le dernier coup a éloigné le cube);
- réussite du réseau sur l'étiquette, sur toutes les marches et sur celles dont l'étiquette est juste.
Usage: python experiments/analysis/walk_labels.py --walks 400 --length 16
"""
import argparse
import json
import os
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
parser.add_argument("--walks", type=int, default=400)
parser.add_argument("--length", type=int, default=16)
parser.add_argument("--table", type=int, default=7, help="rayon D de la table exacte")
parser.add_argument("--model", default="runs/s4/m90_v/final.pt")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="runs/horizon/walk_labels.json")
args = parser.parse_args()

torch.set_num_threads(16)
CHUNK = 1 << 15   # parents par lot: l'empreinte passe par des int64, 12 x 54 x 8 octets par parent
cube = ct.TorchCube("cpu")
ball = lab.Ball.__new__(lab.Ball)
ball.cube = cube
path = f"data/ball{args.table}.pt"
if os.path.exists(path):
    data = torch.load(path)
    ball.keys, ball.dist = data["keys"], data["dist"]
else:
    ball.keys, ball.dist = ball.build(args.table, chunk=CHUNK)
    torch.save({"keys": ball.keys, "dist": ball.dist}, path)


def near_table(state, radius):
    """vrai si un état à <= radius coups de state (radius >= 1) est dans la table."""
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


generator = torch.Generator().manual_seed(args.seed)
states, actions = cube.random_walks(args.walks, args.length, generator)   # (length, walks, 54), (length, walks)
dist = np.zeros((args.length + 1, args.walks), dtype=np.int64)
t0 = time.time()
for j in range(1, args.length + 1):
    found, exact = ball.lookup(cube.state_hash(states[j - 1]))
    for w in range(args.walks):
        d = dist[j - 1, w]
        if found[w]:
            dist[j, w] = int(exact[w])
        elif d - 1 - args.table <= 0:
            dist[j, w] = d + 1
        else:
            dist[j, w] = d - 1 if near_table(states[j - 1, w], d - 1 - args.table) else d + 1
    assert (np.abs(dist[j] - dist[j - 1]) == 1).all(), "graphe biparti : la distance change de 1 à chaque coup"
    print(f"longueur {j} : {np.mean(dist[j] == j):.1%} de plus courts chemins, {time.time() - t0:.0f} s", flush=True)

net, _ = nt.load(args.model)
h = bt.NetworkHeuristic(net, cube, "ac", 1 << 14, 1.0)
with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
    top = h.outputs(states.reshape(-1, 54), "policy").argmax(1).reshape(args.length, args.walks)
hit = (top == cube.inverse_action[actions]).numpy()
outward = dist[1:] > dist[:-1]   # le dernier coup a éloigné le cube: son inverse est un coup améliorant
rows = []
for j in range(1, args.length + 1):
    o = outward[j - 1]
    rows.append({"length": j, "walks": args.walks, "shortest": float(np.mean(dist[j] == j)),
                 "gap_mean": float(np.mean(j - dist[j])), "label_improving": float(o.mean()),
                 "net_top1": float(hit[j - 1].mean()), "net_top1_label_improving": float(hit[j - 1][o].mean())})
    r = rows[-1]
    print(f"k={j:2} | plus court chemin {r['shortest']:6.1%} | k - d moyen {r['gap_mean']:.2f} | étiquette "
          f"améliorante {r['label_improving']:6.1%} | réseau {r['net_top1']:6.1%} (étiquette juste : "
          f"{r['net_top1_label_improving']:6.1%})", flush=True)
json.dump({"args": vars(args), "rows": rows, "dist": dist.tolist()}, open(args.out, "w"))
