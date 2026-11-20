"""
Évaluation des bras de finetune.py sur les cubes impairs du jeu DeepCubeA (jamais vus, même par l'oracle):
- valeur: séparation de d et d + 2 (aire sous la courbe ROC) et écart quadratique à d, sur les préfixes;
- politique: entropie croisée sur des marches neuves (graine fixe) et part du coup optimal en tête;
- recherche: acteur-critique + vues de Kociemba, largeur --width, sur les --cases premiers cubes impairs,
  une ligne par cube dans runs/floor/search/<nom>.jsonl, appariée avec le premier modèle de la liste.
Usage: python experiments/pilots/value_floor/evaluate.py runs/s4/m90_v/final.pt runs/floor/ss.pt runs/floor/exit_drop.pt ...
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
import beam_torch as bt
import cube_torch as ct
import deepcubea_testset as dt
import fast_cube as fc
import network_torch as nt

parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="+")
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--cases", type=int, default=300)
args = parser.parse_args()

torch.set_num_threads(16)
VIEWS = ["id", "inv", "c15", "c20"]
cube = ct.TorchCube("cpu")
os.makedirs("runs/floor/search", exist_ok=True)

scrambles = json.load(open(dt.PATH))
prefix, dist, best, case = [], [], [], []
for i, scramble in enumerate(scrambles):
    st = fc.SOLVED
    for j, m in enumerate(scramble.split(), 1):
        a = fc.ACTIONS.index(m)
        st = fc.apply_sequence(st, [a])
        prefix.append(st)
        dist.append(j)
        best.append(fc.INVERSE_ACTION[a])
        case.append(i)
prefix = cube.from_numpy(np.stack(prefix))
dist, best, odd = np.array(dist), np.array(best), np.array(case) % 2 == 1
walk_states, walk_labels, _ = cube.self_supervised_batch(5000, 26, torch.Generator().manual_seed(123))
CASES = [(cube.from_numpy(s), o, i) for i, (s, o) in enumerate(dt.load()) if i % 2][:args.cases]


def auc(lo, hi):
    both = np.concatenate([lo, hi])
    ranks = np.argsort(np.argsort(both, kind="stable"), kind="stable") + 1.0
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, ranks) / counts)[inv]
    return float((ranks[len(lo):].sum() - len(hi) * (len(hi) + 1) / 2) / (len(lo) * len(hi)))


def paired(a, b):
    d = np.array([y["length"] - x["length"] for x, y in zip(a, b)], dtype=float)
    z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std() > 0 else 0.0
    ob = sum(y["length"] == y["optimal"] and x["length"] != x["optimal"] for x, y in zip(a, b))
    oa = sum(x["length"] == x["optimal"] and y["length"] != y["optimal"] for x, y in zip(a, b))
    return f"{d.mean():+.3f} coups (z {z:+.1f}), optimales {ob}/{oa}"


reference = None
for path in args.models:
    name = os.path.splitext(os.path.basename(path))[0] if "runs/floor" in path else "base"
    net, _ = nt.load(path)
    h = bt.NetworkHeuristic(net, cube, "ac", 1 << 14, 1.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v = h.outputs(prefix, "value").numpy()
        lp = torch.log_softmax(h.outputs(walk_states, "policy"), dim=1)
        pp = h.outputs(prefix, "policy").numpy()
    ce = float(-lp.gather(1, walk_labels[:, None]).mean())
    ds = range(1, 21)
    aucs = [auc(v[odd & (dist == d)], v[odd & (dist == d + 2)]) for d in ds]
    mse = [float(np.mean((v[odd & (dist == d)] - d) ** 2)) for d in ds]
    hit = pp.argmax(1) == best
    top1 = [float(hit[odd & (dist == d)].mean()) for d in ds]
    top1_even = [float(hit[~odd & (dist == d)].mean()) for d in ds]
    lpp = torch.log_softmax(torch.as_tensor(pp).float(), dim=1).numpy()[np.arange(len(best)), best]
    opt_ce = [float(-lpp[odd & (dist == d)].mean()) for d in ds]
    out = f"runs/floor/search/{name}_w{args.width}.jsonl"
    rows = [json.loads(line) for line in open(out)] if os.path.exists(out) else []
    with open(out, "a") as f:
        for state, optimal, i in CASES[len(rows):]:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                r = bt.solve(h, state, width=args.width, views=VIEWS)
            row = {"i": i, "length": len(r["solution"]), "optimal": optimal, "time_s": r["time_s"]}
            f.write(json.dumps(row) + "\n")
            rows.append(row)
    reference = reference or rows
    print(f"{name:12} politique {ce:.4f} | AUC d=10,12,14,15,16,18 : "
          + " ".join(f"{aucs[d - 1]:.3f}" for d in (10, 12, 14, 15, 16, 18))
          + " | écart² d=10,14,18 : " + " ".join(f"{mse[d - 1]:5.2f}" for d in (10, 14, 18))
          + " | top-1 d=10,12,14,15,16,18 : " + " ".join(f"{top1[d - 1]:.3f}" for d in (10, 12, 14, 15, 16, 18))
          + " (pairs d=14,15,16 : " + " ".join(f"{top1_even[d - 1]:.3f}" for d in (14, 15, 16)) + ")"
          + " | -log π(optimal) d=14,16 : " + " ".join(f"{opt_ce[d - 1]:.3f}" for d in (14, 16))
          + f" | recherche {np.mean([r['length'] == r['optimal'] for r in rows]):.1%} "
            f"{np.mean([r['length'] for r in rows]):.3f}"
          + ("" if rows is reference else f" ; contre base {paired(reference, rows)}"), flush=True)
    res = json.load(open("runs/floor/eval.json")) if os.path.exists("runs/floor/eval.json") else {}
    res[name] = {"path": path, "policy_ce": ce, "auc": aucs, "mse": mse, "top1": top1, "top1_even": top1_even,
                 "opt_ce": opt_ce}
    json.dump(res, open("runs/floor/eval.json", "w"))
