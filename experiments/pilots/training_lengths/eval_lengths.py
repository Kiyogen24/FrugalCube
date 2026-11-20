"""
Pilote CPU du choix des longueurs (2026-11-11): ce que chaque réseau a appris, par longueur et par distance.
- précision de l'étiquette des mélanges (inverse du dernier coup) par longueur, sur 4000 mélanges neufs de 26 coups;
- coup optimal connu classé premier, par distance exacte, sur les préfixes des 1000 cubes du jeu de test;
- aire sous la courbe ROC de la valeur entre les distances d et d + 2 sur ces préfixes.
Pour le bandit, la trajectoire de ses probabilités (train_log.jsonl) est résumée par la longueur la plus gardée.
Usage: python experiments/pilots/training_lengths/eval_lengths.py runs/curcpu/uniform_v/final.pt runs/curcpu/focus_v/final.pt ...
"""
import json
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

torch.set_num_threads(16)
cube = ct.TorchCube("cpu")
prefix, dist, best = [], [], []
for scramble in json.load(open(dt.PATH)):
    st = fc.SOLVED
    for j, m in enumerate(scramble.split(), 1):
        a = fc.ACTIONS.index(m)
        st = fc.apply_sequence(st, [a])
        prefix.append(st)
        dist.append(j)
        best.append(fc.INVERSE_ACTION[a])
prefix = cube.from_numpy(np.stack(prefix))
dist, best = np.array(dist), np.array(best)
walks, walk_labels, walk_len = cube.self_supervised_batch(4000, 26, torch.Generator().manual_seed(2026))
walk_len = walk_len.numpy()


def auc(lo, hi):
    both = np.concatenate([lo, hi])
    ranks = np.argsort(np.argsort(both, kind="stable"), kind="stable") + 1.0
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, ranks) / counts)[inv]
    return float((ranks[len(lo):].sum() - len(hi) * (len(hi) + 1) / 2) / (len(lo) * len(hi)))


rows = []
OUT, PATHS = (sys.argv[1], sys.argv[2:]) if sys.argv[1].endswith(".json") else ("runs/curcpu/lengths.json", sys.argv[1:])
for path in PATHS:
    net, meta = nt.load(path)
    h = bt.NetworkHeuristic(net, cube, "ac", 1 << 14, 1.0)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        walk_hit = (h.outputs(walks, "policy").argmax(1) == walk_labels).numpy()
        out = h.head_outputs(prefix, ("policy", "value"))
    hit = out["policy"].argmax(1).numpy() == best
    v = out["value"].float().numpy()
    name = os.path.basename(os.path.dirname(path))
    row = {"model": name, "walk_acc": {k: round(float(walk_hit[walk_len == k].mean()), 4) for k in range(1, 27)},
           "top1": {d: round(float(hit[dist == d].mean()), 4) for d in range(1, 23) if (dist == d).any()},
           "auc": {d: round(auc(v[dist == d], v[dist == d + 2]), 4) for d in range(4, 19)}}
    trunk_log = os.path.join(os.path.dirname(os.path.dirname(path)), name.removesuffix("_v"), "train_log.jsonl")
    if os.path.exists(trunk_log):
        log = [json.loads(line) for line in open(trunk_log)]
        row["examples"] = log[-1]["examples"]
        if "keep" in log[-1]:
            row["bandit_peak"] = [(round(r["time_s"] / 60), int(np.argmax(r["keep"])) + 1) for r in log if "keep" in r]
            row["bandit_keep_end"] = log[-1]["keep"]
    rows.append(row)
    print(f"{name:10} {row.get('examples', 0) / 1e6:6.0f}M ex. | walk acc k=" + " ".join(
        f"{k}:{row['walk_acc'][k]:.2f}" for k in (4, 6, 8, 10, 12, 14, 16, 18, 20)) + " | top-1 d=" + " ".join(
        f"{d}:{row['top1'][d]:.2f}" for d in (6, 8, 10, 12, 14, 16)) + " | AUC d=" + " ".join(
        f"{d}:{row['auc'][d]:.2f}" for d in (6, 8, 10, 12, 14)), flush=True)
    if "bandit_peak" in row:
        print(f"{'':10} bandit, most kept length by minute: {row['bandit_peak'][::5]}", flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
