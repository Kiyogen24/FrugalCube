"""
test d'ajustement (2026-11-06): le réseau entier, entraîné dix fois plus longtemps sur les distances exactes des
préfixes des cubes pairs, sépare-t-il mieux d et d + 2 sur ces cubes mêmes (entraînement) et sur les cubes impairs
(jamais vus)?
- les deux montent: les essais de 3000 pas étaient sous-entraînés;
- l'entraînement monte, pas le test: trop peu d'étiquettes (10 307 états);
- aucun ne monte: limite de représentation du réseau.
Une ligne JSON par modèle dans runs/floor/fit/fit_eval.jsonl.
Usage: python experiments/pilots/value_floor/fit_eval.py runs/s4/m90_v/final.pt runs/floor/fit/oracle30k_s3000.pt ...
"""
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
sys.path.insert(0, "experiments/pilots/value_floor")
import beam_torch as bt
import cube_torch as ct
import deepcubea_testset as dt
import fast_cube as fc
import network_torch as nt

torch.set_num_threads(16)
cube = ct.TorchCube("cpu")
prefix, dist, case = [], [], []
for i, scramble in enumerate(json.load(open(dt.PATH))):
    st = fc.SOLVED
    for j, m in enumerate(scramble.split(), 1):
        st = fc.apply_sequence(st, [fc.ACTIONS.index(m)])
        prefix.append(st)
        dist.append(j)
        case.append(i)
prefix = cube.from_numpy(np.stack(prefix))
dist, odd = np.array(dist), np.array(case) % 2 == 1


def auc(lo, hi):
    """Probabilité qu'un état de hi soit jugé plus loin qu'un état de lo (ex aequo comptés à moitié)."""
    both = np.concatenate([lo, hi])
    ranks = np.argsort(np.argsort(both, kind="stable"), kind="stable") + 1.0
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, ranks) / counts)[inv]
    return float((ranks[len(lo):].sum() - len(hi) * (len(hi) + 1) / 2) / (len(lo) * len(hi)))


os.makedirs("runs/floor/fit", exist_ok=True)
DS = (10, 12, 13, 14, 15, 16, 17, 18)
for path in sys.argv[1:]:
    net, meta = nt.load(path)
    h = bt.NetworkHeuristic(net, cube, "ac", 1 << 14, 1.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v = h.outputs(prefix, "value").numpy()
    row = {"model": path, "steps": meta.get("steps")}
    for name, mask in (("train", ~odd), ("test", odd)):
        row[f"auc_{name}"] = {d: round(auc(v[mask & (dist == d)], v[mask & (dist == d + 2)]), 4) for d in DS}
        row[f"mse_{name}"] = {d: round(float(np.mean((v[mask & (dist == d)] - d) ** 2)), 2) for d in (10, 14, 18)}
    print(f"{os.path.basename(path):24} pas {row['steps']} | AUC entraînement d=" + " ".join(
        f"{d}:{row['auc_train'][d]:.3f}" for d in DS) + " | test " + " ".join(
        f"{d}:{row['auc_test'][d]:.3f}" for d in DS) + f" | écart² d14 {row['mse_train'][14]} / {row['mse_test'][14]}",
        flush=True)
    with open("runs/floor/fit/fit_eval.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
