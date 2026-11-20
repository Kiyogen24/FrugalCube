"""
sonde à étiquettes exactes: les caractéristiques du tronc figé contiennent-elles de quoi séparer d de d + 2
au-delà de 14 coups? Une tête de la même forme que la tête valeur (512 -> 256 -> 1) est entraînée sur les
distances exactes des préfixes du jeu DeepCubeA (solutions optimales retournées), augmentées par les 48
symétries et l'inverse (même distance). Validation croisée à 2 plis par cube: chaque cube est évalué par une
sonde qui ne l'a jamais vu. Comparée à la tête valeur du modèle, entraînée sur la longueur des marches.
Si la sonde sépare nettement mieux, le plancher vient de la cible et non du tronc: une tête suffit.
Usage: python experiments/pilots/value_floor/probe.py [modèle.pt]   (résultats dans runs/floor/probe.json)
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
import cube_torch as ct
import deepcubea_testset as dt
import fast_cube as fc
import network_torch as nt

torch.set_num_threads(16)
torch.manual_seed(0)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/s4/m90_v/final.pt"
EPOCHS, BATCH = 4, 4096
cube = ct.TorchCube("cpu")
net, _ = nt.load(MODEL)
net.requires_grad_(False)

prefix, dist, case = [], [], []
for i, scramble in enumerate(json.load(open(dt.PATH))):
    st = fc.SOLVED
    for j, m in enumerate(scramble.split(), 1):
        st = fc.apply_sequence(st, [fc.ACTIONS.index(m)])
        prefix.append(st)
        dist.append(j)
        case.append(i)
prefix = cube.from_numpy(np.stack(prefix))
dist, case = np.array(dist), np.array(case)


def auc(lo, hi):
    """P(V(hi) > V(lo)) + 0,5 P(égalité)."""
    both = np.concatenate([lo, hi])
    ranks = np.argsort(np.argsort(both, kind="stable"), kind="stable") + 1.0
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, ranks) / counts)[inv]
    return float((ranks[len(lo):].sum() - len(hi) * (len(hi) + 1) / 2) / (len(lo) * len(hi)))


@torch.no_grad()
def features(states):
    out = []
    for s in states.split(16384):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out.append(net.body(cube.encode_stickers(s)).float())
    return torch.cat(out)


def augment(states):
    """les 48 conjugués de chaque état et de son inverse: (N, 54) -> (96 N, 54)."""
    both = torch.cat([states, cube.inverse(states)])
    return cube.conjugates(both).reshape(-1, 54)


def per_d_auc(v):
    return [auc(v[dist == d], v[dist == d + 2]) for d in range(1, int(dist.max()) + 1) if (dist == d + 2).any()]


t0 = time.time()
with torch.no_grad():
    head_v = net.heads["value"](features(prefix)).squeeze(-1).numpy()
probe_v = np.zeros(len(dist))
train_err = []
for fold in (0, 1):
    train = (case % 2) != fold          # cubes pairs / impairs
    states = augment(prefix[torch.as_tensor(train)])
    target = torch.as_tensor(np.tile(np.repeat(dist[train], 48), 2), dtype=torch.float32)
    feats = features(states)
    probe = nn.Sequential(nn.Linear(feats.shape[1], 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    steps = EPOCHS * len(feats) // BATCH
    for step in range(steps):
        idx = torch.randint(len(feats), (BATCH,))
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 * (1 + np.cos(np.pi * step / steps))
        loss = ((probe(feats[idx]).squeeze(-1) - target[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        test = ~train
        probe_v[test] = probe(features(prefix[torch.as_tensor(test)])).squeeze(-1).numpy()
        tr = probe(features(prefix[torch.as_tensor(train)])).squeeze(-1).numpy()
        train_err.append(float(np.mean((tr - dist[train]) ** 2)))
    print(f"pli {fold} : {len(feats)} exemples, {steps} pas, erreur entraînement {train_err[-1]:.2f}, "
          f"{time.time() - t0:.0f} s", flush=True)

res = {"model": MODEL, "head_auc": per_d_auc(head_v), "probe_auc": per_d_auc(probe_v),
       "head_mse": [float(np.mean((head_v[dist == d] - d) ** 2)) for d in range(1, int(dist.max()) + 1)],
       "probe_mse": [float(np.mean((probe_v[dist == d] - d) ** 2)) for d in range(1, int(dist.max()) + 1)],
       "probe_train_mse": train_err}
json.dump(res, open("runs/floor/probe.json", "w"))
ds = [8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]
print("AUC d/d+2      " + " ".join(f"{d:5}" for d in ds))
print("tête (marches) " + " ".join(f"{res['head_auc'][d - 1]:.3f}" for d in ds))
print("sonde (exacte) " + " ".join(f"{res['probe_auc'][d - 1]:.3f}" for d in ds))
print("écart quadratique à d, tête  " + " ".join(f"{res['head_mse'][d - 1]:5.2f}" for d in ds))
print("écart quadratique à d, sonde " + " ".join(f"{res['probe_mse'][d - 1]:5.2f}" for d in ds))
