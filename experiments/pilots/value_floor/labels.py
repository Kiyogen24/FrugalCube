"""
étiquettes de distance par la recherche (itération experte, à la Bootstrap d'Arfaee, Zilles et Holte 2011),
ciblées sur les positions profondes où l'auto-supervision plafonne.
Des positions sont tirées au bout de marches de 12 à 26 coups et résolues par le faisceau acteur-critique +
vues de Kociemba avec le réseau courant. Chaque état du chemin trouvé reçoit pour cible la longueur restante
(borne supérieure de sa distance); le départ garde le minimum de cette longueur et de celle de la marche
(deux bornes supérieures).
Étalonnage d'abord: la même recherche depuis des préfixes du jeu DeepCubeA (distance exacte d) donne l'erreur
de ces cibles par distance.
Usage: python experiments/pilots/value_floor/labels.py --minutes 30 --width 2048 --out runs/floor/labels_w2048.pt
"""
import argparse
import json
import sys
import time

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
parser.add_argument("--model", default="runs/s4/m90_v/final.pt")
parser.add_argument("--minutes", type=float, default=30)
parser.add_argument("--width", type=int, default=2048)
parser.add_argument("--lam", type=float, default=1.0)
parser.add_argument("--k-min", type=int, default=12)
parser.add_argument("--k-max", type=int, default=26)
parser.add_argument("--calibrate", type=int, default=30, help="préfixes par distance pour l'étalonnage (0 : sans)")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", required=True)
args = parser.parse_args()

torch.set_num_threads(16)
VIEWS = ["id", "inv", "c15", "c20"]
cube = ct.TorchCube("cpu")
net, _ = nt.load(args.model)
h = bt.NetworkHeuristic(net, cube, "ac", 1 << 14, args.lam)


def solve(state):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return bt.solve(h, state, width=args.width, views=VIEWS)


calibration = {}
if args.calibrate:
    t0 = time.time()
    scrambles = [s.split() for s in json.load(open(dt.PATH))]
    for d in (12, 14, 16, 18, 20):
        excess = []
        for moves in [s for s in scrambles if len(s) >= d][:args.calibrate]:
            state = cube.from_numpy(fc.apply_sequence(fc.SOLVED, [fc.ACTIONS.index(m) for m in moves[:d]]))
            excess.append(len(solve(state)["solution"]) - d)
        calibration[d] = excess
        print(f"étalonnage d={d} : excès moyen {np.mean(excess):.2f}, exact {np.mean(np.array(excess) == 0):.0%}, "
              f"répartition {np.bincount(excess).tolist()} ({time.time() - t0:.0f} s)", flush=True)

generator = torch.Generator().manual_seed(args.seed)
states, targets, ks, idx, lengths = [], [], [], [], []
start_time, last_save, n_starts = time.time(), time.time(), 0


def save():
    torch.save({"states": torch.stack(states) if states else torch.zeros(0, 54, dtype=torch.uint8),
                "targets": torch.tensor(targets, dtype=torch.float32), "k": torch.tensor(ks),
                "i": torch.tensor(idx), "L": torch.tensor(lengths), "starts": n_starts,
                "seconds": time.time() - start_time, "args": vars(args), "calibration": calibration}, args.out)


while time.time() - start_time < args.minutes * 60:
    walks, _ = cube.random_walks(64, args.k_max, generator)
    k = torch.randint(args.k_min, args.k_max + 1, (64,), generator=generator)
    for j in range(64):
        if time.time() - start_time >= args.minutes * 60:
            break
        start, kj = walks[k[j] - 1, j], int(k[j])
        solution = solve(start)["solution"]
        n_starts += 1
        if solution is None:
            continue
        state, L = start, len(solution)
        for i, a in enumerate(solution):
            states.append(state)
            targets.append(min(L, kj) if i == 0 else L - i)
            ks.append(kj)
            idx.append(i)
            lengths.append(L)
            state = state[cube.moves[a]]
    if time.time() - last_save > 120:
        save()
        last_save = time.time()
        print(f"{n_starts} départs, {len(states)} étiquettes, {time.time() - start_time:.0f} s", flush=True)
save()
L = torch.tensor(lengths)[torch.tensor(idx) == 0].float()
K = torch.tensor(ks)[torch.tensor(idx) == 0].float()
print(f"terminé : {n_starts} départs, {len(states)} étiquettes ; longueur trouvée - longueur de la marche : "
      f"{(L - K).mean():.2f} en moyenne, plus courte dans {(L < K).float().mean():.0%} des cas", flush=True)
