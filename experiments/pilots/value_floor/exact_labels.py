"""
étiquettes exactes pour la borne haute (bras "oracle"): les préfixes des cubes pairs du jeu DeepCubeA, avec
leur distance exacte et le coup optimal connu (inverse du dernier coup du mélange), au format de labels.py.
Les cubes impairs restent pour l'évaluation.
Usage: python experiments/pilots/value_floor/exact_labels.py
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import cube_torch as ct
import deepcubea_testset as dt
import fast_cube as fc

cube = ct.TorchCube("cpu")
states, targets, moves = [], [], []
for i, scramble in enumerate(json.load(open(dt.PATH))):
    if i % 2:
        continue
    st = fc.SOLVED
    for j, m in enumerate(scramble.split(), 1):
        a = fc.ACTIONS.index(m)
        st = fc.apply_sequence(st, [a])
        states.append(st)
        targets.append(j)
        moves.append(fc.INVERSE_ACTION[a])
torch.save({"states": cube.from_numpy(np.stack(states)), "targets": torch.tensor(targets, dtype=torch.float32),
            "moves": torch.tensor(moves), "starts": 500, "seconds": 0.0}, "runs/floor/labels_exact_even.pt")
print(len(states), "étiquettes exactes")
