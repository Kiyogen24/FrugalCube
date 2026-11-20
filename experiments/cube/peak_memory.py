"""
Mémoire de pointe de la recherche (CPU, même code que le GPU): un processus neuf par largeur, quelques cubes du jeu
de test, pic de l'ensemble de travail (psutil) avant et après la recherche, modèle chargé compris.
Usage: python experiments/cube/peak_memory.py WIDTH CASES
"""
import json
import sys

import psutil
import torch

sys.path.insert(0, "experiments/cube")
sys.path.insert(0, "src")
import cube_torch as ct
import deepcubea_testset as dt
import search as lab
import network_torch as nt

torch.set_num_threads(16)
width, cases = int(sys.argv[1]), int(sys.argv[2])
proc = psutil.Process()
cube = ct.TorchCube("cpu")
net, _ = nt.load("runs/s4/m90_v/final.pt", "cpu")
h = lab.LabHeuristic(net, cube, width, chunk=1 << 14)
before = proc.memory_info().peak_wset
times = []
for state, optimal in dt.load(cases):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        r = lab.solve(h, cube.from_numpy(state), width, ["id", "inv", "c15", "c20"])
    times.append(r["time_s"])
peak = proc.memory_info().peak_wset
print(json.dumps({"width": width, "cases": cases, "peak_MB": round(peak / 2 ** 20), "before_MB": round(before / 2 ** 20),
                  "search_MB": round((peak - before) / 2 ** 20), "time_s": round(sum(times) / len(times), 2)}))
