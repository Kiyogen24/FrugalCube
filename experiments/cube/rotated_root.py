"""
pourquoi l'inverse comme seconde racine (2026-11-12): à chaque niveau du faisceau (m90, score V - log pi, 2^10), la
part des places occupées par des copies symétriques d'un autre état du faisceau (même classe sous les 48 symétries,
puis aussi sous l'inversion), et la part du faisceau qui descend de chaque racine.
Usage: python experiments/cube/rotated_root.py [cubes] [threads]
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "experiments/cube")
import search as lab  # noqa: E402  (installe aussi src dans le chemin)
import beam_torch as bt  # noqa: E402
import cube_torch as ct  # noqa: E402
import deepcubea_testset as dt  # noqa: E402
import network_torch as nt  # noqa: E402

N_CUBES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
WIDTH = 1024
cube = ct.TorchCube("cpu")
net, _ = nt.load("runs/s4/m90_v/final.pt")
h = lab.LabHeuristic(net, cube, WIDTH, chunk=1 << 14)
STATS = []          # une ligne par niveau du cube en cours


def classes(states):
    """clé de la classe de chaque état sous les 48 symétries, et sous les 48 symétries et l'inversion."""
    conj = cube.state_hash(cube.conjugates(states)).min(dim=1).values
    conj_inv = cube.state_hash(cube.conjugates(cube.inverse(states))).min(dim=1).values
    return conj, torch.minimum(conj, conj_inv)


init, select = bt._Beam.__init__, bt._Beam.select


def init_with_roots(self, cube_, roots, key_roots):
    init(self, cube_, roots, key_roots)
    self.root_id = torch.arange(len(roots))


def select_with_stats(self, kids, keys, cum, best):
    select(self, kids, keys, cum, best)
    self.root_id = self.root_id[best // 12]
    sym, sym_inv = classes(self.states)
    n = len(self.states)
    STATS.append({"n": n, "copies_sym": 1 - len(sym.unique()) / n, "copies_sym_inv": 1 - len(sym_inv.unique()) / n,
                  "share": (torch.bincount(self.root_id, minlength=self.n_roots).float() / n).tolist()})


bt._Beam.__init__, bt._Beam.select = init_with_roots, select_with_stats

CONFIGS = {"id+inv": ["id", "inv"], "id+c15": ["id", "c15"], "id+ic15": ["id", "ic15"],
           "4 racines": ["id", "inv", "c15", "c20"]}
cases = [cube.from_numpy(s) for s, _ in dt.load()][:N_CUBES]
out = {}
for name, views in CONFIGS.items():
    per_level, lengths = {}, []
    for state in cases:
        STATS.clear()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            r = lab.solve(h, state, WIDTH, views)
        lengths.append(len(r["solution"]) if r["solution"] is not None else np.nan)
        for depth, s in enumerate(STATS, 1):
            per_level.setdefault(depth, []).append(s)
    rows = []
    for depth in sorted(per_level):
        group = per_level[depth]
        if len(group) < N_CUBES // 2:
            break
        rows.append({"depth": depth, "cubes": len(group),
                     "copies_sym": float(np.mean([s["copies_sym"] for s in group])),
                     "copies_sym_inv": float(np.mean([s["copies_sym_inv"] for s in group])),
                     "share": np.mean([s["share"] for s in group], axis=0).round(3).tolist()})
    out[name] = {"mean_length": float(np.nanmean(lengths)), "levels": rows}
    print(f"{name:10} longueur moyenne {np.nanmean(lengths):.3f} sur {len(cases)} cubes", flush=True)
    for row in rows[::2]:
        print(f"   niveau {row['depth']:2}  copies symétriques {row['copies_sym']:6.1%}  avec l'inverse "
              f"{row['copies_sym_inv']:6.1%}  part de chaque racine {row['share']}", flush=True)
json.dump(out, open("runs/cpu9/beam_dupes.json", "w"), indent=1)
