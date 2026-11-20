"""
où un faisceau étroit perd le chemin d'un faisceau large, sur des
cubes neufs, un maître large et un élève étroit, même réseau et même score. Pour chaque cube, on suit le chemin du
maître (dans ses 4 versions, une par vue) dans le faisceau de l'élève: niveau où il en sort, et rang qu'il avait
alors parmi les candidats distincts (rang / largeur; la coupure est au rang = largeur). Un chemin perdu juste sous
la coupure laisse prise à une correction apprise du score; un chemin perdu très loin, non.
Usage: python experiments/analysis/teacher_path.py --cubes 120 --teacher 65536 --student 1024
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

VIEWS = ["id", "inv", "c15", "c20"]


def view_paths(cube, start, solution):
    """pour chaque vue, les empreintes des états du chemin `solution` (repère d'origine) vus dans cette vue."""
    inverse_action = cube.inverse_action.tolist()
    perm = cube.action_perm.tolist()
    paths = []
    for name, (root, _) in zip(VIEWS, bt.equivalent_views(cube, start, VIEWS)):
        if name == "id":
            moves = list(solution)
        elif name == "inv":
            moves = [inverse_action[a] for a in reversed(solution)]
        else:
            moves = [perm[int(name[1:])][a] for a in solution]
        states, s = [], root
        for a in moves:
            s = cube.apply_sequence(s, [a])
            states.append(s)
        if not bool(cube.is_solved(states[-1])):
            raise RuntimeError(f"chemin mal transposé dans la vue {name}")
        paths.append(cube.state_hash(torch.stack(states)))
    return torch.stack(paths, 1)   # (L, 4): empreinte de l'état au niveau k + 1 dans chaque vue


@torch.no_grad()
def student_trace(h, start, width, path_keys, max_depth=40):
    """Le faisceau de l'élève (comme lab.solve), avec, à chaque niveau, la présence du chemin du maître parmi les
    candidats, son meilleur rang parmi les candidats distincts, et s'il est retenu."""
    cube = h.cube
    candidates = bt.equivalent_views(cube, start, VIEWS)
    keys = cube.state_hash(torch.stack([state for state, _ in candidates])).tolist()
    unique = [i for i, key in enumerate(keys) if key not in keys[:i]]
    roots = torch.stack([candidates[i][0] for i in unique])
    forward = bt._Beam(cube, roots, roots)
    levels, length = [], None
    for depth in range(1, max_depth + 1):
        kids, keys, cost, cum, _ = forward.expand(h, lambda x: x)
        best, n_best = forward.candidates(keys, cost, width)
        n_best = int(n_best)
        if depth <= len(path_keys):
            on_path = torch.isin(keys, path_keys[depth - 1]) & torch.isfinite(cost)
            row = {"depth": depth, "present": bool(on_path.any()), "rank": None, "kept": False}
            if row["present"]:
                order = torch.argsort(cost)
                by_key = order[torch.argsort(keys[order], stable=True)]
                first = torch.ones_like(by_key, dtype=torch.bool)
                first[1:] = keys[by_key][1:] != keys[by_key][:-1]
                distinct = cost[by_key][first]
                row["rank"] = int((distinct < cost[on_path].min()).sum())
                row["kept"] = bool(torch.isin(keys[best[:n_best]], path_keys[depth - 1]).any())
            levels.append(row)
        hits = (keys == cube.solved_hash).nonzero().flatten()
        if len(hits) and any(bool(cube.is_solved(kids[j])) for j in hits[:16].tolist()):
            length = depth
            break
        if n_best == 0:
            break
        forward.select(kids, keys, cum, best[:n_best])
    return length, levels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/s4/m90_v/final.pt")
    parser.add_argument("--cubes", type=int, default=120)
    parser.add_argument("--teacher", type=int, default=65536)
    parser.add_argument("--student", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out", default="runs/gates/beam_boundary.jsonl")
    parser.add_argument("--views", default="id,inv,c15,c20", help="racines du maître et de l'élève, par ex. id,inv")
    args = parser.parse_args()
    VIEWS[:] = args.views.split(",")

    torch.set_num_threads(16)
    cube = ct.TorchCube("cpu")
    net, _ = nt.load(args.model, "cpu")
    teacher = lab.LabHeuristic(net, cube, args.teacher, chunk=1 << 14)
    student = lab.LabHeuristic(net, cube, args.student, chunk=1 << 14)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = [json.loads(line) for line in open(args.out)] if os.path.exists(args.out) else []
    cubes = lab.fresh_cubes(cube, args.cubes, args.seed)
    with open(args.out, "a") as f:
        for i in range(len(done), args.cubes):
            t0 = time.time()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                r = lab.solve(teacher, cubes[i], args.teacher, VIEWS)
                path_keys = view_paths(cube, cubes[i], r["solution"])
                length, levels = student_trace(student, cubes[i], args.student, path_keys)
            row = {"i": i, "teacher": len(r["solution"]), "student": length, "levels": levels,
                   "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            lost = next((lv for lv in levels if lv["present"] and not lv["kept"]), None)
            print(f"cube {i} : maître {row['teacher']}, élève {length}, chemin perdu "
                  + (f"au niveau {lost['depth']} (rang {lost['rank']} / {args.student})" if lost else "jamais")
                  + f", {row['seconds']:.0f} s", flush=True)
