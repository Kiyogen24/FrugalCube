"""
Notre solveur sur un jeu d'instances de make_instances.py: faisceau à deux racines (M et M⁻¹), score
V(parent) - lambda log pi, table de fin de partie; lambda réglé sur des instances de validation (graine 7, jamais
le test). Écrit une ligne JSON par instance et par largeur (longueur = nombre de CNOT ou de SWAP, temps,
évaluations, optimum si --table) et, avec --save-circuits, le circuit trouvé (paires (contrôle, cible) ou (a, b) dans
l'ordre du temps, qubits logiques), que ibm_linear.py revérifie avec Qiskit.

Usage: python experiments/cnot/eval_cnot.py --model runs/cnot/line9/final.pt --instances experiments/cnot/inst/cnot_line9_test.json \
            --widths 64,256,1024 --out runs/cnot/line9/eval.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
import beam_torch as bt  # noqa: E402
import circuit_torch as qt  # noqa: E402
import network_torch as nt  # noqa: E402
import perm_search as ps  # noqa: E402

LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--instances", required=True)
parser.add_argument("--widths", default="64,256,1024")
parser.add_argument("--cases", type=int, default=0, help="0 : toutes les instances")
parser.add_argument("--ball-states", type=float, default=2e6, help="taille max de la table de fin de partie")
parser.add_argument("--no-ball", action="store_true")
parser.add_argument("--lam", type=float, help="lambda imposé")
parser.add_argument("--mode", choices=["ac", "policy", "value"], default="ac",
                    help="ac : V(parent) - lambda log pi ; policy : log-probabilité cumulée du chemin ; value : V de "
                         "chaque enfant (n_moves évaluations par état). lambda ne sert qu'en ac")
parser.add_argument("--tune-cases", type=int, default=50)
parser.add_argument("--tune-width", type=int, default=64)
parser.add_argument("--views", default="id,inv")
parser.add_argument("--max-depth", type=int, default=300)
parser.add_argument("--table", help="table exacte (n <= 5 pour cnot) : optimum de chaque instance")
parser.add_argument("--save-circuits", action="store_true")
parser.add_argument("--threads", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

device = torch.device(args.device)
model, meta = nt.load(args.model, device)
train = meta["args"]
p = qt.make_puzzle(train["puzzle"], train["n"], train.get("topology", "line"), device)
with open(args.instances) as f:
    spec = json.load(f)
if (spec["puzzle"], spec["n"], [tuple(e) for e in spec["edges"]]) != (p.name, p.n, p.edges):
    raise ValueError("instances et réseau sur des puzzles différents")
raw = spec["instances"][:args.cases or None]
starts = p.from_matrices(raw) if p.name == "cnot" else torch.tensor(raw, dtype=torch.uint8, device=device)
table = qt.load_or_build_table(p, args.table) if args.table else None
views = args.views.split(",")
t0 = time.time()
ball = None if args.no_ball else ps.Ball(p, 10 ** 6, max_states=args.ball_states)
if ball is not None:
    print(f"table de fin : rayon {ball.radius}, {len(ball.keys):,} états, {time.time() - t0:.0f} s", flush=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)


def record(row):
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")


lam = args.lam
if args.mode != "ac":
    lam = None
elif lam is None:
    valid = p.random_states(args.tune_cases, torch.Generator().manual_seed(7)).to(device)
    scores = {}
    for lam_try in LAMBDAS:
        h = bt.NetworkHeuristic(model, p, mode="ac", lam=lam_try)
        sols = [ps.solve(h, s, args.tune_width, views=views, ball=ball, max_depth=args.max_depth)["solution"]
                for s in valid]
        scores[lam_try] = float(np.mean([len(x) if x is not None else 2 * args.max_depth for x in sols]))
        print(f"lambda {lam_try} : {scores[lam_try]:.2f}", flush=True)
    lam = min(LAMBDAS, key=lambda x: (round(scores[x], 6), x))
    record({"tune": scores, "lambda": lam, "tune_width": args.tune_width})

h = bt.NetworkHeuristic(model, p, mode=args.mode, lam=lam if lam is not None else 0.0)
for width in [int(w) for w in args.widths.split(",")]:
    lengths, times = [], []
    for i, s in enumerate(starts):
        res = ps.solve(h, s, width, views=views, ball=ball, max_depth=args.max_depth)
        sol = res["solution"]
        row = {"i": i, "width": width, "mode": args.mode, "lambda": lam,
               "table": ball.radius if ball is not None else 0,
               "length": None if sol is None else len(sol), "time_s": round(res["time_s"], 4),
               "evaluations": res["evaluations"], "ball_hit": res.get("ball_hit"), "roots": res.get("roots")}
        if table is not None:
            row["optimum"] = int(table[p.rank(s[None])][0])
        if args.save_circuits and sol is not None:
            row["circuit"] = [list(p.pairs[a]) for a in reversed(sol)]      # ordre du temps
        record(row)
        lengths.append(row["length"])
        times.append(res["time_s"])
    ok = [x for x in lengths if x is not None]
    print(f"largeur {width} : {len(ok)}/{len(lengths)} résolus, longueur moyenne {np.mean(ok):.2f}, "
          f"{np.mean(times):.3f} s par instance", flush=True)
