"""
le même faisceau sans réseau ni apprentissage: chaque enfant est noté par une mesure faite à la main de l'écart entre
sa matrice et l'identité, et le faisceau garde les moins chers, depuis M et M^-1, avec la même table de fin de partie.
Mesures (M sur GF(2), n x n, inversible, donc chaque ligne et chaque colonne a un poids >= 1):
- weight: nombre de uns moins n;
- logcol: somme sur les colonnes de log2(poids de la colonne);
- logrowcol: logcol plus la même somme sur les lignes.
La mesure est choisie sur les 50 matrices de validation (graine 7) à --tune-width, comme lambda pour le solveur appris,
puis les matrices de test sont résolues à chaque largeur. Sortie: une ligne JSON par matrice et par largeur, comme
eval_cnot.py.
Usage: python experiments/cnot/eval_heur.py --model runs/cnot/h2h/8qJ/final.pt \\
           --instances experiments/cnot/inst/ibm8qJ_test.json --widths 1024,4096 --out runs/cnot/heur/8qJ.jsonl
(--model ne sert qu'à la topologie, le réseau n'est pas utilisé.)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
import circuit_torch as qt  # noqa: E402
import network_torch as nt  # noqa: E402
import perm_search as ps  # noqa: E402

KINDS = ["weight", "logcol", "logrowcol"]


class MatrixHeuristic:
    """Note chaque enfant par une mesure de sa matrice, sans réseau: le mode 'value' de beam_torch, sans modèle."""

    def __init__(self, puzzle, kind):
        self.cube, self.kind, self.mode = puzzle, kind, "value"

    def evaluations(self, n_states):
        return self.cube.n_moves * n_states

    @torch.no_grad()
    def __call__(self, states, kids, cum):
        p, n = self.cube, len(states)
        flat = kids.reshape(-1, kids.shape[-1])
        bits = p.encode(flat).reshape(len(flat), p.n, p.n)      # [matrice, ligne, colonne]
        if self.kind == "weight":
            h = bits.sum((1, 2)) - p.n
        else:
            h = torch.log2(bits.sum(1)).sum(1)
            if self.kind == "logrowcol":
                h = h + torch.log2(bits.sum(2)).sum(1)
        h = h.reshape(n, -1)
        return h, torch.zeros_like(h), h.shape[1] * n


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="a network of the layout, only to read the layout")
parser.add_argument("--instances", required=True)
parser.add_argument("--widths", default="1024,4096")
parser.add_argument("--cases", type=int, default=0)
parser.add_argument("--kind", choices=KINDS, help="measure imposed (else chosen on the validation matrices)")
parser.add_argument("--tune-cases", type=int, default=50)
parser.add_argument("--tune-width", type=int, default=1024)
parser.add_argument("--ball-states", type=float, default=2e6)
parser.add_argument("--views", default="id,inv")
parser.add_argument("--max-depth", type=int, default=300)
parser.add_argument("--threads", type=int, default=0)
parser.add_argument("--out", required=True)
args = parser.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

device = torch.device("cpu")
_, meta = nt.load(args.model, device)
train = meta["args"]
p = qt.make_puzzle(train["puzzle"], train["n"], train.get("topology", "line"), device)
spec = json.load(open(args.instances))
if (spec["puzzle"], spec["n"], [tuple(e) for e in spec["edges"]]) != (p.name, p.n, p.edges):
    raise ValueError("instances and layout differ")
starts = p.from_matrices(spec["instances"][:args.cases or None])
views = args.views.split(",")
t0 = time.time()
ball = ps.Ball(p, 10 ** 6, max_states=args.ball_states)
print(f"endgame table: radius {ball.radius}, {len(ball.keys):,} states, {time.time() - t0:.0f} s", flush=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)


def record(row):
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")


kind = args.kind
if kind is None:
    valid = p.random_states(args.tune_cases, torch.Generator().manual_seed(7)).to(device)
    scores = {}
    for k in KINDS:
        h = MatrixHeuristic(p, k)
        sols = [ps.solve(h, s, args.tune_width, views=views, ball=ball, max_depth=args.max_depth)["solution"]
                for s in valid]
        scores[k] = float(np.mean([len(x) if x is not None else 2 * args.max_depth for x in sols]))
        print(f"{k}: {scores[k]:.2f} on validation", flush=True)
    kind = min(KINDS, key=lambda k: scores[k])
    record({"tune": scores, "kind": kind, "tune_width": args.tune_width})

h = MatrixHeuristic(p, kind)
for width in [int(w) for w in args.widths.split(",")]:
    lengths, times = [], []
    for i, s in enumerate(starts):
        res = ps.solve(h, s, width, views=views, ball=ball, max_depth=args.max_depth)
        sol = res["solution"]
        row = {"i": i, "width": width, "mode": f"heur_{kind}", "table": ball.radius,
               "length": None if sol is None else len(sol), "time_s": round(res["time_s"], 4),
               "evaluations": res["evaluations"], "roots": res.get("roots")}
        if sol is not None:
            row["circuit"] = [list(p.pairs[a]) for a in reversed(sol)]      # dans l'ordre du temps, vérifié par ibm_linear.py
        record(row)
        lengths.append(row["length"])
        times.append(res["time_s"])
    ok = [x for x in lengths if x is not None]
    print(f"width {width}: {len(ok)}/{len(lengths)} solved, mean length {np.mean(ok) if ok else float('nan'):.2f}, "
          f"{np.mean(times):.3f} s per matrix", flush=True)
