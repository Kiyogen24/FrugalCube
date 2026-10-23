"""
Évaluation des modèles PyTorch sur le jeu de test de DeepCubeA, pour un checkpoint ou pour tous les checkpoints
d'un dossier d'entraînement : chaque checkpoint donne un point de la courbe « qualité en fonction du temps
d'entraînement » (taux de résolution, part de solutions optimales, longueur moyenne, coût de la recherche).
Option --scramble-depth : cubes mélangés de d coups (graine seed x 1000 + d) à la place du jeu
de DeepCubeA ; la longueur optimale n'est alors pas connue.
Option --views : vues multiples (voir beam_torch.equivalent_views), un seul faisceau à plusieurs racines, ou un
faisceau de largeur --width par vue avec --separate-views.

Usage :
    python src/eval_torch.py runs/auto_supervise/final.pt --cases 1000 --width 65536
    python src/eval_torch.py runs/auto_supervise --cases 200 --width 4096 --out logs/courbe.jsonl
    python src/eval_torch.py runs/deux_tetes/final.pt --mode ac --lam 4 --views id,inv --separate-views --width 8192
"""
import argparse
import json
import os

import numpy as np
import torch

import beam_torch as bt
import cube_torch as ct
import deepcubea_testset
import fast_cube as fc
import network_torch as nt


def summarize(runs):
    """runs : liste de (résultat de beam_torch.solve, longueur optimale ou None si inconnue)."""
    solved = [(r, o) for r, o in runs if r["solution"] is not None]
    known = all(o is not None for _, o in runs)
    return {
        "solved": len(solved) / len(runs),
        "optimal_rate": sum(len(r["solution"]) == o for r, o in solved) / len(runs) if known else None,
        "mean_length": float(np.mean([len(r["solution"]) for r, _ in solved])) if solved else None,
        "mean_optimal_of_solved": float(np.mean([o for _, o in solved])) if solved and known else None,
        "mean_evaluations": float(np.mean([r["evaluations"] for r, _ in runs])),
        "mean_time_s": float(np.mean([r["time_s"] for r, _ in runs])),
        "lengths": [len(r["solution"]) if r["solution"] is not None else None for r, _ in runs],
        "optimal_lengths": [o for _, o in runs],
    }


def evaluate(heuristic, cases, width, max_depth=40, bidirectional=False, max_evaluations=None, views=None,
             separate_views=False):
    options = dict(width=width, max_depth=max_depth, bidirectional=bidirectional, max_evaluations=max_evaluations)

    def run(state):
        if views and separate_views:
            return bt.solve_portfolio(heuristic, state, views, **options)
        return bt.solve(heuristic, state, views=views, **options)

    return summarize([(run(state), optimal) for state, optimal in cases])


def checkpoints_of(path):
    """Un fichier, ou les checkpoints d'un dossier dans l'ordre du temps d'entraînement, puis final.pt."""
    if os.path.isfile(path):
        return [path]
    names = sorted(n for n in os.listdir(path) if n.startswith("ckpt_") and n.endswith(".pt"))
    if os.path.exists(os.path.join(path, "final.pt")):
        names.append("final.pt")
    return [os.path.join(path, n) for n in names]


def main():
    parser = argparse.ArgumentParser(description="Évaluation PyTorch sur le jeu de test de DeepCubeA")
    parser.add_argument("path", help="checkpoint .pt ou dossier d'entraînement")
    parser.add_argument("--cases", type=int, default=100, help="nombre de cas (les premiers du jeu de test)")
    parser.add_argument("--scramble-depth", type=int, help="cubes mélangés de d coups au lieu du jeu de DeepCubeA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--mode", choices=["policy", "value", "q", "ac"],
                        help="classement (par défaut : policy, puis q, puis value) ; ac : acteur-critique")
    parser.add_argument("--lam", type=float, default=2.0, help="poids de la politique en mode ac")
    parser.add_argument("--views", help="vues multiples séparées par des virgules, par ex. id,inv,c13,c29")
    parser.add_argument("--separate-views", action="store_true", help="un faisceau par vue au lieu d'un seul")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--max-evaluations", type=int, help="budget d'évaluations du réseau par cube")
    parser.add_argument("--chunk", type=int, default=1 << 16, help="états par passe du réseau")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", help="fichier JSONL, une ligne par checkpoint (ajoutée à la fin)")
    args = parser.parse_args()

    cube = ct.TorchCube(args.device)
    if args.scramble_depth:
        rng = np.random.default_rng(args.seed * 1000 + args.scramble_depth)
        cases = [(cube.from_numpy(fc.scramble(rng, args.scramble_depth)[0]), None) for _ in range(args.cases)]
        print(f"{len(cases)} cubes mélangés de {args.scramble_depth} coups", flush=True)
    else:
        cases = [(cube.from_numpy(s), o) for s, o in deepcubea_testset.load(args.cases)]
        print(f"{len(cases)} cas, longueur optimale moyenne {np.mean([o for _, o in cases]):.2f}", flush=True)
    views = args.views.split(",") if args.views else None
    for checkpoint in checkpoints_of(args.path):
        model, meta = nt.load(checkpoint, args.device)
        heuristic = bt.NetworkHeuristic(model, cube, args.mode, args.chunk, args.lam)
        summary = evaluate(heuristic, cases, args.width, args.max_depth, args.bidirectional, args.max_evaluations,
                           views, args.separate_views)
        entry = {"checkpoint": checkpoint, "elapsed_s": meta.get("elapsed_s"), "examples": meta.get("examples"),
                 "steps": meta.get("steps"), "mode": heuristic.mode, "lam": args.lam if heuristic.mode == "ac" else None,
                 "width": args.width, "bidirectional": args.bidirectional, "views": views,
                 "separate_views": args.separate_views, "cases": len(cases), "scramble_depth": args.scramble_depth,
                 **summary}
        length = f"{summary['mean_length']:.2f}" if summary["mean_length"] is not None else "-"
        optimal = f"{summary['optimal_rate']:.0%}" if summary["optimal_rate"] is not None else "-"
        print(f"{os.path.basename(checkpoint):>22} | {meta.get('elapsed_s', 0) / 60:6.1f} min | "
              f"résolus {summary['solved']:.0%} | optimales {optimal} | longueur {length} | "
              f"évaluations {summary['mean_evaluations']:.0f} | {summary['mean_time_s']:.2f} s/cube", flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
