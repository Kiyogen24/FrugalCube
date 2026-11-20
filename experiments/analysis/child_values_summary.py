"""
Résumé de deux tests (runs/gates):
A. score par la valeur de chaque enfant V(s.a) contre le score partagé V(s), à largeur égale, traduit en doublements
   de largeur de la base (le plafond d'une tête Q distillée);
B. où le chemin d'un maître large sort du faisceau étroit, et à quel rang (la prise qu'aurait un score appris au bord).
Usage: python experiments/analysis/child_values_summary.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def load(path):
    return {r["i"]: r for r in map(json.loads, open(path))}


def gate_a():
    print("== Valeurs des enfants et suites: cubes neufs (graine 777), longueurs appariées à la base de même largeur")
    widths = [w for w in (1024, 2048, 4096, 8192) if os.path.exists(f"runs/gates/base_fresh777_w{w}.jsonl")]
    base = {w: load(f"runs/gates/base_fresh777_w{w}.jsonl") for w in widths}
    ids = sorted(base[1024])
    widths = [w for w in widths if len(base[w]) == len(ids)]   # une base encore en cours est ignorée
    base = {w: base[w] for w in widths}
    L = {w: np.array([d[i]["length"] for i in ids], float) for w, d in base.items()}
    E = {w: np.mean([d[i]["evaluations"] for i in ids]) for w, d in base.items()}
    xs, ys = [math.log2(w) for w in widths], [L[w].mean() for w in widths]
    for w in widths:
        print(f"  base w{w:<5} {L[w].mean():.3f} coups, {np.mean([base[w][i]['time_s'] for i in ids]):.2f} s/cube, "
              f"{E[w] / 1e3:.0f} k évaluations")
    paths = sorted(glob.glob("runs/gates/childv*.jsonl") + glob.glob("runs/gates/wide*.jsonl")
                   + glob.glob("runs/gates/qhead/q_*fresh777*.jsonl"))
    for path in paths:
        w = int(path.rsplit("_w", 1)[1][:-6])
        d = load(path)
        if len(d) < len(ids) or w not in L:
            continue
        x = np.array([d[i]["length"] for i in ids], float)
        diff = x - L[w]
        z = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff)))
        # gain: position de sa longueur moyenne sur la courbe de la base (log2 largeur), en doublements depuis w;
        # coût: log2 de ses évaluations du réseau sur celles de la base à la même largeur
        gain = np.interp(-x.mean(), [-y for y in ys], xs) - math.log2(w) if x.mean() >= ys[-1] else float("nan")
        cost = math.log2(np.mean([d[i]["evaluations"] for i in ids]) / E[w])
        name = os.path.basename(path)[:-6].replace("_fresh777", "")
        print(f"  {name:28} {x.mean():.3f} coups ({diff.mean():+.3f}, z {z:+.1f}), "
              f"{np.mean([d[i]['time_s'] for i in ids]):.2f} s/cube ; gain "
              + (f"{gain:+.2f}" if gain == gain else f"> {xs[-1] - math.log2(w):.0f}")
              + f" doublement, coût {cost:+.2f} doublement d'évaluations")


def gate_b(path="runs/gates/beam_boundary.jsonl"):
    rows = [json.loads(line) for line in open(path)]
    W = 1024
    print(f"\n== Chemin du maître (w65536) dans le faisceau de l'élève (w{W}), {len(rows)} cubes neufs")
    t = np.array([r["teacher"] for r in rows]); s = np.array([r["student"] for r in rows])
    print(f"  maître {t.mean():.3f} coups, élève {s.mean():.3f} ; maître plus court sur {np.mean(t < s):.0%} des cubes")
    for label, keep in [("élève plus long", t < s), ("tous", np.ones(len(rows), bool))]:
        lost = []
        for r, k in zip(rows, keep):
            if not k:
                continue
            lv = next((lv for lv in r["levels"] if lv["present"] and not lv["kept"]), None)
            if lv is not None:
                lost.append((lv["depth"], r["teacher"] - lv["depth"], lv["rank"] / W))
        if not lost:
            continue
        depth, remaining, ratio = map(np.array, zip(*lost))
        print(f"  [{label}] chemin perdu sur {len(lost)} cubes : niveau médian {np.median(depth):.0f} "
              f"(distance restante médiane {np.median(remaining):.0f}) ; rang / largeur : médiane {np.median(ratio):.1f}, "
              f"< 1.5 : {np.mean(ratio < 1.5):.0%}, < 2 : {np.mean(ratio < 2):.0%}, < 4 : {np.mean(ratio < 4):.0%}, "
              f">= 10 : {np.mean(ratio >= 10):.0%}")
        hist = np.bincount(remaining.astype(int), minlength=24)
        print("    distance restante au moment de la perte : "
              + " ".join(f"{k}:{hist[k]}" for k in range(len(hist)) if hist[k]))


if __name__ == "__main__":
    gate_a()
    gate_b()
