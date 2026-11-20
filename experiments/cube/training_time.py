"""
analyse de la session 4: le 3,8 M réentraîné (30, 60, 90 min), comparé cube par cube au 3,8 M de 30 min de la
session 3 (perdu, mais ses résultats par cube sont dans runs/arch/eval) et au 15 M.
Usage: python experiments/cube/training_time.py
"""
import json
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def load(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def summary(rows):
    lengths = [r["length"] for r in rows if r["length"] is not None]
    return (f"{len(rows):4} cas | résolus {len(lengths) / len(rows):6.1%} | optimales "
            f"{np.mean([r['length'] == r['optimal'] for r in rows]):6.1%} | longueur {np.mean(lengths):.3f} | "
            f"{np.mean([r['time_s'] for r in rows]):.3f} s/cube")


def paired(a, b):
    """b contre a sur les cas communs: écart de longueur moyen (z), solutions optimales de b seul / de a seul."""
    ia = {r["i"]: r for r in a}
    common = [(ia[r["i"]], r) for r in b if r["i"] in ia and r["length"] is not None and ia[r["i"]]["length"] is not None]
    d = np.array([y["length"] - x["length"] for x, y in common], dtype=float)
    z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 and d.std() > 0 else 0.0
    ob = sum(y["length"] == y["optimal"] and x["length"] != x["optimal"] for x, y in common)
    oa = sum(x["length"] == x["optimal"] and y["length"] != y["optimal"] for x, y in common)
    zm = (ob - oa) / math.sqrt(ob + oa) if ob + oa else 0.0
    return f"{len(common)} cas communs : {d.mean():+.3f} coups (z {z:+.1f}), optimales {ob}/{oa} (z {zm:+.1f})"


print("Entraînement du 3,8 M (session 4)")
log = load("runs/s4/a38/train_log.jsonl")
for t in (600, 1800, 3600, 5400):
    if log:
        r = min(log, key=lambda r: abs(r["time_s"] - t))
        print(f"  {r['time_s'] / 60:5.1f} min | {r['examples'] / 1e9:5.2f} G exemples | {r['examples_per_s'] / 1e6:.2f} M/s | "
              f"perte {r['policy_loss']:.4f}")
old = load("runs/arch/a38/train_log.jsonl")
if old:
    print(f"  session 3, 30 min : {old[-1]['examples'] / 1e9:.2f} G exemples | perte {old[-1]['policy_loss']:.4f}")
for name in ("m30", "m60", "m90"):
    v = load(f"runs/s4/{name}_v/train_log.jsonl")
    if v:
        print(f"  tête valeur {name} : écart quadratique {v[-1]['value_loss']:.2f}")

runs = {
    "s3 a38 30 min, w4096 (300)": load("runs/arch/eval/sweep_a38_l1.jsonl"),
    "s4 a38 30 min, w4096 (300)": load("runs/s4/eval/m30_w4096.jsonl"),
    "s3 a15 30 min, w93184 (1 s)": load("runs/arch/eval/budget1_a15.jsonl"),
    "s3 t60 60 min, w93184 (1 s)": load("runs/arch/eval/budget1_t60.jsonl"),
    "s3 a38 30 min, w206848 (1 s)": load("runs/arch/eval/budget1_a38.jsonl"),
    "s4 a38 90 min, w206848 (1 s)": load("runs/s4/eval/m90_w206848.jsonl"),
    "s4 a38 90 min, w524288": load("runs/s4/eval/m90_w524288.jsonl"),
}
print()
for name, rows in runs.items():
    if rows:
        print(f"{name:30} {summary(rows)}")
print("\nComparaisons appariées (le second contre le premier)")
for a, b in [("s3 a38 30 min, w4096 (300)", "s4 a38 30 min, w4096 (300)"),
             ("s3 a38 30 min, w206848 (1 s)", "s4 a38 90 min, w206848 (1 s)"),
             ("s3 t60 60 min, w93184 (1 s)", "s4 a38 90 min, w206848 (1 s)"),
             ("s3 a15 30 min, w93184 (1 s)", "s4 a38 90 min, w206848 (1 s)"),
             ("s4 a38 90 min, w206848 (1 s)", "s4 a38 90 min, w524288")]:
    if runs[a] and runs[b]:
        print(f"  {b} vs {a} : {paired(runs[a], runs[b])}")
rows = runs["s4 a38 90 min, w524288"] or runs["s4 a38 90 min, w206848 (1 s)"]
if rows:
    print("\nTaux optimal par distance exacte (meilleur run de la session 4)")
    for d in sorted({r["optimal"] for r in rows}):
        sel = [r for r in rows if r["optimal"] == d]
        print(f"  d={d} : {np.mean([r['length'] == d for r in sel]):6.1%} ({len(sel)} cas)")
