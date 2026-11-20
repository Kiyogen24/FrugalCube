"""
pilote CNOT: notre solveur (bras ss, bellman, improve, par largeur) contre la politique PPO d'IBM (par nombre de
rollouts), sur les mêmes instances. Moyenne de CNOT, instances résolues, temps par instance, et comparaisons
appariées (même instance): écart moyen et z.

    python experiments/cnot/analyse_pilot.py [--dir runs/cnot/pilot]
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs/cnot/pilot")
ap.add_argument("--arms", default="ss,bellman,improve")
args = ap.parse_args()


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


methods = {}            # nom -> {instance: (longueur ou None, temps)}
for arm in args.arms.split(","):
    folder = os.path.join(args.dir, arm)
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.startswith("eval") and f.endswith(".jsonl")) if os.path.isdir(folder) else []
    for r in (r for path in paths for r in rows(path)):
        if "tune" in r:
            print(f"{arm:8s} lambda {r['lambda']} (largeur {r['tune_width']}) : "
                  + ", ".join(f"{k} {v:.1f}" for k, v in r["tune"].items()))
            continue
        methods.setdefault(f"{arm} w{r['width']}", {})[r["i"]] = (r["length"], r["time_s"])
EXTERNAL = (("ibm.jsonl", "IBM {}"), ("alphacnot.jsonl", "{}"))      # références: IBM, AlphaCNOT
for fname, label in EXTERNAL:
    path = os.path.join(args.dir, fname)
    if os.path.exists(path):
        for r in rows(path):
            methods.setdefault(label.format(r["searches"]), {})[r["i"]] = (r["length"] if r["ok"] else None,
                                                                          r["time_s"])

print(f"\n{'méthode':16s} {'résolues':>9s} {'CNOT moyen':>11s} {'médiane':>8s} {'temps/inst.':>12s}")
for name, res in methods.items():
    ok = [v[0] for v in res.values() if v[0] is not None]
    t = [v[1] for v in res.values()]
    print(f"{name:16s} {len(ok):4d}/{len(res):<4d} {np.mean(ok) if ok else float('nan'):11.2f} "
          f"{np.median(ok) if ok else float('nan'):8.1f} {np.mean(t):11.3f}s")

ibm = {k: v for k, v in methods.items() if k.startswith(("IBM", "AlphaCNOT"))}
ours = {k: v for k, v in methods.items() if k not in ibm}
if ibm and ours:
    print("\nappariées (nous - référence, instances résolues par les deux ; négatif = nous plus courts) :")
    for a, ra in ours.items():
        for b, rb in ibm.items():
            common = [i for i in ra if i in rb and ra[i][0] is not None and rb[i][0] is not None]
            if not common:
                continue
            d = np.array([ra[i][0] - rb[i][0] for i in common], dtype=float)
            z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 and d.std() > 0 else 0.0
            print(f"  {a:16s} - {b:10s} : {d.mean():+7.2f} CNOT (z {z:+6.1f}, n {len(d)}), "
                  f"nous plus courts {np.mean(d < 0):5.1%}, égalité {np.mean(d == 0):5.1%}")
