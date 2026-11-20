"""
test de l'horizon (T0), sur CPU, avec les checkpoints de la première session GPU.
Hypothèse: l'auto-supervision apprend des postérieures sur les marches aléatoires; jusqu'à ~17 coups les marches
restent presque géodésiques (étiquettes presque exactes, erreurs d'apprentissage que les données corrigent), au-delà
de ~19 elles ont mélangé (log |G| / log 11 ~ 19) et l'étiquette ne dit plus grand-chose de la distance.
Mesures, pour chaque checkpoint (10 à 60 min, et 90 min pour la lignée "notre" + témoin):
- sur des marches neuves (graine fixe): entropie croisée et précision de l'étiquette de la marche, par longueur k;
- sur les préfixes du jeu DeepCubeA (distance exacte d): part du coup optimal connu en tête, log-probabilité de
  ce coup; pour les réseaux de valeur: moyenne de V et séparation de d et d + 2 (aire sous la courbe ROC).
Prédiction: avec l'entraînement, les mesures progressent pour k, d <= ~17 et plafonnent pour k, d >= ~19.
Usage: python experiments/analysis/by_distance.py   (résultats dans runs/horizon/t0.json)
        python experiments/analysis/by_distance.py runs/horizon/t0_arch.json runs/arch/a034_v/final.pt ...
"""
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
import beam_torch as bt
import cube_torch as ct
import deepcubea_testset as dt
import fast_cube as fc
import network_torch as nt

torch.set_num_threads(16)
OUT = "runs/horizon/t0.json"
LENGTH, WALKS = 26, 20000
cube = ct.TorchCube("cpu")

states, labels, depths = cube.self_supervised_batch(WALKS, LENGTH, generator=torch.Generator().manual_seed(123))
k_of = depths.numpy()

prefix, dist, best = [], [], []
for scramble in json.load(open(dt.PATH)):
    st = fc.SOLVED
    for j, m in enumerate(scramble.split(), 1):
        a = fc.ACTIONS.index(m)
        st = fc.apply_sequence(st, [a])
        prefix.append(st)
        dist.append(j)
        best.append(fc.INVERSE_ACTION[a])
prefix = cube.from_numpy(np.stack(prefix))
dist, best = np.array(dist), np.array(best)


def auc(lo, hi):
    """P(V(hi) > V(lo)) + 0,5 P(égalité), par les rangs."""
    both = np.concatenate([lo, hi])
    ranks = np.argsort(np.argsort(both, kind="stable"), kind="stable") + 1.0
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / counts)[inv]                    # rangs moyens en cas d'égalité
    r_hi = ranks[len(lo):].sum()
    return float((r_hi - len(hi) * (len(hi) + 1) / 2) / (len(lo) * len(hi)))


lineages = {
    "takano": [f"runs/takano/ckpt_000{m}.0min.pt" for m in (10, 20, 30, 40, 50)] + ["runs/takano/final.pt"],
    "notre": [f"runs/notre/ckpt_000{m}.0min.pt" for m in (10, 20, 30, 40, 50)] + ["runs/notre/final.pt"]
             + [f"runs/rl_temoin/ckpt_000{m}.0min.pt" for m in (10, 20)] + ["runs/rl_temoin/final.pt"],
    "cayleypy": [f"runs/cayleypy/ckpt_000{m}.0min.pt" for m in (10, 20, 30, 40, 50)] + ["runs/cayleypy/final.pt"],
    "takano_valeur": ["runs/takano_valeur/final.pt"],
}
if len(sys.argv) > 1:   # autres modèles: python experiments/analysis/by_distance.py sortie.json modèle1.pt modèle2.pt ...
    OUT, lineages = sys.argv[1], {"autres": sys.argv[2:]}
results = json.load(open(OUT)) if os.path.exists(OUT) else {}
for lineage, paths in lineages.items():
    for minutes, path in enumerate(paths, 1):
        key = f"{lineage}:{path}"
        if key in results:
            continue
        model, meta = nt.load(path)
        heads = tuple(model.config["heads"])
        h = bt.NetworkHeuristic(model, cube, heads[0], chunk=8192)
        rec = {"lineage": lineage, "path": path, "examples": meta.get("examples"), "elapsed_s": meta.get("elapsed_s")}
        with torch.autocast("cpu", dtype=torch.bfloat16):
            if "policy" in heads:
                logp = torch.log_softmax(h.head_outputs(states, ("policy",))["policy"], dim=1)
                ce = -logp.gather(1, labels[:, None]).squeeze(1).numpy()
                acc = (logp.argmax(1) == labels).numpy()
                rec["walk_ce"] = [float(ce[k_of == k].mean()) for k in range(1, LENGTH + 1)]
                rec["walk_acc"] = [float(acc[k_of == k].mean()) for k in range(1, LENGTH + 1)]
            out = h.head_outputs(prefix, heads)
        ds = list(range(1, int(dist.max()) + 1))
        rec["d_count"] = [int((dist == d).sum()) for d in ds]
        if "policy" in out:
            lp = torch.log_softmax(out["policy"], dim=1).numpy()
            rec["d_top1"] = [float((lp.argmax(1) == best)[dist == d].mean()) for d in ds]
            rec["d_logp_best"] = [float(lp[np.arange(len(best)), best][dist == d].mean()) for d in ds]
        if "value" in out:
            v = out["value"].numpy()
            rec["d_value"] = [float(v[dist == d].mean()) for d in ds]
            rec["d_auc_plus2"] = [auc(v[dist == d], v[dist == d + 2]) for d in ds if (dist == d + 2).any()]
        results[key] = rec
        json.dump(results, open(OUT, "w"))
        line = f"{lineage:14} {os.path.basename(os.path.dirname(path)):11} {os.path.basename(path):20}"
        if "walk_acc" in rec:
            line += " | acc. marche k=8,14,17,19,22,26 : " + " ".join(
                f"{rec['walk_acc'][k - 1]:.3f}" for k in (8, 14, 17, 19, 22, 26))
        if "d_top1" in rec:
            line += " | top-1 d=8,14,17,19,21 : " + " ".join(f"{rec['d_top1'][d - 1]:.3f}" for d in (8, 14, 17, 19, 21))
        if "d_auc_plus2" in rec:
            line += " | AUC d/d+2 d=8,14,16,18,19,20 : " + " ".join(
                f"{rec['d_auc_plus2'][d - 1]:.3f}" for d in (8, 14, 16, 18, 19, 20))
        print(line, flush=True)
print("terminé", flush=True)
