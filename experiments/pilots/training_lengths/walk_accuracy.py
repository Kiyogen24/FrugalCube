"""
Écart entre mesure et recherche (2026-11-12): une grandeur calculable sur des mélanges, sans étiquette optimale,
classe-t-elle les réseaux comme le fait la recherche? Mesures fixées avant de regarder les résultats:
- M1 précision de l'étiquette des mélanges (inverse du dernier coup);
- M2 entropie croisée de la politique sur cette étiquette;
- M3 aire sous la courbe ROC de la valeur entre mélanges de longueur k et k + 2;
- M4 survie à la coupe, réservoir de même longueur: le coût V(s) − log π(a|s) de l'enfant étiqueté est-il dans le
  dixième le plus bas des coûts de tous les enfants des états de longueur k;
- M5 survie à la coupe, réservoir profond: même test contre les enfants des états de longueur 20 à 26 (presque
  aléatoires, la masse du plancher contre laquelle le faisceau garde le bon chemin);
- M6 référence avec distances exactes: aire ROC de la valeur entre distances d et d + 2 (préfixes du jeu de test);
- M7 petite recherche sur mélanges: longueur moyenne trouvée, faisceau 2^7 à deux racines, 50 mélanges de longueur
  12, 14, 16 et 18.
M1-M5 sont moyennées sur les longueurs 9 à 14 (principal) et 12 à 18; M6 sur 9 à 14. Corrélations de rang avec la
longueur moyenne de la recherche (2 racines, 2^10, 1000 cubes de test, cubes résolus par tous), sur les 10 réseaux
et sur les 8 entraînés au même nombre d'exemples (236M).
Usage: python experiments/pilots/training_lengths/walk_accuracy.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
sys.path.insert(0, "experiments/cube")
import beam_torch as bt
import cube_torch as ct
import deepcubea_testset as dt
import fast_cube as fc
import search as lab
import network_torch as nt

NETS = ["uniform30", "uniform", "uniform_s1", "focus", "focus_s1", "focus_k10", "cut_tail", "cut_front", "bandit",
        "bandit2"] + sys.argv[1:]          # réseaux ajoutés: runs/curcpu/<nom>_v, runs/curcpu/eval_<nom>
EQUAL = [n for n in NETS if n not in ("uniform30", "uniform")]
torch.set_num_threads(16)
cube = ct.TorchCube("cpu")

# mélanges fixes: 4000 marches de 26 coups, un état par préfixe
walks, labels, depth = cube.self_supervised_batch(4000, 26, torch.Generator().manual_seed(2026))
depth = depth.numpy()
# préfixes du jeu de test, distance exacte
prefix, dist = [], []
for scramble in json.load(open(dt.PATH)):
    st = fc.SOLVED
    for j, m in enumerate(scramble.split(), 1):
        st = fc.apply_sequence(st, [fc.ACTIONS.index(m)])
        prefix.append(st)
        dist.append(j)
prefix, dist = cube.from_numpy(np.stack(prefix)), np.array(dist)
# petite recherche: 50 mélanges de chaque longueur, état final d'une marche
small = []
for k, seed in zip((12, 14, 16, 18), (1, 2, 3, 4)):
    w, _ = cube.random_walks(50, k, torch.Generator().manual_seed(9000 + seed))
    small += [(k, w[-1][i]) for i in range(50)]


def auc(lo, hi):
    both = np.concatenate([lo, hi])
    ranks = np.argsort(np.argsort(both, kind="stable"), kind="stable") + 1.0
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, ranks) / counts)[inv]
    return float((ranks[len(lo):].sum() - len(hi) * (len(hi) + 1) / 2) / (len(lo) * len(hi)))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def measures(path):
    net, _ = nt.load(path)
    h = bt.NetworkHeuristic(net, cube, "ac", 1 << 14, 1.0)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = h.head_outputs(walks, ("policy", "value"))
        v_prefix = h.outputs(prefix, "value").float().numpy()
    logp = F.log_softmax(out["policy"].float(), dim=1)
    v = out["value"].float().squeeze(-1) if out["value"].dim() > 1 else out["value"].float()
    cost = (v[:, None] - logp).numpy()                     # coût des 12 enfants, λ = 1
    lab_cost = cost[np.arange(len(cost)), labels.numpy()]
    hit = (logp.argmax(1) == labels).numpy()
    ce = (-logp[np.arange(len(logp)), labels]).numpy()
    v = v.numpy()
    deep = cost[depth >= 20].ravel()
    deep_cut = np.quantile(deep, 0.1)
    m = {k: {} for k in range(1, 27)}
    for k in range(1, 27):
        s = depth == k
        m[k]["M1"] = float(hit[s].mean())
        m[k]["M2"] = float(ce[s].mean())
        if k <= 24:
            m[k]["M3"] = auc(v[s], v[depth == k + 2])          # P(V(k + 2) > V(k)): k + 2 jugée plus loin
        m[k]["M4"] = float((lab_cost[s] <= np.quantile(cost[s].ravel(), 0.1)).mean())
        m[k]["M5"] = float((lab_cost[s] <= deep_cut).mean())
    m6 = {d: auc(v_prefix[dist == d], v_prefix[dist == d + 2]) for d in range(4, 19)}
    found = []
    lh = lab.LabHeuristic(net, cube, 128, chunk=1 << 14)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for k, state in small:
            r = lab.solve(lh, state, 128, ["id", "inv"])
            found.append(len(r["solution"]) if r["solution"] is not None else 40)
    return m, m6, float(np.mean(found))


def search_length():
    rows = {}
    for n in NETS:
        f = f"runs/curcpu/eval_{n}/roots2_all_w1024.jsonl"
        rows[n] = {r["i"]: r["length"] for r in map(json.loads, open(f)) if r["length"] is not None}
    common = set.intersection(*(set(r) for r in rows.values()))
    return {n: float(np.mean([rows[n][i] for i in common])) for n in NETS}, len(common)


if __name__ == "__main__":
    search, n_common = search_length()
    res = {}
    for n in NETS:
        m, m6, m7 = measures(f"runs/curcpu/{n}_v/final.pt")
        res[n] = {"search": search[n], "per_length": m, "M6": m6, "M7": m7}
        for name, ks in (("9-14", range(9, 15)), ("12-18", range(12, 19))):
            for key in ("M1", "M2", "M3", "M4", "M5"):
                res[n][f"{key}_{name}"] = float(np.mean([m[k][key] for k in ks if key in m[k]]))
        res[n]["M6_9-14"] = float(np.mean([m6[d] for d in range(9, 15)]))
        print(f"{n:10} recherche {search[n]:.3f} | " + " ".join(
            f"{key} {res[n][key]:.4f}" for key in ("M1_9-14", "M2_9-14", "M3_9-14", "M4_9-14", "M5_9-14",
                                                    "M6_9-14")) + f" | M7 {m7:.3f}", flush=True)
    keys = [f"{k}_{r}" for k in ("M1", "M2", "M3", "M4", "M5") for r in ("9-14", "12-18")] + ["M6_9-14", "M7"]
    sign = {"M2": 1, "M7": 1}          # plus petit = meilleur pour M2, M7 et la longueur de recherche
    print(f"\nCorrélation de rang avec la longueur de recherche ({n_common} cubes résolus par tous) ; "
          "négative = la mesure classe comme la recherche (sauf M2, M7 : positive)")
    corr = {}
    for key in keys:
        for group, names in (("10", NETS), ("8", EQUAL)):
            corr[f"{key}/{group}"] = spearman([res[n][key] for n in names], [res[n]["search"] for n in names])
        print(f"  {key:9} 10 réseaux {corr[key + '/10']:+.2f}   8 à exemples égaux {corr[key + '/8']:+.2f}")
    json.dump({"nets": res, "corr": corr, "n_common": n_common}, open("runs/cpu10/proxy.json", "w"), indent=1)
