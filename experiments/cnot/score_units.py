"""
Unités du score de la recherche CNOT, c(p, a) = V(p) - lambda log pi(a | p), et part de chaque terme dans le choix du
faisceau: la politique domine-t-elle la valeur avec lambda = 32 ou 64?

A. V sur les instances de test contre la longueur de nos solutions (largeur 16384): V est-il en coups?
B. Le long de nos solutions: V contre le nombre de coups restants, netteté de la politique, écart lambda x (log pi du
   meilleur coup - log pi du second), en coups.
C. Faisceau simple (une racine, sans table, doublons fusionnés) de largeur W depuis l'instance: à chaque niveau, sur
   les enfants candidats, variance des deux termes du coût, et part des W enfants gardés que garderait aussi le terme
   valeur seul ou le terme politique seul.

Usage: python experiments/cnot/score_units.py --model runs/cnot/h2h/8qJ/final.pt --instances experiments/cnot/inst/ibm8qJ_test.json \
            --solutions runs/cnot/h2h/8qJ/ss/eval16k.jsonl --lam 64
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import beam_torch as bt  # noqa: E402
import circuit_torch as qt  # noqa: E402
import network_torch as nt  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--instances", required=True)
parser.add_argument("--solutions", required=True, help="eval_cnot.py --save-circuits (une largeur)")
parser.add_argument("--lam", type=float, required=True)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--cases", type=int, default=20)
parser.add_argument("--levels", type=int, default=30)
parser.add_argument("--out", help="résultats en JSON (pour paper/scripts/make_figures.py)")
args = parser.parse_args()
result = {"lambda": args.lam, "width": args.width, "cases": args.cases}
torch.set_num_threads(1)

model, meta = nt.load(args.model, torch.device("cpu"))
train = meta["args"]
p = qt.make_puzzle(train["puzzle"], train["n"], train.get("topology", "line"), "cpu")
h = bt.NetworkHeuristic(model, p, mode="ac", lam=args.lam)
spec = json.load(open(args.instances))
starts = p.from_matrices(spec["instances"])
rows = [json.loads(x) for x in open(args.solutions)]
rows = {r["i"]: r for r in rows if "i" in r and r.get("circuit") is not None}
index = {pair: a for a, pair in enumerate(p.pairs)}


def heads(states):
    out = h.head_outputs(states, ("policy", "value"))
    return out["value"].reshape(-1), torch.log_softmax(out["policy"], dim=1)


# A. unités
ids = sorted(rows)
v0, _ = heads(starts[ids])
length = np.array([rows[i]["length"] for i in ids], dtype=float)
v0 = v0.numpy()
print(f"A. V(instance) moyen {v0.mean():.2f} (écart-type {v0.std():.2f}) ; nos solutions (largeur "
      f"{rows[ids[0]]['width']}) {length.mean():.2f} CNOT ; corrélation {np.corrcoef(v0, length)[0, 1]:.2f} ; "
      f"V - longueur moyen {np.mean(v0 - length):+.2f}")
result["A"] = {"v_mean": float(v0.mean()), "v_std": float(v0.std()), "length_mean": float(length.mean()),
               "corr": float(np.corrcoef(v0, length)[0, 1]), "solution_width": rows[ids[0]]["width"]}

# B. le long des solutions
by_r = {}
for i in ids:
    sol = [index[tuple(c)] for c in reversed(rows[i]["circuit"])]
    s = starts[i].clone()
    path = [s]
    for a in sol:
        s = p.apply_actions(s[None], torch.tensor([a]))[0]
        path.append(s)
    if not bool(p.is_solved(path[-1][None])[0]):
        raise RuntimeError(f"instance {i} : la solution ne mène pas à l'identité")
    states = torch.stack(path[:-1])
    v, lp = heads(states)
    top2 = lp.topk(2, dim=1).values
    for k in range(len(sol)):
        r = len(sol) - k
        by_r.setdefault(r, []).append((float(v[k]), float(lp[k].exp().max()), float(-lp[k, sol[k]]),
                                       float(args.lam * (top2[k, 0] - top2[k, 1])),
                                       int(lp[k].argmax() == sol[k])))
print("B. le long de nos solutions (r = coups restants sur notre solution, une borne supérieure de la distance)")
print("   r      V    max pi   -log pi(coup joué)   lambda x écart 1er-2e (coups)   coup joué = argmax pi")
for lo, hi in ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 60)):
    xs = [x for r in range(lo, hi + 1) for x in by_r.get(r, [])]
    if xs:
        a = np.array(xs)
        print(f"   {lo:2d}-{hi:2d}  {a[:, 0].mean():5.1f}   {a[:, 1].mean():.3f}    {a[:, 2].mean():5.2f}"
              f"                {np.median(a[:, 3]):6.2f} (médiane)               {a[:, 4].mean():.2f}")
        result.setdefault("B", {})[f"{lo}-{hi}"] = {
            "v": float(a[:, 0].mean()), "max_pi": float(a[:, 1].mean()), "nll_played": float(a[:, 2].mean()),
            "lam_gap_median": float(np.median(a[:, 3])), "played_is_argmax": float(a[:, 4].mean())}

# C. part de chaque terme dans le choix du faisceau
W = args.width
stats = {}
for i in ids[:args.cases]:
    beam = starts[i][None]
    for level in range(1, args.levels + 1):
        v, lp = heads(beam)
        kids = p.children(beam).reshape(-1, p.n)
        cv = v[:, None].expand_as(lp).reshape(-1)
        cp = (-args.lam * lp).reshape(-1)
        cost = cv + cp
        uniq, inv = torch.unique(kids, dim=0, return_inverse=True)
        best = torch.full((len(uniq),), float("inf")).scatter_reduce(0, inv, cost, reduce="amin")
        if len(uniq) <= W:
            beam = uniq
            continue
        # chaque terme seul, sur les mêmes candidats (le meilleur représentant de chaque état); V(p) est le même
        # pour les 14 enfants d'un parent: départage au hasard, sans la politique
        bv = torch.full((len(uniq),), float("inf")).scatter_reduce(0, inv, cv + 1e-6 * torch.rand_like(cv),
                                                                   reduce="amin")
        bp = torch.full((len(uniq),), float("inf")).scatter_reduce(0, inv, cp, reduce="amin")
        keep = set(best.topk(W, largest=False).indices.tolist())
        keep_v = set(bv.topk(W, largest=False).indices.tolist())
        keep_p = set(bp.topk(W, largest=False).indices.tolist())
        st = stats.setdefault(level, [])
        st.append((float(cv.var()), float(cp.var()), len(keep & keep_v) / W, len(keep & keep_p) / W))
        beam = uniq[best.topk(W, largest=False).indices]
        if bool(p.is_solved(beam).any()):
            break
print(f"C. faisceau de largeur {W}, une racine, sans table, {min(args.cases, len(ids))} instances, lambda {args.lam}")
print("   niveaux   var(V(p))   var(lambda x -log pi)   gardés aussi par V seul   gardés aussi par la politique seule")
for lo, hi in ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)):
    xs = [x for lv in range(lo, hi + 1) for x in stats.get(lv, [])]
    if xs:
        a = np.array(xs)
        print(f"   {lo:2d}-{hi:2d}     {a[:, 0].mean():7.2f}     {a[:, 1].mean():9.2f}             "
              f"{a[:, 2].mean():5.2f}                     {a[:, 3].mean():5.2f}")
        result.setdefault("C", {})[f"{lo}-{hi}"] = {
            "var_value_term": float(a[:, 0].mean()), "var_policy_term": float(a[:, 1].mean()),
            "kept_by_value_alone": float(a[:, 2].mean()), "kept_by_policy_alone": float(a[:, 3].mean())}
if args.out:
    json.dump(result, open(args.out, "w"), indent=1)
