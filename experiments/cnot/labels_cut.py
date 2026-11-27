"""
Deux des mesures de la section 5, refaites sur la synthèse CNOT (l'oracle demande des circuits optimaux, inconnus
à 8 et 9 qubits).

1. Première coupe du faisceau. Le graphe est un graphe de Cayley, donc toute matrice a |S_t| matrices à t coups,
   comme l'identité: depuis M et M⁻¹, le niveau t tient 2|S_t| états, et un faisceau de largeur W garde tout tant que
   2|S_t| <= W. Vérifié en comptant, niveau par niveau, les enfants neufs et distincts du faisceau sur les premières
   matrices de test.
2. Étiquettes des marches. Marches tirées par la règle d'entraînement (jamais le coup précédent); distance exacte de
   chaque préfixe par une table de rayon D et un parcours en largeur autour de l'état: si l'état n'est pas dans la
   table, d = D + j, avec j la première couche autour de lui qui touche la table. Pas d'hypothèse de graphe biparti:
   un CNOT peut laisser la distance inchangée.

Usage: python experiments/cnot/labels_cut.py --model models/cnot/8-H.pt --instances experiments/cnot/inst/ibm8qJ_test.json \
           --out runs/cnot/labels_cut_8H.json
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
import beam_torch as bt  # noqa: E402
import circuit_torch as qt  # noqa: E402
import network_torch as nt  # noqa: E402
import perm_search as ps  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--instances", required=True)
parser.add_argument("--table-radius", type=int, default=9)
parser.add_argument("--reach", type=int, default=6, help="couches cherchées autour d'un état: distances <= D + reach")
parser.add_argument("--widths", default="64,256,1024,4096,16384")
parser.add_argument("--cut-cases", type=int, default=20)
parser.add_argument("--walks", type=int, default=400)
parser.add_argument("--length", type=int, default=24)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--threads", type=int, default=16)
parser.add_argument("--out", required=True)
args = parser.parse_args()
torch.set_num_threads(args.threads)

model, meta = nt.load(args.model, torch.device("cpu"))
train = meta["args"]
p = qt.make_puzzle(train["puzzle"], train["n"], train.get("topology", "line"))
t0 = time.time()
ball = ps.Ball(p, args.table_radius)
D, spheres = ball.radius, ball.spheres
CAP = D + args.reach
print(f"{train['topology']}: {p.n_moves} coups, table de rayon {D} ({len(ball.keys):,} matrices), sphères {spheres}, "
      f"{time.time() - t0:.0f} s", flush=True)
out = {"args": vars(args), "layout": train["topology"], "moves": p.n_moves, "table_radius": D, "spheres": spheres}
h = bt.NetworkHeuristic(model, p, mode="ac", lam=1.0)      # lambda sans effet: on compte avant la première coupe

# --- 1. première coupe ------------------------------------------------------------------------------------------
with open(args.instances) as f:
    spec = json.load(f)
starts = p.from_matrices(spec["instances"][:args.cut_cases])
widths = [int(w) for w in args.widths.split(",")]
predicted = {w: next((t for t in range(1, D + 1) if 2 * spheres[t] > w), None) for w in widths}


@torch.no_grad()
def level_counts(start, width, max_depth=12):
    """enfants neufs et distincts à chaque niveau, jusqu'au premier niveau qui en a plus que width."""
    views = bt.equivalent_views(p, start, ["id", "inv"])
    keys = p.state_hash(torch.stack([s for s, _ in views])).tolist()
    roots = torch.stack([views[i][0] for i, k in enumerate(keys) if k not in keys[:i]])
    beam = bt._Beam(p, roots, roots)
    counts = []
    for _ in range(max_depth):
        kids, keys, cost, cum, _ = beam.expand(h, lambda x: x)
        counts.append(int(torch.unique(keys[torch.isfinite(cost)]).numel()))
        if counts[-1] > width:
            break
        best, n_best = beam.candidates(keys, cost, width)
        beam.select(kids, keys, cum, best[:int(n_best)])
    return counts


cut = {}
for w in widths:
    rows = [level_counts(s, w) for s in starts]
    first = [len(c) if c[-1] > w else None for c in rows]
    whole = all(c[t - 1] == 2 * spheres[t] for c in rows for t in range(1, len(c) + 1) if t <= D)
    cut[w] = {"predicted": predicted[w], "observed": first, "counts_match_2S": whole}
    print(f"largeur {w}: première coupe prévue au niveau {predicted[w]}, observée {sorted(set(first))}, "
          f"comptes = 2|S_t|: {whole}", flush=True)
out["first_cut"] = cut


# --- 2. étiquettes des marches -------------------------------------------------------------------------------------
@torch.no_grad()
def distance(state, cap):
    """distance exacte de state si elle est <= cap, sinon None."""
    found, r = ball.lookup(p.state_hash(state[None]))
    if bool(found[0]):
        return int(r[0])
    frontier, older = state[None], torch.empty(0, dtype=torch.long)
    last = p.state_hash(frontier)
    for j in range(1, cap - D + 1):
        kids = p.children(frontier).reshape(-1, p.n)
        k, first = ps.Ball.first_of_each(p.state_hash(kids))
        hit, r = ball.lookup(k)
        if bool(hit.any()):
            return j + int(r[hit].min())
        # voisins de la couche j - 1: couches j - 2, j - 1 et j (graphe non orienté)
        seen = torch.cat([older, last]).sort().values
        pos = torch.searchsorted(seen, k).clamp(max=len(seen) - 1)
        new = seen[pos] != k
        frontier, older, last = kids[first[new]], last, k[new]
    return None


gen = torch.Generator().manual_seed(args.seed)
states, actions = p.random_walks(args.walks, args.length, gen)        # (longueur, marches, n), (longueur, marches)
L, W = args.length, args.walks
dist = np.full((L + 1, W), -1, dtype=np.int64)
dist[0] = 0
t0 = time.time()
for k in range(1, L + 1):
    for w in range(W):
        d = distance(states[k - 1, w], CAP)
        dist[k, w] = -1 if d is None else d
    print(f"k={k}: {np.mean(dist[k] >= 0):.1%} des distances certifiées, {time.time() - t0:.0f} s", flush=True)

with torch.no_grad():
    top = h.outputs(states.reshape(-1, p.n), "policy").argmax(1).reshape(L, W)
label = actions                                                       # chaque CNOT est son propre inverse
improves = np.full((L, W), -1, dtype=np.int64)                        # -1: inconnu
for k in range(1, L + 1):
    for w in range(W):
        d = dist[k, w]
        if d < 0:
            continue
        if int(top[k - 1, w]) == int(label[k - 1, w]):
            improves[k - 1, w] = int(dist[k - 1, w] >= 0 and dist[k - 1, w] < d)
            continue
        child = p.apply_actions(states[k - 1, w][None], top[k - 1, w][None])[0]
        dc = distance(child, d - 1) if d >= 1 else None                # plus proche si et seulement si d(enfant) <= d - 1
        improves[k - 1, w] = int(dc is not None)

rows = []
for k in range(1, L + 1):
    known = (dist[k] >= 0) & (dist[k - 1] >= 0)
    d, dp = dist[k][known], dist[k - 1][known]
    hit = (top[k - 1] == label[k - 1]).numpy()
    rows.append({"length": k, "certified": float(known.mean()), "shortest": float(np.mean(d == k)),
                 "gap_mean": float(np.mean(k - d)), "label_improving": float(np.mean(d > dp)),
                 "label_sideways": float(np.mean(d == dp)), "net_top1_label": float(hit[known].mean()),
                 "net_top_improves": float(np.mean(improves[k - 1][known] == 1))})
    r = rows[-1]
    print(f"k={k:2} | certifiées {r['certified']:6.1%} | plus court chemin {r['shortest']:6.1%} | k - d {r['gap_mean']:.2f}"
          f" | étiquette améliorante {r['label_improving']:6.1%} (neutre {r['label_sideways']:5.1%}) | réseau: étiquette "
          f"{r['net_top1_label']:6.1%}, coup améliorant {r['net_top_improves']:6.1%}", flush=True)
out["labels"] = rows
out["dist"] = dist.tolist()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print("fini", flush=True)
