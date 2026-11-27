"""
AlphaCNOT (Cossio et al., arXiv 2604.13812, code MIT github.com/Jaccos01/AlphaCNOT, cloné dans ext/AlphaCNOT) sur nos
instances, avec ses modèles publiés. À lancer avec l'environnement .aenv (JAX, haiku, mctx) depuis la racine:

    .aenv/Scripts/python experiments/cnot/alphacnot_run.py --topology H --instances experiments/cnot/inst/ibm8qJ_test.json \
        --shots 1,10 --out runs/cnot/h2h/8qJ/alphacnot.jsonl

Même boucle que leur apply_AlphaCNOT / run_validation (gumbel_muzero_policy, réglages MCTS du checkpoint, au plus
4 n^2 coups), mais l'état de départ est notre matrice (mêmes conventions: le coup (c, t) ajoute la ligne c à la
ligne t, but = identité). "k shots" = meilleur de k épisodes à graines différentes (leur AlphaCNOT_100 = 100 shots).
Chaque solution est rejouée de notre côté (circuit_torch) pour vérifier qu'elle atteint l'identité.
Les checkpoints .pkl sont lus par un unpickler restreint aux trois classes numpy qu'ils contiennent.
Si leur topologie est la nôtre à une renumérotation des qubits près (8-F et 8-T1 d'AlphaCNOT contre 8qF et 8qT1
d'IBM), la matrice est renumérotée dans leur sens et leurs coups ramenés dans le nôtre: le nombre de CNOT ne change pas.
--start et --cases découpent un run entre plusieurs processus (mêmes graines qu'un run unique).
"""
import argparse
import functools
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ext", "AlphaCNOT"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mctx  # noqa: E402
from scripts import model_loading  # noqa: E402

ALLOWED = {("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}


def restricted_find_class(self, module, name):
    if (module, name) not in ALLOWED:
        raise pickle.UnpicklingError(f"classe refusée dans le checkpoint : {module}.{name}")
    return pickle.Unpickler.find_class(self, module, name)


model_loading._InferenceCheckpointUnpickler.find_class = restricted_find_class
from scripts.utils import load_alphacnot_model, get_recurrent_fn_simple  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=8)
parser.add_argument("--topology", required=True, help="nom AlphaCNOT : H, T2, T1, F, L, T, Y, ALL")
parser.add_argument("--instances", required=True)
parser.add_argument("--start", type=int, default=0, help="indice de la première instance")
parser.add_argument("--cases", type=int, default=0)
parser.add_argument("--shots", default="1,10", help="meilleur de k épisodes, pour chaque k")
parser.add_argument("--out", required=True)
args = parser.parse_args()

params, config, V_net, pi_net, env = load_alphacnot_model(args.n, args.topology, source="trained_models",
                                                           terminal_mode="identity")
num_actions = env.num_actions()
pairs = [tuple(int(x) for x in p) for p in np.asarray(env.topology)]
spec = json.load(open(args.instances))
if spec["n"] != args.n:
    raise ValueError("nombre de qubits")
ours = sorted({(min(a, b), max(a, b)) for a, b in spec["edges"]})
theirs = sorted({(min(a, b), max(a, b)) for a, b in pairs})
# notre qubit q est leur qubit perm[q]; l'identité est essayée en premier, donc rien ne change si les graphes sont égaux
perm = next((p for p in itertools.permutations(range(args.n))
             if sorted((min(p[a], p[b]), max(p[a], p[b])) for a, b in ours) == theirs), None)
if perm is None:
    raise ValueError(f"topologies différentes : instances {ours}, AlphaCNOT {theirs}")
inv = [perm.index(q) for q in range(args.n)]
first = args.start
raw = spec["instances"][first:first + args.cases if args.cases else None]
# leur matrice: M'[perm[i]][perm[j]] = M[i][j]
mats = [np.asarray(m, dtype=np.float32) for m in raw]
theirs_mats = [m[np.ix_(inv, inv)] for m in mats]
print(f"AlphaCNOT {args.n}{args.topology} : {num_actions} coups, {config['num_simulations']} simulations MCTS, "
      f"au plus {config.get('max_steps', 4 * args.n ** 2)} coups, renumérotation {list(perm)}", flush=True)

rec_fn = get_recurrent_fn_simple(env, V_net.apply, pi_net.apply, config)


@jax.jit
def get_action(params, key, state):
    obs = env.get_observation(state)
    obs_batch = jnp.expand_dims(obs.astype(float), 0)
    state_batch = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), state)
    root = mctx.RootFnOutput(prior_logits=jnp.reshape(pi_net.apply(params["pi"], obs_batch), (1, num_actions)),
                             value=jnp.reshape(V_net.apply(params["V"], obs_batch), (1,)), embedding=state_batch)
    out = mctx.gumbel_muzero_policy(
        params=params, rng_key=key, root=root, recurrent_fn=rec_fn, num_simulations=config["num_simulations"],
        max_depth=config.get("max_depth"), max_num_considered_actions=num_actions,
        gumbel_scale=config.get("gumbel_scale", 1.0),
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value,
                                     use_mixed_value=config.get("use_mixed_value", True),
                                     value_scale=config.get("value_scale", 0.1),
                                     maxvisit_init=config.get("maxvisit_init", 50.0),
                                     rescale_values=config.get("q_rescale", True),
                                     epsilon=config.get("q_epsilon", 1e-8)))
    return out.action[0]


max_steps = config.get("max_steps", 4 * args.n ** 2)


def episode(matrix, key):
    state = (jnp.asarray(matrix), jnp.array(0), jnp.array(0), jax.random.PRNGKey(0))
    moves = []
    for _ in range(max_steps):
        key, sub = jax.random.split(key)
        a = int(get_action(params, sub, state))
        state, _, _, done, info = env.step(a, state)
        c, t = pairs[a]
        moves.append((inv[c], inv[t]))                           # ramené dans notre numérotation
        if bool(info["is_solved"]):
            return moves
        if bool(done):
            return None
    return None


def replay(matrix, moves):
    rows = [int("".join(str(int(b)) for b in reversed(r)), 2) for r in matrix]
    for c, t in moves:
        rows[t] ^= rows[c]
    return rows == [1 << i for i in range(len(rows))]


episode(theirs_mats[0], jax.random.PRNGKey(1))                  # compilation
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
for k in [int(x) for x in args.shots.split(",")]:
    lengths, times = [], []
    for j, (m, mt) in enumerate(zip(mats, theirs_mats)):
        i = first + j                                            # indice global: mêmes graines qu'un run unique
        t0 = time.perf_counter()
        best = None
        for s in range(k):
            moves = episode(mt, jax.random.PRNGKey(1000 * i + s))
            if moves is not None:
                if not replay(m, moves):
                    raise RuntimeError(f"solution AlphaCNOT fausse (instance {i})")
                if best is None or len(moves) < len(best):
                    best = moves
        dt = time.perf_counter() - t0
        row = {"i": i, "searches": f"AlphaCNOT{k}", "time_s": round(dt, 4), "ok": best is not None,
               "length": None if best is None else len(best)}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        times.append(dt)
        if best is not None:
            lengths.append(len(best))
    print(f"AlphaCNOT {args.n}{args.topology} {k} shots : {len(lengths)}/{len(mats)} résolus, "
          f"CNOT moyen {np.mean(lengths):.2f}, {np.mean(times):.2f} s par instance", flush=True)
