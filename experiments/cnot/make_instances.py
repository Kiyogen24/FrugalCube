"""
jeux d'instances communs à notre solveur (eval_cnot.py) et aux références IBM/Qiskit (ibm_linear.py, dans .qenv):
matrices uniformes de GL(n, 2) (cnot) ou permutations uniformes (swap), en JSON, lignes 0/1 (M[i][j] = bit j de la
ligne i; le circuit réalise x -> M x).

Usage: python experiments/cnot/make_instances.py --puzzle cnot --n 9 --topology line --count 200 --seed 2026 \
            --out experiments/cnot/inst/cnot_line9_test.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import circuit_torch as qt  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--puzzle", choices=["cnot", "swap"], default="cnot")
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--topology", default="line")
parser.add_argument("--count", type=int, default=200)
parser.add_argument("--seed", type=int, default=2026, help="2026 : test ; 7 : validation (réglage de lambda)")
parser.add_argument("--out", required=True)
args = parser.parse_args()

p = qt.make_puzzle(args.puzzle, args.n, args.topology)
states = p.random_states(args.count, torch.Generator().manual_seed(args.seed))
if args.puzzle == "cnot":
    instances = p.to_matrices(states).tolist()
else:
    instances = states.long().tolist()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"puzzle": args.puzzle, "n": args.n, "topology": args.topology, "edges": p.edges, "seed": args.seed,
               "instances": instances}, f)
print(f"{len(instances)} instances -> {args.out}")
