"""
ablation de la recherche pour l'article, sur le modèle final (m90) et les 1000 cubes DeepCubeA, à largeur égale:
score de la politique seule (cumul de -log pi, celui de Takano) ou score politique-valeur, racine unique ou 4 vues
de Kociemba, règle de bande, mu; et le score par valeur des enfants (DeepCubeA, CayleyPy), à largeur égale
(13 appels par parent) et à coût égal (largeur / 13).
Usage: python experiments/cube/ablation.py --width 1024
"""
import argparse
import json
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
sys.path.insert(0, "experiments/cube")
import beam_torch as bt
import cube_torch as ct
import search as lab
import network_torch as nt

VIEWS = ["id", "inv", "c15", "c20"]
VIEWS2 = ["id", "inv"]   # deux racines (s, s⁻¹), le solveur final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/s4/m90_v/final.pt")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--out-dir", default="runs/ablation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--configs", help="noms séparés par des virgules (défaut : tous)")
    args = parser.parse_args()
    torch.set_num_threads(16)
    cube = ct.TorchCube(args.device)
    net, _ = nt.load(args.model, args.device)
    cases = [(cube.from_numpy(s), o, i) for i, (s, o) in enumerate(lab.dt.load())][:args.cases]
    W, chunk = args.width, 1 << (14 if args.device == "cpu" else 17)
    configs = [   # (nom, heuristique, vues, largeur)
        ("policy_1v", bt.NetworkHeuristic(net, cube, "policy", chunk), ["id"], W),
        ("policy_4v", bt.NetworkHeuristic(net, cube, "policy", chunk), VIEWS, W),
        ("policy_2v", bt.NetworkHeuristic(net, cube, "policy", chunk), VIEWS2, W),
        ("pv_2v", lab.LabHeuristic(net, cube, W, chunk=chunk), VIEWS2, W),
        ("childv_2v_eqcost", bt.NetworkHeuristic(net, cube, "value", chunk), VIEWS2, max(1, W // 13)),
        ("childv_2v", bt.NetworkHeuristic(net, cube, "value", chunk), VIEWS2, W),
        ("pv_1v", lab.LabHeuristic(net, cube, W, chunk=chunk), ["id"], W),
        ("pv_4v", lab.LabHeuristic(net, cube, W, chunk=chunk), VIEWS, W),
        ("pv_4v_band", lab.LabHeuristic(net, cube, W, lamb=2, tau=15, chunk=chunk), VIEWS, W),
        ("pv_4v_band_mu", lab.LabHeuristic(net, cube, W, lamb=2, tau=15, mu=0.1, chunk=chunk), VIEWS, W),
        ("childv_4v_eqcost", bt.NetworkHeuristic(net, cube, "value", chunk), VIEWS, max(1, W // 13)),
        ("childv_4v", bt.NetworkHeuristic(net, cube, "value", chunk), VIEWS, W),
    ]
    if args.configs:
        keep = args.configs.split(",")
        configs = [c for c in configs if c[0] in keep]
    os.makedirs(args.out_dir, exist_ok=True)
    reference = None
    for name, h, views, width in configs:
        out = f"{args.out_dir}/{name}_w{W}.jsonl"
        rows = [json.loads(line) for line in open(out)] if os.path.exists(out) else []
        with open(out, "a") as f:
            for state, optimal, i in cases[len(rows):]:
                # sur GPU, NetworkHeuristic active lui-même l'autocast bf16
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=args.device == "cpu"):
                    r = lab.solve(h, state, width, views)
                row = {"i": i, "length": len(r["solution"]) if r["solution"] is not None else None,
                       "optimal": optimal, "time_s": r["time_s"], "evaluations": r["evaluations"], "width": width}
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
        reference = reference or rows
        lab.summary(name, rows, reference)
