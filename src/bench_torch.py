"""
Mesure des débits sur un device, pour calibrer le plan d'une session GPU :
- génération des données (états par seconde) ;
- entraînement (exemples par seconde, génération comprise) pour plusieurs architectures ;
- recherche en faisceau (évaluations du réseau par seconde) pour plusieurs largeurs.

Usage :
    python src/bench_torch.py --seconds 20 --archs 5000x1000x4,2000x1000x2 --widths 512,8192,65536
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import beam_torch as bt
import cube_torch as ct
import fast_cube as fc
import network_torch as nt


def timed(fn, seconds, sync):
    """Répète fn pendant `seconds` ; retourne (somme des quantités renvoyées par fn, durée)."""
    fn()                                # préchauffage
    sync()
    total, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        total += fn()
    sync()
    return total, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Débits de génération, d'entraînement et de recherche")
    parser.add_argument("--seconds", type=float, default=10.0, help="durée de chaque mesure")
    parser.add_argument("--archs", default="5000x1000x4,2000x1000x2", help="hidden x width x blocs, séparés par des virgules")
    parser.add_argument("--walks", type=int, default=1000)
    parser.add_argument("--length", type=int, default=26)
    parser.add_argument("--widths", default="512,8192")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out")
    args = parser.parse_args()

    device = torch.device(args.device)
    amp = device.type == "cuda"
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    cube = ct.TorchCube(device)
    results = {"device": str(device), "gpu": torch.cuda.get_device_name() if amp else None, "torch": torch.__version__}

    n, dt = timed(lambda: len(cube.self_supervised_batch(args.walks, args.length)[0]), args.seconds, sync)
    results["generation_states_per_s"] = round(n / dt)
    print(f"génération : {n / dt:,.0f} états/s", flush=True)

    results["training"] = []
    models = []
    for arch in args.archs.split(","):
        hidden, width, blocks = (int(x) for x in arch.split("x"))
        net = nt.ResMLP("pieces", hidden, width, blocks, ("policy",)).to(device)
        model = torch.compile(net) if args.compile else net
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        def step():
            states, labels, _ = cube.self_supervised_batch(args.walks, args.length)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(cube.encode(states))["policy"]
            loss = F.cross_entropy(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            return len(states)

        net.train()
        n, dt = timed(step, args.seconds, sync)
        entry = {"arch": arch, "parameters_m": round(net.n_parameters() / 1e6, 2), "examples_per_s": round(n / dt)}
        results["training"].append(entry)
        print(f"entraînement {arch} ({entry['parameters_m']} M paramètres) : {n / dt:,.0f} exemples/s", flush=True)
        models.append((arch, net.eval()))

    results["search"] = []
    state = cube.from_numpy(fc.scramble(np.random.default_rng(0), 30)[0])
    for arch, net in models:
        for mode in ("policy", "value"):
            if mode == "value":   # la recherche par la valeur a besoin d'une tête value
                net = nt.ResMLP("pieces", net.config["hidden"], net.config["width"], net.config["blocks"],
                                ("value",)).to(device).eval()
            heuristic = bt.NetworkHeuristic(net, cube, mode)
            for width in (int(w) for w in args.widths.split(",")):
                n, dt = timed(lambda: bt.solve(heuristic, state, width=width, max_depth=8)["evaluations"], args.seconds, sync)
                entry = {"arch": arch, "mode": mode, "width": width, "evaluations_per_s": round(n / dt)}
                results["search"].append(entry)
                print(f"recherche {arch} {mode} largeur {width} : {n / dt:,.0f} évaluations/s", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
