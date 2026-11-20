"""
politique de contrôle du faisceau apprise par stratégies d'évolution (Salimans et al. 2017), sans toucher au
réseau ni au jeu de test. Le coût d'un enfant (p, a) est
    V(p) - lam(m) * log pi(a | p) + eta(m) * H(pi(. | p)) + mu * (-log pi cumulé le long du chemin),
où m est la valeur médiane du niveau du faisceau (où en est la recherche dans la bande) et lam, eta sont
interpolés linéairement entre 5 nœuds de m. La récompense d'un épisode est moins la longueur de la solution
trouvée, sur des cubes tirés au hasard (200 coups aléatoires), neufs à chaque génération; les paires
antithétiques sont jugées sur les mêmes cubes.
Départ: la règle de bande (lam = 2 pour m >= 15, 1 sinon; eta = 0).
Usage: python experiments/cube/es.py --device cuda --width 16384 --cubes 128 --generations 25
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
sys.path.insert(0, "experiments/cube")
import beam_torch as bt
import cube_torch as ct
import search as lab
import network_torch as nt

KNOTS = [12.0, 14.0, 16.0, 18.0, 20.0]


def interpolate(m, knots, values):
    """interpolation linéaire de values aux nœuds knots, en m (tenseur 0-d), bornée, sans synchronisation."""
    m = m.clamp(knots[0], knots[-1])
    i = torch.searchsorted(knots, m).clamp(1, len(knots) - 1)
    w = (m - knots[i - 1]) / (knots[i] - knots[i - 1])
    return values[i - 1] * (1 - w) + values[i] * w


class ControllerHeuristic(bt.NetworkHeuristic):
    def __init__(self, net, cube, theta, chunk=1 << 17):
        super().__init__(net, cube, "ac", chunk, 1.0)
        theta = torch.as_tensor(theta, dtype=torch.float32, device=cube.device)
        k = len(KNOTS)
        self.knots = torch.tensor(KNOTS, device=cube.device)
        self.lams, self.etas = theta[:k].exp(), theta[k:2 * k]
        # poids du -log pi cumulé le long du chemin (absent des contrôleurs à 10 paramètres); négatif, il
        # favoriserait les chemins improbables: borné à 0
        self.mu = theta[2 * k].clamp(min=0) if len(theta) > 2 * k else None

    def __call__(self, states, kids, cum):
        n = len(states)
        out = self.head_outputs(states, ("policy", "value"))
        v, logp = out["value"], torch.log_softmax(out["policy"], dim=1)
        m = v.median()
        entropy = -(logp.exp() * logp).sum(1, keepdim=True)
        cost = v[:, None] - interpolate(m, self.knots, self.lams) * logp + interpolate(m, self.knots, self.etas) * entropy
        nll = cum[:, None] - logp
        if self.mu is not None:
            cost = cost + self.mu * nll
        return cost, nll, n


def band_theta():
    lams = [0.0 if k < 15 else np.log(2.0) for k in KNOTS]
    return np.array(lams + [0.0] * len(KNOTS) + [0.0])


def mean_length(net, cube, theta, cubes, width, failed=40):
    """Longueur moyenne trouvée; une recherche sans solution compte pour `failed` coups."""
    h = ControllerHeuristic(net, cube, theta)
    solutions = [lab.solve(h, s, width, VIEWS)["solution"] for s in cubes]
    lengths = [len(s) if s is not None else failed for s in solutions]
    return float(np.mean(lengths)), lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/s4/m90_v/final.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=16384)
    parser.add_argument("--cubes", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--out", default="runs/gpu5/es/es.jsonl")
    parser.add_argument("--stop", type=float, default=0, help="aucune génération ne démarre après (epoch)")
    args = parser.parse_args()

    torch.set_num_threads(16)
    VIEWS = ["id", "inv", "c15", "c20"]
    cube = ct.TorchCube(args.device)
    net, _ = nt.load(args.model, args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log = [json.loads(line) for line in open(args.out)] if os.path.exists(args.out) else []
    theta = np.array(log[-1]["theta_next"]) if log else band_theta()
    adam_m, adam_v = np.zeros_like(theta), np.zeros_like(theta)
    for gen in range(len(log), args.generations):
        if args.stop and time.time() > args.stop:
            print(f"arrêt à l'heure limite après {gen} générations", flush=True)
            break
        t0 = time.time()
        g = torch.Generator(args.device).manual_seed(10_000 + gen)
        walks, _ = cube.random_walks(args.cubes, 200, g)
        cubes = list(walks[-1])
        rng = np.random.default_rng(gen)
        eps = rng.standard_normal((args.pairs, len(theta)))
        center, _ = mean_length(net, cube, theta, cubes, args.width)
        diffs = []
        for e in eps:
            plus, _ = mean_length(net, cube, theta + args.sigma * e, cubes, args.width)
            minus, _ = mean_length(net, cube, theta - args.sigma * e, cubes, args.width)
            diffs.append(plus - minus)
        grad = (np.array(diffs)[:, None] * eps).mean(0) / (2 * args.sigma)
        # Adam sur la longueur moyenne (à minimiser)
        adam_m = 0.9 * adam_m + 0.1 * grad
        adam_v = 0.999 * adam_v + 0.001 * grad ** 2
        step = args.lr * (adam_m / (1 - 0.9 ** (gen + 1))) / (np.sqrt(adam_v / (1 - 0.999 ** (gen + 1))) + 1e-8)
        theta_next = theta - step
        k = len(KNOTS)
        row = {"gen": gen, "center_length": center, "diffs": diffs, "theta": theta.tolist(),
               "theta_next": theta_next.tolist(), "lam": np.exp(theta[:k]).round(3).tolist(),
               "eta": theta[k:2 * k].round(3).tolist(), "mu": round(max(0.0, float(theta[2 * k])), 3) if len(theta) > 2 * k else None,
               "seconds": round(time.time() - t0, 1)}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"génération {gen} : longueur moyenne au centre {center:.3f}, lam {row['lam']}, eta {row['eta']}, "
              f"mu {row['mu']}, {row['seconds']:.0f} s", flush=True)
        theta = theta_next
