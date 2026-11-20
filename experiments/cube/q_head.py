"""
Juger chaque enfant sur sa propre valeur V(s.a) vaut environ
un doublement de largeur à w1024, mais coûte 13 appels du réseau par parent. Une tête Q(s, a) qui prédit les 12
valeurs des enfants en un seul appel rend-elle ce gain? Tête ajoutée au tronc figé de m90_v (distillation, pas de
RL): Q(s, .) = V(s) + MLP(z(s)), apprise sur l'écart V(s.a) - V(s) donné par le même réseau, sur les préfixes de
mélanges aléatoires. Recherche: coût Q(s, a) - lamb log pi(a | s), un appel par parent.
Usage:
    python experiments/cube/q_head.py train --minutes 30
    python experiments/cube/q_head.py finetune --minutes 75 --lr 3e-4 --init-head runs/gates/qhead/qhead.pt \
        --out-dir runs/gates/qhead_ft
    python experiments/cube/q_head.py search lamb=0 lamb=0.5 [--seed 777] [--width 1024]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
sys.path.insert(0, "experiments/cube")
import cube_torch as ct
import search as lab
import network_torch as nt

VIEWS = ["id", "inv", "c15", "c20"]


class QNet(nn.Module):
    """le réseau entraîné plus la tête q: q (N, 12) = V(s) + écart appris sur les caractéristiques du tronc."""

    def __init__(self, net, hidden=1024):
        super().__init__()
        self.net = net
        self.q = nn.Sequential(nn.Linear(net.config["width"], hidden), nn.ReLU(), nn.Linear(hidden, 12))
        self.config = dict(net.config, heads=net.config["heads"] + ["q"])

    def forward(self, x):
        z = self.net.body(x)
        out = {h: layer(z) for h, layer in self.net.heads.items()}
        out["value"] = out["value"].squeeze(-1)
        out["q"] = out["value"][:, None] + self.q(z)
        return out


class QHeuristic(lab.LabHeuristic):
    def __call__(self, states, kids, cum):
        out = self.head_outputs(states, ("policy", "value", "q"))
        v, logp = out["value"], torch.log_softmax(out["policy"], dim=1)
        self.last_v = v
        self.trace.append(torch.stack([v.min(), v.median()]))
        return out["q"] - self.lamb * logp, cum[:, None] - logp, len(states)


@torch.no_grad()
def teacher(net, cube, states, chunk=1 << 15):
    """Caractéristiques du tronc z(s), V(s) et l'écart cible V(s.a) - V(s) (N, 12), réseau figé, en bf16."""
    encode = cube.encode_stickers
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z = net.body(encode(states)).float()
        v = net.heads["value"](z).squeeze(-1).float()
        kids = cube.children(states).reshape(-1, 54)
        vk = torch.cat([net(encode(kids[i:i + chunk]))["value"].float() for i in range(0, len(kids), chunk)])
    return z, v, vk.reshape(-1, 12) - v[:, None]


def batch(cube, walks, length, generator):
    states, _ = cube.random_walks(walks, length, generator)
    return states.reshape(-1, 54)


def train(args):
    cube = ct.TorchCube("cpu")
    net, _ = nt.load(args.model, "cpu")
    net.requires_grad_(False)
    model = QNet(net, args.hidden)
    optimizer = torch.optim.Adam(model.q.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.train_seed)
    # validation fixe, mélanges d'une autre graine; référence: la politique seule, qui choisit argmax pi
    zv, vv, dv = teacher(net, cube, batch(cube, 400, args.length, torch.Generator().manual_seed(12345)))
    with torch.no_grad():
        policy_top1 = (net.heads["policy"](zv).argmax(1) == dv.argmin(1)).float().mean().item()
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(os.path.join(args.out_dir, "train_log.jsonl"), "a")
    start, steps, examples, next_log = time.time(), 0, 0, 0.0
    budget = args.minutes * 60
    while (elapsed := time.time() - start) < budget:
        for group in optimizer.param_groups:
            group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * elapsed / budget))
        z, _, d = teacher(net, cube, batch(cube, args.walks, args.length, generator))
        loss = ((model.q(z) - d) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps, examples = steps + 1, examples + len(z)
        if elapsed >= next_log or time.time() - start >= budget:
            with torch.no_grad():
                p = model.q(zv)
                mse = ((p - dv) ** 2).mean().item()
                row = {"minutes": round(elapsed / 60, 2), "steps": steps, "examples": examples,
                       "train_mse": round(loss.item(), 4), "val_mse": round(mse, 4),
                       "val_r2": round(1 - mse / dv.var().item(), 4),
                       "val_top1": round((p.argmin(1) == dv.argmin(1)).float().mean().item(), 4),
                       "policy_top1": round(policy_top1, 4)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            next_log = elapsed + args.log_seconds
    torch.save({"q": model.q.state_dict(), "hidden": args.hidden, "model": args.model, "steps": steps,
                "examples": examples}, os.path.join(args.out_dir, "qhead.pt"))


def finetune(args):
    """
    tout le réseau apprend la cible Q (tronc compris), tête q partie de celle du tronc figé (--init-head);
    politique et valeur restent ancrées sur celles du réseau de départ (KL des politiques, écart des valeurs).
    """
    cube = ct.TorchCube("cpu")
    frozen, _ = nt.load(args.model, "cpu")
    frozen.requires_grad_(False)
    student, _ = nt.load(args.model, "cpu")
    model = QNet(student, args.hidden)
    if args.init_head:
        model.q.load_state_dict(torch.load(args.init_head, weights_only=True)["q"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.train_seed)
    sv = batch(cube, 400, args.length, torch.Generator().manual_seed(12345))
    zv, vv, dv = teacher(frozen, cube, sv)
    with torch.no_grad():
        pv = torch.log_softmax(frozen.heads["policy"](zv), dim=1)
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(os.path.join(args.out_dir, "train_log.jsonl"), "a")
    start, steps, examples, next_log = time.time(), 0, 0, 0.0
    budget = args.minutes * 60
    while (elapsed := time.time() - start) < budget:
        for group in optimizer.param_groups:
            group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * elapsed / budget))
        s = batch(cube, args.walks, args.length, generator)
        zt, vt, dt = teacher(frozen, cube, s)
        with torch.no_grad():
            pt = torch.log_softmax(frozen.heads["policy"](zt), dim=1)
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(cube.encode_stickers(s))
        q, v, logp = out["q"].float(), out["value"].float(), torch.log_softmax(out["policy"].float(), dim=1)
        loss_q = ((q - (vt[:, None] + dt)) ** 2).mean()
        loss_anchor = (pt.exp() * (pt - logp)).sum(1).mean() + ((v - vt) ** 2).mean()
        loss = loss_q + args.anchor * loss_anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps, examples = steps + 1, examples + len(s)
        if elapsed >= next_log or time.time() - start >= budget:
            model.eval()
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                o = model(cube.encode_stickers(sv))
            p = o["q"].float() - vv[:, None]
            mse = ((p - dv) ** 2).mean().item()
            kl = (pv.exp() * (pv - torch.log_softmax(o["policy"].float(), dim=1))).sum(1).mean().item()
            row = {"minutes": round(elapsed / 60, 2), "steps": steps, "examples": examples,
                   "train_q": round(loss_q.item(), 4), "train_anchor": round(loss_anchor.item(), 4),
                   "val_mse": round(mse, 4), "val_r2": round(1 - mse / dv.var().item(), 4),
                   "val_top1": round((p.argmin(1) == dv.argmin(1)).float().mean().item(), 4),
                   "val_kl": round(kl, 4), "val_v_mse": round(((o["value"].float() - vv) ** 2).mean().item(), 4)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            next_log = elapsed + args.log_seconds
    torch.save({"q": model.q.state_dict(), "net": student.state_dict(), "hidden": args.hidden, "model": args.model,
                "steps": steps, "examples": examples}, os.path.join(args.out_dir, "qhead.pt"))


def search(args):
    torch.set_num_threads(16)
    cube = ct.TorchCube("cpu")
    net, _ = nt.load(args.model, "cpu")
    saved = torch.load(os.path.join(args.out_dir, "qhead.pt"), weights_only=True)
    if "net" in saved:   # réseau entier affiné (finetune)
        net.load_state_dict(saved["net"])
    model = QNet(net, saved["hidden"])
    model.q.load_state_dict(saved["q"])
    model.eval()
    cases = list(enumerate(lab.fresh_cubes(cube, args.fresh, args.seed)))
    ref_path = args.reference or f"runs/gates/base_fresh{args.seed}_w{args.width}.jsonl"
    res_dir = args.results_dir or args.out_dir
    os.makedirs(res_dir, exist_ok=True)
    reference = None
    if os.path.exists(ref_path):
        reference = {r["i"]: r for r in map(json.loads, open(ref_path))}
        reference = [reference[i] for i, _ in cases]
        lab.summary("référence (base)", reference, None)
    for config in args.configs:
        options = lab.parse(config)
        h = QHeuristic(model, cube, args.width, chunk=1 << 14, **options)
        name = "q_" + config.replace("=", "").replace(",", "_")
        out = f"{res_dir}/{name}_fresh{args.seed}_w{args.width}.jsonl"
        rows = [json.loads(line) for line in open(out)] if os.path.exists(out) else []
        with open(out, "a") as f:
            for i, state in cases[len(rows):]:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    r = lab.solve(h, state, args.width, VIEWS)
                row = {"i": i, "length": len(r["solution"]) if r["solution"] is not None else None,
                       "optimal": None, "time_s": r["time_s"], "evaluations": r["evaluations"]}
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
        lab.summary(name, rows, reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "finetune", "search"])
    parser.add_argument("configs", nargs="*", help="search : lamb=..., une recherche par réglage")
    parser.add_argument("--model", default="runs/s4/m90_v/final.pt")
    parser.add_argument("--out-dir", default="runs/gates/qhead")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--walks", type=int, default=256)
    parser.add_argument("--length", type=int, default=26)
    parser.add_argument("--log-seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=777, help="cubes neufs de la recherche")
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--init-head", help="finetune : tête q de départ (qhead.pt du tronc figé)")
    parser.add_argument("--anchor", type=float, default=1.0, help="finetune : poids de l'ancrage politique et valeur")
    parser.add_argument("--fresh", type=int, default=200)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--views", default="id,inv,c15,c20", help="search : racines du faisceau, par ex. id,inv")
    parser.add_argument("--results-dir", help="search : dossier des résultats (défaut : --out-dir)")
    parser.add_argument("--reference", help="search : jsonl de base pour l'appariement")
    args = parser.parse_args()
    VIEWS[:] = args.views.split(",")
    torch.set_num_threads(16)
    {"train": train, "finetune": finetune, "search": search}[args.mode](args)
