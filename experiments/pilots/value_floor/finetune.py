"""
réentraînement de la tête valeur seule (tronc et politique figés), ou du réseau entier avec --trunk (perte de
politique auto-supervisée + --value-weight x perte de valeur), avec des cibles d'apprentissage par
renforcement, ciblées sur les positions profondes (marche de --k-deep coups ou plus), à nombre de pas égal:
- ss: témoin, auto-supervision (longueur de la marche) partout;
- exit_keep: auto-supervision partout + étiquettes de la recherche (labels.py);
- exit_drop: auto-supervision sur les marches courtes seulement, les positions profondes n'apprennent que des
              étiquettes de la recherche;
- bellman: positions profondes: 1 + min des valeurs des 12 enfants (copie cible de la tête), bornée par la
              longueur de la marche (CayleyPy RL, 2025); marches courtes: auto-supervision.
- policy: auto-supervision partout + la politique apprend aussi le coup des étiquettes (coup optimal connu, ou
              coup joué par la recherche, retrouvé le long des chemins de labels.py): itération experte pour la
              politique, ou borne haute avec les étiquettes exactes. Symétrie seule (le coup se transpose), pas
              d'inverse.
Les étiquettes de la recherche sont augmentées par une symétrie et un passage à l'inverse tirés au hasard (même
distance, et le chemin trouvé se transpose).
Usage: python experiments/pilots/value_floor/finetune.py --arm exit_drop --labels runs/floor/labels_w2048.pt --out runs/floor/exit_drop.pt
"""
import argparse
import copy
import json
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
import cube_torch as ct
import network_torch as nt

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["ss", "exit_keep", "exit_drop", "bellman", "policy"], required=True)
parser.add_argument("--model", default="runs/s4/m90_v/final.pt")
parser.add_argument("--labels")
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--walks", type=int, default=200)
parser.add_argument("--length", type=int, default=26)
parser.add_argument("--label-fraction", type=float, default=0.25, help="part du lot tirée des étiquettes")
parser.add_argument("--k-deep", type=int, default=12)
parser.add_argument("--target-every", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--trunk", action="store_true", help="le tronc et la politique apprennent aussi")
parser.add_argument("--value-weight", type=float, default=0.1, help="poids de la perte de valeur avec --trunk")
parser.add_argument("--label-weight", type=float, default=1.0, help="poids de la perte de politique des étiquettes")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save-every", type=int, default=0, help="points de contrôle <out>_s<pas>.pt en cours de route")
parser.add_argument("--out", required=True)
args = parser.parse_args()
assert args.trunk or args.arm != "policy", "le bras policy entraîne le réseau entier (--trunk)"

torch.set_num_threads(16)
torch.manual_seed(args.seed)
generator = torch.Generator().manual_seed(args.seed)
cube = ct.TorchCube("cpu")
net, meta = nt.load(args.model)
net.requires_grad_(args.trunk)
head = net.heads["value"]
head.requires_grad_(True)
target_net = copy.deepcopy(net).eval().requires_grad_(False)
opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)

labelled = args.arm.startswith("exit") or args.arm == "policy"
if labelled:
    data = torch.load(args.labels)
    lab_states, lab_targets = data["states"], data["targets"]
    print(f"{len(lab_states)} étiquettes de {data['starts']} départs ({data['seconds'] / 60:.0f} min de recherche)",
          flush=True)
if args.arm == "policy":
    lab_moves = data.get("moves")
    if lab_moves is None:
        # chemins de labels.py, rangés à la suite: l'état suivant est l'étiquette suivante, ou le cube résolu
        nxt = torch.roll(lab_states, -1, 0)
        last = data["i"] == data["L"] - 1
        nxt[last] = cube.solved
        match = (cube.children(lab_states) == nxt[:, None]).all(-1)
        assert (match.sum(1) == 1).all(), "chemin rompu"
        lab_moves = match.float().argmax(1)
n_ss = args.walks * args.length
n_lab = round(n_ss * args.label_fraction / (1 - args.label_fraction)) if labelled else 0


@torch.no_grad()
def target_values(states):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return target_net(cube.encode_stickers(states))["value"].float()


def symmetric(states):
    """une symétrie au hasard et, une fois sur deux, l'inverse, pour chaque état."""
    inv = torch.rand(len(states), generator=generator) < 0.5
    states = torch.where(inv[:, None], cube.inverse(states), states)
    j = torch.randint(48, (len(states),), generator=generator)
    return cube.conjugates(states)[torch.arange(len(states)), j]


def conjugate_with_moves(states, moves):
    """une symétrie au hasard pour chaque état, et le coup qui lui correspond."""
    j = torch.randint(48, (len(states),), generator=generator)
    return cube.conjugates(states)[torch.arange(len(states)), j], cube.action_perm[j, moves]


log, t0 = [], time.time()
for step in range(args.steps):
    for g in opt.param_groups:
        g["lr"] = args.lr * 0.5 * (1 + np.cos(np.pi * step / args.steps))
    states, labels, depths = cube.self_supervised_batch(args.walks, args.length, generator)
    target = depths.float()
    keep = torch.ones(len(states), dtype=torch.bool)
    deep = depths >= args.k_deep
    if args.arm == "exit_drop":
        keep = ~deep
    if args.arm == "bellman" and step % args.target_every == 0:
        target_net.load_state_dict(net.state_dict())
    if args.arm == "bellman":
        kids = cube.children(states[deep]).reshape(-1, 54)
        v = target_values(kids).clamp(min=0).masked_fill(cube.is_solved(kids), 0.0)
        target[deep] = torch.minimum(target[deep], 1.0 + v.reshape(-1, 12).min(dim=1).values)
    n_ss = len(states)
    if n_lab and args.arm == "policy":
        idx = torch.randint(len(lab_states), (n_lab,), generator=generator)
        lab, lab_labels = conjugate_with_moves(lab_states[idx], lab_moves[idx])
        states = torch.cat([states, lab])
    elif n_lab:
        idx = torch.randint(len(lab_states), (n_lab,), generator=generator)
        states = torch.cat([states, symmetric(lab_states[idx])])
        target = torch.cat([target, lab_targets[idx]])
        keep = torch.cat([keep, torch.ones(n_lab, dtype=torch.bool)])
    if args.trunk:
        net.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = net(cube.encode_stickers(states))
        ce = torch.nn.functional.cross_entropy(out["policy"][:n_ss].float(), labels)
        if args.arm == "policy":
            se = ((out["value"][:n_ss].float() - target) ** 2).mean()
            ce_lab = torch.nn.functional.cross_entropy(out["policy"][n_ss:].float(), lab_labels)
            loss = ce + args.value_weight * se + args.label_weight * ce_lab
        else:
            se = ((out["value"].float() - target) ** 2)[keep].mean()
            loss = ce + args.value_weight * se
    else:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            z = net.body(cube.encode_stickers(states[keep])).float()
        ce = torch.zeros(())
        se = ((head(z).squeeze(-1) - target[keep]) ** 2).mean()
        loss = se
    opt.zero_grad()
    loss.backward()
    opt.step()
    if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < args.steps:
        nt.save(net, args.out.replace(".pt", f"_s{step + 1}.pt"), arm=args.arm, steps=step + 1,
                seconds=time.time() - t0, args=vars(args))
    if (step + 1) % 200 == 0:
        log.append({"step": step + 1, "policy_loss": round(float(ce), 4), "value_loss": round(float(se), 4),
                    "s": round(time.time() - t0, 1)})
        if args.arm == "policy":
            log[-1]["label_loss"] = round(float(ce_lab), 4)
        print(json.dumps(log[-1]), flush=True)
net.eval()
nt.save(net, args.out, arm=args.arm, steps=args.steps, seconds=time.time() - t0, args=vars(args),
        examples=meta.get("examples"), elapsed_s=meta.get("elapsed_s"))
print(f"terminé : {args.steps} pas en {time.time() - t0:.0f} s", flush=True)
