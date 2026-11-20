"""
Entraînement à temps égal des bras du test du critère (docs/lrx_prediction.md), sur un puzzle de permutation
(perm_torch) ou sur le cube, depuis zéro, réseau à deux têtes (policy, value):
- ss: auto-supervisé, politique = inverse du dernier coup, valeur = longueur du mélange (notre recette);
- bellman: valeur = min(longueur, 1 + min_a V_cible(s·a)), 0 au but; politique comme ss;
- improve: valeur comme bellman; politique = l'enfant de plus petit 1 + V_cible, où l'enfant du mélange est
            borné par la longueur du mélange (à égalité, l'étiquette du mélange);
- oracle: valeur = distance exacte, politique = uniforme sur les coups optimaux (table exacte, n <= 12).
Le réseau cible (bellman, improve) est recopié tous les --target-every pas. La valeur est apprise en unités de
--value-scale coups; le réseau sauvegardé la rend en coups (dernière couche remise à l'échelle).

Usage:
    python src/train_perm.py --puzzle lrx --n 11 --arm bellman --minutes 15 --out-dir runs/lrx/b_s0
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import circuit_torch as qt
import cube_torch as ct
import network_torch as nt
import perm_torch as pt
import train_torch as tt


def make_puzzle(name, n, device, topology="line"):
    if name == "cube":
        return ct.TorchCube(device)
    if name in ("cnot", "swap"):
        return qt.make_puzzle(name, n, topology, device)        # synthèse de circuits (circuit_torch)
    return pt.PermPuzzle(name, n, device)


def network_for(puzzle, args):
    if isinstance(puzzle, ct.TorchCube):
        return nt.ResMLP(encoding=args.encoding, hidden=args.hidden, width=args.width, blocks=args.blocks,
                         heads=("policy", "value"))
    return nt.ResMLP(encoding="perm", hidden=args.hidden, width=args.width, blocks=args.blocks,
                     heads=("policy", "value"), n_inputs=puzzle.n * puzzle.n, n_moves=puzzle.n_moves)


@torch.no_grad()
def child_values(target, puzzle, states, scale, amp, chunk=1 << 16):
    """Valeur (en coups) de chaque enfant selon le réseau cible, 0 pour le but: (N, G)."""
    kids = puzzle.children(states).reshape(-1, states.shape[-1])
    out = []
    for start in range(0, len(kids), chunk):
        with torch.autocast(puzzle.device.type, dtype=torch.bfloat16, enabled=amp):
            out.append(target(puzzle.encode(kids[start:start + chunk]))["value"].float())
    v = (torch.cat(out) * scale).clamp(min=0).masked_fill(puzzle.is_solved(kids), 0.0)
    return v.reshape(len(states), -1)


@torch.no_grad()
def targets(arm, puzzle, states, labels, depths, target, table, scale, amp):
    """(cible de politique: indices ou probabilités, cible de valeur en coups)."""
    if arm == "ss":
        return labels, depths.float()
    if arm == "oracle":
        d = table[puzzle.rank(states)].long()
        kids = puzzle.children(states).reshape(-1, states.shape[-1])
        kid_d = table[puzzle.rank(kids)].long().reshape(len(states), -1)
        optimal = (kid_d == (d - 1).unsqueeze(1)).float()          # aucun au but: perte de politique nulle
        probs = optimal / optimal.sum(1, keepdim=True).clamp(min=1)
        return probs, d.float()
    q = 1.0 + child_values(target, puzzle, states, scale, amp)
    rows = torch.arange(len(states), device=states.device)
    q[rows, labels] = torch.minimum(q[rows, labels], depths.float())   # le mélange borne son propre enfant
    value = q.min(1).values.masked_fill(puzzle.is_solved(states), 0.0)
    if arm == "bellman":
        return labels, value
    q[rows, labels] -= 1e-3                                            # à égalité, l'étiquette du mélange
    return q.argmin(1), value


@torch.no_grad()
def probe_accuracy(net, puzzle, probe, length, bin_of, bins, amp, chunk=1 << 18):
    """bandit C: part des étiquettes du jeu fixe classées premières, par intervalle de longueurs (CPU)."""
    states, labels, depths = probe
    net.eval()
    hits = []
    for s in range(0, len(states), chunk):
        with torch.autocast(puzzle.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = net(puzzle.encode(states[s:s + chunk]))["policy"].float()
        hits.append((logits.argmax(1) == labels[s:s + chunk]).float())
    b = bin_of.to(states.device)[depths - 1]
    acc = torch.bincount(b, weights=torch.cat(hits), minlength=bins) / torch.bincount(b, minlength=bins)
    return acc.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--puzzle", choices=["lrx", "pancake", "cube", "cnot", "swap"], default="lrx")
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--topology", default="line",
                        help="cnot, swap : graphe de couplage (line, ring, full, grid:RxC ou fichier JSON)")
    parser.add_argument("--arm", choices=["ss", "bellman", "improve", "oracle"], required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--walks", type=int, default=256, help="mélanges par pas (lot = walks x length)")
    parser.add_argument("--gen-factor", type=int, default=1, help="mélanges générés par appel, en multiples de --walks")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=1000)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=10.0)
    parser.add_argument("--target-every", type=int, default=500)
    parser.add_argument("--warmup", type=float, default=0.25,
                        help="bellman, improve : part du budget entraînée comme ss avant les cibles de Bellman "
                             "(échauffement de CayleyPy RL : partir de zéro, les valeurs se propagent trop lentement)")
    parser.add_argument("--table", help="table des distances exactes (défaut : data/<puzzle><n>_dist.u8)")
    parser.add_argument("--log-seconds", type=float, default=60.0)
    parser.add_argument("--checkpoint-minutes", type=float, default=0.0,
                        help="checkpoints réguliers (pour comparer aussi à nombre d'exemples égal) ; 0 : aucun")
    parser.add_argument("--bandit", action="store_true",
                        help="bandit C de l'article (bras ss) : longueurs choisies par le progrès de la précision de "
                             "l'étiquette sur un jeu fixe ; les longueurs sont groupées en --bandit-bins intervalles")
    parser.add_argument("--bandit-bins", type=int, default=26, help="26 : la structure du bandit du cube")
    parser.add_argument("--bandit-probe", type=int, default=4000, help="mélanges du jeu fixe")
    parser.add_argument("--bandit-decisions", type=int, default=90, help="décisions sur le budget (90 sur le cube)")
    parser.add_argument("--bandit-floor", type=float, default=0.25)
    parser.add_argument("--encoding", choices=["pieces", "stickers"], default="stickers",
                        help="cube seulement (stickers : le réseau principal de l'article)")
    parser.add_argument("--threads", type=int, default=0, help="threads CPU de torch (0 : défaut)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    puzzle = make_puzzle(args.puzzle, args.n, device, args.topology)
    table = None
    if args.arm == "oracle" and args.puzzle in ("cnot", "swap"):
        table = qt.load_or_build_table(puzzle, args.table)
    elif args.arm == "oracle":
        table = pt.load_or_build_table(puzzle, args.table or f"data/{args.puzzle}{args.n}_dist.u8")
    net = network_for(puzzle, args).to(device)
    if args.puzzle == "cube" and args.encoding == "stickers":
        puzzle.encode = puzzle.encode_stickers      # dans ce script, l'encodage du réseau (la recherche lit la config)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    target = copy.deepcopy(net).eval().requires_grad_(False) if args.arm in ("bellman", "improve") else None
    amp = device.type == "cuda"
    generator = torch.Generator(device=device).manual_seed(args.seed)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    open(log_path, "w").close()
    print(f"{args.arm} : {net.n_parameters():,} paramètres, lot {args.walks * args.length:,}", flush=True)

    budget, start = args.minutes * 60, time.time()
    steps = examples = 0
    bellman_started, bellman_step, target_copies = False, 0, 0
    sums, n_sums, next_log = {}, 0, args.log_seconds
    next_checkpoint = args.checkpoint_minutes * 60
    pending = []
    bandit = None
    if args.bandit:
        if args.arm != "ss":
            raise ValueError("--bandit : bras ss seulement")
        bins = args.bandit_bins
        bin_of = torch.arange(args.length) * bins // args.length          # longueur k (indice k - 1) -> intervalle
        bandit = tt.LengthBandit(bins, device, floor=args.bandit_floor, alp=True, horizon=8, top=8, inertia=0.5)
        probe = puzzle.self_supervised_batch(args.bandit_probe, args.length,
                                             torch.Generator(device=device).manual_seed(args.seed + 1))
        interval = budget / args.bandit_decisions
        next_decision = interval
    while True:
        elapsed = time.time() - start
        if elapsed >= budget:
            break
        lr = args.lr if args.lr_schedule == "constant" else args.lr * 0.5 * (1 + math.cos(math.pi * elapsed / budget))
        for group in optimizer.param_groups:
            group["lr"] = lr
        if bandit is not None and elapsed >= next_decision:
            bandit.update(probe_accuracy(net, puzzle, probe, args.length, bin_of, args.bandit_bins, amp))
            next_decision += interval
            pending = []                                       # les lots suivants suivent le nouveau choix
        while not pending and bandit is not None:
            # marge de 2 %: avec peu de lots par appel, le dernier peut manquer (lot vide écarté, on retire)
            pending = [b for b in tt.focused_batches(puzzle, args.walks, args.length, args.gen_factor,
                                                     bandit.keep[bin_of], generator) if len(b[0])]
        if not pending:
            # génération en un appel pour gen_factor pas (sur GPU, elle est limitée par le lancement des noyaux)
            batch = puzzle.self_supervised_batch(args.walks * args.gen_factor, args.length, generator)
            pending = tt.split_batches(*batch, args.length, args.gen_factor)[::-1]
        states, labels, depths = pending.pop()
        arm = args.arm
        if target is not None and elapsed < args.warmup * budget:
            arm = "ss"
        elif target is not None and not bellman_started:
            target.load_state_dict(net.state_dict())           # fin de l'échauffement: cible = réseau courant
            bellman_started, bellman_step = True, steps
            print(f"cibles de Bellman à partir du pas {steps} ({elapsed:.0f} s)", flush=True)
        pol, val = targets(arm, puzzle, states, labels, depths, target, table, args.value_scale, amp)
        net.train()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = net(puzzle.encode(states))
        logits = out["policy"].float()
        policy_loss = F.cross_entropy(logits, pol)
        value_loss = F.mse_loss(out["value"].float(), val / args.value_scale)
        loss = policy_loss + args.value_weight * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps += 1
        examples += len(states)
        if bellman_started and (steps - bellman_step) % args.target_every == 0:
            target.load_state_dict(net.state_dict())
            target_copies += 1
        hard = pol if pol.dim() == 1 else pol.argmax(1)
        # métriques cumulées sur le device: pas de synchronisation par pas (environ 1 ms chacune sur un GPU distant)
        for key, value in (("policy_loss", policy_loss), ("value_loss", value_loss),
                           ("label_agree", (hard == labels).float().mean()),
                           ("policy_acc", (logits.argmax(1) == hard).float().mean())):
            sums[key] = sums.get(key, 0.0) + value.detach()
        n_sums += 1
        if args.checkpoint_minutes and time.time() - start >= next_checkpoint:
            save_scaled(net, os.path.join(args.out_dir, f"ckpt_{next_checkpoint / 60:06.1f}min.pt"), args,
                        elapsed_s=time.time() - start, steps=steps, examples=examples, target_copies=target_copies)
            next_checkpoint += args.checkpoint_minutes * 60
        if time.time() - start >= next_log:
            entry = {"time_s": round(time.time() - start, 1), "steps": steps, "examples": examples, "lr": lr,
                     "examples_per_s": round(examples / (time.time() - start)), "phase": arm,
                     "target_copies": target_copies}
            entry.update({k: round(float(v) / n_sums, 4) for k, v in sums.items()})
            if bandit is not None:
                entry["keep"] = [round(float(x), 2) for x in bandit.keep]
            sums, n_sums = {}, 0
            next_log += args.log_seconds
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)

    save_scaled(net, os.path.join(args.out_dir, "final.pt"), args, elapsed_s=time.time() - start, steps=steps,
                examples=examples, target_copies=target_copies)
    print(f"fin : {steps} pas, {examples:,} exemples, {time.time() - start:.0f} s", flush=True)


def save_scaled(net, path, args, **metadata):
    """sauvegarde une copie dont la tête value rend des coups (dernière couche multipliée par l'échelle)."""
    copy_net = copy.deepcopy(net)
    with torch.no_grad():
        layer = copy_net.heads["value"]
        layer = layer[-1] if isinstance(layer, torch.nn.Sequential) else layer
        layer.weight.mul_(args.value_scale)
        layer.bias.mul_(args.value_scale)
    nt.save(copy_net, path, args=vars(args), **metadata)


if __name__ == "__main__":
    main()
