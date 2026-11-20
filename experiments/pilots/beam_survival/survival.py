"""
pilote "survie dans le faisceau" (2026-11-06, CPU). Le réseau apprend la décision même du faisceau: à chaque
niveau, garder les W enfants les moins chers. Un professeur (faisceau large) trouve une solution plus courte que
l'élève (faisceau de largeur W); on suit ce chemin dans le faisceau de l'élève, gardé de force comme dans l'oracle
de la section 5.3, et à chaque niveau où il y a une vraie coupure (plus de W candidats distincts), on apparie
l'enfant du chemin avec K concurrents classés juste avant la coupure (rangs [W/2, W)). Une perte par paires apprend
au coût c(p, a) = V(p) - log pi(a | p) à placer l'enfant du chemin avant eux.
Étapes:
  teach: solutions du professeur sur des cubes neufs (marches de 200/199 coups, lab.fresh_cubes);
  pairs: paires le long du faisceau de l'élève, chemin du professeur gardé de force;
  train: réentraînement depuis le réseau de base (réseau entier avec l'auto-supervision, ou tête de valeur seule),
            aux mêmes réglages que le témoin runs/floor/ss.pt (3000 pas, lr 3e-4 cosinus, poids de valeur 0.1);
  survive: sans forcer, niveau où le chemin du professeur sort du faisceau, et précision des paires tenues à part.
Usage:
  python experiments/pilots/beam_survival/survival.py teach --n 1000 --seed 780
  python experiments/pilots/beam_survival/survival.py pairs --seed 780
  python experiments/pilots/beam_survival/survival.py train --pairs runs/survival/pairs_seed780.pt --out runs/survival/rank.pt
  python experiments/pilots/beam_survival/survival.py survive --seed 781 runs/s4/m90_v/final.pt runs/survival/rank.pt
"""
import argparse
import json
import math
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

VIEWS = ["id", "inv", "c15", "c20"]
BASE = "runs/s4/m90_v/final.pt"
DIR = "runs/survival"
torch.set_num_threads(16)
cube = ct.TorchCube("cpu")


def teacher_file(seed, width):
    return f"{DIR}/teacher_seed{seed}_w{width}.jsonl"


def teach(args):
    """solution du professeur (faisceau de largeur --width, 4 racines, réseau de base) pour chaque cube neuf."""
    net, _ = nt.load(args.model)
    h = lab.LabHeuristic(net, cube, args.width, chunk=1 << 14)
    out = teacher_file(args.seed, args.width)
    done = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    cubes = lab.fresh_cubes(cube, args.n, args.seed)
    with open(out, "a") as f:
        for i in range(done, args.n):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                r = lab.solve(h, cubes[i], args.width, VIEWS)
            f.write(json.dumps({"i": i, "state": cubes[i].tolist(), "solution": r["solution"],
                                "time_s": round(r["time_s"], 2)}) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"professeur : {i + 1}/{args.n} cubes, {time.strftime('%H:%M')}", flush=True)


def path_keys(start, moves):
    """Empreintes des états du chemin depuis start, profondeurs 1 à len(moves)."""
    states, st = [], start
    for a in moves:
        st = cube.children(st[None])[0, a]
        states.append(st)
    assert bool(cube.is_solved(st)), "le chemin du professeur ne résout pas le cube"
    return cube.state_hash(torch.stack(states))


@torch.no_grad()
def replay(h, start, keys_path, width, k, force, generator):
    """
    Faisceau de l'élève (4 racines, coût V - log pi, largeur width) sur start, en suivant le chemin du professeur
    depuis la racine s. À chaque profondeur où l'enfant du chemin est candidat: son rang parmi les candidats
    distincts, gardé ou non, et, s'il y a une coupure, k concurrents de rangs [width/2, width) (parent et coup).
    force: l'enfant du chemin est gardé de force (données d'entraînement); sinon le suivi s'arrête quand il sort.
    """
    views = bt.equivalent_views(cube, start, VIEWS)
    hashes = cube.state_hash(torch.stack([s for s, _ in views])).tolist()
    roots = torch.stack([views[i][0] for i, key in enumerate(hashes) if key not in hashes[:i]])
    forward = bt._Beam(cube, roots, roots)
    records = []
    for depth in range(1, len(keys_path) + 1):
        parents = forward.states
        kids, keys, cost, cum, _ = forward.expand(h, lambda x: x)
        if bool((keys == cube.solved_hash).any()):
            break                                   # le faisceau s'arrête ici: plus de décision à apprendre
        best, n_best = forward.candidates(keys, cost, width)
        target = keys_path[depth - 1]
        js = torch.nonzero((keys == target) & torch.isfinite(cost)).flatten()
        if len(js) == 0:
            # l'état du chemin n'est pas candidat (déjà vu par le faisceau): le chemin est perdu
            records.append({"depth": depth, "kept": False, "absent": True})
            break
        j = js[cost[js].argmin()]
        # même tri que _Beam.candidates: un représentant par état distinct, les autres à l'infini
        order = torch.argsort(cost)
        by_key = order[torch.argsort(keys[order], stable=True)]
        first = torch.ones(len(by_key), dtype=torch.bool)
        first[1:] = keys[by_key][1:] != keys[by_key][:-1]
        unique_cost = cost.scatter(0, by_key, cost[by_key].masked_fill(~first, float("inf")))
        sorted_cost, sorted_idx = torch.sort(unique_cost)
        n_valid = int(torch.isfinite(sorted_cost).sum())
        rank = int((unique_cost < cost[j]).sum())
        kept = bool((keys[best[:int(n_best)]] == target).any())
        rec = {"depth": depth, "rank": rank, "kept": kept, "n_valid": n_valid, "path_cost": float(cost[j])}
        if n_valid > width:
            ranks = torch.randint(width // 2, width, (k,), generator=generator)
            idx = sorted_idx[ranks]
            idx = idx[keys[idx] != target]
            if len(idx):
                rec.update(p_parent=parents[j // 12], p_action=int(j % 12), c_parent=parents[idx // 12],
                           c_action=idx % 12, c_cost=cost[idx])
        records.append(rec)
        if not kept:
            if not force:
                break
            best, n_best, _ = lab.force_keep(keys, cost, best, n_best, target)
        n_best = int(n_best)
        if n_best == 0:
            break
        forward.select(kids, keys, cum, best[:n_best])
    return records


def load_teacher(seed, width):
    rows = [json.loads(line) for line in open(teacher_file(seed, width))]
    return [(cube.from_numpy(r["state"]), r["solution"], r["i"]) for r in rows if r["solution"] is not None]


def pairs(args):
    """paires d'entraînement le long du faisceau de l'élève (réseau --model), chemin gardé de force."""
    net, _ = nt.load(args.model)
    h = bt.NetworkHeuristic(net, cube, "ac", 1 << 14, 1.0)
    generator = torch.Generator().manual_seed(args.seed)
    P, PA, C, CA, meta = [], [], [], [], []
    t0 = time.time()
    teacher = load_teacher(args.seed, args.teacher_width)
    for n, (start, moves, i) in enumerate(teacher):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            recs = replay(h, start, path_keys(start, moves), args.width, args.k, True, generator)
        for r in recs:
            if "p_parent" in r:
                m = len(r["c_action"])
                P.append(r["p_parent"])
                PA.append(r["p_action"])
                # toujours k concurrents par paire (rare: l'état du chemin tiré parmi eux, on complète)
                fill = torch.arange(args.k) % m
                C.append(r["c_parent"][fill])
                CA.append(r["c_action"][fill])
                meta.append([i, r["depth"], r["rank"], int(r["kept"]), len(moves)])
        if (n + 1) % 100 == 0:
            print(f"paires : {n + 1}/{len(teacher)} cubes, {len(P)} niveaux, {time.time() - t0:.0f} s", flush=True)
    meta = torch.tensor(meta)
    out = args.out or f"{DIR}/pairs_seed{args.seed}{args.tag}.pt"
    torch.save({"p_parent": torch.stack(P), "p_action": torch.tensor(PA), "c_parent": torch.stack(C),
                "c_action": torch.stack(CA), "meta": meta, "model": args.model, "width": args.width}, out)
    kept = meta[:, 3].float()
    print(f"{len(P)} niveaux avec coupure, {len(teacher)} cubes ; chemin gardé sans aide à {kept.mean():.1%} de ces "
          f"niveaux ; rang médian / largeur {meta[:, 2].float().median() / args.width:.2f} -> {out}", flush=True)


def conjugate(states, actions, generator):
    """une symétrie au hasard par état, et le coup correspondant."""
    j = torch.randint(48, (len(states),), generator=generator)
    return cube.conjugates(states)[torch.arange(len(states)), j], cube.action_perm[j, actions]


def scores(net, states, actions, value_only):
    """Coût V(p) - log pi(a | p) de l'enfant (p, a); avec value_only, seule la tête de valeur reçoit le gradient."""
    x = cube.encode_stickers(states)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        if value_only:
            with torch.no_grad():
                z = net.body(x)
                logits = net.heads["policy"](z)
            v = net.heads["value"](z.float()).squeeze(-1)
        else:
            out = net(x)
            v, logits = out["value"], out["policy"]
    logp = torch.log_softmax(logits.float(), dim=1).gather(1, actions[:, None]).squeeze(1)
    return v.float() - logp


def train(args):
    net, meta = nt.load(args.model)
    # plusieurs fichiers: les paires des tours précédents s'ajoutent (agrégation à la DAgger)
    data = [torch.load(p) for p in args.pairs]
    P, PA, C, CA = (torch.cat([d[k] for d in data]) for k in ("p_parent", "p_action", "c_parent", "c_action"))
    R, K = C.shape[:2]
    print(f"{R} niveaux x {K} concurrents ({', '.join(args.pairs)})", flush=True)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    if args.value_only:
        net.requires_grad_(False)
        net.heads["value"].requires_grad_(True)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    t0, log = time.time(), []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.5 * (1 + np.cos(np.pi * step / args.steps))
        idx = torch.randint(R, (args.batch,), generator=generator)
        p, pa = conjugate(P[idx], PA[idx], generator)
        c, ca = conjugate(C[idx].reshape(-1, 54), CA[idx].reshape(-1), generator)
        states, labels, depths = cube.self_supervised_batch(args.walks, args.length, generator)
        n = len(states)
        if args.value_only:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                z = net.body(cube.encode_stickers(states))
            v = net.heads["value"](z.float()).squeeze(-1)
            ce = torch.zeros(())
            se = ((v - depths.float()) ** 2).mean()
            cost = scores(net, torch.cat([p, c]), torch.cat([pa, ca]), True)
            loss = se
        else:
            net.train()
            all_states = torch.cat([states, p, c])
            all_actions = torch.cat([labels, pa, ca])
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = net(cube.encode_stickers(all_states))
            logp = torch.log_softmax(out["policy"].float(), dim=1)
            ce = -logp[:n].gather(1, labels[:, None]).mean()
            se = ((out["value"][:n].float() - depths.float()) ** 2).mean()
            cost = out["value"][n:].float() - logp[n:].gather(1, all_actions[n:, None]).squeeze(1)
            loss = ce + args.value_weight * se
        cp, cc = cost[:args.batch], cost[args.batch:].reshape(args.batch, K)
        rank = torch.nn.functional.softplus((cp[:, None] - cc) / args.temperature).mean()
        loss = loss + args.rank_weight * rank
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            acc = float((cp[:, None] < cc).float().mean())
            log.append({"step": step + 1, "policy_loss": round(float(ce), 4), "value_loss": round(float(se), 4),
                        "rank_loss": round(float(rank), 4), "pair_acc": round(acc, 4), "s": round(time.time() - t0)})
            print(json.dumps(log[-1]), flush=True)
    net.eval()
    nt.save(net, args.out, arm="survival", steps=args.steps, seconds=time.time() - t0, args=vars(args),
            examples=meta.get("examples"), elapsed_s=meta.get("elapsed_s"))
    print(f"terminé : {args.steps} pas en {time.time() - t0:.0f} s -> {args.out}", flush=True)


@torch.no_grad()
def survive(args):
    """Pour chaque modèle: précision des paires tenues à part, et survie du chemin du professeur sans aide."""
    held = torch.load(args.pairs) if args.pairs else None
    teacher = load_teacher(args.seed, args.teacher_width)[:args.cubes]
    for path in args.models:
        net, _ = nt.load(path)
        row = {"model": path}
        if held is not None:
            R, K = held["c_parent"].shape[:2]
            with torch.autocast("cpu", dtype=torch.bfloat16):
                cp = torch.cat([scores(net, held["p_parent"][s:s + 4096], held["p_action"][s:s + 4096], False)
                                for s in range(0, R, 4096)])
                cc = torch.cat([scores(net, held["c_parent"].reshape(-1, 54)[s:s + 4096],
                                       held["c_action"].reshape(-1)[s:s + 4096], False)
                                for s in range(0, R * K, 4096)]).reshape(R, K)
            depth = held["meta"][:, 1]
            win = (cp[:, None] < cc).float().mean(1)
            row["pair_acc"] = round(float(win.mean()), 4)
            row["pair_acc_by_depth"] = {f"{a}-{b}": round(float(win[(depth >= a) & (depth <= b)].mean()), 4)
                                        for a, b in ((1, 4), (5, 7), (8, 10), (11, 14), (15, 30))}
        h = bt.NetworkHeuristic(net, cube, "ac", 1 << 14, 1.0)
        generator = torch.Generator().manual_seed(0)
        lost, ranks = [], []
        for start, moves, i in teacher:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                recs = replay(h, start, path_keys(start, moves), args.width, 1, False, generator)
            gone = [r for r in recs if not r.get("kept", True)]
            lost.append(gone[0]["depth"] if gone else len(moves))
            if gone and "rank" in gone[0]:
                ranks.append(gone[0]["rank"] / args.width)
        lost = np.array(lost)
        row.update(cubes=len(lost), lost_depth_mean=round(float(lost.mean()), 3),
                   survived_10=round(float(np.mean(lost > 10)), 4), never_lost=round(float(np.mean(
                       [len(m) == d for (_, m, _), d in zip(teacher, lost)])), 4),
                   rank_at_loss_median=round(float(np.median(ranks)), 3) if ranks else None)
        print(json.dumps(row), flush=True)
        with open(f"{DIR}/survive.jsonl", "a") as f:
            f.write(json.dumps(dict(row, seed=args.seed, width=args.width, time=time.strftime("%H:%M"))) + "\n")


def compare(args):
    """longueurs appariées de plusieurs fichiers de search.py sur les mêmes cubes, contre le premier."""
    runs = [{r["i"]: r for r in map(json.loads, open(p))} for p in args.files]
    ids = sorted(set.intersection(*(set(r) for r in runs)))
    ref = np.array([runs[0][i]["length"] for i in ids], float)
    for path, rows in zip(args.files, runs):
        L = np.array([rows[i]["length"] for i in ids], float)
        d = L - ref
        z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std() > 0 else 0.0
        print(f"{path:70} {len(ids)} cubes, {L.mean():.3f} coups, contre le premier {d.mean():+.3f} (z {z:+.1f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("teach")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=780)
    p.add_argument("--width", type=int, default=1 << 14)
    p.add_argument("--model", default=BASE)
    p = sub.add_parser("pairs")
    p.add_argument("--seed", type=int, default=780)
    p.add_argument("--teacher-width", type=int, default=1 << 14)
    p.add_argument("--width", type=int, default=1 << 10)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--model", default=BASE)
    p.add_argument("--tag", default="")
    p.add_argument("--out")
    p = sub.add_parser("train")
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--model", default=BASE)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value-weight", type=float, default=0.1)
    p.add_argument("--rank-weight", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=256, help="niveaux par pas (chacun avec ses K concurrents)")
    p.add_argument("--walks", type=int, default=200)
    p.add_argument("--length", type=int, default=26)
    p.add_argument("--value-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("survive")
    p.add_argument("models", nargs="+")
    p.add_argument("--seed", type=int, default=781)
    p.add_argument("--teacher-width", type=int, default=1 << 14)
    p.add_argument("--width", type=int, default=1 << 10)
    p.add_argument("--cubes", type=int, default=200)
    p.add_argument("--pairs", help="paires tenues à part (cubes de --seed), pour la précision")
    p = sub.add_parser("compare")
    p.add_argument("files", nargs="+")
    args = parser.parse_args()
    os.makedirs(DIR, exist_ok=True)
    {"teach": teach, "pairs": pairs, "train": train, "survive": survive, "compare": compare}[args.cmd](args)
