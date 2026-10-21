"""
Entraînement PyTorch (CPU ou GPU) à budget de temps fixe, avec des checkpoints réguliers pour tracer la qualité
en fonction du temps d'entraînement. Les données sont toujours neuves et générées sur le device : chaque pas
tire --walks mélanges de --length coups et utilise tous leurs préfixes.

Objectifs (têtes du réseau, combinables avec --heads) :
- policy : auto-supervisé, inverse du dernier coup du mélange (notre méthode, aussi celle de Takano) ;
- value  : régression de la longueur du mélange (principe de CayleyPy) ;
- q      : Q-learning, cible de Bellman 1 + min_a' Q_cible(enfant, a') (0 pour le cube résolu), bornée par la
           longueur du mélange ; réseau cible recopié tous les --target-every pas.
Options :
- --canonical : états ramenés à leur forme canonique parmi leurs 48 symétriques ;
- --per : replay priorisé (Schaul et al., 2016) : une part --per-fraction de chaque lot est tirée d'un buffer,
  avec une probabilité qui croît avec la perte de l'exemple à son dernier passage, corrigée par des poids
  d'importance.

Usage :
    python src/train_torch.py --minutes 90 --heads policy --out-dir runs/auto_supervise
    python src/train_torch.py --minutes 90 --heads value --encoding stickers --out-dir runs/cayleypy
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import cube_torch as ct
import network_torch as nt


class PrioritizedBuffer:
    """Buffer circulaire sur le device : états, étiquettes, profondeurs et priorités."""

    def __init__(self, capacity, device, alpha, beta):
        self.capacity, self.alpha, self.beta = capacity, alpha, beta
        self.states = torch.zeros(capacity, 54, dtype=torch.uint8, device=device)
        self.labels = torch.zeros(capacity, dtype=torch.long, device=device)
        self.depths = torch.zeros(capacity, dtype=torch.long, device=device)
        self.priority = torch.zeros(capacity, device=device)
        self.size = self.pos = 0

    def add(self, states, labels, depths, priority):
        idx = (self.pos + torch.arange(len(states), device=states.device)) % self.capacity
        self.states[idx], self.labels[idx], self.depths[idx], self.priority[idx] = states, labels, depths, priority
        self.pos = (self.pos + len(states)) % self.capacity
        self.size = min(self.size + len(states), self.capacity)

    def sample(self, n, generator=None):
        p = self.priority[:self.size] ** self.alpha
        p = p / p.sum()
        idx = torch.multinomial(p, n, replacement=True, generator=generator)
        weights = (self.size * p[idx]) ** (-self.beta)
        return idx, self.states[idx], self.labels[idx], self.depths[idx], weights / weights.max()

    def update(self, idx, priority):
        self.priority[idx] = priority


@torch.no_grad()
def q_targets(target, cube, encode, states, labels, depths, canonical, amp, chunk=1 << 16):
    """
    Cibles de Bellman (N, 12) : 1 + distance estimée de chaque enfant, bornées par le mélange (un enfant est
    à au plus depth + 1 coups, et l'inverse du dernier coup ramène à au plus depth - 1 coups).
    """
    kids = cube.children(states).reshape(-1, 54)
    best = []
    for start in range(0, len(kids), chunk):
        k = kids[start:start + chunk]
        if canonical:
            k, _ = cube.canonicalize(k)     # le minimum sur les actions ne dépend pas du repère
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            best.append(target(encode(k))["q"].float().min(dim=1).values)
    distance = torch.cat(best).clamp(min=0).masked_fill(cube.is_solved(kids), 0.0).reshape(-1, 12)
    bound = (depths + 2).float().unsqueeze(1).repeat(1, 12)
    bound.scatter_(1, labels.unsqueeze(1), depths.float().unsqueeze(1))
    return torch.minimum(1.0 + distance, bound)


class Trainer:
    """Réseau, optimiseur et réseau cible ; un appel à step fait un pas de gradient sur un lot."""

    def __init__(self, net, cube, args):
        self.net, self.cube, self.args = net, cube, args
        self.heads, self.canonical = net.config["heads"], net.config["canonical"]
        self.encode = cube.encode if net.config["encoding"] == "pieces" else cube.encode_stickers
        self.model = torch.compile(net) if args.compile else net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        self.target = copy.deepcopy(net).eval().requires_grad_(False) if "q" in self.heads else None
        self.amp = cube.device.type == "cuda"
        self.steps = 0

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self, states, labels, depths, weights=None, n_metrics=None):
        """
        states (N, 54), labels (coup à jouer), depths (borne supérieure de la distance), weights (poids
        d'importance, optionnels). Retourne (perte par exemple, métriques sur les n_metrics premiers exemples).
        """
        args, cube = self.args, self.cube
        n_metrics = n_metrics or len(states)
        if self.canonical:
            states, k = cube.canonicalize(states)
            labels = cube.action_perm[k, labels]
        self.model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(self.encode(states))
        per_example = torch.zeros(len(states), device=states.device)
        metrics = {}
        if "policy" in self.heads:
            logits = out["policy"].float()
            ce = F.cross_entropy(logits, labels, reduction="none")
            per_example = per_example + args.policy_weight * ce
            metrics["policy_loss"] = ce[:n_metrics].mean().detach()
            metrics["policy_acc"] = (logits[:n_metrics].argmax(dim=1) == labels[:n_metrics]).float().mean()
        if "value" in self.heads:
            se = (out["value"].float() - depths.float()) ** 2
            per_example = per_example + args.value_weight * se
            metrics["value_loss"] = se[:n_metrics].mean().detach()
        if "q" in self.heads:
            y = q_targets(self.target, cube, self.encode, states, labels, depths, self.canonical, self.amp)
            se_q = ((out["q"].float() - y) ** 2).mean(dim=1)
            per_example = per_example + args.q_weight * se_q
            metrics["q_loss"] = se_q[:n_metrics].mean().detach()
        loss = (per_example * weights).mean() if weights is not None else per_example.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.target is not None and self.steps % args.target_every == 0:
            self.target.load_state_dict(self.net.state_dict())
        self.steps += 1
        return per_example.detach(), metrics


class RunLog:
    """Journal JSONL périodique et checkpoints réguliers d'un entraînement à budget de temps."""

    def __init__(self, net, args):
        self.net, self.args = net, args
        self.path = os.path.join(args.out_dir, "train_log.jsonl")
        self.budget = args.minutes * 60
        self.examples = 0
        self.sums, self.n_sums = {}, 0
        self.next_log, self.next_checkpoint = args.log_seconds, args.checkpoint_minutes * 60
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def lr(self):
        if self.args.lr_schedule == "constant":
            return self.args.lr
        return self.args.lr * 0.5 * (1 + math.cos(math.pi * min(self.elapsed() / self.budget, 1.0)))

    def save(self, name, steps):
        nt.save(self.net, os.path.join(self.args.out_dir, name), elapsed_s=self.elapsed(), steps=steps,
                examples=self.examples, args=vars(self.args))

    def record(self, steps, n_examples, metrics, extra=None):
        """À appeler après chaque pas : cumule les métriques, écrit le journal et les checkpoints à échéance."""
        self.examples += n_examples
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + value
        self.n_sums += 1
        elapsed = self.elapsed()
        if elapsed >= self.next_log or elapsed >= self.budget:
            entry = {"time_s": round(elapsed, 1), "steps": steps, "examples": self.examples, "lr": self.lr(),
                     "examples_per_s": round(self.examples / elapsed)}
            entry.update({key: round(float(value) / self.n_sums, 4) for key, value in self.sums.items()})
            entry.update(extra or {})
            self.sums, self.n_sums = {}, 0
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            self.next_log += self.args.log_seconds
        if elapsed >= self.next_checkpoint:
            self.save(f"ckpt_{self.next_checkpoint / 60:07.1f}min.pt", steps)
            self.next_checkpoint += self.args.checkpoint_minutes * 60


def add_training_arguments(parser):
    """Options communes à train_torch et expert_iteration_torch."""
    parser.add_argument("--minutes", type=float, required=True, help="budget de temps total (génération comprise)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--heads", default="policy", help="têtes, séparées par des virgules : policy,value,q")
    parser.add_argument("--encoding", choices=list(nt.INPUT_SIZES), default="pieces")
    parser.add_argument("--hidden", type=int, default=5000)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--init", help="checkpoint de départ (son architecture remplace --hidden/--width/...)")
    parser.add_argument("--walks", type=int, default=1000, help="mélanges par pas (lot = walks x length)")
    parser.add_argument("--length", type=int, default=26)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                        help="cosine : décroissance selon la fraction du budget de temps écoulée")
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--target-every", type=int, default=1000, help="pas entre deux copies du réseau cible (q)")
    parser.add_argument("--checkpoint-minutes", type=float, default=10.0)
    parser.add_argument("--log-seconds", type=float, default=60.0)
    parser.add_argument("--compile", action="store_true", help="torch.compile (GPU sous Linux)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)


def setup(args):
    """Retourne (cube, générateur aléatoire, réseau) sur le device demandé."""
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    cube = ct.TorchCube(device)
    generator = torch.Generator(device).manual_seed(args.seed)
    if args.init:
        net, _ = nt.load(args.init, device)
    else:
        net = nt.ResMLP(args.encoding, args.hidden, args.width, args.blocks, args.heads.split(","),
                        args.canonical).to(device)
    return cube, generator, net


def main():
    parser = argparse.ArgumentParser(description="Entraînement PyTorch à budget de temps fixe")
    add_training_arguments(parser)
    parser.add_argument("--per", action="store_true", help="replay priorisé")
    parser.add_argument("--per-fraction", type=float, default=0.5)
    parser.add_argument("--per-capacity", type=int, default=2_000_000)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    args = parser.parse_args()

    cube, generator, net = setup(args)
    trainer = Trainer(net, cube, args)
    n_fresh = args.walks * args.length
    n_replay = round(n_fresh * args.per_fraction / (1 - args.per_fraction)) if args.per else 0
    buffer = PrioritizedBuffer(args.per_capacity, cube.device, args.per_alpha, args.per_beta) if args.per else None
    print(f"{net.n_parameters() / 1e6:.1f} M paramètres, têtes {net.config['heads']}, device {cube.device}, "
          f"lot {n_fresh} + {n_replay} rejoués", flush=True)

    log = RunLog(net, args)
    while log.elapsed() < log.budget:
        trainer.set_lr(log.lr())
        states, labels, depths = cube.self_supervised_batch(args.walks, args.length, generator)
        weights = None
        if buffer is not None and buffer.size >= n_replay:
            idx, r_states, r_labels, r_depths, r_weights = buffer.sample(n_replay, generator)
            states, labels, depths = torch.cat([states, r_states]), torch.cat([labels, r_labels]), torch.cat([depths, r_depths])
            weights = torch.cat([torch.ones(n_fresh, device=cube.device), r_weights])
        # métriques sur les exemples neufs seulement, comparables avec ou sans replay
        per_example, metrics = trainer.step(states, labels, depths, weights, n_metrics=n_fresh)
        if buffer is not None:
            priority = per_example + 1e-3
            if weights is not None:
                buffer.update(idx, priority[n_fresh:])
            buffer.add(states[:n_fresh], labels[:n_fresh], depths[:n_fresh], priority[:n_fresh])
        log.record(trainer.steps, len(states), metrics)
    log.save("final.pt", trainer.steps)
    print(f"Terminé : {trainer.steps} pas, {log.examples} exemples, {log.elapsed():.0f} s", flush=True)


if __name__ == "__main__":
    main()
