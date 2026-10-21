"""
Boucle recherche -> étiquettes (itération experte) en PyTorch, à budget de temps fixe.

À chaque tour, des cubes mélangés de --depth-min à --depth-max coups sont résolus par le faisceau avec le réseau
courant. Quand une solution n'est pas plus longue que le mélange (--max-length-ratio), chacun de ses états
devient un exemple : le coup suivant de la solution comme étiquette de politique, la longueur restante comme
borne de distance. C'est là que ces exemples apportent quelque chose : loin du cube résolu, l'inverse du dernier
coup du mélange n'est souvent pas un bon coup. Ils vont dans un buffer, et chaque lot d'entraînement mélange une
part --search-fraction tirée du buffer avec des exemples auto-supervisés neufs, pour ne rien oublier.
Le temps de recherche compte dans le budget. Le témoin à temps égal est train_torch.py avec le même --init.

Usage :
    python src/expert_iteration_torch.py --init runs/auto_supervise/final.pt --minutes 60 --out-dir runs/exit
"""
import argparse
import time

import numpy as np
import torch

import beam_torch as bt
import train_torch as tt


def search_examples(heuristic, cube, n_cubes, depth_min, depth_max, width, max_depth, max_length_ratio, generator,
                    deadline=None):
    """
    Résout jusqu'à n_cubes cubes mélangés, en s'arrêtant à l'heure `deadline` (time.time()) si elle est donnée.
    Retourne (états, coups, longueurs restantes) des solutions retenues, ou None si aucune ne l'est, et les
    statistiques du tour.
    """
    lengths = torch.randint(depth_min, depth_max + 1, (n_cubes,), device=cube.device, generator=generator)
    walks, _ = cube.random_walks(n_cubes, depth_max, generator)
    starts = walks[lengths - 1, torch.arange(n_cubes, device=cube.device)]
    states, labels, remaining = [], [], []
    stats = {"attempted": 0, "solved": 0, "kept": 0, "length_ratios": []}
    for start, length in zip(starts, lengths.tolist()):
        if deadline is not None and time.time() >= deadline:
            break
        stats["attempted"] += 1
        solution = bt.solve(heuristic, start, width=width, max_depth=max_depth)["solution"]
        if not solution:
            continue
        stats["solved"] += 1
        stats["length_ratios"].append(len(solution) / length)
        if len(solution) > max_length_ratio * length:
            continue   # plus longue que le mélange : moins bonne que l'étiquette gratuite
        stats["kept"] += 1
        state = start
        for i, action in enumerate(solution):
            states.append(state)
            labels.append(action)
            remaining.append(len(solution) - i)
            state = state[cube.moves[action]]
    if not states:
        return None, stats
    device = cube.device
    return (torch.stack(states), torch.tensor(labels, device=device), torch.tensor(remaining, device=device)), stats


def main():
    parser = argparse.ArgumentParser(description="Boucle recherche -> étiquettes à budget de temps fixe")
    tt.add_training_arguments(parser)
    parser.add_argument("--cubes-per-round", type=int, default=256)
    parser.add_argument("--depth-min", type=int, default=20)
    parser.add_argument("--depth-max", type=int, default=30)
    parser.add_argument("--search-width", type=int, default=4096)
    parser.add_argument("--search-max-depth", type=int, default=40)
    parser.add_argument("--max-length-ratio", type=float, default=1.0)
    parser.add_argument("--steps-per-round", type=int, default=200)
    parser.add_argument("--search-fraction", type=float, default=0.25, help="part des lots tirée du buffer")
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    args = parser.parse_args()

    cube, generator, net = tt.setup(args)
    trainer = tt.Trainer(net, cube, args)
    heuristic = bt.NetworkHeuristic(net, cube)
    buffer = tt.PrioritizedBuffer(args.buffer_size, cube.device, alpha=0.0, beta=0.0)   # tirage uniforme
    n_fresh = args.walks * args.length
    n_search = round(n_fresh * args.search_fraction / (1 - args.search_fraction))
    print(f"{net.n_parameters() / 1e6:.1f} M paramètres, recherche {heuristic.mode} largeur {args.search_width}, "
          f"lot {n_fresh} neufs + {n_search} issus de la recherche", flush=True)

    log = tt.RunLog(net, args)
    totals = {"attempted": 0, "solved": 0, "kept": 0, "length_ratios": [], "search_s": 0.0}
    while log.elapsed() < log.budget:
        t0 = time.time()
        net.eval()
        examples, stats = search_examples(heuristic, cube, args.cubes_per_round, args.depth_min, args.depth_max,
                                          args.search_width, args.search_max_depth, args.max_length_ratio, generator,
                                          deadline=log.start + log.budget)
        if examples is not None:
            buffer.add(*examples, torch.ones(len(examples[0]), device=cube.device))
        for key in ("attempted", "solved", "kept", "length_ratios"):
            totals[key] += stats[key]
        totals["search_s"] += time.time() - t0
        extra = {"buffer": buffer.size, "search_attempted": totals["attempted"], "search_solved": totals["solved"],
                 "search_kept": totals["kept"], "search_s": round(totals["search_s"], 1),
                 "mean_length_ratio": round(float(np.mean(totals["length_ratios"])), 3) if totals["length_ratios"] else None}

        for _ in range(args.steps_per_round):
            if log.elapsed() >= log.budget:
                break
            trainer.set_lr(log.lr())
            states, labels, depths = cube.self_supervised_batch(args.walks, args.length, generator)
            if buffer.size > 0:
                _, b_states, b_labels, b_depths, _ = buffer.sample(n_search, generator)
                states, labels, depths = torch.cat([states, b_states]), torch.cat([labels, b_labels]), torch.cat([depths, b_depths])
            _, metrics = trainer.step(states, labels, depths, n_metrics=n_fresh)
            log.record(trainer.steps, len(states), metrics, extra)
    log.save("final.pt", trainer.steps)
    print(f"Terminé : {trainer.steps} pas, {totals['kept']} solutions retenues sur {totals['attempted']} cubes, "
          f"{log.elapsed():.0f} s", flush=True)


if __name__ == "__main__":
    main()
