import json
import os
import subprocess
import sys

import torch

import beam_torch as bt
import cube_torch as ct
import expert_iteration_torch as exit_torch
import network_torch as nt

CUBE = ct.TorchCube("cpu")


def test_search_examples_follow_verified_solutions():
    torch.manual_seed(0)
    net = nt.ResMLP(hidden=32, width=16, blocks=1).eval()
    heuristic = bt.NetworkHeuristic(net, CUBE)
    g = torch.Generator().manual_seed(0)
    # faisceau plus large que l'arbre à 3 coups : toutes les solutions sont trouvées, et jamais plus longues
    examples, stats = exit_torch.search_examples(heuristic, CUBE, 6, 3, 3, 2048, 5, 1.0, g)
    assert stats["attempted"] == 6 and stats["solved"] == 6 and stats["kept"] == 6
    states, labels, remaining = examples
    assert len(states) == len(labels) == len(remaining) and len(states) <= 18
    after = CUBE.apply_actions(states, labels)
    for i in range(len(states)):
        if remaining[i] == 1:
            assert CUBE.is_solved(after[i])            # le dernier coup de la solution résout le cube
        else:
            assert torch.equal(after[i], states[i + 1]) and remaining[i + 1] == remaining[i] - 1


def test_search_stops_at_deadline():
    net = nt.ResMLP(hidden=32, width=16, blocks=1).eval()
    examples, stats = exit_torch.search_examples(bt.NetworkHeuristic(net, CUBE), CUBE, 50, 20, 26, 256, 40, 1.0,
                                                 torch.Generator().manual_seed(0), deadline=0.0)
    assert examples is None and stats["attempted"] == 0


def run_exit(tmp_path, *extra):
    out = tmp_path / "exit"
    cmd = [sys.executable, os.path.join(os.path.dirname(exit_torch.__file__), "expert_iteration_torch.py"),
           "--minutes", "0.05", "--out-dir", str(out), "--hidden", "32", "--width", "16", "--blocks", "1",
           "--walks", "8", "--length", "6", "--log-seconds", "1", "--device", "cpu", *extra]
    subprocess.run(cmd, check=True, capture_output=True, cwd=tmp_path, timeout=120)
    return out


def test_expert_iteration_run(tmp_path):
    out = run_exit(tmp_path, "--cubes-per-round", "3", "--depth-min", "2", "--depth-max", "3",
                   "--search-width", "2048", "--steps-per-round", "3")
    entries = [json.loads(line) for line in open(out / "train_log.jsonl")]
    assert entries[-1]["search_kept"] > 0 and entries[-1]["buffer"] > 0
    model, meta = nt.load(out / "final.pt")
    assert meta["steps"] > 0


def test_expert_iteration_respects_time_budget_with_hard_cubes(tmp_path):
    # réseau non entraîné, cubes complètement mélangés : chaque recherche échoue après 40 profondeurs
    out = run_exit(tmp_path, "--cubes-per-round", "256", "--depth-min", "20", "--depth-max", "26",
                   "--search-width", "256")
    _, meta = nt.load(out / "final.pt")
    assert meta["elapsed_s"] < 3 + 10   # budget de 3 s, plus au plus une recherche en cours
