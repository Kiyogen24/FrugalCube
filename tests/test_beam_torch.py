import math

import numpy as np
import pytest
import torch

import beam_torch as bt
import cube_torch as ct
import fast_cube as fc
import network_torch as nt

CUBE = ct.TorchCube("cpu")


class Uniform:
    """Heuristique constante : le faisceau devient un parcours en largeur exact s'il est assez large."""

    def __init__(self, mode="policy"):
        self.cube, self.mode = CUBE, mode

    def evaluations(self, n):
        return 12 * n if self.mode == "value" else n

    def __call__(self, states, kids, cum):
        n = len(states)
        if self.mode == "policy":
            child_cum = cum[:, None].expand(n, 12) + math.log(12)
            return child_cum, child_cum, n
        zeros = torch.zeros(n, 12)
        return zeros, zeros, self.evaluations(n)


def bfs_distances(max_depth):
    """Empreinte -> distance exacte au cube résolu, pour tous les états à au plus max_depth coups."""
    distances = {int(CUBE.solved_hash): 0}
    frontier = CUBE.solved[None]
    for d in range(1, max_depth + 1):
        kids = CUBE.children(frontier).reshape(-1, 54)
        keys = CUBE.state_hash(kids)
        _, first = np.unique(keys.numpy(), return_index=True)
        new = [i for i in first if int(keys[i]) not in distances]
        for i in new:
            distances[int(keys[i])] = d
        frontier = kids[new]
    return distances


DISTANCES = bfs_distances(5)


def scrambled(seed, depth):
    rng = np.random.default_rng(seed)
    state = CUBE.from_numpy(fc.scramble(rng, depth)[0])
    return state, DISTANCES[int(CUBE.state_hash(state[None])[0])]


def check(state, result):
    assert result["solution"] is not None
    assert CUBE.is_solved(CUBE.apply_sequence(state, result["solution"]))


@pytest.mark.parametrize("mode", ["policy", "q", "value"])
def test_wide_uniform_beam_finds_optimal_solutions(mode):
    for seed in range(12):
        state, distance = scrambled(seed, 1 + seed % 4)
        result = bt.solve(Uniform(mode), state, width=2048, max_depth=6)
        check(state, result)
        assert len(result["solution"]) == distance


def test_bidirectional_beams_meet_in_the_middle():
    for seed in range(8):
        state, distance = scrambled(100 + seed, 5)
        result = bt.solve(Uniform(), state, width=4096, max_depth=4, bidirectional=True)
        check(state, result)
        assert len(result["solution"]) == distance
        assert result["depth"] <= 3   # chaque faisceau n'a fait qu'environ la moitié du chemin


def test_evaluation_budget_and_failure():
    state = CUBE.from_numpy(fc.scramble(np.random.default_rng(7), 20)[0])
    result = bt.solve(Uniform(), state, width=64, max_depth=30, max_evaluations=500)
    assert result["solution"] is None and result["evaluations"] <= 500
    assert bt.solve(Uniform(), CUBE.solved, width=8)["solution"] == []


def test_canonical_heuristic_is_symmetry_equivariant(tmp_path):
    torch.manual_seed(0)
    model = nt.ResMLP(hidden=64, width=32, blocks=1, heads=("policy", "value", "q"), canonical=True).eval()
    heuristic = bt.NetworkHeuristic(model, CUBE)
    rng = np.random.default_rng(3)
    s = CUBE.from_numpy(fc.scramble(rng, 25)[0])
    conj = CUBE.conjugates(s[None])[0]                               # (48, 54)
    policy = heuristic.outputs(conj, "policy")
    value = heuristic.outputs(conj, "value")
    # l'action a sur s correspond à l'action ACTION_PERM[j, a] sur le j-ième conjugué
    mapped = policy.gather(1, CUBE.action_perm)
    assert torch.allclose(mapped, policy[0].expand_as(mapped), atol=1e-5)
    assert torch.allclose(value, value[0].expand_as(value), atol=1e-5)

    nt.save(model, tmp_path / "m.pt", elapsed_s=12.5)
    loaded, meta = nt.load(tmp_path / "m.pt")
    assert meta["elapsed_s"] == 12.5 and loaded.config == model.config
    x = CUBE.encode(conj)
    assert torch.allclose(loaded(x)["q"], model(x)["q"])
