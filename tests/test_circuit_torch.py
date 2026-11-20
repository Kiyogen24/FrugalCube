import itertools
from collections import deque

import numpy as np
import pytest
import torch

import beam_torch as bt
import circuit_torch as qt
import network_torch as nt
import perm_search as ps
import perm_torch as pt


def test_coupling_edges():
    assert qt.coupling_edges("line", 5) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert len(qt.coupling_edges("full", 6)) == 15
    assert len(qt.coupling_edges("ring", 6)) == 6
    assert len(qt.coupling_edges("grid:2x3", 6)) == 7
    with pytest.raises(ValueError):
        qt.coupling_edges("grid:2x2", 5)


@pytest.mark.parametrize("topology", ["line", "full"])
def test_moves_children_and_self_inverse(topology):
    p = qt.CnotPuzzle(5, topology)
    s = p.random_states(30, torch.Generator().manual_seed(0))
    kids = p.children(s)
    for a in range(p.n_moves):
        one = p.apply_actions(s, torch.full((len(s),), a))
        assert torch.equal(kids[:, a], one)
        assert torch.equal(p.apply_actions(one, torch.full((len(s),), a)), s)
        assert torch.equal(p.apply_sequence(s[0], [a]), one[0])
    assert p.n_moves == 2 * len(p.edges)


def test_inverse_compose_encode_and_uniform_states():
    p = qt.CnotPuzzle(6, "full")
    s = p.random_states(200, torch.Generator().manual_seed(1))
    assert p.invertible(s).all()
    inv = p.inverse(s)
    eye = p.solved.expand_as(s)
    assert torch.equal(p.compose(s, inv), eye) and torch.equal(p.compose(inv, s), eye)
    m = p.to_matrices(s).numpy().astype(np.int64)
    assert torch.equal(p.encode(s), torch.as_tensor(m.reshape(200, -1), dtype=torch.float32))
    assert torch.equal(p.from_matrices(m), s)
    prod = (m @ p.to_matrices(inv).numpy().astype(np.int64)) % 2      # vérifié contre numpy
    assert (prod == np.eye(6, dtype=np.int64)).all()
    singular = p.from_matrices(np.array([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]] + np.eye(6, dtype=int)[2:].tolist()))
    assert not bool(p.invertible(singular[None])[0])
    # le rejet garde des matrices uniformes: sur GL(2, 2) (6 éléments), chaque élément environ 1/6
    p2 = qt.CnotPuzzle(2, "full")
    counts = torch.bincount(p2.rank(p2.random_states(6000, torch.Generator().manual_seed(2))), minlength=16)
    assert int((counts > 0).sum()) == 6 and counts.max() < 1150 and counts[counts > 0].min() > 850


def python_bfs(p):
    """Parcours en largeur indépendant, sur des matrices numpy."""
    n = p.n
    start = tuple(map(tuple, np.eye(n, dtype=int)))
    dist, queue = {start: 0}, deque([start])
    while queue:
        m = queue.popleft()
        for c, t in p.pairs:
            k = [list(r) for r in m]
            k[t] = [x ^ y for x, y in zip(k[t], k[c])]
            k = tuple(map(tuple, k))
            if k not in dist:
                dist[k] = dist[m] + 1
                queue.append(k)
    return dist


@pytest.mark.parametrize("topology", ["line", "full"])
def test_distance_table_matches_python_bfs(topology):
    p = qt.CnotPuzzle(3, topology)
    dist, spheres = qt.cnot_distance_table(p)
    ref = python_bfs(p)
    assert len(ref) == qt.gl_order(3) == 168 == sum(spheres)
    mats = np.array(list(ref.keys()))
    r = p.rank(p.from_matrices(mats))
    assert dist[r].tolist() == list(ref.values())


def test_walks_and_both_roots_give_circuits_that_realise_the_matrix():
    p = qt.CnotPuzzle(4, "line")
    walks, actions = p.random_walks(50, 12, torch.Generator().manual_seed(3))
    m, acts = walks[-1], actions.T                       # état final de chaque mélange, ses coups
    inv = p.inverse(m)
    _, back_inv = bt.equivalent_views(p, m[0], ["id", "inv"])[1]
    for i in range(len(m)):
        path = acts[i].tolist()
        assert bool(p.is_solved(p.apply_sequence(m[i], path[::-1])))          # coups inverses = mêmes coups
        assert bool(p.is_solved(p.apply_sequence(inv[i], path)))                # M⁻¹ = E_1 ... E_k
    # la vue inverse ramène une solution de M⁻¹ à une solution de M
    sol_inv = acts[0].tolist()
    assert bool(p.is_solved(p.apply_sequence(m[0], back_inv(sol_inv))))
    # le circuit (coups de la solution lus à l'envers) réalise M sur chaque vecteur de base
    sol = acts[0].tolist()[::-1]
    mat = p.to_matrices(m[:1])[0].numpy()
    for j in range(4):
        x = np.zeros(4, dtype=int)
        x[j] = 1
        for a in reversed(sol):
            c, t = p.pairs[a]
            x[t] ^= x[c]
        assert (x == mat[:, j]).all()


@pytest.mark.parametrize("topology", ["line", "full"])
def test_exhaustive_beam_and_ball_are_optimal(topology):
    n = 4
    p = qt.CnotPuzzle(n, topology)
    dist, spheres = qt.cnot_distance_table(p)
    net = nt.ResMLP(encoding="perm", hidden=0, width=16, blocks=0, heads=("policy", "value"), n_inputs=n * n,
                    n_moves=p.n_moves).eval()
    h = bt.NetworkHeuristic(net, p, mode="ac", lam=1.0)
    ball = ps.Ball(p, 3)
    assert ball.spheres == spheres[:4]
    starts = p.random_states(15, torch.Generator().manual_seed(4))
    for s in starts:
        exact = int(dist[p.rank(s[None])][0])
        for b in (None, ball):
            res = ps.solve(h, s, 10 ** 6, ball=b, max_depth=40)
            assert res["solution"] is not None and len(res["solution"]) == exact


def test_swap_line_distance_is_the_number_of_inversions():
    p = qt.SwapPuzzle(6, "line")
    assert p.n_moves == 5 and torch.equal(p.inverse_action, torch.arange(5))
    dist, _ = pt.distance_table(p)
    perms = torch.tensor(list(itertools.permutations(range(6))), dtype=torch.uint8)
    inversions = (perms[:, :, None] > perms[:, None, :]).triu(1).sum((1, 2))
    assert torch.equal(dist[p.rank(perms)].long(), inversions)
    grid = qt.SwapPuzzle(6, "grid:2x3")
    assert grid.n_moves == 7 and grid.name == "swap"
