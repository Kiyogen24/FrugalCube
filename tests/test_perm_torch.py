import itertools
import math
from collections import deque

import pytest
import torch

import beam_torch as bt
import network_torch as nt
import perm_torch as pt


@pytest.mark.parametrize("name", ["lrx", "pancake"])
def test_inverse_actions(name):
    p = pt.PermPuzzle(name, 7)
    s = p.random_states(50)
    for a in range(p.n_moves):
        back = p.apply_actions(p.apply_actions(s, torch.full((50,), a)), torch.full((50,), int(p.inverse_action[a])))
        assert torch.equal(back, s)


def test_inverse_and_compose():
    p = pt.PermPuzzle("lrx", 9)
    s = p.random_states(100)
    assert p.is_solved(p.compose(s, p.inverse(s))).all()
    assert p.is_solved(p.compose(p.inverse(s), s)).all()


def test_rank_is_a_bijection():
    p = pt.PermPuzzle("lrx", 6)
    perms = torch.tensor(list(itertools.permutations(range(6))), dtype=torch.uint8)
    r = p.rank(perms)
    assert sorted(r.tolist()) == list(range(720))
    assert int(p.rank(p.solved[None])[0]) == 0


@pytest.mark.parametrize("name", ["lrx", "pancake"])
def test_distance_table_matches_python_bfs(name):
    n = 6
    p = pt.PermPuzzle(name, n)
    dist, spheres = pt.distance_table(p, chunk=37)
    gens = p.moves.tolist()
    start = tuple(range(n))
    seen = {start: 0}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        for g in gens:
            t = tuple(s[i] for i in g)
            if t not in seen:
                seen[t] = seen[s] + 1
                queue.append(t)
    perms = torch.tensor(list(seen), dtype=torch.uint8)
    assert torch.equal(dist[p.rank(perms)].long(), torch.tensor(list(seen.values())))
    assert sum(spheres) == math.factorial(n)


@pytest.mark.parametrize("name", ["lrx", "pancake"])
def test_walk_rule_and_uniformity(name):
    p = pt.PermPuzzle(name, 8)
    states, actions = p.random_walks(20000, 30, generator=torch.Generator().manual_seed(0))
    inv = p.inverse_action[actions[:-1]]
    assert not (actions[1:] == inv).any()                    # jamais l'inverse du coup précédent
    run = torch.ones_like(actions[0])
    for t in range(1, len(actions)):
        run = torch.where(actions[t] == actions[t - 1], run + 1, torch.ones_like(run))
        assert (run <= p.max_run[actions[t]]).all()          # jamais au-delà de la moitié de l'ordre
    # uniformité parmi les coups permis: après X sur LRX, L et R chacun la moitié du temps
    if name == "lrx":
        after_x = actions[1:][actions[:-1] == 2]
        share = (after_x == 0).float().mean().item()
        assert abs(share - 0.5) < 0.02
    # les états suivent les coups
    s = p.solved.expand(20000, 8)
    for t in range(30):
        s = p.apply_actions(s, actions[t])
        assert torch.equal(s, states[t])


def test_max_run_generalises_the_cube_rule():
    p = pt.PermPuzzle("lrx", 11)
    assert p.max_run[:3].tolist() == [5, 5, 1]               # L et R d'ordre 11, X d'ordre 2
    assert pt.permutation_order([1, 2, 3, 0]) == 4           # un quart de tour: au plus 2 de suite, comme le cube


@pytest.mark.parametrize("name", ["lrx", "pancake"])
def test_exhaustive_beam_is_optimal_with_two_roots(name):
    n = 7
    p = pt.PermPuzzle(name, n)
    dist, _ = pt.distance_table(p)
    net = nt.ResMLP(encoding="perm", hidden=0, width=32, blocks=0, heads=("policy", "value"), n_inputs=n * n,
                    n_moves=p.n_moves).eval()
    heuristic = bt.NetworkHeuristic(net, p, mode="ac", lam=1.0)
    starts = p.random_states(20, generator=torch.Generator().manual_seed(1))
    for s in starts:
        result = bt.solve(heuristic, s, width=10 ** 5, max_depth=60, views=["id", "inv"])
        assert result["solution"] is not None
        assert len(result["solution"]) == int(dist[p.rank(s[None])][0])
    heuristic = bt.NetworkHeuristic(net, p, mode="value")
    result = bt.solve(heuristic, starts[0], width=64, max_depth=200)
    assert result["solution"] is not None


def test_arm_targets():
    import copy
    import train_perm as tp
    p = pt.PermPuzzle("lrx", 7)
    dist, _ = pt.distance_table(p)
    states, labels, depths = p.self_supervised_batch(200, 30, torch.Generator().manual_seed(3))
    pol, val = tp.targets("oracle", p, states, labels, depths, None, dist, 10.0, False)
    d = dist[p.rank(states)].long()
    assert torch.equal(val.long(), d)
    kids_d = dist[p.rank(p.children(states).reshape(-1, 7))].long().reshape(len(states), -1)
    assert torch.allclose(pol[d > 0].sum(1), torch.ones(int((d > 0).sum())))
    assert (kids_d[pol > 0] == (d.unsqueeze(1).expand_as(pol))[pol > 0] - 1).all()
    net = nt.ResMLP(encoding="perm", hidden=0, width=16, blocks=0, heads=("policy", "value"), n_inputs=49,
                    n_moves=3).eval()
    target = copy.deepcopy(net)
    for arm in ("bellman", "improve"):
        pol, val = tp.targets(arm, p, states, labels, depths, target, None, 10.0, False)
        assert (val <= depths.float()).all() and (val >= 0).all()
        assert (val[p.is_solved(states)] == 0).all()
        assert pol.shape == labels.shape and pol.max() < 3
    assert torch.equal(tp.targets("bellman", p, states, labels, depths, target, None, 10.0, False)[0], labels)


def test_split_keeps_layout():
    import train_torch as tt
    p = pt.PermPuzzle("lrx", 8)
    states, labels, depths = p.self_supervised_batch(12, 5, torch.Generator().manual_seed(0))
    parts = tt.split_batches(states, labels, depths, 5, 3)
    assert len(parts) == 3 and all(len(s) == 20 for s, _, _ in parts)
    for s, lab, dep in parts:
        assert torch.equal(dep, torch.arange(1, 6).repeat_interleave(4))
        # chaque préfixe suivi de son étiquette revient au préfixe précédent du même mélange
        back = p.apply_actions(s[4:], lab[4:])
        assert torch.equal(back, s[:-4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU seulement")
@pytest.mark.parametrize("arm", ["ss", "bellman", "improve", "oracle"])
def test_training_step_never_synchronizes_with_the_cpu(arm):
    import copy
    import train_perm as tp
    p = pt.PermPuzzle("lrx", 8, "cuda")
    dist, _ = pt.distance_table(p)
    net = nt.ResMLP(encoding="perm", hidden=32, width=16, blocks=1, heads=("policy", "value"), n_inputs=64,
                    n_moves=3).cuda()
    target = copy.deepcopy(net).eval()
    g = torch.Generator("cuda").manual_seed(0)
    torch.cuda.set_sync_debug_mode("error")      # toute synchronisation lève une erreur
    try:
        states, labels, depths = p.self_supervised_batch(100, 40, g)
        pol, val = tp.targets(arm, p, states, labels, depths, target, dist, 10.0, True)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("n", [7, 8])
def test_longest_element_and_ball(n):
    import perm_search as ps
    p = pt.PermPuzzle("lrx", n)
    dist, spheres = pt.distance_table(p)
    longest = torch.tensor(ps.lrx_longest(n), dtype=torch.uint8)
    assert int(dist[p.rank(longest[None])][0]) == n * (n - 1) // 2 == len(spheres) - 1
    ball = ps.Ball(p, 6)
    assert ball.spheres == spheres[:7]                       # graphe non biparti pour n impair: niveaux d-2 et d-1
    s = p.random_states(200, torch.Generator().manual_seed(2))
    found, r = ball.lookup(p.state_hash(s))
    exact = dist[p.rank(s)].long()
    assert torch.equal(found, exact <= 6) and torch.equal(r[found], exact[found])
    net = nt.ResMLP(encoding="perm", hidden=0, width=16, blocks=0, heads=("policy", "value"), n_inputs=n * n,
                    n_moves=3).eval()
    h = bt.NetworkHeuristic(net, p, mode="ac", lam=1.0)
    for st in list(s[:10]) + [longest]:
        res = ps.solve(h, st, 10 ** 5, ball=ball, max_depth=60)  # faisceau exhaustif: longueur optimale
        assert res["solution"] is not None
        assert len(res["solution"]) == int(dist[p.rank(st[None])][0])


def test_checkpoint_round_trip(tmp_path):
    p = pt.PermPuzzle("lrx", 8)
    net = nt.ResMLP(encoding="perm", hidden=16, width=8, blocks=1, heads=("policy", "value"), n_inputs=64, n_moves=3)
    nt.save(net, tmp_path / "m.pt")
    model, _ = nt.load(tmp_path / "m.pt")
    out = model(p.encode(p.random_states(4)))
    assert out["policy"].shape == (4, 3) and out["value"].shape == (4,)
    cube_net = nt.ResMLP(hidden=16, width=8, blocks=1)
    assert "n_moves" not in cube_net.config                  # config du cube inchangée
