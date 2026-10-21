import random

import numpy as np

import fast_cube as fc
from Cube import RubiksCube


def pycuber_state(sequence):
    cube = RubiksCube()
    for a in sequence:
        cube.rotate(fc.ACTIONS[a])
    return cube


def test_moves_match_pycuber_and_encoding_matches_cube_py():
    rng = random.Random(0)
    for _ in range(200):
        sequence = [rng.randrange(12) for _ in range(rng.randint(0, 30))]
        state = fc.apply_sequence(fc.SOLVED, sequence)
        cube = pycuber_state(sequence)
        assert fc.kociemba_string(state) == cube.get_kociemba_definition_string()
        assert np.array_equal(fc.encode(state[None])[0], cube.get_matrix_state())


def test_inverse_and_compose():
    rng = np.random.default_rng(1)
    for _ in range(50):
        state, sequence = fc.scramble(rng, 20)
        inv = fc.inverse(state[None])[0]
        # l'inverse s'obtient aussi en appliquant la séquence inverse au cube résolu
        assert np.array_equal(inv, fc.apply_sequence(fc.SOLVED, [fc.INVERSE_ACTION[a] for a in reversed(sequence)]))
        assert fc.is_solved(fc.compose(state[None], inv[None]))[0]
        assert fc.is_solved(fc.compose(inv[None], state[None]))[0]


def test_children_and_batch_apply():
    rng = np.random.default_rng(2)
    states = np.stack([fc.scramble(rng, 10)[0] for _ in range(5)])
    kids = fc.children(states)
    assert kids.shape == (5, 12, 54)
    for a in range(12):
        assert np.array_equal(kids[:, a], fc.apply_actions(states, np.full(5, a)))
        undone = fc.apply_actions(kids[:, a], np.full(5, fc.INVERSE_ACTION[a]))
        assert np.array_equal(undone, states)


def test_hash_distinguishes_bfs_states():
    frontier = fc.SOLVED[None]
    seen = {int(fc.state_hash(frontier)[0])}
    total = 1
    for _ in range(3):
        kids = fc.children(frontier).reshape(-1, 54)
        hashes = fc.state_hash(kids)
        _, first = np.unique(hashes, return_index=True)
        new = [i for i in first if int(hashes[i]) not in seen]
        seen.update(int(hashes[i]) for i in new)
        frontier = kids[new]
        total += len(new)
    assert total == 1 + 12 + 114 + 1068


def test_symmetry_group():
    assert len(fc.SYMMETRIES) == 48
    assert len({tuple(g) for g in fc.SYMMETRIES.tolist()}) == 48
    assert np.array_equal(fc.SYMMETRIES[0], fc.SOLVED)
    for row in fc.ACTION_PERM:
        assert sorted(row.tolist()) == list(range(12))
    # les symétries avec miroir changent le sens de rotation : R devient un quart de tour anti-horaire
    clockwise = {a for a, name in enumerate(fc.ACTIONS) if not name.endswith("'")}
    flips = [fc.ACTION_PERM[k][fc.ACTIONS.index("R")] not in clockwise for k in range(48)]
    assert sum(flips) == 24


def test_conjugation_preserves_solutions_and_distance():
    rng = np.random.default_rng(3)
    for _ in range(30):
        state, sequence = fc.scramble(rng, 12)
        solution = [int(fc.INVERSE_ACTION[a]) for a in reversed(sequence)]
        conj = fc.conjugates(state[None])[0]
        for k in range(48):
            mapped = [int(fc.ACTION_PERM[k, a]) for a in solution]
            assert fc.is_solved(fc.apply_sequence(conj[k], mapped))
    assert fc.is_solved(fc.conjugates(fc.SOLVED[None])[0]).all()


def test_canonical_form_is_symmetry_invariant():
    rng = np.random.default_rng(4)
    for _ in range(20):
        state, _ = fc.scramble(rng, 15)
        canonical, _ = fc.canonicalize(state[None])
        for sym_state in fc.conjugates(state[None])[0]:
            assert np.array_equal(fc.canonicalize(sym_state[None])[0], canonical)


def test_canonical_policy_mapping():
    # le coup qui défait le dernier coup du mélange, exprimé dans l'état canonique, défait bien l'état canonique
    rng = np.random.default_rng(5)
    states, labels = [], []
    for _ in range(50):
        state, sequence = fc.scramble(rng, 1 + int(rng.integers(0, 10)))
        states.append(state)
        labels.append(int(fc.INVERSE_ACTION[sequence[-1]]))
    canonical, k = fc.canonicalize(np.stack(states))
    for c, s, label, sym in zip(canonical, states, labels, k):
        undone_original = fc.apply_sequence(s, [label])
        undone_canonical = fc.apply_sequence(c, [int(fc.ACTION_PERM[sym, label])])
        assert np.array_equal(fc.canonicalize(undone_canonical[None])[0], fc.canonicalize(undone_original[None])[0])


#FIN
