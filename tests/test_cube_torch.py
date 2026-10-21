import numpy as np
import pytest
import torch

import cube_torch as ct
import fast_cube as fc

CUBE = ct.TorchCube("cpu")


def random_states(n, seed=0, depth=25):
    rng = np.random.default_rng(seed)
    return np.stack([fc.scramble(rng, depth)[0] for _ in range(n)])


def test_tables_and_operations_match_fast_cube():
    states = random_states(64)
    t = CUBE.from_numpy(states)
    actions = np.random.default_rng(1).integers(0, 12, len(states))
    assert np.array_equal(CUBE.to_numpy(CUBE.apply_actions(t, torch.as_tensor(actions))), fc.apply_actions(states, actions))
    assert np.array_equal(CUBE.to_numpy(CUBE.children(t)), fc.children(states))
    assert np.array_equal(CUBE.to_numpy(CUBE.inverse(t)), fc.inverse(states))
    other = random_states(64, seed=2)
    assert np.array_equal(CUBE.to_numpy(CUBE.compose(t, CUBE.from_numpy(other))), fc.compose(states, other))
    assert CUBE.is_solved(CUBE.solved[None]).item()
    assert not CUBE.is_solved(t).any()


def test_hash_matches_fast_cube_as_int64():
    states = random_states(100, seed=3)
    expected = fc.state_hash(states).view(np.int64)
    assert np.array_equal(CUBE.state_hash(CUBE.from_numpy(states)).numpy(), expected)


def test_encodings():
    states = random_states(50, seed=4)
    t = CUBE.from_numpy(states)
    assert np.array_equal(CUBE.encode(t).numpy(), fc.encode(states))
    stickers = CUBE.encode_stickers(t)
    assert stickers.shape == (50, 324)
    assert torch.equal(stickers.reshape(50, 54, 6).sum(-1), torch.ones(50, 54))
    # chaque couleur apparaît 9 fois
    assert torch.equal(stickers.reshape(50, 54, 6).sum(1), torch.full((50, 6), 9.0))


def test_symmetries_match_fast_cube():
    states = random_states(40, seed=5)
    t = CUBE.from_numpy(states)
    assert np.array_equal(CUBE.to_numpy(CUBE.conjugates(t)), fc.conjugates(states))
    canonical, k = CUBE.canonicalize(t, chunk=16)
    expected_canonical, expected_k = fc.canonicalize(states)
    assert np.array_equal(CUBE.to_numpy(canonical), expected_canonical)
    assert np.array_equal(k.numpy(), expected_k)


def test_random_walks_follow_scramble_rule_and_labels_undo_last_move():
    g = torch.Generator().manual_seed(0)
    states, actions = CUBE.random_walks(200, 20, generator=g)
    assert states.shape == (20, 200, 54)
    a = actions.numpy()
    # jamais l'inverse du coup précédent, jamais trois fois le même coup
    assert not (a[1:] == fc.INVERSE_ACTION[a[:-1]]).any()
    assert not ((a[2:] == a[1:-1]) & (a[1:-1] == a[:-2])).any()
    # le mélange est bien la suite des coups appliqués au cube résolu
    first = CUBE.to_numpy(states[:, 0])
    assert np.array_equal(first[-1], fc.apply_sequence(fc.SOLVED, a[:, 0].tolist()))

    batch_states, labels, depths = CUBE.self_supervised_batch(50, 12, generator=g)
    assert batch_states.shape == (600, 54) and labels.shape == (600,) and depths.min() == 1 and depths.max() == 12
    # l'étiquette ramène au préfixe précédent : au cube résolu pour la profondeur 1
    undone = CUBE.apply_actions(batch_states, labels)
    assert CUBE.is_solved(undone[depths == 1]).all()
    depth2 = (depths == 2).nonzero().squeeze(1)
    assert torch.equal(undone[depth2], batch_states[depths == 1])


def test_random_walks_draw_uniformly_among_allowed_moves():
    g = torch.Generator().manual_seed(1)
    a = CUBE.random_walks(200_000, 3, generator=g)[1].numpy()

    def frequencies(moves):
        return np.bincount(moves, minlength=12) / len(moves)

    assert np.allclose(frequencies(a[0]), 1 / 12, atol=0.005)
    # après le coup 0 : tous les coups sauf son inverse
    second = frequencies(a[1][a[0] == 0])
    allowed = np.arange(12) != fc.INVERSE_ACTION[0]
    assert second[~allowed].sum() == 0 and np.allclose(second[allowed], 1 / 11, atol=0.01)
    # après deux fois le coup 0 : ni son inverse, ni une troisième fois
    third = frequencies(a[2][(a[0] == 0) & (a[1] == 0)])
    allowed &= np.arange(12) != 0
    assert third[~allowed].sum() == 0 and np.allclose(third[allowed], 1 / 10, atol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU seulement")
def test_data_generation_never_synchronizes_with_the_cpu():
    cube = ct.TorchCube("cuda")
    g = torch.Generator("cuda").manual_seed(0)
    torch.cuda.set_sync_debug_mode("error")      # toute synchronisation lève une erreur
    try:
        cube.self_supervised_batch(100, 26, g)
    finally:
        torch.cuda.set_sync_debug_mode("default")
