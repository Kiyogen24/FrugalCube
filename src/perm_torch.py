"""
puzzles de permutation: graphes de Cayley du groupe symétrique S_n, avec la même interface que
cube_torch.TorchCube, pour que la recherche en faisceau (beam_torch) et l'entraînement servent tels quels.

- lrx: décalage circulaire à gauche L, à droite R, échange X des deux premières positions (OEIS A186783,
  étudié par CayleyPy RL, Chervov et al., 2025);
- pancake: retournement des k premiers éléments, k = 2..n.

Un état est une permutation de 0..n-1 stockée en uint8; un coup g agit par s -> s[g], comme les mouvements du
cube; l'état résolu est l'identité. Pour n <= 12, distance_table donne la distance exacte de tous les états par
un parcours en largeur, indexée par le rang de Lehmer de la permutation.
"""
import math

import numpy as np
import torch


def lrx_generators(n):
    left = list(range(1, n)) + [0]
    right = [n - 1] + list(range(n - 1))
    swap = [1, 0] + list(range(2, n))
    return [left, right, swap]


def pancake_generators(n):
    return [list(range(k - 1, -1, -1)) + list(range(k, n)) for k in range(2, n + 1)]


GENERATORS = {"lrx": lrx_generators, "pancake": pancake_generators}


def permutation_order(g):
    """Ordre de la permutation g (ppcm des longueurs de ses cycles)."""
    seen, order = [False] * len(g), 1
    for i in range(len(g)):
        length, j = 0, i
        while not seen[j]:
            seen[j], j, length = True, g[j], length + 1
        if length:
            order = order * length // math.gcd(order, length)
    return order


class PermPuzzle:
    """Tables d'un puzzle de permutation sur un device et opérations par lots (interface de TorchCube)."""

    def __init__(self, name, n, device="cpu"):
        gens = GENERATORS[name](n)
        self.name, self.n, self.n_moves = name, n, len(gens)
        self.device = torch.device(device)
        identity = list(range(n))
        inverse = [next(j for j, h in enumerate(gens) if [g[x] for x in h] == identity) for g in gens]

        def t(array, dtype=torch.long):
            return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

        self.moves = t(gens)                                            # (G, n)
        self.inverse_action = t(inverse)                                # (G,)
        self.inverse_or_none = t(inverse + [self.n_moves])              # (G + 1,), G: pas de coup
        # un coup d'ordre m répété plus de m // 2 fois s'écrit plus court avec son inverse: la règle des mélanges
        # l'interdit (pour le cube, m = 4: pas trois fois de suite le même coup)
        self.max_run = t([permutation_order(g) // 2 for g in gens] + [1 << 30])
        self.solved = t(identity, torch.uint8)
        rng = np.random.default_rng(12345)
        self.hash_weights = t(rng.integers(1, 1 << 62, size=n, dtype=np.int64) * 2 + 1)
        self.action_perm = t([list(range(self.n_moves))])               # pas de symétrie: l'identité seule
        self.solved_hash = self.state_hash(self.solved[None])[0]
        self.factorials = t([math.factorial(n - 1 - i) for i in range(n)])

    def to_numpy(self, states):
        return states.cpu().numpy().astype(np.int16)

    def from_numpy(self, states):
        return torch.as_tensor(np.asarray(states), dtype=torch.uint8, device=self.device)

    def apply_actions(self, states, actions):
        return states.gather(1, self.moves[actions])

    def apply_sequence(self, state, actions):
        for a in actions:
            state = state[self.moves[a]]
        return state

    def children(self, states):
        """les G enfants de chaque état: (N, n) -> (N, G, n)."""
        return states[:, self.moves]

    def inverse(self, states):
        index = torch.arange(self.n, device=states.device, dtype=torch.uint8).expand_as(states).contiguous()
        return torch.empty_like(states).scatter_(-1, states.long(), index)

    def compose(self, a, b):
        return a.gather(-1, b.long())

    def is_solved(self, states):
        return (states == self.solved).all(dim=-1)

    def state_hash(self, states):
        return (states.long() * self.hash_weights).sum(dim=-1)

    def encode(self, states, dtype=torch.float32):
        """one-hot n x n: quel élément occupe chaque position."""
        return torch.nn.functional.one_hot(states.long(), self.n).to(dtype).reshape(len(states), -1)

    def rank(self, states, chunk=1 << 18):
        """rang de Lehmer de chaque permutation, de 0 à n! - 1 (l'identité a le rang 0)."""
        out = []
        upper = torch.triu(torch.ones(self.n, self.n, dtype=torch.bool, device=states.device), diagonal=1)
        for start in range(0, len(states), chunk):
            s = states[start:start + chunk].long()
            smaller_after = (s[:, None, :] < s[:, :, None]) & upper    # [b, i, j]: j > i et s[j] < s[i]
            out.append((smaller_after.sum(2) * self.factorials).sum(1))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=states.device)

    def random_states(self, count, generator=None):
        """permutations tirées uniformément dans S_n (tous les états sont atteignables)."""
        return torch.rand(count, self.n, generator=generator, device=self.device).argsort(1).to(torch.uint8)

    def random_walks(self, n, length, generator=None):
        """
        n mélanges de `length` coups, règle du cube généralisée: jamais l'inverse du coup précédent, jamais un coup
        répété au-delà de la moitié de son ordre; chaque coup tiré uniformément parmi les coups permis, sans
        synchronisation. Retourne les états après chaque coup (length, n, size) et les coups (length, n).
        """
        G = self.n_moves
        states = self.solved.expand(n, self.n)
        none = torch.full((n,), G, dtype=torch.long, device=self.device)
        last, run = none, torch.zeros(n, dtype=torch.long, device=self.device)
        uniform = torch.rand(length, n, device=self.device, generator=generator)
        all_states, all_actions = [], []
        for step in range(length):
            inv = self.inverse_or_none[last]
            # le coup précédent est interdit s'il a atteint sa limite (sauf s'il est son propre inverse: déjà exclu)
            second = torch.where((run >= self.max_run[last]) & (inv != last), last, none)
            low, high = torch.minimum(inv, second), torch.maximum(inv, second)
            n_allowed = G - (inv < G).long() - (second < G).long()
            actions = torch.minimum((uniform[step] * n_allowed).long(), n_allowed - 1)
            actions = actions + (actions >= low)
            actions = actions + (actions >= high)
            states = self.apply_actions(states, actions)
            run = torch.where(actions == last, run + 1, torch.ones_like(run))
            last = actions
            all_states.append(states)
            all_actions.append(actions)
        return torch.stack(all_states), torch.stack(all_actions)

    def self_supervised_batch(self, n_walks, length, generator=None):
        """Comme TorchCube.self_supervised_batch: (états (n·length, size), étiquettes, profondeurs)."""
        states, actions = self.random_walks(n_walks, length, generator)
        labels = self.inverse_action[actions]
        depths = torch.arange(1, length + 1, device=self.device).unsqueeze(1).expand(length, n_walks)
        return states.reshape(-1, self.n), labels.reshape(-1), depths.reshape(-1)


def distance_table(puzzle, chunk=1 << 18, verbose=False):
    """
    Distance exacte de chaque état au résolu, par un parcours en largeur depuis l'identité (le graphe est non
    orienté: les générateurs sont fermés par inverse). Tenseur uint8 de taille n!, indexé par puzzle.rank.
    """
    total = math.factorial(puzzle.n)
    dist = torch.full((total,), 255, dtype=torch.uint8, device=puzzle.device)
    frontier = puzzle.solved[None]
    dist[puzzle.rank(frontier)] = 0
    d, spheres = 0, [1]
    while len(frontier):
        fresh_states = []
        for start in range(0, len(frontier), chunk):
            kids = puzzle.children(frontier[start:start + chunk]).reshape(-1, puzzle.n)
            r = puzzle.rank(kids)
            new = dist[r] == 255
            kids, r = kids[new], r[new]
            if not len(r):
                continue
            unique, inverse = torch.unique(r, return_inverse=True)
            first = torch.full((len(unique),), len(r), dtype=torch.long, device=r.device)
            first.scatter_reduce_(0, inverse, torch.arange(len(r), device=r.device), reduce="amin")
            dist[unique] = d + 1
            fresh_states.append(kids[first])
        if not fresh_states:
            break
        frontier = torch.cat(fresh_states)
        d += 1
        spheres.append(len(frontier))
        if verbose:
            print(f"  distance {d}: {len(frontier):,} états", flush=True)
    assert sum(spheres) == total, "graphe non connexe ?"
    return dist, spheres


def load_or_build_table(puzzle, path, verbose=True):
    """table des distances exactes, construite une fois puis lue sur disque (uint8, n! octets)."""
    import os
    import time
    if os.path.exists(path):
        dist = torch.from_numpy(np.fromfile(path, dtype=np.uint8)).to(puzzle.device)
        assert len(dist) == math.factorial(puzzle.n)
        return dist
    t0 = time.time()
    dist, spheres = distance_table(puzzle, verbose=verbose)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dist.cpu().numpy().tofile(path)
    if verbose:
        print(f"table {puzzle.name} n={puzzle.n} : {len(dist):,} états, diamètre {len(spheres) - 1}, "
              f"{time.time() - t0:.0f} s", flush=True)
    return dist
