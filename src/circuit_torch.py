"""
synthèse de circuits quantiques vue comme un graphe de Cayley, avec l'interface de perm_torch.PermPuzzle (donc de
TorchCube): la recherche (beam_torch, perm_search) et l'entraînement (train_perm) servent tels quels.

- cnot: fonctions linéaires réversibles, matrices M inversibles n x n sur GF(2) (le groupe GL(n, 2)). Le coup
  CNOT(c, t), sur une paire de qubits couplés, ajoute la ligne c à la ligne t (M -> E M avec E = I + e_t e_c^T);
  l'état résolu est l'identité. Si E_k ... E_1 M = I, alors M = E_1 ... E_k (chaque CNOT est son propre inverse):
  le circuit qui réalise M applique les coups trouvés dans l'ordre inverse, et son nombre de CNOT est la longueur
  du chemin.
- swap: permutations des qubits (token swapping); le coup SWAP(a, b) échange deux positions couplées.

Un état cnot est une matrice rangée par lignes, chaque ligne un masque de n bits (int64, n <= 62); un état swap est
une permutation (uint8), comme dans perm_torch. Deux racines: la solution de M⁻¹ lue à l'envers (coups inversés)
résout M, comme pour le cube (beam_torch.equivalent_views).

Topologies: line, ring, full (toutes les paires), grid:RxC, ou un fichier JSON {"n": .., "edges": [[a, b], ...]}
(p. ex. un bloc d'un processeur IBM).
"""
import json
import math
import os

import numpy as np
import torch

import perm_torch as pt


def coupling_edges(topology, n):
    """Arêtes non orientées (a, b), a < b, triées, du graphe de couplage (connexe)."""
    if topology == "line":
        edges = [(i, i + 1) for i in range(n - 1)]
    elif topology == "ring":
        edges = [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)]
    elif topology == "full":
        edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    elif topology.startswith("grid:"):
        rows, cols = map(int, topology[5:].split("x"))
        if rows * cols != n:
            raise ValueError(f"grille {rows}x{cols} pour n = {n}")
        edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
        edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
    else:
        with open(topology) as f:
            spec = json.load(f)
        if spec["n"] != n:
            raise ValueError(f"{topology} : {spec['n']} qubits, pas {n}")
        edges = [(min(a, b), max(a, b)) for a, b in spec["edges"]]
    edges = sorted(set(edges))
    reached, todo = {0}, [0]
    while todo:
        a = todo.pop()
        for u, v in edges:
            for x, y in ((u, v), (v, u)):
                if x == a and y not in reached:
                    reached.add(y)
                    todo.append(y)
    if len(reached) != n:
        raise ValueError(f"graphe de couplage non connexe ({topology}, n = {n})")
    return edges


def gl_order(n):
    """nombre de matrices inversibles n x n sur GF(2)."""
    return math.prod((1 << n) - (1 << i) for i in range(n))


class CnotPuzzle(pt.PermPuzzle):
    """
    fonctions linéaires réversibles sur un graphe de couplage; coups = CNOT dans les deux sens de chaque arête.
    Mélanges: la règle des puzzles de permutation (jamais le coup précédent, son propre inverse). Mesuré le
    2026-11-19 (topologie IBM 9qL, solutions gloutonnes de la politique IBM): les mélanges se replient beaucoup
    (300 coups -> environ 69 CNOT, un état uniforme -> environ 78). Imposer l'ordre des CNOT qui commutent replie
    davantage (400 coups -> 53): les coups d'indice élevé alternent alors avec CX(b, a), d'ordre 3 avec CX(a, b).
    """

    def __init__(self, n, topology="line", device="cpu"):
        if not 2 <= n <= 62:
            raise ValueError("2 <= n <= 62 (une ligne par int64)")
        self.name, self.n, self.topology = "cnot", n, topology
        self.device = torch.device(device)
        self.edges = coupling_edges(topology, n)
        self.pairs = [p for a, b in self.edges for p in ((a, b), (b, a))]     # (contrôle, cible)
        G = self.n_moves = len(self.pairs)

        def t(array, dtype=torch.long):
            return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

        self.control = t([c for c, _ in self.pairs])
        self.target = t([x for _, x in self.pairs])
        self.inverse_action = t(range(G))                       # chaque CNOT est son propre inverse
        self.inverse_or_none = t(list(range(G)) + [G])
        self.max_run = t([1] * G + [1 << 30])
        self.solved = t([1 << i for i in range(n)])
        rng = np.random.default_rng(12345)
        self.hash_weights = t(rng.integers(1, 1 << 62, size=n, dtype=np.int64) * 2 + 1)
        self.action_perm = t([list(range(G))])
        self.solved_hash = self.state_hash(self.solved[None])[0]
        self.bits = t(range(n))

    # --- conversions -------------------------------------------------------------------------------------------
    def to_numpy(self, states):
        return states.cpu().numpy().astype(np.int64)

    def from_numpy(self, states):
        return torch.as_tensor(np.asarray(states, dtype=np.int64), device=self.device)

    def to_matrices(self, states):
        """(N, n) masques -> (N, n, n) uint8, M[i, j] = bit j de la ligne i."""
        return ((states[..., :, None] >> self.bits) & 1).to(torch.uint8)

    def from_matrices(self, matrices):
        m = torch.as_tensor(np.asarray(matrices), device=self.device).long()
        return (m << self.bits).sum(-1)

    # --- coups ------------------------------------------------------------------------------------------------
    def apply_actions(self, states, actions):
        rows = torch.arange(len(states), device=states.device)
        c, t = self.control[actions], self.target[actions]
        out = states.clone()
        out[rows, t] = states[rows, t] ^ states[rows, c]
        return out

    def apply_sequence(self, state, actions):
        state = state.clone()
        for a in actions:
            c, t = self.pairs[int(a)]
            state[t] = state[t] ^ state[c]
        return state

    def children(self, states):
        """les G enfants de chaque état: (N, n) -> (N, G, n)."""
        N, G = len(states), self.n_moves
        kids = states[:, None, :].expand(N, G, self.n).clone()
        new = states[:, self.target] ^ states[:, self.control]          # (N, G): nouvelle ligne cible
        kids.scatter_(2, self.target.view(1, G, 1).expand(N, G, 1), new.unsqueeze(2))
        return kids

    def _eliminate(self, states):
        """Gauss-Jordan sur GF(2) par lots: (inverses, masque des matrices inversibles)."""
        a = states.clone()
        b = self.solved.expand_as(states).clone()
        N, n = a.shape
        rows = torch.arange(N, device=a.device)
        idx = torch.arange(n, device=a.device)
        ok = torch.ones(N, dtype=torch.bool, device=a.device)
        for j in range(n):
            bit = ((a >> j) & 1).bool() & (idx >= j)
            ok &= bit.any(1)
            p = bit.to(torch.uint8).argmax(1)                            # premier pivot (singulière: sans objet)
            aj, ap, bj, bp = a[rows, j], a[rows, p], b[rows, j], b[rows, p]
            a[rows, p], b[rows, p] = aj, bj
            a[rows, j], b[rows, j] = ap, bp
            mask = ((a >> j) & 1).bool()
            mask[:, j] = False
            a = a ^ torch.where(mask, a[:, j:j + 1], 0)
            b = b ^ torch.where(mask, b[:, j:j + 1], 0)
        return b, ok

    def inverse(self, states):
        inv, ok = self._eliminate(states)
        return inv

    def invertible(self, states):
        return self._eliminate(states)[1]

    def compose(self, a, b):
        """Produit matriciel A B sur GF(2) (lignes en masques)."""
        out = torch.zeros_like(a)
        for j in range(self.n):
            out = out ^ torch.where(((a >> j) & 1).bool(), b[..., j:j + 1], 0)
        return out

    def encode(self, states, dtype=torch.float32):
        """les n x n bits de la matrice."""
        return ((states[:, :, None] >> self.bits) & 1).to(dtype).reshape(len(states), -1)

    def rank(self, states, chunk=None):
        """indice de la matrice parmi les 2^(n^2) (table exacte, n <= 5)."""
        return (states << (self.bits * self.n)).sum(-1)

    def random_states(self, count, generator=None):
        """matrices tirées uniformément dans GL(n, 2) (rejet des singulières, environ 71 %)."""
        out, have = [], 0
        while have < count:
            m = torch.randint(0, 1 << self.n, (4 * count + 16, self.n), generator=generator, device=self.device)
            m = m[self.invertible(m)]
            out.append(m)
            have += len(m)
        return torch.cat(out)[:count]


class SwapPuzzle(pt.PermPuzzle):
    """Permutations des qubits; coups = SWAP sur les arêtes du graphe de couplage (token swapping)."""

    def __init__(self, n, topology="line", device="cpu"):
        self.edges = coupling_edges(topology, n)
        gens = []
        for a, b in self.edges:
            g = list(range(n))
            g[a], g[b] = b, a
            gens.append(g)
        key = f"swap:{topology}:{n}"
        pt.GENERATORS[key] = lambda m, gens=gens: gens          # PermPuzzle lit ses générateurs dans ce registre
        super().__init__(key, n, device)
        self.name, self.topology = "swap", topology
        self.pairs = list(self.edges)


def make_puzzle(name, n, topology, device="cpu"):
    if name == "cnot":
        return CnotPuzzle(n, topology, device)
    if name == "swap":
        return SwapPuzzle(n, topology, device)
    raise ValueError(name)


def cnot_distance_table(puzzle, chunk=1 << 18, verbose=False):
    """distance exacte de toutes les matrices inversibles (n <= 5): uint8 de taille 2^(n^2), 255 hors du groupe."""
    if puzzle.n > 5:
        raise ValueError("table exacte : n <= 5 (GL(6, 2) a 2·10^10 éléments)")
    dist = torch.full((1 << (puzzle.n * puzzle.n),), 255, dtype=torch.uint8, device=puzzle.device)
    frontier = puzzle.solved[None]
    dist[puzzle.rank(frontier)] = 0
    d, spheres = 0, [1]
    while len(frontier):
        fresh = []
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
            fresh.append(kids[first])
        if not fresh:
            break
        frontier = torch.cat(fresh)
        d += 1
        spheres.append(len(frontier))
        if verbose:
            print(f"  distance {d}: {len(frontier):,} matrices", flush=True)
    assert sum(spheres) == gl_order(puzzle.n), "graphe non connexe ?"
    return dist, spheres


def load_or_build_table(puzzle, path, verbose=True):
    """Table exacte (cnot: cnot_distance_table; swap: perm_torch.distance_table), construite une fois."""
    if os.path.exists(path):
        return torch.from_numpy(np.fromfile(path, dtype=np.uint8)).to(puzzle.device)
    build = cnot_distance_table if puzzle.name == "cnot" else pt.distance_table
    dist, spheres = build(puzzle, verbose=verbose)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dist.cpu().numpy().tofile(path)
    if verbose:
        print(f"table {puzzle.name} {puzzle.topology} n={puzzle.n} : sphères {spheres}", flush=True)
    return dist
