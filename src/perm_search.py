"""
recherche du solveur sur un puzzle quelconque (perm_torch, cube_torch): le faisceau de beam_torch (racines
multiples, score V(parent) - lambda log pi) avec la table de fin de partie, construite par un parcours en largeur
générique (le graphe n'a pas à être biparti).

Table: tous les états à au plus `radius` coups du résolu, empreintes triées et distance exacte. Dès qu'un enfant
du faisceau y tombe, la fin du chemin est exacte et la recherche s'arrête: un état qui entre dans la table depuis
l'extérieur y est à distance radius, donc aucun niveau suivant ne fait mieux que le meilleur enfant du premier niveau
touché.
"""
import time

import torch

import beam_torch as bt


def lrx_longest(n):
    """Élément le plus long conjecturé de LRX (CayleyPy RL, éq. 1): (1 0 n-1 n-2 ... 3 2), de longueur n(n-1)/2."""
    return [1, 0] + list(range(n - 1, 1, -1))


class Ball:
    """les états à <= radius coups du résolu: empreintes triées (int64) et distance exacte (uint8)."""

    def __init__(self, puzzle, radius, chunk=1 << 19, verbose=False, max_states=None):
        """max_states: le rayon s'arrête au dernier niveau complet qui garde la table sous cette taille."""
        self.puzzle, self.radius = puzzle, radius
        t0 = time.time()
        frontier = puzzle.solved[None]
        levels = [puzzle.state_hash(frontier)]
        older = torch.empty(0, dtype=torch.long, device=puzzle.device)    # niveaux d - 2 et d - 1, triés
        self.spheres = [1]
        for d in range(1, radius + 1):
            keys, states = [], []
            for s in range(0, len(frontier), chunk):
                kids = puzzle.children(frontier[s:s + chunk]).reshape(-1, frontier.shape[-1])
                k, first = self.first_of_each(puzzle.state_hash(kids))
                keys.append(k)
                if d < radius:
                    states.append(kids[first])
            k, first = self.first_of_each(torch.cat(keys))
            # voisins d'un état du niveau d - 1: niveaux d - 2, d - 1 et d (graphe non orienté)
            seen = torch.cat([older, levels[-1]]).sort().values
            if len(seen):
                pos = torch.searchsorted(seen, k).clamp(max=len(seen) - 1)
                new = seen[pos] != k
                k, first = k[new], first[new]
            if max_states and sum(self.spheres) + len(k) > max_states:
                self.radius = d - 1
                break
            frontier = torch.cat(states)[first] if d < radius else None
            older = levels[-1]
            levels.append(k)
            self.spheres.append(len(k))
            if verbose:
                print(f"table : distance {d}, {len(k):,} états, {time.time() - t0:.0f} s", flush=True)
            if not len(k):
                break
        self.radius = len([k for k in levels if len(k)]) - 1      # graphe épuisé ou table plafonnée
        keys = torch.cat(levels)
        dist = torch.cat([torch.full((len(k),), d, dtype=torch.uint8, device=keys.device)
                          for d, k in enumerate(levels)])
        order = torch.argsort(keys)
        self.keys, self.dist = keys[order], dist[order]
        assert bool((self.keys[1:] != self.keys[:-1]).all()), "collision d'empreintes dans la table"

    @staticmethod
    def first_of_each(keys):
        order = torch.argsort(keys)
        k = keys[order]
        first = torch.ones_like(k, dtype=torch.bool)
        first[1:] = k[1:] != k[:-1]
        return k[first], order[first]

    def lookup(self, keys):
        idx = torch.searchsorted(self.keys, keys).clamp(max=len(self.keys) - 1)
        found = self.keys[idx] == keys
        return found, torch.where(found, self.dist[idx].long(), 255)

    def descend(self, state):
        """un plus court chemin exact vers le résolu, depuis un état de la table."""
        p, path = self.puzzle, []
        _, r = self.lookup(p.state_hash(state[None]))
        r = int(r)
        while r > 0:
            kids = p.children(state[None])[0]
            _, rk = self.lookup(p.state_hash(kids))
            a = int(torch.nonzero(rk == r - 1)[0])
            path.append(a)
            state, r = kids[a], r - 1
        return path


@torch.no_grad()
def solve(heuristic, start, width, views=("id", "inv"), ball=None, max_depth=100):
    """
    Faisceau à racines multiples (vues équivalentes, dédoublonnées), avec la table en option. Retourne un dict:
    solution (vérifiée, ou None), evaluations, depth, time_s, et ball_hit (profondeur où la table a été touchée).
    """
    p = heuristic.cube                         # le puzzle, quel qu'il soit (nom hérité de beam_torch)
    t0 = time.perf_counter()
    result = {"solution": None, "evaluations": 0, "depth": 0}

    def done(solution):
        if solution is not None and not bool(p.is_solved(p.apply_sequence(start, solution))):
            solution = None                    # collision d'empreintes: jamais observée, mais pas de faux succès
            result["error"] = "solution fausse"
        result["solution"] = solution
        result["time_s"] = time.perf_counter() - t0
        return result

    start = start.to(p.device)
    if p.is_solved(start):
        return done([])
    candidates = bt.equivalent_views(p, start, list(views))
    keys = p.state_hash(torch.stack([state for state, _ in candidates])).tolist()
    unique = [i for i, key in enumerate(keys) if key not in keys[:i]]    # l'inverse peut être l'état lui-même
    roots, backs = torch.stack([candidates[i][0] for i in unique]), [candidates[i][1] for i in unique]
    result["roots"] = len(unique)
    if ball is not None:
        found, r = ball.lookup(p.state_hash(roots))
        if bool(found.any()):
            j = int(r.argmin())
            result["ball_hit"] = 0
            return done(backs[j](ball.descend(roots[j])))
    forward = bt._Beam(p, roots, roots)
    for depth in range(1, max_depth + 1):
        result["depth"] = depth
        kids, keys, cost, cum, evaluations = forward.expand(heuristic, lambda x: x)
        result["evaluations"] += evaluations
        best, n_best = forward.candidates(keys, cost, width)
        if ball is not None:
            _, r = ball.lookup(keys)
            r_min, j = r.min(0)
            n_best, r_min, j = bt._to_host(n_best, r_min, j)
            if r_min < 255:
                moves, root = forward.child_path(j)
                result["ball_hit"] = depth
                return done(backs[root](moves + ball.descend(kids[j])))
        else:
            hits = keys == p.solved_hash
            n_best, any_hit = bt._to_host(n_best, hits.any())
            if any_hit:
                for j in hits.nonzero().flatten()[:16].tolist():
                    if p.is_solved(kids[j]):
                        moves, root = forward.child_path(j)
                        return done(backs[root](moves))
        if n_best == 0:
            break
        forward.select(kids, keys, cum, best[:n_best])
    result["time_s"] = time.perf_counter() - t0
    return result
