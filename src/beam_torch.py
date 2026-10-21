"""
Recherche en faisceau en PyTorch, sur CPU ou GPU : un cube à la fois, tout le faisceau
évalué en un lot. Une seule synchronisation avec le CPU par profondeur (taille du faisceau suivant, solution
trouvée ou non), plus une copie pour reconstruire la solution : sur un GPU distant, chacune coûte environ 1 ms.

Le classement des enfants dépend de la tête du réseau :
- policy : log-probabilité cumulée du chemin (Takano, 2023), une évaluation par état du faisceau ;
- q      : distance estimée après chaque coup, lue en une passe sur le parent (Q* search, Agostinelli et al.,
           2024), une évaluation par état du faisceau ;
- value  : distance estimée de chaque enfant (DeepCubeA, CayleyPy), douze évaluations par état du faisceau.
Option bidirectionnelle : un second faisceau résout le cube inverse s⁻¹ avec le même réseau. Si z = s⁻¹ ∘ V
est un de ses états, b = s ∘ z = V est l'état atteint depuis le cube résolu par les mouvements V, et la
distance de b à s est celle de z au cube résolu : le réseau guide donc aussi la recherche partie de la
solution. Dès qu'un état avant s ∘ U coïncide avec un état arrière V, la solution est U suivie de l'inverse
de V.
"""
import bisect
import time

import torch


class NetworkHeuristic:
    """Classement des enfants par un réseau network_torch.ResMLP (conversion de symétrie comprise)."""

    def __init__(self, model, cube, mode=None, chunk=1 << 16):
        self.model, self.cube, self.chunk = model, cube, chunk
        config = getattr(model, "_orig_mod", model).config
        heads = config["heads"]
        self.mode = mode or ("policy" if "policy" in heads else "q" if "q" in heads else "value")
        if self.mode not in heads:
            raise ValueError(f"le réseau n'a pas de tête {self.mode} (têtes : {heads})")
        self.canonical = config["canonical"]
        self.encode = cube.encode if config["encoding"] == "pieces" else cube.encode_stickers
        self.amp = cube.device.type == "cuda"

    def evaluations(self, n_states):
        """Nombre d'évaluations du réseau pour développer n_states états du faisceau."""
        return 12 * n_states if self.mode == "value" else n_states

    @torch.no_grad()
    def outputs(self, states, head):
        results = []
        for start in range(0, len(states), self.chunk):
            s = states[start:start + self.chunk]
            if self.canonical:
                s, k = self.cube.canonicalize(s)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                out = self.model(self.encode(s))[head].float()
            if self.canonical and head != "value":
                out = out.gather(1, self.cube.action_perm[k])   # repère canonique -> repère d'origine
            results.append(out)
        return torch.cat(results)

    def __call__(self, states, kids, cum):
        """Retourne (coût de chaque enfant (N, 12), coût cumulé transmis aux enfants, évaluations)."""
        n = len(states)
        if self.mode == "policy":
            child_cum = cum[:, None] - torch.log_softmax(self.outputs(states, "policy"), dim=1)
            return child_cum, child_cum, n
        if self.mode == "q":
            q = self.outputs(states, "q")
            return q, torch.zeros_like(q), n
        h = self.outputs(kids.reshape(-1, 54), "value").reshape(n, 12)
        return h, torch.zeros_like(h), 12 * n


class _Beam:
    """
    Un faisceau : états courants, coût cumulé, et pour chaque profondeur le parent et le coup de chaque état
    retenu (pour reconstruire les chemins). Les empreintes de tous les états visités sont gardées triées,
    avec un identifiant global (rang de visite) pour retrouver leur chemin.
    """

    def __init__(self, cube, state, key_state):
        self.cube = cube
        self.states = state[None]
        self.cum = torch.zeros(1, device=cube.device)
        self.parents, self.moves = [], []
        self.offsets = [0]          # identifiant global du premier état de chaque profondeur
        self.size = 1
        self.seen_keys = cube.state_hash(key_state[None])
        self.seen_ids = torch.zeros(1, dtype=torch.long, device=cube.device)

    def lookup(self, keys):
        """(masque des empreintes déjà visitées, identifiant global de l'état correspondant)."""
        idx = torch.searchsorted(self.seen_keys, keys).clamp(max=len(self.seen_keys) - 1)
        return self.seen_keys[idx] == keys, self.seen_ids[idx]

    def path(self, gid):
        depth = bisect.bisect_right(self.offsets, gid) - 1
        if depth == 0:
            return []
        # une seule copie vers le CPU ; l'état d'identifiant g >= 1 a son parent et son coup en position g - 1
        parents, moves = torch.stack([torch.cat(self.parents[:depth]), torch.cat(self.moves[:depth])]).to(
            torch.int32).cpu().numpy()
        path = []
        while depth > 0:
            path.append(int(moves[gid - 1]))
            gid = self.offsets[depth - 1] + int(parents[gid - 1])
            depth -= 1
        return path[::-1]

    def child_path(self, j):
        """Chemin de l'enfant j (indice dans le dernier développement : parent j // 12, coup j % 12)."""
        return self.path(self.offsets[-1] + j // 12) + [j % 12]

    def expand(self, heuristic, key_fn):
        kids = self.cube.children(self.states)
        cost, cum, evaluations = heuristic(self.states, kids, self.cum)
        kids = kids.reshape(-1, 54)
        keys = self.cube.state_hash(key_fn(kids))
        seen, _ = self.lookup(keys)
        return kids, keys, cost.reshape(-1).masked_fill(seen, float("inf")), cum.reshape(-1), evaluations

    @staticmethod
    def candidates(keys, cost, width):
        """
        Les `width` meilleurs enfants neufs et distincts, sans synchronisation : retourne leurs indices triés par
        coût (les invalides, doublons ou états déjà vus, à la fin) et leur nombre, encore sur le device.
        """
        order = torch.argsort(cost)
        by_key = order[torch.argsort(keys[order], stable=True)]   # doublons regroupés, meilleur coût en premier
        grouped = keys[by_key]
        first = torch.ones_like(grouped, dtype=torch.bool)
        first[1:] = grouped[1:] != grouped[:-1]
        unique_cost = cost.scatter(0, by_key, cost[by_key].masked_fill(~first, float("inf")))
        best_cost, best = torch.sort(unique_cost)
        return best[:width], torch.isfinite(best_cost[:width]).sum()

    def select(self, kids, keys, cum, best):
        """Le faisceau suivant : les enfants d'indices `best` (non vide)."""
        self.parents.append(best // 12)
        self.moves.append(best % 12)
        self.states, self.cum = kids[best], cum[best]
        ids = torch.arange(self.size, self.size + len(best), device=keys.device)
        self.offsets.append(self.size)
        self.size += len(best)
        all_keys = torch.cat([self.seen_keys, keys[best]])
        order = torch.argsort(all_keys)
        self.seen_keys = all_keys[order]
        self.seen_ids = torch.cat([self.seen_ids, ids])[order]


def _to_host(*scalars):
    """Plusieurs scalaires du device copiés vers le CPU en une seule synchronisation."""
    return torch.stack([s.long() for s in scalars]).tolist()


@torch.no_grad()
def solve(heuristic, start, width=512, max_depth=40, bidirectional=False, max_evaluations=None, max_checks=16):
    """
    Cherche une solution pour l'état `start` (54,). Retourne un dict : solution (indices d'actions,
    vérifiée, ou None), evaluations (évaluations du réseau), depth (profondeur atteinte), time_s.
    """
    cube = heuristic.cube
    t0 = time.perf_counter()
    result = {"solution": None, "evaluations": 0, "depth": 0}

    def done(solution):
        result["solution"] = solution
        result["time_s"] = time.perf_counter() - t0
        return result

    def inverse_path(path):
        return [int(cube.inverse_action[a]) for a in reversed(path)]

    def verified(solution):
        return bool(cube.is_solved(cube.apply_sequence(start, solution)))

    start = start.to(cube.device)
    if cube.is_solved(start):
        return done([])
    forward = _Beam(cube, start, start)
    backward_key = lambda z: cube.compose(start.expand_as(z), z)   # b = s ∘ z
    if bidirectional:
        start_inverse = cube.inverse(start[None])[0]
        backward = _Beam(cube, start_inverse, backward_key(start_inverse[None])[0])
    forward_width = width // 2 if bidirectional else width

    for depth in range(1, max_depth + 1):
        result["depth"] = depth
        n_states = len(forward.states) + (len(backward.states) if bidirectional else 0)
        if max_evaluations is not None and result["evaluations"] + heuristic.evaluations(n_states) > max_evaluations:
            break

        kids, keys, cost, cum, evaluations = forward.expand(heuristic, lambda x: x)
        result["evaluations"] += evaluations
        best, n_best = forward.candidates(keys, cost, forward_width)
        hits = keys == cube.solved_hash
        if bidirectional:
            found, ids = backward.lookup(keys)
            n_best, any_hit, any_found = _to_host(n_best, hits.any(), found.any())
        else:
            (n_best, any_hit), any_found = _to_host(n_best, hits.any()), False
        if any_hit:
            for j in hits.nonzero().flatten()[:max_checks].tolist():
                if cube.is_solved(kids[j]):
                    return done(forward.child_path(j))
        if any_found:
            for j in found.nonzero().flatten()[:max_checks].tolist():
                solution = forward.child_path(j) + inverse_path(backward.path(int(ids[j])))
                if verified(solution):
                    return done(solution)
        if n_best == 0:
            break
        forward.select(kids, keys, cum, best[:n_best])

        if bidirectional:
            kids, keys, cost, cum, evaluations = backward.expand(heuristic, backward_key)
            result["evaluations"] += evaluations
            best, n_best = backward.candidates(keys, cost, width - forward_width)
            found, ids = forward.lookup(keys)
            n_best, any_found = _to_host(n_best, found.any())
            if any_found:
                for j in found.nonzero().flatten()[:max_checks].tolist():
                    solution = forward.path(int(ids[j])) + inverse_path(backward.child_path(j))
                    if verified(solution):
                        return done(solution)
            if n_best == 0:
                break
            backward.select(kids, keys, cum, best[:n_best])
    result["time_s"] = time.perf_counter() - t0
    return result
