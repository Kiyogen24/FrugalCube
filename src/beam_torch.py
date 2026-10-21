"""
Recherche en faisceau en PyTorch, sur CPU ou GPU : un cube à la fois, tout le faisceau
évalué en un lot, sans aller-retour avec le CPU sauf pour reconstruire la solution.

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
        i = gid - self.offsets[depth]
        moves = []
        while depth > 0:
            moves.append(int(self.moves[depth - 1][i]))
            i = int(self.parents[depth - 1][i])
            depth -= 1
        return moves[::-1]

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

    def select(self, kids, keys, cost, cum, width):
        """Garde les `width` meilleurs enfants neufs et distincts ; retourne leur nombre."""
        order = torch.argsort(cost)
        by_key = torch.argsort(keys[order], stable=True)     # doublons regroupés, meilleur coût en premier
        grouped = keys[order][by_key]
        first = torch.ones_like(grouped, dtype=torch.bool)
        first[1:] = grouped[1:] != grouped[:-1]
        best = order[by_key[first]]
        best = best[torch.isfinite(cost[best])]
        if len(best) > width:
            best = best[torch.topk(cost[best], width, largest=False).indices]
        if len(best) == 0:
            return 0
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
        return len(best)


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
        for j in (keys == cube.solved_hash).nonzero().flatten()[:max_checks].tolist():
            if cube.is_solved(kids[j]):
                return done(forward.child_path(j))
        if bidirectional:
            found, ids = backward.lookup(keys)
            for j in found.nonzero().flatten()[:max_checks].tolist():
                solution = forward.child_path(j) + inverse_path(backward.path(int(ids[j])))
                if verified(solution):
                    return done(solution)
        if forward.select(kids, keys, cost, cum, forward_width) == 0:
            break

        if bidirectional:
            kids, keys, cost, cum, evaluations = backward.expand(heuristic, backward_key)
            result["evaluations"] += evaluations
            found, ids = forward.lookup(keys)
            for j in found.nonzero().flatten()[:max_checks].tolist():
                solution = forward.path(int(ids[j])) + inverse_path(backward.child_path(j))
                if verified(solution):
                    return done(solution)
            if backward.select(kids, keys, cost, cum, width - forward_width) == 0:
                break
    result["time_s"] = time.perf_counter() - t0
    return result
