"""
Recherche en faisceau en PyTorch, sur CPU ou GPU : un cube à la fois, tout le faisceau
évalué en un lot. Une seule synchronisation avec le CPU par profondeur (taille du faisceau suivant, solution
trouvée ou non), plus une copie pour reconstruire la solution : sur un GPU distant, chacune coûte environ 1 ms.

Le classement des enfants dépend de la tête du réseau :
- policy : log-probabilité cumulée du chemin (Takano, 2023), une évaluation par état du faisceau ;
- q      : distance estimée après chaque coup, lue en une passe sur le parent (Q* search, Agostinelli et al.,
           2024), une évaluation par état du faisceau ;
- value  : distance estimée de chaque enfant (DeepCubeA, CayleyPy), douze évaluations par état du faisceau ;
- ac     : acteur-critique, coût V(p) - lam * log pi(a | p) pour l'enfant (p, a) : la valeur du parent compare
           les états du faisceau entre eux, la politique choisit le coup ; un réseau à deux têtes (policy et
           value), une évaluation par état du faisceau.
Option bidirectionnelle : un second faisceau résout le cube inverse s⁻¹ avec le même réseau. Si z = s⁻¹ ∘ V
est un de ses états, b = s ∘ z = V est l'état atteint depuis le cube résolu par les mouvements V, et la
distance de b à s est celle de z au cube résolu : le réseau guide donc aussi la recherche partie de la
solution. Dès qu'un état avant s ∘ U coïncide avec un état arrière V, la solution est U suivie de l'inverse
de V.
Vues multiples : le même réseau cherche depuis plusieurs états équivalents à s, dont les solutions se ramènent à
une solution de s de même longueur : l'inverse s⁻¹ et les conjugués g⁻¹ ∘ s ∘ g par les 48 symétries. Un réseau
qui n'est pas équivariant s'y trompe différemment, d'où une diversité sans entraînement supplémentaire. Soit un
seul faisceau à plusieurs racines (solve, option views), soit un faisceau par vue dont on garde la solution la
plus courte (solve_portfolio).
"""
import bisect
import time

import torch


class NetworkHeuristic:
    """Classement des enfants par un réseau network_torch.ResMLP (conversion de symétrie comprise)."""

    def __init__(self, model, cube, mode=None, chunk=1 << 16, lam=2.0):
        self.model, self.cube, self.chunk, self.lam = model, cube, chunk, lam
        config = getattr(model, "_orig_mod", model).config
        heads = config["heads"]
        self.mode = mode or ("policy" if "policy" in heads else "q" if "q" in heads else "value")
        missing = [h for h in (("policy", "value") if self.mode == "ac" else (self.mode,)) if h not in heads]
        if missing:
            raise ValueError(f"le réseau n'a pas de tête {', '.join(missing)} (têtes : {heads})")
        self.canonical = config["canonical"]
        self.encode = cube.encode if config["encoding"] == "pieces" else cube.encode_stickers
        self.amp = cube.device.type == "cuda"

    def evaluations(self, n_states):
        """Nombre d'évaluations du réseau pour développer n_states états du faisceau."""
        return 12 * n_states if self.mode == "value" else n_states

    @torch.no_grad()
    def head_outputs(self, states, heads):
        """Sorties des têtes demandées (dict), en une passe du réseau par paquet, dans le repère d'origine."""
        results = {h: [] for h in heads}
        for start in range(0, len(states), self.chunk):
            s = states[start:start + self.chunk]
            if self.canonical:
                s, k = self.cube.canonicalize(s)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                out = self.model(self.encode(s))
            for h in heads:
                o = out[h].float()
                if self.canonical and h != "value":
                    o = o.gather(1, self.cube.action_perm[k])   # repère canonique -> repère d'origine
                results[h].append(o)
        return {h: torch.cat(r) for h, r in results.items()}

    def outputs(self, states, head):
        return self.head_outputs(states, (head,))[head]

    def __call__(self, states, kids, cum):
        """Retourne (coût de chaque enfant (N, 12), coût cumulé transmis aux enfants, évaluations)."""
        n = len(states)
        if self.mode == "policy":
            child_cum = cum[:, None] - torch.log_softmax(self.outputs(states, "policy"), dim=1)
            return child_cum, child_cum, n
        if self.mode == "ac":
            out = self.head_outputs(states, ("policy", "value"))
            cost = out["value"][:, None] - self.lam * torch.log_softmax(out["policy"], dim=1)
            return cost, torch.zeros_like(cost), n
        if self.mode == "q":
            q = self.outputs(states, "q")
            return q, torch.zeros_like(q), n
        h = self.outputs(kids.reshape(-1, 54), "value").reshape(n, 12)
        return h, torch.zeros_like(h), 12 * n


def equivalent_views(cube, start, names):
    """
    États équivalents à `start` et, pour chacun, la fonction qui ramène une de ses solutions à une solution de
    start, de même longueur. Noms : id, inv (inverse s⁻¹), c<j> (conjugué par la symétrie j de 0 à 47),
    ic<j> (conjugué de l'inverse).
    """
    inverse_action = cube.inverse_action.tolist()
    perm_inv = torch.argsort(cube.action_perm, dim=1).tolist()   # coup joué sur le conjugué j -> coup sur s
    inverse = cube.inverse(start[None])[0]

    def invert(path):
        return [inverse_action[a] for a in reversed(path)]

    def relabel(path, j):
        return [perm_inv[j][b] for b in path]

    views = []
    for name in names:
        if name == "id":
            views.append((start, list))
        elif name == "inv":
            views.append((inverse, invert))
        elif name.startswith("ic"):
            j = int(name[2:])
            views.append((cube.conjugates(inverse[None])[0, j], lambda path, j=j: invert(relabel(path, j))))
        elif name.startswith("c"):
            j = int(name[1:])
            views.append((cube.conjugates(start[None])[0, j], lambda path, j=j: relabel(path, j)))
        else:
            raise ValueError(f"vue inconnue : {name}")
    return views


class _Beam:
    """
    Un faisceau : états courants, coût cumulé, et pour chaque profondeur le parent et le coup de chaque état
    retenu (pour reconstruire les chemins). Les empreintes de tous les états visités sont gardées triées,
    avec un identifiant global (rang de visite) pour retrouver leur chemin. Il peut partir de plusieurs racines
    (vues multiples) : un état atteint depuis deux racines est le même problème, il n'est gardé qu'une fois.
    """

    def __init__(self, cube, roots, key_roots):
        self.cube = cube
        self.states = roots
        self.n_roots = len(roots)
        self.cum = torch.zeros(self.n_roots, device=cube.device)
        self.parents, self.moves = [], []
        self.offsets = [0]          # identifiant global du premier état de chaque profondeur
        self.size = self.n_roots
        keys = cube.state_hash(key_roots)
        order = torch.argsort(keys)
        self.seen_keys = keys[order]
        self.seen_ids = torch.arange(self.n_roots, device=cube.device)[order]

    def lookup(self, keys):
        """(masque des empreintes déjà visitées, identifiant global de l'état correspondant)."""
        idx = torch.searchsorted(self.seen_keys, keys).clamp(max=len(self.seen_keys) - 1)
        return self.seen_keys[idx] == keys, self.seen_ids[idx]

    def path(self, gid):
        """(coups depuis la racine, indice de la racine) de l'état d'identifiant global gid."""
        depth = bisect.bisect_right(self.offsets, gid) - 1
        if depth == 0:
            return [], gid
        # une seule copie vers le CPU ; avec k racines, l'état d'identifiant g >= k a son parent et son coup en
        # position g - k
        parents, moves = torch.stack([torch.cat(self.parents[:depth]), torch.cat(self.moves[:depth])]).to(
            torch.int32).cpu().numpy()
        k = self.n_roots
        path = []
        while depth > 0:
            path.append(int(moves[gid - k]))
            gid = self.offsets[depth - 1] + int(parents[gid - k])
            depth -= 1
        return path[::-1], gid

    def child_path(self, j):
        """(chemin, racine) de l'enfant j (indice dans le dernier développement : parent j // 12, coup j % 12)."""
        moves, root = self.path(self.offsets[-1] + j // 12)
        return moves + [j % 12], root

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
def solve(heuristic, start, width=512, max_depth=40, bidirectional=False, max_evaluations=None, max_checks=16,
          views=None):
    """
    Cherche une solution pour l'état `start` (54,). Retourne un dict : solution (indices d'actions,
    vérifiée, ou None), evaluations (évaluations du réseau), depth (profondeur atteinte), time_s.
    views : noms de vues (voir equivalent_views), racines d'un seul faisceau ; pas avec bidirectional.
    """
    if views and bidirectional:
        raise ValueError("les vues multiples et la recherche bidirectionnelle ne se combinent pas")
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
    if views:
        candidates = equivalent_views(cube, start, views)
        keys = cube.state_hash(torch.stack([state for state, _ in candidates])).tolist()
        unique = [i for i, key in enumerate(keys) if key not in keys[:i]]   # une vue peut coïncider avec une autre
        roots, backs = torch.stack([candidates[i][0] for i in unique]), [candidates[i][1] for i in unique]
    else:
        roots, backs = start[None], [list]
    forward = _Beam(cube, roots, roots)
    backward_key = lambda z: cube.compose(start.expand_as(z), z)   # b = s ∘ z
    if bidirectional:
        start_inverse = cube.inverse(start[None])
        backward = _Beam(cube, start_inverse, backward_key(start_inverse))
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
                    moves, root = forward.child_path(j)
                    solution = backs[root](moves)
                    if verified(solution):
                        return done(solution)
        if any_found:
            for j in found.nonzero().flatten()[:max_checks].tolist():
                solution = forward.child_path(j)[0] + inverse_path(backward.path(int(ids[j]))[0])
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
                    solution = forward.path(int(ids[j]))[0] + inverse_path(backward.child_path(j)[0])
                    if verified(solution):
                        return done(solution)
            if n_best == 0:
                break
            backward.select(kids, keys, cum, best[:n_best])
    result["time_s"] = time.perf_counter() - t0
    return result


def solve_portfolio(heuristic, start, views, width=512, **options):
    """
    Un faisceau par vue (chacun de largeur `width`, options de solve comprises) ; garde la solution la plus
    courte, ramenée à `start`. Les évaluations et le temps sont la somme de ceux des faisceaux.
    """
    cube = heuristic.cube
    start = start.to(cube.device)
    best = {"solution": None, "evaluations": 0, "depth": 0, "time_s": 0.0}
    for state, back in equivalent_views(cube, start, views):
        result = solve(heuristic, state, width=width, **options)
        best["evaluations"] += result["evaluations"]
        best["time_s"] += result["time_s"]
        best["depth"] = max(best["depth"], result["depth"])
        if result["solution"] is None:
            continue
        solution = back(result["solution"])
        shorter = best["solution"] is None or len(solution) < len(best["solution"])
        if shorter and bool(cube.is_solved(cube.apply_sequence(start, solution))):
            best["solution"] = solution
    return best
