"""
Laboratoire de recherche de la session GPU 5: le faisceau acteur-critique + vues de Kociemba de beam_torch (m90),
avec en option, par configuration (clé=valeur séparées par des virgules):
- lamb, tau: règle de bande de band.py (poids de la politique lamb tant que la valeur médiane du niveau >= tau);
- inner, tauw: hors de la bande (valeur médiane < tauw), largeur réduite à inner x largeur: le budget suit la
  cécité du réseau;
- ball: table exacte de tous les états à <= ball coups du cube résolu (recherche périmétrique, Dillenburg et
  Nelson 1994). Dès qu'un enfant du faisceau y tombe, la fin du chemin est exacte et la recherche s'arrête: un
  état qui entre dans la table depuis l'extérieur y est à distance ball, donc aucun niveau suivant ne fait mieux
  que le meilleur enfant du premier niveau touché.
Une ligne par cube dans <out-dir>/<config>_<part>_w<largeur>.jsonl, appariée avec la configuration de base.
Usage: python experiments/cube/search.py --device cuda --split all --width 206848 base ball=8 lamb=2,tau=15,ball=8
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
import beam_torch as bt
import cube_torch as ct
import deepcubea_testset as dt
import fast_cube as fc
import network_torch as nt

QTM_SPHERES = [1, 12, 114, 1068, 10011, 93840, 878880, 8221632, 76843595, 717789576]


class Ball:
    """les états à <= depth coups du cube résolu: empreintes triées et distance exacte."""

    def __init__(self, cube, depth, cache="data"):
        self.cube = cube
        path = f"{cache}/ball{depth}.pt"
        if os.path.exists(path):
            data = torch.load(path)
            self.keys, self.dist = data["keys"].to(cube.device), data["dist"].to(cube.device)
        else:
            self.keys, self.dist = self.build(depth)
            os.makedirs(cache, exist_ok=True)
            torch.save({"keys": self.keys.cpu(), "dist": self.dist.cpu()}, path)

    def build(self, depth, chunk=1 << 19):
        cube, t0 = self.cube, time.time()
        frontier = cube.solved[None]
        levels = [cube.state_hash(frontier)]
        previous = torch.empty(0, dtype=torch.long, device=cube.device)
        for d in range(1, depth + 1):
            keys, states = [], []
            for s in range(0, len(frontier), chunk):
                kids = cube.children(frontier[s:s + chunk]).reshape(-1, 54)
                k = cube.state_hash(kids)
                k, first = self.first_of_each(k)
                keys.append(k)
                if d < depth:
                    states.append(kids[first])
            k = torch.cat(keys)
            del keys
            k, first = self.first_of_each(k)
            # graphe biparti (chaque quart de tour change la parité des coins): les voisins du niveau d - 1 sont
            # aux niveaux d - 2 et d
            if len(previous):
                pos = torch.searchsorted(previous, k).clamp(max=len(previous) - 1)
                new = previous[pos] != k
                k, first = k[new], first[new]
            frontier = torch.cat(states)[first] if d < depth else None
            del states
            if cube.device.type == "cuda":
                torch.cuda.empty_cache()
            previous = levels[-1].sort().values
            levels.append(k)
            expected = QTM_SPHERES[d] if d < len(QTM_SPHERES) else None
            print(f"table : distance {d}, {len(k)} états (attendu {expected}), {time.time() - t0:.0f} s", flush=True)
            assert expected is None or len(k) == expected, "table fausse"
        keys = torch.cat(levels)
        dist = torch.cat([torch.full((len(k),), d, dtype=torch.uint8, device=keys.device) for d, k in enumerate(levels)])
        order = torch.argsort(keys)
        return keys[order], dist[order]

    @staticmethod
    def first_of_each(keys):
        """empreintes distinctes, triées, et l'indice d'une occurrence de chacune."""
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
        """Un plus court chemin exact vers le cube résolu, depuis un état de la table."""
        path = []
        _, r = self.lookup(self.cube.state_hash(state[None]))
        r = int(r)
        while r > 0:
            kids = self.cube.children(state[None])[0]
            _, rk = self.lookup(self.cube.state_hash(kids))
            a = int(torch.nonzero(rk == r - 1)[0])
            path.append(a)
            state, r = kids[a], r - 1
        return path


class LabHeuristic(bt.NetworkHeuristic):
    def __init__(self, net, cube, width, lamb=1.0, tau=99.0, inner=1.0, tauw=0.0, tmin=0.0, mu=0.0, childv=0.0,
                 chunk=1 << 17):
        super().__init__(net, cube, "ac", chunk, 1.0)
        self.width, self.lamb, self.tau, self.inner, self.tauw, self.tmin = width, lamb, tau, inner, tauw, tmin
        self.mu = mu
        # childv: chaque enfant est jugé sur sa propre valeur V(s.a) (12 appels de plus par parent) au lieu de la
        # valeur partagée V(s); plafond de ce qu'une tête Q(s, a) distillée pourrait apporter en un seul appel
        self.childv = childv
        self.trace = []   # (valeur min, valeur médiane) des parents à chaque niveau, sur le device

    def __call__(self, states, kids, cum):
        n = len(states)
        out = self.head_outputs(states, ("policy", "value"))
        v, logp = out["value"], torch.log_softmax(out["policy"], dim=1)
        self.last_v = v
        median = v.median()
        self.trace.append(torch.stack([v.min(), median]))
        lam = torch.where(median >= self.tau, self.lamb, self.lam)
        evaluations = n
        if self.childv:
            vk = self.head_outputs(kids.reshape(-1, 54), ("value",))["value"].reshape(n, 12)
            cost = vk - lam * logp
            evaluations = 13 * n
        else:
            cost = v[:, None] - lam * logp
        # -log pi cumulé le long du chemin (le score de Takano): dans la bande, l'avantage de la politique est faible
        # à chaque coup mais constant, et s'accumule
        nll = cum[:, None] - logp
        if self.mu:
            cost = cost + self.mu * nll
        k = max(1, int(self.inner * self.width))
        if self.inner < 1 and 12 * n > k:
            # là où le réseau est fiable (valeur médiane < tauw, ou meilleur parent à moins de tmin coups), seuls les
            # k meilleurs enfants restent candidats
            threshold = cost.flatten().kthvalue(k).values
            narrow = (median < self.tauw) | (v.min() < self.tmin)
            cost = torch.where(narrow & (cost > threshold), float("inf"), cost)
        return cost, nll, evaluations


class Memory:
    """
    ce qu'un essai de recherche a appris sur ce cube (programmation dynamique en temps réel, Barto, Bradtke et Singh
    1995): pour chaque état retenu par le faisceau, la correction delta = b - V, où b est sa valeur sauvegardée sur
    l'arbre du faisceau (1 + min des enfants retenus; V aux feuilles; 1 au parent de la solution). Un essai suivant
    ajoute delta au coût de ces états: un sous-arbre décevant est repoussé, un sous-arbre prometteur rapproché, et
    les états jamais vus (delta = 0) prennent la place des premiers.
    """

    def __init__(self, levels, values, hit_parent=None, dead=False):
        # levels[k] = (empreintes, indice du parent au niveau k - 1) des états retenus au niveau k;
        # values[k] = leur valeur V, calculée quand ils ont été développés. dead: un état dont aucun enfant n'a été
        # retenu (branche abandonnée par le faisceau) vaut au moins 1 + la valeur médiane du niveau suivant, au lieu
        # de garder sa valeur optimiste
        K = len(values) - 1
        b = values[K].clone()
        if hit_parent is not None:
            b[hit_parent] = 1.0
        keys, deltas = [], []
        for k in range(K, 0, -1):
            keys.append(levels[k][0])
            deltas.append(b - values[k])
            child_min = torch.full((len(values[k - 1]),), float("inf"), device=b.device)
            child_min.scatter_reduce_(0, levels[k][1].long(), b, "amin")   # parents en int32 dans beam_torch
            leaf = torch.maximum(values[k - 1], 1 + values[k].median()) if dead else values[k - 1]
            b = torch.where(torch.isfinite(child_min), 1 + child_min, leaf)
        keys, deltas = torch.cat(keys), torch.cat(deltas)
        order = torch.argsort(keys)
        self.keys, self.deltas = keys[order], deltas[order]

    def merge(self, other):
        """les corrections des deux essais; celles de l'essai le plus récent (other) l'emportent."""
        keys, deltas = torch.cat([other.keys, self.keys]), torch.cat([other.deltas, self.deltas])
        order = torch.argsort(keys, stable=True)
        keys, deltas = keys[order], deltas[order]
        first = torch.ones_like(keys, dtype=torch.bool)
        first[1:] = keys[1:] != keys[:-1]
        self.keys, self.deltas = keys[first], deltas[first]
        return self

    def correction(self, keys):
        idx = torch.searchsorted(self.keys, keys).clamp(max=len(self.keys) - 1)
        return torch.where(self.keys[idx] == keys, self.deltas[idx], 0.0)


def force_keep(keys, cost, best, n_best, target):
    """
    oracle: l'enfant d'empreinte target (l'état du chemin optimal connu à cette profondeur) est gardé dans le
    faisceau, à la place du dernier retenu s'il n'y est pas déjà. Rend (best, n_best, inséré ou non).
    """
    n = int(n_best)
    js = torch.nonzero((keys == target) & torch.isfinite(cost)).flatten()
    if len(js) == 0 or bool((keys[best[:n]] == target).any()):
        return best, n_best, False
    best = best.clone()
    j = js[cost[js].argmin()]
    if n < len(best):
        best[n] = j
        n += 1
    else:
        best[n - 1] = j
    return best, torch.tensor(n, device=keys.device), True


@torch.no_grad()
def solve(h, start, width, views, ball=None, max_depth=40, max_checks=16, memory=None, beta=1.0, record=False,
          dead=False, wide=1.0, vwide=99.0, force=None):
    """
    bt.solve en avant seulement (vues multiples), avec la table de fin de partie en option. memory: corrections
    d'un essai précédent (Memory), ajoutées au coût des enfants avec le poids beta. record: rend dans
    result["memory"] ce que cet essai a appris (sans table). wide: largeur multipliée par wide tant que la valeur
    médiane des parents est au moins vwide (le chemin optimal sort du faisceau tôt, à 15-19 coups).
    force: (T, empreintes du chemin optimal depuis la racine id, profondeur 1, 2, ...); jusqu'à la profondeur T,
    l'état du chemin optimal est gardé de force (borne de ce que vaudrait un classement parfait sur ces niveaux).
    """
    cube = h.cube
    t0 = time.perf_counter()
    result = {"solution": None, "evaluations": 0, "depth": 0}

    def done(solution):
        if solution is not None and not bool(cube.is_solved(cube.apply_sequence(start, solution))):
            raise RuntimeError("solution fausse")
        result["solution"] = solution
        result["time_s"] = time.perf_counter() - t0
        if hasattr(h, "trace") and h.trace:
            result["trace"] = torch.stack(h.trace).tolist()
        return result

    if hasattr(h, "trace"):
        h.trace = []
    start = start.to(cube.device)
    if cube.is_solved(start):
        return done([])
    candidates = bt.equivalent_views(cube, start, views)
    keys = cube.state_hash(torch.stack([state for state, _ in candidates])).tolist()
    unique = [i for i, key in enumerate(keys) if key not in keys[:i]]
    roots, backs = torch.stack([candidates[i][0] for i in unique]), [candidates[i][1] for i in unique]
    if ball is not None:
        found, r = ball.lookup(cube.state_hash(roots))
        if bool(found.any()):
            j = int(r.argmin())
            result["hits"], result["root"] = int(found.sum()), j
            return done(backs[j](ball.descend(roots[j])))
    forward = bt._Beam(cube, roots, roots)
    levels, values = [(None, None)], []
    for depth in range(1, max_depth + 1):
        result["depth"] = depth
        kids, keys, cost, cum, evaluations = forward.expand(h, lambda x: x)
        result["evaluations"] += evaluations
        if record:
            values.append(h.last_v)
        if memory is not None:
            cost = cost + beta * memory.correction(keys)
        w = int(wide * width) if wide > 1 and float(h.trace[-1][1]) >= vwide else width
        best, n_best = forward.candidates(keys, cost, w)
        if force is not None and depth <= min(force[0], len(force[1])):
            best, n_best, inserted = force_keep(keys, cost, best, n_best, force[1][depth - 1])
            result["inserted"] = result.get("inserted", 0) + int(inserted)
        if ball is not None:
            _, r = ball.lookup(keys)
            r_min, j = r.min(0)
            n_best, r_min, j = bt._to_host(n_best, r_min, j)
            if r_min < 255:
                moves, root = forward.child_path(j)
                # nombre d'enfants distincts dans la table au premier niveau touché: indice de facilité du cube
                result["hits"], result["root"] = int(keys[r < 255].unique().numel()), root
                return done(backs[root](moves + ball.descend(kids[j])))
        else:
            hits = keys == cube.solved_hash
            n_best, any_hit = bt._to_host(n_best, hits.any())
            if any_hit:
                for j in hits.nonzero().flatten()[:max_checks].tolist():
                    if cube.is_solved(kids[j]):
                        moves, root = forward.child_path(j)
                        if record:
                            result["memory"] = Memory(levels, values, j // 12, dead)
                        return done(backs[root](moves))
        if n_best == 0:
            break
        forward.select(kids, keys, cum, best[:n_best])
        if record:
            levels.append((keys[best[:n_best]], forward.parents[-1]))
    if record and values:
        result["memory"] = Memory(levels, values, dead=dead)
    result["time_s"] = time.perf_counter() - t0
    return result


def retry(h, start, width, first, trials, beta, altviews, dead=False):
    """
    Essais suivants sur le même cube: avec la mémoire des précédents (RTDP), ou, en contrôle (altviews), une
    recherche neuve depuis d'autres vues, sans mémoire. Seule une solution plus courte d'au moins 2 coups (parité)
    améliore: chaque essai s'arrête à la profondeur L - 2. Temps et évaluations s'additionnent; la plus courte
    solution est gardée.
    """
    best, memory = first, first.pop("memory", None)
    seconds, evaluations = first["time_s"], first["evaluations"]
    for _ in range(trials - 1):
        if best["solution"] is None or len(best["solution"]) <= 2:
            break
        L = len(best["solution"])
        if altviews:
            r = solve(h, start, width, VIEWS2, max_depth=L - 2)
        else:
            r = solve(h, start, width, VIEWS, max_depth=L - 2, memory=memory, beta=beta, record=True, dead=dead)
            new = r.pop("memory", None)
            if new is not None:
                memory = new if memory is None else memory.merge(new)
        seconds, evaluations = seconds + r["time_s"], evaluations + r["evaluations"]
        if r["solution"] is not None and len(r["solution"]) < L:
            best = r
    best = dict(best, time_s=seconds, evaluations=evaluations, improved=best is not first)
    return best


def fresh_cubes(cube, n, seed):
    """n cubes indépendants: marches aléatoires de 200 coups (cubes pairs) ou 199 (impairs), une marche par cube."""
    walks, _ = cube.random_walks(n, 200, torch.Generator(cube.device).manual_seed(seed))
    return [walks[-1 - k % 2][k] for k in range(n)]


def parse(config):
    if config == "base":
        return {}
    options = dict(item.split("=") for item in config.split(","))
    return {k: v if k == "views" else float(v) for k, v in options.items()}


def summary(name, rows, reference):
    lengths = np.array([r["length"] for r in rows], dtype=float)
    optimal = np.array([r["optimal"] for r in rows])
    known = rows[0]["optimal"] is not None   # cubes neufs: longueur optimale inconnue
    rate = f"{np.mean(lengths == optimal):6.1%}" if known else "     —"
    line = (f"{name:28} {rate} optimal, {np.nanmean(lengths):.3f} coups, "
            f"{np.mean([r['time_s'] for r in rows]):.2f} s/cube, {np.mean([r["evaluations"] for r in rows]) / 1e6:.3f} M évaluations")
    if reference is not None and rows is not reference:
        ref = np.array([r["length"] for r in reference[:len(rows)]], dtype=float)
        d = lengths - ref
        z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std() > 0 else 0.0
        better = int(np.sum((lengths == optimal) & (ref != optimal)))
        worse = int(np.sum((ref == optimal) & (lengths != optimal)))
        line += f" ; contre base {d.mean():+.3f} coups (z {z:+.1f}), optimales {better}/{worse}"
    print(line, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="base, ou clé=valeur séparées par des virgules")
    parser.add_argument("--model", default="runs/s4/m90_v/final.pt")
    parser.add_argument("--split", choices=["odd", "even", "all"], default="all")
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="runs/gpu5/lab")
    parser.add_argument("--reference", help="jsonl de référence pour l'appariement (sinon la config base, calculée)")
    parser.add_argument("--es-log", default="runs/gpu5/es/es.jsonl", help="journal de es.py pour es=1 (dernier theta)")
    parser.add_argument("--fresh", type=int, default=0,
                        help="N cubes neufs (marches aléatoires de 200 ou 199 coups, les deux parités) au lieu du jeu "
                             "DeepCubeA : pour choisir une méthode sans user le jeu de test")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--no-base", action="store_true",
                        help="sans référence, ne pas ajouter la configuration base (4 racines) : la première sert de "
                             "référence")
    args = parser.parse_args()

    torch.set_num_threads(16)
    VIEWS = ["id", "inv", "c15", "c20"]
    VIEWS2 = ["c5", "c33", "ic15", "ic20"]   # contrôle des essais répétés: autres vues équivalentes
    SCRAMBLES = json.load(open(dt.PATH))
    cube = ct.TorchCube(args.device)
    net, _ = nt.load(args.model, args.device)
    keep = {"odd": lambda i: i % 2 == 1, "even": lambda i: i % 2 == 0, "all": lambda i: True}[args.split]
    if args.fresh:
        # --cases limite aussi les cubes neufs: les 50 premiers des 1000 (graine fixée) pour une vérification rapide
        CASES = [(s, None, i) for i, s in enumerate(fresh_cubes(cube, args.fresh, args.seed))][:args.cases]
        args.split = f"fresh{args.seed}"
    else:
        CASES = [(cube.from_numpy(s), o, i) for i, (s, o) in enumerate(dt.load()) if keep(i)][:args.cases]
    os.makedirs(args.out_dir, exist_ok=True)
    balls, reference = {}, None
    if args.reference:
        reference = [json.loads(line) for line in open(args.reference)]
        reference = {r["i"]: r for r in reference}
        reference = [reference[i] for _, _, i in CASES]
        summary("référence", reference, None)
    configs = args.configs if args.reference or args.no_base else ["base"] + [c for c in args.configs if c != "base"]
    for config in configs:
        options = parse(config)
        depth = int(options.pop("ball", 0))
        controller = options.pop("es", 0)
        trials, beta, altviews = int(options.pop("trials", 1)), options.pop("beta", 1.0), options.pop("altviews", 0)
        dead = bool(options.pop("dead", 0))
        wide, vwide = options.pop("wide", 1.0), options.pop("vwide", 99.0)
        force_depth = int(options.pop("force", 0))
        # nombre de racines: 1 (s), 2 (s, s⁻¹), 4 (défaut, celles de Kociemba), 6 (+ conjugués de l'inverse), 8
        views = {1: ["id"], 2: ["id", "inv"], 4: VIEWS, 6: VIEWS + ["ic15", "ic20"],
                 8: VIEWS + ["ic15", "ic20", "c5", "c33"]}[int(options.pop("roots", 4))]
        if "views" in options:      # liste explicite, par ex. views=id+c15
            views = options.pop("views").split("+")
        if force_depth and args.fresh:
            raise ValueError("force : chemins optimaux connus seulement sur le jeu de test")
        if depth and depth not in balls:
            if args.device == "cuda":
                torch.cuda.empty_cache()
            balls[depth] = Ball(cube, depth)
        name = config.replace("=", "").replace(",", "_")
        out = f"{args.out_dir}/{name}_{args.split}_w{args.width}.jsonl"
        rows = [json.loads(line) for line in open(out)] if os.path.exists(out) else []
        if controller:
            # contrôleur appris par es.py: dernier theta du journal
            import es
            theta = [json.loads(line) for line in open(args.es_log)][-1]["theta_next"]
            h = es.ControllerHeuristic(net, cube, theta, chunk=1 << (14 if args.device == "cpu" else 17))
        else:
            h = LabHeuristic(net, cube, args.width, chunk=1 << (14 if args.device == "cpu" else 17), **options)
        with open(out, "a") as f:
            for state, optimal, i in CASES[len(rows):]:
                force = None
                if force_depth:
                    # le mélange publié est l'inverse d'une solution optimale: ses préfixes, du plus long au plus
                    # court, sont les états du chemin optimal depuis le cube
                    st, prefixes = fc.SOLVED, []
                    for m in SCRAMBLES[i].split():
                        st = fc.apply_sequence(st, [fc.ACTIONS.index(m)])
                        prefixes.append(st)
                    path = cube.state_hash(cube.from_numpy(np.stack(prefixes[::-1])))
                    assert int(path[0]) == int(cube.state_hash(state.to(cube.device)[None])[0])
                    force = (force_depth, path[1:])
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=args.device == "cpu"):
                    r = solve(h, state, args.width, views, balls.get(depth), record=trials > 1 and not altviews,
                              dead=dead, wide=wide, vwide=vwide, force=force)
                    if trials > 1:
                        r = retry(h, state, args.width, r, trials, beta, altviews, dead)
                row = {"i": i, "length": len(r["solution"]) if r["solution"] is not None else None,
                       "optimal": optimal, "time_s": r["time_s"], "evaluations": r["evaluations"]}
                for key in ("depth", "hits", "root", "improved", "inserted"):
                    if key in r:
                        row[key] = r[key]
                if "trace" in r:
                    row["trace"] = [[round(x, 3) for x in level] for level in r["trace"]]
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
        reference = reference or rows
        summary(name, rows, reference)
