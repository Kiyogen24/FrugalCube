"""
Cube vectorisé avec numpy, pour la recherche par lots.

Un état est la permutation des 54 facettes, dans l'ordre Kociemba (faces U, R, F, D, L, B, lues ligne par
ligne comme get_face de pycuber) : state[i] est l'indice, dans le cube résolu, de la facette qui se trouve
maintenant en position i. Un lot d'états est un tableau (N, 54).

Les permutations des mouvements et la position des pièces sont déduites de pycuber en suivant chaque
facette : le module est cohérent par construction avec Cube.py (mêmes mouvements, même ordre d'actions,
même encodage 480 que RubiksCube.get_matrix_state), ce que vérifient les tests.

Contrairement à pycuber, cette représentation donne directement l'inverse d'un état (argsort), utilisé par
la recherche bidirectionnelle.
"""
import numpy as np
import pycuber as pc

from Cube import RubiksCube, CORNER_SLOTS, EDGE_SLOTS, inverse_action

FACES = "URFDLB"
ACTIONS = RubiksCube().action_list              # même ordre que la tête de politique du réseau
INVERSE_ACTION = np.array([ACTIONS.index(inverse_action(a)) for a in ACTIONS])
SOLVED = np.arange(54, dtype=np.int16)
FACELET_FACE = np.arange(54) // 9               # face d'origine (0..5 = URFDLB) de chaque facette


def _tagged_cube():
    """Cube pycuber résolu dont chaque facette porte son indice Kociemba."""
    cube = pc.Cube()
    for i, square in enumerate(sq for f in FACES for row in cube.get_face(f) for sq in row):
        square.tag = i
    return cube


def _read_tags(cube):
    return np.array([sq.tag for f in FACES for row in cube.get_face(f) for sq in row], dtype=np.int16)


def _build_tables():
    # Mouvements : après le mouvement, la facette en position i vient de la position moves[a, i]
    moves = []
    for action in ACTIONS:
        cube = _tagged_cube()
        cube(action)
        moves.append(_read_tags(cube))
    moves = np.stack(moves)

    # Indices des facettes de chaque emplacement de coin / d'arête, dans l'ordre des faces du tuple
    cube = _tagged_cube()
    slots = {frozenset(c.facings): {face: sq.tag for face, sq in c.facings.items()}
             for c in cube.children if len(c.facings) > 1}
    corner_idx = np.array([[slots[frozenset(s)][f] for f in s] for s in CORNER_SLOTS])
    edge_idx = np.array([[slots[frozenset(s)][f] for f in s] for s in EDGE_SLOTS])

    # Tables couleur -> colonne one-hot (même définition de pièce et d'orientation que Cube.py) :
    # la pièce k est celle de l'emplacement k dans le cube résolu, sa couleur de référence est celle
    # de la première face de son emplacement, et l'orientation est la position de cette couleur.
    face_id = {f: i for i, f in enumerate(FACES)}
    corner_lut = np.full(6 ** 3, -1, dtype=np.int64)
    for piece, slot in enumerate(CORNER_SLOTS):
        colours = [face_id[f] for f in slot]
        for perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)):
            c = [colours[p] for p in perm]
            corner_lut[c[0] * 36 + c[1] * 6 + c[2]] = 3 * piece + c.index(colours[0])
    edge_lut = np.full(6 ** 2, -1, dtype=np.int64)
    for piece, slot in enumerate(EDGE_SLOTS):
        a, b = face_id[slot[0]], face_id[slot[1]]
        edge_lut[a * 6 + b] = 2 * piece
        edge_lut[b * 6 + a] = 2 * piece + 1
    return moves, corner_idx, edge_idx, corner_lut, edge_lut


MOVES, CORNER_IDX, EDGE_IDX, CORNER_LUT, EDGE_LUT = _build_tables()
_HASH_WEIGHTS = np.random.default_rng(12345).integers(1, 2 ** 63, size=54, dtype=np.uint64)


def _build_symmetries():
    """
    Les 48 symétries du cube (24 rotations du cube entier, avec ou sans miroir) comme permutations des
    54 positions, et la correspondance des actions : action_perm[k, a] est l'action qui joue, dans l'état
    conjugué par la symétrie k, le rôle de l'action a dans l'état d'origine.
    """
    def rotation(name):
        cube = _tagged_cube()
        cube(name)
        return _read_tags(cube)

    # Miroir gauche-droite : R et L échangées, colonnes de chaque face inversées
    mirror = np.empty(54, dtype=np.int16)
    for f, face in enumerate(FACES):
        target = FACES.index({"R": "L", "L": "R"}.get(face, face))
        for r in range(3):
            for c in range(3):
                mirror[f * 9 + r * 3 + c] = target * 9 + r * 3 + (2 - c)

    generators = [rotation("x"), rotation("y"), mirror]
    group = {tuple(SOLVED.tolist()): SOLVED}
    frontier = [SOLVED]
    while frontier:
        new = []
        for g in frontier:
            for h in generators:
                k = g[h]
                if tuple(k.tolist()) not in group:
                    group[tuple(k.tolist())] = k
                    new.append(k)
        frontier = new
    syms = [SOLVED] + [g for key, g in sorted(group.items()) if key != tuple(SOLVED.tolist())]
    syms = np.stack(syms).astype(np.int16)
    syms_inv = np.argsort(syms, axis=1).astype(np.int16)

    move_index = {tuple(m.tolist()): a for a, m in enumerate(MOVES)}
    action_perm = np.empty((len(syms), 12), dtype=np.int64)
    for k, (g, g_inv) in enumerate(zip(syms, syms_inv)):
        for a in range(12):
            conjugated = g_inv[MOVES[a][g]]            # g⁻¹ ∘ M_a ∘ g
            action_perm[k, a] = move_index[tuple(conjugated.tolist())]  # KeyError si le miroir est faux
    return syms, syms_inv, action_perm


SYMMETRIES, SYMMETRIES_INV, ACTION_PERM = _build_symmetries()


def apply_actions(states, actions):
    """Applique une action par état : states (N, 54), actions (N,) -> (N, 54)."""
    return np.take_along_axis(states, MOVES[actions], axis=1)


def apply_sequence(state, actions):
    """Applique une séquence d'actions (indices) à un seul état (54,)."""
    for a in actions:
        state = state[MOVES[a]]
    return state


def children(states):
    """Les 12 enfants de chaque état : (N, 54) -> (N, 12, 54)."""
    return states[:, MOVES]


def inverse(states):
    """Inverse de chaque état (élément inverse dans le groupe du cube)."""
    return np.argsort(states, axis=-1).astype(states.dtype)


def compose(a, b):
    """État obtenu en appliquant à a les mouvements qui mènent du cube résolu à b."""
    return np.take_along_axis(a, b, axis=-1)


def is_solved(states):
    return (states == SOLVED).all(axis=-1)


def state_hash(states):
    """Empreinte 64 bits d'états (N, 54) ; les collisions, très improbables, sont écartées par vérification."""
    return (states.astype(np.uint64) * _HASH_WEIGHTS).sum(axis=-1)


def encode(states):
    """Encodage one-hot 480 identique à RubiksCube.get_matrix_state."""
    colours = FACELET_FACE[states]                                    # (N, 54) faces d'origine 0..5
    c = colours[:, CORNER_IDX]                                        # (N, 8, 3)
    corner_cols = CORNER_LUT[c[..., 0] * 36 + c[..., 1] * 6 + c[..., 2]]
    e = colours[:, EDGE_IDX]                                          # (N, 12, 2)
    edge_cols = EDGE_LUT[e[..., 0] * 6 + e[..., 1]]
    n = len(states)
    out = np.zeros((n, 20, 24), dtype=np.float32)
    rows = np.arange(n)[:, None]
    out[rows, np.arange(8), corner_cols] = 1.0
    out[rows, 8 + np.arange(12), edge_cols] = 1.0
    return out.reshape(n, 480)


def conjugates(states):
    """Les 48 états symétriques g⁻¹ ∘ s ∘ g de chaque état : (N, 54) -> (N, 48, 54)."""
    moved = states[:, SYMMETRIES]                                   # s[g[i]]
    return SYMMETRIES_INV[np.arange(len(SYMMETRIES))[None, :, None], moved]


def canonicalize(states, chunk=20000):
    """
    Forme canonique de chaque état : celui de ses 48 symétriques qui a la plus petite empreinte (identique
    pour tous les états d'une même classe de symétrie). Retourne (états canoniques, indice de symétrie k) ;
    une action a sur l'état d'origine correspond à l'action ACTION_PERM[k, a] sur l'état canonique.
    """
    canonical = np.empty_like(states)
    sym_index = np.empty(len(states), dtype=np.int64)
    for start in range(0, len(states), chunk):
        conj = conjugates(states[start:start + chunk])
        k = state_hash(conj).argmin(axis=1)
        canonical[start:start + chunk] = conj[np.arange(len(k)), k]
        sym_index[start:start + chunk] = k
    return canonical, sym_index


def kociemba_string(state):
    return "".join(FACES[f] for f in FACELET_FACE[state])


def scramble(rng, depth):
    """Mélange de `depth` coups sans deux mouvements consécutifs sur la même face (règle d'évaluation)."""
    sequence = []
    for _ in range(depth):
        choices = [a for a in range(12) if not sequence or ACTIONS[a][0] != ACTIONS[sequence[-1]][0]]
        sequence.append(int(rng.choice(choices)))
    return apply_sequence(SOLVED, sequence), sequence
