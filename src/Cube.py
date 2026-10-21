import pycuber as pc
import random
import numpy as np


# Emplacements des coins : la première face est toujours U ou D (face de référence pour l'orientation)
CORNER_SLOTS = [
    ('U', 'R', 'F'), ('U', 'F', 'L'), ('U', 'L', 'B'), ('U', 'B', 'R'),
    ('D', 'F', 'R'), ('D', 'L', 'F'), ('D', 'B', 'L'), ('D', 'R', 'B'),
]

# Emplacements des arêtes : la première face est U/D, ou F/B pour la couronne du milieu
EDGE_SLOTS = [
    ('U', 'R'), ('U', 'F'), ('U', 'L'), ('U', 'B'),
    ('D', 'R'), ('D', 'F'), ('D', 'L'), ('D', 'B'),
    ('F', 'R'), ('F', 'L'), ('B', 'L'), ('B', 'R'),
]


def _read_cubies(pycuber_cube):
    """Retourne {emplacement (frozenset des faces): {face: couleur}} pour les coins et arêtes."""
    return {
        frozenset(cubie.facings): {face: square.colour for face, square in cubie.facings.items()}
        for cubie in pycuber_cube.children if len(cubie.facings) > 1
    }


# Identité des pièces et couleur de référence, déduites du cube résolu de pycuber
_SOLVED_CUBIES = _read_cubies(pc.Cube())
CORNER_PIECES = {frozenset(_SOLVED_CUBIES[frozenset(slot)].values()): i for i, slot in enumerate(CORNER_SLOTS)}
EDGE_PIECES = {frozenset(_SOLVED_CUBIES[frozenset(slot)].values()): i for i, slot in enumerate(EDGE_SLOTS)}
# Couleur qui se trouve sur la face de référence de l'emplacement d'origine de chaque pièce
CORNER_REF_COLOUR = [_SOLVED_CUBIES[frozenset(slot)][slot[0]] for slot in CORNER_SLOTS]
EDGE_REF_COLOUR = [_SOLVED_CUBIES[frozenset(slot)][slot[0]] for slot in EDGE_SLOTS]

SOLVED_KOCIEMBA_STRING = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9


def inverse_action(action):
    """Retourne le mouvement inverse ('R' -> 'R\'', 'R\'' -> 'R')."""
    return action[0] if action.endswith("'") else action + "'"


class RubiksCube:
    """
    Classe représentant un Rubik's Cube utilisant pycuber.
    Fournit une interface pour manipuler le cube et obtenir sa représentation.
    """
    _solved_str = str(pc.Cube())
    _CENTER_COLOR_MAP = None  # défini après la classe

    def __init__(self,cube=None):
        if cube is None:
            self.cube = pc.Cube()
        else:
            self.cube=cube
        # Dictionnaire des actions possibles
        self.actions = {
            'F': lambda: self.cube("F"),
            'F\'': lambda: self.cube("F'"),
            'B': lambda: self.cube("B"),
            'B\'': lambda: self.cube("B'"),
            'L': lambda: self.cube("L"),
            'L\'': lambda: self.cube("L'"),
            'R': lambda: self.cube("R"),
            'R\'': lambda: self.cube("R'"),
            'U': lambda: self.cube("U"),
            'U\'': lambda: self.cube("U'"),
            'D': lambda: self.cube("D"),
            'D\'': lambda: self.cube("D'"),
        }
        self.action_list = list(self.actions.keys())
        self.last_scramble_depth = None
        self.last_scramble_sequence = None

        # Mappage des couleurs des centres aux caractères Kociemba
        # IMPORTANT: Vérifiez que ces couleurs correspondent aux centres de VOTRE cube
        # Pycuber utilise des objets Color, str(color) donne le nom (ex: "white", "red")
        # Kociemba attend URFDLB.
        # Standard (mais à vérifier sur votre cube physique et comment pycuber les nomme):
        # U -> White, R -> Red, F -> Blue, D -> Yellow, L -> Orange, B -> Green
        # Nous allons obtenir les couleurs des centres directement du cube à l'état résolu
        # pour rendre cela plus robuste.

        # Le mappage est calculé une seule fois pour la classe (voir _CENTER_COLOR_MAP) :
        # le recréer à chaque instance doublait le coût de clone()
        self.center_color_map = RubiksCube._CENTER_COLOR_MAP


    def reset(self):
        """Réinitialise le cube à l'état résolu"""
        self.cube = pc.Cube()
        self.last_scramble_depth = None
        self.last_scramble_sequence = None
        return self.get_matrix_state()

    def rotate(self, action):
        """
        Applique une action au cube ('F', 'F\'', 'B', etc.)

        """
        if action in self.actions:
            self.actions[action]()
        else:
            raise ValueError(f"Action non reconnue: {action}")

        return self.get_matrix_state()

    def apply_actions(self, action_sequence):
        """Applique une séquence d'actions au cube"""
        for action in action_sequence:
            self.rotate(action)
        return self.get_matrix_state()

    def scramble(self, num_moves, p=False, sequence=None):
        """
        Mélange le cube avec un nombre spécifié de mouvements aléatoires ou une séquence donnée.
        """
        if sequence is None:
            generated_sequence = []
            for _ in range(num_moves):
                # Évite les mouvements qui s'annulent (R puis R') et les triplés (R R R = R').
                # Les doublés (R R = demi-tour R2) restent autorisés.
                available_actions = list(self.action_list)
                if generated_sequence:
                    available_actions.remove(inverse_action(generated_sequence[-1]))
                if len(generated_sequence) >= 2 and generated_sequence[-1] == generated_sequence[-2]:
                    available_actions.remove(generated_sequence[-1])
                action = random.choice(available_actions)
                generated_sequence.append(action)
            self.last_scramble_sequence = generated_sequence # Stocker la séquence générée
        else:
            self.last_scramble_sequence = list(sequence) # Stocker la séquence fournie
        self.last_scramble_depth = len(self.last_scramble_sequence)

        for action_to_apply in self.last_scramble_sequence:
            self.rotate(action_to_apply)

        if p:
            print("Scramble sequence:", self.last_scramble_sequence)
        return self.last_scramble_sequence

    def is_solved(self):
        """Vérifie si le cube est dans un état résolu"""
        return str(self.cube) == RubiksCube._solved_str

    def get_matrix_state(self):
        """
        Retourne une représentation one-hot 20x24 du cube, aplatie en 480 éléments.
        - 20 lignes: les 20 emplacements (8 coins puis 12 arêtes)
        - 24 colonnes: quelle pièce occupe l'emplacement et avec quelle orientation
          (coins: 8 pièces x 3 orientations, arêtes: 12 pièces x 2 orientations)

        L'orientation est la position, dans l'emplacement, de la couleur de référence de la pièce
        (couleur U/D pour les coins et les arêtes U/D, couleur F/B pour la couronne du milieu).
        Les couleurs sont lues pièce par pièce via pycuber, cette représentation est donc sans perte.
        """
        cubies = _read_cubies(self.cube)
        state_matrix = np.zeros((20, 24), dtype=np.float32)

        for i, slot in enumerate(CORNER_SLOTS):
            facing = cubies[frozenset(slot)]
            piece = CORNER_PIECES.get(frozenset(facing.values()))
            if piece is None:
                raise ValueError(f"Coin invalide dans l'emplacement {''.join(slot)}: {facing}")
            orientation = [facing[face] for face in slot].index(CORNER_REF_COLOUR[piece])
            state_matrix[i][3 * piece + orientation] = 1

        for i, slot in enumerate(EDGE_SLOTS):
            facing = cubies[frozenset(slot)]
            piece = EDGE_PIECES.get(frozenset(facing.values()))
            if piece is None:
                raise ValueError(f"Arête invalide dans l'emplacement {''.join(slot)}: {facing}")
            orientation = [facing[face] for face in slot].index(EDGE_REF_COLOUR[piece])
            state_matrix[8 + i][2 * piece + orientation] = 1

        return state_matrix.flatten()  # Retourne un vecteur de 20*24 = 480 éléments

    def clone(self):
        """Crée une copie du cube actuel"""
        cube_copy = self.cube.copy()

        # Créer un nouveau RubiksCube avec cette copie
        new_cube = RubiksCube(cube_copy)
        new_cube.last_scramble_depth = self.last_scramble_depth
        new_cube.last_scramble_sequence = self.last_scramble_sequence.copy() if self.last_scramble_sequence is not None else None

        return new_cube

    def get_face(self, face):
        """Retourne la face spécifiée du cube."""
        return self.cube.get_face(face)

    def display(self):
        """Affiche l'état actuel du cube."""
        print(self.cube)

    def get_action_space(self):
        return self.action_list

    def take_action(self, action):
        new_state = self.clone()
        new_state.rotate(action)
        return new_state

    def __hash__(self):
        return hash(str(self.cube))  # ou tu peux hasher la matrice si tu préfères

    def __eq__(self, other):
        return str(self.cube) == str(other.cube)


    def get_direct_children_if_not_solved(self):
        """
        Génère les enfants directs du cube actuel en appliquant toutes les actions possibles
        si le cube n'est pas dans un état résolu.
        """
        direct_children = []
        if not self.is_solved():
            for action in self.action_list:
                # Crée une copie du cube actuel
                cube_copy = self.clone()
                # Applique l'action au cube cloné (sans calculer l'encodage comme le ferait rotate())
                cube_copy.actions[action]()
                # Ajoute le cube modifié à la liste des enfants
                direct_children.append(cube_copy)

        return direct_children

    def get_kociemba_definition_string(self):
        """
        Retourne la chaîne de 54 caractères pour Kociemba.
        Ordre des faces: U, R, F, D, L, B.
        Pour chaque face, les 9 facettes sont lues de gauche à droite, de haut en bas.
        """
        kociemba_str_list = []
        # Ordre des faces tel que Kociemba les attend
        kociemba_face_order = ['U', 'R', 'F', 'D', 'L', 'B']

        for face_char_kociemba_meaning in kociemba_face_order: # 'U', 'R', 'F', ...
            face_data = self.cube.get_face(face_char_kociemba_meaning) # Obtient les données de la face actuelle du cube
            # face_data est une liste de 3 listes (lignes), chaque sous-liste a 3 Square
            for row in face_data:
                for square in row:
                    # str(square.colour) donne le nom de la couleur (ex: "white", "red")
                    sticker_color_name = str(square.colour)
                    if sticker_color_name not in self.center_color_map:
                        # Cela ne devrait pas arriver si toutes les couleurs du cube sont standards
                        # et présentes dans les centres.
                        raise ValueError(
                            f"Couleur de facette inconnue '{sticker_color_name}' non trouvée dans le mappage des centres. "
                            f"Mappage actuel: {self.center_color_map}. "
                            "Vérifiez les couleurs de votre cube et le mappage dans __init__."
                        )
                    kociemba_str_list.append(self.center_color_map[sticker_color_name])

        return "".join(kociemba_str_list)

    @classmethod
    def from_kociemba_string(cls, definition_string):
        """
        Construit un RubiksCube à partir d'une chaîne Kociemba de 54 caractères (faces U, R, F, D, L, B).
        Le cube est reconstruit en appliquant à un cube résolu l'inverse de la solution de Kociemba,
        ce qui valide aussi l'état scanné. Lève ValueError si la chaîne ne décrit pas un cube valide.
        Le résultat est indépendant des couleurs du cube physique (seules les faces comptent).
        """
        import kociemba

        definition_string = definition_string.strip().upper()
        if len(definition_string) != 54 or definition_string[4::9] != "URFDLB":
            raise ValueError(f"Chaîne Kociemba invalide (54 caractères, centres URFDLB attendus): {definition_string}")

        cube = cls()
        if definition_string == SOLVED_KOCIEMBA_STRING:
            return cube

        # kociemba.solve lève ValueError si l'état est impossible
        solution = kociemba.solve(definition_string).split()
        quarter_turns = []
        for move in solution:
            if move.endswith("2"):
                quarter_turns += [move[0], move[0]]
            else:
                quarter_turns.append(move)
        cube.apply_actions([inverse_action(move) for move in reversed(quarter_turns)])

        if cube.get_kociemba_definition_string() != definition_string:
            raise ValueError("La reconstruction du cube ne correspond pas à la chaîne Kociemba fournie.")
        return cube


# Obtenir les couleurs des centres à partir d'un cube résolu pour le mappage vers les lettres Kociemba
_solved_cube_for_centers = pc.Cube()
RubiksCube._CENTER_COLOR_MAP = {
    str(_solved_cube_for_centers.get_face(face)[1][1].colour): face  # Centre de chaque face
    for face in ['U', 'R', 'F', 'D', 'L', 'B']
}


if __name__ == "__main__":
    cube = RubiksCube()
    cube.display()
