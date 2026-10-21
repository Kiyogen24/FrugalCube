"""
Jeu de test de référence de DeepCubeA (Agostinelli et al., 2019) : 1 000 cubes, utilisé aussi par Takano (2023)
et CayleyPy (2025). On utilise la version JSON publiée avec EfficientCube : chaque cas est un mélange obtenu en
inversant la solution optimale publiée par DeepCubeA, donc sa longueur est la longueur optimale du cas
(moyenne 20,64 coups en quart de tour).
Le fichier est téléchargé dans data/ au premier usage (dossier non versionné).
"""
import json
import os
import urllib.request

import fast_cube as fc
from paths import PROJECT_ROOT

URL = "https://raw.githubusercontent.com/kyo-takano/efficientcube/main/data/deepcubea-dataset--cube3.json"
PATH = os.path.join(PROJECT_ROOT, "data", "deepcubea-dataset--cube3.json")


def load(n=None):
    """Retourne une liste de (état fast_cube, longueur optimale) pour les n premiers cas (tous par défaut)."""
    if not os.path.exists(PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        urllib.request.urlretrieve(URL, PATH)
    with open(PATH) as f:
        scrambles = json.load(f)
    cases = []
    for scramble in scrambles[:n]:
        moves = [fc.ACTIONS.index(m) for m in scramble.split()]
        cases.append((fc.apply_sequence(fc.SOLVED, moves), len(moves)))
    return cases
