"""
Version PyTorch de fast_cube, pour générer les données et chercher sur GPU.

Mêmes conventions et mêmes tables que fast_cube (elles en sont copiées), ce que vérifient les tests : un état
est la permutation des 54 facettes dans l'ordre Kociemba, stockée en uint8 ; un lot d'états est un tenseur
(N, 54). Toutes les opérations restent sur le device des tenseurs, sans aller-retour avec le CPU.
"""
import numpy as np
import torch

import fast_cube as fc

ACTIONS = fc.ACTIONS
_SIGN_BIT = -(2 ** 63)


class TorchCube:
    """Tables du cube sur un device (CPU ou GPU) et opérations par lots."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        def t(array, dtype=torch.long):
            return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

        self.moves = t(fc.MOVES)                                  # (12, 54)
        self.inverse_action = t(fc.INVERSE_ACTION)                # (12,)
        self.solved = t(fc.SOLVED, torch.uint8)                   # (54,)
        self.facelet_face = t(fc.FACELET_FACE)                    # (54,)
        self.corner_idx = t(fc.CORNER_IDX)                        # (8, 3)
        self.edge_idx = t(fc.EDGE_IDX)                            # (12, 2)
        self.corner_lut = t(fc.CORNER_LUT)
        self.edge_lut = t(fc.EDGE_LUT)
        # mêmes poids que fast_cube : l'arithmétique int64 modulo 2^64 donne la même empreinte (vue en int64)
        self.hash_weights = t(fc._HASH_WEIGHTS.view(np.int64))
        self.symmetries = t(fc.SYMMETRIES)                        # (48, 54)
        self.symmetries_inv = t(fc.SYMMETRIES_INV)
        self.action_perm = t(fc.ACTION_PERM)                      # (48, 12)
        self.solved_hash = self.state_hash(self.solved[None])[0]

    def to_numpy(self, states):
        return states.cpu().numpy().astype(np.int16)

    def from_numpy(self, states):
        return torch.as_tensor(np.asarray(states), dtype=torch.uint8, device=self.device)

    def apply_actions(self, states, actions):
        """Applique une action par état : states (N, 54), actions (N,) -> (N, 54)."""
        return states.gather(1, self.moves[actions])

    def apply_sequence(self, state, actions):
        for a in actions:
            state = state[self.moves[a]]
        return state

    def children(self, states):
        """Les 12 enfants de chaque état : (N, 54) -> (N, 12, 54)."""
        return states[:, self.moves]

    def inverse(self, states):
        """Inverse de chaque état dans le groupe du cube : inv[s[i]] = i."""
        index = torch.arange(54, device=states.device, dtype=torch.uint8).expand_as(states).contiguous()
        return torch.empty_like(states).scatter_(-1, states.long(), index)

    def compose(self, a, b):
        """État obtenu en appliquant à a les mouvements qui mènent du cube résolu à b."""
        return a.gather(-1, b.long())

    def is_solved(self, states):
        return (states == self.solved).all(dim=-1)

    def state_hash(self, states):
        """Empreinte 64 bits, égale à fast_cube.state_hash vue en int64."""
        return (states.long() * self.hash_weights).sum(dim=-1)

    def encode(self, states, dtype=torch.float32):
        """Encodage one-hot 480 (pièce et orientation de chaque emplacement), identique à fast_cube.encode."""
        colours = self.facelet_face[states.long()]                        # (N, 54) faces d'origine 0..5
        c = colours[:, self.corner_idx]                                    # (N, 8, 3)
        corner_cols = self.corner_lut[c[..., 0] * 36 + c[..., 1] * 6 + c[..., 2]]
        e = colours[:, self.edge_idx]                                      # (N, 12, 2)
        edge_cols = self.edge_lut[e[..., 0] * 6 + e[..., 1]]
        cols = torch.cat([corner_cols, edge_cols], dim=1)                  # (N, 20)
        out = torch.zeros(len(states), 20, 24, dtype=dtype, device=states.device)
        out.scatter_(2, cols.unsqueeze(-1), 1.0)
        return out.reshape(len(states), 480)

    def encode_stickers(self, states, dtype=torch.float32):
        """Encodage one-hot 54 x 6 de la couleur de chaque facette (celui de DeepCubeA, Takano, CayleyPy)."""
        colours = self.facelet_face[states.long()]
        return torch.nn.functional.one_hot(colours, 6).to(dtype).reshape(len(states), 324)

    def conjugates(self, states):
        """Les 48 états symétriques g⁻¹ ∘ s ∘ g de chaque état : (N, 54) -> (N, 48, 54)."""
        moved = states[:, self.symmetries]                                 # s[g[i]]
        sym_inv = self.symmetries_inv.to(states.dtype).unsqueeze(0).expand(len(states), -1, -1)
        return sym_inv.gather(2, moved.long())

    def canonicalize(self, states, chunk=1 << 16):
        """
        Forme canonique (le symétrique de plus petite empreinte, comme fast_cube.canonicalize) et indice de
        symétrie k ; l'action a sur l'état d'origine correspond à action_perm[k, a] sur l'état canonique.
        """
        canonical = torch.empty_like(states)
        sym_index = torch.empty(len(states), dtype=torch.long, device=states.device)
        for start in range(0, len(states), chunk):
            conj = self.conjugates(states[start:start + chunk])
            # le bit de signe inversé donne en int64 l'ordre des empreintes non signées de fast_cube
            k = (self.state_hash(conj) ^ _SIGN_BIT).argmin(dim=1)
            canonical[start:start + chunk] = conj[torch.arange(len(k), device=states.device), k]
            sym_index[start:start + chunk] = k
        return canonical, sym_index

    def random_walks(self, n, length, generator=None):
        """
        n mélanges de `length` coups, règle de Cube.scramble (pas l'inverse du coup précédent, pas trois fois
        de suite le même coup). Retourne les états après chaque coup (length, n, 54) et les coups (length, n).
        """
        states = self.solved.expand(n, 54)
        last = torch.full((n,), -1, dtype=torch.long, device=self.device)
        before_last = last.clone()
        all_states, all_actions = [], []
        for _ in range(length):
            actions = torch.randint(0, 12, (n,), device=self.device, generator=generator)
            while True:
                bad = (last >= 0) & ((actions == self.inverse_action[last.clamp(min=0)]) |
                                     ((actions == last) & (last == before_last)))
                n_bad = int(bad.sum())
                if n_bad == 0:
                    break
                actions[bad] = torch.randint(0, 12, (n_bad,), device=self.device, generator=generator)
            states = self.apply_actions(states, actions)
            before_last, last = last, actions
            all_states.append(states)
            all_actions.append(actions)
        return torch.stack(all_states), torch.stack(all_actions)

    def self_supervised_batch(self, n_walks, length, generator=None):
        """
        Exemples auto-supervisés : chaque préfixe de chaque mélange donne un état, avec pour étiquette
        l'inverse du dernier coup et pour profondeur la longueur du préfixe (borne supérieure de la vraie
        distance). Retourne (états (n·length, 54), étiquettes, profondeurs).
        """
        states, actions = self.random_walks(n_walks, length, generator)
        labels = self.inverse_action[actions]
        depths = torch.arange(1, length + 1, device=self.device).unsqueeze(1).expand(length, n_walks)
        return states.reshape(-1, 54), labels.reshape(-1), depths.reshape(-1)
