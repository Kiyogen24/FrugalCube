"""
Réseau PyTorch : MLP résiduel, l'architecture de DeepCubeA (reprise par Takano, et en plus petit par CayleyPy),
avec des têtes au choix :
- policy : 12 logits, le coup à jouer (appris sur l'inverse du dernier coup du mélange) ;
- value  : distance estimée au cube résolu, en coups ;
- q      : distance estimée après chacun des 12 coups (Q-learning, « Q* search » d'Agostinelli et al., 2024).
Pas de dropout ni de régularisation L2 : les données sont toujours neuves, il n'y a pas de surapprentissage.
"""
import torch
from torch import nn

HEAD_SIZES = {"policy": 12, "value": 1, "q": 12}
INPUT_SIZES = {"pieces": 480, "stickers": 324}


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(),
                                    nn.Linear(width, width), nn.BatchNorm1d(width))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class ResMLP(nn.Module):
    """
    encoding : "pieces" (one-hot 480 de cube_torch.encode) ou "stickers" (one-hot 324 de encode_stickers) ;
    canonical : le réseau attend des états ramenés à leur forme canonique (symétries), et ses sorties par
    action sont exprimées dans ce repère ;
    value_detach : la tête value apprend sur les caractéristiques du tronc sans les modifier (pas de gradient
    vers le tronc), pour qu'une tête value ajoutée ne dégrade pas la politique.
    Ces réglages voyagent avec les poids dans les checkpoints.
    """

    def __init__(self, encoding="pieces", hidden=5000, width=1000, blocks=4, heads=("policy",), canonical=False,
                 value_detach=False):
        super().__init__()
        self.config = dict(encoding=encoding, hidden=hidden, width=width, blocks=blocks, heads=list(heads),
                           canonical=canonical, value_detach=value_detach)
        self.body = nn.Sequential(
            nn.Linear(INPUT_SIZES[encoding], hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, width), nn.BatchNorm1d(width), nn.ReLU(),
            *[ResidualBlock(width) for _ in range(blocks)])
        self.heads = nn.ModuleDict({h: nn.Linear(width, HEAD_SIZES[h]) for h in heads})

    def forward(self, x):
        z = self.body(x)
        out = {h: layer(z.detach() if h == "value" and self.config["value_detach"] else z)
               for h, layer in self.heads.items()}
        if "value" in out:
            out["value"] = out["value"].squeeze(-1)
        return out

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def save(model, path, **metadata):
    """
    Enregistre la configuration, les poids et des métadonnées (temps d'entraînement, exemples vus...) ; les
    métadonnées doivent être des types simples (nombres, chaînes, listes, dictionnaires, tenseurs).
    """
    model = getattr(model, "_orig_mod", model)   # modèle passé par torch.compile
    torch.save({"config": model.config, "state_dict": model.state_dict(), **metadata}, path)


def load(path, device="cpu"):
    """Retourne (modèle en mode évaluation, métadonnées du checkpoint)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)   # pas de pickle arbitraire
    model = ResMLP(**checkpoint.pop("config")).to(device)
    model.load_state_dict(checkpoint.pop("state_dict"))
    model.eval()
    return model, checkpoint
