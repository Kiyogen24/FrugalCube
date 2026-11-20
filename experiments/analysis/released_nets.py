"""
les mesures par distance de l'article sur des réseaux entraînés par d'autres: le réseau cube3 publié par DeepCubeA
(itération de valeur, ext/deepcubea) et celui publié par CayleyPy (modèle 333, ext/cayleypy-cube), à côté de nos
réseaux de valeur. Sur les préfixes des solutions optimales des 1000 cubes de test (distance exacte d):
- AUC entre les distances d et d + 2 (probabilité que la valeur classe l'état à d + 2 au-dessus de celui à d);
- valeur moyenne par d;
- top-1 glouton: l'enfant de plus petite valeur est l'enfant optimal connu (12 appels par état, la politique que
  donne une valeur).
Aucun code des deux dépôts n'est exécuté. Les deux réseaux sont reconstruits ici d'après leur source (mêmes couches,
mêmes noms), leurs poids sont chargés avec weights_only, et les états de test de DeepCubeA sont lus par un unpickler
restreint (des tableaux numpy et un remplaçant passif de Cube3State, rien d'autre). On passe de nos états aux leurs
par une permutation des facettes et un renommage des couleurs, lus sur les 1000 cubes de test que les deux dépôts ont
aussi, et vérifiés sur les 1000.
Usage: python experiments/analysis/released_nets.py   (résultats dans runs/horizon/released.json)
"""
import json
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
import beam_torch as bt  # noqa: E402
import cube_torch as ct  # noqa: E402
import deepcubea_testset as dt  # noqa: E402
import fast_cube as fc  # noqa: E402
import network_torch as nt  # noqa: E402

torch.set_num_threads(16)
OUT = "runs/horizon/released.json"
DCA = "ext/deepcubea"
CPY = "ext/cayleypy-cube"
cube = ct.TorchCube("cpu")


# ---------- les cubes de test et les préfixes de leurs solutions optimales (comme dans by_distance.py) ----------
prefix, dist, best, finals = [], [], [], []
for scramble in json.load(open(dt.PATH)):
    st = fc.SOLVED
    for j, m in enumerate(scramble.split(), 1):
        a = fc.ACTIONS.index(m)
        st = fc.apply_sequence(st, [a])
        prefix.append(st)
        dist.append(j)
        best.append(fc.INVERSE_ACTION[a])
    finals.append(st)
prefix_np, dist, best = np.stack(prefix).astype(np.int64), np.array(dist), np.array(best)
finals = np.stack(finals).astype(np.int64)
prefix = cube.from_numpy(np.stack(prefix))
kids = cube.children(prefix)                                    # (N, 12, 54), dans l'ordre de fc.ACTIONS
# l'enfant optimal connu d'un préfixe à distance j est le préfixe à distance j - 1 (ou le cube résolu)
prev = np.concatenate([[fc.SOLVED], prefix_np[:-1]])
prev[dist == 1] = fc.SOLVED
assert (kids[np.arange(len(best)), best].numpy() == prev).all(), "move order of children and of the labels differ"
kids_np = kids.numpy().astype(np.int64)
our_colour = lambda s: s // 9                                    # nos facettes: id // 9 est la couleur de la face


def colour_map(theirs, ours):
    """Permutation des facettes et renommage des couleurs, avec theirs[i, p] == relabel[ours[i, sigma[p]]] pour chaque
    cube de test i."""
    sigma, relabel = [], {}
    for p in range(theirs.shape[1]):
        cands = []
        for q in range(ours.shape[1]):
            pairs = set(zip(ours[:, q].tolist(), theirs[:, p].tolist()))
            if len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs}):
                cands.append((q, dict(pairs)))
        if len(cands) != 1:
            # les 6 centres ne bougent jamais, donc on les apparie par leur couleur
            cands = [(q, m) for q, m in cands if all(relabel.get(a, b) == b for a, b in m.items())]
        assert len(cands) == 1, f"position {p}: {len(cands)} candidates"
        q, m = cands[0]
        for a, b in m.items():
            assert relabel.setdefault(a, b) == b, "inconsistent colour relabelling"
        sigma.append(q)
    sigma = np.array(sigma)
    lut = np.zeros(max(relabel) + 1, dtype=np.int64)
    for a, b in relabel.items():
        lut[a] = b
    assert (lut[ours[:, sigma]] == theirs).all(), "conversion does not reproduce the test cubes"
    return lambda s: lut[our_colour(s)[..., sigma]]


# ---------- DeepCubeA: le réseau cube3 publié ----------
class _Cube3State:
    def __setstate__(self, state):
        slots = state[1] if isinstance(state, tuple) else state
        self.colors = slots["colors"]


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("environments.cube3", "Cube3State"):
            return _Cube3State
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            import numpy._core.multiarray as mu
            return getattr(mu, name)
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"refused: {module}.{name}")


class DeepCubeANet(nn.Module):
    """utils/pytorch_models.ResnetModel(324, 6, 5000, 1000, 4, 1, True), mêmes couches et mêmes noms."""

    def __init__(self):
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(324, 5000), nn.BatchNorm1d(5000)
        self.fc2, self.bn2 = nn.Linear(5000, 1000), nn.BatchNorm1d(1000)
        self.blocks = nn.ModuleList([nn.ModuleList([nn.Linear(1000, 1000), nn.BatchNorm1d(1000),
                                                    nn.Linear(1000, 1000), nn.BatchNorm1d(1000)]) for _ in range(4)])
        self.fc_out = nn.Linear(1000, 1)

    def forward(self, colours):
        x = F.one_hot(colours.long(), 6).float().view(-1, 324)
        x = F.relu(self.bn2(self.fc2(F.relu(self.bn1(self.fc1(x))))))
        for f1, b1, f2, b2 in self.blocks:
            x = F.relu(b2(f2(F.relu(b1(f1(x))))) + x)
        return self.fc_out(x).flatten()


class PilgrimNet(nn.Module):
    """pilgrim/model.Pilgrim(54, 700, 643, 4, num_classes=6) de CayleyPy, mêmes couches et mêmes noms, en mode éval."""

    def __init__(self, hd1, hd2, nrd):
        super().__init__()
        self.input_layer, self.bn1 = nn.Linear(54 * 6, hd1), nn.BatchNorm1d(hd1)
        self.hidden_layer, self.bn2 = nn.Linear(hd1, hd2), nn.BatchNorm1d(hd2)
        block = lambda: nn.ModuleDict(dict(fc1=nn.Linear(hd2, hd2), bn1=nn.BatchNorm1d(hd2),
                                           fc2=nn.Linear(hd2, hd2), bn2=nn.BatchNorm1d(hd2)))
        self.residual_blocks = nn.ModuleList([block() for _ in range(nrd)])
        self.output_layer = nn.Linear(hd2, 1)

    def forward(self, colours):
        x = F.one_hot(colours.long(), 6).float().view(len(colours), -1)
        x = F.relu(self.bn1(self.input_layer(x)))
        x = F.relu(self.bn2(self.hidden_layer(x)))
        for b in self.residual_blocks:
            x = F.relu(b["bn2"](b["fc2"](F.relu(b["bn1"](b["fc1"](x))))) + x)
        return self.output_layer(x).flatten()


def strip(state, prefix="module."):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}


nets = {}
with open(f"{DCA}/data/cube3/test/data_0.pkl", "rb") as f:
    dca_test = _SafeUnpickler(f).load()
dca_colours = np.stack([s.colors for s in dca_test["states"]]).astype(np.int64) // 9
to_dca = colour_map(dca_colours, our_colour(finals))
m = DeepCubeANet()
m.load_state_dict(strip(torch.load(f"{DCA}/saved_models/cube3/current/model_state_dict.pt", map_location="cpu",
                                   weights_only=True)))
nets["DeepCubeA (released, value iteration)"] = (m.eval(), to_dca)

cfg = json.load(open(f"{CPY}/logs/model_p054-t000_333.json"))
V0 = torch.load(f"{CPY}/targets/p054-t000.pt", map_location="cpu", weights_only=True).long()
cpy_test = torch.load(f"{CPY}/datasets/p054-t000-deepcubea.pt", map_location="cpu", weights_only=True).long()
cpy_colours = (cpy_test - V0.min()).numpy()
to_cpy = colour_map(cpy_colours, our_colour(finals))
m = PilgrimNet(cfg["hd1"], cfg["hd2"], cfg["nrd"])
state = torch.load(f"{CPY}/weights/p054-t000_333_e08192.pth", map_location="cpu", weights_only=True)
m.load_state_dict({k.replace("residual_blocks.", "residual_blocks.", 1): v for k, v in state.items()})
nets["CayleyPy (released, model 333)"] = (m.eval(), to_cpy)
print("conversions checked on the 1000 test cubes for both repositories", flush=True)


def ours(path):
    net, _ = nt.load(path)
    h = bt.NetworkHeuristic(net, cube, "value", chunk=1 << 13)
    return h, None


for name, path in (("ours, uniform 90 min (value head on the frozen trunk)", "runs/s4/m90_v/final.pt"),
                   ("ours, CayleyPy-style value network (session 1)", "runs/cayleypy/final.pt")):
    nets[name] = ours(path)


def values(net, convert, states):
    out = []
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        for s in range(0, len(states), 8192):
            chunk = states[s:s + 8192]
            if convert is None:
                out.append(net.outputs(cube.from_numpy(chunk), "value").float().flatten())
            else:
                out.append(net(torch.as_tensor(convert(chunk))).float())
    return torch.cat(out).numpy()


def auc(lo, hi):
    both = np.concatenate([lo, hi])
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    ranks = np.argsort(np.argsort(both, kind="stable"), kind="stable") + 1.0
    ranks = (np.bincount(inv, ranks) / counts)[inv]
    return float((ranks[len(lo):].sum() - len(hi) * (len(hi) + 1) / 2) / (len(lo) * len(hi)))


solved = fc.SOLVED[None].astype(np.int64)
results = {}
ds = list(range(1, int(dist.max()) + 1))
for name, (net, convert) in nets.items():
    v = values(net, convert, prefix_np)
    vk = values(net, convert, kids_np.reshape(-1, 54)).reshape(len(prefix_np), 12)
    greedy = vk.argmin(1) == best
    rec = {"value_solved": float(values(net, convert, solved)[0]),
           "d_count": [int((dist == d).sum()) for d in ds],
           "d_value": [float(v[dist == d].mean()) for d in ds],
           "d_auc_plus2": [auc(v[dist == d], v[dist == d + 2]) for d in ds if (dist == d + 2).any()],
           "d_greedy_top1": [float(greedy[dist == d].mean()) for d in ds]}
    results[name] = rec
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"{name}\n  V(goal) {rec['value_solved']:.2f} | mean V d=1,5,10,15,20: "
          + " ".join(f"{rec['d_value'][d - 1]:.1f}" for d in (1, 5, 10, 15, 20))
          + "\n  AUC d/d+2 d=8,10,12,14,15,16,18,20: "
          + " ".join(f"{rec['d_auc_plus2'][d - 1]:.3f}" for d in (8, 10, 12, 14, 15, 16, 18, 20))
          + "\n  greedy top-1 d=8,10,12,14,16,18,20: "
          + " ".join(f"{rec['d_greedy_top1'][d - 1]:.3f}" for d in (8, 10, 12, 14, 16, 18, 20)), flush=True)
print("done", flush=True)
