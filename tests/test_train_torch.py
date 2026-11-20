import json
import os
import subprocess
import sys

import torch

import cube_torch as ct
import network_torch as nt
import train_torch

CUBE = ct.TorchCube("cpu")
SRC_DIR = os.path.dirname(train_torch.__file__)


class ConstantQ(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return {"q": torch.full((len(x), 12), self.value)}


def test_detached_value_head_learns_without_changing_the_trunk():
    torch.manual_seed(0)
    states, _, depths = CUBE.self_supervised_batch(30, 6, generator=torch.Generator().manual_seed(0))
    x = CUBE.encode(states)
    for detach in (True, False):
        net = nt.ResMLP(hidden=32, width=16, blocks=1, heads=("policy", "value"), value_detach=detach)
        ((net(x)["value"] - depths.float()) ** 2).mean().backward()
        trunk = [p.grad for p in net.body.parameters()]
        assert all(g is None for g in trunk) == detach
        assert net.heads["value"].weight.grad.abs().sum() > 0
    assert nt.ResMLP(heads=("policy",), hidden=8, width=8, blocks=1).config["value_detach"] is False


def test_one_hidden_layer_network_and_unchanged_checkpoint_layout():
    net = nt.ResMLP("stickers", hidden=0, width=16, blocks=0, heads=("policy", "value"))
    linear = [m for m in net.body if isinstance(m, torch.nn.Linear)]
    assert len(linear) == 1 and linear[0].in_features == 324
    assert net(torch.zeros(3, 324))["policy"].shape == (3, 12)
    # avec une première couche, les noms des poids sont ceux des anciens checkpoints
    keys = nt.ResMLP("stickers", hidden=32, width=16, blocks=1).state_dict()
    assert keys["body.0.weight"].shape == (32, 324) and keys["body.3.weight"].shape == (16, 32)


def test_split_batches_keep_the_layout_of_smaller_batches():
    length, n, k = 5, 4, 3
    states, labels, depths = CUBE.self_supervised_batch(n * k, length, generator=torch.Generator().manual_seed(0))
    parts = train_torch.split_batches(states, labels, depths, length, k)
    assert len(parts) == k
    for j, (s, l, d) in enumerate(parts):
        assert s.shape == (length * n, 54) and torch.equal(d, torch.arange(1, length + 1).repeat_interleave(n))
        # le mélange i du lot j est le mélange j·n + i du grand lot
        assert torch.equal(s.reshape(length, n, 54), states.reshape(length, n * k, 54)[:, j * n:(j + 1) * n])
        assert torch.equal(l.reshape(length, n), labels.reshape(length, n * k)[:, j * n:(j + 1) * n])
        # l'étiquette ramène bien au préfixe précédent
        back = CUBE.apply_sequence(s.reshape(length, n, 54)[2, 0], [int(l.reshape(length, n)[2, 0])])
        assert torch.equal(back, s.reshape(length, n, 54)[1, 0])


def test_focused_batches_favour_the_focus_lengths_at_equal_batch_size():
    walks, length, k = 50, 26, 8
    p = train_torch.focus_keep(length, (9, 18), 0.25)
    parts = train_torch.focused_batches(CUBE, walks, length, k, p, torch.Generator().manual_seed(0))
    assert len(parts) >= k - 1
    depths = torch.cat([d for _, _, d in parts])
    assert all(len(s) == len(l) == len(d) == walks * length for s, l, d in parts)
    assert depths.min() >= 1 and depths.max() <= length
    per_length = torch.bincount(depths, minlength=length + 1)[1:].float() / len(parts)
    inside, outside = per_length[8:18].mean(), torch.cat([per_length[:8], per_length[18:]]).mean()
    # 4 fois plus d'exemples par longueur dans l'intervalle, et plus qu'avec des lots uniformes (walks par longueur)
    assert 3.5 < inside / outside < 4.5 and inside > 1.7 * walks
    # les exemples gardés restent des exemples justes: l'étiquette ramène au préfixe précédent
    s, l, d = parts[0]
    i = int((d > 1).nonzero()[0])
    back = CUBE.apply_sequence(s[i], [int(l[i])])
    assert CUBE.state_hash(back[None]).item() != CUBE.state_hash(s[i][None]).item()


def test_length_bandit_favours_the_lengths_whose_loss_falls():
    length = 10
    bandit = train_torch.LengthBandit(length, "cpu", every=1, floor=0.25)
    depths = torch.arange(1, length + 1).repeat(20)
    assert torch.equal(bandit.keep, torch.ones(length))
    # la perte baisse aux longueurs 4 à 6 seulement, elle est stable ailleurs
    for t in range(30):
        loss = torch.ones(len(depths))
        falling = (depths >= 4) & (depths <= 6)
        loss[falling] = 2.0 - 0.05 * t
        bandit.observe(loss, depths)
    # progrès égal aux longueurs 4 à 6, lissé sur les voisines: 1 au centre, 3/4 du progrès à côté, 1/4 au-delà
    expected = torch.tensor([0.25, 0.25, 0.4375, 0.8125, 1.0, 0.8125, 0.4375, 0.25, 0.25, 0.25])
    assert torch.allclose(bandit.keep, expected, atol=1e-5)
    # même résultat quand les pertes moyennes par longueur viennent d'un jeu fixe (version 2)
    probed = train_torch.LengthBandit(length, "cpu", every=1, floor=0.25)
    for t in range(30):
        mean = torch.ones(length)
        mean[3:6] = 2.0 - 0.05 * t
        probed.update(mean)
    assert torch.allclose(probed.keep, expected, atol=1e-5)


def test_alp_bandit_uses_absolute_progress_and_soft_allocation():
    length, horizon = 10, 4
    rising = train_torch.LengthBandit(length, "cpu", every=1, floor=0.25, alp=True, horizon=horizon, top=3, inertia=0.0)
    falling = train_torch.LengthBandit(length, "cpu", every=1, floor=0.25, alp=True, horizon=horizon, top=3,
                                       inertia=0.0)
    for t in range(horizon):
        up, down = torch.full((length,), 0.5), torch.full((length,), 0.5)
        up[3:6] = 0.2 + 0.05 * t
        down[3:6] = 0.8 - 0.05 * t
        if t < horizon - 1:
            assert torch.equal(rising.keep, torch.ones(length))   # pas de décision avant `horizon` mesures
        rising.update(up)
        falling.update(down)
    # progrès égal aux longueurs 4 à 6, lissé: 1/4, 3/4, 1, 3/4, 1/4; référence = moyenne des 3 plus forts (5/6)
    expected = torch.tensor([0.25, 0.25, 0.475, 0.925, 1.0, 0.925, 0.475, 0.25, 0.25, 0.25])
    assert torch.allclose(rising.keep, expected, atol=1e-5)
    assert torch.allclose(falling.keep, expected, atol=1e-5)   # une hausse compte comme une baisse
    # progrès faible mais égal partout: rien n'est amplifié, toutes les longueurs intérieures restent gardées
    even = train_torch.LengthBandit(length, "cpu", every=1, floor=0.25, alp=True, horizon=horizon, inertia=0.0)
    for t in range(horizon):
        even.update(torch.full((length,), 0.5 + 1e-2 * t))
    assert torch.allclose(even.keep[1:-1], torch.ones(length - 2), atol=1e-3)


def test_probe_losses_use_the_eval_mode_and_one_value_per_length():
    net = nt.ResMLP("stickers", hidden=32, width=16, blocks=1, heads=("policy",))
    args = type("Args", (), {"compile": False, "bf16": False, "heads": "policy", "canonical": False, "lr": 1e-3,
                             "freeze_body": False})()
    trainer = train_torch.Trainer(net, CUBE, args)
    probe = CUBE.self_supervised_batch(20, 6, generator=torch.Generator().manual_seed(1))
    mean = train_torch.probe_losses(trainer, probe, 6)
    assert mean.shape == (6,) and torch.isfinite(mean).all()
    assert not net.training
    acc = train_torch.probe_losses(trainer, probe, 6, signal="acc")
    assert acc.shape == (6,) and bool(((acc >= 0) & (acc <= 1)).all())


def test_path_consistency_vanishes_exactly_on_consistent_values():
    length, n, alpha = 5, 3, 0.7
    logp = -torch.rand(length, n)
    depth = torch.arange(1, length + 1).float()[:, None].expand(length, n)
    # V(s_k) = k + alpha * somme des log pi le long du chemin jusqu'au cube résolu: cohérent par construction
    consistent = depth + alpha * logp.cumsum(dim=0)
    residual = train_torch.path_consistency(consistent.reshape(-1), logp.reshape(-1), length, alpha)
    assert torch.allclose(residual, torch.zeros(length * n), atol=1e-6)
    assert torch.allclose(train_torch.path_consistency(depth.reshape(-1), torch.zeros(length * n), length, alpha),
                          torch.zeros(length * n))
    values = torch.randn(length * n)
    v = values.reshape(length, n)
    expected = v[2] - 1 - v[1] - alpha * logp[2]
    assert torch.allclose(train_torch.path_consistency(values, logp.reshape(-1), length, alpha).reshape(length, n)[2],
                          expected)


def test_q_targets_use_solved_children_and_scramble_bounds():
    g = torch.Generator().manual_seed(0)
    states, labels, depths = CUBE.self_supervised_batch(20, 3, generator=g)
    y = train_torch.q_targets(ConstantQ(100.0), CUBE, CUBE.encode, states, labels, depths, canonical=False, amp=False)
    rows = torch.arange(len(states))
    # profondeur 1: l'inverse du dernier coup résout le cube, cible exacte 1
    assert torch.all(y[rows[depths == 1], labels[depths == 1]] == 1.0)
    # sinon la borne du mélange s'applique: depth pour l'inverse du dernier coup, depth + 2 pour les autres
    assert torch.all(y[rows, labels][depths > 1] == depths[depths > 1].float())
    others = torch.ones_like(y, dtype=torch.bool)
    others[rows, labels] = False
    assert torch.all(y[others] <= (depths + 2).float().unsqueeze(1).expand_as(y)[others])
    # un réseau cible qui prédit 0 donne la cible 1 partout (pas de cible négative ni nulle)
    y0 = train_torch.q_targets(ConstantQ(-5.0), CUBE, CUBE.encode, states, labels, depths, canonical=False, amp=False)
    assert torch.all(y0 == 1.0)


def test_prioritized_buffer_samples_high_priority_items_more():
    buffer = train_torch.PrioritizedBuffer(100, "cpu", alpha=1.0, beta=0.4)
    states = CUBE.solved.expand(10, 54).clone()
    labels = torch.arange(10)
    buffer.add(states, labels, torch.ones(10, dtype=torch.long), torch.tensor([1.0] * 9 + [91.0]))
    assert buffer.size == 10
    idx, _, sampled_labels, _, weights = buffer.sample(2000, torch.Generator().manual_seed(0))
    assert (sampled_labels == 9).float().mean() > 0.8
    assert torch.all(weights <= 1.0) and torch.all(weights > 0)
    buffer.update(torch.tensor([9]), torch.tensor([1.0]))
    _, _, sampled_labels, _, _ = buffer.sample(2000, torch.Generator().manual_seed(1))
    assert (sampled_labels == 9).float().mean() < 0.2


def run_training(tmp_path, *extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "run"
    cmd = [sys.executable, os.path.join(SRC_DIR, "train_torch.py"), "--minutes", "0.05", "--out-dir", str(out),
           "--hidden", "64", "--width", "32", "--blocks", "1", "--walks", "8", "--length", "6",
           "--log-seconds", "1", "--checkpoint-minutes", "0.02", "--device", "cpu", *extra]
    subprocess.run(cmd, check=True, capture_output=True, cwd=tmp_path)
    return out


def test_training_writes_logs_and_loadable_checkpoints(tmp_path):
    out = run_training(tmp_path, "--heads", "policy,value,q", "--canonical", "--per", "--target-every", "5")
    entries = [json.loads(line) for line in open(out / "train_log.jsonl")]
    assert entries and all(k in entries[-1] for k in ("policy_loss", "policy_acc", "value_loss", "q_loss"))
    checkpoints = sorted(p for p in os.listdir(out) if p.startswith("ckpt_"))
    assert checkpoints
    model, meta = nt.load(out / "final.pt")
    assert meta["steps"] > 0 and meta["examples"] > 0 and meta["args"]["per"]
    assert model.config["canonical"] and model.config["heads"] == ["policy", "value", "q"]


def test_training_can_start_from_a_checkpoint(tmp_path):
    out = run_training(tmp_path, "--heads", "policy", "--encoding", "stickers")
    out2 = run_training(tmp_path / "suite", "--init", str(out / "final.pt"))
    model, _ = nt.load(out2 / "final.pt")
    assert model.config["encoding"] == "stickers"


def test_value_head_added_to_a_frozen_trained_network(tmp_path):
    out = run_training(tmp_path, "--heads", "policy", "--encoding", "stickers")
    grown = run_training(tmp_path / "tete", "--init", str(out / "final.pt"), "--heads", "policy,value",
                         "--value-hidden", "16", "--freeze-body", "--bf16")
    before, _ = nt.load(out / "final.pt")
    after, _ = nt.load(grown / "final.pt")
    assert after.config["heads"] == ["policy", "value"] and after.config["value_hidden"] == 16
    # tronc (statistiques des BatchNorm comprises) et politique strictement inchangés
    for key, tensor in before.state_dict().items():
        assert torch.equal(tensor, after.state_dict()[key]), key
    x = CUBE.encode_stickers(CUBE.self_supervised_batch(4, 5)[0])
    assert torch.equal(before(x)["policy"], after(x)["policy"]) and after(x)["value"].shape == (20,)
    entries = [json.loads(line) for line in open(grown / "train_log.jsonl")]
    assert "value_loss" in entries[-1]


def test_path_consistency_training(tmp_path):
    out = run_training(tmp_path, "--heads", "policy,value", "--encoding", "stickers", "--pcl-weight", "1",
                       "--gen-factor", "3")
    entries = [json.loads(line) for line in open(out / "train_log.jsonl")]
    assert "pcl_loss" in entries[-1]
    # critique seul, sans autre perte: la politique ne change pas, la valeur si
    critic = run_training(tmp_path / "critique", "--init", str(out / "final.pt"), "--heads", "policy,value",
                          "--pcl-weight", "1", "--pcl-critic-only", "--policy-weight", "0", "--value-weight", "0")
    before, _ = nt.load(out / "final.pt")
    after, _ = nt.load(critic / "final.pt")
    assert torch.equal(before.heads["policy"].weight, after.heads["policy"].weight)
    assert not torch.equal(before.heads["value"].weight, after.heads["value"].weight)
