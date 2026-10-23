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


def test_q_targets_use_solved_children_and_scramble_bounds():
    g = torch.Generator().manual_seed(0)
    states, labels, depths = CUBE.self_supervised_batch(20, 3, generator=g)
    y = train_torch.q_targets(ConstantQ(100.0), CUBE, CUBE.encode, states, labels, depths, canonical=False, amp=False)
    rows = torch.arange(len(states))
    # profondeur 1 : l'inverse du dernier coup résout le cube, cible exacte 1
    assert torch.all(y[rows[depths == 1], labels[depths == 1]] == 1.0)
    # sinon la borne du mélange s'applique : depth pour l'inverse du dernier coup, depth + 2 pour les autres
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
