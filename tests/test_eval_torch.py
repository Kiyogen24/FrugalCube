import eval_torch


def run(solution, evaluations=10):
    return {"solution": solution, "evaluations": evaluations, "time_s": 0.5}


def test_summary_counts_optimal_solutions_over_all_cases():
    runs = [(run([0] * 20), 20), (run([0] * 22), 20), (run(None), 21)]
    summary = eval_torch.summarize(runs)
    assert summary["solved"] == 2 / 3
    assert summary["optimal_rate"] == 1 / 3
    assert summary["mean_length"] == 21 and summary["mean_optimal_of_solved"] == 20
    assert summary["lengths"] == [20, 22, None]


def test_summary_without_known_optimal_lengths():
    summary = eval_torch.summarize([(run([1, 2]), None), (run(None), None)])
    assert summary["solved"] == 0.5 and summary["optimal_rate"] is None and summary["mean_optimal_of_solved"] is None


def test_checkpoints_are_listed_in_training_order(tmp_path):
    for name in ["final.pt", "ckpt_00020.0min.pt", "ckpt_00010.0min.pt", "train_log.jsonl"]:
        (tmp_path / name).write_text("")
    names = [p.split("\\")[-1].split("/")[-1] for p in eval_torch.checkpoints_of(str(tmp_path))]
    assert names == ["ckpt_00010.0min.pt", "ckpt_00020.0min.pt", "final.pt"]
    assert eval_torch.checkpoints_of(str(tmp_path / "final.pt")) == [str(tmp_path / "final.pt")]
