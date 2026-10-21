# FrugalCube

Solve the Rubik's Cube in a frugal way: self-supervised learning and reinforcement learning, with the goal of matching CayleyPy's results with much less compute.

## Background

FrugalCube continues the UTT project [Reinforcement-Learning-for-Rubiks-Cube](https://github.com/Kiyogen24/Reinforcement-Learning-for-Rubiks-Cube) (Romain Goldenchtein and Simon Gelbart), whose report is on [HAL](https://utt.hal.science/hal-05185396v2). That project started from DeepCube (autodidactic iteration and MCTS) and stalled at scrambles of about 10 moves.

Its self-supervised objective, predicting the inverse of the last scramble move, was found independently in that project; the same objective was published by Takano (2023). With fresh data at every step and a beam search at solving time, a model trained for about 2 hours on a laptop CPU already solves 96 % of the 1,000 DeepCubeA test cubes (beam width 512, 2.5 s per cube). Its solutions are long, though: 31.3 moves on average against 20.6 for the optimal ones, and only 0.5 % are optimal, where CayleyPy reports 98.4 %. Closing that gap cheaply is the goal of this repository.

## What is compared

All methods share the same cube simulator, data generator, network architecture (the residual MLP of DeepCubeA), time budget and beam search code, so that differences come from the method alone.

| Method | Training target | Search |
| --- | --- | --- |
| Takano-style | next move (inverse of the last scramble move), sticker encoding | beam ranked by the policy |
| CayleyPy-style | distance (scramble length), sticker encoding | beam ranked by the value, width / 12 for an equal search cost |
| Ours | next move, piece encoding, states reduced to a canonical form under the 48 cube symmetries | policy beam, optionally bidirectional: a second beam solves the inverse cube s⁻¹ with the same network and the two meet in the middle |
| RL variants, from the same model | prioritized replay; search → labels loop (beam solutions shorter than the scramble become training examples); Q-learning head | as above, or ranked by the Q head |

Each run saves a checkpoint every few minutes. Every checkpoint is evaluated on the DeepCubeA test set (1,000 fully scrambled cubes with known optimal lengths): solve rate, share of optimal solutions, mean length and network evaluations. This gives quality-versus-training-time curves at equal budget.

## Status

Work in progress: the PyTorch pipeline is tested on CPU, and the first GPU session (RTX A6000) is next.

## Layout

| File | Role |
| --- | --- |
| `src/cube_torch.py` | Batched cube on CPU or GPU: moves, inverse, hashing, encodings, 48 symmetries, scrambles |
| `src/fast_cube.py`, `src/Cube.py` | NumPy and pycuber versions of the cube, from which the move tables are derived and checked |
| `src/network_torch.py` | Residual MLP with `policy`, `value` and `q` heads |
| `src/train_torch.py` | Time-budgeted training with regular checkpoints, prioritized replay (`--per`) and canonical form (`--canonical`) |
| `src/expert_iteration_torch.py` | Search → labels loop |
| `src/beam_torch.py` | Beam search ranked by the policy, the Q head or the value, forward or bidirectional |
| `src/eval_torch.py` | Evaluation on the DeepCubeA test set, for one checkpoint or a whole run |
| `src/bench_torch.py` | Data generation, training and search throughput |
| `gpu_session.sh` | Full GPU session: tests, throughput, the three methods, curves, RL variants, final evaluation |

## Installation and tests

```bash
pip install -r requirements.txt
python -m pytest
```

## Usage

```bash
python src/train_torch.py --minutes 90 --heads policy --canonical --out-dir runs/ours
python src/eval_torch.py runs/ours --cases 200 --width 4096 --out runs/curve_ours.jsonl
nohup bash gpu_session.sh > runs/session.log 2>&1 &
```

The DeepCubeA test set is downloaded to `data/` on first use, from the JSON version published with EfficientCube.

## References

- S. McAleer, F. Agostinelli, A. Shmakov, P. Baldi. *Solving the Rubik's Cube Without Human Knowledge*, 2018 (DeepCube).
- F. Agostinelli, S. McAleer, A. Shmakov, P. Baldi. *Solving the Rubik's cube with deep reinforcement learning and search*, Nature Machine Intelligence, 2019 (DeepCubeA, test set).
- K. Takano. *Self-Supervision is All You Need for Solving Rubik's Cube*, TMLR, 2023 (EfficientCube).
- CayleyPy, NeurIPS 2025.
- F. Agostinelli et al. *Q\* Search: Heuristic Search with Deep Q-Networks*, 2024.
- T. Anthony, Z. Tian, D. Barber. *Thinking Fast and Slow with Deep Learning and Tree Search*, NeurIPS 2017 (expert iteration).
- T. Schaul, J. Quan, I. Antonoglou, D. Silver. *Prioritized Experience Replay*, ICLR 2016.

## License

MIT
