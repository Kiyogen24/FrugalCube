# FrugalCube

Compute-efficient learned search, from the Rubik's cube to CNOT synthesis.

Learned Rubik's cube solvers find optimal solutions for most random cubes, but with beams that take ten seconds or
more per cube. FrugalCube targets quality per second of search. A small network is trained on random scrambles, without
exact distances or labels from search, and a beam search scores the twelve children of a state with a single network
call, searches the cube and its inverse together, and stops at an exact table of the states near the goal. The same
search, with a network trained on a laptop, also synthesises short CNOT circuits for IBM qubit layouts.

## Results

**Rubik's cube**, on the 1000 cubes of the DeepCubeA test set, quarter-turn metric, one NVIDIA RTX A6000:

| Solver | Optimal solutions | Mean length | Time per cube |
| --- | --- | --- | --- |
| FrugalCube, width 2^18.5 + endgame table (mean of three trainings) | 84.8 % | 20.94 | 0.89 s |
| FrugalCube, width 2^21 + endgame table | 92.5 % | 20.79 | 4.5 s |
| CayleyPy's released network, width 2^18, same GPU (first 200 cubes) | 77.5 % | 21.09 | 15.9 s |

On those 200 cubes, FrugalCube at one second finds 85.8 % optimal solutions.

**CNOT synthesis**, 200 random matrices per layout, one CPU core per method: on IBM's 8-qubit layouts, FrugalCube in
under one second per circuit matches or beats AlphaCNOT's best of 100 searches (about 27 s). From width 1024 it beats
IBM's best of 10,000 rollouts of its PPO policy (13 to 15 s).

The paper, to be added here, gives the comparisons, the ablations and an analysis by exact distance of where a
learned beam loses the optimal path.

## How it works

- **Network.** A residual MLP of 3.9M parameters with a policy head and a value head, trained for 100 minutes on
  random scrambles. The policy learns the inverse of the last scramble move, the value learns the scramble length,
  and prefixes of length 9 to 18 are kept more often than the others.
- **Score.** A child is scored by V(parent) − λ log π(move | parent), so one network call on a parent scores all twelve
  of its children.
- **Two roots.** The cube and its inverse share one beam, and a solution of the inverse, reversed and inverted, solves
  the cube.
- **Endgame table.** The 86M states within 8 moves of the goal (775 MB), built on the GPU by breadth-first search. The
  search stops at the first child found in the table.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | The cube on CPU or GPU (`cube_torch.py`), the network (`network_torch.py`), training (`train_torch.py`), the beam (`beam_torch.py`), the DeepCubeA test set (`deepcubea_testset.py`); CNOT synthesis on the same code (`perm_torch.py`, `circuit_torch.py`, `perm_search.py`, `train_perm.py`) |
| `experiments/` | The scripts behind each result of the paper, by part of the paper (`cube/`, `analysis/`, `pilots/`, `cnot/`), with a map in `experiments/README.md` |
| `models/` | The trained networks of the paper |
| `tests/` | Tests of the cube, the beam, training and CNOT synthesis |

Raw results, intermediate checkpoints (`runs/`) and the endgame tables (`data/`) are not tracked.

## Trained networks

| File | Network |
| --- | --- |
| `models/cube/focused_seed0.pt`, `_seed1.pt`, `_seed2.pt` | The final cube network (3.9M parameters, lengths 9 to 18 kept more often), three trainings with different seeds; seed 0 is the network studied in the paper |
| `models/cube/uniform_seed0.pt`, `_seed1.pt`, `_seed2.pt` | The same network trained on uniform lengths, three seeds |
| `models/cnot/8-H.pt`, `8-T2.pt`, `9-L.pt` | The CNOT networks (0.22M parameters) of IBM's three layouts |

Each file holds the weights, the architecture and the training arguments, and loads with
`torch.load(path, weights_only=True)`. To solve the 1000 test cubes at one second per cube with the first one:

```bash
python experiments/cube/search.py --model models/cube/focused_seed0.pt --device cuda roots=2,ball=8 \
    --split all --cases 1000 --width 360448 --no-base --out-dir runs/focused_seed0
```

and the 200 test matrices of the 8-H layout:

```bash
python experiments/cnot/eval_cnot.py --model models/cnot/8-H.pt --instances experiments/cnot/inst/ibm8qJ_test.json \
    --widths 1024,4096 --lam 64 --threads 1 --device cpu --out runs/cnot_8-H.jsonl
```

## Installation and tests

```bash
pip install -r requirements.txt
python -m pytest
```

The DeepCubeA test set is downloaded to `data/` on first use, from the JSON version published with EfficientCube.

## Reproducing the cube results

Train the network (about 97 minutes on an A6000 for 8.62 billion examples), then fit the value head for 3 minutes on
the frozen trunk:

```bash
cd src
python train_torch.py --out-dir ../runs/final --heads policy --encoding stickers --hidden 2000 --width 512 \
    --blocks 4 --walks 1000 --length 26 --gen-factor 32 --minutes 170 --max-examples 8624746000 \
    --focus 9-18 --focus-keep 0.25 --seed 0
python train_torch.py --init ../runs/final/final.pt --out-dir ../runs/final_v --minutes 3 \
    --heads policy,value --value-hidden 256 --freeze-body --encoding stickers --walks 1000 --length 26 \
    --gen-factor 32 --seed 0
cd ..
```

Search the 1000 test cubes at one second per cube, with the endgame table (built into `data/ball8.pt` on the first
run) and without it:

```bash
python experiments/cube/search.py --model runs/final_v/final.pt --device cuda roots=2,ball=8 \
    --split all --cases 1000 --width 360448 --no-base --out-dir runs/final/table
python experiments/cube/search.py --model runs/final_v/final.pt --device cuda roots=2 \
    --split all --cases 1000 --width 206848 --no-base --out-dir runs/final/no_table
```

The paper's three trainings use seeds 0, 1 and 2. Without `--focus`, the same command trains the uniform network.

## Reproducing the CNOT results

Train on 3 CPU threads for 45 minutes, then search the 200 test matrices of the layout:

```bash
python src/train_perm.py --puzzle cnot --n 8 --topology experiments/cnot/topo/ibm_8qJ.json --minutes 45 \
    --length 300 --walks 32 --value-scale 45 --target-every 500 --hidden 256 --width 256 --blocks 1 \
    --threads 3 --device cpu --arm ss --out-dir runs/cnot/8qJ
python experiments/cnot/eval_cnot.py --model runs/cnot/8qJ/final.pt \
    --instances experiments/cnot/inst/ibm8qJ_test.json --widths 64,256,1024,4096 --tune-width 64 \
    --tune-cases 50 --ball-states 2e6 --save-circuits --threads 1 --device cpu --out runs/cnot/8qJ/eval.jsonl
```

IBM's and AlphaCNOT's released models run in their own Python environments (`.qenv` and `.aenv`), described in
`experiments/README.md`.

## Background

FrugalCube continues the UTT project
[Reinforcement-Learning-for-Rubiks-Cube](https://github.com/Kiyogen24/Reinforcement-Learning-for-Rubiks-Cube)
(Romain Goldenchtein and Simon Gelbart).

## References

- F. Agostinelli, S. McAleer, A. Shmakov, P. Baldi. *Solving the Rubik's cube with deep reinforcement learning and
  search*, Nature Machine Intelligence, 2019 (DeepCubeA and its test set).
- K. Takano. *Self-Supervision is All You Need for Solving Rubik's Cube*, TMLR, 2023 (EfficientCube).
- A. Chervov et al. *A machine learning approach that beats Rubik's cubes*, NeurIPS 2025 (CayleyPy).
- F. Agostinelli et al. *Q\* Search: Heuristic Search with Deep Q-Networks*, ICAPS 2024 PRL workshop.
- D. Kremer et al. *Practical and efficient quantum circuit synthesis and transpiling with Reinforcement Learning*,
  2024 (IBM's models).
- J. Cossio et al. *AlphaCNOT: Learning CNOT Minimization with Model-Based Planning*, 2026.

## License

MIT
