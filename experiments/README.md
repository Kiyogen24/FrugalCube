# Experiments

The scripts behind the paper's results, one folder per part of the paper. Run them from the repository root, for
example `python experiments/analysis/top_move.py`. They import the modules of `src/` and write their results to
`runs/`, which is not tracked, and the paper's figures, tables and numbers are built from those results. Each script
gives its usage at the top, and comments are in French.

## `cube/`: the solver and what each part is worth

| File | What it does | In the paper |
| --- | --- | --- |
| `search.py` | The search of every cube result: one beam from the cube and its inverse, score V(parent) − λ log π(move), optional endgame table (`ball=8`), on GPU or CPU | Sections 4 and 5, Appendices B and C |
| `es.py` | A beam controller trained by evolution strategies, imported by `search.py` for its `es=1` option | Not reported |
| `ablation.py` | Search ablation at equal width: policy score, one root, child values | Table 2 |
| `rotated_root.py` | Share of the beam taken by symmetric copies when the second root is a rotation | Section 4.3 |
| `q_head.py` | A Q-head distilled from the network's own child values | Section 4.3, Appendix F |
| `training_time.py` | Training for 30, 60 and 90 minutes, compared cube by cube | Appendix D |
| `peak_memory.py` | Peak memory of the search per beam state | Appendix A |

The laptop results of Appendix C use `search.py` with `--device cpu`.

## `analysis/`: why width helps

| File | What it does | In the paper |
| --- | --- | --- |
| `by_distance.py` | Value AUC (d against d + 2) and policy accuracy by exact distance | Section 5.3 |
| `walk_labels.py`, `top_move.py` | Whether scramble labels are right, and whether the network's top move brings the cube closer | Table 3 |
| `released_nets.py` | The same measurements on the released networks of DeepCubeA and CayleyPy (cloned in `ext/`, loaded with `weights_only`) | Section 5.3 |
| `teacher_path.py`, `child_values_summary.py` | Where a narrow beam loses a wide teacher's path, and child values against the shared score | Sections 4.3 and 5.2, Appendix F |

The oracle of Section 5.2 is an option of `search.py`. It keeps an optimal path in the beam for the first T levels:

```bash
python experiments/cube/search.py --model models/cube/uniform_seed0.pt roots=2,force=10 \
    --cases 1000 --width 1024 --out-dir runs/oracle
```

## `pilots/`: changing what the network learns

| Folder | What it does | In the paper |
| --- | --- | --- |
| `value_floor/` | Training targets where the value stops ranking: Bellman targets, exact-distance labels, labels from search (expert iteration), a probe on the frozen trunk, a ten-times longer fit | Section 6, Appendix E |
| `beam_survival/` | A ranking loss at the beam cutoff | Section 6 |
| `training_lengths/` | Laptop pilot of the training lengths and of the learning-progress curriculum, and the walk accuracy used as its reward (`walk_accuracy.py`) | Section 6, Appendices D and E |

## `cnot/`: CNOT synthesis

| File | What it does | In the paper |
| --- | --- | --- |
| `topo/`, `inst/` | IBM's 8- and 9-qubit layouts and the 200 test matrices per layout (`make_instances.py`, seed 2026) | Section 7 |
| `eval_cnot.py` | FrugalCube's beam on CNOT synthesis, with λ set on validation matrices; the network is trained by `src/train_perm.py` | Section 7, Appendix G |
| `ibm_linear.py` | IBM's released PPO policies (needs a `.qenv` environment with qiskit 2.1, qiskit-gym and qiskit-ibm-transpiler), and a Qiskit check of every circuit | Section 7 |
| `alphacnot_run.py` | AlphaCNOT's released models (its repository cloned in `ext/AlphaCNOT`, run in a `.aenv` environment with JAX and mctx) | Section 7 |
| `eval_heur.py` | The same beam without a network, guided by a hand-made measure of the matrix | Section 7, Appendix G |
| `score_units.py`, `analyse_pilot.py` | The share of each term of the score in the beam's choices, and the first comparison with IBM | Section 7, Appendix G |

Times are measured with one process at a time, on one thread, with nothing else running on the machine.
