"""
Références IBM (Kremer et al., 2024, arXiv 2405.13196) sur nos instances, dans l'environnement .qenv (qiskit 2.1,
qiskit-gym, qiskit-ibm-transpiler; HF_HOME=.qenv/hf). Les politiques PPO publiées (Hugging Face
qiskit/ai-transpiler_linear-functions, celles du mode local du transpileur) sont appelées directement par
RLSynthesis.synth avec num_searches rollouts échantillonnés (10: le défaut du transpileur; on garde le plus court)
ou en glouton. Pour chaque instance: nombre de CNOT (un SWAP compte pour 3), profondeur en CNOT, temps, et
vérification par Qiskit que le circuit réalise la matrice.
--ours vérifie de la même façon les circuits de notre solveur (eval_cnot.py --save-circuits).

Usage (depuis la racine du dépôt):
    .qenv/Scripts/python experiments/cnot/ibm_linear.py --model 9qL --export-topology experiments/cnot/topo/ibm_9qL.json
    .qenv/Scripts/python experiments/cnot/ibm_linear.py --model 9qL --instances experiments/cnot/inst/ibm9qL_test.json \
        --searches greedy,1,10,100 --out runs/cnot/ibm/ibm9qL_test.jsonl
    .qenv/Scripts/python experiments/cnot/ibm_linear.py --instances ... --ours runs/cnot/line9/eval.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction


def model_files(name, kind="linear_function"):
    from qiskit_ibm_transpiler.hf_models_client import HFInterface
    from qiskit_ibm_transpiler.model_bootstrap import TYPE_CONFIGS
    cfg = TYPE_CONFIGS[kind]
    snap = Path(HFInterface().download_models(repo_id=cfg.default_repo_id, revision=cfg.default_revision))
    config = snap / f"{kind}_{name}.json"
    if not config.exists():
        raise FileNotFoundError(f"{config.name} absent ; modèles : {sorted(p.stem for p in snap.glob('*.json'))}")
    return config, config.with_suffix(".safetensors")


def edges_of(config):
    gates = json.load(open(config))["env"]["gateset"]
    return sorted({(min(q), max(q)) for _, q in gates})


def cnots(qc):
    ops = qc.count_ops()
    return int(ops.get("cx", 0) + 3 * ops.get("swap", 0))


def cnot_depth(qc):
    flat = qc.decompose(["swap"]) if "swap" in qc.count_ops() else qc
    return flat.depth(lambda inst: inst.operation.num_qubits > 1)


def realises(qc, matrix):
    return bool(np.array_equal(np.asarray(LinearFunction(qc).linear, dtype=int), np.asarray(matrix, dtype=int)))


def check_convention():
    """notre convention: le circuit réalise x -> M x, CX(c, t) fait x_t ^= x_c; Qiskit doit donner M[t][c] = 1."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    assert np.array_equal(np.asarray(LinearFunction(qc).linear, dtype=int), [[1, 0], [1, 1]]), "convention"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="9qL", help="nom du modèle IBM (linear_function_<nom>.json)")
    parser.add_argument("--export-topology", help="écrit le graphe de couplage du modèle (format circuit_torch)")
    parser.add_argument("--instances")
    parser.add_argument("--cases", type=int, default=0)
    parser.add_argument("--searches", default="greedy,1,10,100", help="greedy ou nombre de rollouts")
    parser.add_argument("--ours", help="jsonl de eval_cnot.py --save-circuits à vérifier")
    parser.add_argument("--out")
    args = parser.parse_args()
    check_convention()

    if args.export_topology:
        config, _ = model_files(args.model)
        n = json.load(open(config))["env"]["num_qubits"]
        os.makedirs(os.path.dirname(args.export_topology) or ".", exist_ok=True)
        with open(args.export_topology, "w") as f:
            json.dump({"n": n, "edges": edges_of(config), "source": f"qiskit/ai-transpiler_linear-functions "
                                                                     f"{config.name}"}, f)
        print(f"{n} qubits, arêtes {edges_of(config)} -> {args.export_topology}")
        return

    spec = json.load(open(args.instances))
    mats = spec["instances"][:args.cases or None]

    if args.ours:
        rows = [json.loads(line) for line in open(args.ours) if '"circuit"' in line]
        bad = 0
        for r in rows:
            qc = QuantumCircuit(spec["n"])
            for c, t in r["circuit"]:
                assert (min(c, t), max(c, t)) in {tuple(e) for e in spec["edges"]}, "CX hors du graphe"
                qc.cx(c, t)
            ok = realises(qc, mats[r["i"]]) and cnots(qc) == r["length"]
            bad += not ok
        print(f"nos circuits : {len(rows) - bad}/{len(rows)} vérifiés par Qiskit (matrice réalisée, CX permis, "
              f"longueur)")
        return

    from qiskit_gym.rl import RLSynthesis
    config, weights = model_files(args.model)
    if [tuple(e) for e in spec["edges"]] != edges_of(config):
        raise ValueError("les instances ne sont pas sur le graphe du modèle IBM")
    model = RLSynthesis.from_config_json(str(config), model_path=str(weights))
    n = spec["n"]

    def circuit_of(m):
        qc = QuantumCircuit(n)
        qc.append(LinearFunction(np.asarray(m, dtype=bool)), range(n))
        return qc

    model.synth(input=circuit_of(mats[0]), num_searches=1)          # échauffement (chargement, compilation)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for k in args.searches.split(","):
        lengths, times, fails = [], [], 0
        for i, m in enumerate(mats):
            qc = circuit_of(m)
            t0 = time.perf_counter()
            if k == "greedy":
                out = model.synth(input=qc, deterministic=True, num_searches=1)
            else:
                out = model.synth(input=qc, num_searches=int(k))
            dt = time.perf_counter() - t0
            row = {"i": i, "searches": k, "time_s": round(dt, 4)}
            if out is None or not realises(out, m):
                row.update(length=None, ok=False)
                fails += 1
            else:
                row.update(length=cnots(out), depth=cnot_depth(out), ok=True)
                lengths.append(row["length"])
            times.append(dt)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"IBM {args.model} searches={k} : {len(lengths)}/{len(mats)} résolus, "
              f"CNOT moyen {np.mean(lengths):.2f}, {np.mean(times):.3f} s par instance", flush=True)


if __name__ == "__main__":
    main()
