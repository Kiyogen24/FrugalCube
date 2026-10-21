#!/usr/bin/env bash
# Session GPU complète, à lancer avec nohup depuis la racine du dépôt :
#     nohup bash gpu_session.sh > runs/session.log 2>&1 &
# Chaque étape est sautée si son résultat existe déjà : on peut relancer le script après une coupure.
# Les durées et tailles se règlent par variables d'environnement (valeurs minuscules pour un essai sur CPU) :
#     MINUTES_TRAIN=0.5 MINUTES_RL=0.5 CHECKPOINT_MINUTES=0.2 CASES_CURVE=3 WIDTH_CURVE=128 CASES_FINAL=3 \
#         WIDTH_FINAL=256 ARCH="--hidden 128 --width 64 --blocks 1" WALKS=50 DEVICE=cpu PYTHON=python \
#         BENCH_ARGS="--seconds 1 --archs 128x64x1 --widths 128" RUNS=/tmp/essai bash gpu_session.sh
set -euo pipefail

PYTHON=${PYTHON:-python3}
DEVICE=${DEVICE:-cuda}
MINUTES_TRAIN=${MINUTES_TRAIN:-90}      # par méthode, à budget égal
MINUTES_RL=${MINUTES_RL:-45}            # par variante RL, depuis le même modèle
CHECKPOINT_MINUTES=${CHECKPOINT_MINUTES:-10}
CASES_CURVE=${CASES_CURVE:-200}         # cas DeepCubeA par point de courbe
WIDTH_CURVE=${WIDTH_CURVE:-4096}
CASES_FINAL=${CASES_FINAL:-1000}
WIDTH_FINAL=${WIDTH_FINAL:-65536}
# la recherche par la valeur évalue les 12 enfants : largeur divisée par 12 pour un coût de recherche égal
WIDTH_FINAL_VALUE=${WIDTH_FINAL_VALUE:-$((WIDTH_FINAL / 12))}
ARCH=${ARCH:---hidden 5000 --width 1000 --blocks 4}   # architecture de DeepCubeA / Takano, la même pour tous
WALKS=${WALKS:-1000}
RUNS=${RUNS:-runs}
BENCH_ARGS=${BENCH_ARGS:---seconds 15 --archs 5000x1000x4,2000x1000x2 --widths 512,4096,65536}
TRAIN_EXTRA=${TRAIN_EXTRA:-}            # options ajoutées à tous les entraînements, par ex. --compile
RUN_Q=${RUN_Q:-0}                       # 1 : entraîne aussi la tête Q (Q-learning) au même budget, à la fin

mkdir -p "$RUNS"
step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }

train() {   # nom, options...
    local name=$1; shift
    if [ -f "$RUNS/$name/final.pt" ]; then echo "$name : déjà entraîné"; return; fi
    step "entraînement $name"
    "$PYTHON" src/train_torch.py --device "$DEVICE" --walks "$WALKS" --checkpoint-minutes "$CHECKPOINT_MINUTES" \
        --out-dir "$RUNS/$name" $TRAIN_EXTRA "$@" 2>&1 | tee "$RUNS/$name.log"
}

tag() { local t; t=$(echo "$@" | tr -d ' -'); echo "${t:+_$t}"; }

curve() {   # nom, options d'évaluation...
    local name=$1; shift
    local out="$RUNS/curves/$name$(tag "$@").jsonl"
    if [ -f "$out" ]; then echo "courbe $out : déjà faite"; return; fi
    step "courbe $name $*"
    mkdir -p "$RUNS/curves"
    "$PYTHON" src/eval_torch.py "$RUNS/$name" --device "$DEVICE" --cases "$CASES_CURVE" --width "$WIDTH_CURVE" \
        --out "$out.tmp" "$@" && mv "$out.tmp" "$out"
}

final_eval() {   # nom, largeur, options d'évaluation...
    local name=$1 width=$2; shift 2
    local out="$RUNS/final/$name$(tag "$@")_w$width.jsonl"
    if [ -f "$out" ]; then echo "évaluation finale $out : déjà faite"; return; fi
    step "évaluation finale $name largeur $width $*"
    mkdir -p "$RUNS/final"
    "$PYTHON" src/eval_torch.py "$RUNS/$name/final.pt" --device "$DEVICE" --cases "$CASES_FINAL" \
        --width "$width" --out "$out.tmp" "$@" && mv "$out.tmp" "$out"
}

# 0. Vérifications et débits (quelques minutes)
step "tests"
"$PYTHON" -m pytest -q
if [ ! -f "$RUNS/bench.json" ]; then
    step "débits"
    "$PYTHON" src/bench_torch.py --device "$DEVICE" $BENCH_ARGS --out "$RUNS/bench.json"
fi

# 1. Trois méthodes au même budget, même architecture, même générateur de mélanges
train takano   --minutes "$MINUTES_TRAIN" $ARCH --heads policy --encoding stickers
train cayleypy --minutes "$MINUTES_TRAIN" $ARCH --heads value  --encoding stickers
train notre    --minutes "$MINUTES_TRAIN" $ARCH --heads policy --encoding pieces --canonical

# 2. Courbes qualité / temps d'entraînement sur les checkpoints
curve takano
curve cayleypy
curve notre
curve notre --bidirectional

# 3. Variantes RL depuis le même modèle (notre), à temps égal
train rl_temoin --minutes "$MINUTES_RL" --init "$RUNS/notre/final.pt"
train rl_per    --minutes "$MINUTES_RL" --init "$RUNS/notre/final.pt" --per
if [ ! -f "$RUNS/rl_exit/final.pt" ]; then
    step "entraînement rl_exit"
    "$PYTHON" src/expert_iteration_torch.py --device "$DEVICE" --walks "$WALKS" \
        --checkpoint-minutes "$CHECKPOINT_MINUTES" --minutes "$MINUTES_RL" --init "$RUNS/notre/final.pt" \
        --out-dir "$RUNS/rl_exit" 2>&1 | tee "$RUNS/rl_exit.log"
fi
curve rl_temoin
curve rl_per
curve rl_exit

# 4. Évaluation finale, faisceau large, jeu complet, à coût de recherche égal
final_eval takano    "$WIDTH_FINAL"
final_eval cayleypy  "$WIDTH_FINAL_VALUE"
final_eval notre     "$WIDTH_FINAL"
final_eval notre     "$WIDTH_FINAL" --bidirectional
final_eval rl_temoin "$WIDTH_FINAL"
final_eval rl_per    "$WIDTH_FINAL"
final_eval rl_exit   "$WIDTH_FINAL"

# 5. (Optionnel) Q-learning : même budget et même architecture, recherche guidée par la tête Q
if [ "$RUN_Q" = 1 ]; then
    train q_learning --minutes "$MINUTES_TRAIN" $ARCH --heads q --encoding pieces --canonical
    curve q_learning
    final_eval q_learning "$WIDTH_FINAL"
fi
step "session terminée"
