#!/bin/bash

set -e

# Put this file in:
#   $HOME/sh_files/run_new_regimes_const_values8_selectable_models.sh
#
# Expected project folders directly in $HOME:
#   training_data/single_new_regimes_const
#   DynaMix_context_phi
#   DynaMix-python-b-tipping
#   DynaMix_diagonal_coupling
#   DynaMix_polynomial
#   diffrentbutconst_prams
#
# First run on the server:
# sed -i 's/\r$//' ~/sh_files/run_new_regimes_const_values8_selectable_models.sh
# chmod +x ~/sh_files/run_new_regimes_const_values8_selectable_models.sh
#
# Usage:
# ~/sh_files/run_new_regimes_const_values8_selectable_models.sh
# ~/sh_files/run_new_regimes_const_values8_selectable_models.sh 2
# GPU_ID=3 ~/sh_files/run_new_regimes_const_values8_selectable_models.sh

ROOT_DIR="${ROOT_DIR:-$HOME}"
CONTEXT_REPO="$ROOT_DIR/DynaMix_context_phi"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
DIAGONAL_REPO="$ROOT_DIR/DynaMix_diagonal_coupling"
POLYNOMIAL_REPO="$ROOT_DIR/DynaMix_polynomial"
TOOLS_DIR="$ROOT_DIR/diffrentbutconst_prams"
DATA_ROOT="$ROOT_DIR/training_data/single_new_regimes_const"

source "$BTIP_REPO/venv/bin/activate"

GPU_ID=${1:-${GPU_ID:-0}}
echo "Using GPU_ID=$GPU_ID"

latent_dim=30
experts=10
slides_context_steps=2000
rmse_steps=500
save_folder="results/single_new_regimes_const_values8_selectable"
baseline_save_folder="results/single_new_regimes_const_values8"

# Baseline for all slides. Set to 0 if this vanilla run already exists and
# should only be reused for comparisons.
TRAIN_VANILLA=0

# Select phi-conditioned models to train and compare against vanilla.
# Comment out entries you do not want to run.
TRAIN_MODELS=(
  # "context_phi"
  # "b_tipping"
  "diagonal"
  # "polynomial"
)

REGIMES=(
  # "rossler_a_005_to_035"
  # "thomas_b_004_to_023"
  # "dadras_e_49_to_64"
  # "burke_shaw_nu_15_to_30"
  # "tsucs1_e_18_to_20"
  # "tsucs1_e_20_to_245"
  "shimizu_morioka_a_085_to_113"
  # "shimizu_morioka_b_05_to_105"
  "rikitake_mu_10_to_45"
)

dataset_dir() {
  local regime="$1"
  local variant="values_8"
  local with_windows="$DATA_ROOT/$regime/windows/$variant"
  local flat="$DATA_ROOT/$regime/$variant"

  if [ -f "$with_windows/data.npy" ]; then
    echo "$with_windows"
    return 0
  fi
  if [ -f "$flat/data.npy" ]; then
    echo "$flat"
    return 0
  fi

  echo "Could not find values_8 dataset for $regime" >&2
  echo "Tried: $with_windows and $flat" >&2
  return 1
}

model_repo() {
  case "$1" in
    vanilla) echo "$CONTEXT_REPO" ;;
    context_phi) echo "$CONTEXT_REPO" ;;
    b_tipping) echo "$BTIP_REPO" ;;
    diagonal) echo "$DIAGONAL_REPO" ;;
    polynomial) echo "$POLYNOMIAL_REPO" ;;
    *) echo "Unknown model key: $1" >&2; return 1 ;;
  esac
}

model_suffix() {
  case "$1" in
    vanilla) echo "no_phi" ;;
    context_phi) echo "context_phi" ;;
    b_tipping) echo "b_tipping_phi" ;;
    diagonal) echo "diagonal_phi" ;;
    polynomial) echo "polynomial_phi" ;;
    *) echo "Unknown model key: $1" >&2; return 1 ;;
  esac
}

run_dir_for() {
  local model_key="$1"
  local run_name="$2"
  local repo
  local suffix
  repo="$(model_repo "$model_key")"
  suffix="$(model_suffix "$model_key")"
  if [ "$model_key" = "vanilla" ]; then
    echo "$repo/$baseline_save_folder/${run_name}_${suffix}"
    return 0
  fi
  echo "$repo/$save_folder/${run_name}_${suffix}"
}

train_vanilla() {
  local base="$1"
  local run_name="$2"
  local repo="$BTIP_REPO"
  local suffix
  suffix="$(model_suffix vanilla)"

  echo "Training vanilla/no-phi baseline in context repo..."
  cd "$repo"
  python -m src.dynamix.training.training_setup \
    --data_path "$base/data.npy" \
    --context_path "$base/context.npy" \
    --test_path "$base/test.npy" \
    --device cuda \
    --gpu_id "$GPU_ID" \
    --threads 4 \
    --latent_dim "$latent_dim" \
    --experts "$experts" \
    --pwl_units 2 \
    --batch_size 8 \
    --batches_per_epoch 20 \
    --epochs 500 \
    --ssi 25 \
    --noise_level 0.02 \
    --save_path "$baseline_save_folder/${run_name}_${suffix}"
}

train_phi_model() {
  local model_key="$1"
  local base="$2"
  local run_name="$3"
  local repo
  local suffix
  repo="$(model_repo "$model_key")"
  suffix="$(model_suffix "$model_key")"

  echo "Training $model_key model..."
  cd "$repo"
  python -m src.dynamix.training.training_setup \
    --data_path "$base/data.npy" \
    --context_path "$base/context.npy" \
    --test_path "$base/test.npy" \
    --phi_path "$base/phi.npy" \
    --context_phi_path "$base/context_phi.npy" \
    --test_phi_path "$base/test_phi.npy" \
    --device cuda \
    --gpu_id "$GPU_ID" \
    --threads 4 \
    --latent_dim "$latent_dim" \
    --experts "$experts" \
    --pwl_units 2 \
    --batch_size 8 \
    --batches_per_epoch 20 \
    --epochs 500 \
    --ssi 25 \
    --noise_level 0.02 \
    --save_path "$save_folder/${run_name}_${suffix}"
}

make_pair_slides() {
  local data_dir="$1"
  local with_model_key="$2"
  local no_model_key="$3"
  local with_run="$4"
  local no_run="$5"
  local output_dir="$6"
  local with_repo
  local no_repo

  with_repo="$(model_repo "$with_model_key")"
  no_repo="$(model_repo "$no_model_key")"

  cd "$TOOLS_DIR"
  python constant_parameter_stochastic_median_slides_server.py \
    --repo-root "$with_repo" \
    --with-phi-repo-root "$with_repo" \
    --no-phi-repo-root "$no_repo" \
    --data-dir "$data_dir" \
    --with-phi-run "$with_run" \
    --no-phi-run "$no_run" \
    --output-dir "$output_dir" \
    --device cuda \
    --gpu-id "$GPU_ID" \
    --context-steps "$slides_context_steps" \
    --rmse-steps "$rmse_steps" \
    --nonpositive-phi-metric rmse \
    --positive-phi-metric rmse \
    --torch-threads 4
}

for REGIME in "${REGIMES[@]}"; do
  BASE="$(dataset_dir "$REGIME")"
  RUN_NAME="${REGIME}_values_8"
  VANILLA_RUN="$(run_dir_for vanilla "$RUN_NAME")"

  echo "==============================================="
  echo "Regime:  $REGIME"
  echo "Dataset: $BASE"
  echo "Vanilla baseline: $VANILLA_RUN"
  echo "Models: ${TRAIN_MODELS[*]}"
  echo "==============================================="

  if [ "$TRAIN_VANILLA" = "1" ]; then
    train_vanilla "$BASE" "$RUN_NAME"
  elif [ ! -f "$VANILLA_RUN/config.json" ]; then
    echo "TRAIN_VANILLA=0 but missing vanilla baseline:"
    echo "$VANILLA_RUN/config.json"
    echo "Set TRAIN_VANILLA=1 or check save_folder/RUN_NAME."
    exit 1
  else
    echo "Reusing existing vanilla baseline."
  fi

  for MODEL_KEY in "${TRAIN_MODELS[@]}"; do
    MODEL_RUN="$(run_dir_for "$MODEL_KEY" "$RUN_NAME")"
    MODEL_REPO="$(model_repo "$MODEL_KEY")"
    MODEL_SUFFIX="$(model_suffix "$MODEL_KEY")"

    train_phi_model "$MODEL_KEY" "$BASE" "$RUN_NAME"

    echo "Creating slides: $MODEL_KEY vs vanilla..."
    make_pair_slides \
      "$BASE" \
      "$MODEL_KEY" \
      "vanilla" \
      "$MODEL_RUN" \
      "$VANILLA_RUN" \
      "$MODEL_REPO/$save_folder/slides/${RUN_NAME}_${MODEL_SUFFIX}_vs_vanilla"

    echo "Finished $REGIME / $MODEL_KEY"
  done
done

echo "==============================================="
echo "Selected model trainings and slides finished."
echo "==============================================="
