#!/bin/bash

set -e

# Put this file in:
#   $HOME/sh_files/run_new_regimes_const_slides_only.sh
#
# Expected project folders:
#   $HOME/training_data/single_new_regimes_const
#   $HOME/DynaMix_context_phi
#   $HOME/DynaMix-python-b-tipping
#   $HOME/diffrentbutconst_prams
#
# First run on the server:
# sed -i 's/\r$//' ~/sh_files/run_new_regimes_const_slides_only.sh
# chmod +x ~/sh_files/run_new_regimes_const_slides_only.sh
#
# Usage:
# ~/sh_files/run_new_regimes_const_slides_only.sh
# ~/sh_files/run_new_regimes_const_slides_only.sh 2
# GPU_ID=3 ~/sh_files/run_new_regimes_const_slides_only.sh

ROOT_DIR="${ROOT_DIR:-$HOME}"
CONTEXT_REPO="$ROOT_DIR/DynaMix_context_phi"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
TOOLS_DIR="$ROOT_DIR/diffrentbutconst_prams"
DATA_ROOT="$ROOT_DIR/training_data/single_new_regimes_const"

source "$BTIP_REPO/venv/bin/activate"

GPU_ID=${1:-${GPU_ID:-0}}
echo "Using GPU_ID=$GPU_ID"

slides_context_steps=2000
rmse_steps=500

context_save_folder="results/single_new_regimes_const"
btip_save_folder="results/single_new_regimes_const"

REGIMES=(
  # "rossler_a_005_to_035"
  "thomas_b_004_to_023"
  "dadras_e_49_to_64"
  # "burke_shaw_nu_15_to_30"
  # "tsucs1_e_18_to_20"
  # "tsucs1_e_20_to_245"
  "shimizu_morioka_a_085_to_113"
  "shimizu_morioka_b_05_to_105"
  "rikitake_mu_10_to_45"
)

TRAIN_VARIANTS=(
  "random_full"
  "random_phi_lt_0"
  "random_phi_gt_0"
)

dataset_dir() {
  local regime="$1"
  local variant="$2"
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

  echo "Could not find dataset for $regime / $variant" >&2
  echo "Tried: $with_windows and $flat" >&2
  return 1
}

make_pair_slides() {
  local data_dir="$1"
  local repo_root="$2"
  local with_repo_root="$3"
  local no_repo_root="$4"
  local with_run="$5"
  local no_run="$6"
  local output_dir="$7"

  cd "$TOOLS_DIR"
  python constant_parameter_stochastic_median_slides_server.py \
    --repo-root "$repo_root" \
    --with-phi-repo-root "$with_repo_root" \
    --no-phi-repo-root "$no_repo_root" \
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
  SLIDE_DATA="$(dataset_dir "$REGIME" "values_8")"

  for VARIANT in "${TRAIN_VARIANTS[@]}"; do
    RUN_NAME="${REGIME}_${VARIANT}"

    CONTEXT_NO_PHI_RUN="$CONTEXT_REPO/$context_save_folder/${RUN_NAME}_no_phi"
    CONTEXT_PHI_RUN="$CONTEXT_REPO/$context_save_folder/${RUN_NAME}_context_phi"
    BTIP_PHI_RUN="$BTIP_REPO/$btip_save_folder/${RUN_NAME}_b_tipping_phi"

    echo "==============================================="
    echo "Regime: $REGIME"
    echo "Train variant: $VARIANT"
    echo "Slide data: $SLIDE_DATA"
    echo "==============================================="

    make_pair_slides \
      "$SLIDE_DATA" \
      "$CONTEXT_REPO" \
      "$CONTEXT_REPO" \
      "$CONTEXT_REPO" \
      "$CONTEXT_PHI_RUN" \
      "$CONTEXT_NO_PHI_RUN" \
      "$CONTEXT_REPO/$context_save_folder/slides/${RUN_NAME}_context_phi_vs_no_phi"

    make_pair_slides \
      "$SLIDE_DATA" \
      "$CONTEXT_REPO" \
      "$BTIP_REPO" \
      "$CONTEXT_REPO" \
      "$BTIP_PHI_RUN" \
      "$CONTEXT_NO_PHI_RUN" \
      "$BTIP_REPO/$btip_save_folder/slides/${RUN_NAME}_btip_phi_vs_no_phi"

    make_pair_slides \
      "$SLIDE_DATA" \
      "$CONTEXT_REPO" \
      "$BTIP_REPO" \
      "$CONTEXT_REPO" \
      "$BTIP_PHI_RUN" \
      "$CONTEXT_PHI_RUN" \
      "$BTIP_REPO/$btip_save_folder/slides/${RUN_NAME}_btip_phi_vs_context_phi"

    echo "Finished slides for $RUN_NAME"
  done
done

echo "==============================================="
echo "All pairwise slides finished."
echo "==============================================="
