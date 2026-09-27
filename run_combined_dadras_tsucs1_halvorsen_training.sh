#!/bin/bash

set -e

# Put this file in:
#   $HOME/sh_files/run_combined_dadras_tsucs1_halvorsen_training.sh
#
# Expected server project folders:
#   $HOME/training_data/single_new_regimes_const
#   $HOME/training_data/single_new_regimes_const/dadras_e_49_to_64/values_8
#   $HOME/training_data/single_new_regimes_const/tsucs1_e_20_to_245/values_8
#   $HOME/training_data/single_halvorsen_constparams/14to209_noisy
#   $HOME/DynaMix_context_phi
#   $HOME/DynaMix-python-b-tipping
#   $HOME/diffrentbutconst_prams
#
# First run on the server:
# sed -i 's/\r$//' ~/sh_files/run_combined_dadras_tsucs1_halvorsen_training.sh
# chmod +x ~/sh_files/run_combined_dadras_tsucs1_halvorsen_training.sh
#
# Usage:
# ~/sh_files/run_combined_dadras_tsucs1_halvorsen_training.sh
# ~/sh_files/run_combined_dadras_tsucs1_halvorsen_training.sh 2
# GPU_ID=3 ~/sh_files/run_combined_dadras_tsucs1_halvorsen_training.sh

ROOT_DIR="${ROOT_DIR:-$HOME}"
CONTEXT_REPO="$ROOT_DIR/DynaMix_context_phi"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
TOOLS_DIR="$ROOT_DIR/diffrentbutconst_prams"
DATA_ROOT="$ROOT_DIR/training_data/single_new_regimes_const"
USUAL="${USUAL:-$ROOT_DIR/training_data/single_halvorsen_constparams}"
HALVORSEN_DIR="${HALVORSEN_DIR:-$USUAL/14to209_noisy}"

source "$BTIP_REPO/venv/bin/activate"

GPU_ID=${1:-${GPU_ID:-0}}
echo "Using GPU_ID=$GPU_ID"

latent_dim=30
experts=10
slides_context_steps=2000
rmse_steps=500

save_folder="results/combined_dadras_tsucs1_halvorsen_values8"
combined_data_dir="$ROOT_DIR/training_data/combined/dadras_tsucs1_halvorsen_values8"
run_name="dadras_tsucs1_halvorsen_values8"

dataset_dir() {
  local regime="$1"
  local variant="$2"
  local candidates=(
    "$DATA_ROOT/$regime/$variant"
    "$DATA_ROOT/windows/$regime/$variant"
    "$DATA_ROOT/$regime/windows/$variant"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "$candidate/data.npy" ]; then
      echo "$candidate"
      return 0
    fi
  done

  echo "Could not find dataset for $regime / $variant" >&2
  printf 'Tried:\n' >&2
  printf '  %s\n' "${candidates[@]}" >&2
  return 1
}

require_dataset() {
  local data_dir="$1"
  local label="$2"
  local required=(
    "data.npy"
    "context.npy"
    "test.npy"
    "phi.npy"
    "context_phi.npy"
    "test_phi.npy"
  )

  local file
  for file in "${required[@]}"; do
    if [ ! -f "$data_dir/$file" ]; then
      echo "Missing $file for $label in $data_dir" >&2
      return 1
    fi
  done
}

build_combined_dataset() {
  local dadras_dir="$1"
  local tsucs1_dir="$2"
  local halvorsen_dir="$3"
  local output_dir="$4"

  mkdir -p "$output_dir"

  DADRAS_DIR="$dadras_dir" \
  TSUCS1_DIR="$tsucs1_dir" \
  HALVORSEN_DIR="$halvorsen_dir" \
  OUTPUT_DIR="$output_dir" \
  python - <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np

sources = [
    ("dadras_e_49_to_64_values_8", Path(os.environ["DADRAS_DIR"])),
    ("tsucs1_e_20_to_245_values_8", Path(os.environ["TSUCS1_DIR"])),
    ("halvorsen_14to209_noisy_windows", Path(os.environ["HALVORSEN_DIR"])),
]
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

required_files = [
    "data.npy",
    "context.npy",
    "test.npy",
    "phi.npy",
    "context_phi.npy",
    "test_phi.npy",
]

summary = {"sources": [], "files": {}}
for label, source in sources:
    summary["sources"].append({"label": label, "path": str(source)})
    for filename in required_files:
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {filename} for {label}: {path}")

for filename in required_files:
    arrays = []
    shapes = []
    dtypes = []
    for label, source in sources:
        arr = np.load(source / filename)
        arrays.append(arr)
        shapes.append((label, arr.shape))
        dtypes.append((label, str(arr.dtype)))

    tail_shapes = {shape[1:] for _, shape in shapes}
    if len(tail_shapes) != 1:
        lines = "\n".join(f"  {label}: {shape}" for label, shape in shapes)
        raise ValueError(
            f"Cannot concatenate {filename}: shapes differ after axis 0:\n{lines}"
        )

    combined = np.concatenate(arrays, axis=0)
    np.save(output_dir / filename, combined)
    summary["files"][filename] = {
        "shape": list(combined.shape),
        "dtype": str(combined.dtype),
        "source_shapes": {label: list(shape) for label, shape in shapes},
        "source_dtypes": {label: dtype for label, dtype in dtypes},
    }
    print(f"Wrote {output_dir / filename}: {combined.shape} {combined.dtype}")

(output_dir / "metadata_combined.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
PY
}

mark_btip_config_as_expert_phi() {
  local config_path="$1/config.json"
  python - "$config_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text(encoding="utf-8"))
config["use_expert_phi"] = True
path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY
}

train_no_phi() {
  local repo="$1"
  local base="$2"
  local save_path="$3"

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
    --save_path "$save_path"
}

train_with_phi() {
  local repo="$1"
  local base="$2"
  local save_path="$3"

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
    --save_path "$save_path"
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

DADRAS_DIR="$(dataset_dir "dadras_e_49_to_64" "values_8")"
TSUCS1_DIR="$(dataset_dir "tsucs1_e_20_to_245" "values_8")"

require_dataset "$DADRAS_DIR" "Dadras"
require_dataset "$TSUCS1_DIR" "Tsucs1"
require_dataset "$HALVORSEN_DIR" "Halvorsen"

echo "==============================================="
echo "Dadras:    $DADRAS_DIR"
echo "Tsucs1:    $TSUCS1_DIR"
echo "Halvorsen: $HALVORSEN_DIR"
echo "Combined:  $combined_data_dir"
echo "==============================================="

echo "Building combined training dataset..."
build_combined_dataset "$DADRAS_DIR" "$TSUCS1_DIR" "$HALVORSEN_DIR" "$combined_data_dir"

CONTEXT_NO_PHI_RUN="$CONTEXT_REPO/$save_folder/${run_name}_context_no_phi"
CONTEXT_PHI_RUN="$CONTEXT_REPO/$save_folder/${run_name}_context_phi"
BTIP_NO_PHI_RUN="$BTIP_REPO/$save_folder/${run_name}_btip_no_phi"
BTIP_PHI_RUN="$BTIP_REPO/$save_folder/${run_name}_btip_cphi"

echo "==============================================="
echo "Training on combined dataset"
echo "Run name: $run_name"
echo "==============================================="

echo "Training context repo WITHOUT phi/Cphi..."
train_no_phi "$CONTEXT_REPO" "$combined_data_dir" "$save_folder/${run_name}_context_no_phi"

echo "Training context repo WITH context_phi, WITHOUT expert Cphi..."
train_with_phi "$CONTEXT_REPO" "$combined_data_dir" "$save_folder/${run_name}_context_phi"

echo "Training b-tipping repo WITHOUT phi/Cphi..."
train_no_phi "$BTIP_REPO" "$combined_data_dir" "$save_folder/${run_name}_btip_no_phi"

echo "Training b-tipping repo WITH context_phi and expert Cphi..."
train_with_phi "$BTIP_REPO" "$combined_data_dir" "$save_folder/${run_name}_btip_cphi"
mark_btip_config_as_expert_phi "$BTIP_PHI_RUN"

SLIDE_DATASETS=(
  "dadras:$DADRAS_DIR"
  "tsucs1:$TSUCS1_DIR"
  "halvorsen:$HALVORSEN_DIR"
)

for ITEM in "${SLIDE_DATASETS[@]}"; do
  LABEL="${ITEM%%:*}"
  DATA_DIR="${ITEM#*:}"

  echo "==============================================="
  echo "Creating slides on $LABEL"
  echo "Data: $DATA_DIR"
  echo "==============================================="

  make_pair_slides \
    "$DATA_DIR" \
    "$CONTEXT_REPO" \
    "$CONTEXT_REPO" \
    "$CONTEXT_REPO" \
    "$CONTEXT_PHI_RUN" \
    "$CONTEXT_NO_PHI_RUN" \
    "$CONTEXT_REPO/$save_folder/slides/${LABEL}_${run_name}_context_phi_vs_context_no_phi"

  make_pair_slides \
    "$DATA_DIR" \
    "$BTIP_REPO" \
    "$BTIP_REPO" \
    "$BTIP_REPO" \
    "$BTIP_PHI_RUN" \
    "$BTIP_NO_PHI_RUN" \
    "$BTIP_REPO/$save_folder/slides/${LABEL}_${run_name}_btip_cphi_vs_btip_no_phi"

  make_pair_slides \
    "$DATA_DIR" \
    "$BTIP_REPO" \
    "$BTIP_REPO" \
    "$CONTEXT_REPO" \
    "$BTIP_PHI_RUN" \
    "$CONTEXT_PHI_RUN" \
    "$BTIP_REPO/$save_folder/slides/${LABEL}_${run_name}_btip_cphi_vs_context_phi"
done

echo "==============================================="
echo "Combined training and all per-dataset slides finished."
echo "==============================================="
