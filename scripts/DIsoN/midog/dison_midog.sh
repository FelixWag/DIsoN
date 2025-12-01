#!/bin/bash

# Ensure the script is called with two arguments:
# 1. id_or_ood, and
# 2. chunk (an integer like 1, 2, 3, etc.)
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <ID|OOD> <chunk_number (e.g., 1,2,3,...)>"
  exit 1
fi

id_or_ood="$1"
chunk="$2"

# Verify that chunk is numeric.
if ! [[ "$chunk" =~ ^[0-9]+$ ]]; then
  echo "Error: chunk_number must be an integer."
  exit 1
fi

# Define the number of samples
num_samples=1896

# Define the base experiment name
base_experiment_name="MIDOG_2"

# Create a unique experiment name by appending ID and OOD sample numbers
experiment_name=$(echo "$base_experiment_name" | sed "s/{Num_samples}/$num_samples/")_ID_${num_samples}_OOD_1

# Compute sample range based on chunk:
# For chunk 1: [50,162]
# For chunk 2: [162,274]
# For chunk 3: [274,386]
# For chunk 4: [386,500]
if [ "$chunk" -eq 1 ]; then
  sample_range="[50,162]"
elif [ "$chunk" -eq 2 ]; then
  sample_range="[162,274]"
elif [ "$chunk" -eq 3 ]; then
  sample_range="[274,386]"
elif [ "$chunk" -eq 4 ]; then
  sample_range="[386,500]"
else
  echo "Error: Only chunks 1-4 are supported with predefined ranges."
  exit 1
fi

echo "Using sample range: $sample_range"

python3 isolation_network.py \
    general.num_workers=8 \
    dataset=midog_midog_2 \
    ood_settings.num_no_support_device_samples="$num_samples" \
    general.experiment_name="$experiment_name" \
    dataset.transform=midog_basic \
    ood_settings.exclude_target_transform=true \
    model.normalization.type="instance" \
    general.gpu=1 \
    model.model_type=resnet18_pretrained \
    training.batch_size=16 \
    ood_settings.minority_sampler=true \
    training.learning_rate=0.01 \
    wandb.group=true \
    wandb.project_name="DIsoN" \
    ood_settings.hidden_sizes="[]" \
    ood_settings.grayscale=false \
    training.up_to_layer=0 \
    training.optimizer=sgd \
    training.log_interval=5 \
    ood_settings.pacemaker_frequency=1 \
    data.num_target_id_samples="$sample_range" \
    data.num_target_ood_samples="$sample_range" \
    training.decentralized=true \
    decentralized.num_communication_rounds=100 \
    decentralized.num_local_iterations=50 \
    decentralized.training_algorithm="default" \
    regularizer.lambda_reg=0.1 \
    decentralized.aggregation_method='avg' \
    ood_settings.weight_decay=0.0 \
    ood_settings.id_or_ood="$id_or_ood" \
    data.data_root="PATH_TO_DATA" \
    ood_settings.subsample=true \
    ood_settings.subsample_threshold=0.00 \
    training.grad_clipping=true


