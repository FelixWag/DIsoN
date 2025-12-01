#!/bin/bash

# Ensure the script is called with two arguments:
# 1. id_or_ood, and
# 2. chunk (an integer like 1, 2, 3, etc.)
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <ID|OOD> <chunk_number (e.g., 1,2,3,...)>"
  exit 1
fi

num_samples=469

# Define the base experiment name
base_experiment_name="ULTRASOUND"


id_or_ood="$1"
split="$2"

# Set the correct split based on the provided split number
case $split in
  1)
    id_split="[0,14]"
    ood_split="[0,57]"
    ;;
  2)
    id_split="[14,28]"
    ood_split="[57,114]"
    ;;
  3)
    id_split="[28,42]"
    ood_split="[114,171]"
    ;;
  4)
    id_split="[42,56]"
    ood_split="[171,228]"
    ;;
  5)
    id_split="[0,10]"
    ood_split="[0,10]"
    ;;
  *)
    echo "Invalid split number: $split. Must be 1, 2, 3, 4 or 5."
    exit 1
    ;;
esac

# Create experiment name
experiment_name=$(echo "$base_experiment_name" | sed "s/{Num_samples}/$num_samples/")_ALPHA_0.2_ID_${num_samples}_OOD_1

python3 isolation_network.py \
    general.num_workers=8 \
    dataset=breastmnist \
    ood_settings.num_no_support_device_samples="$num_samples" \
    general.experiment_name="$experiment_name" \
    dataset.transform=breastmnist_basic \
    ood_settings.exclude_target_transform=true \
    model.normalization.type="instance" \
    general.gpu=0 \
    model.model_type=resnet18_pretrained \
    training.batch_size=8 \
    ood_settings.minority_sampler=true \
    training.learning_rate=0.001 \
    wandb.group=true \
    wandb.project_name="DIsoN" \
    ood_settings.hidden_sizes="[]" \
    ood_settings.grayscale=true \
    training.up_to_layer=0 \
    training.optimizer=adam \
    training.log_interval=5 \
    ood_settings.pacemaker_frequency=1 \
    data.data_root="PATH_TO_DATA" \
    data.num_target_id_samples="$id_split" \
    data.num_target_ood_samples="$ood_split" \
    training.decentralized=true \
    decentralized.num_communication_rounds=300 \
    decentralized.num_local_iterations=27 \
    decentralized.training_algorithm="default" \
    regularizer.lambda_reg=0.1 \
    decentralized.aggregation_method='avg' \
    ood_settings.weight_decay=0.0 \
    ood_settings.id_or_ood="$id_or_ood" \
    ood_settings.subsample=true \
    ood_settings.subsample_threshold=0.00 \
    decentralized.alpha=0.2 \
    training.grad_clipping=false


