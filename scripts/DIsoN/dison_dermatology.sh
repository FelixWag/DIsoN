#!/bin/bash

# Ensure the script is called with two arguments: ID/OOD and split number (1-4)
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <ID|OOD> <split_number (1-4)>"
  exit 1
fi

id_or_ood="$1"
split="$2"

# Set the correct split based on the provided split number
case $split in
  1)
    id_split="[0,47]"
    ood_split="[0,62]"
    ;;
  2)
    id_split="[47,94]"
    ood_split="[62,125]"
    ;;
  3)
    id_split="[94,142]"
    ood_split="[125,188]"
    ;;
  4)
    id_split="[105,142]"
    ood_split="[188,251]"
    ;;
  5)
    id_split="[0,10]"
    ood_split="[0,10]"
    ;;
  *)
    echo "Invalid split number: $split. Must be 1, 2, 3, or 4 or 5."
    exit 1
    ;;
esac

# Define the number of samples
num_samples=1261

# Define the base experiment name
base_experiment_name="Dermatology"

# Create a unique experiment name by appending ID and OOD sample numbers
experiment_name=$(echo "$base_experiment_name" | sed "s/{Num_samples}/$num_samples/")_ID_${num_samples}_OOD_1


python3 isolation_network.py \
    general.num_workers=8 \
    dataset=skin_lesion \
    ood_settings.num_no_support_device_samples="$num_samples" \
    general.experiment_name="$experiment_name" \
    dataset.transform=skin_lesion_basic \
    ood_settings.exclude_target_transform=true \
    model.normalization.type="instance" \
    general.gpu=1 \
    model.model_type=resnet18_pretrained \
    training.batch_size=16 \
    ood_settings.minority_sampler=true \
    training.learning_rate=0.001 \
    wandb.group=true \
    wandb.project_name="DIsoN" \
    ood_settings.hidden_sizes="[]" \
    ood_settings.grayscale=false \
    training.optimizer=adam \
    training.up_to_layer=0 \
    training.log_interval=5 \
    ood_settings.pacemaker_frequency=1 \
    data.data_root="PATH_TO_DATA" \
    data.num_target_id_samples="$id_split" \
    data.num_target_ood_samples="$ood_split" \
    training.decentralized=true \
    decentralized.num_communication_rounds=300 \
    decentralized.num_local_iterations=37 \
    decentralized.training_algorithm="default" \
    regularizer.lambda_reg=0.1 \
    decentralized.aggregation_method="avg" \
    ood_settings.weight_decay=0.0 \
    ood_settings.id_or_ood="$id_or_ood" \
    ood_settings.subsample=true \
    ood_settings.subsample_threshold=0.00 \
    decentralized.alpha=0.2 \
    training.grad_clipping=false

