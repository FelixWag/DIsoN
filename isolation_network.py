import copy
import random
import torch
import numpy as np
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb
import logging
import sys
import os
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataloader.medical_dataset import MedicalDataset, get_transform
from src.decentralized.node import Node
from src.decentralized.coordinator import Coordinator
from src.networks.pre_trained_resnet import PretrainedResNet18
from src.utils.subsampling_utils import predict_class_from_image, filter_dataset_by_class

def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logFormatter = logging.Formatter('[%(asctime)s] %(message)s')
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(consoleHandler)

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(model_type, ood_settings, cfg, device, change_last_layer=True):
    set_seed(1)
    if model_type == 'resnet18_pretrained':
        model = PretrainedResNet18(pretrained_path=cfg.dataset.pretrained_model_path, grayscale=cfg.ood_settings.grayscale, device=device,
                                   small_resolution=cfg.dataset.small_resolution, num_classes=cfg.dataset.num_classes, change_last_layer=change_last_layer,
                                   use_instance_norm=(cfg.model.normalization.type == 'instance'))
        model.freeze_backbone(up_to_layer=cfg.training.up_to_layer)
        model.print_trainable_parameters()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def create_dataloader(no_support_device_file, pacemaker_file,
                      transform, num_no_support_device_samples, num_pacemaker_samples, num_workers, cfg,
                      batch_size=16, shuffle=True, seed=42, val_transform=None,
                      exclude_target_transform=False, only_target_samples=False, i=0):
    dataset = MedicalDataset(root_data_path=cfg.data.data_root,
                             no_support_device_file=no_support_device_file,
                             pacemaker_file=pacemaker_file,
                             num_no_support_device_samples=num_no_support_device_samples,
                             num_pacemaker_samples=num_pacemaker_samples,
                             transform=transform, seed=seed, val_transform=val_transform,
                             exclude_target_transform=exclude_target_transform, i=i)

    if only_target_samples:
        target_indices = dataset.get_target_indices()
        target_subset = Subset(dataset, target_indices)
        dataloader = DataLoader(target_subset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return dataloader

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader

def evaluate_binary_classification_model(dataloader, model, device, threshold=0.5):
    # Define loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Move model to GPU if available
    print('Using device:', device)
    model = model.to(device)

    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    correct_predictions_per_class = {0: 0, 1: 0}
    total_samples_per_class = {0: 0, 1: 0}
    ood_score = 0.0
    max_id_scores = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are of shape (batch_size, 1)

            _, outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate loss
            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) >= threshold).float()
            correct_predictions += (predicted == labels).sum().item()
            for label in [0, 1]:
                correct_predictions_per_class[label] += ((predicted == labels) & (labels == label)).sum().item()
                total_samples_per_class[label] += (labels == label).sum().item()

            # Print the score of the OOD sample with label 1
            for output, label in zip(torch.sigmoid(outputs), labels):
                if label.item() == 1:
                    ood_score = output.item()
                    print(f'OOD Sample Score: {output.item():.4f}')
                else:
                    max_id_scores.append(output.item())

    total_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / len(dataloader.dataset)
    print(f'Total samples for ID: {total_samples_per_class[0]}; Total samples for Pacemaker: {total_samples_per_class[1]}')
    print(f'Correct samples for ID: {correct_predictions_per_class[0]}; Correct samples for Pacemaker: {correct_predictions_per_class[1]}')
    accuracy_per_class = {label: correct_predictions_per_class[label] / total_samples_per_class[label] if total_samples_per_class[label] != 0 else 0. for label in [0, 1]}


    max_id_score = max(max_id_scores) if max_id_scores else None
    return total_loss, accuracy, accuracy_per_class, ood_score, max_id_score

def get_decentralized_dataloaders(batch_size, no_support_device_file, pacemaker_file, num_no_support_device_samples, train_transform,
                                  val_transform, seed, num_workers, data_root, i, target_class=None, label_file=None, cfg=None):
    source_train_dataset = MedicalDataset(root_data_path=data_root,
                                          no_support_device_file=no_support_device_file,
                                          pacemaker_file=pacemaker_file,
                                          num_no_support_device_samples=num_no_support_device_samples,
                                          num_pacemaker_samples=0,
                                          transform=train_transform, seed=seed)

    source_val_dataset = MedicalDataset(root_data_path=data_root,
                                        no_support_device_file=no_support_device_file,
                                        pacemaker_file=pacemaker_file,
                                        num_no_support_device_samples=num_no_support_device_samples,
                                        num_pacemaker_samples=0,
                                        transform=val_transform, seed=seed)
    
    if target_class is not None:
        source_train_dataset = filter_dataset_by_class(source_train_dataset, target_class, label_file, data_root, cfg.dataset.id_name)
        source_val_dataset = filter_dataset_by_class(source_val_dataset, target_class, label_file, data_root, cfg.dataset.id_name)

    target_train_dataset = MedicalDataset(root_data_path=data_root,
                                          no_support_device_file=no_support_device_file,
                                          pacemaker_file=pacemaker_file,
                                          num_no_support_device_samples=0,
                                          num_pacemaker_samples=1,
                                          duplicate_pacemaker_samples=num_no_support_device_samples,
                                          transform=train_transform, seed=seed, i=i)

    target_val_dataset = MedicalDataset(root_data_path=data_root,
                                        no_support_device_file=no_support_device_file,
                                        pacemaker_file=pacemaker_file,
                                        num_no_support_device_samples=0,
                                        num_pacemaker_samples=1,
                                        duplicate_pacemaker_samples=num_no_support_device_samples,
                                        transform=val_transform, seed=seed, i=i)

    source_train_dataloader = DataLoader(source_train_dataset, batch_size=batch_size - 1, shuffle=True, num_workers=num_workers)
    source_val_dataloader = DataLoader(source_val_dataset, batch_size=batch_size - 1, shuffle=False, num_workers=num_workers)
    target_train_dataloader = DataLoader(target_train_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    target_val_dataloader = DataLoader(target_val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return source_train_dataloader, source_val_dataloader, target_train_dataloader, target_val_dataloader

def setup_decentralized(model_type, batch_size, num_no_support_device_samples, seed,
                        device, num_workers, ood_settings, cfg, id_data_file, target_data_file, optimized_val_dataloader, i):
    node_names = ['source', 'target']
    model = get_model(model_type=model_type, ood_settings=ood_settings, cfg=cfg, device=device)
    
    if cfg.ood_settings.subsample:
        if model_type == 'resnet18_pretrained_intermediate':
            pre_trained_model = get_model(model_type='resnet18_pretrained', ood_settings=ood_settings, cfg=cfg, device=device,
                                          change_last_layer=False)
        else:
             pre_trained_model = get_model(model_type=model_type, ood_settings=ood_settings, cfg=cfg, device=device, change_last_layer=False)
        _, target_class, max_probability = predict_class_from_image(optimized_val_dataloader, pre_trained_model, device,
                                                                    cfg.dataset.id_name)
        if max_probability < cfg.ood_settings.subsample_threshold:
            target_class = None
    else:
        target_class = None
    
    coordinator = Coordinator(global_model=model, node_names_training_enabled=node_names, aggregation_method=cfg.decentralized.aggregation_method,
                              device=device, training_algorithm=cfg.decentralized.training_algorithm, target_node_name=None)

    train_transform, val_transform = get_transform(setting=cfg.dataset.transform)
    
    use_label_file = hasattr(cfg.dataset, 'label_file') and cfg.dataset.label_file is not None
    
    (source_train_dataloader, source_val_dataloader,
     target_train_dataloader, target_val_dataloader) = get_decentralized_dataloaders(batch_size=batch_size,
                                                                                      no_support_device_file=id_data_file,
                                                                                      pacemaker_file=target_data_file,
                                                                                      num_no_support_device_samples=num_no_support_device_samples,
                                                                                      train_transform=train_transform,
                                                                                      val_transform=val_transform,
                                                                                      seed=seed, num_workers=num_workers,
                                                                                      data_root=cfg.data.data_root,
                                                                                      i=i, 
                                                                                      target_class=target_class,
                                                                                      label_file=cfg.dataset.label_file if use_label_file else None,
                                                                                      cfg=cfg)

    for node_name in node_names:
        node_model = copy.deepcopy(coordinator.global_model)
        if node_name == 'source':
            train_dataloader = source_train_dataloader
            val_dataloader = source_val_dataloader
        else:
            train_dataloader = target_train_dataloader
            val_dataloader = target_val_dataloader
        node = Node(model=node_model, name=node_name, device=device, train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader, coordinator=coordinator, cfg=cfg, optimized_val_dataloader=optimized_val_dataloader)
        coordinator.add_node(node)

    return coordinator

def train_decentralized_model(coordinator, num_communication_rounds, device, optimized_val_dataloader, sample_id, cfg, threshold=0.5, lambda_reg=0.0):
    wandb_metrics = {}
    consecutive_stopping_criteria = 0

    for com_round in range(num_communication_rounds):
        train_losses = coordinator.train(wandb_metrics=wandb_metrics, cfg=cfg, communication_round=com_round, lambda_reg=lambda_reg)
        step = com_round + 1
        
        if coordinator.aggregation_method == 'avg_bn':
                val_loss, val_accuracy, val_accuracy_per_class, ood_score, max_id_score = evaluate_binary_classification_model(dataloader=optimized_val_dataloader, model=coordinator.target_node.model,
                                                                                                                        threshold=threshold, device=device)
        else:
            val_loss, val_accuracy, val_accuracy_per_class, ood_score, max_id_score = evaluate_binary_classification_model(dataloader=optimized_val_dataloader, model=coordinator.global_model,
                                                                                                                        threshold=threshold, device=device)

        wandb.log({f"train/train_loss_{sample_id}_source_node": train_losses['source'],
                   f"train/train_loss_{sample_id}_target_node": train_losses['target'],
                   f"val/val_ood_score_{sample_id}": ood_score,
                   f"val/val_max_id_score_{sample_id}": max_id_score,
                   f"global_step": step})

        if ood_score >= 0.65:
            consecutive_stopping_criteria += 1
        else:
            consecutive_stopping_criteria = 0

        if consecutive_stopping_criteria >= 5:
            if val_accuracy_per_class[0] >= 0.85:
                print(f'Stopping early as target sample correctly classified for 5 consecutive epochs at epoch {step}')
                return coordinator.global_model, step

    print(f'Stopping at iteration {num_communication_rounds}')
    return coordinator.global_model, num_communication_rounds

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg : DictConfig) -> None:
    setup_logging()
    logging.info(f'Running in decentralized mode: {cfg.training.decentralized}')
    logging.info(f'Using GPU: {cfg.general.gpu}')
    
    device = torch.device(f'cuda:{cfg.general.gpu}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    num_epochs_list_OOD = []
    num_epochs_list_ID = []

    EXPERIMENT_NAME = cfg.general.experiment_name
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    output_dir = os.path.join("output", cfg.general.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if cfg.wandb.group:
        wandb.init(project=cfg.wandb.project_name, name=EXPERIMENT_NAME, group=cfg.general.experiment_name, config=config_dict)
    else:
        wandb.init(project=cfg.wandb.project_name, name=EXPERIMENT_NAME, config=config_dict)

    ID_seeds = list(range(cfg.data.num_target_id_samples[0], cfg.data.num_target_id_samples[1]))
    OOD_seeds = list(range(cfg.data.num_target_ood_samples[0], cfg.data.num_target_ood_samples[1]))

    for mode in [cfg.ood_settings.id_or_ood]:
        seeds = ID_seeds if mode == 'ID' else OOD_seeds
        for seed, i in zip(seeds, seeds):
            set_seed(seed)

            no_support_device_file = cfg.dataset.id_data_paths
            if mode == 'OOD':
                pacemaker_file = cfg.dataset.target_data_ood_paths
            else:
                pacemaker_file = cfg.dataset.target_data_id_paths

            NUM_NO_SUPPORT_DEVICE_SAMPLES = cfg.ood_settings.num_no_support_device_samples
            NUM_PACEMAKER_SAMPLES = 1
            BATCH_SIZE = cfg.training.batch_size
            SEED = seed

            logging.info(f"Starting the experiment with the following config:\n {OmegaConf.to_yaml(cfg)}")
            train_transform, val_transform = get_transform(setting=cfg.dataset.transform)

            optimized_val_dataloader = create_dataloader(no_support_device_file=no_support_device_file, pacemaker_file=pacemaker_file,
                                                         transform=val_transform, num_no_support_device_samples=NUM_NO_SUPPORT_DEVICE_SAMPLES,
                                                         num_pacemaker_samples=NUM_PACEMAKER_SAMPLES, batch_size=BATCH_SIZE,
                                                         shuffle=False, seed=SEED, num_workers=cfg.general.num_workers,
                                                         only_target_samples=True, cfg=cfg, i=i)

            coordinator = setup_decentralized(model_type=cfg.model.model_type, batch_size=cfg.training.batch_size,
                                               num_no_support_device_samples=cfg.ood_settings.num_no_support_device_samples,
                                               seed=seed, device=device, num_workers=cfg.general.num_workers, ood_settings=cfg.ood_settings, cfg=cfg,
                                               id_data_file=no_support_device_file, target_data_file=pacemaker_file, optimized_val_dataloader=optimized_val_dataloader, i=i)

            model, num_epochs = train_decentralized_model(coordinator, num_communication_rounds=cfg.decentralized.num_communication_rounds, device=device,
                                                           sample_id=str(seed)+'_'+mode, optimized_val_dataloader=optimized_val_dataloader,
                                                           cfg=cfg, lambda_reg=cfg.regularizer.lambda_reg)

            wandb.log({f"num_epochs_{mode}": num_epochs})

            if mode == 'OOD':
                num_epochs_list_OOD.append(num_epochs)
            else:
                num_epochs_list_ID.append(num_epochs)

    if cfg.ood_settings.id_or_ood == "ID":
        selected_epochs = num_epochs_list_ID
    elif cfg.ood_settings.id_or_ood == "OOD":
        selected_epochs = num_epochs_list_OOD
    else:
        raise ValueError("Invalid ood_settings.id_or_ood value. Expected 'ID' or 'OOD'.")

    id_or_ood_label = cfg.ood_settings.id_or_ood
    if id_or_ood_label == "ID":
        sample_range = cfg.data.num_target_id_samples
    elif id_or_ood_label == "OOD":
        sample_range = cfg.data.num_target_ood_samples
    else:
        raise ValueError("Invalid ood_settings.id_or_ood value. Expected 'ID' or 'OOD'.")

    summary_file = os.path.join(output_dir, f"{EXPERIMENT_NAME}_{id_or_ood_label}_{sample_range[0]}-{sample_range[1]}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Epoch numbers for {cfg.ood_settings.id_or_ood} experiments:\n")
        f.write(f"{selected_epochs}\n")

    logging.info(f"Summary saved to {summary_file}")
    wandb.finish()

if __name__ == "__main__":
    main()