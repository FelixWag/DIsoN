import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import numpy as np
import copy

from src.networks.pre_trained_resnet import PretrainedResNet18


def extract_features(model, dataloader, device):
    # Define the nodes from which you want to extract features
    return_nodes = {'avgpool': 'features'}

    # Copy the model (to prevent modifying the original model)
    model_copy = copy.deepcopy(model)

    # Create a feature extractor
    feature_extractor = create_feature_extractor(model_copy, return_nodes=return_nodes)
    feature_extractor.eval()
    feature_extractor.to(device)

    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            
            # Extract features
            outputs = feature_extractor(images)['features']
            outputs = torch.flatten(outputs, 1)  # Flatten the output for the fully connected layer

            features.append(outputs.cpu().detach().numpy())
            labels.extend(lbls.cpu().detach().numpy())

    features = np.concatenate(features)
    labels = np.array(labels)
    return features, labels

def get_optimizer(cfg):
    if cfg.training.optimizer == 'adam':
        optimizer_class = torch.optim.Adam
        optimizer_params = {'lr': cfg.training.learning_rate, 'betas': (0.9, 0.999), 'weight_decay': cfg.ood_settings.weight_decay} #, 'weight_decay': 5e-4}
    elif cfg.training.optimizer == 'adamw':
        optimizer_class = torch.optim.AdamW
        optimizer_params = {'lr': cfg.training.learning_rate, 'weight_decay': cfg.ood_settings.weight_decay}
    elif cfg.training.optimizer == 'sgd':
        optimizer_class = torch.optim.SGD
        optimizer_params = {'lr': cfg.training.learning_rate, 'weight_decay': cfg.ood_settings.weight_decay, 'nesterov': True, 'momentum': 0.9}
    else:
        raise ValueError(f'Optimizer {cfg.optimizer.optimizer} not supported')

    return optimizer_class, optimizer_params


def get_normalization_layer_names(state_dict, model):
    # For each new architecture we need to specify the corresponding layer names
    if isinstance(model, torchvision.models.densenet.DenseNet):
        return [k for k in list(state_dict.keys()) if 'norm' in k]
    elif isinstance(model, torchvision.models.resnet.ResNet):
        return [k for k in list(state_dict.keys()) if 'bn' in k]
    elif isinstance(model, PretrainedResNet18):
        return [k for k in list(state_dict.keys()) if 'bn' in k]
    elif isinstance(model, ResNetPenultimate):
        return [k for k in list(state_dict.keys()) if 'bn' in k]
    else:
        raise ValueError(f"{type(model)} is not supported")