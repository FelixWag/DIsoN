import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import logging

from src.utils.logging_utils import update_wandb_metrics_client
from src.utils.model_utils import get_optimizer

def evaluate_binary_classification_model(dataloader, model, device, threshold=0.5):
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print('Using device:', device)
    model = model.to(device)

    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    correct_predictions_per_class = {0: 0, 1: 0}
    total_samples_per_class = {0: 0, 1: 0}
    ood_score = 0.0
    max_id_scores = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            _, outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            predicted = (torch.sigmoid(outputs) >= threshold).float()
            correct_predictions += (predicted == labels).sum().item()
            for label in [0, 1]:
                correct_predictions_per_class[label] += ((predicted == labels) & (labels == label)).sum().item()
                total_samples_per_class[label] += (labels == label).sum().item()

            for output, label in zip(torch.sigmoid(outputs), labels):
                if label.item() == 1:
                    ood_score = output.item()
                    print(f'OOD Sample Score: {output.item():.4f}')
                else:
                    max_id_scores.append(output.item())

    total_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / len(dataloader.dataset)
    print(f'Total samples for ID: {total_samples_per_class[0]}; Total samples for Target: {total_samples_per_class[1]}')
    print(f'Correct samples for ID: {correct_predictions_per_class[0]}; Correct samples for Target: {correct_predictions_per_class[1]}')
    accuracy_per_class = {label: correct_predictions_per_class[label] / total_samples_per_class[label] if total_samples_per_class[label] != 0 else 0. for label in [0, 1]}

    max_id_score = max(max_id_scores) if max_id_scores else None
    return total_loss, accuracy, accuracy_per_class, ood_score, max_id_score


class Node:
    """
    Represents a node in the decentralized learning network.
    Each node maintains its own model and trains on local data.
    """
    def __init__(self, model: torch.nn.Module, name: str, device, cfg,
                 train_dataloader, val_dataloader, coordinator, optimized_val_dataloader=None):
        self._model = model
        self._name = name
        self._device = device
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        
        optimizer_class, optimizer_params = get_optimizer(cfg)
        self._optimizer = optimizer_class(self._model.parameters(), **optimizer_params)
        
        self._num_local_iterations = cfg.decentralized.num_local_iterations

        self._coordinator = coordinator

        self.dataset_length = len(self._train_dataloader.dataset)
        logging.info(f'Dataset length of node {self._name}: {self.dataset_length}')

        self.embedding = None
        self.aggregated_mean = None
        self.aggregated_std = None

        self._model.to(device)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def model_weights(self) -> dict:
        return self._model.state_dict()

    @property
    def name(self) -> str:
        return self._name

    @property
    def coordinator(self):
        return self._coordinator

    def get_dataloader(self, train_test_val: str):
        if train_test_val == 'train':
            return self._train_dataloader
        elif train_test_val == 'val':
            return self._val_dataloader
        else:
            raise ValueError(
                f'Invalid input: train_test_val must be either train, val or test but was {train_test_val}.')

    def update_model(self, new_model, strict=True) -> None:
        if isinstance(new_model, nn.Module):
            self._model.load_state_dict(new_model.state_dict(), strict=strict)
        elif isinstance(new_model, dict):
            self._model.load_state_dict(new_model, strict=strict)
        else:
            raise ValueError('Invalid input: new_model must be an instance of nn.Module or a state dictionary.')


    def train(self, wandb_metrics: dict, cfg, method: str = 'default', proxy_model=None, mu=None, embedding=None, lambda_reg=0.0, epoch=None):
        if method == 'default':
            loss = local_train(train_dataloader=self.get_dataloader('train'), model=self._model,
                               num_local_iterations=self._num_local_iterations, optimizer=self._optimizer, device=self._device,
                               node_name=self.name, cfg=cfg)
        elif method == 'regularization':
            loss = local_train_w_regularization(train_dataloader=self.get_dataloader('train'), model=self._model,
                               num_local_iterations=self._num_local_iterations, optimizer=self._optimizer,
                               device=self._device, lambda_reg=lambda_reg)
        elif method == 'regularization_source':
            if self.name == 'source':
                loss = local_train_w_regularization(train_dataloader=self.get_dataloader('train'), model=self._model,
                               num_local_iterations=self._num_local_iterations, optimizer=self._optimizer,
                               device=self._device, lambda_reg=lambda_reg)
            elif self.name == 'target':
                loss = local_train(train_dataloader=self.get_dataloader('train'), model=self._model,
                                   num_local_iterations=self._num_local_iterations, optimizer=self._optimizer, device=self._device, cfg=cfg)
            else:
                raise ValueError('Unsupported type of node')
        elif method == 'synthetic':
            if self.name == 'source':
                if self.aggregated_mean is None:
                    _ = local_train(train_dataloader=self.get_dataloader('train'), model=self._model,
                                    num_local_iterations=2500, optimizer=self._optimizer,
                                    device=self._device, cfg=cfg)
                    self.coordinator.return_node('target').update_model(new_model=self._model)
                loss, aggregated_mean, aggregated_std = local_train_aggregate_synthetic(train_dataloader=self.get_dataloader('train'), model=self._model,
                                                    num_local_iterations=self._num_local_iterations,
                                                    optimizer=self._optimizer,
                                                    device=self._device)
                self.aggregated_mean = aggregated_mean
                self.aggregated_std = aggregated_std
            elif self.name == 'target':
                loss = local_train_with_synthetic_features(train_dataloader=self.get_dataloader('train'), model=self._model,
                                                    aggregated_mean=self._coordinator.return_node('source').aggregated_mean,
                                                    aggregated_std=self._coordinator.return_node('source').aggregated_std,
                                                    num_local_iterations=self._num_local_iterations,
                                                    optimizer=self._optimizer,
                                                    device=self._device)
            else:
                raise ValueError(f"Unsupported node: {self.name}")
        elif method == 'regularization_prototype':
            if self.name == 'source':
                loss = local_train_w_regularization(train_dataloader=self.get_dataloader('train'), model=self._model,
                                                    num_local_iterations=self._num_local_iterations,
                                                    optimizer=self._optimizer,
                                                    device=self._device, lambda_reg=lambda_reg)
                self.embedding = self.compute_mean_embedding()
            elif self.name == 'target':
                loss = local_train_prototype(train_dataloader=self.get_dataloader('train'), model=self._model,
                                             num_local_iterations=self._num_local_iterations, optimizer=self._optimizer,
                                             device=self._device,
                                             prototype=self._coordinator.return_node('source').embedding)
            else:
                raise ValueError(f"Unsupported training method: {method}")
        elif method == 'prototype':
            if self._name == 'source':
                loss = local_train(train_dataloader=self.get_dataloader('train'), model=self._model,
                                   num_local_iterations=self._num_local_iterations, optimizer=self._optimizer,
                                   device=self._device)
                self.embedding = self.compute_mean_embedding()
            elif self._name == 'target':
                loss = local_train_prototype(train_dataloader=self.get_dataloader('train'), model=self._model,
                                            num_local_iterations=self._num_local_iterations, optimizer=self._optimizer,
                                            device=self._device, prototype=self._coordinator.return_node('source').embedding)
            else:
                raise ValueError(f"Unsupported training method: {method}")
        elif method == 'fedprox':
            loss = local_train_prox(train_dataloader=self.get_dataloader('train'), model=self._model,
                                    num_local_iterations=self._num_local_iterations, optimizer=self._optimizer,
                                    proxy_model=proxy_model, mu=mu, device=self._device)
        else:
            raise ValueError(f"Unsupported training method: {method}")

        return loss

    def evaluate(self, train_test_val, step, wandb_metrics):
        logging.info('-' * 50 + '\n' + f'Evaluate node: {self.name} on step {step}')
        loss, accuracy, accuracy_per_class = evaluate_model(dataloader=self.get_dataloader('val'), model=self.model,
                                                            device=self._device)

        logging.info(f'Node {self.name} - {train_test_val} loss: {loss:.2f}' + '\n' + '-' * 50)

        update_wandb_metrics_client(wandb_metrics=wandb_metrics, train_val_test=train_test_val, client_name=self.name,
                                    loss=loss, acc=accuracy)

        return loss, accuracy, accuracy_per_class

    def compute_mean_embedding(self):
        self.model.eval()
        embedding_sum = 0
        total_samples = 0
        dataloader = self.get_dataloader('train')
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc='Computing mean embedding', unit='batch'):
                images = images.to(self._device)
                embeddings = self.model.penult_feature(images)
                embedding_sum += embeddings.sum(dim=0)
                total_samples += embeddings.size(0)

        mean_embedding = embedding_sum / total_samples
        return mean_embedding.cpu().numpy()

def local_train(train_dataloader, model, num_local_iterations, optimizer, device, cfg, node_name):
    criterion = torch.nn.BCEWithLogitsLoss()
    model = model.to(device)
    model.train()
    iterations_loss = []

    data_iter = iter(train_dataloader)

    for _ in range(num_local_iterations):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            images, labels = next(data_iter)

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        _, outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        
        if cfg.training.grad_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        iterations_loss.append(loss.item())

    return np.mean(iterations_loss)

def evaluate_model(dataloader, model, device, threshold=0.5):
    criterion = torch.nn.BCEWithLogitsLoss()
    model = model.to(device)
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    correct_predictions_per_class = {0: 0, 1: 0}
    total_samples_per_class = {0: 0, 1: 0}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            predicted = (torch.sigmoid(outputs) >= threshold).float()
            correct_predictions += (predicted == labels).sum().item()
            for label in [0, 1]:
                correct_predictions_per_class[label] += ((predicted == labels) & (labels == label)).sum().item()
                total_samples_per_class[label] += (labels == label).sum().item()

            for output, label in zip(torch.sigmoid(outputs), labels):
                ood_sample_score = None
                if label.item() == 1:
                    ood_sample_score = output.item()

    print(f'OOD Sample Score: {ood_sample_score}')
    total_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / len(dataloader.dataset)
    logging.info(f'Total samples for ID: {total_samples_per_class[0]}; Total samples for Target: {total_samples_per_class[1]}')
    logging.info(f'Correct samples for ID: {correct_predictions_per_class[0]}; Correct samples for Target: {correct_predictions_per_class[1]}')
    accuracy_per_class = {label: (correct_predictions_per_class[label] / total_samples_per_class[label]) if total_samples_per_class[label] > 0. else 0. for label in [0, 1]}

    return total_loss, accuracy, accuracy_per_class


