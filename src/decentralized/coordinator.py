import copy
import statistics

import torch
import torch.nn as nn

from src.decentralized.decentralized_algorithms import weighted_avg, weighted_avg_batch_norm
import logging

from src.utils.logging_utils import update_wandb_metrics


class Coordinator:
    """
    Coordinates training across multiple nodes in the decentralized learning network.
    Manages model aggregation and synchronization between nodes.
    """
    def __init__(self, global_model: torch.nn.Module, node_names_training_enabled: list[str],
                 aggregation_method: str, device, training_algorithm: str = 'default', target_node_name: str = None):
        self._global_model = global_model
        self.all_nodes = {}
        self.training_nodes = {}
        self._node_names_training_enabled = node_names_training_enabled
        self._training_algorithm = training_algorithm
        self._target_node_name = target_node_name
        self._aggregation_method = aggregation_method

        self._best_selection_metric = 0.0

        self.proxy_models = {}

        self._device = device
        self._global_model.to(device)

    @property
    def global_model(self) -> nn.Module:
        return self._global_model

    @property
    def global_model_weights(self) -> dict:
        return self._global_model.state_dict()

    @property
    def target_node(self) -> nn.Module:
        return self.all_nodes['target']

    @property
    def source_node(self) -> nn.Module:
        return self.all_nodes['source']

    def return_node(self, name: str):
        return self.training_nodes[name]

    @property
    def target_node_name(self) -> str:
        return self._target_node_name

    @property
    def aggregation_method(self) -> str:
        return self._aggregation_method

    @property
    def training_algorithm(self) -> str:
        return self._training_algorithm

    @training_algorithm.setter
    def training_algorithm(self, new_training_algorithm: str) -> None:
        self._training_algorithm = new_training_algorithm

    def update_model(self, new_model) -> None:
        if isinstance(new_model, nn.Module):
            self._global_model.load_state_dict(new_model.state_dict())
        elif isinstance(new_model, dict):
            self._global_model.load_state_dict(new_model)
        else:
            raise ValueError('Invalid input: new_model must be an instance of nn.Module or a state dictionary.')

    def add_node(self, node) -> None:
        assert node.name not in self.all_nodes
        self.all_nodes[node.name] = node
        if node.name in self._node_names_training_enabled:
            self.training_nodes[node.name] = node

    def train(self, wandb_metrics: dict, cfg, communication_round=0, lambda_reg=0.0) -> dict:
        losses = {}
        for _, node in self.training_nodes.items():
            loss = node.train(wandb_metrics=wandb_metrics, method=self._training_algorithm, lambda_reg=lambda_reg,
                                epoch=communication_round, cfg=cfg)
            losses[node.name]=loss
        
        with torch.no_grad():
            if self._aggregation_method == 'avg':
                if communication_round >= 2:
                    alpha = cfg.decentralized.alpha
                else:
                    alpha = cfg.decentralized.alpha
                w_avg = weighted_avg(coordinator=self, nodes=self.training_nodes, equal_weighting=False, alpha=alpha)
                for _, node in self.all_nodes.items():
                    node.update_model(copy.deepcopy(w_avg))
            elif self._aggregation_method == 'avg_bn':
                w_avg, w_avg_non_norm_params = weighted_avg_batch_norm(coordinator=self, nodes=self.training_nodes,
                                                        equal_weighting=False, alpha=0.2)
                for _, node in self.all_nodes.items():
                    node.update_model(w_avg_non_norm_params, strict=False)
            else:
                raise ValueError(f"Unsupported aggregation method: {self._aggregation_method}")
            self.update_model(copy.deepcopy(w_avg))
        return losses

    def evaluate(self, train_test_val, step, wandb_metrics) -> None:
        loss_all = []
        correct_all = 0
        total_all = 0

        node_accuracies = []

        for _, node in self.all_nodes.items():
            loss, accuracy, accuracy_per_class = node.evaluate(train_test_val=train_test_val,
                                                                 step=step, wandb_metrics=wandb_metrics)
            print(f'Node {node.name}, Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')
            print(f'Node {node.name}, Validation Accuracy per Class: {accuracy_per_class}')
            loss_all.append(loss)
            logging.info('[INFO] This is just the average of the accuracies of the nodes')
            node_accuracies.append(accuracy)

        mean_loss = statistics.fmean(loss_all)
        mean_acc = statistics.fmean(node_accuracies)

        logging.info(
            f'Evaluate coordinator on step {step}. Mean loss is: {mean_loss:.2f}, Mean accuracy is: {mean_acc:.2%}')

        update_wandb_metrics(wandb_metrics=wandb_metrics, train_val_test=train_test_val,
                             loss_all=mean_loss, acc_all_mean=mean_acc)

