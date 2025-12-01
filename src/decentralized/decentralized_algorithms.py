import torch
import copy

from src.utils.model_utils import get_normalization_layer_names


def weighted_avg(coordinator, nodes, alpha, equal_weighting=False):
    """
    Weighted average aggregation for decentralized learning.
    
    Args:
        coordinator: Coordinator managing the nodes
        nodes: Dictionary of nodes participating in training
        alpha: Weight for target node (1-alpha for source node)
        equal_weighting: If True, use equal weights; if False, use alpha-based weighting
        
    Returns:
        Aggregated model weights
    """
    with torch.no_grad():
        w_avg = copy.deepcopy(coordinator.global_model_weights)
        total_datapoints = sum(node.dataset_length for _, node in nodes.items())

        for k in coordinator.global_model_weights.keys():
            if equal_weighting:
                w_avg[k] = sum(n.model_weights[k] for _, n in nodes.items()) / len(nodes)
            else:
                w_avg[k] = torch.zeros_like(w_avg[k])
                for _, n in nodes.items():
                    if 'num_batches_tracked' in k:
                        w_avg[k] = n.model_weights[k]
                    else:
                        if n.name == 'source':
                            w_avg[k] += (1-alpha) * n.model_weights[k]
                        else:
                            w_avg[k] += alpha * n.model_weights[k]

        coordinator.update_model(new_model=w_avg)
        return w_avg


def weighted_avg_batch_norm(coordinator, nodes, alpha, equal_weighting=False):
    """
    Weighted average aggregation with batch normalization handling.
    
    Args:
        coordinator: Coordinator managing the nodes
        nodes: Dictionary of nodes participating in training
        alpha: Weight for target node
        equal_weighting: If True, use equal weights
        
    Returns:
        Tuple of (full aggregated weights, weights without normalization layers)
    """
    w_avg = weighted_avg(coordinator=coordinator, nodes=nodes, equal_weighting=equal_weighting, alpha=alpha)
    w_avg_non_norm_params = copy.deepcopy(w_avg)
    norm_layer_names = get_normalization_layer_names(state_dict=w_avg_non_norm_params, model=coordinator.global_model)
    for key in norm_layer_names:
        del w_avg_non_norm_params[key]
    return w_avg, w_avg_non_norm_params


