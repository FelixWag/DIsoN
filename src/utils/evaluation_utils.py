import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score, roc_curve


def print_thresholds(ID_epochs, OOD_epochs, iteration=None):
    # Compute the mean of means and the mean of medians
    mean_of_means = np.mean([np.mean(ID_epochs), np.mean(OOD_epochs)])
    mean_of_medians = np.mean([np.median(ID_epochs), np.median(OOD_epochs)])
    # Combine data and labels
    epochs = np.array(ID_epochs + OOD_epochs)
    labels = np.array([0] * len(ID_epochs) + [1] * len(OOD_epochs))  # 0 for ID, 1 for OOD
    print(labels)
    # Possible thresholds (unique sorted epochs)
    # thresholds = sorted(set(epochs))
    thresholds = sorted(set(epochs) | {mean_of_means, mean_of_medians})

    # Initialize lists to store metrics
    threshold_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []

    # Loop over possible thresholds
    for T in thresholds:
        # Predictions based on threshold
        # Predict OOD (1) if epochs <= T, else ID (0)
        predictions = (epochs <= T).astype(int)

        # Compute TP, TN, FP, FN
        TP = np.sum((predictions == 1) & (labels == 1))
        TN = np.sum((predictions == 0) & (labels == 0))
        FP = np.sum((predictions == 1) & (labels == 0))
        FN = np.sum((predictions == 0) & (labels == 1))

        # Compute metrics
        accuracy = (TP + TN) / len(labels)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Store metrics
        threshold_list.append(T)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        tp_list.append(TP)
        tn_list.append(TN)
        fp_list.append(FP)
        fn_list.append(FN)

    # Create a DataFrame to display metrics
    metrics_df = pd.DataFrame({
        'Threshold': threshold_list,
        'Accuracy': accuracy_list,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1 Score': f1_list,
        'TP': tp_list,
        'TN': tn_list,
        'FP': fp_list,
        'FN': fn_list
    })

    # Compute AUROC and FPR@95
    try:
        auroc = roc_auc_score(labels, -epochs)

        # Compute ROC curve
        fpr, tpr, thresholds_roc = roc_curve(labels, -epochs)

        # Find the threshold where TPR >= 0.95
        idx = np.argmax(tpr >= 0.95)
        if tpr[idx] >= 0.95:
            fpr95 = fpr[idx]
        else:
            fpr95 = np.nan  # Handle case where TPR never reaches 0.95
    except Exception as e:
        print(f"Error computing AUROC and FPR@95: {e}")
        auroc = np.nan
        fpr95 = np.nan

    # Track the maximum F1 score to Weights & Biases if it is the last evaluation
    max_f1 = metrics_df['F1 Score'].max()
    max_accuracy = metrics_df['Accuracy'].max()

    # Compute class proportions
    num_class_0 = len(ID_epochs)
    num_class_1 = len(OOD_epochs)
    total = num_class_0 + num_class_1
    weight_0 = num_class_0 / total
    weight_1 = num_class_1 / total

    # Compute recall for each class
    recall_0 = (metrics_df['TN'] / (metrics_df['TN'] + metrics_df['FP'])).replace([np.inf, -np.inf], 0).fillna(0)
    recall_1 = metrics_df['Recall']

    # Compute weighted accuracy
    weighted_accuracy = (recall_0 * weight_0 + recall_1 * weight_1).max()

    # Display the table and additional metrics
    additional_metrics = (
        f"Max F1 Score: {max_f1:.4f}\n"
        f"Max Accuracy: {max_accuracy:.4f}\n"
        f"AUROC: {auroc:.4f}\n"
        f"Max Weighted Accuracy: {weighted_accuracy:.4f}"
    )

    # Format ID_epochs and OOD_epochs as strings
    id_epochs_str = "ID Epochs: " + ', '.join(map(str, ID_epochs))
    ood_epochs_str = "OOD Epochs: " + ', '.join(map(str, OOD_epochs))

    # Combine all parts into a single string
    full_output = (
        metrics_df.to_string(index=False) + "\n\n" +
        additional_metrics + "\n\n" +
        id_epochs_str + "\n" +
        ood_epochs_str
    )

    # Log metrics to Weights & Biases
    wandb.log({
        "f1": max_f1,
        "accuracy": max_accuracy,
        "auroc": auroc,
        "weighted_accuracy": weighted_accuracy,
        "fpr@95": fpr95
    }, step=iteration)

    return full_output