#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import wandb
import argparse
from ast import literal_eval
from sklearn.metrics import roc_auc_score, roc_curve

def print_thresholds_with_wandb(ID_epochs, OOD_epochs, folder_name, dataset_name):
    """
    Calculate metrics based on ID and OOD epochs and log them to Weights & Biases.
    
    Parameters:
        ID_epochs (list): List of epochs for in-distribution experiments
        OOD_epochs (list): List of epochs for out-of-distribution experiments
        folder_name (str): Name of the folder containing the experiments
        dataset_name (str): Name of the dataset being evaluated
        
    Returns:
        str: Formatted string with metric results
    """
    # Compute the mean of means and the mean of medians
    mean_of_means = np.mean([np.mean(ID_epochs), np.mean(OOD_epochs)])
    mean_of_medians = np.mean([np.median(ID_epochs), np.median(OOD_epochs)])
    
    # Combine data and labels
    epochs = np.array(ID_epochs + OOD_epochs)
    labels = np.array([0] * len(ID_epochs) + [1] * len(OOD_epochs))  # 0 for ID, 1 for OOD
    
    # Possible thresholds (unique sorted epochs)
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

    # Find maximum F1 score and accuracy
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

    # Log metrics to W&B
    wandb.log({
        "folder": folder_name,
        "dataset": dataset_name,
        "num_ID_experiments": num_class_0,
        "num_OOD_experiments": num_class_1,
        "AUROC": auroc,
        "FPR@95": fpr95,
        "weighted_accuracy": weighted_accuracy,
        "max_f1": max_f1,
        "max_accuracy": max_accuracy
    })

    # Display the table and additional metrics
    additional_metrics = (
        f"Max F1 Score: {max_f1:.4f}\n"
        f"Max Accuracy: {max_accuracy:.4f}\n"
        f"AUROC: {auroc:.4f}\n"
        f"FPR@95: {fpr95:.4f}\n"
        f"Max Weighted Accuracy: {weighted_accuracy:.4f}\n"
        f"Num ID experiments: {num_class_0}\n"
        f"Num OOD experiments: {num_class_1}"
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

    return full_output

def extract_experiment_epochs(folder_path):
    """
    Given a folder path, this function scans for .txt files within the folder.
    
    Each file is expected to have the following format:
      Line 1: "Epoch numbers for ID experiments:" OR "Epoch numbers for OOD experiments:"
      Line 2: A string representation of a list, e.g., "[15, 7]"

    The function extracts epochs from each file and returns two lists:
      - id_experiments: A list of epoch lists from files related to ID experiments.
      - ood_experiments: A list of epoch lists from files related to OOD experiments.
    
    Parameters:
        folder_path (str): The path to the folder containing the .txt files.
    
    Returns:
        tuple: (id_experiments, ood_experiments)
    """
    id_experiments = []
    ood_experiments = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Process only .txt files
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                # Read the header and the list line
                header = file.readline().strip()
                list_str = file.readline().strip()
                
                try:
                    # Safely evaluate the list string to a Python list
                    epochs = literal_eval(list_str)
                    if not isinstance(epochs, list):
                        print(f"Warning: The parsed content in {filename} is not a list.")
                        continue
                except Exception as e:
                    print(f"Error parsing epochs from {filename}: {e}")
                    continue

                # Determine if the file is for ID or OOD experiments and add accordingly
                if header.lower().startswith("epoch numbers for id experiments"):
                    id_experiments.append(epochs)
                elif header.lower().startswith("epoch numbers for ood experiments"):
                    ood_experiments.append(epochs)
                else:
                    print(f"Warning: Unrecognized header in {filename}: {header}")
    
    flat_id_experiments = [item for sublist in id_experiments for item in sublist]
    flat_ood_experiments = [item for sublist in ood_experiments for item in sublist]

    return flat_id_experiments, flat_ood_experiments

def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate FedOOD experiments and log to W&B.')
    parser.add_argument('--folder', type=str, required=True, 
                        help='Path to the folder containing experiment data')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Name of the dataset being evaluated')
    parser.add_argument('--wandb_project', type=str, default='DIsoN', 
                        help='W&B project name')
    args = parser.parse_args()
    
    # Initialize W&B
    wandb.init(project=args.wandb_project, name=f"{os.path.basename(args.folder)}_{args.dataset}")
    
    # Extract experiment epochs
    id_epochs, ood_epochs = extract_experiment_epochs(args.folder)
    print(f"Found {len(id_epochs)} ID epochs and {len(ood_epochs)} OOD epochs")
    
    # Calculate metrics and log to W&B
    folder_name = os.path.basename(args.folder)
    results = print_thresholds_with_wandb(id_epochs, ood_epochs, folder_name, args.dataset)
    
    # Print results
    print(results)
    
    # Finish W&B run
    wandb.finish()
    
if __name__ == "__main__":
    main() 