import torch
import torch.nn as nn
from torch.utils.data import Subset
from pathlib import Path
import datetime
import inspect

MIDOG_CLASSES = [0, 1, 2]

CHEXPERT_CLASSES = [0, 1]

SKIN_LESION_CLASSES = [0, 1]

BREASTMNIST_CLASSES = list(range(0, 3))

# Add a cache for label files
_label_file_cache = {}

def get_dataset_class_mapping(dataset_name):
    if dataset_name == 'midog':
        return MIDOG_CLASSES
    elif dataset_name == 'chexpert':
        return CHEXPERT_CLASSES
    elif dataset_name == 'skin_lesion':
        return SKIN_LESION_CLASSES
    elif dataset_name == 'breastmnist':
        return BREASTMNIST_CLASSES
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def predict_class_from_image(dataloader, model, device, dataset_name):
    """
    Predict the class of an image using a pretrained model.
    Supports both multi-class classification and binary classification with a single output neuron.
    
    Args:
        dataloader (DataLoader): DataLoader containing a single image
        model (nn.Module): Pretrained model
        device (torch.device): Device to run the model on
        dataset_name (str): Name of the dataset
        
    Returns:
        int: Predicted class index
        str: Class name
        float: Maximum probability
    """
    # Get the class mapping based on dataset name
    class_mapping = get_dataset_class_mapping(dataset_name)
    # Ensure class_mapping is a list for proper indexing
    if not isinstance(class_mapping, list):
        class_mapping = list(class_mapping)

    # Get the single image from dataloader
    image_tensor = next(iter(dataloader))[0].to(device)
    model.to(device)
    
    # Set model to eval mode and get prediction
    model.eval()
    with torch.no_grad():
        _, outputs = model(image_tensor)
        print(f'outputs: {outputs}')
        
        # Check if binary classification (single output neuron) or multi-class
        if outputs.shape[1] == 1:  # Binary classification with single output neuron
            # Apply sigmoid to get probability
            sigmoid_output = torch.sigmoid(outputs).item()
            
            # If probability > 0.5, predict class 1, else class 0
            class_idx = 1 if sigmoid_output > 0.5 else 0
            
            # Calculate probability for the predicted class
            max_probability = sigmoid_output if class_idx == 1 else (1 - sigmoid_output)
            
            # For display purposes
            binary_probs = [1.0 - sigmoid_output, sigmoid_output]
        else:  # Multi-class classification
            # Apply softmax to get class probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get the predicted class index
            max_prob_idx = torch.argmax(probabilities).item()
            class_idx = max_prob_idx
            
            # Get the probability for the predicted class
            max_probability = probabilities[int(max_prob_idx)].item()
            
            # For display purposes
            binary_probs = None
    
    # Get the class name
    if isinstance(class_idx, (int, float)):
        class_name = class_mapping[int(class_idx)]
    else:
        class_name = class_mapping[class_idx]
    
    print(f"Predicted class: {class_name} (index: {class_idx})")
    
    # Print probabilities for each class
    print("\nClass Probabilities:")
    print("-" * 50)
    print(f"{'Class Name':<20} {'Index':<8} {'Probability':<10}")
    print("-" * 50)
    
    # Display probabilities
    if outputs.shape[1] == 1:  # Binary classification
        # Display probabilities for binary classification
        for i in range(min(len(class_mapping), 2)):
            if binary_probs is not None:
                print(f"{class_mapping[i]:<20} {i:<8} {binary_probs[i]:.6f}")
            else:
                print(f"{class_mapping[i]:<20} {i:<8} {'N/A'}")
    else:  # Multi-class classification
        # Get top probabilities (up to 5)
        top_count = min(len(class_mapping), 5)
        top_probs, top_indices = torch.topk(probabilities, top_count)
        
        # Display top probabilities
        for i, (idx, prob) in enumerate(zip(top_indices.cpu().numpy(), top_probs.cpu().numpy())):
            print(f"{class_mapping[idx]:<20} {idx:<8} {prob:.6f}")
    
    print("-" * 50)
    if outputs.shape[1] == 1:
        print(f"Binary classification with single output neuron")
    else:
        print(f"Multi-class classification with {outputs.shape[1]} output neurons")
    
    return class_idx, class_name, max_probability

# This one is the new one
def predict_classes_from_image(dataloader, model, device, dataset_name, k=5):
    """
    Predict the class of an image using a pretrained model.
    
    Args:
        dataloader (DataLoader): DataLoader containing a single image
        model (nn.Module): Pretrained model
        device (torch.device): Device to run the model on
        k (int): Number of top predictions to return
        
    Returns:
        list: Top-k predicted class indices
        list: Top-k class names
        list: Top-k probabilities
    """
    # Get the class mapping based on dataset name
    class_mapping = get_dataset_class_mapping(dataset_name)

    # Get the single image from dataloader
    image_tensor = next(iter(dataloader))[0].to(device)
    model.to(device)
    
    # Set model to eval mode and get prediction
    model.eval()
    with torch.no_grad():
        _, outputs = model(image_tensor)
        print(f'outputs: {outputs}')
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get top-k probabilities and indices
    top_probs, top_indices = torch.topk(probabilities, k)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    # Get class names for top indices
    top_class_names = [class_mapping[idx] for idx in top_indices]
    
    # For backward compatibility, still show the top-1 prediction
    class_idx = top_indices[0]
    class_name = top_class_names[0]
    max_probability = top_probs[0]
    
    print(f"Predicted class: {class_name} (index: {class_idx})")
    
    # Print probabilities for each class
    print("\nClass Probabilities:")
    print("-" * 50)
    print(f"{'Class Name':<20} {'Index':<8} {'Probability':<10}")
    print("-" * 50)
    
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        print(f"{class_mapping[idx]:<20} {idx:<8} {prob:.6f}")
    
    print("-" * 50)
    print(f"Full probabilities tensor shape: {probabilities.shape}")
    
    return class_idx, top_class_names, max_probability
# END NEW ONE

def extract_class_from_path(file_path):
    """
    Extract the class name from a file path.
    
    Args:
        file_path (str or Path): Path to the file
        
    Returns:
        str: Class name
    """
    # Convert to string if it's a Path object
    path_str = str(file_path)
    
    # Split the path and get the second to last element
    parts = path_str.split('/')
    return parts[-2]

def extract_class_from_file(file_path, label_file, root_dir):
    """
    Extract the class from a file path using a label file.
    The label file should be in the format: path label
    Example: 1a/044/044_1060_2.tiff 2
    Note: Paths can contain spaces, the label is always the last whitespace-separated value
    
    Args:
        file_path (str or Path): Path to the file
        label_file (str): Path to the label file
        root_dir (str): Root directory to prepend to paths
        
    Returns:
        str: Class label
    """
    # Convert to string if it's a Path object
    path_str = str(file_path)
    
    # Check if we have the label file in cache
    if label_file not in _label_file_cache:
        # Read the label file and store in cache
        label_dict = {}
        reversed_dict = {}  # Map full paths to labels for quick lookup
        
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split on last whitespace to handle paths with spaces
                last_space_idx = line.rfind(' ')
                if last_space_idx == -1:
                    continue
                path = line[:last_space_idx]
                label = line[last_space_idx + 1:]
                # Prepend root_dir to path
                full_path = str(Path(root_dir) / path)
                label_dict[path] = label
                reversed_dict[full_path] = label
        
        _label_file_cache[label_file] = {
            'dict': label_dict,
            'reversed': reversed_dict
        }
    
    # Get the label dictionary from cache
    cache_data = _label_file_cache[label_file]
    reversed_dict = cache_data['reversed']
    
    # Direct lookup first (most efficient)
    if path_str in reversed_dict:
        return int(reversed_dict[path_str])
    
    # If direct lookup fails, try to find a relative path match
    relative_path = str(Path(path_str).relative_to(root_dir) if root_dir else path_str)
    label_dict = cache_data['dict']
    if relative_path in label_dict:
        return int(label_dict[relative_path])
    
    # As a last resort, try the old substring matching approach
    for path in reversed_dict:
        if path in path_str or path_str.endswith(path):
            return int(reversed_dict[path])
    
    raise ValueError(f"Could not find class for file {file_path} in label file {label_file}")

def extract_class_from_path_or_file(file_path, label_file=None, root_dir=None):
    """
    Extract the class from a file path, either using the path structure or a label file.
    
    Args:
        file_path (str or Path): Path to the file
        label_file (str, optional): Path to the label file. If None, uses path-based extraction.
        
    Returns:
        str: Class name or label
    """
    if label_file is not None:
        return extract_class_from_file(file_path, label_file, root_dir)
    else:
        return extract_class_from_path(file_path)

def precompute_class_indices(dataset, label_file, root_dir=None, dataset_name=None):
    """
    Pre-computes indices for all classes in the dataset and saves them to files.
    This is more efficient than computing indices one class at a time.
    
    Args:
        dataset: The dataset to process
        label_file (str): Path to the label file
        root_dir (str, optional): Root directory to prepend to paths
        dataset_name (str): Name of the dataset
    """
    if not label_file:
        print("Error: label_file must be provided to precompute class indices")
        return

    # Get class mapping for the dataset
    class_mapping = get_dataset_class_mapping(dataset_name)
    
    # Create a dictionary to store indices for each class
    class_indices = {cls: [] for cls in class_mapping}
    
    print(f"Pre-computing indices for all classes in {dataset_name}...")
    # Iterate through the dataset only once
    total_items = len(dataset.data)
    update_interval = max(1, total_items // 100)  # Update progress every 1%
    
    for i, (path, _) in enumerate(dataset.data):
        try:
            class_id = extract_class_from_path_or_file(path, label_file, root_dir)
            if class_id in class_indices:
                class_indices[class_id].append(i)
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
            
        # Print progress update
        if i % update_interval == 0 or i == total_items - 1:
            progress = (i + 1) / total_items * 100
            print(f"Progress: {progress:.2f}% ({i+1}/{total_items})")
    
    # Create directory to save indices
    indices_dir = Path(label_file).parent / "class_indices"
    indices_dir.mkdir(exist_ok=True, parents=True)
    
    # Save indices for each class
    for class_id, indices in class_indices.items():
        if indices:  # Only save if there are indices for this class
            indices_file = indices_dir / f"class_{class_id}_indices.pt"
            torch.save(torch.tensor(indices), indices_file)
            print(f"Saved {len(indices)} indices for class {class_id} to {indices_file}")

    # Save metadata file to indicate that precomputation is complete
    meta_file = indices_dir / "precomputed_complete.txt"
    with open(meta_file, "w") as f:
        f.write(f"Precomputation completed at {datetime.datetime.now()}\n")
        f.write(f"Total classes: {len(class_mapping)}\n")
    
    print(f"Pre-computation complete. Saved indices for {len(class_mapping)} classes.")

def filter_dataset_by_class(dataset, target_class, label_file=None, root_dir=None, dataset_name=None, use_centroids=False, centroid_dir=None, centroid_percentage=0.7):
    """
    Filters the dataset to only include samples from the target class.
    
    Args:
        dataset: The dataset to filter
        target_class (str): The class to filter by
        label_file (str, optional): Path to the label file. If None, uses path-based extraction.
        root_dir (str, optional): Root directory to prepend to paths
        dataset_name (str, optional): Name of the dataset
        use_centroids (bool, optional): Whether to use centroid-based indices if available.
        centroid_dir (str, optional): Directory containing centroid indices. If None, uses label_file parent directory.
        centroid_percentage (float, optional): Percentage used when computing centroid indices (default: 0.7).
        
    Returns:
        Subset: A filtered dataset containing only samples from the target class
    """
    # Check if we should use centroid-based indices
    if use_centroids and centroid_dir is not None:
        # Try to load centroid indices
        centroid_indices = load_centroid_indices(target_class, centroid_dir, centroid_percentage)
        if centroid_indices:
            print(f"Using pre-computed centroid indices for class {target_class} ({len(centroid_indices)} samples)")
            return Subset(dataset, centroid_indices)
        else:
            print(f"No pre-computed centroid indices found for class {target_class}, falling back to standard filtering")
    
    # Check if precomputation is available
    if label_file is not None and dataset_name is not None:
        # Construct the path to the pre-computed indices file
        indices_dir = Path(label_file).parent / "class_indices"
        indices_file = indices_dir / f"class_{target_class}_indices.pt"
        meta_file = indices_dir / "precomputed_complete.txt"
        
        # Check if precomputation is needed
        if not meta_file.exists():
            print("Pre-computed indices not found. Running precomputation for all classes...")
            precompute_class_indices(dataset, label_file, root_dir, dataset_name)
        
        # Load the pre-computed indices
        if indices_file.exists():
            filtered_indices = torch.load(indices_file).tolist()
            print(f"Loaded {len(filtered_indices)} pre-computed indices for class {target_class}")
            return Subset(dataset, filtered_indices)
        else:
            print(f"Warning: No pre-computed indices found for class {target_class}")
    
    # Regular processing when precomputed indices aren't available
    filtered_indices = []
    
    for i, (path, _) in enumerate(dataset.data):
        if extract_class_from_path_or_file(path, label_file, root_dir) == target_class:
            filtered_indices.append(i)
    
    # If we don't have enough samples, print a warning
    if len(filtered_indices) < 10:  # Arbitrary threshold
        print(f"Warning: Only found {len(filtered_indices)} samples for class {target_class}")
    
    # Create a subset of the dataset with only the filtered indices
    return Subset(dataset, filtered_indices)

def filter_dataset_by_classes(dataset, target_classes, label_file=None, root_dir=None, dataset_name=None, use_centroids=False, centroid_dir=None, centroid_percentage=0.7):
    """
    Filters the dataset to only include samples from the target classes.
    
    Args:
        dataset: The dataset to filter
        target_classes (str or list): The class or classes to filter by
        label_file (str, optional): Path to the label file. If None, uses path-based extraction.
        root_dir (str, optional): Root directory to prepend to paths
        dataset_name (str, optional): Name of the dataset
        use_centroids (bool, optional): Whether to use centroid-based indices if available.
        centroid_dir (str, optional): Directory containing centroid indices. If None, uses label_file parent directory.
        centroid_percentage (float, optional): Percentage used when computing centroid indices (default: 0.7).
        
    Returns:
        Subset: A filtered dataset containing only samples from the target classes
    """
    # Handle single class case by converting to list
    if isinstance(target_classes, str):
        target_classes = [target_classes]
    
    # Check if we should use centroid-based indices
    if use_centroids and centroid_dir is not None:
        # Collect indices for all target classes
        all_centroid_indices = []
        found_centroid_indices = False
        
        for target_class in target_classes:
            # Try to load centroid indices
            class_indices = load_centroid_indices(target_class, centroid_dir, centroid_percentage)
            if class_indices:
                all_centroid_indices.extend(class_indices)
                found_centroid_indices = True
                print(f"Using pre-computed centroid indices for class {target_class} ({len(class_indices)} samples)")
        
        if found_centroid_indices:
            return Subset(dataset, all_centroid_indices)
        else:
            print("No pre-computed centroid indices found, falling back to standard filtering")
    
    # Check if precomputation is available
    if label_file is not None and dataset_name is not None:
        # Check if precomputation is needed
        indices_dir = Path(label_file).parent / "class_indices"
        meta_file = indices_dir / "precomputed_complete.txt"
        
        if not meta_file.exists():
            print("Pre-computed indices not found. Running precomputation for all classes...")
            precompute_class_indices(dataset, label_file, root_dir, dataset_name)
        
        # Collect indices for all target classes
        all_filtered_indices = []
        found_precomputed = False
        
        for target_class in target_classes:
            indices_file = indices_dir / f"class_{target_class}_indices.pt"
            
            if indices_file.exists():
                class_indices = torch.load(indices_file).tolist()
                all_filtered_indices.extend(class_indices)
                found_precomputed = True
                print(f"Loaded {len(class_indices)} pre-computed indices for class {target_class}")
        
        if found_precomputed:
            return Subset(dataset, all_filtered_indices)
    
    # Regular processing when precomputed indices aren't available
    filtered_indices = []
    
    for i, (path, _) in enumerate(dataset.data):
        if extract_class_from_path_or_file(path, label_file, root_dir) in target_classes:
            filtered_indices.append(i)
    
    # If we don't have enough samples, print a warning
    if len(filtered_indices) < 10:  # Arbitrary threshold
        print(f"Warning: Only found {len(filtered_indices)} samples for classes {target_classes}")
    
    # Create a subset of the dataset with only the filtered indices
    return Subset(dataset, filtered_indices)

def clear_label_file_cache():
    """
    Clear the label file cache. Useful if you want to reload label files.
    """
    _label_file_cache.clear()

def compute_centroid_based_subset(dataset, target_class, feature_extractor, device, percentage=0.7, save_dir=None):
    """
    Computes a subset of data by selecting examples closest to the class centroid.
    
    Args:
        dataset: The dataset to filter
        target_class: The class to filter by
        feature_extractor: Model to extract features from images
        device: Device to run the model on
        percentage: Percentage of closest samples to keep (default: 0.7)
        save_dir: Directory to save indices to (optional)
        
    Returns:
        Subset: A filtered dataset containing only the closest samples to the centroid
        list: Indices of the selected samples
    """
    # First filter to get only samples of the target class
    class_subset = filter_dataset_by_class(dataset, target_class)
    
    if len(class_subset) == 0:
        print(f"No samples found for class {target_class}")
        return Subset(dataset, []), []
    
    print(f"Computing centroid for class {target_class} with {len(class_subset)} samples")
    
    # Extract features for all samples
    features = []
    feature_extractor.eval()
    
    # Check if the model is ResNet18_32x32 which has a different forward method
    model_forward_params = inspect.signature(feature_extractor.forward).parameters
    uses_return_feature_param = 'return_feature' in model_forward_params
    
    with torch.no_grad():
        for idx in class_subset.indices:
            # Get the image
            img, _ = dataset[idx]
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img).float()
            
            # Ensure image has batch dimension and is on correct device
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            img = img.to(device)
            
            # Extract features based on model type
            if uses_return_feature_param:
                # For models with return_feature parameter (like some ResNet implementations)
                feature = feature_extractor(img, return_feature=True)[0]
            else:
                try:
                    # Try the standard implementation that returns features, logits
                    feature, _ = feature_extractor(img)
                except (ValueError, TypeError):
                    # If that fails, the model might just return the output
                    # We'll use the output before the final layer as features
                    # This is a fallback and might not work for all models
                    # Get the penultimate layer by removing the final layer
                    temp_model = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
                    feature = temp_model(img)
                    if len(feature.shape) > 2:
                        feature = torch.flatten(feature, 1)
            
            features.append(feature.cpu())
    
    # Stack all features
    features = torch.cat(features, dim=0)
    
    # Compute centroid
    centroid = torch.mean(features, dim=0, keepdim=True)
    
    # Compute distances from each sample to the centroid
    distances = torch.norm(features - centroid, dim=1)
    
    # Select the closest percentage of samples
    num_to_keep = max(1, int(len(class_subset) * percentage))
    _, closest_indices = torch.topk(distances, num_to_keep, largest=False)
    
    # Map back to original dataset indices
    selected_indices = [class_subset.indices[i] for i in closest_indices.tolist()]
    
    # Save indices if save_dir is provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        indices_file = save_dir / f"centroid_class_{target_class}_{percentage:.2f}.pt"
        torch.save(torch.tensor(selected_indices), indices_file)
        print(f"Saved {len(selected_indices)} indices for class {target_class} to {indices_file}")
    
    return Subset(dataset, selected_indices), selected_indices

def compute_all_centroids(dataset, dataset_name, feature_extractor, device, percentage=0.7, save_dir=None):
    """
    Computes centroid-based subsets for all classes in the dataset.
    
    Args:
        dataset: The dataset to process
        dataset_name: Name of the dataset
        feature_extractor: Model to extract features from images
        device: Device to run the model on
        percentage: Percentage of closest samples to keep (default: 0.7)
        save_dir: Directory to save indices to (optional)
        
    Returns:
        dict: Dictionary mapping class indices to selected indices
    """
    class_mapping = get_dataset_class_mapping(dataset_name)
    all_selected_indices = {}
    
    # Create save directory if provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    for class_idx in class_mapping:
        _, selected_indices = compute_centroid_based_subset(
            dataset, class_idx, feature_extractor, device, percentage, save_dir
        )
        all_selected_indices[class_idx] = selected_indices
        
        # Print progress
        print(f"Processed class {class_idx}: {len(selected_indices)} samples selected")
    
    # Save metadata
    if save_dir:
        meta_file = save_dir / f"centroid_metadata_{percentage:.2f}.txt"
        with open(meta_file, "w") as f:
            f.write(f"Centroid computation completed at {datetime.datetime.now()}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Percentage kept: {percentage}\n")
            f.write(f"Total classes processed: {len(class_mapping)}\n")
    
    return all_selected_indices

def load_centroid_indices(class_idx, save_dir, percentage=0.7):
    """
    Loads pre-computed centroid-based indices for a class.
    
    Args:
        class_idx: The class index
        save_dir: Directory where indices are saved
        percentage: Percentage used when computing indices
        
    Returns:
        list: List of selected indices
    """
    save_dir = Path(save_dir)
    indices_file = save_dir / f"centroid_class_{class_idx}_{percentage:.2f}.pt"
    
    if indices_file.exists():
        return torch.load(indices_file).tolist()
    else:
        print(f"Warning: No pre-computed centroid indices found for class {class_idx}")
        return []

def create_centroid_subset(dataset, dataset_name, save_dir, percentage=0.7):
    """
    Creates a dataset subset using pre-computed centroid-based indices.
    
    Args:
        dataset: The dataset to filter
        dataset_name: Name of the dataset
        save_dir: Directory where indices are saved
        percentage: Percentage used when computing indices
        
    Returns:
        Subset: A filtered dataset containing only the selected samples
    """
    class_mapping = get_dataset_class_mapping(dataset_name)
    all_indices = []
    
    for class_idx in class_mapping:
        indices = load_centroid_indices(class_idx, save_dir, percentage)
        all_indices.extend(indices)
    
    if len(all_indices) == 0:
        print(f"Warning: No centroid indices found in {save_dir}")
    else:
        print(f"Loaded {len(all_indices)} total indices from centroid computations")
    
    return Subset(dataset, all_indices)