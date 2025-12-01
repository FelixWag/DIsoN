from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
import torch
from torch.utils.data import Sampler
import time
import logging
from itertools import cycle
import hashlib


def deterministic_random_key(x, seed=42):
    return int(hashlib.md5((str(x) + str(seed)).encode('utf-8')).hexdigest(), 16)

class MedicalDataset(Dataset):
    def __init__(self, root_data_path, no_support_device_file, pacemaker_file,
                 transform = None, num_no_support_device_samples=None, num_pacemaker_samples=None, seed=42, duplicate_pacemaker_samples=None,
                 exclude_target_transform=False, val_transform=None, i=0):
        """
        General dataset class for medical imaging with support for ID and OOD samples.
        
        Args:
            root_data_path (str): Path to the root data directory
            no_support_device_file (str): Path to the text file with in-distribution image paths
            pacemaker_file (str): Path to the text file with target/OOD image paths
            num_no_support_device_samples (int, optional): Number of ID samples to include. Defaults to None (include all)
            num_pacemaker_samples (int, optional): Number of target samples to include. Defaults to None (include all)
            transform (callable, optional): Optional transform to be applied on an image
            seed (int): Random seed for reproducibility
            duplicate_pacemaker_samples (int, optional): Number of times to duplicate target samples. Defaults to None (no duplication)
            exclude_target_transform (bool): If True, apply val_transform to target samples instead of train transform
            val_transform (callable, optional): Validation transform
            i (int): Index for selecting specific target sample
        """
        root_data_path = Path(root_data_path)

        with open(no_support_device_file, 'r') as f:
            no_support_device_paths = f.read().splitlines()

        with open(pacemaker_file, 'r') as f:
            pacemaker_paths = f.read().splitlines()
        
        no_support_device_paths_absolute = Path(no_support_device_paths[0]).is_absolute() if no_support_device_paths else False
        pacemaker_paths_absolute = Path(pacemaker_paths[0]).is_absolute() if pacemaker_paths else False
        
        if num_no_support_device_samples is not None:
            random.seed(seed+1)
            no_support_device_paths = random.sample(no_support_device_paths, min(num_no_support_device_samples, len(no_support_device_paths)))

        if no_support_device_paths_absolute:
            self.data = [(Path(path), 0) for path in no_support_device_paths]
        else:
            self.data = [(root_data_path / path, 0) for path in no_support_device_paths]

        pacemaker_paths = list(set(pacemaker_paths) - set(no_support_device_paths))
        pacemaker_paths = sorted(pacemaker_paths, key=lambda x: deterministic_random_key(x))

        if num_pacemaker_samples is not None:
            if not pacemaker_paths:
                logging.info("No target paths available, sampling from in-distribution paths")
                random.seed(seed)
                pacemaker_samples = random.sample(no_support_device_paths, min(num_pacemaker_samples, len(no_support_device_paths)))
                no_support_device_paths = [path for path in no_support_device_paths if path not in pacemaker_samples]
                self.data = [(root_data_path / path, 0) for path in no_support_device_paths]
            else:
                random.seed(seed)
                if isinstance(i, int):
                    if num_pacemaker_samples == 0:
                        pacemaker_samples = []
                    else:
                        pacemaker_samples = [pacemaker_paths[i]]
                else:
                    pacemaker_samples = [i]
                overlap = set(pacemaker_samples) & set(no_support_device_paths)
                if overlap:
                    raise Exception(f"Target samples overlap with in-distribution paths: {overlap}")
            if pacemaker_paths_absolute:
                self.data += [(Path(path), 1) for path in pacemaker_samples]
            else:
                self.data += [(root_data_path / path, 1) for path in pacemaker_samples]
        else:
            overlap = set(pacemaker_paths) & set(no_support_device_paths)
            if overlap:
                raise Exception(f"Target samples overlap with in-distribution paths: {overlap}")
            if pacemaker_paths_absolute:
                self.data += [(Path(path), 1) for path in pacemaker_paths]
            else:
                self.data += [(root_data_path / path, 1) for path in pacemaker_paths]

        if duplicate_pacemaker_samples is not None:
            if num_pacemaker_samples is None or num_pacemaker_samples == 0:
                raise Exception("num_pacemaker_samples must be specified and greater than 0 to use duplicate_pacemaker_samples.")
            if pacemaker_paths_absolute:
                self.data += [(Path(path), 1) for path in pacemaker_samples] * (duplicate_pacemaker_samples - 1)
            else:
                self.data += [(root_data_path / path, 1) for path in pacemaker_samples] * (duplicate_pacemaker_samples - 1)

        random.shuffle(self.data)

        self.transform = transform
        self.exclude_target_transform = exclude_target_transform
        self.val_transform = val_transform

    def __len__(self):
        return len(self.data)
    
    def get_target_indices(self):
        """
        Returns a list of indices where the label is 1 (target samples).
        """
        target_indices = [idx for idx, (_, label, *_ ) in enumerate(self.data) if label == 1]
        return target_indices

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)
        
        if self.transform:
            if self.exclude_target_transform:
                if label == 0:
                    image = self.transform(image)
                else:
                    image = self.val_transform(image)
            else:
                image = self.transform(image)

        return image, label


class TargetSampler(Sampler):
    """
    Custom sampler that ensures a specific number of target samples per batch.
    """
    def __init__(self, data_source, batch_size, target_per_batch=1, target_frequency=1):
        """
        Args:
            data_source: Dataset to sample from
            batch_size (int): Total batch size
            target_per_batch (int): Number of target samples per batch (when target batch is selected)
            target_frequency (int): Frequency of batches that contain target samples (1 = every batch)
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.target_per_batch = target_per_batch
        self.target_frequency = target_frequency
        start_time = time.time()
        
        labels = [label for _, label in data_source.data]
        self.target_indices = [i for i, l in enumerate(labels) if l == 1]
        self.id_indices = [i for i, l in enumerate(labels) if l == 0]
        
        end_time = time.time()
        logging.debug(f"TargetSampler initialization time: {end_time - start_time:.4f} seconds")

    def __iter__(self):
        random.shuffle(self.id_indices)
        random.shuffle(self.target_indices)

        target_cycle = cycle(self.target_indices)
        batch_idx = 0
        i = 0
        
        while i < len(self.id_indices):
            batch_idx += 1

            if batch_idx % self.target_frequency == 0:
                id_needed = self.batch_size - self.target_per_batch
            else:
                id_needed = self.batch_size

            batch = self.id_indices[i:i + id_needed]
            i += id_needed

            if batch_idx % self.target_frequency == 0:
                target_samples = [next(target_cycle) for _ in range(self.target_per_batch)]
                batch.extend(target_samples)

            if len(batch) < self.batch_size:
                logging.debug(f"Yielding a smaller batch of size {len(batch)}")
            
            random.shuffle(batch)
            yield batch

    def __len__(self):
        target_batch_size = self.target_per_batch
        id_batch_size = self.batch_size - target_batch_size
        total_batches = (len(self.id_indices) + id_batch_size - 1) // id_batch_size
        return total_batches


# Basic transforms for grayscale images
basic_transform_grayscale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop(224, padding=25),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5066162], std=[0.28903392]),
])

basic_transform_grayscale_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5066162], std=[0.28903392]),
])

# BreastMNIST transforms
breastmnist_transform = transforms.Compose([
    transforms.Resize((224)),
    transforms.CenterCrop(224),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0.32283312, std=0.2032362)
])

breastmnist_transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0.32283312, std=0.2032362)
])

# Skin lesion transforms
transform_skin_lesion = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop(224, padding=25),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.72662437, 0.6243302, 0.5687489], std=[0.22084126, 0.22352666, 0.22693515])
])

transform_skin_lesion_no_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.72662437, 0.6243302, 0.5687489], std=[0.22084126, 0.22352666, 0.22693515])
])

# MIDOG transforms
transform_midog = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize(50),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.712, 0.496, 0.756], 
                        std=[0.167, 0.167, 0.110])
])

transform_midog_no_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize(50),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.712, 0.496, 0.756], 
                        std=[0.167, 0.167, 0.110])
])

def get_transform(setting):
    """
    Get data transforms based on the specified setting.
    
    Args:
        setting (str): Transform setting name
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    transform_dict = {
        'grayscale_basic': {
            'train': basic_transform_grayscale,
            'val': basic_transform_grayscale_val
        },
        'grayscale_no_transform': {
            'train': basic_transform_grayscale_val,
            'val': basic_transform_grayscale_val
        },
        'skin_lesion_basic': {
            'train': transform_skin_lesion,
            'val': transform_skin_lesion_no_transform
        },
        'skin_lesion_no_transform': {
            'train': transform_skin_lesion_no_transform,
            'val': transform_skin_lesion_no_transform
        },
        'breastmnist_basic': {
            'train': breastmnist_transform,
            'val': breastmnist_transform_val
        },
        'breastmnist_no_transform': {
            'train': breastmnist_transform_val,
            'val': breastmnist_transform_val
        },
        'midog_basic': {
            'train': transform_midog,
            'val': transform_midog_no_transform
        },
        'midog_no_transform': {
            'train': transform_midog_no_transform,
            'val': transform_midog_no_transform
        }
    }

    if setting not in transform_dict:
        raise ValueError(f"Invalid setting: {setting}, available settings are: {list(transform_dict.keys())}")
    
    print(f"Selected setting: {setting}")
    print("Train transforms:")
    train_transform = transform_dict[setting]['train']
    for t in train_transform.transforms:
        print(f"  - {t.__class__.__name__}: {t}")
    
    print("Test/Validation transforms:")
    val_transform = transform_dict[setting]['val']
    for t in val_transform.transforms:
        print(f"  - {t.__class__.__name__}: {t}")
    
    return transform_dict[setting]['train'], transform_dict[setting]['val']


# Legacy alias for backward compatibility
CheXpertDataset = MedicalDataset

