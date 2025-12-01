# Start training
import os
import pandas as pd
import numpy as np
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
# Import optimizers from specific submodules
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torchvision.transforms import TrivialAugmentWide

import pytorch_lightning as pl
from pathlib import Path
import torch.nn as nn
import wandb
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Custom Dataset for CheXpert
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, img_dir, target_class, grayscale, transform=None, num_classes=2):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes
        self.grayscale = grayscale

        # For multi-class one-hot encoded data, keep all class columns
        if num_classes > 2 and isinstance(target_class, list):
            self.data = self.data[['image_path'] + target_class]
            self.multi_class_columns = target_class
        else:
            # For binary classification, keep only the specified target class
            self.data = self.data[['image_path', f'{target_class}']]
            self.target_class = target_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['image_path'])
        if self.grayscale:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')

        if self.num_classes > 2:
            one_hot = self.data.iloc[idx][self.multi_class_columns].values
            label = np.argmax(one_hot)
            label = torch.tensor(label)
        else:
            label = self.data.iloc[idx][self.target_class].astype(np.float32)
            label = torch.FloatTensor([label])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transform(dataset):
    if dataset == 'x_ray':
        return transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.CenterCrop(224),
                                   transforms.RandomRotation(degrees=15),
                                   transforms.RandomCrop(224, padding=25),
                                   transforms.ToTensor(),
                                   transforms.ConvertImageDtype(torch.float),
                                   transforms.Normalize(mean=[0.5066162], std=[0.28903392]), ])
    elif dataset == 'skin_lesion':
        return transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.CenterCrop(224),
                                   transforms.RandomRotation(degrees=15),
                                   transforms.RandomCrop(224, padding=25),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomPerspective(distortion_scale=0.2),
                                   transforms.GaussianBlur((3, 3), sigma=(0.1, 2.0)),
                                   transforms.ToTensor(),
                                   transforms.ConvertImageDtype(torch.float),
                                   transforms.Normalize(mean=[0.72662437, 0.6243302, 0.5687489],
                                                        std=[0.22084126, 0.22352666, 0.22693515])])
    elif dataset == 'breastmnist':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),  # to make the images square
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=0.32283312, std=0.2032362)])
    elif dataset == 'midog':
        return transforms.Compose([
            transforms.Resize((50, 50)),
            TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.712, 0.496, 0.756], 
                                 std=[0.167, 0.167, 0.110])
        ])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}. Supported types are 'x_ray', 'skin_lesion', 'breastmnist', and 'midog'.")


def get_image_paths(dataset):
    if dataset == 'x_ray':
        train_csv = 'CHANGE TO FULL PATH  data_pretrain folder'
        img_dir = 'CHANGE TO FULL PATH  data path'
        val_csv = 'CHANGE TO FULL PATH  data_pretrain folder'
        return img_dir, train_csv, val_csv
    elif dataset == 'skin_lesion':
        train_csv = 'CHANGE TO FULL PATH  data_pretrain folder'
        img_dir = 'CHANGE TO FULL PATH  data path'
        val_csv = 'CHANGE TO FULL PATH  data_pretrain folder'
        return img_dir, train_csv, val_csv
    elif dataset == 'breastmnist':
        train_csv = 'CHANGE TO FULL PATH  data_pretrain folder'
        img_dir = 'CHANGE TO FULL PATH  data path'
        val_csv = 'CHANGE TO FULL PATH  data_pretrain folder'
        return img_dir, train_csv, val_csv
    elif dataset == 'midog':
        train_csv = 'CHANGE TO FULL PATH  data_pretrain folder'
        val_csv = 'CHANGE TO FULL PATH  data_pretrain folder'
        img_dir = 'CHANGE TO FULL PATH  data path'
        return img_dir, train_csv, val_csv
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}. Supported types are 'x_ray', 'skin_lesion', 'breastmnist', and 'midog'.")


def get_training_data_information(dataset):
    transform = get_data_transform(dataset)
    img_dir, train_csv, val_csv = get_image_paths(dataset)
    
    if dataset == 'x_ray':
        target_class = 'cardiomegaly'
        grayscale = True
        num_classes = 2
    elif dataset == 'skin_lesion':
        target_class = 'nevus'
        grayscale = False
        num_classes = 2
    elif dataset == 'breastmnist':
        target_class = ['normal', 'benign', 'malignant']
        grayscale = True
        num_classes = 3
    elif dataset == 'midog':
        target_class = None
        grayscale = False
        num_classes = 3
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}")

    return transform, img_dir, train_csv, val_csv, target_class, grayscale, num_classes


def replace_bn_with_in(module):
    """
    Recursively replace all BatchNorm2d layers with InstanceNorm2d layers in a given module.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            instance_norm = nn.InstanceNorm2d(num_features, affine=True)
            setattr(module, name, instance_norm)
        else:
            replace_bn_with_in(child)


# Add mixup augmentation for training with limited data
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Define the LightningModule
class CheXpertCardiomegalyModel(pl.LightningModule):
    def __init__(self, grayscale, learning_rate=0.001, model_type='densenet121', 
                 num_classes=2, use_instance_norm=True, dataset='skin_lesion', 
                 weight_decay=1e-4, max_epochs=100, use_mixup=False, 
                 freeze_strategy='none', scheduler_milestones=None, scheduler_gamma=None):
        super(CheXpertCardiomegalyModel, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.use_instance_norm = use_instance_norm
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.use_mixup = use_mixup
        self.mixup_alpha = 0.2  # Mixup alpha parameter
        self.current_epoch_num = 0
        self.freeze_strategy = freeze_strategy  # 'none', 'all_except_head', or 'first_three_blocks'

        # Always use num_classes as output dimension
        output_dim = num_classes
        
        # Load pre-trained model based on model_type
        if model_type == 'densenet121':
            # Determine whether to use pretrained weights
            use_pretrained = (dataset == 'midog')
            self.model = models.densenet121(pretrained=use_pretrained)
            if grayscale:
                # Modify the first convolutional layer to accept one input channel (grayscale)
                self.model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(num_ftrs, output_dim)
        elif model_type == 'resnet18':
            # Determine whether to use pretrained weights
            # Only use pretrained if NOT using InstanceNorm, or if dataset is MIDOG
            use_pretrained = not self.use_instance_norm or dataset == 'midog' # Adjusted logic
            print(f"Using pretrained weights for ResNet18: {use_pretrained}")
            self.model = models.resnet18(pretrained=use_pretrained)
            if grayscale:
                # Modify the first convolutional layer to accept one input channel (grayscale)
                self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # For MIDOG dataset, modify the architecture for small images
            if dataset == 'midog':
                # Modify first conv layer to handle 50x50 inputs with smaller kernel and stride
                self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                # Replace maxpool with a no-op maxpool that maintains dimensions
                self.model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1)
            
            num_ftrs = self.model.fc.in_features
            
            # Modify the last layer to output the correct number of classes
            self.model.fc = torch.nn.Linear(num_ftrs, output_dim)
            
            # Apply freezing strategy based on parameter
            if self.freeze_strategy == 'all_except_head':
                # Freeze everything except the final fully connected layer
                for name, param in self.model.named_parameters():
                    if 'fc' not in name:  # Don't freeze the fc layer
                        param.requires_grad = False
                    
            elif self.freeze_strategy == 'first_three_blocks':
                # Freeze only the first three residual blocks (as described in the paper)
                for name, param in self.model.named_parameters():
                    # Freeze conv1, bn1, layer1, layer2, layer3
                    if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']):
                        param.requires_grad = False
                    else:
                        # Keep layer4 and fc trainable
                        param.requires_grad = True
            
            # Print which layers are being trained
            print("\nTrainable parameters:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}")
            print("\nFrozen parameters:")
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    print(f"  {name}")
        elif model_type == 'resnet34':
            # Determine whether to use pretrained weights
            # Only use pretrained if NOT using InstanceNorm, or if dataset is MIDOG
            use_pretrained = not self.use_instance_norm or dataset == 'midog'
            print(f"Using pretrained weights for ResNet34: {use_pretrained}")
            self.model = models.resnet34(pretrained=use_pretrained)
            if grayscale:
                # Modify the first convolutional layer to accept one input channel (grayscale)
                self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # For MIDOG dataset, modify the architecture for small images
            if dataset == 'midog':
                # Modify first conv layer to handle 50x50 inputs with smaller kernel and stride
                self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                # Replace maxpool with a no-op maxpool that maintains dimensions
                self.model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1)
            
            num_ftrs = self.model.fc.in_features
            
            # Modify the last layer to output the correct number of classes
            self.model.fc = torch.nn.Linear(num_ftrs, output_dim)
            
            # Apply freezing strategy based on parameter
            if self.freeze_strategy == 'all_except_head':
                # Freeze everything except the final fully connected layer
                for name, param in self.model.named_parameters():
                    if 'fc' not in name:  # Don't freeze the fc layer
                        param.requires_grad = False
                    
            elif self.freeze_strategy == 'first_three_blocks':
                # Freeze only the first three residual blocks (as described in the paper)
                for name, param in self.model.named_parameters():
                    # Freeze conv1, bn1, layer1, layer2, layer3
                    if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']):
                        param.requires_grad = False
                    else:
                        # Keep layer4 and fc trainable
                        param.requires_grad = True
            
            # Print which layers are being trained
            print("\nTrainable parameters:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}")
            print("\nFrozen parameters:")
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    print(f"  {name}")
        elif model_type == 'resnet50':
            # Determine whether to use pretrained weights
            use_pretrained = (dataset == 'midog')
            self.model = models.resnet50(pretrained=use_pretrained)
            if grayscale:
                # Modify the first convolutional layer to accept one input channel (grayscale)
                self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, output_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Only replace BatchNorm with InstanceNorm if specified
        if self.use_instance_norm:
            replace_bn_with_in(self.model)

        # Use CrossEntropyLoss for all cases
        self.criterion = torch.nn.CrossEntropyLoss()

        # Track accuracy for all classification problems
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        # ← Right after you save hparams, store them on the model:
        #    fall back to your defaults if None was passed
        self.scheduler_milestones = (
            scheduler_milestones if scheduler_milestones is not None else [15, 25]
        )
        self.scheduler_gamma = (
            scheduler_gamma if scheduler_gamma is not None else 0.1
        )

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.current_epoch_num = self.current_epoch
        
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Disable mixup for now - just do a standard forward pass
        outputs = self(images)
        
        # Convert labels properly
        if labels.dim() > 1:
            labels = labels.view(-1).long()
        else:
            labels = labels.long()
            
        loss = self.criterion(outputs, labels)
        
        # Log loss and accuracy
        if batch_idx % 50 == 0:  # Log more frequently
            self.log('train_loss', loss, on_step=True, on_epoch=True, 
                    sync_dist=True, reduce_fx='mean')
        
            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            self.train_accuracy.update(preds, labels)
        
        return loss

    def on_train_epoch_end(self):
        train_acc = self.train_accuracy.compute()
        self.log('train_accuracy_epoch', train_acc, prog_bar=True, sync_dist=True)
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        
        # Convert labels properly
        if labels.dim() > 1:
            labels = labels.view(-1).long()
        else:
            labels = labels.long()
        
        loss = self.criterion(outputs, labels)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, 
                 sync_dist=True, reduce_fx='mean')
        
        # Calculate validation accuracy
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy.update(preds, labels)
        
        return loss

    def on_validation_epoch_end(self):
        val_acc = self.val_accuracy.compute()
        # Standard logging for all datasets
        self.log('val_accuracy', val_acc, prog_bar=True, sync_dist=True)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        if self.dataset == 'midog':
            # Add the model configuration with SGD for MIDOG
            # SGD with momentum of 0.9 as specified
            # Use SGD directly
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams['learning_rate'],
                momentum=0.9,  # β=0.9 as specified
                weight_decay=self.hparams['weight_decay']
            )
            
            # OneCycle learning rate scheduler as specified in the paper
            # Calculate exact steps: MIDOG dataset has 1896 samples with batch size 128
            # This means 15 batches per epoch * 300 epochs = 4500 total steps
            # Previously we had max.epochs as total steps, but this is incorrect
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams['learning_rate'] * 10,  # 10x the base lr
                total_steps=4500,  # Exactly 15 batches/epoch * 300 epochs
                pct_start=0.3,  # Spend 30% of training time warming up
                div_factor=10.0,  # Initial lr is max_lr/div_factor
                final_div_factor=1000.0,  # Final lr is initial_lr/final_div_factor
                anneal_strategy='cos'  # Use cosine annealing
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'  # OneCycleLR is updated each step
                }
            }
        elif self.dataset == 'breastmnist':
            # Use Adam optimizer for breastmnist as specified in the reference script
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams['learning_rate'],
                weight_decay=self.hparams['weight_decay']
            )
            return optimizer
        else:
            # Fix optimizer reference
            # Use Adam directly
            optimizer = Adam(
                self.parameters(), 
                lr=self.hparams['learning_rate'],  # Access as dictionary item
                weight_decay=self.hparams['weight_decay']
            )
            return optimizer

# Class for MIDOG text file dataset
class MIDOGFileDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the text file with image paths and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = []
        self.labels = []
        self.root_dir = root_dir
        self.transform = transform
        
        # Read the text file
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) != 2:
                    continue
                    
                path, label = parts
                self.image_paths.append(path)
                self.labels.append(int(label))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

###################### SETTING UP TRAINING #########################################################
# Change this to the dataset you want to use
DATASET = 'skin_lesion'  
model_type = 'resnet34' 

# Define hyperparameters based on dataset
if DATASET == 'midog':
    # Existing MIDOG code
    lr = 5e-4
    batch_size = 128
    max_epochs = 300
    weight_decay = 1e-1
    use_instance_norm = False
    use_mixup = False
    freeze_strategy = 'none'
    scheduler_milestones = None # MIDOG uses OneCycleLR, not MultiStep
    scheduler_gamma = None
elif DATASET == 'breastmnist':
    # Match the hyperparameters from provided script
    lr = 0.001
    batch_size = 32
    max_epochs = 1000
    weight_decay = 0.0  # No weight decay specified in the reference script
    use_instance_norm = True
    use_mixup = False
    freeze_strategy = 'none'
    scheduler_milestones = None
    scheduler_gamma = None
elif DATASET == 'skin_lesion':
    # Custom configuration for skin_lesion dataset
    lr = 0.001                  # Requested learning rate
    batch_size = 32             # Requested batch size
    max_epochs = 750            # Requested number of epochs
    weight_decay = 0.0          # No weight decay
    use_instance_norm = True    # Use instance normalization
    use_mixup = False           # No mixup augmentation
    freeze_strategy = 'none'    # Train all parameters
    scheduler_milestones = None
    scheduler_gamma = None
elif DATASET == 'x_ray':
    # Custom configuration for x_ray dataset
    lr = 0.001                  # Same learning rate as skin_lesion
    batch_size = 32             # Same batch size as skin_lesion
    max_epochs = 500            # 500 epochs for x_ray
    weight_decay = 0.0          # No weight decay
    use_instance_norm = True    # Use instance normalization
    use_mixup = False           # No mixup augmentation
    freeze_strategy = 'none'    # Train all parameters
    scheduler_milestones = None
    scheduler_gamma = None
else:
    lr = 0.003  # Default for other datasets
    batch_size = 32
    max_epochs = 1200
    weight_decay = 0.0
    use_instance_norm = True
    use_mixup = False
    freeze_strategy = 'none'

transform, img_dir, train_csv, val_csv, target_class, grayscale, num_classes = get_training_data_information(dataset=DATASET)

# Create a simplified dataset loading section
if DATASET == 'midog':
    # Use the MIDOGFileDataset class for MIDOG dataset
    train_dataset = MIDOGFileDataset(
        txt_file=train_csv,
        root_dir=img_dir,
        transform=transform
    )
    
    # Now use the separate validation file
    val_dataset = MIDOGFileDataset(
        txt_file=val_csv,
        root_dir=img_dir,
        transform=transform
    )
    
    # Add validation loader for MIDOG dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=8
    )
else:
    # Create training dataset
    train_dataset = CheXpertDataset(csv_file=train_csv, img_dir=img_dir, target_class=target_class,
                                   transform=transform, grayscale=grayscale, num_classes=num_classes)
    
    # Create validation dataset if val_csv is provided
    if val_csv is not None:
        val_dataset = CheXpertDataset(csv_file=val_csv, img_dir=img_dir, target_class=target_class,
                                     transform=transform, grayscale=grayscale, num_classes=num_classes)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=8
        )
    else:
        val_loader = None

# Optimize the DataLoader for faster processing
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=8
)

# Update config with training information
config = {
    "dataset": DATASET,
    "batch_size": batch_size,
    "learning_rate": lr,
    "model_type": model_type,
    "num_classes": num_classes,
    "grayscale": grayscale,
    "max_epochs": max_epochs,
    "normalization": "IN" if use_instance_norm else "BN",
    "optimizer": "SGD+momentum" if DATASET == 'midog' else "Adam",
    "weight_decay": weight_decay,
    "validation": val_loader is not None,
    "scheduler": "OneCycleLR" if DATASET == 'midog' else "None",
    "imagenet_pretrained": DATASET == 'midog',
    "freeze_strategy": freeze_strategy
}

# Initialize W&B Logger with config
wandb_logger = WandbLogger(
    project=f'MedicalImageClassification',
    name=f'CHECK_{DATASET}_{model_type}_{max_epochs}_epochs_{num_classes}_classes_norm_{config["normalization"]}',
    log_model=True,
    save_dir='wandb_logs',
    config=config
)

model = CheXpertCardiomegalyModel(
    learning_rate=lr, 
    model_type=model_type,
    grayscale=grayscale, 
    num_classes=num_classes,
    use_instance_norm=use_instance_norm,
    dataset=DATASET,
    weight_decay=config["weight_decay"],
    max_epochs=config["max_epochs"],
    use_mixup=use_mixup,
    freeze_strategy=freeze_strategy,
    scheduler_milestones=scheduler_milestones,
    scheduler_gamma=scheduler_gamma
)

# Define the Trainer with a single GPU
trainer = pl.Trainer(
    max_epochs=config["max_epochs"], 
    accelerator='gpu', 
    devices=[0],  # Specify GPU 0
    logger=wandb_logger,
    enable_progress_bar=True,
    val_check_interval=1.0,
    callbacks=[
        ModelCheckpoint(
            monitor='val_loss' if val_loader is not None else 'train_loss',
            mode='min',
            save_top_k=3,
            filename=f'{DATASET}-{model_type}-{{epoch:02d}}-{{val_loss:.2f}}'
        ),
        ModelCheckpoint(
            save_last=True,
            filename=f'{DATASET}-{model_type}-last'
        ),
        LearningRateMonitor(logging_interval='step')
    ] if val_loader is not None else []
)

# Train the model
if val_loader is not None:
    trainer.fit(model, train_loader, val_loader)
else:
    trainer.fit(model, train_loader)