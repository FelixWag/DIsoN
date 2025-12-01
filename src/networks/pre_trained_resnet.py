import torch
import torch.nn as nn
import torchvision


class ResidualIsolationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim // 4
        self.linear = nn.Linear(in_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # original linear path (what gave you 87%)
        out0 = self.linear(x)
        # small MLP "correction"
        out1 = self.mlp(x)
        # fuse: preserves the linear solution if out1≈0
        return out0 + out1

def replace_bn_with_in(module):
    """
    Recursively replace all BatchNorm2d layers with InstanceNorm2d layers in a given module.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Get the number of features from the BatchNorm layer
            num_features = child.num_features
            # Create an InstanceNorm layer with the same number of features
            instance_norm = nn.InstanceNorm2d(num_features, affine=True)
            # Replace the BatchNorm layer with the InstanceNorm layer
            setattr(module, name, instance_norm)
        else:
            # Recursively apply to child modules
            replace_bn_with_in(child)

def strip_prefix(state_dict, prefix):
    """
    Strips the specified prefix from the state_dict keys.

    Args:
        state_dict (dict): The original state_dict with prefixed keys.
        prefix (str): The prefix to remove.

    Returns:
        dict: A new state_dict with the prefix removed from keys.
    """
    stripped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        stripped_state_dict[new_key] = value
    return stripped_state_dict

class PretrainedResNet18(nn.Module):
    def __init__(self, pretrained_path, grayscale, device, num_classes, change_last_layer=True,small_resolution=False,
                 use_instance_norm=False, model_type='resnet18'):
        super(PretrainedResNet18, self).__init__()
        num_classes = 1 if num_classes == 2 else num_classes
        
        # Choose the appropriate ResNet model based on model_type
        if model_type == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=False)
        elif model_type == 'resnet34':
            self.model = torchvision.models.resnet34(pretrained=False)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        if grayscale:
            # Modify the first convolutional layer to accept one input channel (grayscale)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_instance_norm:
            print("Using instance normalization")
            replace_bn_with_in(self.model)
        else:
            print("Using batch normalization")
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if small_resolution:
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()  # Remove maxpool layer
        print("Pretrained path:", pretrained_path)
        state_dict = torch.load(pretrained_path, map_location=device)["state_dict"]
        state_dict = strip_prefix(state_dict, "model.")

        # Rename keys if they come from a model saved with dropout in the fc layer
        if 'fc.1.weight' in state_dict:
            state_dict['fc.weight'] = state_dict.pop('fc.1.weight')
        if 'fc.1.bias' in state_dict:
            state_dict['fc.bias'] = state_dict.pop('fc.1.bias')

        self.model.load_state_dict(state_dict, strict=True)

        #self.model.bn1 = nn.Identity()
        # Now randomly initialize the last layer
        if change_last_layer:
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        # Re-implement the forward pass so we can capture the features
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)

        # Features after first layer
        # features_after_first_layer = x.clone()
        # features_after_first_layer = torch.flatten(features_after_first_layer, 1)

        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)  # penultimate features

        logits = self.model.fc(features)  # final output
        # # Use model.fc (which is now Identity) and then our new head
        # features = self.model.fc(features)
        # logits = self.head(features) if hasattr(self, 'head') else features
        return features, logits
    
    def penult_feature(self, x):
        """
        Return only the penultimate-layer features (flattened after avgpool).
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        penultimate_feats = torch.flatten(x, 1)
        return penultimate_feats

    def freeze_backbone(self, up_to_layer=0):
        """
        Freezes the backbone of the model starting below the specified layer index.

        Args:
            up_to_layer (int): The layer index from which to start unfreezing.
                                - 0: Do not freeze anything (all layers trainable).
                                - 1: Unfreeze layer1, layer2, layer3, layer4, and fc.
                                - 2: Unfreeze layer2, layer3, layer4, and fc.
                                - 3: Unfreeze layer3, layer4, and fc.
                                - 4: Unfreeze layer4 and fc.
                                - 5: Unfreeze fc only.
        """
        # Dictionary mapping layer indices to model layers
        layer_mapping = {
            1: self.model.layer1,
            2: self.model.layer2,
            3: self.model.layer3,
            4: self.model.layer4,
            5: self.model.fc
        }

        if up_to_layer == 0:
            # Do not freeze anything; ensure all parameters are trainable
            for param in self.model.parameters():
                param.requires_grad = True
            print("All layers are set to trainable.")
        else:
            # Freeze all parameters first
            for param in self.model.parameters():
                param.requires_grad = False
            print("All layers have been frozen.")

            # Unfreeze layers from up_to_layer to fc (layer5)
            for idx in range(up_to_layer, 5 + 1):  # Include fc layer
                layer = layer_mapping.get(idx, None)
                if layer:
                    for param in layer.parameters():
                        param.requires_grad = True
                    print(f"Layer{idx} and its parameters are now trainable.")

            # Additional confirmation
            print("Selected layers have been unfrozen based on up_to_layer parameter.")

    def print_trainable_parameters(self):
        """
        Prints out which parameters are trainable.
        """
        for name, param in self.model.named_parameters():
            status = "Trainable" if param.requires_grad else "Frozen"
            print(f"{status}: {name}")
    
    def get_parameter_groups(self):
        """
        Returns separate parameter groups for the backbone and head components.
        """
        backbone_params = []
        head_linear_params = []
        head_mlp_params = []
        
        # Get backbone parameters (everything except the head)
        for name, param in self.model.named_parameters():
            if not name.startswith('head.'):
                backbone_params.append(param)
        
        # Get head parameters
        if hasattr(self, 'head'):
            for name, param in self.head.named_parameters():
                if name.startswith('linear.'):
                    head_linear_params.append(param)
                elif name.startswith('mlp.'):
                    head_mlp_params.append(param)
        
        return backbone_params, head_linear_params, head_mlp_params
    
    