import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import torchvision.models as M



data_dir = '/home/ccuser/data'

transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

test_dataset  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)




# Move to CPU/GPU device -- DO NOT MODIFY
device = "cuda" if torch.cuda.is_available() else "cpu"

## YOUR SOLUTION HERE ##
# Load ResNet18 and adapt for CIFAR-10 
model_resnet18 = M.resnet18(weights=None)
in_feats = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(in_feats,10)

# Load pre-trained weights fine-tuned on CIFAR-10
state_dict_resnet18 = torch.load("models/resnet18_cifar10.pt",map_location=device, weights_only=True)
model_resnet18.load_state_dict(state_dict_resnet18)

# Move to device and set to evaluation mode
model_resnet18 = model_resnet18.to(device)
model_resnet18.eval()

# Show output
# from custom_torchinfo import custom_summary                         
custom_summary(model_resnet18, input_size=(1, 3, 224, 224))



## YOUR SOLUTION HERE ##
def predict(model, dataloader, device="cpu"):
    model.eval()
    all_logits = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device),batch_y.to(device)
            # Forward pass

            logits = model(batch_X)
    
            # Convert logits to predicted labels
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            # Save predictions
            
            all_logits.append(logits.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            
    # Join predictions
    y_logits = np.concatenate(all_logits,axis=0)
    y_pred   = np.concatenate(all_predictions,axis=0)
    y_true   = np.concatenate(all_labels,axis=0)
    return y_logits, y_pred, y_true

# Generate test predictions
y_logits_resnet18, y_pred_resnet18, y_true = predict(model_resnet18,test_loader,device=device)

# Show output
print("First 10 predictions:", y_pred_resnet18[:10])
print("First 10 labels:", y_true[:10])








## YOUR SOLUTION HERE ##
test_accuracy_resnet18 = accuracy_score(y_true,y_pred_resnet18)
report_resnet18 = classification_report(y_true,y_pred_resnet18, target_names=test_dataset.classes)

# Show output
print(f"Accuracy: {test_accuracy_resnet18:.4f}")
print(report_resnet18)




# Introduction to Neural Network Architectures
# Convolutional Neural Networks: Lightweight Models
# 45 min
# In practice, we often rely on pre-trained models and compare their performance against simple neural networks, which serve as baselines that we train ourselves.

# Pre-trained models offer several benefits:

# Performance: leverages their pre-trained knowledge from massive datasets
# Speed: fewer training 
# epochs
# Preview: Docs Loading link description
#  required
# Flexibility: can be adapted and fine-tuned to specific datasets
# Here are a few pre-trained models for image tasks:

# ResNet50 (25M Parameters): deep residual network with many convolutional, pooling, activation, and normalization layers, pre-trained on ImageNet
# ResNet18 (11M Parameters): smaller version of ResNet50 that maintains performance, pre-trained on ImageNet
# MobileNet_V3 (1.5M Parameters): lightweight model designed for efficiency on mobile devices, pre-trained on ImageNet
# First, let’s load the MobileNet architecture from the torchvision module, and adapt it for the CIFAR-10 dataset (10 classes):

# import torchvision.models as M
# import torch.nn as nn
# model = M.mobilenet_v3_small(weights=None) 
# in_feats = model.classifier[-1].in_features
# model.classifier[-1] = nn.Linear(in_feats, 10)

# Copy to Clipboard

# M.mobilenet_v3_small() specifies the MobileNet architecture from torchvision.models.
# weights=None loads just the architecture without ImageNet weights, since we’ll load custom weights next.
# nn.Linear() replaces the final classification layer to output predictions for the 10 classes.
# Then, let’s load the pre-trained weights that have already been fine-tuned on the CIFAR-10 training dataset:

# state_dict = torch.load("models/mobilenet_cifar10.pt", map_location=device, weights_only=True)
# model.load_state_dict(state_dict)
# model = model.to(device).eval()

# Copy to Clipboard

# Prediction Loop
# Next, we’ll build the prediction loop similar to the MLP:

# Pass the input batch through the network’s forward pass
# Obtain the raw outputs (logits)
# Convert logits to probabilities using softmax
# Select the predicted class using argmax
# Save the predictions for evaluation
# all_predictions = []
# with torch.no_grad():
#     for batch_X, batch_y in dataloader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)

#         # Forward pass
#         logits = model(batch_X)

#         # Convert logits to predicted labels
#         probs = torch.softmax(logits, dim=1)
#         preds = probs.argmax(dim=1)
        
#         # Save predictions
#         all_predictions.append(preds.cpu().numpy())

# # Join predictions
# y_pred = np.concatenate(all_predictions, axis=0)

# Copy to Clipboard

# One difference from the prediction loop before is that we’ll use the following components to output predicted probabilities for each class:

# softmax: outputs probability (confidence) scores for each class
# argmax: selects the class with the highest probability score as the predicted label