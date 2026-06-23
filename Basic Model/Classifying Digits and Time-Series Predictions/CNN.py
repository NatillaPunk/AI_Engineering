import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# We set the object transform which we will apply to the images we want to adapt
transform = T.Compose([
    T.Resize(28),
    T.ToTensor(),
    T.Normalize(
        mean=(0.1307),
        std=(0.3081)
    )    
])


dataset_dir = 'Datasets'
train_dataset = datasets.MNIST(root=dataset_dir, train=True,download=True, transform=transform)
test_dataset = datasets.MNIST(root=dataset_dir, train=False,download=True, transform=transform)


train_dataloader = DataLoader(train_dataset,32,shuffle=True)
test_dataloader = DataLoader(train_dataset,32,shuffle=False)





