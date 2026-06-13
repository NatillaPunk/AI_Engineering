import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

    
    

## YOUR SOLUTION HERE ##
transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# Show output
print(transform)
    
    
    
data_dir = '/home/ccuser/data'

## YOUR SOLUTION HERE ##
train_dataset = datasets.CIFAR10(root=data_dir,train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root=data_dir,train=False, download=True, transform=transform)

# Show output
print("CIFAR-10 Training Set: \n", train_dataset)
print("CIFAR-10 Testing Set: \n", test_dataset)
    

## YOUR SOLUTION HERE ##
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=False)
test_loader  = DataLoader(test_dataset,batch_size=64,shuffle=False)

# Show output
first_batch = next(iter(test_loader))
images, labels = first_batch
print("Batch shape:", images.shape)
print("Testing labels:", labels) 
    
    

    
    

    
    
    
# #Convolutional Neural Networks: Processing Images

# Moving on from tabular data, let’s introduce an architecture for modeling image data: Convolutional Neural Networks (CNNs).

# Convolutional Neural Networks (CNNs) are a type of deep neural network that features specialized layers for tasks such as image classification and other computer vision tasks. They are particularly effective for images because they exploit spatial locality (nearby pixels are related) and translation invariance (a cat is a cat, regardless of its position in the image).

# There are two key layers in CNNs: convolution layers and pooling layers.

# Convolution layers learn to identify important local patterns and relationships within the image. Here are the general steps for processing images:

# Apply a filter (or kernel), a small grid containing learnable weights, that slides over the input image.
# A convolution is performed by computing a weighted sum of the pixel values and filter weights at every location. This outputs a feature map capturing the filter’s activations across different regions of the image.
# The filter slides to the next location based on the stride (step size) — typically one pixel over — performs a convolution, and the result is added to the feature map.
# The filter continually shifts until the feature map contains activations for the entire image.
# Multiple filters can be applied to create multiple feature maps, with each filter learning to detect different patterns (edges, textures, shapes, etc.).
# Pooling layers reduce the spatial dimensions of feature maps while retaining the most important information through a process called downsampling:

# A small window (e.g., 2x2 grid) slides across the feature map with a specific stride.
# We pool all the values into a single, representative value (like the maximum, minimum, or average value) for each region.
# Preprocessing Images Using Torchvision
# Before building image models, let’s quickly understand an image’s key properties in the context of 
# machine learning
# Preview: Docs Loading link description
# .

# Each image contains the following:

# Pixel values: Each pixel holds intensity values where the number of values equals the number of color channels (e.g., grayscale images have one channel and RGB images have three channels).
# Color channels: Most color images use three channels—Red, Green, and Blue (RGB)—where the final color of a pixel is determined by combining the values across each channel.
# Resolution and aspect ratio: Resolution determines the level of detail, while the aspect ratio (width vs height) affects how the image is displayed or processed. Oftentimes, we’ll need to resize or standardize image dimensions.
# Using these properties of images, we’ll preprocess them using an open-sourced library called torchvision. This library is built on top of PyTorch, which provides us with datasets, processing tools, and pre-trained models. Preprocessing is necessary to standardize images into a consistent format that aligns with the input shape expected by pre-trained models.

# We’ll preprocess the images with torchvision using the following sequential pipeline:

# import torchvision.transforms as T

# transform = T.Compose([
#     T.Resize(32),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406),
#                 std=(0.229, 0.224, 0.225)),
# ])

# Copy to Clipboard

# .Resize(): resizes images to a fixed shape for model input
# .ToTensor(): converts each image into PyTorch tensors and scales values to the range [0.0, 1.0]
# .Normalize(): normalizes the pixel values within each channel based on the specified mean and standard deviation
# Note: The normalization values shown here come from the ImageNet dataset, which was used to pre-train many popular vision models. If you’re training from scratch or using a very different dataset, calculate the mean and standard deviation from your training data.

# Loading and Batching the CIFAR-10 Dataset
# For our image exercises, we’ll fine-tune models to classify images in a multiclass image classification task using a popular benchmark dataset called CIFAR-10. This dataset contains 60,000 color 32x32 images, each belonging to one of 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

# We’ll use torchvision to load the dataset and apply our preprocessing pipeline to the images:

# import torchvision.datasets as datasets
# train = datasets.CIFAR10(root="folder", train=True, download=False, transform=transform)
# test  = datasets.CIFAR10(root="folder", train=False, download=False, transform=transform)

# Copy to Clipboard

# root: specifies the directory to store the dataset
# train=True: indicates the training set (False indicates the testing set)
# download: downloads the dataset if it is not already in the directory
# Then, we’ll use PyTorch’s DataLoader to load the dataset in mini-batches:

# from torch.utils.data import DataLoader

# train_loader = DataLoader(train, batch_size=64, shuffle=True)
# test_loader  = DataLoader(test,  batch_size=128, shuffle=False)

# Copy to Clipboard

# Batch loading helps fit the images into memory, speeds up training with GPU parallelism, and improves generalization with the random shuffling of training images.

# Instructions
# Checkpoint 1 Passed
# 1.
# Let’s create a transformation pipeline that processes the CIFAR-10 images using transforms.Compose() with the following:

# Resizing the images to 224x224 pixels
# Convert the images into PyTorch tensors scaled between 0 and 1.
# Normalize each RGB color channel with the mean values (0.485, 0.456, 0.406) and standard deviation values (0.229, 0.224, 0.225).
# Create a transformation to resize an image to 64x64 pixels. Use the transform on the sample image, image. Visualize the results by plotting the original and resized image.

# Checkpoint 2 Passed
# 2.
# Next, load the CIFAR-10 training and testing sets while applying the transformation pipeline to the images in each:

# Load/save the dataset within the data_dir directory.
# Set train to True for the training set (False for the testing set).
# Use transforms= to apply the transform pipeline to each image.
# Set download to True.
# Save the training set to the variable train_dataset and the testing set to the variable test_dataset.

# Checkpoint 3 Passed
# 3.
# Lastly, create dataloaders for the training and testing sets using the DataLoader utility class to load the images in batches:

# For the training set:

# Load 32 images per batch.
# Shuffle the images.
# Save the dataloader to the variable train_loader.
# For the testing set:

# Load 64 images per batch for the testing set.
# Do not shuffle the images.
# Save the dataloader to the variable test_loader.


