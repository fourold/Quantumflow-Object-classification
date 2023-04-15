import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import qiskit  
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm
import functools



# Weiwen: modify the target classes starting from 0. Say, [3,6] -> [0,1]
def data_generate():
    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),   # Resize the images to 32x32 pixels
        transforms.ToTensor(),        # Convert the images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize the images by (mean,standard deviation values )
    ])

    # Load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                download=True, transform=transform)

    # Create a list of the class names in the CIFAR-10 dataset
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Extract the indices of the dog and cat classes
    dog_cat_indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i] in [3, 5]]

    # Create a subset of the CIFAR-10 dataset with only the dog and cat images
    dog_cat_dataset = torch.utils.data.Subset(train_dataset, dog_cat_indices)
    return dog_cat_dataset

dog_cat_dataset = data_generate()
# Get a random image from the dataset
image, label = dog_cat_dataset[10]


print(image[0].shape)