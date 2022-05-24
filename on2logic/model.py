import logging

import pandas as  pd
import numpy as np
import torch
from torch.cuda import is_available as cuda_is_available
from torch.cuda import get_device_name as cuda_get_device_name

import timm
from scipy import spatial
import seaborn as sns
import matplotlib.pyplot as plt

def get_device() -> str:
    """Detects CUDA availability and returns the appropriate torch device.
    Returns:
        device (str): device to-be used in torch
    """
    if cuda_is_available():
        logging.info(f'using GPU: {cuda_get_device_name()}')
        device = 'cuda'
    else:
        logging.info('No GPU found, using CPU')
        device = 'cpu'
    return device

def load_image_model(model_name:str ='resnet50') -> torch.nn.Module:
    device = get_device()
    image_model = timm.create_model(model_name, pretrained=True).to(device)
    image_model.eval() # this turns off dropout etc, important!
    return image_model

def generate_vectors_from_preprocessed_folder(image_model, image_dataset, device:str = None):
    if device is None:
        device = get_device()
    
    image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False)
    image_vectors = []
    image_labels = []
    for batch_of_images, batch_of_labels in image_dataloader:
        batch_of_vectors = image_model(batch_of_images.to(device))
        image_vectors.append(batch_of_vectors)
        image_labels.append(batch_of_labels)
        
    image_vectors = torch.cat(image_vectors).detach().numpy()
    image_labels = torch.cat(image_labels).detach().numpy()
    return image_vectors, image_labels

def generate_vector_for_pil_image(pil_image, image_model, torchvision_transform, device:str = None):
    if device is None:
        device = get_device()
    transformed_image = torchvision_transform(pil_image).unsqueeze(0).to(device)
    image_vector = image_model(transformed_image).detach().numpy()
    return image_vector