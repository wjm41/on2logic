import pandas as  pd
import numpy as np
import torch
import timm
from scipy import spatial
import seaborn as sns
import matplotlib.pyplot as plt

def load_image_model(model_name:str ='restnet50') -> torch.nn.Module:
    image_model = timm.create_model(model_name, pretrained=True)
    image_model.eval() # this turns off dropout etc, important!
    return image_model

def generate_vectors_from_preprocessed_folder(image_model, image_dataset):
    image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False)
    image_vectors = []
    image_labels = []
    for batch_of_images, batch_of_labels in image_dataloader:
        batch_of_vectors = image_model(batch_of_images)
        image_vectors.append(batch_of_vectors)
        image_labels.append(batch_of_labels)
        
    image_vectors = torch.cat(image_vectors).detach().numpy()
    image_labels = torch.cat(image_labels).detach().numpy()
    return image_vectors, image_labels
