from pathlib import Path

import numpy as np
import pandas as pd
from torchvision import datasets, transforms
    
def return_default_transforms(image_size:int = 500):
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size-1),
                                    transforms.ToTensor()])
    return transform

def load_image_folder_and_return_dataset(folder_name:str = '../data/images/',image_size:int = 500):

    transform = return_default_transforms(image_size)
    
    image_dataset = datasets.ImageFolder(folder_name, transform=transform)
    image_label_to_name =  {value:key for key, value in image_dataset.class_to_idx.items()}
    
    page_numbers = [name[0].split('.')[-2].split('-')[-1] for name in image_dataset.imgs] # page numbers numerically not strictly correct
    return image_dataset, image_label_to_name, page_numbers

def generate_manuscript_dataframe(image_vectors, image_labels, label_dictionary, page_numbers):
    manuscript_dataframe = pd.DataFrame({'vector': [x for x in image_vectors], 'label': image_labels})
    manuscript_dataframe['manuscript'] = [label_dictionary[label] for label in manuscript_dataframe['label']]
    manuscript_dataframe['page'] = page_numbers
    return manuscript_dataframe

def case_study_setup():
    parent_dirname = str(Path.cwd().parents[0])
    image_dirname = parent_dirname+'/data/images/case_study/'
    numpy_dirname = parent_dirname+'/notebooks/numpy/'
    manuscript_dataset, manuscript_label_to_name, manuscript_page_numbers = load_image_folder_and_return_dataset(image_dirname)

    manuscript_vectors = np.load(numpy_dirname+'vectors.npy')
    manuscript_labels = np.load(numpy_dirname+'labels.npy')
    
    manuscript_dataframe = generate_manuscript_dataframe(manuscript_vectors, manuscript_labels, manuscript_label_to_name, manuscript_page_numbers)
    return manuscript_dataset, manuscript_dataframe