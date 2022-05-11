import pandas as  pd
import numpy as np
import torch
import timm
from scipy import spatial
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def load_image_model(model_name:str ='restnet50') -> torch.nn.Module:
    image_model = timm.create_model(model_name, pretrained=True)
    image_model.eval() # this turns off dropout etc, important!
    return image_model
    
def load_image_folder_and_return_dataset(folder_name:str = '../data/images/', image_size:int = 500):

    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size-1),
                                    transforms.ToTensor()])
    
    image_dataset = datasets.ImageFolder(folder_name, transform=transform)
    image_label_to_name =  {value:key for key, value in image_dataset.class_to_idx.items()}
    return image_dataset, image_label_to_name

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


def plot_image_from_index(image_dataset, image_index, gray=True, title=None):
    image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False)

    image_batch, label_batch = next(iter(image_dataloader))
    
    image_tensor = image_batch[image_index]
    image_numpy = image_tensor.numpy()
    image_numpy=np.swapaxes(image_numpy,0,1)
    image_numpy=np.swapaxes(image_numpy,1,2)

    fig = plt.figure(figsize=(10,10))
    plt.imshow(image_numpy)
    if title is not None:
        plt.title(title)
    plt.show()
    return


def generate_manuscript_dataframe(image_vectors, image_labels, label_dictionary):
    manuscript_dataframe = pd.DataFrame({'vector': [x for x in image_vectors], 'label': image_labels})
    manuscript_dataframe['manuscript'] = [label_dictionary[label] for label in manuscript_dataframe['label']]
    return manuscript_dataframe


def cosine_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)


def plot_top_n_similar_images_for_query(image_dataset_to_search, 
                                        image_dataframe_to_search: pd.DataFrame, 
                                        image_data_to_query: pd.DataFrame, 
                                        query_index: int, 
                                        n: int = 10, 
                                        search_type: str ='same'):
    
    row_for_this_image = image_dataframe_to_search.iloc[query_index]
    vector_for_this_image = row_for_this_image['vector']
    manuscript_for_this_image = row_for_this_image['manuscript']

    if search_type == 'same':
        search_dataframe_grouped_by_manuscript = image_dataframe_to_search.groupby(by = 'manuscript')
        rows_from_the_same_manuscript = search_dataframe_grouped_by_manuscript.get_group(manuscript_for_this_image).copy()
        rows_to_search = rows_from_the_same_manuscript.query('manuscript.isin([@manuscript_for_this_image])')
    elif search_type == 'different':
        search_dataframe_grouped_by_manuscript = image_dataframe_to_search.groupby(by = 'manuscript')
        rows_from_the_same_manuscript = search_dataframe_grouped_by_manuscript.get_group(manuscript_for_this_image).copy()
        rows_to_search = rows_from_the_same_manuscript.query('~manuscript.isin([@manuscript_for_this_image])')
    elif search_type == 'all':
        rows_to_search = image_dataframe_to_search.copy()
    else:
        raise ValueError('search_type must be one of "same", "different", "all"')

    rows_to_search['cosine'] = rows_to_search['vector'].apply(lambda x: cosine_similarity(x, vector_for_this_image))
    top_n = rows_to_search[['manuscript', 'cosine']].sort_values(by='cosine', ascending=False).index.values[:n]
    
    plot_image_from_index(image_dataset=image_dataset_to_search, image_index=similar_index, title=f'Original Image')
    for similar_index in top_n:
        plot_image_from_index(image_dataset=image_dataset_to_search, image_index=similar_index, title=f'similarity = {rows_to_search:.3f}')
        
    return