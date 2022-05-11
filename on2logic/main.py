import pandas as  pd
import numpy as np
import torch
import timm
from scipy import spatial
import seaborn as sns
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
    
    page_numbers = [name[0].split('.')[-2].split('-')[-1] for name in image_dataset.imgs] # page numbers numerically not strictly correct
    return image_dataset, image_label_to_name, page_numbers

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


def plot_image_from_index(image_dataset, image_index, gray=True, title=None, ax=None):
    sns.set_style('white')
    image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=128, shuffle=False)

    image_batch, label_batch = next(iter(image_dataloader))
    
    image_tensor = image_batch[image_index]
    image_numpy = image_tensor.numpy()
    image_numpy=np.swapaxes(image_numpy,0,1)
    image_numpy=np.swapaxes(image_numpy,1,2)

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        plt.imshow(image_numpy)
        if title is not None:
            plt.title(title)
        plt.show()
    else:
        ax.imshow(image_numpy)
        if title is not None:
            ax.set_title(title)
    return


def generate_manuscript_dataframe(image_vectors, image_labels, label_dictionary, page_numbers):
    manuscript_dataframe = pd.DataFrame({'vector': [x for x in image_vectors], 'label': image_labels})
    manuscript_dataframe['manuscript'] = [label_dictionary[label] for label in manuscript_dataframe['label']]
    manuscript_dataframe['page'] = page_numbers
    return manuscript_dataframe


def cosine_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)


def plot_top_n_similar_images_for_query(image_dataset_to_search, 
                                        image_dataframe_to_search: pd.DataFrame, 
                                        image_data_to_query: pd.DataFrame, 
                                        query_index: int, 
                                        n: int = 10, 
                                        search_type: str ='same',
                                        manuscript_name_to_search:str = 'astro-christ',
                                        plot_rows = True):
    
    row_for_this_image = image_dataframe_to_search.iloc[query_index]
    vector_for_this_image = row_for_this_image['vector']
    manuscript_for_this_image = row_for_this_image['manuscript']
    
    search_dataframe_grouped_by_manuscript = image_dataframe_to_search.groupby(by = 'manuscript')


    if search_type == 'same':
        rows_from_the_same_manuscript = search_dataframe_grouped_by_manuscript.get_group(manuscript_for_this_image).copy()
        rows_to_search = rows_from_the_same_manuscript.query('manuscript.isin([@manuscript_for_this_image])')
    elif search_type == 'different':
        rows_from_the_same_manuscript = search_dataframe_grouped_by_manuscript.get_group(manuscript_for_this_image).copy()
        rows_to_search = rows_from_the_same_manuscript.query('~manuscript.isin([@manuscript_for_this_image])')
    elif search_type == 'specific':
        rows_to_search = search_dataframe_grouped_by_manuscript.get_group(manuscript_name_to_search).copy()
    
    elif search_type == 'all':
        rows_to_search = image_dataframe_to_search.copy()
    else:
        raise ValueError('search_type must be one of "same", "different", "all"')

    rows_to_search['cosine'] = rows_to_search['vector'].apply(lambda x: cosine_similarity(x, vector_for_this_image))
    top_n = rows_to_search[['manuscript', 'cosine']].sort_values(by='cosine', ascending=False).index.values[:n]
    
    plot_image_from_index(image_dataset=image_dataset_to_search, image_index=query_index, 
                          title=f'Original Image, search_type = ({search_type})')
    plot_image_row(image_dataset_to_search, top_n, rows_to_search)
    # for similar_index in top_n:
    #     plot_image_from_index(image_dataset=image_dataset_to_search,
    #                           image_index=similar_index, 
    #                           title=f'similarity = {rows_to_search.loc[similar_index, "cosine"]:.3f}, page_number = {rows_to_search.loc[similar_index, "page"]}')
        
    return

def plot_image_row(image_dataset, indices, rows):
    sns.set_style('white')

    fig, axs = plt.subplots(figsize=(20,10), ncols=len(indices))
    for i,index in enumerate(indices):
        plot_image_from_index(image_dataset=image_dataset,
                              image_index=index, 
                              ax=axs[i],
                              title=f'similarity = {rows.loc[index, "cosine"]:.3f}\npage_number = {rows.loc[index, "page"]}')
        
    plt.show()
    return

def similarity_histogram(image_dataset_to_search, 
                        image_dataframe_to_search: pd.DataFrame, 
                        image_data_to_query: pd.DataFrame, 
                        query_index: int, 
                        search_type: str ='same',
                        manuscript_name_to_search:str = 'astro-christ',
                        plot_rows = True):

    sns.set_style('white')
    sns.set_palette('deep')

    row_for_this_image = image_dataframe_to_search.iloc[query_index]
    vector_for_this_image = row_for_this_image['vector']
    manuscript_for_this_image = row_for_this_image['manuscript']
    search_dataframe_grouped_by_manuscript = image_dataframe_to_search.groupby(by = 'manuscript')

    if search_type == 'same':
        rows_from_the_same_manuscript = search_dataframe_grouped_by_manuscript.get_group(manuscript_for_this_image).copy()
        rows_to_search = rows_from_the_same_manuscript.query('manuscript.isin([@manuscript_for_this_image])')
    elif search_type == 'different':
        rows_from_the_same_manuscript = search_dataframe_grouped_by_manuscript.get_group(manuscript_for_this_image).copy()
        rows_to_search = rows_from_the_same_manuscript.query('~manuscript.isin([@manuscript_for_this_image])')
    elif search_type == 'specific':
        rows_to_search = search_dataframe_grouped_by_manuscript.get_group(manuscript_name_to_search).copy()
    
    elif search_type == 'all':
        rows_to_search = image_dataframe_to_search.copy()
    else:
        raise ValueError('search_type must be one of "same", "different", "all"')

    rows_to_search['cosine'] = rows_to_search['vector'].apply(lambda x: cosine_similarity(x, vector_for_this_image))

    plot_image_from_index(image_dataset=image_dataset_to_search, image_index=query_index, 
                          title=f'Original Image, search_type = ({search_type})')    
    
    fig = plt.figure(figsize=(20,10))
    # sns.set(rc={'figure.figsize':(14, 14)})
    ax = plt.gca()
    sns.histplot(data=rows_to_search, 
                x='cosine', 
                hue='manuscript', 
                multiple='dodge',
                ax=ax,
                )
    sns.kdeplot(data=rows_to_search, 
                x='cosine', 
                hue='manuscript', 
                multiple='layer',
                ax=ax,
                )
    plt.title(f'Histogram of Similarity for a page from {manuscript_for_this_image}')
    plt.show()
    return
    
    
def case_study_setup(parent_dirname):
    image_dirname = parent_dirname+'/data/images'
    numpy_dirname = parent_dirname+'/notebooks/numpy/'
    manuscript_dataset, manuscript_label_to_name, manuscript_page_numbers = load_image_folder_and_return_dataset(image_dirname)

    manuscript_vectors = np.load(numpy_dirname+'vectors.npy')
    manuscript_labels = np.load(numpy_dirname+'labels.npy')
    
    manuscript_dataframe = generate_manuscript_dataframe(manuscript_vectors, manuscript_labels, manuscript_label_to_name, manuscript_page_numbers)
    
    return manuscript_dataset, manuscript_dataframe