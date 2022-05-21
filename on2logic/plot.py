import pandas as  pd
import numpy as np
import torch
from scipy import spatial
import seaborn as sns
import matplotlib.pyplot as plt

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

    fig, axs = plt.subplots(figsize=(20,20), ncols=len(indices))
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
    print('Loading...')
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
    
