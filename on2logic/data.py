#%%
import re
import requests
from pathlib import Path
from os.path import abspath, dirname

import pandas as pd

module_dir = dirname(abspath(__file__))
import time 

from fire import Fire


    # item = "MS-NN-00003-00074"
def load_iiif_manifest_from_item_name(item_name):
    
    manifest_url = "https://cudl.lib.cam.ac.uk/iiif/" + item_name
    r = requests.get(manifest_url)
    json_data = r.json()

    return json_data

def download_and_save_single_image(image_id: str,
                                   save_folder:str = None,
                                   relax:bool = False,):
    print(f'Image ID: {image_id}')
    # IIIF
    download_image = 'https://images.lib.cam.ac.uk/iiif/' + image_id + '/full/max/0/default.jpg'

    print(f'URL: {download_image}')

    file_name = image_id.replace('.jp2', '.jpg')
    save_path = f'{save_folder}/{file_name}'

    if relax:
        # Backoff to prevent the IIIF server being overloaded
        time.sleep(0.1)

    # Exception Handling for invalid requests
    try:
        # Creating a request object to store the response
        ImgRequest = requests.get(download_image)

        # Verifying whether the specified URL exist or not
        if ImgRequest.status_code == requests.codes.ok:
            # Opening a file to write bytes from response content
            # Storing this object as an image file on the hard drive
            img = open(save_path, "wb")
            img.write(ImgRequest.content)
            img.close()

            print(f'Saved: {save_path}\n')

        else:
            print(ImgRequest.status_code)
    except Exception as e:
        print(str(e))
    return download_image

def default_folder_name_for_item(item_name):
    project_dir = Path(abspath(__file__)).parents[1]

    default_folder_name = f'{project_dir}/data/images/cudl/{item_name}'
    
    return default_folder_name

def loop_through_json_data_and_save_images(item_name:str ,
                                           json_data, 
                                           save_folder:str = None,
                                           relax:bool = False,
                                           save_metadata: bool = True):
    
    if save_folder is None:
        save_folder = default_folder_name_for_item(item_name)
        # print(save_folder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    
    image_labels = []
    image_urls = []
    for sequence in json_data['sequences']:

        for canvas in sequence['canvases']:

            for image in canvas['images']:

                iiif_image = image['resource']['@id']

                image_id = iiif_image.rsplit('/', 1)[-1]
                # print(image_id)
                image_url = download_and_save_single_image(image_id, save_folder, relax)
                

                image_labels.append(canvas['label'])
                image_urls.append(image_url)
    if save_metadata:
        
        full_item_label = json_data['label']
        short_item_label = re.search('\((.*)\)', full_item_label).group(1)
        metadata_dataframe = pd.DataFrame({'img-label': image_labels,
                                           'img-url': image_urls})
        metadata_dataframe['item-name'] = item_name
        metadata_dataframe['item-label'] = short_item_label
        
        metadata_dataframe = metadata_dataframe[['item-name', 'item-label', 'img-label', 'img-url']]
        metadata_dataframe.to_csv(f'{save_folder}/img_data.csv', index=False)
        return metadata_dataframe
    else:
        return
        

def download_images_from_item_name(item_name:str,
                                   save_metadata:bool = True):

    json_data = load_iiif_manifest_from_item_name(item_name)

    # TODO save json metadata as txt file - display in app
    loop_through_json_data_and_save_images(item_name, 
                                           json_data,
                                           save_metadata=save_metadata)
    
    
def item_name_dataloader(item_name):
    json_data = load_iiif_manifest_from_item_name(item_name)
    
    for sequence in json_data['sequences']:

        for canvas in sequence['canvases']:

            for image in canvas['images']:

                iiif_image = image['resource']['@id']

                image_id = iiif_image.rsplit('/', 1)[-1]
                
                yield image_id
                
#%%
if __name__ == '__main__':
    Fire(download_images_from_item_name)