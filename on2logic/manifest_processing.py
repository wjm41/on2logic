import requests
from io import BytesIO

import numpy as np
from PIL import Image
from tqdm import tqdm
from fire import Fire

from on2logic.model import load_image_model, generate_vector_for_pil_image, get_device
from on2logic.data import item_name_dataloader, default_folder_name_for_item, image_url_from_image_id
from on2logic.utils import return_default_transforms


def process_item_name(item_name, data_dir:str = 'data/images/cudl'):
    device = get_device()
    
    image_model = load_image_model()
    images_in_item = list(item_name_dataloader(item_name)) # convert generator item to memory in list
    number_of_images_in_item = sum(1 for _ in images_in_item)
    transform = return_default_transforms()
    
    image_vectors_for_item = []
    for image_id in tqdm(images_in_item, total=number_of_images_in_item):
        image_url = image_url_from_image_id(image_id)

                
        ImgRequest = requests.get(image_url)
        if ImgRequest.status_code == requests.codes.ok:
            pil_for_image_id = Image.open(BytesIO(ImgRequest.content))
            image_vector = generate_vector_for_pil_image(pil_for_image_id, image_model, transform, device)
            image_vectors_for_item.append(image_vector)
        else:
            print(ImgRequest.status_code)
            raise Exception
    image_vectors_for_item = np.vstack(image_vectors_for_item)
    np.save(f'{data_dir}/{item_name}/image_vectors.npy', image_vectors_for_item)
    return


if __name__ == '__main__':
    Fire(process_item_name)