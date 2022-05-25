
import numpy as np
import pandas as pd
from fire import Fire

from on2logic.manifest_processing import process_item_name


def process_case_study_items(data_dir):
    path_to_case_study_items = f'{data_dir}/case_study_list.txt'
    df_items = pd.read_csv(path_to_case_study_items, header=None, names=['item_ids'])
    
    for item in df_items.item_ids:
        process_item_name(item, data_dir=data_dir)
    
    return


if __name__ == '__main__':
    Fire(process_case_study_items)