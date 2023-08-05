import os
import json
import pandas as pd

from utils.utils import df_siamese_to_single
from tqdm import tqdm

def get_filenames_not_saved(files):
    files_to_remove = []
    print('check files..')
    for filename in tqdm(files):
        filename_dst = CONFIG["images_path_ssd"] + filename 
        if CONFIG["img_format"] in filename_dst:
            if not os.path.exists(filename_dst):
                files_to_remove.append(filename)
        else: 
            files_to_remove.append(filename)
    return files_to_remove

def handler(whitelist_path):
    print('checking ssd images - path:', whitelist_path)
    df = pd.read_csv(whitelist_path)
    files_to_remove = list(set(get_filenames_not_saved(df.file_name_x) + get_filenames_not_saved(df.file_name_y)))
    print('files to remove:', files_to_remove)
    for c in ['file_name_x', 'file_name_y']:
        for i in files_to_remove:
            idx = df[df[c] == i].index
            df.iloc[idx, df.columns.get_loc('location')] = None
    df.dropna().to_csv(whitelist_path, index=False)
    df_siamese_to_single(df).to_csv(whitelist_path.replace('siamese', 'no-siamese'), index=False)


MAPPER = json.load(open('/home/luiz/experiments/.whitelist/.scripts/utils/map.json'))

DATASETS = ['serengeti', 'caltech', 'wcs']
for NAME in DATASETS:
    CONFIG = MAPPER[NAME]

    handler(CONFIG['path_target_time_siamese'] + 'short/train.csv')
    handler(CONFIG['path_target_time_siamese'] + 'short/val.csv')
    handler(CONFIG['path_target_time_siamese'] + 'short/test.csv')