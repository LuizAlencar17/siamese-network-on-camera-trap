import os
import json
import pandas as pd

from utils.utils import create_dir
from tqdm import tqdm
from PIL import Image

def get_files_from_subset(path):
    print('path:', path)
    subset = pd.read_csv(path)
    all_files = list(set(subset.file_name_x)) + list(set(subset.file_name_y)) 
    return list(set(all_files))

def copy_files(files):
    print('check files in source directory..')
    files_to_copy = []
    for filename in tqdm(files):
        filename_src = CONFIG['images_path'] + filename 
        if CONFIG['img_format'] in filename_src:
            filename_dst = filename_src.replace(CONFIG['images_path'], CONFIG['images_path_ssd'])
            if not os.path.exists(filename_dst):
                files_to_copy.append([filename_src, filename_dst])
            

    print('coping files to ssd..')
    for i in tqdm(files_to_copy):
        try:
            filename_src, filename_dst = i[0], i[1]
            img = Image.open(filename_src)
            img = img.resize((max_width, max_width), Image.ANTIALIAS)
            create_dir('/'.join(filename_dst.split('/')[:-1]))
            img.save(filename_dst)
        except Exception as e:
            print(f'{filename_src}: {e}')

max_width = 256
MAPPER = json.load(open('/home/luiz/experiments/.whitelist/.scripts/utils/map.json'))

DATASETS = ['serengeti', 'caltech', 'wcs']
for NAME in DATASETS:
    CONFIG = MAPPER[NAME]

    files = get_files_from_subset(CONFIG['path_target_time_siamese'] + 'short/train.csv')
    copy_files(files)
    files = get_files_from_subset(CONFIG['path_target_time_siamese'] + 'short/val.csv')
    copy_files(files)
    files = get_files_from_subset(CONFIG['path_target_time_siamese'] + 'short/test.csv')
    copy_files(files)