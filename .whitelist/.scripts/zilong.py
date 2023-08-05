import os
import json
import shutil
import subprocess
import pandas as pd
import tensorflow as tf
import pandas as pd

MAPPER = json.load(open('/home/luiz/experiments/.whitelist/.scripts/utils/map.json'))

ZILONG_PATH_OUTPUT_ANIMAL = MAPPER["models"]["zilong"]["path_output_animal"]
ZILONG_PATH_OUTPUT_EMPTY = MAPPER["models"]["zilong"]["path_output_empty"]
ZILONG_CSV = MAPPER["models"]["zilong"]["path_csv"]
DIR_TMP = MAPPER["models"]["zilong"]["path_tmp"]
PATH_TEST = MAPPER["models"]["zilong"]["path_test"]
ZILONG_RESULTS_PATH = MAPPER["models"]["zilong"]["path_results"]
ZILONG_COMMAND = MAPPER["models"]["zilong"]["command"]

def run_zilong(img_type, input_path, output_path):
    cmd = ZILONG_COMMAND.replace('<INP>', input_path).replace('<OUT>', output_path).replace('<TYPE>', img_type)
    subprocess.call(cmd, shell=True)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_files(files, dir_target):
    create_dir(dir_target)
    for i in files:
        shutil.copy2(i, dir_target)

def get_class(origin):
    with open(origin, 'r') as origin_f:
        return int(origin_f.read() != '')
        
def delete_files(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


for dataset_name in ['caltech', 'serengeti', 'wcs']:
    CONFIG = MAPPER[dataset_name]
    df = pd.read_csv(PATH_TEST.replace('<dataset>', dataset_name))
    result = {
        'pred_confidence': [],
        'pred': [],
        'real': [],
        'file_name_x': [],
        'file_name_y': [],
    }
    for i in df.index:
        copy_files([
            MAPPER[dataset_name]['images_path_ssd'] + df.iloc[i].file_name_x,
            MAPPER[dataset_name]['images_path_ssd'] + df.iloc[i].file_name_y
        ], DIR_TMP)

        run_zilong(CONFIG["img_format"], DIR_TMP, ZILONG_RESULTS_PATH)

        _class = get_class('/home/luiz/experiments/Zilong/output/animal_images.txt')
        pred_class = int(tf.one_hot(_class, 1)[0])
        real_class = int(tf.one_hot(df.iloc[i].category, 1)[0])

        result['pred_confidence'].append(pred_class)
        result['pred'].append(pred_class)
        result['real'].append(real_class)
        result['file_name_x'].append(df.iloc[i].file_name_x)
        result['file_name_y'].append(df.iloc[i].file_name_y)
        delete_files(DIR_TMP)
        
    pd.DataFrame(result).to_csv(ZILONG_CSV.replace('<dataset>', dataset_name), index=False)