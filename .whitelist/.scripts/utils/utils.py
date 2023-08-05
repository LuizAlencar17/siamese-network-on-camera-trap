import os
import pandas as pd

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def df_siamese_to_single(df):
    return df.dropna().rename(columns={'file_name_x': 'file_name'})[['file_name','category','location']]

def get_different_between_two_dates(d1, d2):
    start_day = d2.replace(hour=0, minute=0, second=0, microsecond=0)
    return abs((d1 - start_day).seconds - (d2 - start_day).seconds)

def get_seconds_from_date(d):
    return (d - d.replace(hour=0, minute=0, second=0, microsecond=0)).seconds

def save_data_backup_with_pairs(subset, CONFIG):
    create_dir(CONFIG['path_target_time_siamese'])
    subset.to_csv(CONFIG['path_target_time_siamese'] + 'time_backup.csv', index=False)

def load_data_backup_with_pairs(CONFIG):
    return pd.read_csv(CONFIG['path_target_time_siamese'] + 'time_backup.csv')