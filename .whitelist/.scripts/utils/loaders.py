import json
import datetime
import pandas as pd

def get_annotations(path):
    with open(path) as handle:
        return json.loads(handle.read())

def get_day_time(time):
    day_start = datetime.time(0, 0, 0)
    morning_start = datetime.time(6, 0, 0) 
    morning_end = datetime.time(14, 0, 0)
    afternoon_end = datetime.time(18, 0, 0)
    night_end = datetime.time(23, 59, 59)
    status = ''
    if morning_start <= time <= morning_end:
        status = 'morning'
    if morning_end <= time <= afternoon_end:
        status = 'afternoon'
    if (afternoon_end <= time <= night_end) or (day_start <= time <= morning_start):
        status = 'night'
    return status


def get_serengeti_dataset(CONFIG):
        metadata = get_annotations(CONFIG['metadata'])
        df = pd.DataFrame(metadata['annotations'])

        df['season_num'] = df['season'].map(lambda a: int(a[-2:].replace('S', '')))
        df_filter = df[df['season_num'] < 7]

        df_filter['file_name'] = df_filter['image_id'].map(lambda a: a + '.JPG')
        df_filter['location'] = df_filter['file_name'].map(lambda a: a.split('/')[0]+'_'+a.split('/')[2])

        df_filter['datetime'] = pd.to_datetime(df_filter['datetime'])
        df_filter['day_time'] = df_filter['datetime'].map(lambda a: get_day_time(a.time()))
        df_filter = df_filter.rename(columns={"category_id": "category"})
        
        df_filter = df_filter[['file_name', 'location', 'day_time', 'datetime', 'category']]
        df_filter.category = df_filter.category.map(lambda a: int(a != 0) )
        return df_filter


def get_caltech_dataset(CONFIG):
        metadata = get_annotations(CONFIG['metadata'])
        df = pd.DataFrame(metadata['images'])
        annotations = pd.DataFrame(metadata['annotations'])

        df['datetime'] = pd.to_datetime(df['date_captured'], errors = 'coerce')
        df = df.iloc[df['datetime'].dropna().index].reset_index(drop=1)
        df['day_time'] = df['datetime'].map(lambda a: get_day_time(a.time()))

        df = df.rename(columns={"id": "image_id"})
        annotations = annotations.rename(columns={"category_id": "category"})

        df = pd.merge(df, annotations, on="image_id", how="right")
        
        empty_class = 30
        df.category = (df.category != empty_class).replace(True, 1).replace(False, 0)
        df = df[['file_name', 'location', 'day_time', 'datetime', 'category']].dropna()
        df.file_name = df.file_name.map(lambda a: 'cct_images/' + a.replace('.jpg', '.JPG'))
        return df

def get_wellington_dataset(CONFIG):
        metadata = get_annotations(CONFIG['metadata'])
        df = pd.DataFrame(metadata['images'])
        annotations = pd.DataFrame(metadata['annotations'])

        # 12 id is unclassifiable
        # 2 id is empty class
        empty_class = 2
        annotations = annotations[annotations.category_id != 12]
        annotations['category'] = annotations.category_id.map(lambda a: int(a != empty_class))
        df['datetime'] = pd.to_datetime(df['datetime'], errors = 'coerce')
        df = df.iloc[df['datetime'].dropna().index].reset_index(drop=1)
        df['day_time'] = df['datetime'].map(lambda a: get_day_time(a.time()))
        df = df.rename(columns={"id": "image_id"})

        dataset = pd.merge(df, annotations, on="image_id", how="right")
        dataset['location'] = dataset['site'] + dataset['camera']
        return dataset[['file_name', 'location', 'day_time', 'datetime', 'category']]

def get_wcs_dataset(CONFIG):
    metadata = get_annotations(CONFIG['metadata'])
    df = pd.DataFrame(metadata['images'])
    annotations = pd.DataFrame(metadata['annotations'])
    empty_class = 0
    annotations['category'] = annotations.category_id.map(lambda a: int(a != empty_class))
    df = df[df.corrupt == False]
    df['datetime'] = pd.to_datetime(df['datetime'], errors = 'coerce')
    df = df.loc[df['datetime'].dropna().index].reset_index(drop=1)
    df['day_time'] = df['datetime'].map(lambda a: get_day_time(a.time()))
    df = df.rename(columns={"id": "image_id"})

    dataset = pd.merge(df, annotations, on="image_id", how="right")
    dataset = dataset[['file_name', 'location', 'day_time', 'datetime', 'category']].dropna()
    dataset['location'] = dataset.index.map(lambda a: dataset.file_name[a].split('/')[1] + str(dataset.location[a]))
    return dataset