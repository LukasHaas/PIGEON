from os import path
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Image
from preprocessing import GeoAugmentor
from config import METADATA_PATH, IMAGE_PATH

CLIMATE_DICT = {
    'Arid, desert, cold': 0,
    'Arid, desert, hot': 1,
    'Arid, steppe, cold': 2,
    'Arid, steppe, hot': 3,
    'Cold, dry summer, cold summer': 4,
    'Cold, dry summer, hot summer': 5,
    'Cold, dry summer, warm summer': 6,
    'Cold, dry winter, cold summer': 7,
    'Cold, dry winter, hot summer': 8,
    'Cold, dry winter, warm summer': 9,
    'Cold, no dry season, cold summer': 10,
    'Cold, no dry season, hot summer': 11,
    'Cold, no dry season, very cold winter': 12,
    'Cold, no dry season, warm summer': 13,
    'Polar, frost': 14,
    'Polar, tundra': 15,
    'Temperate, dry summer, cold summer': 16,
    'Temperate, dry summer, hot summer': 17,
    'Temperate, dry summer, warm summer': 18,
    'Temperate, dry winter, cold summer': 19,
    'Temperate, dry winter, hot summer': 20,
    'Temperate, dry winter, warm summer': 21,
    'Temperate, no dry season, cold summer': 22,
    'Temperate, no dry season, hot summer': 23,
    'Temperate, no dry season, warm summer': 24,
    'Tropical, monsoon': 25,
    'Tropical, rainforest': 26,
    'Tropical, savannah': 27
 }

def create_dataset_split(df: pd.DataFrame, shuffle: bool=False,
                         panorama: bool=False) -> Dataset:
    """Creates an image dataset from the given dataframe.

    Args:
        df (pd.DataFrame): metadata dataframe
        shuffle (bool): if dataset should be shuffled
        panorama (bool, optional): whether four images are passed in as a panorama.
            Defaults to False.

    Returns:
        Dataset: HuggingFace dataset
    """
    image_col = 'image_path' if 'image_path' in df.columns else 'image'
    data_dict = {
        'image': df[image_col].tolist(),
        'lng': df['lng'].values,
        'lat': df['lat'].values,
        'elevation': df['elevation_reg'],
        'population': df['population_reg'],
        'temp_avg': df['temp_avg_reg'],
        'temp_diff': df['temp_diff_reg'],
        'prec_avg': df['prec_avg_reg'],
        'prec_diff': df['prec_diff_reg'],
        'index': df.index.values
    }

    if 'climate' in df.columns:
        data_dict['labels_climate'] = df['climate'].values

    if 'climate_zone' in df.columns:
        data_dict['labels_climate'] = df['climate_zone'].values

    if type(data_dict['labels_climate'][0]) == str:
        data_dict['labels_climate'] = np.array([CLIMATE_DICT[x] for x in data_dict['labels_climate']])

    if 'heading' in df.columns:
        data_dict['heading'] = df['heading']

    if 'month' in df.columns:
        data_dict['labels_month'] = df['month']

    if panorama:
        data_dict['image_2'] = df['image_2'].tolist()
        data_dict['image_3'] = df['image_3'].tolist()
        data_dict['image_4'] = df['image_4'].tolist()

    dataset = Dataset.from_dict(data_dict).cast_column('image', Image())
    if panorama:
        dataset = dataset.cast_column('image_2', Image())
        dataset = dataset.cast_column('image_3', Image())
        dataset = dataset.cast_column('image_4', Image())

    if shuffle:
        dataset = dataset.shuffle(seed=330)
    
    return dataset

def generate_finetune_dataset(sample: int=None, geo_augment: bool=True, 
                              metadata_path: str=METADATA_PATH,
                              image_path: str=IMAGE_PATH) -> DatasetDict:
    """Generates the Geoguessr dataset

    Args:
        sample (int, optional): how many samples to pick from dataset. Defaults to None (all).
        geo_augment (bool, optional): whether to augment the dataset with additional geo data
            Defaults to True.
        metadata_path (str, optional): metadata path. Defaults to METADATA_PATH.
        image_path (str, optional): image folder path. Defaults to IMAGE_PATH.

    Returns:
        DatasetDict: HuggingFace DatasetDict
    """
    panorama = False
    data_df = pd.read_csv(metadata_path)

    # Process image paths
    image_cols = [x for x in data_df.columns if 'image' in x]
    if len(image_cols) == 1:
        print('Detected single image dataset.')
        if 'image_idx' in data_df.columns:
            data_df['image_path'] = data_df.apply(lambda row: path.join(image_path, f'{row.image_idx}.jpg'), axis=1)
            data_df = data_df.rename(columns={'image': 'image_path'})

        else:
            data_df['image'] = data_df.apply(lambda row: path.join(image_path, row.image), axis=1)

    else:
        print('Detected multi-image dataset.')
        panorama = True
        for col in image_cols:
            data_df[col] = data_df[col].apply(lambda x: path.join(image_path, x))

    if sample is not None:
        data_df = data_df.sample(int(sample)).copy()

    if geo_augment and 'geo_area' not in data_df.columns:
        augmentor = GeoAugmentor()
        data_df = augmentor(data_df)

    splits = []
    for split in ['train', 'val', 'test']:
        data_split = data_df.loc[data_df['selection'] == split]
        splits.append(create_dataset_split(data_split, panorama=panorama))

    dataset = DatasetDict(
        train=splits[0],
        val=splits[1],
        test=splits[2]
    )
    
    return dataset

if __name__ == '__main__':
    dataset = generate_finetune_dataset()
    print(dataset)