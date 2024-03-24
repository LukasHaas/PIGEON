import os
import logging
import pygeos
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from geopandas import GeoDataFrame
from torch import Tensor
from typing import Dict, List, Any
from shapely.geometry import Point
from datasets import DatasetDict
from transformers import AutoFeatureExtractor
from config import *
from shapely import wkt
from functools import partial
from preprocessing.embed import embed_images

logger = logging.getLogger('preprocessing')

def change_labels_for_classification(dataset: DatasetDict) -> DatasetDict:
    """Changes regression to classification labels

    Args:
        dataset (DatasetDict): original dataset

    Returns:
        DatasetDict: modified dataset
    """
    print(dataset['train']['labels_clf'])
    dataset = dataset.remove_columns(['labels_clf'])
    geocell_df = load_geocell_df()

    train_labels = generate_cell_labels_vector(dataset['train']['labels'], geocell_df['polygon'])
    val_labels = generate_cell_labels_vector(dataset['val']['labels'], geocell_df['polygon'])
    test_labels = generate_cell_labels_vector(dataset['test']['labels'], geocell_df['polygon'])

    dataset = DatasetDict(
        train=dataset['train'].add_column('labels_clf', train_labels.numpy()),
        val=dataset['val'].add_column('labels_clf', val_labels.numpy()),
        test=dataset['test'].add_column('labels_clf', test_labels.numpy())
    )

    dataset = dataset.with_format('torch')
    return dataset

def load_geocell_df(path: str=GEOCELL_PATH) -> gpd.GeoDataFrame:
    """Loads geocell dataframe

    Returns:
        gpd.GeoDataFrame: geocell dataframe
    """
    geocell_df = pd.read_csv(path)
    geocell_df['polygon'] = geocell_df['polygon'].apply(wkt.loads)
    geocell_df = gpd.GeoDataFrame(geocell_df, geometry=gpd.points_from_xy(geocell_df.lng, geocell_df.lat), crs='EPSG:4326')
    geocell_df = geocell_df.set_geometry('geometry')
    return geocell_df

def generate_cell_labels(point: Point, geocell_df: GeoDataFrame,
                         one_hot: bool=True) -> np.ndarray:
    """Generates one-hot labels for classification task

    Args:
        point (Point): point to find label for
        polygons (List): list of geocell polygons
        one_hot (bool, optional): whether to use one-hot labels.
            Defaults to False.

    Returns:
        np.ndarray: one-hot label
    """
    if one_hot:
        one_hot_array = np.zeros((len(geocell_df.index)))
    try:
        indices = geocell_df.sindex.query(point, predicate='covered_by')
        if len(indices) > 0:
            if one_hot:
                one_hot_array[indices[0]] = 1
                return one_hot_array
            else:
                return indices[0]

    except pygeos.GEOSException:
        print(f'Point {point} could not be assigned to a geocell.')
        pass
        
    # Assigning point to closest geocell
    closest_cell = geocell_df.sindex.nearest(point)[1][0]
    if one_hot:
        one_hot_array[closest_cell] = 1
        return one_hot_array
    else:
        return closest_cell

def _add_heading(heading: float, add_times_90: float):
    return (heading + (add_times_90 * (np.pi / 2))) % (2 * np.pi)

def preprocess_heading(example: Dict) -> Dict:
    """Transforms heading via sine and cosine.

    Args:
        example (Dict): data sample

    Returns:
        Dict: modified data sample
    """
    heading = np.deg2rad(example['heading'])
    return {
        'heading': np.array([[np.sin(heading), np.cos(heading)],
                             [np.sin(_add_heading(heading, 1)), np.cos(_add_heading(heading, 1))],
                             [np.sin(_add_heading(heading, 2)), np.cos(_add_heading(heading, 2))],
                             [np.sin(_add_heading(heading, 3)), np.cos(_add_heading(heading, 3))]])
    }

def generate_cell_labels_vector(labels: Tensor, polygons: List) -> np.ndarray:
    """Generates one-hot labels for classification task

    Args:
        point (Point): point to find label for
        polygons (List): list of geocell polygons

    Returns:
        np.ndarray: one-hot label
    """
    one_hots = torch.zeros((labels.size(0), len(polygons)))
    for i, polygon in enumerate(polygons):
        coords = list(polygon.exterior.coords)[:4]
        right_upper, left_lower = torch.tensor(coords[0]), torch.tensor(coords[2])
        mask = torch.all((labels <= right_upper) * (labels >= left_lower), dim=1)
        one_hots[mask, i] = 1

    return torch.argmax(one_hots, dim=-1)


def generate_label_cells(example: Dict, geocell_df: gpd.GeoDataFrame=None,
                         one_hot: bool=True) -> Dict:
    """Generates labels from longitude and latitude.

    Args:
        example (Dict): dataset sample
        geocell_df (GeoDataFrame, optional): dataframe containing geocell centroids and polygons.
        one_hot (bool, optional): whether one-hot labels should be generated.
            Defaults to True.

    Returns:
        Dict: modified dataset sample
    """
    try:
        longitude, latitude = example['lng'], example['lat']
    except KeyError:
        longitude, latitude = example['labels'][0], example['labels'][1]

    label_dict = {}
    assert geocell_df is not None, \
        'In the classification setting, the geocell dataframe is required.'

    point = Point(longitude, latitude)
    label_dict = {
        'labels': np.array([longitude, latitude]).transpose(),
        'labels_clf': generate_cell_labels(point, geocell_df, one_hot)
    }

    return label_dict

def generate_label_mt(example: Dict) -> Dict:
    """Generates labels for the multitask setup.

    Args:
        example (Dict): dataset sample

    Returns:
        Dict: modified dataset sample
    """
    label_dict = {}
    values = [example['elevation'], example['population'], example['temp_avg'], \
                example['temp_diff'], example['prec_avg'], example['prec_diff']]
    label_dict['labels_multi_task'] = np.array(values).transpose()

    return label_dict

def extract_features(example: Dict, feature_extractor: AutoFeatureExtractor) -> Dict:
    """Extracts features from images for vision input

    Args:
        example (Dict): data sample
        feature_extractor (AutoFeatureExtractor): vision feature extractor

    Returns:
        Dict: modified data sample
    """
    x = {}
    x['pixel_values'] = feature_extractor(images=example['image'], return_tensors='pt')['pixel_values']
    if 'image_2' in example:
        for i in range(2, 5):
            x[f'pixel_values_{i}'] = feature_extractor(images=example[f'image_{i}'],
                                                       return_tensors='pt')['pixel_values']

        x['pixel_values'] = torch.concat((x['pixel_values'], x['pixel_values_2'], x['pixel_values_3'],
                                          x['pixel_values_4']), dim=1)

    return {
        'pixel_values': x['pixel_values'],
    }

def extract_features_yfcc(example: Dict, feature_extractor: AutoFeatureExtractor) -> Dict:
    """Extracts features from images for vision input from YFCC data.

    Args:
        example (Dict): data sample
        feature_extractor (AutoFeatureExtractor): vision feature extractor

    Returns:
        Dict: modified data sample
    """
    pixel_values = feature_extractor(images=example['image'], return_tensors='pt')['pixel_values']
    return {
        'image': pixel_values
    }

def get_embeddings(example: Dict, embedder: Any) -> Dict:
    """Computes embeddings of the given image.

    Args:
        example (Dict): Dataset entry.
        embedder (CLIPEmbedding): Model to use for embeddings.

    Returns:
        Dict: Modified dataste entry.
    """
    embeddings = embedder(example['image'])
    return {
        'embedding': embeddings
    }

def add_embeddings(example: Dict, index: int, emb_array: np.ndarray) -> np.ndarray:
    """Adds embeddings for the given image."""
    return {
        'embedding': emb_array[index]
    }

def __find_index(index: int, indices: np.ndarray) -> int:
    """Finds index in indices array.

    Args:
        index (int): _description_
        indices (np.ndarray): _description_

    Returns:
        int: _description_
    """
    real_index = index
    flat_index = indices.flatten()
    while flat_index[real_index] != index and real_index >= 0:
        real_index -= 1

    if real_index < 0:
        raise Exception('Index not found.')
        
    return real_index

def preprocess(dataset: DatasetDict, geocell_path: str, embedder: Any=None,
               multi_task: bool=False) -> DatasetDict:
    """Preproccesses image dataset for vision input

    Args:
        dataset (DatasetDict): image dataset.
        geocell_path (str): path to geocells.
        embedder (CLIPEmbedding): CLIP embedding model. Defaults to None.
        multi_task (bool, optional): if labels for multi-task setup should be generated.
            Defaults to False.

    Returns:
        DatasetDict: transformed dataset
    """
    # Load geocells
    geocell_df = load_geocell_df(geocell_path)
    geocell_df = gpd.GeoDataFrame(geocell_df, geometry='polygon', crs='EPSG:4326')

    # Preprocess heading
    if geocell_path == GEOCELL_PATH:
        dataset = dataset.map(preprocess_heading, keep_in_memory=True)

    # Preprocess impages
    if embedder is not None:
        
        # Compute embeddings and save to disk
        #if os.path.exists(f'data/yfcc_embeddings/train.npy') == False:
        # embed_images(embedder, dataset)

        # # Load embeddings from disk
        for split_name, split_dataset in dataset.items():
            print(f'... loading file: {split_name}.npy')

            num_samples = len(split_dataset)
            embeds = np.load(f'data/landmark_embeddings/{split_name}.npy')
            indices = np.load(f'data/landmark_embeddings/{split_name}_indices.npy')

            arg_indices = np.argsort(indices.flatten()[:num_samples])
            embeds = embeds.reshape((-1, 1024))[arg_indices]

            add_embeds = partial(add_embeddings, emb_array=embeds)
            dataset[split_name] = split_dataset.map(add_embeds, with_indices=True,
                                                    keep_in_memory=False, num_proc=1)

        # Delete columns
        cols = ['image']
        if geocell_path == GEOCELL_PATH:
            cols += ['image_2', 'image_3', 'image_4']

        dataset = dataset.remove_columns(cols)
        dataset.save_to_disk('data/hf_landmarks')

    else:
        feature_extractor = AutoFeatureExtractor(CLIP_MODEL)
        dataset = dataset.map(lambda x: extract_features(x, feature_extractor), 
                            batch_size=128, batched=True, num_proc=64)

    # Create labels for cells
    dataset = dataset.map(lambda x: generate_label_cells(x, geocell_df, False),
                          num_proc=8, keep_in_memory=False)  # TODO: batch label creation
    dataset = dataset.remove_columns(['lng', 'lat'])

    # Create labels for multi-task
    if multi_task:
        dataset = dataset.map(generate_label_mt, num_proc=8, keep_in_memory=False)  # TODO: batch creation
        
    cols = ['elevation', 'population', 'temp_avg', 'temp_diff', 'prec_avg', 'prec_diff']
    dataset = dataset.remove_columns(cols)
        
    dataset = dataset.with_format('torch')
    return dataset
