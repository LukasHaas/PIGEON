import logging
import joblib
import numpy as np
import geopandas as gpd
from typing import Dict, List
from preprocessing import haversine_np
from shapely.geometry import Point, MultiPolygon
from config import DECAY_CONSTANT, SCALER_PATH, SCALER_PATH_YFCC, COUNTRY_PATH
from sklearn.metrics import accuracy_score

CRS = 'EPSG:4326'

# Logger
logger = logging.getLogger('evaluation')

# Load country geojson
geo_df = gpd.read_file(COUNTRY_PATH)
geo_df = geo_df.set_crs(crs=CRS)
geo_df['geometry'] = geo_df['geometry'].apply(lambda x: x.buffer(0))
country_shapes = geo_df['geometry'].values

def mae(labels: np.ndarray, preds: np.ndarray) -> float:
    error = np.mean(np.abs(labels - preds))
    if type(error) == tuple:
        return error[0]
    
    return error

def recover_regression_values(values: np.ndarray, yfcc: bool=False) -> np.ndarray:
    """Reconstructs the multi-task regression values.

    Args:
        values (np.ndarray): array of predictions or labels
        yfcc (bool, optional): whether yfcc input data was used.

    Returns:
        np.ndarray: recovered regression values
    """
    # Variables: ['elevation_reg', 'population_reg', 'temp_avg_reg',
    #             'temp_diff_reg', 'prec_avg_reg', 'prec_diff_reg']

    # Scaler
    path = SCALER_PATH_YFCC if yfcc else SCALER_PATH
    scaler = joblib.load(path)
    vals = scaler.inverse_transform(values)

    # All variariables except average temperature have been log-transformed
    vals[:, :2] = np.exp(vals[:, :2])
    vals[:, 3:] = np.exp(vals[:, 3:])

    # Reconstruct offset
    offset_val = 416 if yfcc else 408
    vals = vals - np.array([offset_val, 1, 0, 1, 1, 1]).transpose()
    return vals

def find_country(point: Point, countries: List[MultiPolygon]=country_shapes) -> MultiPolygon:
    """Finds the country a given point lies in.

    Args:
        point (Point): shapely point
        countries (List[MultiPolygon], optional): list of country polygons

    Returns:
        MultiPolygon: country polygone, None if not found
    """
    country = geo_df['geometry'].sindex.query(point, predicate='covered_by')
    if len(country > 0):
        return geo_df.iloc[country[0]]['geometry']
        
    logger.warn(f'Point not part of any country polygon: {point}')
    country = geo_df['geometry'].sindex.nearest(point, return_all=False)[1]
    return geo_df.iloc[country[0]]['geometry']

def country_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Computes if the country was predicted correctly

    Args:
        predictions (np.ndarray): array of (longitude, latitude)
        labels (np.ndarray): array of (longitude, latitude)

    Returns:
        float: country prediction accuracy
    """
    p_points = gpd.points_from_xy(predictions[:, 0], predictions[:, 1], crs=CRS)
    l_points = gpd.points_from_xy(labels[:, 0], labels[:, 1], crs=CRS)
    countries = [find_country(point) for point in l_points]
    correct = np.array([c.contains(p) if c is not None else False for c, p in zip(countries, p_points)], dtype=int)
    return np.mean(correct)

def percentage_within_radius(distances: np.ndarray, km: float) -> float:
    """Calculates what percentage of 

    Args:
        distances (np.ndarray): distances in kilometers
        km (float): maximum radius

    Returns:
        float: percentage of points within radius
    """
    percentage = (distances < km).sum() / len(distances)
    return percentage

def geoguessr_score(distances: np.ndarray) -> float:
    """Calculates the mean Geoguessr score of the batch

    Args:
        distances (np.ndarray): haversine distances between labels 
                                and predictions in km

    Returns:
        float: mean Geoguessr batch score
    """
    # Geoguessr exponential decay scoring function
    scores = np.round(5000 * np.exp(-distances / DECAY_CONSTANT))
    return np.mean(scores)

def topk_geocell_accuracy(cell_labels: np.ndarray, topk_preds: np.ndarray) -> float:
    """Computes top-5 geocell accuracy.

    Note:
        This metric is important for developing good guess refinement models
        as the cell classification model needs to surface good cell candidates
        for refinement.

    Args:
        cell_labels (np.ndarray): ground truth labels. Shape (N).
        topk_preds (np.ndarray): top-5 predictions. Shape (N, 5).

    Returns:
        float: top-5 geocell accuracy
    """
    num_correct = 0
    for label, top5 in zip(cell_labels, topk_preds):
        if label in top5:
            num_correct += 1

    return num_correct / len(cell_labels)

def compute_geoguessr_metrics(results: np.ndarray) -> Dict:
    """Computes metrics useful for evaluating the geoguessr

    Args:
        results (np.ndarray): output of Huggingface trainer.
        yfcc (bool, optional): whether yfcc input data was used.

    Returns:
        Dict: evaluation metrics
    """
    try:
        predictions, labels = results.predictions, results.label_ids
    except AttributeError:
        predictions, cell_preds, preds_mt, preds_climate, preds_month, top5_geocells, \
        labels, cell_labels, labels_mt, labels_climate, labels_month = results

    if cell_labels.ndim == 1:
        num_cells = 2076 #cell_preds.shape[-1]
        one_hot_matrix = np.zeros((cell_labels.size, num_cells))
        one_hot_matrix[np.arange(cell_labels.size), cell_labels] = 1
        cell_labels = one_hot_matrix

    cell_labels = np.argmax(cell_labels, axis=-1)
    distances = haversine_np(predictions, labels)
    yfcc = (labels_month is None)

    eval_dict = {
        'Mean_km_error': np.mean(distances),
        'Median_km_error': np.median(distances),
        'Under_1_km': percentage_within_radius(distances, 1),
        'Under_5_km': percentage_within_radius(distances, 5),
        'Under_10_km': percentage_within_radius(distances, 10),
        'Under_25_km': percentage_within_radius(distances, 25),
        'Under_50_km': percentage_within_radius(distances, 50),
        'Under_100_km': percentage_within_radius(distances, 100),
        'Under_200_km': percentage_within_radius(distances, 200),
        'Under_750_km': percentage_within_radius(distances, 750),
        'Under_1000_km': percentage_within_radius(distances, 1000),
        'Under_2500_km': percentage_within_radius(distances, 2500),
        'Country_accuracy': country_accuracy(predictions, labels),
        'Geoguessr_score': geoguessr_score(distances),
        'Geocell_accuracy': accuracy_score(cell_labels, cell_preds),
        'Geocell_top5_accuracy': topk_geocell_accuracy(cell_labels, top5_geocells)
    }

    if labels_mt is not None:
        preds_mt = recover_regression_values(preds_mt, yfcc)
        labels_mt = recover_regression_values(labels_mt)
        preds_climate = np.argmax(preds_climate, axis=-1)

        eval_dict['Mean_elevation_error'] = mae(labels_mt[:, 0], preds_mt[:, 0])
        eval_dict['Mean_population_error'] = mae(labels_mt[:, 1], preds_mt[:, 1])
        eval_dict['Mean_temperature_error'] = mae(labels_mt[:, 2], preds_mt[:, 2])
        eval_dict['Mean_temp_diff_error'] = mae(labels_mt[:, 3], preds_mt[:, 3])
        eval_dict['Mean_precipitation_error'] = mae(labels_mt[:, 4], preds_mt[:, 4])
        eval_dict['Mean_prec_diff_error'] = mae(labels_mt[:, 5], preds_mt[:, 5])
        eval_dict['Climate_accuracy'] = accuracy_score(labels_climate, preds_climate)

        if not yfcc:
            preds_month = np.argmax(preds_month, axis=-1)
            eval_dict['Month_accuracy'] = accuracy_score(labels_month, preds_month)

    print(eval_dict)
    return eval_dict