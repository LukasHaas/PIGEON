import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # The script is being run directly
    # Get the project directory (parent of the script directory)
    project_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(project_dir)
else:
    # The script is being imported
    # In this case, we'll assume the current directory is the project directory
    project_dir = os.getcwd()

# Add the project directory to sys.path
if project_dir not in sys.path:
    sys.path.append(project_dir)

# FILE STARTS HERE

import numpy as np
import pandas as pd
import geopandas as gpd
from pandarallel import pandarallel
from typing import Tuple, List
from tqdm import tqdm
from sklearn.cluster import OPTICS
from datasets import DatasetDict
from preprocessing import haversine_matrix_np
from config import *

# DEFAULT_CLUSTER_ARGS = (3, 0.15) # PIGEON
DEFAULT_CLUSTER_ARGS = (100, 0.1)

class ProtoDataset:
    def __init__(self, df: pd.DataFrame, embedding_path: str, 
                 output_file: str, cluster_args: Tuple[float, float]=DEFAULT_CLUSTER_ARGS,
                 sample: int=None):
        """Initializes a Prototype Dataset for in-cell refinement.

        Args:
            df (pd.DataFrame): Dataframe to create prototypes from.
            embedding_path (str): Path to Huggingface dataset containing embeddings.
            output_file (str): Where to save the prototypes.
            cluster_args (Tuple[float, float], optional): Clustering arguments for prototype
                creation. Defaults to DEFAULT_CLUSTER_ARGS.
            sample (int, optional): If to sample from the dataframe. Defaults to None.
        """
        super().__init__()

        self.df = df[df['selection'] == 'train'].copy().reset_index(drop=True)
        self.embeddings = DatasetDict.load_from_disk(embedding_path)
        self.embeddings = self.embeddings['train']
        self.output = output_file
        self.cluster_args = cluster_args
        self._sample = sample
        self.clusterer = OPTICS(min_samples=self.cluster_args[0],
                                xi=self.cluster_args[1],
                                metric='precomputed')

        # Add geocell index if not already applied
        if 'geocell_idx' not in self.df.columns:
            self.df['labels_clf'] = np.nan
            train_cell_idx = self.embeddings['labels_clf'].numpy()
            self.df.loc[self.df['selection'] == 'train', 'geocell_idx'] = train_cell_idx

    def generate(self):
        """Generates prototypes for a given dataset and saves it to file.
        """
        # Sample if necessary
        if self._sample:
            self.df = self.df.sample(self._sample).copy()

        # Compute clusters
        pandarallel.initialize(nb_workers=64, progress_bar=True)
        self._clusters = self.df.groupby('geocell_idx').parallel_apply(self._compute_clusters)
        np.save('tmp/clusters_100.npy', self._clusters)
        # self._clusters = np.load('tmp/clusters.npy', allow_pickle=True)

        # Select clusters
        tqdm.pandas()
        self.df['cluster'] = self.df.progress_apply(self.__get_cluster_id, axis=1)

        # Compute prototypes from clusters
        centroids = self.df.groupby(['geocell_idx', 'cluster']).agg(lng=('lng', 'mean'),
                                                                    lat=('lat', 'mean'),
                                                                    count=('lng', len),
                                                                    indices=('lng', self.__get_indices))
        centroids = centroids.reset_index(drop=False)
        centroids = centroids.loc[centroids['cluster'] != -1].copy()
        
        # Save to disk
        centroids.to_csv(self.output, index=False)
    
    def _proto_embedding(self, series: pd.Series) -> List[float]:
        """Computes the prototype for a single cluster.

        Args:
            series (pd.Series): Series containing cluster indices.

        Returns:
            List[float]: Embedding as a list.
        """
        indices = series.index.values
        proto_emb = self.embeddings['embedding'][indices].mean(dim=0)
        return proto_emb.numpy().tolist()

    def __get_indices(self, series: pd.Series) -> List[int]:
        """Helper function to retrieve indices of a series.

        Args:
            series (pd.Series): Series.

        Returns:
            List[int]: List of indices.
        """
        return series.index.values.tolist()
        
    def _compute_distances(self, df: pd.DataFrame) -> np.ndarray:
        """Computes the haversine distances in a dataframe.

        Args:
            df (pd.DataFrame): Dataframe.

        Returns:
            np.ndarray: Distances.
        """
        points = df[['lng', 'lat']].values
        distances = haversine_matrix_np(points, points.T)
        distances = np.where(distances == 0, 1e-5, distances)
        return distances

    def _compute_clusters(self, df: pd.DataFrame) -> np.ndarray:
        """Computes clusters for a given dataframe.

        Args:
            df (pd.DataFrame): Dataframe.

        Returns:
            np.ndarray: Array of cluster indices.
        """
        if len(df.index) < self.cluster_args[0]:
            return [0] * len(df.index)
        
        distances = self._compute_distances(df)
        clusters = self.clusterer.fit_predict(distances)
        return clusters

    def __find_index_in_geocell_slice(self, row: pd.Series) -> int:
        """Retrieves the index of the given row in the Datframe.

        Args:
            row (pd.Series): Row to retrieve.

        Returns:
            int: Index of row.
        """
        slice_df = self.df[self.df['geocell_idx'] == row.geocell_idx]
        index = slice_df.index.get_loc(row.name)
        return index

    def __get_cluster_id(self, row: pd.Series) -> int:
        """Retrieves cluster corresponding to given row.

        Args:
            row (pd.Series): Row to retrieve cluster for.

        Returns:
            int: Cluster.
        """
        index = self.__find_index_in_geocell_slice(row)
        return self._clusters[row.geocell_idx][index]

if __name__ == '__main__':
    data_df = pd.read_csv(METADATA_PATH_YFCC)
    data_df = gpd.GeoDataFrame(data_df, geometry=gpd.points_from_xy(data_df.lng, data_df.lat), crs='EPSG:4326')
    dataset = ProtoDataset(data_df, DATASET_PATH_YFCC, 'data_prototypes_YFCC_100.csv')
    dataset.generate()