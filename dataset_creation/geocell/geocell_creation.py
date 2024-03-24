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
import warnings
import functools
import numpy as np
import pandas as pd
import geopandas as gpd

from typing import Tuple, List
from tqdm import tqdm

from cell import Cell
from cell_collection import CellCollection
from config import COUNTRY_PATH, ADMIN_1_PATH, ADMIN_2_PATH, MIN_CELL_SIZE, MAX_CELL_SIZE

# Constants
CRS = 'EPSG:4326'
NEEDED_COLS = ['id', 'lat', 'lng', 'selection', 'country_name']
LEVEL_NAMES = ['country_id', 'admin_1_id', 'admin_2_id']

# Special Country Lists
COPY_COUNTRY_ALLOW = ['Faroe Islands', 'Isle of Man', 'United States Minor Outlying Isl', 'Curaçao', 'Jersey']
SPECIAL_COUNTRIES = ['Hong Kong', 'Christmas Island', 'Curaçao']
COPY_A1_ALLOW = ['Latvia', 'Virgin Islands, U.S.', 'Uruguay', 'Greenland']

class GeocellCreator:
    def __init__(self, df: pd.DataFrame, output_file: str) -> None:
        """Creates geocells based on a supplied dataframe.

        Args:
            df (pd.DataFrame): Pandas dataframe used during training.
            output_file (str): Where the geocells should be saved to.
        """
        assert all([x in df.columns for x in NEEDED_COLS]), f'Dataframe must contain all of the following \
            columns: {NEEDED_COLS}. Also, "selection" is used to filter out all rows with value "train".'
        
        # Save properties
        self.output = output_file
        self.cells = None

        # Load dataframe
        self.df = df[df['selection'] == 'train'].copy().reset_index()

        keep_cols = [x for x in self.df.columns if x in NEEDED_COLS or x in LEVEL_NAMES]
        self.df = self.df[keep_cols].copy()
        self.df = gpd.GeoDataFrame(self.df, geometry=gpd.points_from_xy(self.df.lng, self.df.lat), crs='EPSG:4326')

    def generate(self, min_cell_size: int=MIN_CELL_SIZE, max_cell_size: int=MAX_CELL_SIZE):
        """Generate geocells.

        Args:
            min_cell_size (int, optional): Minimum number of training examples per geocell.
                Defaults to MIN_CELL_SIZE.
            max_cell_size (int, optional): Maximum number of training examples per geocell.
                Defaults to MAX_CELL_SIZE.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.cells = self._initialize_cells(min_cell_size)
            self.cells.balance(min_cell_size, max_cell_size)
            self.geo_cell_df = self.cells.to_pandas()
            self.geo_cell_df.to_csv(self.output, index=False)

        return self.cells

    def _initialize_cells(self, min_cell_size: int) -> CellCollection:
        """Assigns IDs to each location in the dataset based on geographic boundaries.

        Args:
            min_cell_size (int): Suggested minimum cell size.

        Returns:
            CellCollection: Initial geocells on admin 2 hierarchy level.

        Note: This helper function is created such that the boundaries, which are huge
            files, can go out of scope as quickly as possible to free up memory.
        """
        # Load boundaries and assign IDs
        cols = [str(x) for x in self.df.columns]
        if all([x in cols for x in LEVEL_NAMES]) == False or self.df[LEVEL_NAMES].isnull().sum().sum() > 0:
            boundaries = self._load_geo_boundaries()
            for name, boundary in zip(LEVEL_NAMES, boundaries):
                self.df = self._assign_boundary_ids(self.df, boundary, name)
                self.df = self._apply_nearest_match(self.df, name)

            #self.df.to_csv('data/data_yfcc_augmented_non_contaminated.csv', index=False)
        
        else:
            boundaries = self._load_geo_boundaries(most_granular=True)

        # Initialize all geocells
        initialize_cell_fnc = functools.partial(self.__initialize_cell, boundaries[2], min_cell_size)
        tqdm.pandas(desc='Initializing geocells for every admin 2 area')
        cells = self.df.groupby(LEVEL_NAMES[2]).progress_apply(initialize_cell_fnc)
        cells = [item for sublist in cells for item in sublist]

        # Add unassigned areas to cells
        self._assign_unassinged_areas(cells, boundaries[2])
        return CellCollection(cells)
    
    def __initialize_cell(self, admin_2_boundary: gpd.GeoDataFrame,
                          min_cell_size: int, df: gpd.GeoDataFrame) -> Cell:
        """Initializes a geocell based on an admin 2 boundary level.

        Args:
            admin_2_boundary (gpd.GeoDataFrame): file containing admin 2 polygons.
            min_cell_size (int): suggested minimum cell size.
            df (gpd.GeoDataFrame): Dataframe containing all coordinates of a given
                admin 2 level.

        Returns:
            Cell: Geocell.
        """
        # Get metadata
        name = df.iloc[0][LEVEL_NAMES[2]]
        admin_1 = df.iloc[0][LEVEL_NAMES[1]]
        country = df.iloc[0][LEVEL_NAMES[0]]

        # Get shapes
        polygon_ids = np.array([int(x) for x in df[LEVEL_NAMES[2]].unique()])
        points = df['geometry'].values.tolist()
        polygons = admin_2_boundary.iloc[polygon_ids].geometry.unique().tolist()

        return [Cell(name, admin_1, country, points, polygons)]

    def _load_geo_boundaries(self, most_granular: bool=False) -> Tuple[gpd.GeoDataFrame]:
        """Loads the geographic boundaries for countries and other admin levels.

        Args:
            most_granular (bool, optional): only loads most granular area.
                Defaults to False.

        Returns:
            Tuple[gpd.GeoDataFrame]: countries, admin 1, and admin 2 level
                geographic boundaries.
        """
        print('Loading geographic boundaries ...')

        # Load smaller administrative areas
        admin_2 = gpd.read_file(ADMIN_2_PATH)
        admin_2 = admin_2.set_crs(crs=CRS)
        admin_2['geometry'] = admin_2['geometry'].apply(lambda x: x.buffer(0))
        print(' ... loaded admin 2 boundaries.')

        if most_granular:
            return None, None, admin_2

         # Load Geo areas
        admin_1 = gpd.read_file(ADMIN_1_PATH)
        admin_1 = admin_1.set_crs(crs=CRS)
        admin_1['geometry'] = admin_1['geometry'].apply(lambda x: x.buffer(0))
        print(' ... loaded admin 1 boundaries.')

        # Load countries
        countries = gpd.read_file(COUNTRY_PATH)
        countries = countries.set_crs(crs=CRS)
        countries['geometry'] = countries['geometry'].apply(lambda x: x.buffer(0))
        print(' ... loaded countries.')

        return countries, admin_1, admin_2

    def _assign_boundary_ids(self, df: gpd.GeoDataFrame, ref_df: gpd.GeoDataFrame,
                                  col: str) -> gpd.GeoDataFrame:
        """Assigns geographic IDs to a dataframe given a boundary reference dataframe.

        Args:
            df (gpd.GeoDataFrame): Dataframe to assign IDs to.
            ref_df (gpd.GeoDataFrame): Reference dataframe containing boundary polygons.
            col (str): Name of the new ID column.

        Returns:
            gpd.GeoDataFrame: df augmented with boundary ID data.
        """
        found_points = df.sindex.query_bulk(ref_df.geometry, predicate='covers')
        for i in range(len(ref_df.index)):
            mask = (found_points[0] == i)
            indices = found_points[1][mask].tolist()
            if len(indices) == 0:
                continue
                
            df.loc[indices, col] = str(i)
            
        return df

    def _assign_unassinged_areas(self, cells: List[Cell], admin_2: gpd.GeoDataFrame):
        """Adds unassigned admin 2 areas to the existing cells.

        Args:
            cells (List[Cell]): Existing geocells.
            admin_2 (gpd.GeoDataFrame): Admin 2 boundary GeoDataFrame.
        """
        # Determined assigned and unassigned polygons
        cell_map = {int(cell.cell_id): cell for cell in cells}
        cell_idx = list(cell_map.keys())

        assigned = admin_2.loc[admin_2.index.isin(cell_idx)].copy().reset_index()
        assigned['centroid'] = [row.geometry.centroid for _, row in assigned.iterrows()]
        assigned = gpd.GeoDataFrame(assigned, geometry='centroid')

        unassigned = admin_2.loc[admin_2.index.isin(cell_idx) == False].reset_index(drop=True)
        unassigned['centroid'] = [row.geometry.centroid for _, row in unassigned.iterrows()]
        unassigned = gpd.GeoDataFrame(unassigned, geometry='centroid')

        # Find assignments
        closest_match = assigned.sindex.nearest(unassigned.centroid)[1]
        assignments = assigned.iloc[closest_match]['index'].values

        # Add polygons to closest cells
        for i, row in unassigned.iterrows():
            closest_cell = assignments[i]
            cell_map[closest_cell].add_polygons([row['geometry']])

    def _apply_nearest_match(self, df: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
        """Fill NaN values in a column based on the closest geographic match.

        Args:
            df (gpd.GeoDataFrame): Dataframe to fill NaN values in.
            col (str): Column to substitute NaNs in.

        Returns:
            gpd.GeoDataFrame: Dataframe with all NaNs replaced.
        """
        missing = df[df[col].isnull()].copy().reset_index(drop=True)
        not_missing = df[df[col].isnull() == False].copy().reset_index(drop=True)
        nearest = not_missing.sindex.nearest(missing.geometry, return_all=False)
        values = not_missing.iloc[nearest[1, :]][col].values
        df.loc[df[col].isnull(), col] = values
        return df


if __name__ == '__main__':
    df = pd.read_csv('data/data_yfcc_augmented_non_contaminated.csv')
    geocell_creator = GeocellCreator(df, 'data/geocells_yfcc.csv')
    geocells = geocell_creator.generate()