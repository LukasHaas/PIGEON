import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # The script is being run directly
    # Get the project directory (parent of the script directory)
    project_dir = os.path.dirname(script_dir)
else:
    # The script is being imported
    # In this case, we'll assume the current directory is the project directory
    project_dir = os.getcwd()

# Add the project directory to sys.path
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Imports
import srtm
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import georasters as gr
from pandarallel import pandarallel
from concurrent.futures import ThreadPoolExecutor
from georasters.georasters import RasterGeoError
from latlon_utils import get_climate
from tqdm import tqdm
from srtm.utils import FileHandler
from typing import List
from config import GADM_PATH, GHSL_PATH, WORLDCLIM_SAVE_PATH, \
                   SRTM_SAVE_PATH, KOPPEN_GEIGER_PATH

# Pandas parallel
tqdm.pandas()
pandarallel.initialize(progress_bar=False)

# Constants
CRS = 'EPSG:4326'
MOLLWEIDE = 'ESRI:54009'
COUNTRY_LAYER = 'ADM_0'
AREA_LAYER = 'ADM_1'
CLIMATE_VARS = ['tavg', 'prec']
CLIMATE_DICT = { # Köppen-Geiger Climate Zones
    1:  ('Af',   'Tropical, rainforest', 'a tropical rainforest climate'),     
    2:  ('Am',   'Tropical, monsoon', 'a tropical monsoon climate'),                 
    3:  ('Aw',   'Tropical, savannah', 'a tropical savanna climate'),                
    4:  ('BWh',  'Arid, desert, hot', 'an arid, hot desert climate'),       
    5:  ('BWk',  'Arid, desert, cold', 'an arid, cold desert climate'),                  
    6:  ('BSh',  'Arid, steppe, hot', 'a hot, semi-arid climate'), 
    7:  ('BSk',  'Arid, steppe, cold', 'a cold, semi-arid climate'),                
    8:  ('Csa',  'Temperate, dry summer, hot summer', 'a Mediterranean climate with a hot summer'), 
    9:  ('Csb',  'Temperate, dry summer, warm summer', 'a Mediterranean climate with a warm summer'), 
    10: ('Csc',  'Temperate, dry summer, cold summer', 'a Mediterranean climate with a cold summer'),  
    11: ('Cwa',  'Temperate, dry winter, hot summer', 'a humid subtropical monsoon climate'), 
    12: ('Cwb',  'Temperate, dry winter, warm summer', 'a temperate oceanic monsoon climate'), 
    13: ('Cwc',  'Temperate, dry winter, cold summer', 'a subpolar oceanic monsoon climate'), 
    14: ('Cfa',  'Temperate, no dry season, hot summer', 'a humid subtropical climate'), 
    15: ('Cfb',  'Temperate, no dry season, warm summer', 'a temperate oceanic climate'), 
    16: ('Cfc',  'Temperate, no dry season, cold summer', 'a subpolar oceanic climate'), 
    17: ('Dsa',  'Cold, dry summer, hot summer', 'a Mediterranean humid continental climate with a hot summer'), 
    18: ('Dsb',  'Cold, dry summer, warm summer', 'a Mediterranean humid continental climate with a warm summer'),   
    19: ('Dsc',  'Cold, dry summer, cold summer', 'a Mediterranean subarctic climate with a cold summer'), 
    20: ('Dsd',  'Cold, dry summer, very cold winter', 'a Mediterranean humid continental climate with a warm summer'),   
    21: ('Dwa',  'Cold, dry winter, hot summer', 'a humid continental monsoon climate with a hot summer'), 
    22: ('Dwb',  'Cold, dry winter, warm summer', 'a humid continental monsoon climate with a warm summer'),  
    23: ('Dwc',  'Cold, dry winter, cold summer', 'a subarctic monsoon climate'), 
    24: ('Dwd',  'Cold, dry winter, very cold winter', 'an extremely cold subarctic monsoon climate'),   
    25: ('Dfa',  'Cold, no dry season, hot summer', 'a humid continental climate with a hot summer'), 
    26: ('Dfb',  'Cold, no dry season, warm summer', 'a humid continental climate with a warm summer'), 
    27: ('Dfc',  'Cold, no dry season, cold summer', 'a subarctic climate'), 
    28: ('Dfd',  'Cold, no dry season, very cold winter', 'an extremely cold subarctic climate'), 
    29: ('ET',   'Polar, tundra', 'a polar tundra climate'), 
    30: ('EF',   'Polar, frost', 'a polar ice cap climate')
}

# Init
logger = logging.getLogger('run')

class GeoAugmentor:
    def __init__(self, output_file: str, path_prefix: str='') -> None:
        self.country_df = None
        self.area_df = None
        self.elevation_data = None
        self.population_file = None
        self.climate_zone_file = None

        self.output_file = output_file
        self.path_prefix = path_prefix

        # Set temporary file storage location
        os.environ['LATLONDATA'] =  path_prefix + WORLDCLIM_SAVE_PATH

    def augment_country(self, data: pd.DataFrame) -> pd.DataFrame:
        """Augment dataset with GADM country data

        Args:
            data (pd.DataFrame): dataset to augment

        Returns:
            pd.DataFrame: augmented dataset
        """
        logger.warning('Augmenting dataset with GADM country names.')
        if self.country_df is None:
            self.country_df = gpd.read_file(f'{self.path_prefix}{GADM_PATH}', layer=COUNTRY_LAYER)

        # Compute country
        geo_data = gpd.GeoDataFrame(data, crs=CRS, geometry=gpd.points_from_xy(data.lng, data.lat))
        for _, row in tqdm(self.country_df.iterrows()):
            name = row['COUNTRY']
            indices = geo_data.sindex.query(row.geometry, predicate='covers')
            geo_data.loc[indices, 'country_name'] = name

        # Compute missing countries as nearest neighbors
        missing = geo_data[geo_data['country_name'].isnull()]
        not_missing = geo_data[geo_data['country_name'].isnull() == False]
        nearest = not_missing.sindex.nearest(missing.geometry, return_all=False)
        not_missing = geo_data[geo_data['country_name'].isnull() == False].copy().reset_index(drop=True)
        countries = not_missing.iloc[nearest[1, :]]['country_name'].values
        geo_data.loc[geo_data['country_name'].isnull(), 'country_name'] = countries

        return geo_data

    @property
    def all_countries(self) -> List:
        """Returns a list of all possible countries.

        Returns:
            List: all countries.
        """
        if self.country_df is None:
            self.country_df = gpd.read_file(f'{self.path_prefix}{GADM_PATH}', layer=COUNTRY_LAYER)
        
        return self.country_df.sort_values(by='COUNTRY')['COUNTRY'].unique().tolist()

    def augment_geo_area(self, data: pd.DataFrame) -> pd.DataFrame:
        """Augment dataset with GADM Level 1 area name 

        Args:
            data (pd.DataFrame): dataset to augment

        Returns:
            pd.DataFrame: augmented dataset
        """
        logger.warning('Augmenting dataset with GADM Level 1 area names.')
        if self.area_df is None:
            self.area_df = gpd.read_file(f'{self.path_prefix}{GADM_PATH}', layer=AREA_LAYER)

        geo_data = gpd.GeoDataFrame(data, crs=CRS, geometry=gpd.points_from_xy(data.lng, data.lat))

        # Compute geo area
        geo_data = gpd.GeoDataFrame(data, crs=CRS, geometry=gpd.points_from_xy(data.lng, data.lat))
        for _, row in tqdm(self.area_df.iterrows()):
            name = row['NAME_1']
            indices = geo_data.sindex.query(row.geometry, predicate='covers')
            geo_data.loc[indices, 'geo_area'] = name

        # Compute missing countries as nearest neighbors
        missing = geo_data[geo_data['geo_area'].isnull()]
        not_missing = geo_data[geo_data['geo_area'].isnull() == False]
        nearest = not_missing.sindex.nearest(missing.geometry, return_all=False)
        not_missing = geo_data[geo_data['geo_area'].isnull() == False].copy().reset_index(drop=True)
        countries = not_missing.iloc[nearest[1, :]]['geo_area'].values
        geo_data.loc[geo_data['geo_area'].isnull(), 'geo_area'] = countries
        
        return geo_data

    def all_geo_areas(self, country: str) -> List:
        """Generates a list of all geo areas in the specified country.

        Args:
            country (str): country to retrieve geoareas for.

        Returns:
            List: all geo areas in country.
        """
        if self.area_df is None:
            self.area_df = gpd.read_file(f'{self.path_prefix}{GADM_PATH}', layer=AREA_LAYER)

        df_slice = self.area_df[self.area_df['COUNTRY'] == country]
        if len(df_slice) == 0:
            return []

        return self.df_slice.sort_values(by='NAME_1')['NAME_1'].values.tolist()

    def augment_climate(self, data: pd.DataFrame, batch_size: int=8096) -> pd.DataFrame:
        """Augment dataset with WorldClim climate data

        Args:
            data (pd.DataFrame): dataset to augment
            batch_size (int, optional): how many coordinates to process at a time.
                Defaults to 8192.

        Returns:
            pd.DataFrame: augmented dataset
        """
        logger.warning('Augmenting dataset with WorldClim climate data.')
        
        # Get climate data
        coords = data[['lat', 'lng']].values.T
        temp_avg, temp_diff, prec_avg, prec_diff = [], [], [], []
        for i in tqdm(range(0, len(data.index), batch_size)):
            batch_coords = coords[:, i:i+batch_size]

            try:
                climate_data = get_climate(*batch_coords, variables=CLIMATE_VARS, load_data=True)
            except IndexError as e:
                print(batch_coords)
                raise IndexError(e)

            # Perform climate computations
            temp_avg += climate_data['tavg']['ann'].values.tolist()
            temp_diff += (climate_data['tavg'].max(axis=1).values - climate_data['tavg'].min(axis=1).values).tolist()
            prec_avg += climate_data['prec']['ann'].values.tolist()
            prec_diff += (climate_data['prec'].max(axis=1).values - climate_data['prec'].min(axis=1).values).tolist()

            # Save the arrays to a file in each iteration
            np.save(f'data/tmp/temp_avg.npy', temp_avg)
            np.save(f'data/tmp/temp_diff.npy', temp_diff)
            np.save(f'data/tmp/prec_avg.npy', prec_avg)
            np.save(f'data/tmp/prec_diff.npy', prec_diff)
        
        # Add to dataframe
        data['temp_avg'] = temp_avg
        data['temp_diff'] = temp_diff
        data['prec_avg'] = prec_avg
        data['prec_diff'] = prec_diff
        
        return data

    def __augment_elevation_point(self, row: pd.Series) -> int:
        return self.elevation_data.get_elevation(row.lat, row.lng)

    def augment_elevation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Augment dataset with SRTM elevation data

        Args:
            data (pd.DataFrame): dataset to augment

        Returns:
            pd.DataFrame: augmented dataset
        """
        logger.warning('Augmenting dataset with SRTM elevation data.')
        if self.elevation_data is None:
            handler = FileHandler(local_cache_dir=f'{self.path_prefix}{SRTM_SAVE_PATH}')
            self.elevation_data = srtm.get_data(srtm1=False, file_handler=handler)

        data['elevation'] = data[['lat', 'lng']].progress_apply(self.__augment_elevation_point, axis=1)
        return data

    def __augment_population_point(self, point: pd.Series) -> float:
        try:
            return self.population_file.map_pixel(point.geometry.x, point.geometry.y)
        except RasterGeoError:
            return 0

    def augment_population(self, data: pd.DataFrame) -> pd.DataFrame:
        """Augment dataset with GHSL population density data

        Args:
            data (pd.DataFrame): dataset to augment

        Returns:
            pd.DataFrame: augmented dataset
        """
        logger.warning('Augmenting dataset with GHSL population density data.')
        if self.population_file is None:
            self.population_file = gr.from_file(f'{self.path_prefix}{GHSL_PATH}')

        geo_data = gpd.GeoDataFrame(data, crs=CRS, geometry=gpd.points_from_xy(data.lng, data.lat))
        geo_data = geo_data.to_crs(MOLLWEIDE)
        data['population'] = geo_data.progress_apply(self.__augment_population_point, axis=1)
        return data

    def __augment_climate_zone_point(self, row: pd.Series) -> float:
        zone_id = self.climate_zone_file.map_pixel(row.lng, row.lat)
        if zone_id == 0:
            return np.nan
            
        return zone_id - 1

    def augment_climate_zone(self, data: pd.DataFrame) -> pd.DataFrame:
        """Augment dataset with Koppen-Geiger climate zones

        Args:
            data (pd.DataFrame): dataset to augment

        Returns:
            pd.DataFrame: augmented dataset
        """
        logger.warning('Augmenting dataset with Koppen-Geiger climate zones.')
        if self.climate_zone_file is None:
            self.climate_zone_file = gr.from_file(f'{self.path_prefix}{KOPPEN_GEIGER_PATH}')

        data['climate_zone'] = data.progress_apply(self.__augment_climate_zone_point, axis=1)
        return data

    def augment(self, **kwargs) -> pd.DataFrame:
        self.__call__(**kwargs)

    def __call__(self, dataset: pd.DataFrame, country: bool=False, geo_area: bool=True, climate: bool=True,
                 elevation: bool=True, density: bool=True, climate_zone: bool=True) -> pd.DataFrame:
        """Augment the dataset

        Args:
            dataset (pd.DataFrame): dataset to augment
            country (bool, optional): whether GADM country names should be added.
                Defaults to False.
            geo_area (bool, optional): whether GADM Level 1 area name should be added.
                Defaults to True.
            climate (bool, optional): whether climate data should be added. Defaults to True.
            elevation (bool, optional): whether elevation data should be added. Defaults to True.
            density (bool, optional): whether population density data should be added.
                Defaults to True.
            climate_zone (bool, optional): whether Koppen-Geiger climate zone data should be added.
                Defaults to True.

        Returns:
            pd.DataFrame: augmented dataset
        """
        if country:
            dataset = self.augment_country(dataset)
            dataset.to_csv(self.output_file, index=False)

        if geo_area:
            dataset = self.augment_geo_area(dataset)
            dataset.to_csv(self.output_file, index=False)

        if climate:
            dataset = self.augment_climate(dataset)
            dataset.to_csv(self.output_file, index=False)

        if climate_zone:
            dataset = self.augment_climate_zone(dataset)
            dataset.to_csv(self.output_file, index=False)

        if elevation:
            dataset = self.augment_elevation(dataset)
            dataset.to_csv(self.output_file, index=False)

        if density:
            dataset = self.augment_population(dataset)
            dataset.to_csv(self.output_file, index=False)

        return dataset

if __name__ == '__main__':
    df = pd.read_csv('data/data_landmarks.csv')
    augmentor = GeoAugmentor('data/data_landmarks_aug.csv')
    _ = augmentor(df, country=True, geo_area=True, climate=True, elevation=True, density=True, climate_zone=True)