import concurrent.futures
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from typing import Iterable, Any, List
from shapely.affinity import scale
from cell import Cell

CRS = 'EPSG:4326'
GEOCELL_COLUMNS = ['name', 'admin_1', 'country', 'size', 'num_polygons', 'geometry']
OPTICS_PARAMS_GEOGUESSR = [(8, 0.05), (10, 0.025), (15, 0.015)]
OPTICS_PARAMS_YFCC = [(300, 0.05), (400, 0.005), (1000, 0.0001)]

class CellCollection(set):
    def __init__(self, cells: Iterable[Cell]):
        """A collection of geocells and a wrapper around a set of geocells.

        Args:
            cells (Iterable[Cell]): cells to create the CellCollection from.

        Raises:
            TypeError: Thrown if any supplied element is not of type Cell.
        """
        cs = [x for x in cells if not x.empty]
        super(CellCollection, self).__init__(set(cs))
        
        # Ensure all elements are instances of Cell
        for cell in cells:
            if not isinstance(cell, Cell):
                raise TypeError("All elements must be instances of Cell.")

    @property
    def countries(self):
        return sorted(list(set([x.country for x in self])))

    def clean(self) -> Any:
        """Removes empty cells.

        Returns:
            Any: Cleaned CellCollection
        """
        return CellCollection([x for x in self if x.empty == False])

    def find(self, cell_id: str) -> Cell:
        """Finds the geocells with the given id in a collection.

        Args:
            cell_id (str): cell id to search for.

        Returns:
            Cell: geocell
        """
        if type(cell_id) == int:
            cell_id = str(cell_id)

        for cell in self:
            if cell_id == cell.cell_id:
                return cell

        raise KeyError(f'Cell {cell_id} is not in collection.')

    def copy(self) -> Any:
        """Creates copy of cells.

        Args:
            cell_list (List): list of geocells

        Returns:
            List: copy of list of geocells
        """
        return CellCollection([Cell(x.cell_id, x.admin_1, x.country, x.points, x.polygons) \
                               for x in self])

    def to_pandas(self, country: str=None) -> gpd.GeoDataFrame:
        """Converts a list of cells to a geopandas DataFrame.

        Args:
            country (str, optional): Country to filter by. Defaults to None.

        Returns:
            gpd.GeoDataFrame: geopandas DataFrame.
        """
        if country is not None:
            cells = [x for x in self if x.country == country and x.empty == False]
        else:
            cells = [x for x in self if x.empty == False]

        df = pd.DataFrame(data=[x.tolist() for x in cells], columns=GEOCELL_COLUMNS)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=CRS)
        return df

    def save(self, output_file: str):
        """Saves the cell collection to file.

        Args:
            output_file (str): Output filename.
        """
        np.save(output_file, list(self))

    def overwrite(self, file: str):
        """Overwrites current CellCollection with file.

        Args:
            file (str): Filename to load.
        """
        collection = np.load(file, allow_pickle=True)
        cs = [x for x in collection if not x.empty]
        super(CellCollection, self).__init__(set(cs))
        print('Overwrote with contents of:', file)

    @classmethod
    def load(cls, file: str):
        """Load a CellCollection from file.

        Args:
            file (str): Filename to load.
        """
        return cls(set(np.load(file, allow_pickle=True)))

    def balance(self, min_cell_size: int, max_cell_size: int):
        """Balances all contained cells such that most cells are not smaller than min_cell_size.

        Args:
            min_cell_size: (int): Minimum cell size.
            max_cell_size (int): Minimum cell size.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._balance_country, country, min_cell_size) for country in self.countries[::-1]]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Fusing cells within countries', unit='country'):
                pass
            
        self._split_geocells(min_cell_size, max_cell_size)

    def _balance_country(self, country: str, min_cell_size: int):
        """Helper function to parallelize cell fusing across countries.

        Args:
            country (str): Country ID
            min_cell_size (int): Minimum cell size.
        """
        cells = CellCollection([x for x in self if x.country == country and x.empty == False])
        cells.__fuse(min_cell_size=min_cell_size)

    def _split_geocells(self, min_cell_size: int, max_cell_size: int):
        """Split large geocells into cells smaller or equal to max_cell_size.

        Args:
            min_cell_size: (int): Minimum cell size.
            max_cell_size (int): Maximum cell size.
        """
        for args in OPTICS_PARAMS_YFCC:
            print('||| NEW OPTICS PARAMS ||| ', args)
            new_cells = []

            large_cells = [x for x in self if x.size > max_cell_size]
            round = 1
            while len(large_cells) > 0:

                # Progress bar
                desc = f'Round {round}: splitting large cells'
                pbar = tqdm(total=len(large_cells), desc=desc, dynamic_ncols=True, unit='cell')

                # Parallalize the splitting of cells across cores
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(cell._split_cell, self, args, min_cell_size, max_cell_size)for cell in large_cells]
                    for future in concurrent.futures.as_completed(futures):
                        nc = future.result()
                        new_cells.extend(nc)
                        pbar.update(1)

                    concurrent.futures.wait(futures)

                # Update variables
                large_cells = new_cells
                new_cells = []
                round += 1

                # Close the progress bar
                pbar.close()

            self.save('data/yfcc_geocell_collection_new.npy')

    def __fuse(self, min_cell_size: int):
        """Fuses all contained cells.

        Args:
            min_cell_size (int): Minimum cell size.
        """
        exclude_list = CellCollection({})
        while True:
            consider_cells = self - exclude_list
            cell_df = consider_cells.to_pandas()

            cell_df['scaled'] = cell_df['geometry'].apply(lambda x: scale(x, xfact=1.01, yfact=1.01))
            cell_df = cell_df.set_geometry('scaled')
            df_small = cell_df.loc[cell_df['size'] < min_cell_size].copy()
            if len(df_small.index) == 0:
                break
                
            # Sample one cell
            center = df_small.sample(random_state=330).iloc[0]        
            df_slice = df_small[df_small['name'] != center['name']].reset_index(drop=True)
            
            # Find sorrounding cells - prioritize small adjacent cells in same ADMIN 1 area
            look_df = df_slice[df_slice['admin_1'] == center['admin_1']].reset_index(drop=True)
            indices = look_df.sindex.query(center.scaled, predicate='intersects')
            found_indices = look_df.iloc[indices]['name'].values
            
            # Find sorrounding cells - prioritize big adjacent cells in same ADMIN 1 area
            if len(found_indices) == 0:
                look_df = cell_df[(cell_df['admin_1'] == center['admin_1']) & (cell_df['name'] != center['name'])].reset_index(drop=True)
                indices = look_df.sindex.query(center.scaled, predicate='intersects')
                found_indices = look_df.iloc[indices]['name'].values
                
            # Find sorrounding cells - prioritize small adjacent cells in other ADMIN areas
            if len(found_indices) == 0:
                indices = df_slice.sindex.query(center.scaled, predicate='intersects')
                found_indices = df_slice.iloc[indices]['name'].values
                    
            # Find sorrounding cells - prioritize big adjacent cells in other ADMIN areas
            if len(found_indices) == 0:
                indices = cell_df[(cell_df['name'] != center['name'])].sindex.query(center.scaled, predicate='intersects')
                found_indices = cell_df[(cell_df['name'] != center['name'])].iloc[indices]['name'].values
                        
            # Try again but enlarge 2 times as much
            if len(found_indices) == 0:
                new_shape = scale(center.scaled, xfact=2, yfact=2)
                indices = cell_df[(cell_df['name'] != center['name'])].sindex.query(new_shape, predicate='intersects')
                found_indices = cell_df[(cell_df['name'] != center['name'])].iloc[indices]['name'].values
                        
            if len(found_indices) == 0:
                exclude_list.add(self.find(center['name']))
                continue
            
            sorrounds = cell_df[(cell_df['name'].isin(found_indices)) & (cell_df['name'] != center['name'])]
            sorrounds = sorrounds.sort_values(by='size', ascending=False)
            
            # Get cells
            c_cell = self.find(center['name'])
            s_cell = self.find(sorrounds.iloc[0]['name'])
            
            # Merge
            c_cell.combine([s_cell])
    
    def __sub__(self, other):
        return CellCollection(super().__sub__(other))
    
    def __add__(self, other):
        return CellCollection(self.union(other))