import numpy as np
import shapely.wkt
import shapely.ops
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.errors import TopologicalError
from typing import List, Any, Tuple
from sklearn.cluster import OPTICS
from scipy.spatial import Voronoi
from voronoi import voronoi_finite_polygons

CRS = 'EPSG:4326'
CELL_COLUMNS = ['id', 'lng', 'lat']
GEOCELL_COLUMNS = ['name', 'admin_1', 'country', 'size', 'num_polygons', 'geometry']

class Cell:
    """Abstraction of a geocell.
    """
    def __init__(self, cell_id: str, admin_1: str, country: str,
                 points: List[Point], polygons: List[Polygon]):
        """Initializes a geocell.

        Args:
            cell_id (str): name
            admin_1 (str): name of Admin 1 area
            country (str): name of country
            points (List[Point]): collection of coordinates
            polygons (List[Polygon]): collection of polygons
        """
        self.cell_id = str(cell_id)
        self.admin_1 = str(admin_1)
        self.country = str(country)
        self._points = points

        if isinstance(polygons, Polygon):
            self._polygons = [polygons]
        else:
            self._polygons = list(polygons)
        
    @property
    def size(self) -> int:
        """Returns the number of coordinates in cell.

        Returns:
            int: coordinates in cell
        """
        return len(self.points)
        
    @property
    def shape(self) -> Polygon:
        """Combines cell's collection of polygons to a geocell shape.

        Returns:
            Polygon: geocell shape
        """
        union = shapely.ops.unary_union(self.polygons)
        union = union.buffer(0)
        return union

    @property
    def points(self) -> List[Point]:
        """Safely loads points.

        Returns:
            List[Point]: all points.
        """
        try:
            p = [shapely.wkt.loads(x) for x in self._points]
        except TypeError:
            p = self._points

        return p

    @property
    def polygons(self) -> List[Polygon]:
        """Safely loads polygons.

        Returns:
            List[Polygon]: all polygons.
        """
        try:
            p = [shapely.wkt.loads(x) for x in self._polygons]
        except TypeError:
            p = self._polygons

        return p

    @property
    def multi_point(self) -> MultiPoint:
        """Generates a multi-point shape from points

        Returns:
            MultiPoint: multi-point shape
        """
        return MultiPoint(self.points)

    @property
    def coords(self) -> np.ndarray:
        """Generate coordinates from points in the cell.

        Returns:
            np.ndarray: coordinates (lng, lat)
        """
        return np.array([[x.x, x.y] for x in self.points])

    @property
    def centroid(self) -> np.ndarray:
        """Computes the centroid of the geocell.

        Returns:
            np.ndarray: coordinates of centroid (lng,lat)
        """
        # COMPUTATION BASED ON POINTS:
        return np.mean(self.coords, axis=0)

        # NEW COMPUTATION BASED ON SHAPE
        # return self.shape.centroid
    
    @property
    def empty(self) -> bool:
        """Whether the geocell is empty.

        Returns:
            bool: whether the geocell is empty.
        """
        return len(self.points) == 0

    def subtract(self, other: Any):
        """Subtracts other cell from current cell.

        Args:
            other (Any): other cell
        """
        try:
            diff_shape = self.shape.difference(other.shape)
            
        except TopologicalError as e:
            print(f'Error occurred during subtracting in cell: {self.cell_id}')
            raise TopologicalError(e)
        
        self._polygons = [diff_shape.buffer(0)]

        # Convert Point objects to tuples
        A_tuples = {(point.x, point.y) for point in self.points}
        B_tuples = {(point.x, point.y) for point in other.points}

        # Find tuples in A that are not in B
        difference_tuples = A_tuples - B_tuples

        # Convert tuples back to Point objects if needed
        self._points = [x for x in self.points if (x.x, x.y) in difference_tuples]

    def combine(self, others: List):
        """Combines cell with other cells and deletes other cells' shapes and points.

        Args:
            others (List): list of other geocells.
        """
        for other in others:
            if other is self:
                print('Tried to combine cell with itself')
                continue 

            self.add_points(other.points)
            self.add_polygons(other.polygons)
            other._points = []
            other._polygons = []
        
    def add_polygons(self, polygons: List[Polygon]):
        """Adds list of polygons to current cell.

        Args:
            polygons (List[Polygon]): polygons
        """
        self._polygons += polygons
        
    def add_points(self, points: List[Point]):
        """Adds list of points to current cell.

        Args:
            points (List[Point]): points
        """
        try:
            self._points += points
        except TypeError:
            self._points += points.tolist()
        
    def tolist(self) -> List:
        """Converts cell to a list.

        Returns:
            List: output
        """
        return [self.cell_id, self.admin_1, self.country, len(self.points), len(self.polygons), self.shape]

    def to_pandas(self) -> gpd.GeoDataFrame:
        """Converts a cell to a geopandas DataFrame.

        Returns:
            gpd.GeoDataFrame: geopandas DataFrame.
        """
        data = [[self.cell_id, p.x, p.y] for p in self.points]
        df = pd.DataFrame(data=data, columns=CELL_COLUMNS)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs=CRS)
        return df

    def __separate_points(self, points: List[Point], polygons: List[Polygon],
                          contain_points: bool) -> Any:
        """Separates the given points and polygons from the current cell.

        Args:
            points (List[Point]): points in the new cell.
            polygons (List[Polygon]): polygons in the new cell.
            contain_points (bool): whether to smooth a cell to contain all inner points

        Returns:
            Any: New cell
        """
        coords = tuple((p.x, p.y) for p in points)
        new_name = str(hash(coords))[:12] 

        # Create new shape           
        new_shape = shapely.ops.unary_union(polygons)       
        new_shape = new_shape.buffer(0)
        if contain_points and isinstance(new_shape, MultiPolygon) == False:
            new_shape = Polygon(new_shape.exterior)

        # Create new cell
        new_cell = Cell(new_name, self.admin_1, self.country, points, [new_shape])
        return new_cell

    def voronoi_polygons(self, coords: np.ndarray=None) -> List[Polygon]:
        """Generates voronoi shapes that fill out the cell shape.

        Args:
            coords (np.ndarray): Coordinates to be tesselated.
                Defaults to None.

        Returns:
            List[Polygon]: List of polygons.

        Note:
            If coords is none, cell's own points will be tesselated.
        """
        # Get Voronoi Regions
        if coords is None:
            v_coords = np.unique(self.coords, axis=0)
        else:
            v_coords = np.unique(coords, axis=0)

        vor = Voronoi(v_coords)
        regions, vertices = voronoi_finite_polygons(vor)
        
        # Create Polygons
        polys = []
        for region in regions:
            polygon = Polygon(vertices[region])
            polys.append(polygon)
        
        # Intersect with original cell shape
        try:
            polys = [x.intersection(self.shape) for x in polys]
        except TopologicalError as e:
            print(f'Error occurred in cell: {self.cell_id}')
            raise TopologicalError(e)

        # Return area belonging to each Point
        df = pd.DataFrame({'geometry': polys})
        df = gpd.GeoDataFrame(df, geometry='geometry')
        points = [Point(p[0], p[1]) for p in coords] if coords is not None else self.points
        indices = df.sindex.nearest(points, return_all=False)[1]
        return [polys[i] for i in indices]

    def _separate_single_cluster(self, df: pd.DataFrame, cluster: int=0) -> Tuple[List[Any]]:
        """Separates a single cluster from a geocell.

        Args:
            df (pd.DataFrame): Dataframe of points.
            cluster (int): Cluster to seperate out. Defaults to 0.

        Returns:
            Tuple[List[Any]]: New cells.
        """

        # Create polygon map
        polygons = self.voronoi_polygons()

        # Separate out points
        cluster_df = df[df['cluster'] == cluster][['lng', 'lat']]
        assert len(cluster_df.index) > 0, 'Dataframe does not contain a cluster'
        cluster_points = [self.points[i] for i in cluster_df.index]
        cluster_polys = [polygons[i] for i in cluster_df.index]

        # Create new cell
        new_cell = self.__separate_points(cluster_points, cluster_polys, contain_points=True)
        return [new_cell], []

    def _separate_multi_cluster(self, df: pd.DataFrame, non_null_large_clusters: List[int]) -> List[Any]:
        """Separates multiple cluster from a geocell.

        Args:
            df (pd.DataFrame): Dataframe of points.
            non_null_large_clusters (pd.Series): Large clusters that are not unassigned.

        Returns:
            List[Any]: New cells.
        """
        # Assign unassigned points based on cluster centroids
        assigned_df = df[df['cluster'].isin(non_null_large_clusters)]
        unassigned_df = df[df['cluster'].isin(non_null_large_clusters) == False]
        cc = assigned_df.groupby(['cluster'])[['lng', 'lat']].mean().reset_index()
        cc = gpd.GeoDataFrame(cc, geometry=gpd.points_from_xy(cc.lng, cc.lat), crs=CRS)

        # Assign unassigned points
        nearest_index = cc.sindex.nearest(unassigned_df.geometry, return_all=False)[1]
        df.loc[df['cluster'].isin(non_null_large_clusters) == False, 'cluster'] = cc.iloc[nearest_index]['cluster'].values   

        # Get polygons
        if len(cc.index) == 2:
            return self._separate_single_cluster(df, cluster=cc.iloc[0]['cluster'])
        
        else:
            polygons = self.voronoi_polygons(coords=cc[['lng', 'lat']].values)

            # Separate out clusters
            new_cells = []
            for cluster, polygon in zip(cc['cluster'].unique(), polygons):
                cluster_coords = df[df['cluster'] == cluster][['lng', 'lat']]
                cluster_points = [Point(row.lng, row.lat) for _, row in cluster_coords.iterrows()]
                new_cell = self.__separate_points(cluster_points, [polygon], contain_points=True)
                new_cells.append(new_cell)

            return new_cells, [self]

    def _split_cell(self, add_to: Any, cluster_args: Tuple[float], min_cell_size: int,
                    max_cell_size: int) -> List[Any]:
        """Splits a cell into two. 

        Args:
            add_to (Any): CellCollection to add new geocells to.
            cluster_args (Tuple[float]): OPTICS clusterer arguments.
            min_cell_size (int): Minimum size of a cell.
            max_cell_size (int): Maximum size of a cell.

        Returns:
            List[Cell]: new cells which need processing.
        """
        # Don't need to cluster small cells
        if self.size < max_cell_size:
            return []

        # Get dataframe
        df = pd.DataFrame(data=self.coords, columns=['lng', 'lat'])
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs=CRS)

        # Cluster
        clusterer = OPTICS(min_samples=cluster_args[0], xi=cluster_args[1])
        df['cluster'] = clusterer.fit_predict(df[['lng', 'lat']].values)
        
        # No clusters found
        unique_clusters = df['cluster'].nunique()
        if unique_clusters < 2:
            return []

        # Erase small clusters
        cluster_counts = df['cluster'].value_counts()
        small_clusters = cluster_counts[cluster_counts < min_cell_size].index.tolist()
        df.loc[df['cluster'].isin(small_clusters), 'cluster'] = -1

        # Count clusters
        cluster_counts = df['cluster'].value_counts()
        large_clusters = cluster_counts[cluster_counts >= min_cell_size].index
        non_null_large_clusters = [x for x in large_clusters if x != -1]

        # Return if not at least two large clusters
        if len(large_clusters) < 2:
            return []

        # Dougnut extraction possible
        if len(large_clusters) == 2 and len(non_null_large_clusters) == 1:
            null_df = df[df['cluster'] == -1]
            if len(null_df) > max_cell_size:
                return []

            # Separate a single cluster
            new_cells, remove_cells = self._separate_single_cluster(df, non_null_large_clusters[0])

        # At least 2 assigned large clusters exist
        else:
            new_cells, remove_cells = self._separate_multi_cluster(df, non_null_large_clusters)

        # Detach new cells
        for new_cell in new_cells:
            self.subtract(new_cell)
            add_to.add(new_cell)

        # Remove cells as needed
        for cell in remove_cells:
            add_to.remove(cell)

        # Clean dirty splits
        clean_cells = new_cells
        if len(remove_cells) == 0:
            clean_cells += [self]

        self.__clean_dirty_splits(clean_cells)

        # Look at what cells need further processing
        proc_cells = []
        if self.size > max_cell_size and self not in remove_cells:
            proc_cells.append(self)

        for cell in new_cells:
            if cell.size > max_cell_size:
                proc_cells.append(cell)

        return proc_cells

    def __clean_dirty_splits(self, cells: List[Any]):
        """Cleans messy splits that split polygons into multiple parts.
        """
        df = pd.DataFrame(data=[x.tolist() for x in cells], columns=GEOCELL_COLUMNS)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=CRS)

        # Identifying Multipolygons
        multi_polys = df[df['geometry'].type == 'MultiPolygon']

        # Iterate through rows with Multipolygons
        for index, row in multi_polys.iterrows():

            # Find points
            points = cells[index].to_pandas()['geometry'] # .to_crs('EPSG:3857')
            
            # Splitting Multipolygons
            all_polygons = list(row['geometry'].geoms)
            
            # Finding the Largest Sub-Polygon
            largest_poly = max(all_polygons, key=lambda polygon: polygon.area)
            
            # Flag
            did_assign = False
            
            # Assigning Smaller Polygons
            for small_poly in all_polygons:
                if small_poly != largest_poly:
                    
                    # Creating a GeoSeries with the same index and CRS as 'test'
                    small_poly_gseries = gpd.GeoSeries([small_poly], index=[index], crs=CRS)
                    
                    # Exclude the original polygon during the intersection calculation
                    other_polys = df.drop(index)
                    
                    # Create a small buffer around the small polygon to account for mismatched borders
                    buffered_poly = small_poly_gseries.buffer(0.01)
                    
                    # Identify polygons that intersect with the buffered small polygon
                    intersecting_polys = other_polys[other_polys.intersects(buffered_poly.unary_union)]

                    if len(intersecting_polys) == 0:
                        continue

                    did_assign = True
                    
                    # Find the polygon that has the largest intersection area
                    largest_intersect_index = intersecting_polys.geometry.apply(
                        lambda poly: poly.intersection(buffered_poly.unary_union).area
                    ).idxmax()

                    # Checking which points fall into 'small_poly'
                    mask = points.within(small_poly)
                    points_in_small_poly = points[mask]
                    cells[index]._points = [x for x in cells[index].points if x not in points_in_small_poly]
                    cells[largest_intersect_index].add_points(points_in_small_poly)
                    
                    # Union the small polygon with the polygon having largest common border
                    cells[largest_intersect_index]._polygons = [cells[largest_intersect_index].shape.union(small_poly)]

            if did_assign:
                # Keeping the largest polygon in the original GeoDataFrame
                cells[index]._polygons = [largest_poly]

    def __eq__(self, other):
        return self.cell_id == other.cell_id
    
    def __ne__(self, other):
        return self.cell_id != other.cell_id

    def __hash__(self):
        return hash(self.cell_id)
        
    def __repr__(self):
        rep = f'Cell(id={self.cell_id}, admin_1={self.admin_1}, country={self.country}, size={len(self.points)}, num_polys={len(self.polygons)})'
        return rep
        
    def __str__(self):
        return self.__repr__()