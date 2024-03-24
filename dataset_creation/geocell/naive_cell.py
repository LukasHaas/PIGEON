import numpy as np
from typing import NamedTuple, Set
from shapely import geometry

class Point(NamedTuple):
    idx: int
    lng: float
    lat: float

class NaiveCell:
    def __init__(self, points: Set[Point]):
        assert len(points) > 0, 'No points were passed.'
        self.points = points
        
        self.min_lng = min(points, key=lambda p: p.lng).lng
        self.max_lng = max(points, key=lambda p: p.lng).lng
        self.min_lat = min(points, key=lambda p: p.lat).lat
        self.max_lat = max(points, key=lambda p: p.lat).lat
        
    @property
    def longitudes(self):
        return np.array([x.lng for x in self.points])
    
    @property
    def latitudes(self):
        return np.array([x.lat for x in self.points])
    
    @property
    def centroid(self):
        return np.mean(self.longitudes), np.mean(self.latitudes) 
    
    @property
    def area(self):
        return (self.max_lng - self.min_lng) * (self.max_lat - self.min_lat)
    
    @property
    def polygon(self):
        points = [(self.max_lng, self.max_lat), (self.max_lng, self.min_lat),
                  (self.min_lng, self.min_lat), (self.min_lng, self.max_lat)]
        poly = geometry.Polygon(points)
        return poly
    
    def split(self):
        split_on_lat = self.__should_split_on_lat()
        p1, p2 = set(), set()
        
        if split_on_lat:
            thresh = (self.max_lat + self.min_lat) / 2
            for p in self.points:
                if p.lat < thresh: 
                    p1.add(p)
                else: 
                    p2.add(p)
        else:
            thresh = (self.max_lng + self.min_lng) / 2
            for p in self.points:
                if p.lng < thresh: 
                    p1.add(p)
                else: 
                    p2.add(p)
                    
        return [Cell(p1), Cell(p2)]
    
    def __should_split_on_lat(self):
        lat_range = self.max_lat - self.min_lat
        lng_range = self.max_lng - self.min_lng
        return lat_range > lng_range

    def __len__(self):
        """
        Number of datapoints in cell
        """
        return len(self.points)
    
    def __str__(self):
        """
        String representation
        """
        if len(self) == 0:
            return "Cell is empty!"
        return (
            f"Num points: {len(self)}"
            f"\nCentroid: {self.centroid}"
        )
    
    def __repr__(self):
        return f'Cell(num_cells={len(self)}, centroid: {self.centroid})'