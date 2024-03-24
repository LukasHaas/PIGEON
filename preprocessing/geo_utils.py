import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

# Constants
rad_np = np.float64(6378137.0)        # Radius of the Earth (in meters)
f_np = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model

rad_torch = torch.tensor(6378137.0, dtype=torch.float64)
f_torch = torch.tensor(1.0/298.257223563, dtype=torch.float64)

PREC = 2.2204e-16

# ECEF to LLA coordinate transformation
a_sq = rad_np ** 2
e = 8.181919084261345e-2
e_sq = 6.69437999014e-3
b = rad_np * (1- f_np)
ep_sq  = (rad_np**2 - b**2) / b**2
ee = (rad_np**2-b**2)

def haversine_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the haversine distance between two sets of points

    Args:
        x (np.ndarray): points 1 (lon, lat)
        y (np.ndarray): points 2 (lon, lat)

    Returns:
        np.ndarray: haversine distance in km
    """
    x_rad, y_rad = map(np.radians, [x, y])
    delta = y_rad - x_rad
    a = np.sin(delta[:, 1] / 2)**2 + np.cos(x_rad[:, 1]) * np.cos(y_rad[:, 1]) * np.sin(delta[:, 0] / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = (rad_np * c) / 1000
    return km

def haversine(x: Tensor, y: Tensor) -> Tensor:
    """Computes the haversine distance between two sets of points

    Args:
        x (Tensor): points 1 (lon, lat)
        y (Tensor): points 2 (lon, lat)

    Returns:
        Tensor: haversine distance in km
    """
    x_rad, y_rad = torch.deg2rad(x), torch.deg2rad(y)
    delta = y_rad - x_rad
    a = torch.sin(delta[:, 1] / 2)**2 + torch.cos(x_rad[:, 1]) * torch.cos(y_rad[:, 1]) * torch.sin(delta[:, 0] / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    km = (rad_torch * c) / 1000
    return km

# Implementation to calculate all possible combinations of distances in parallel
def haversine_matrix(x: Tensor, y: Tensor) -> Tensor:
    """Computes the haversine distance between two matrices of points

    Args:
        x (Tensor): matrix 1 (lon, lat) -> shape (N, 2)
        y (Tensor): matrix 2 (lon, lat) -> shape (2, M)

    Returns:
        Tensor: haversine distance in km -> shape (N, M)
    """
    x_rad, y_rad = torch.deg2rad(x), torch.deg2rad(y)
    delta = x_rad.unsqueeze(2) - y_rad
    p = torch.cos(x_rad[:, 1]).unsqueeze(1) * torch.cos(y_rad[1, :]).unsqueeze(0)
    a = torch.sin(delta[:, 1, :] / 2)**2 + p * torch.sin(delta[:, 0, :] / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    km = (rad_torch * c) / 1000
    return km

# Implementation to calculate all possible combinations of distances in parallel
def haversine_matrix_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the haversine distance between two matrices of points

    Args:
        x (np.ndarray): matrix 1 (lon, lat) -> shape (N, 2)
        y (np.ndarray): matrix 2 (lon, lat) -> shape (2, M)

    Returns:
        np.ndarray: haversine distance in km -> shape (N, M)
    """
    x_rad, y_rad = np.deg2rad(x), np.deg2rad(y)
    delta = np.expand_dims(x_rad, axis=2) - y_rad
    p = np.expand_dims(np.cos(x_rad[:, 1]), axis=1) * np.expand_dims(np.cos(y_rad[1, :]), axis=0)
    a = np.sin(delta[:, 1, :] / 2)**2 + p * np.sin(delta[:, 0, :] / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = (rad_np * c) / 1000
    return km

# Inspired by https://github.com/kvenkman/ecef2lla
def lla2ecef_np(coords: np.ndarray) -> np.ndarray:
    """Converts longitude and latitude to 3D coordinates

    Args:
        coords (np.ndarray): array of (lon, lat)

    Returns:
        np.ndarray: 3d coordinates
    """
    coords = np.radians(coords)
    cosLat = np.cos(coords[:, 1])
    sinLat = np.sin(coords[:, 1])

    ff = (1.0 - f_np)**2
    c = 1 / np.sqrt(cosLat**2 + ff * sinLat**2)
    s = c * ff

    x = (rad_np * c) * cosLat * np.cos(coords[:, 0])
    y = (rad_np * c) * cosLat * np.sin(coords[:, 0])
    z = (rad_np * s) * sinLat
    spatial_coords = np.array([x, y, z]).transpose()
    return spatial_coords

# Inspired by https://github.com/kvenkman/ecef2lla
def lla2ecef(coords: Tensor) -> Tensor:
    """Converts longitude and latitude to 3D coordinates

    Args:
        coords (Tensor): array of (lon, lat)

    Returns:
        Tensor: 3d coordinates
    """
    coords = torch.deg2rad(coords)
    cosLat = torch.cos(coords[:, 1])
    sinLat = torch.sin(coords[:, 1])

    ff = (1.0 - f_torch)**2
    c = 1 / torch.sqrt(cosLat**2 + ff * sinLat**2)
    s = c * ff

    x = (rad_torch * c) * cosLat * torch.cos(coords[:, 0])
    y = (rad_torch * c) * cosLat * torch.sin(coords[:, 0])
    z = (rad_torch * s) * sinLat

    spatial_coords = torch.vstack([x, y, z]).t()
    return spatial_coords

# Inspired by https://github.com/kvenkman/ecef2lla
def ecef2lla_np(coords: np.ndarray) -> np.ndarray: 
    """Converts ECEF coordinates to longitude and latitude

    Args:
        coords (np.ndarray): array of (x, y, z)

    Returns:
        np.ndarray: array of (longitude, latitude)
    """
    # Extract
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    # Compute
    r = np.sqrt(x**2 + y**2)
    f = (54*b**2)*(z**2)
    g = r**2 + (1 - e_sq) * (z**2) - e_sq * ee * 2
    c = (((f / (g**2)) * (r**2)) / g) * (e_sq**2)
    s = (1 + c + np.sqrt(c**2 + 2*c)) ** (1/3.)
    p = f / (3.0 * (g**2) * (s + (1.0 / s) + 1) **2)
    q = np.sqrt(1 + 2 * p * e_sq**2)
    r_0 = -(p * e_sq * r) / (1 + q) + np.sqrt(0.5 * (rad_np**2) * (1 + (1. / q)) \
          - p * (z**2) * (1 - e_sq) / (q * (1 + q)) - 0.5 * p * (r**2))
    v = np.sqrt((r - e_sq * r_0)**2 + (1 - e_sq) * z**2)
    z_0 = (b**2) * z / (rad_np * v)
    phi = np.arctan((z + ep_sq * z_0) / r)
    lambd = np.arctan2(y, x)

    return np.array([lambd * 180 / np.pi, phi * 180 / np.pi]).transpose()

# Inspired by https://github.com/kvenkman/ecef2lla
def cylindrical2geodetic(rho: Tensor, z: Tensor, a: Tensor, f: Tensor,
                         device: torch.device) -> Tensor:

    # Reshape
    zz = z.unsqueeze(1)

    # Spheroid properties
    b = (1 - f) * a       # Semiminor axis
    e2 = f * (2 - f)      # Square of (first) eccentricity
    ae2 = a * e2
    bep2 = b * e2 / (1 - e2)   # b * (square of second eccentricity)

    # Starting value for parametric latitude (beta), following Bowring 1985
    vector = torch.hstack([rho, zz])
    r = torch.norm(vector, dim=-1, keepdim=True)
    u = a * rho                    # vs. u = b * rho (Bowring 1976)
    v = b * zz * (1 + bep2 / r)   # vs. v = a * z   (Bowring 1976)

    w = v / u
    vector = torch.hstack([torch.ones_like(w).to(device), w])
    cosbeta = torch.sign(u) / torch.norm(vector, dim=-1, keepdim=True)

    w = u / v
    vector = torch.hstack([torch.ones_like(w).to(device), w])
    sinbeta = torch.sign(v) / torch.norm(vector, dim=-1, keepdim=True)

    # Fixed-point iteration with Bowring's formula
    # (typically converges within three iterations or less)
    count = 0
    iterate = True
    while iterate and count < 5:
        cosprev = cosbeta
        sinprev = sinbeta
        u = rho - ae2  * (cosbeta**3)
        v = zz   + bep2 * (sinbeta**3)
        au = a * u
        bv = b * v

        w = bv / au
        vector = torch.hstack([torch.ones_like(w).to(device), w])
        cosbeta = torch.sign(au) / torch.norm(vector, dim=-1, keepdim=True)

        w = au / bv
        vector = torch.hstack([torch.ones_like(w).to(device), w])
        sinbeta = torch.sign(bv) / torch.norm(vector, dim=-1, keepdim=True)

        vector = torch.hstack([cosbeta - cosprev, sinbeta - sinprev])
        iterate = torch.any(torch.norm(vector, dim=-1, keepdim=True) > PREC)
        count += 1
 
    # Final latitude in degrees or radians
    phi = torch.atan2(v, u)
    return phi

# Inspired by https://github.com/kvenkman/ecef2lla
def ecef2lla(coords: Tensor, device: torch.device) -> Tensor: 
    """Converts ECEF coordinates to longitude and latitude

    Args:
        coords (Tensor): array of (x, y, z)

    Returns:
        Tensor: array of (longitude, latitude)
    """
    # Extract
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Compute longitude
    lon = torch.atan2(y, x).unsqueeze(1)
    
    # Determine radial distance from polar axis
    rho = torch.norm(coords[:, :2], dim=-1, keepdim=True)

    # Compute latitude
    lat = cylindrical2geodetic(rho, z, rad_torch, f_torch, device)

    # Longitude and latitude
    ll = torch.hstack([lon, lat]) * 180 / torch.pi
    return ll