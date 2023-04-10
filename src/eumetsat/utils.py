import pyresample as pr
from pyproj import CRS

from src.eumetsat import AREA_EXTENT, IMG_SIZE, TARGET_PROJ


def get_area_def():
    area_extent = AREA_EXTENT
    area_id = "UK"
    proj_crs = CRS.from_user_input(TARGET_PROJ).to_dict()  # Target Projection EPSG:4326 standard lat lon geograpic
    output_res = IMG_SIZE  # Target res in pixels
    area_def = pr.geometry.AreaDefinition.from_extent(area_id, proj_crs, output_res, area_extent)
    return area_def