from pyproj import CRS
from pyresample.geometry import AreaDefinition

from eumetsat import AREA_EXTENT, IMG_SIZE, TARGET_PROJ


class _DataSpecs:
    @property
    def min_lat(self) -> float:
        return AREA_EXTENT[1]

    @property
    def max_lat(self) -> float:
        return AREA_EXTENT[3]

    @property
    def min_lon(self) -> float:
        return AREA_EXTENT[0]

    @property
    def max_lon(self) -> float:
        return AREA_EXTENT[2]

dataspec = _DataSpecs()

def get_area_def() -> AreaDefinition:
    area_extent = AREA_EXTENT
    area_id = "UK"
    proj_crs = CRS.from_user_input(TARGET_PROJ).to_dict()  # Target Projection EPSG:4326 standard lat lon geograpic
    output_res = IMG_SIZE  # Target res in pixels
    area_def = AreaDefinition.from_extent(area_id, proj_crs, output_res, area_extent)
    return area_def
