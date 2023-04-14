from absl import flags
from absl.testing import parameterized

import eumetsat
from eumetsat.utils import dataspec

FLAGS = flags.FLAGS


class test_utils(parameterized.TestCase):

    def test_dataspec_hardcode(self):
        # # UK coords in degrees as per WSG84 [llx, lly, urx, ury] in the form of [Lon Lat Lon Lat]
        # AREA_EXTENT = [-12., 48., 5., 61.]
        assert dataspec.max_lat == 61
        assert dataspec.min_lat == 48
        assert dataspec.max_lon == 5
        assert dataspec.min_lon == -12

    def test_dataspec_area_extent(self):
        # # UK coords in degrees as per WSG84 [llx, lly, urx, ury] in the form of [Lon Lat Lon Lat]
        # AREA_EXTENT = [-12., 48., 5., 61.]
        assert dataspec.max_lat == eumetsat.AREA_EXTENT[3]
        assert dataspec.min_lat == eumetsat.AREA_EXTENT[1]
        assert dataspec.max_lon == eumetsat.AREA_EXTENT[2]
        assert dataspec.min_lon == eumetsat.AREA_EXTENT[0]
