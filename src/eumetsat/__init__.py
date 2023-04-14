
# CONSTANTS
"""
These are the constants that get used for UK satellite imagery.
In the future perhaps theses should be read from a confing file so that they can be changed
TODO: use the same presses a hemera to read the config file
"""
# MSG15-RSS
COLLECTION_ID = 'EO:EUM:DAT:MSG:HRSEVIRI'
# Image layers
IMG_LAYERS = ["HRV", "VIS006", "VIS008", "IR_016",
              "IR_039", "WV_062", "WV_073", "IR_087",
              "IR_097", "IR_108", "IR_120", "IR_134"]

# satpy reader
READER = "seviri_l1b_native"
# Target Projection EPSG:4326 standard lat lon geographic
TARGET_PROJ = 4326
# UK coords in degrees as per WSG84 [llx, lly, urx, ury] in the form of [Lon Lat Lon Lat]
AREA_EXTENT = [-12., 48., 5., 61.]
# Image size in pixels
IMG_SIZE = [500, 500]
