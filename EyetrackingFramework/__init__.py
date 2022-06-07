import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config # as long as config is first imported here the variables set in config are valid everywhere
import landmark_utils
import aoi
import aoi_mapper
import calibration
import etstatistics as statistics
import face_mapper
import mainparser
import preprocessor
import result_classes
import scanpath_analysis
import utilities
import visualizations
