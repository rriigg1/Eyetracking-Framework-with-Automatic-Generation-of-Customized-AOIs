# general settings
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
SCREEN_SIZE = {"width": 1920, "height": 1080}
VIDEO_SIZE = {"width": 1920, "height": 1080}
# size of the image the default landmarks where taken frome
DEFAULT_LANDMARK_SIZE = (700, 700)

# =======================
# files
# xls is the older excel format and may either be a binary excel file an xml file or even a csv file
# because of this uncertainty xls files may sometimes not work. In this case maybe try exporting it as either xlsx or csv
EXCEL_FILES = ["xlsx", "xls", "xml"]
EXCEL_FILES_NEW = ["xlsx"] # as long as openpyxl supports it it should work. Add extensions at your own risk
EXTENSIONS = ["xls", "xlsx", "csv", "txt"]
DELIMITERS = [",", "\t", ";", ":", "|"]

VIDEO_FORMATS = ["mp4", "avi", "mov", "mpeg", "flv", "wmv"]
VIDEO_DIRECTORY = "videos"
ET_FILE_DIRECTORY = "."
VIDEO_DEFAULT_FPS = 30  # only used if the fps can not be determined

# default values
SELECTED_CALIBRATOR = {"CALIBRATE": "CROSS",
                       "CORRECT": "SHIFT"}
SELECTED_AOI_MANAGER = {"GENERATOR": "SIMPLE_POINTS",
                        "GETTER": "CLOSEST_AOI"}
SELECTED_GROUPING = "FACE_SIMPLE"


import result_classes
# data types
DATA_TYPES = {
    "aois": result_classes.FixatedAOIsData,
    "landmarks": result_classes.Landmarks,
    "fixations": result_classes.FixationsDataset,
    "scanpath_distance": result_classes.ScanpathDistances,
    "fixation_counts": result_classes.FixationCounts,
    "aoi_saccades": result_classes.AOISaccadesData,
    "dwell_times": result_classes.Dwelltimes
}


# ==================================
# eye tracking data file structure
# missing columns can either be set to None or be ommited.
DEFAULT_ET_FILE = {
    "fixation_x": "CURRENT_FIX_X",  # x position of the fixation
    "fixation_y": "CURRENT_FIX_Y",  # y position of the fixation
    "frame_start": "VIDEO_FRAME_INDEX_START",  # frame of the video in which the fixation starts
    "frame_end":  "VIDEO_FRAME_INDEX_END",  # frame of the video in which the fixation ends
    "video":      "video",  # Video that corresponds to the eytracking data
    "participant_id": "RECORDING_SESSION_LABEL",  # ID of the participant
    "trial_id": "TRIAL_INDEX",  # ID of the trial
    "timestamp_start": None,  # time in ms when the fixation started
    "timestamp_end": None,  # time in ms when the fixation ended
    "fixation_duration": "CURRENT_FIX_DURATION",  # duration of the fixation in ms
    "additional_info": "message"  # additional information that can be used later on to distinguish types of data
}
# header for fixation-exports by pupillabs
PUPILLABS_ET_FILE = {
    "fixation_x": "CURRENT_FIX_X",  # x position of the fixation
    "fixation_y": "CURRENT_FIX_Y",  # y position of the fixation
    "frame_start": "start_frame_index",  # frame of the video in which the fixation starts
    "frame_end":  "end_frame_index",  # frame of the video in which the fixation ends
    "timestamp_start": "start_timestamp",  # time in ms when the fixation started
    "fixation_duration": "duration"  # duration of the fixation in ms
}

# header for a simple file containing only basic information. this file needs
# to be loaded with --list because the participant and video are not included
MINIMAL_FILE = {
    "fixation_x": "x",  # x position of the fixation
    "fixation_y": "y",  # y position of the fixation
    "frame_start": "start_frame",  # frame of the video in which the fixation starts
    "frame_end":  "end_frame"  # frame of the video in which the fixation ends
}

ET_FILE_HEADERS = {
    "default": DEFAULT_ET_FILE,
    "pupillabs": PUPILLABS_ET_FILE,
    "minimalistic": MINIMAL_FILE
}

# ===================================
# calibration
# CALIBRATE is used to calculate the correction data and CORRECT is used
# to correct a given coordinate
CALIBRATION_FUNCTIONS = {"CALIBRATE": {}, "CORRECT": {}}
from calibration import calibrate_cross, shift_correct
CALIBRATION_FUNCTIONS["CALIBRATE"]["CROSS"] = calibrate_cross
CALIBRATION_FUNCTIONS["CORRECT"]["SHIFT"] = shift_correct

CALIBRATION_INDICES = 2  # if greater than one, uses every n-th trial as calibration data


import preprocessor
PREPROCESSING_FUNCTIONS = [
    # (columns, preprocessing function, needs list file)
    (("fixation_x", "fixation_y", "participant_id", "video", "frame_start", "frame_end"), preprocessor.process_file, False),
    (("fixation_x", "fixation_y", "participant_id", "video", "timestamp_start", "timestamp_end"), preprocessor.process_file, False),
    (("fixation_x", "fixation_y", "participant_id", "video", "timestamp_start", "fixation_duration"), preprocessor.process_file, False),
    (("fixation_x", "fixation_y", "frame_start", "frame_end"), preprocessor.process_list_file, True),
    (("fixation_x", "fixation_y", "timestamp_start", "timestamp_end"), preprocessor.process_list_file, True),
    (("fixation_x", "fixation_y", "timestamp_start", "fixation_duration"), preprocessor.process_list_file, True)
]

# data loading
LOADING_FUNCTIONS = {
   "aois": result_classes.FixatedAOIsData.load,
   "landmarks": result_classes.Landmarks.load,
   "fixations": result_classes.FixationsDataset.load
}

# ====================================================
# AOI generation and detection

# landmarks of a default face. Can be used when no face is available or as a placeholder
DEFAULT_LANDMARKS = [(70, 201), (79, 277), (92, 353), (104, 427),
                     (125, 497), (163, 558), (217, 609), (281, 645),
                     (352, 653), (423, 643), (487, 605), (540, 555),
                     (579, 493), (600, 422), (613, 348), (626, 272),
                     (632, 197), (111, 151), (145, 117), (195, 106),
                     (247, 112), (296, 135), (404, 135), (454, 111),
                     (506, 105), (556, 115), (593, 148), (350, 196),
                     (350, 253), (350, 309), (350, 365), (297, 395),
                     (323, 402), (350, 409), (378, 401), (404, 394),
                     (172, 204), (205, 188), (243, 190), (274, 214),
                     (239, 221), (200, 220), (427, 214), (459, 189),
                     (497, 188), (530, 203), (500, 219), (462, 220),
                     (232, 474), (274, 460), (317, 453), (349, 462),
                     (381, 453), (421, 459), (465, 470), (425, 513),
                     (384, 530), (349, 534), (313, 531), (270, 515),
                     (250, 478), (316, 482), (349, 486), (382, 482),
                     (446, 474), (382, 484), (349, 489), (315, 485)]

# landmarks in background only used when generator "FACE_AND_BACKGROUND" is used
BACKGROUND_LANDMARKS = [(100, 100),
                        (SCREEN_SIZE["width"]-100, 100),
                        (100, SCREEN_SIZE["height"]-100),
                        (SCREEN_SIZE["width"]-100, SCREEN_SIZE["height"]-100)]

# needed to resolve circular import
AOI_MANAGERS = {"GENERATOR": {}, "GETTER": {}}
# =================================

import landmark_utils
import aoi
# ======================================
# dict of value type: (function, AOI type)
# The type is used to determine whether a GENERATOR and GETTER are compatible.
AOI_MANAGERS["GENERATOR"]["SIMPLE_POINTS"] = landmark_utils.generate_landmarks_simple
AOI_MANAGERS["GENERATOR"]["FACE_AND_BACKGROUND"] = landmark_utils.generate_landmarks_and_background
AOI_MANAGERS["GETTER"]["CLOSEST"] = aoi.get_closest_point
AOI_MANAGERS["GETTER"]["AOI"] = aoi.get_aoi
AOI_MANAGERS["GETTER"]["CLOSEST_AOI"] = aoi.get_closest_aoi

# functions to calculate distances to various shapes
# The tuples contain the function and a dict of additional parameters. The names do not have to match the actual parameter names.
# The parameters are only use in the aoi creator tool and can therefore be left out if not needed.
DISTANCE_FUNCTIONS = {
    "POINT": (aoi.near_point, {"radius": (float,20)}),
    "LINE": (aoi.on_polyline, {"thickness": (float,20), "loop": (bool, False)}),
    "POLYGON": (aoi.in_polygon, {"padding": (float,10)}),
    "HULL": (aoi.in_hull, {"padding": (float,10)}),
    "VORONOI-HESSELS": (aoi.in_hessels_voronoi, {"maximum_distance": (float,-1)})
}

AOI_NONE = "NONE"    # used when no aoi was fixated
AOI_NO_FACE = "NONE" # used when no face is present in the frame

# AOIs
# "Grouping":{"AOI1":[(landmarks, type, *args),...], "AOI2":...}
AOI_GROUPINGS = {
    "PARTS_OUTLINES":
        {
            "JAW": [(list(range(17)), "LINE", [15])],
            "LEFT_EYEBROW": [(list(range(17, 22)), "LINE", [20])],
            "RIGHT_EYEBROW": [(list(range(22, 27)), "LINE", [20])],
            "NOSE": [(list(range(27, 36)), "HULL", [10])],
            "LEFT_EYE": [(list(range(36, 42)), "POLYGON", [20])],
            "RIGHT_EYE": [(list(range(42, 48)), "POLYGON", [20])],
            "MOUTH": [(list(range(48, 60)), "POLYGON", [20])],
            "FACE": [(list(range(68)), "HULL", [1])]
        },
    "FACE_SIMPLE":
        {
            "EYEBROWS": [(list(range(17, 22)), "LINE", [20]), (list(range(22, 27)), "LINE", [20])],
            "NOSE": [(list(range(27, 36)), "HULL", [20])],
            "EYES": [(list(range(36, 42)), "HULL", [25]), (list(range(42, 48)), "HULL", [25])],
            "MOUTH": [(list(range(48, 60)), "POLYGON", [25])],
            "JAW": [(list(range(17)), "LINE", [20])],
            "FACE": [(list(range(68)), "HULL", [20])]
        },
    "FACE_OUTLINES":
        {
            "FACE": [(list(range(68)), "HULL", [20])]
        },
    "HESSELS_VORONOI":
        {
            "NOSE": [([30], "VORONOI-HESSELS", [150])],
            "LEFT_EYE": [(list(range(36, 42)), "VORONOI-HESSELS", [150])],
            "RIGHT_EYE": [(list(range(42, 48)), "VORONOI-HESSELS", [150])],
            "MOUTH": [(list(range(48, 68)), "VORONOI-HESSELS", [150])]
        },
    "HESSELS_AOIS":
        {
            "NOSE": [(list(range(27, 36)), "HULL", [20])],
            "EYES": [(list(range(36, 42)), "HULL", [25]), (list(range(42, 48)), "HULL", [25])],
            "MOUTH": [(list(range(48, 68)), "HULL", [25])]
        },
    "BACKGROUND":
        {
            "ANY": [(list(range(72)), "POINT", [30])]
        }
}

# =======================================
# visualizations
# =======================================

# list of colors to use for visualizations
# can be adapted to ensure better visability
COLOR_LIST = [
    0x92B3C5,
    0x6A1346,
    0xF3DC95,
    0x34A44A,
    0x13466A,
    0xCFF99F,
    0xF3CC25,
    0xa4942A,
    (0, 240, 255),
    (189, 189, 255),
    (238, 171, 0),
    (0, 127, 255),
    (138, 0, 235),
    (0, 255, 0)
]

LANDMARK_COLOR = (0, 255, 0)  # color used to visualize landmarks
LANDMARK_SIZE = 3             # radius of landmarks when visualized
FIXATION_SIZE = 5             # radius of fixations when visualized
ARROW_THICKNESS = 2           # thickness of the arrows drawn to visualize scanpath
ARROW_TIP_SIZE = 6            # size of the tip of the arrow

# ========================================
# configs for different visualizations
VISUALIZATIONS = {
    "FIXATIONS":
        {
            "draw_aois": True,         # draw aois
            "draw_fixations": True,   # draw circles where fixations occurred
            "draw_arrows": False,     # draw arrows between fixations to visualize scanpath
            "draw_landmarks": False,  # draw landmarks as dots
            "random_colors": -1,   # use random colors / seed / -1=use color list
            "label_aois": False,       # label the aois
            "label_fixations": True,  # label the fixations
            "fixation_count": 5,      # number of fixations to draw at a time. Last n fixations are shown
            "fill_polygons": True    # only draw outlines of the polygons
        },
    "AOIS":
        {
            "draw_aois": True,
            "draw_fixations": False,
            "draw_arrows": False,
            "draw_landmarks": False,
            "random_colors": -1,
            "label_aois": True,
            "label_fixations": False,
            "fill_polygons": True
        },
    "VORONOI":
        {
            "draw_aois": True,
            "draw_fixations": False,
            "draw_arrows": False,
            "draw_landmarks": False,
            "random_colors": False,
            "label_aois": False,
            "label_fixations": False,
            "fill_polygons": True
        },
    "SCANPATH":
        {
            "draw_aois": True,
            "draw_fixations": True,
            "draw_arrows": True,
            "draw_landmarks": False,
            "random_colors": -1,
            "label_aois": False,
            "label_fixations": False,
            "fill_polygons": True,
            "fixation_count": 5
        },
    "LANDMARKS":
        {
            "draw_aois": False,
            "draw_fixations": False,
            "draw_arrows": False,
            "draw_landmarks": True,
            "random_colors": -1,
            "label_aois": False,
            "label_fixations": False,
            "fill_polygons": True,
        }
}


# ========================================
# vector based similarity measures
import scanpath_analysis
VECTOR_DIFF_FUNCTIONS = {
    "FIXATIONS":
        {
            "POSITION": scanpath_analysis.get_fixation_pos_diff,
            "DURATION": scanpath_analysis.get_fixation_duration_diff
        },
    "SACCADES":
        {
        }
}
