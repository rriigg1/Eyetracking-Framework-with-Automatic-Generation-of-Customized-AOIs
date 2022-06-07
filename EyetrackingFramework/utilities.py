import os
import cv2
import glob
import random
import dlib
import xml.etree.cElementTree as ELT
import xlrd
import csv

# framework imports
import config
import landmark_utils
import argparse

LAST_COLOR_COUNT = 0


def resolve_file_list(file_list):
    """
    For a list of files resolves files given using regex and returns a concatenated list of files.
    """
    files = []
    for file in file_list:
        if (not os.path.exists(file) or not os.path.isfile(file)):
            glob_files = list(filter(lambda f: os.path.isfile(f), glob.glob(file)))
            if (len(glob_files) > 0):
                files += glob_files
        else:
            files.append(file)
    return files


def check_file(file):
    """
    Tests whether the given string is a path to an existing file.
    """
    if (not os.path.exists(file) or not os.path.isfile(file)):
        # resolve regex to test if files are found
        files = list(filter(lambda f: os.path.isfile(f), glob.glob(file)))
        if (len(files) > 0):
            return file
        else:
            raise argparse.ArgumentTypeError(f"\'{file}\' is not a valid path to a file.")
    else:
        return file


def check_char(string):
    """
    Tests whether the given string is a single char (length = 1).
    """
    if (len(string) != 1):
        raise argparse.ArgumentTypeError(str(string) + " needs to be a char.")
    else:
        return string


def check_dir(string):
    """
    Tests if the given name is a directory and creates it if necessary.
    """
    if (not os.path.exists(string)):
        os.mkdir(string)
    if (os.path.isdir(string)):
        return string
    else:
        raise argparse.ArgumentTypeError(str(string) + " needs to be a valid directory.")


def check_existing_dir(string):
    """
    Tests whether a given string is a valid existing directory.
    """
    if (not os.path.exists(string) or not os.path.isdir(string)):
        raise argparse.ArgumentTypeError(str(string) + " needs to be an existing directory.")
    else:
        return string


def test_dtypes(indata, dtypes):
    """
    Tests if a specified types of data are included in a list of data.
    Args:
        indata : list
            A list of different types of data.
        dtypes : list
            A list of types of data to retrieve the indices for.
    Returns:
        False if one of the given types is not present in the current data.
        A list of the indices where the given types of data are save otherwise.
    """
    idx = []
    for dtype in dtypes:
        if (dtype not in config.DATA_TYPES):
            print("No data type named: {}".format(dtype))
            return False
        needed_type = config.DATA_TYPES[dtype]
        for i, d in enumerate(indata):
            if (isinstance(d, needed_type)):
                idx.append(i)
                break
        else:
            return False
    return idx


def get_xls_type(file):
    """
    For a given xls file tests whether it actually contains binary excel data, xml or even csv.
    ### Returns:
        string - A String dependent on data in the file (XLS, XML, CSV, UNKNOWN).
        If the file could not be found this function returns None
    """
    if (not os.path.exists(file) or not os.path.isfile(file)):
        return None
    try:
        wb = xlrd.open_workbook(file)
        return "XLS"
    except xlrd.XLRDError:
        try:
            tree = ELT.parse(file)
            return "XML"
        except ELT.ParseError:
            try:
                with open(file, "rt") as fh:
                    csvdoc = csv.DictReader(filter(lambda row: row[0] != "#", fh), delimiter = "\t")
                    firstrow = next(row for row in csvdoc)
                    return "CSV"
            except Exception:
                return "UNKNOWN"



def get_fps(video):
    """
    For a given video file returns the fps of the video.

        ### Args:
            video : file
                Video file for which to determine the fps.
        ### Returns:
            float
                Returns the fps of the video.
    """
    if (not os.path.isfile(video)):
        return config.VIDEO_DEFAULT_FPS # maybe None should be returned here instead
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if (fps is None or fps == 0):
        return config.VIDEO_DEFAULT_FPS # fps can not be determined automatically
    return fps


def find_video(title, directory=None, recursive=True):
    """
    For a given title and directory returns a video file with a supported format.

        ### Args:
            title : str
                Title of the video to search for.
            directory : str
                Directory in which to look for the video.
            recursive : bool
                Whether to recursively search subdirectories.

        ### Returns:
            str
                Path to the found video file or None if no file was found.
    """
    if (directory is None):
        directory = config.VIDEO_DIRECTORY
    if (directory[-1] != os.sep):
        directory += os.sep
    if (recursive):
        search_path = directory + "**" + os.sep + title + ".*"
        files = glob.glob(search_path, recursive=True)
    else:
        search_path = directory + "*" + title + ".*"
        files = glob.glob(search_path, recursive=False)
    if (files is None or len(files) == 0):
        return None # No files are matching the name
    else:
        for f in files:
            if (os.path.splitext(f)[1].replace(".", "") in config.VIDEO_FORMATS):
                return f
        else:
            print("No video with a supported format was found for '{}'.".format(title))
            print("Supported formats:\n\t{}".format(config.VIDEO_FORMATS))
            return None # No file with a supported format


def get_frame_of_video(video_file, frame=0):
    """
    Opens th given video_file and returns the image which represents the given frame.
    By default the first frame of the video is returned.
    """
    video = cv2.VideoCapture(video_file)
    if (video is None):
        print(f"\'{video_file}\' is not a valid video.")
        return None
    if (not video.set(cv2.CAP_PROP_POS_FRAMES, frame)):
        print(f"Can't get frame no. {frame} of the video \'{video_file}\'.")
        return None
    success, image = video.read()
    video.release()
    if (not success):
        print(f"Can't get frame no. {frame} of the video \'{video_file}\'.")
        return None
    else:
        return image


def get_color(random_color=False, gray_scale=False, max_brightness=255, min_brightness=0):
    """
    Returns either a random color or a color from COLOR_LIST.
    For random colors a maximum and minimum brightness can be set.
    Additionally all colors can be converted to gray scale.
    """
    global LAST_COLOR_COUNT
    if (random_color is True):
        if (gray_scale):
            val = random.randint(min_brightness, max_brightness)
            return (val, val, val)
        else:
            return (random.randint(min_brightness, max_brightness), random.randint(min_brightness, max_brightness), random.randint(min_brightness, max_brightness))
    else:
        if (random_color is False):
            idx = random.randint(0, len(config.COLOR_LIST) - 1)
        else:
            idx = random_color
        LAST_COLOR_COUNT = idx
        col = config.COLOR_LIST[idx%len(config.COLOR_LIST)]
        if (isinstance(col, int)):
            col = int_to_bgr(col)
        if (gray_scale):
            val = (col[0] + col[1] + col[2]) / 3
            return (val, val, val)
        else:
            return col

def int_to_bgr(val):
    """
    For a given integer that describes a rgb value returns the corresponding
    bgr tuple.
    """
    r = (val & 0xff0000) >> 16
    g = (val & 0x00ff00) >> 8
    b = val & 0x0000ff
    return (b, g, r)


def get_base_face(base_image_file: str):
    """
    For a given path of an image returns a tuple of the corresponding image
    and the coordinates of the calculated landmarks.
    Args:
        base_image_file : str
            A path to an image.
    Returns:
        tuple
            A tuple (image, coordinates of the landmarks in the image).
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR)
    image = cv2.imread(base_image_file)
    timage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(timage, 0)
    if (not dets or len(dets) > 1):
        raise ValueError("Exactly one face should be present in the image.")
    dets = dets[0]
    coords = landmark_utils.generate_landmarks_simple(predictor, timage, dets)
    return image, coords
