import openpyxl
import xlrd
import csv
import xml.etree.cElementTree as ELT
import os
import glob
import itertools
import numpy as np
from result_classes import FixationsDataset, FixationData, Fixation
import re
import config
import cv2
import utilities

def get_process_func(columns):
    """
    For a given header returns a function that is able to preprocess
    data of this kind.
    """
    for processing_func in config.PREPROCESSING_FUNCTIONS:
        for column in processing_func[0]:
            if (column not in columns) or (columns[column] is None):
                break
        else:
            return processing_func[1], processing_func[2]
    return None, None


def find_delimiter(file):
    """
    Tries to determine the delimiter used in the given csv file.

    ### Args:
        file : path
            The file the delimiter is to be determined for.

    ### Returns:
        The delimiter used in the given csv file.
    """
    argmax = ","
    with open(file, "r") as fh:
        line = fh.readline()
        max_count = 0
        for delimiter in config.DELIMITERS:
            count = line.count(delimiter)
            if (count > max_count):
                max_count = count
                argmax = delimiter
    return argmax


def prep_excel_file(file, columns=config.ET_FILE_HEADERS["default"]):
    """
    Converts all sheets of a given excel file to seperate csv files.
    The data is grouped by participant and video.

    ### Args:
        file : path
            The file to be preprocessed.

    ### Returns:
        FixationDataset
            Returns FixationDataset containing all the generated fixation data.
    """
    all_data = get_excel_data_save(file)
    if (all_data is None):
        return None
    fix_data = process_data(all_data, file, columns=columns)
    return fix_data


def convert_all_sheets(dir, delimiter="", output_delimiter=",", output_dir="preprocessed/", save=True):
    """
    Converts all the excel files in a given directory to csv files.
    ### Returns:
        A List of all created csv files.
    """
    output_files = []
    for ext in config.EXCEL_FILES:
        for file in glob.glob(os.path.join(dir, "*." + ext)):
            data = prep_excel_file(file)
            if (data is None):
                continue
            csv_files = data.save(dir=output_dir, delimiter=output_delimiter)
            output_files = output_files + csv_files
    return output_files


def process_file(file, delimiter="", columns=config.ET_FILE_HEADERS["default"]):
    """
    Preprocesses the given file.
    The files are split by participants and videos
    and the data is corrected with the calculated calibration data.

    ### Args:
        file : path
            The file to be preprocessed.

        delimiter : char
            The delimiter used in the input file.
            If none is given the delimiter is determined automatically.

    ### Returns:
        FixationDataset
            Returns FixationDataset containing all the generated fixation data.
    """
    file_ext = os.path.splitext(file)[1][1:]
    if (file_ext in config.EXCEL_FILES):
        data = get_excel_data_save(file)
        if (data is None):
            return None
        fix_data = process_data(data, file, columns=columns)
        return fix_data
    elif (file_ext in config.EXTENSIONS):
        data = get_csv_data(file, delimiter)
        if (data is None):
            return None
        fix_data = process_data(data, file, columns=columns)
        return fix_data
    else:
        print(f"Filetype: .{file_ext} is not supported.")
        return None


def process_data(data, file="et_data.csv",
                 columns=config.ET_FILE_HEADERS["default"]):
    """
    Preprocesses the given data.
    The files are split by participants and videos
    and the data ist corrected with the calculated calibration data.

    ### Args:
        data : list
            The raw data as a list of rows as dictionaries with columns as keys.
        file : path
            Path to the file the data is read from.
        columns : dict
            The header which is used to interpret the data.
        calibrator : func
            A function used to generate correction data that can be used to
            improve the accuracy of the data.

    ### Returns:
        FixationDataset
            Returns FixationDataset containing all the generated fixation data.
    """
    if (config.SELECTED_CALIBRATOR["CALIBRATE"] not in config.CALIBRATION_FUNCTIONS["CALIBRATE"]):
        calibrator = None
    else:
        calibrator = config.CALIBRATION_FUNCTIONS["CALIBRATE"][config.SELECTED_CALIBRATOR["CALIBRATE"]]
    consistent = test_data(data, columns=columns)
    if (not consistent):
        print("Inconsistent data in file {}".format(file))
        print("Make sure the correct header is selected.")
        return None
    # data should fit the specified header so from here on all columns
    # can be assumed to be present
    groups = group_data(data, columns=columns)
    fixation_data = []
    for p_id, participant_data in groups.items():
        trial_count = 0
        c_data = None
        for t, trial in participant_data.items():
            # if calibration data is in the file
            if (config.CALIBRATION_INDICES is not None and config.CALIBRATION_INDICES > 1 and "trial_id" in columns):
                # if the current trial is calibration data
                if (trial_count % config.CALIBRATION_INDICES == 0):
                    trial_count += 1
                    if (calibrator is not None):
                        c_data = calibrator(trial, columns=columns)
                    continue
            formatted_data = format_data(trial, columns=columns, c_data=c_data)
            vid = os.path.splitext(trial[0][columns["video"]])[0]
            if ("additional_info" in columns and columns["additional_info"] is not None):
                info = trial[0][columns["additional_info"]]
                fixation_data.append(FixationData(formatted_data, vid, p_id, file, info=info))
            else:
                fixation_data.append(FixationData(formatted_data, vid, p_id, file))
            trial_count += 1
    return FixationsDataset(fixation_data)


def process_list_file(list_file, delimiter="", columns=config.ET_FILE_HEADERS["default"]):
    """
    Processes a file that lists files containing et data and their
    corresponding video and participant.
    """
    if (delimiter == ""):
        delimiter = find_delimiter(list_file)
    with open(list_file, "r") as csvfile:
        rd = csv.DictReader(filter(lambda r: r[0] != "#", csvfile), delimiter=delimiter)
        data = [row for row in rd]
    if (data is None):
        print("Could not open {}".format(list_file))
        return None
    if ("participant" not in data[0] or "video" not in data[0] or "file" not in data[0]):
        print("The header of the list file must contain:\n\tparticipant, video, file")
        return None
    fixation_data = []
    for row in data:
        file = row["file"]
        if (not os.path.isfile(file)):
            directory = config.ET_FILE_DIRECTORY
            if (directory[-1] != os.sep):
                directory += os.sep
            file_name, ext = os.path.splitext(file)
            search_path = directory + "**" + os.sep + file_name + ".*"
            files = glob.glob(search_path, recursive=True)
            for fi in files:
                if (os.path.splitext(fi)[1].replace(".", "") in config.EXTENSIONS + [ext]):
                    file = fi
                    break
            else:
                print(f"No file found named \'{file}\'.")
                continue
        et_data = process_single_file(file, row["video"], row["participant"], columns=columns)
        fixation_data.append(et_data)
    return FixationsDataset(fixation_data)


def process_single_file(file, video_title, participant, delimiter="", columns=config.ET_FILE_HEADERS["default"]):
    """
    Processes an eye tracking data file that contains only data of one participant and one video.
    """
    if (os.path.splitext(file)[1] in config.EXCEL_FILES):
        data = get_excel_data_save(file)
    else:
        data = get_csv_data(file, delimiter)
    if (data is None):
        print(f"Could not open {file}")
        return None
    et_data = process_single_data(data, video_title, participant, file=file, columns=columns)
    return et_data


def get_excel_data_save(file):
    """
    Returns all data contained in an excel file as a concatenated list of rows
    which are dictionaries with the columns as keys.
    This function reads newer excel files like .xlsx aswell as older files like .xls.
    This function does not throw an exception but may return None if the file could not be read.
    """
    file_name = os.path.split(file)[1]
    file_ext = os.path.splitext(file_name)[1]
    if (file_ext[1:] not in config.EXCEL_FILES):
        print("Not an excel file.")
        return None
    if (file_ext[1:] in config.EXCEL_FILES_NEW):
        return get_xlsx_data(file)
    else:
        data_type = utilities.get_xls_type(file)
        if (data_type is None):
            print(f"File \'{file}\' not found or not a file.")
            return None
        if (data_type == "UNKNOWN"):
            print(f"File could not be read. The file may be malformatted or corrupted.")
            return None
        if (data_type == "XLS"):
            return get_xls_data(file)
        if (data_type == "XML"):
            return get_xml_data(file)
        if (data_type == "CSV"):
            return get_csv_data(file)
        print(f"Unexpected data type \'{data_type}\' when trying to open ")
        return None



def get_xlsx_data(file):
    """
    Returns all data contained in an excel file as a concatenated list of rows
    which are dictionaries with the columns as keys.
    This function reads newer excel files like .xlsx.
    """
    all_data = None
    try:
        excel_file = openpyxl.load_workbook(file)
        all_data = []
        for sheet in excel_file.worksheets:
            data = [",".join([str(cell.value) for cell in row]) for row in sheet.rows]
            data = [r for r in csv.DictReader(data, delimiter=",")]
            if (len(all_data) == 0):
                all_data = data
            elif (all_data[0].keys() == data[0].keys()):
                all_data += data
            else:
                print("Collumns of different sheets do not match. Skipping sheet.")
    except Exception:
        # excel file read failed
        print(f"Failed to read Excel file \'{file}\'")
    return all_data


def get_xls_data(file):
    """
    Reads data from an xls file and returns the data as a list of dictionaries with column names as keys.
    """
    all_data = None
    try:
        excel_file = xlrd.open_workbook(file)
        sheets = excel_file.sheets()
        all_data = []
        for i in range(len(sheets)):
            data = [",".join([str(v) for v in sheets[i].row_values(r)]) for r in range(sheets[i].nrows)]
            data = [r for r in csv.DictReader(data, delimiter=",")]
            if (len(all_data) == 0):
                all_data = data
            elif (all_data[0].keys() == data[0].keys()):
                all_data += data
            else:
                print("Collumns of different sheets do not match.")
    except xlrd.XLRDError:
        # excel file malformated fall back to interpretation as csv
        print(f"Failed to read Excel file \'{file}\'")
    return all_data


def get_xml_data(file):
    """
    Reads an xml file and returns the data as a list of dictionaries with column names as keys.
    """
    data = None
    try:
        # adapted version of
        # https://stackoverflow.com/questions/61548942/attempting-to-parse-an-xls-xml-file-using-python
        name_space = {"doc": "urn:schemas-microsoft-com:office:spreadsheet"}
        tree = ELT.parse(file)
        root = tree.getroot()
        raw_data = []
        for i, row in enumerate(root.findall('.//doc:Row', name_space)):
            raw_data.append(",".join([value.text.replace(",", ".") for cell in row for value in cell]))
        csvrd = csv.DictReader(raw_data, delimiter=",")
        data = [row for row in csvrd]
    except ELT.ParseError:
        print(f"Failed to read xml file \'{file}\'")
    return data


def get_csv_data(file, delimiter=""):
    """
    Reads a csv file and returns its content as lists of dictionaries with their column names as keys.
    """
    data = None
    try:
        if (delimiter == ""):
            delimiter = find_delimiter(file)
        with open(file, "r") as csvfile:
            rd = csv.DictReader(filter(lambda r: r[0] != "#", csvfile), delimiter=delimiter)
            data = [r for r in rd]
    except Exception:
        print(f"Failed to read the file \'{file}\'")
    return data


def process_single_data(data, video, participant, file="et_file", columns=config.ET_FILE_HEADERS["default"]):
    """
    Given raw eye tracking data from one participant and given a video and participant id
    creates a FixationData object which can be used for further processsing and added to a FixationDataset.
    """
    formatted_data = format_single_data(data, video, columns=columns)
    if (format_data is None):
        return None
    vid = os.path.splitext(video)[0]
    if ("additional_info" in columns and columns["additional_info"] is not None):
        info = data[0][columns["additional_info"]]
        return FixationData(formatted_data, vid, participant, file, info=info)
    else:
        return FixationData(formatted_data, vid, participant, file)



def process_files(files, delimiter="", columns=config.ET_FILE_HEADERS["default"], is_list=False):
    """
    Preprocesses the given list of files.
    The files are split by participants and videos
    and the data ist corrected with the calculated calibration data.

    ### Args:
        files : list
            A list containing all the files to preprocess.
        delimiter : char
            The delimiter used in the input files.
            If none is given the delimiter is determined automatically.
        columns : dict
            Names of the columns that are present in the table and the type
            of data they contain.

    ### Returns:
        FixationDataset
            Returns FixationDataset containing all the generated fixation data.
    """
    return_data = None
    prep_func, needs_list = get_process_func(columns)
    # no function was found that takes a file with the given header as input
    if prep_func is None:
        return None

    # A list file is needed that matches videos and participants with fixation data
    if (needs_list and not is_list):
        print("\nThe given header needs a list file to be specified in order to match eye tracking data to videos.")
        print("Use the argument --list <file> to do this.")
        return None
    elif (not needs_list and is_list):
        print("\nThe given header does not need a list file to be processed but the files are specified as list files.")
        print("Maybe the wrong header was chosen or the --list flag was set by accident.")
        return None

    # execute preprocessing and concatenate Data to a large dataset
    for file in files:
        data = prep_func(file, delimiter=delimiter, columns=columns)
        if (data is None):
            continue
        if (return_data is None):
            return_data = data
        else:
            return_data.append_prep(data)

    return return_data


def test_data(data, columns=config.ET_FILE_HEADERS["default"]):
    """
    Tests whether all columns that are specified are present in the file.

    ### Args:
        data : list
            The data to test for consistency.
        columns : dict
            Columns that should be present in the file.
    ### Returns:
        bool
            Returns True if the data seems to be consistent.
    """
    if (len(data) <= 0):
        return False
    if (not isinstance(data[0], dict)):
        return False
    for k, v in columns.items():
        if (v is not None):
            if (v not in data[0]):
                return False
    return True


def group_data(data, columns=config.ET_FILE_HEADERS["default"]):
    """
    Groups the given data by the video watched by the participant.

    ### Args:
        data : list
            A list containig all rows of an input file.

    ### Returns:
        dict
            A dict with the participants as keys containing dicts with
            the trial as key and the rows in the data as values.
    """
    if ("trial_id" in columns):
        trial_lambda = lambda r: r[columns["trial_id"]]
    else:
        trial_lambda = lambda r: r[columns["video"]]
    part_lambda = lambda r: r[columns["participant_id"]]
    key_lambda = lambda r: (part_lambda(r), trial_lambda(r))
    # group all rows by trial and participant
    groups = {}
    for key, group in itertools.groupby(data, key_lambda):
        group = list(group)
        if (key[0] in groups):
            groups[key[0]][key[1]] = group
        else:
            groups[key[0]] = {key[1]:group}
    return groups


def get_frame_from_time(time, fps=config.VIDEO_DEFAULT_FPS):
    """
    For a given time in milliseconds and a given framerate returns the frame
    that was displayed at time milliseconds after the video started.
    """
    if (time is None or time < 0):
        return None
    return int(time / 1000 * fps)


def get_number_from_cell(value):
    """
    Takes a value that is either a number or string.
    If it is a string it is converted to a float and returns its value.
    """
    num_regex = r"(^-?\d+(\.|\,)?\d*$)|(^-?\d*(\.|\,)\d+$)"
    if (not isinstance(value, str)):
        return value
    if re.match(num_regex, value):
        return float(value.replace(",", "."))
    else:
        return None


def format_data(data, columns=config.ET_FILE_HEADERS["default"],
                corrector=config.CALIBRATION_FUNCTIONS["CORRECT"][config.SELECTED_CALIBRATOR["CORRECT"]], c_data=None):
    """
    Formats the given data so it can be processed by VideoThreads.
    The format is: (fixation x, fixation y, start frame, end frame)

    ### Args:
        data : list
            A list of rows from the input file.
        corrector: func
            A function used to correct the data with the given c_data.
        c_data : dict
            Data used for shift correction.

    ### Returns:
        list
            A list of rows of the format: (fixation x, fixation y, start frame, end frame)
    """
    new_data = []
    if (len(data) < 1):
        return new_data

    # get fps
    vid_name = data[0]["video"]
    vid = utilities.find_video(os.path.splitext(vid_name)[0], directory=config.VIDEO_DIRECTORY, recursive=True)
    if (vid is None):
        print("No video found for {}".format(os.path.splitext(vid_name)[0]))
        fps = config.VIDEO_DEFAULT_FPS
    else:
        fps = utilities.get_fps(vid)

    # start of fixation
    if ("frame_start" in columns):
        get_start = lambda l: get_number_from_cell(l[columns["frame_start"]])
    elif ("timestamp_start" in columns):
        get_start = lambda l: get_frame_from_time(get_number_from_cell(l[columns["timestamp_start"]]), fps)
    else:
        raise ValueError("No column marking the start of a fixation present.")

    # end of fixation
    if ("frame_end" in columns):
        get_end = lambda l: get_number_from_cell(l[columns["frame_end"]])
    elif ("timestamp_end" in columns):
        get_end = lambda l: get_frame_from_time(get_number_from_cell(l[columns["timestamp_end"]]), fps)
    elif ("fixation_duration" in columns and "timestamp_start" in columns):
        get_end = lambda l: get_frame_from_time(get_number_from_cell(l[columns["timestamp_start"]]) + get_number_from_cell(l[columns["fixation_duration"]]), fps)

    # read and format lines
    for line in data:
        start = get_start(line)
        end = get_end(line)
        x = line[columns["fixation_x"]]
        y = line[columns["fixation_y"]]
        x = get_number_from_cell(x)
        y = get_number_from_cell(y)
        if (c_data is not None and corrector is not None):
            x, y = corrector((x, y), c_data)
        x = int(x)
        y = int(y)
        # test if row contains valid data
        if (start is None or end is None or x is None or y is None):
            continue
        if (end < start or start < 0 or end < 0):
            continue
        new_data.append(Fixation(x, y, int(start), int(end)))

    if len(new_data) < 1:
        print("No valid data found.")
        return []
    return new_data


def format_single_data(data, video, columns=config.ET_FILE_HEADERS["default"]):
    """
    Formats the given data of a file containing only data from one participant
    for one video so it can be processed by VideoThreads.
    The format is: (fixation x, fixation y, start frame, end frame)

    ### Args:
        data : list
            A list of rows from the input file.
        video : str
            Name of the video that corresponds to the data.

    ### Returns:
        list
            A list of rows of the format: (fixation x, fixation y, start frame, end frame)
    """
    new_data = []
    if (len(data) < 1):
        return new_data

    # get fps
    vid = utilities.find_video(os.path.splitext(video)[0], directory=config.VIDEO_DIRECTORY, recursive=True)
    if (vid is None):
        print("No video found for {}".format(os.path.splitext(video)[0]))
        fps = config.VIDEO_DEFAULT_FPS
    else:
        fps = utilities.get_fps(vid)

    # start of fixation
    if ("frame_start" in columns):
        get_start = lambda l: get_number_from_cell(l[columns["frame_start"]])
    elif ("timestamp_start" in columns):
        get_start = lambda l: get_frame_from_time(get_number_from_cell(l[columns["timestamp_start"]]), fps)
    else:
        raise ValueError("No column marking the start of a fixation present.")

    # end of fixation
    if ("frame_end" in columns):
        get_end = lambda l: get_number_from_cell(l[columns["frame_end"]])
    elif ("timestamp_end" in columns):
        get_end = lambda l: get_frame_from_time(get_number_from_cell(l[columns["timestamp_end"]]), fps)
    elif ("fixation_duration" in columns and "timestamp_start" in columns):
        get_end = lambda l: get_frame_from_time(get_number_from_cell(l[columns["timestamp_start"]]) + get_number_from_cell(l[columns["fixation_duration"]]), fps)

    # read and format lines
    for line in data:
        start = get_start(line)
        end = get_end(line)
        x = line[columns["fixation_x"]]
        y = line[columns["fixation_y"]]
        x = get_number_from_cell(x)
        y = get_number_from_cell(y)
        x = int(x)
        y = int(y)
        # test if row contains valid data
        if (start is None or end is None or x is None or y is None):
            continue
        if (end < start or start < 0 or end < 0):
            continue
        new_data.append(Fixation(x, y, int(start), int(end)))

    if len(new_data) < 1:
        print("No valid data found.")
        return None

    return new_data
