import os
import sys
import argparse
from tqdm import tqdm
import cv2
import inspect
import numpy as np

# modules of this package
import config
import preprocessor
import utilities
import aoi_mapper
import landmark_utils
import result_classes
import etstatistics
import scanpath_analysis
import face_mapper
import visualizations
import heatmap_generator


def preprocess(args, indata):
    """
    Preprocesses the given files so they can be used for further analysis.
    """
    print("Preprocessing... ", end="")
    if (args.header not in config.ET_FILE_HEADERS):
        print("\'{}\' is not a valid header".format(args.header))
    header = config.ET_FILE_HEADERS[args.header]

    if (args.calibrator == "NONE" or args.correct == "NONE"):
        print("No calibration is used to correct the data.")
        config.SELECTED_CALIBRATOR["CALIBRATE"] = None
        config.SELECTED_CALIBRATOR["CORRECT"] = None
    else:
        if (args.calibrator is not None):
            config.SELECTED_CALIBRATOR["CALIBRATE"] = args.calibrator
        if (args.correct is not None):
            config.SELECTED_CALIBRATOR["CORRECT"] = args.correct
    config.VIDEO_DIRECTORY = args.video_directory
    if (args.delimiter is None):
        args.delimiter = ""
    if (args.input is not None):
        files = utilities.resolve_file_list(args.input)
        data = preprocessor.process_files(files, delimiter=args.delimiter, columns=header, is_list=args.list)
        if (data is not None):
            indata.append(data)
    print("done")
    return indata


def save(args, indata):
    """
    Saves the last produced data to the given directory.
    """
    print("Saving... ", end="")
    if (len(indata) <= 0):
        print("No data to save.")
        return
    for t in config.DATA_TYPES.values():
        if (isinstance(indata[-1], t)):
            if ("save" in vars(type(indata[-1]))):
                indata[-1].save(dir=args.output, delimiter=args.delimiter)
            else:
                print("Saving for dtype {} is not supported.".format(type(indata[-1]).__name__))
            break
    else:
        print("Data is not of a supported dtype ({}).".format(type(indata[-1]).__name__))
        print("Make sure to add custom data types to config.DATA_TYPES.")
    print("done")
    return indata


def guess_dtype(path):
    """
    Determines the data type from the name of the given file.
    """
    if ("_fixations.csv" in path):
        return "fixations"
    elif ("_fixated_aois.csv" in path):
        return "aois"
    elif ("_landmarks.csv" in path):
        return "landmarks"
    else:
        return None


def load(args, indata):
    """
    Loads data of a given type from the specified files.
    """
    print("Loading ", end="")
    files = utilities.resolve_file_list(args.files)
    if (args.type == "AUTO"):
        type = guess_dtype(files[0])
        if (type is None):
            print("of file \'{}\'{} failed".format(files[0], f" and {len(files)-1} more" if len(files) > 1 else ""))
            print("Could not determine the type of data automatically. Please specify it manually.")
            return indata
        else:
            print(type + "... ", end="")
        args.type = type
    data = config.LOADING_FUNCTIONS[args.type](files, args.delimiter)
    type_idx = utilities.test_dtypes(indata, [args.type])
    if (type_idx):
        type_idx = type_idx[0]
        if (args.type == "aois"):
            indata[type_idx].append_fixated_aois(data)
        elif (args.type == "fixations"):
            indata[type_idx].append_prep(data)
        else:
            indata.append(data)
    else:
        indata.append(data)
    print("done")
    return indata


def create_landmarks(args, indata):
    """
    Creates landmarks for all specified videos in args.videos.
    """
    print("Generating landmarks... ")
    video_list = []
    if (args.ignore_filter):
        video_list = utilities.resolve_file_list(args.videos)
    else:
        prep_idx = None
        for i, d in enumerate(indata):
            if (isinstance(d, list) and isinstance(d[0], result_classes.Landmarks)):
                prep_idx = i
                break

        if (prep_idx is not None):
            tmp_list = utilities.resolve_file_list(args.videos)
            for video in tmp_list:
                if (os.path.splitext(os.path.split(video)[1])[0] in indata[prep_idx].videos):
                    video_list.append(video)
        else:
            video_list = utilities.resolve_file_list(args.videos)
    observer = landmark_utils.generate_all_landmarks(video_list, config.AOI_MANAGERS["GENERATOR"][args.generator])
    observer.join()
    landmarks = None
    if (len(observer.results) <= 0):
        print("Generation of landmarks failed.")
        return indata
    landmarks = result_classes.Landmarks([observer.results[0][0]], [observer.results[0][1]])
    for res in observer.results[1:]:
        landmarks.add_video(res[0], res[1])
    indata.append(landmarks)
    print("done")
    return indata


def get_aoi(args, indata):
    """
    For each video in the aoi data determines for each preprocessed file
    for each frame of the video determines which aoi was fixated by the participant.
    """
    print("Getting aois...")
    if (args.visualize is not None):
        vis = config.VISUALIZATIONS[args.visualize]
        arg_list = inspect.getargspec(aoi_mapper.visualize_aois).args
        cur_vis = {}
        for k in vis:
            if (k in arg_list):
                cur_vis[k] = vis[k]
        data = aoi_mapper.visualize_aois(indata, getter=args.getter, grouping=args.grouping, **cur_vis)
    else:
        data = aoi_mapper.get_all_aois(indata, getter=args.getter, grouping=args.grouping)
    indata.append(data)
    return indata


def levenshtein_func(args, indata):
    """
    Calulates the Levenshtein distance for a given base scanpath and a list of scanpaths.
    A FixatedAOIsData object needs to be in indata.
    Returns:
        ScanpathDistances
            A list of the Levenshtein distances for each of the pairs.
    """
    print("Levenshtein distance... ", end="")
    type_idx = utilities.test_dtypes(indata, ["aois"])
    if (not type_idx):
        print("Aois needed for analysis.")
        return
    distances = scanpath_analysis.get_fixations_edit_distances(args.video, args.participant, indata[type_idx[0]], start_frame=args.start_frame, end_frame=args.end_frame)
    indata.append(distances)
    print("\nMean: {}, std_error: {}".format(*distances.get_distribution()))
    print("done")
    return indata


def mannan_func(args, indata):
    """
    Calulates the Mannan distance for a given base scanpath and a list of scanpaths.
    A FixatedAOIsData object needs to be in indata.
    Returns:
        ScanpathDistances
            A list of the Mannan distances for each of the pairs.
    """
    print("Mannan distance... ")
    type_idx = utilities.test_dtypes(indata, ["fixations"])
    if (not type_idx):
        print("Fixation data is needed for analysis.")
        return indata
    fix_data = indata[type_idx[0]]
    distances = scanpath_analysis.get_fixations_mannan_distance(args.video, args.participant, fix_data, video_size=args.video_size, seed=args.seed, start_frame=args.start_frame, end_frame=args.end_frame)
    if (distances is None):
        raise ValueError("No data for {} and participant {}".format(args.video, args.participant))
    indata.append(distances)
    print("done")
    return indata


def aoi_count_func(args, indata):
    """
    Calculates the fixation count statistic.
    """
    print("Counting aoi fixations... ", end="")
    type_idx = utilities.test_dtypes(indata, ["aois"])
    if (not type_idx):
        print("AOI fixation data is needed for analysis.")
        return indata
    aoi_data = indata[type_idx[0]]
    count_data = etstatistics.fixations_per_AOI(aoi_data, start_frame=args.start_frame, end_frame=args.end_frame)
    if (args.average):
        count_data.average()
    indata.append(count_data)
    print("done")
    return indata


def dwelltime_func(args, indata):
    """
    Calculates the dwell time statistic.
    """
    print("Calculating dwell times... ", end="")
    type_idx = utilities.test_dtypes(indata, ["aois"])
    if (not type_idx):
        print("AOI fixation data is needed for analysis.")
        # TODO maybe exit instead of returning
        return indata
    aoi_data = indata[type_idx[0]]
    count_data = etstatistics.fixations_per_AOI(aoi_data, start_frame=args.start_frame, end_frame=args.end_frame)
    if (args.average):
        count_data.average()
    count_data = result_classes.Dwelltimes(count_data)
    indata.append(count_data)
    print("done")
    return indata


def aoi_saccades_func(args, indata):
    """
    Calculates the saccade count statistic.
    """
    print("Counting aoi saccades... ", end="")
    type_idx = utilities.test_dtypes(indata, ["aois"])
    if (not type_idx):
        print("AOI fixation data is needed for analysis.")
        return indata
    aoi_data = indata[type_idx[0]]
    count_data = etstatistics.saccades_between_AOIs(aoi_data, filter_equal=args.ignore_doubles, start_frame=args.start_frame, end_frame=args.end_frame)
    indata.append(count_data)
    print("done")
    return indata


def vector_based_func(args, indata):
    """
    Calculates the vector based similarity measure using the selected distance function.
    """
    print("Vector based similarity measure...")
    type_idx = utilities.test_dtypes(indata, ["fixations"])
    if (not type_idx):
        print("Fixation data is needed for analysis.")
        return indata
    fix_data = indata[type_idx[0]]
    fix_data.simplify()
    vid_name = os.path.splitext(os.path.split(args.video)[1])[0]
    # find data matching video and participant
    vid_data = fix_data.get_video_data(vid_name)
    base_data = next((a for a in vid_data if (a.participant == str(args.participant))), None)
    if (not base_data):
        raise ValueError("No data for {} and participant {}".format(args.video, args.participant))
    comp_data = list(filter(lambda x: x.participant != base_data.participant or x.video != base_data.video, fix_data.data_set))
    scores = []
    p_bar = tqdm(total=len(comp_data), position=0, leave=True)
    for data in comp_data:
        score = scanpath_analysis.get_vector_based_fixation_diff(base_data, data,
                                                                 diff_func=config.VECTOR_DIFF_FUNCTIONS[args.diff_func],
                                                                 start_frame=args.start_frame,
                                                                 end_frame=args.end_frame)
        scores.append(score)
        p_bar.update(1)
    distances = []
    for c, score in zip(comp_data, scores):
        distances.append((c.video, c.participant, c.info, score))
    indata.append(result_classes.ScanpathDistances(distances, base_data.video, base_data.participant, info=base_data.info, analysis="VECTOR"))
    print("done")
    return indata


def gen_heatmap_func(args, indata):
    """
    Generates a saliency map using the fixations data and landmarks in indata.
    """
    print("Generating heatmap... ")
    if (args.image is not None):
        type_idx = utilities.test_dtypes(indata, ["fixations", "landmarks"])
        if (not type_idx):
            print("Fixation data and landmarks are needed for heatmap generation.")
            return indata
        fix_data = indata[type_idx[0]]
        landmarks = indata[type_idx[1]]
    else:
        type_idx = utilities.test_dtypes(indata, ["fixations"])
        if (not type_idx):
            print("Fixation data and landmarks are needed for heatmap generation.")
            return indata
        fix_data = indata[type_idx[0]]
        landmarks = None
    img = None
    base_landmarks = None
    fixations = []

    if (args.image is not None):
        image_ext = os.path.splitext(args.image)[1][1:]
        # image is actually video. Get a specific frame instead.
        if (image_ext in config.VIDEO_FORMATS):
            img = utilities.get_frame_of_video(args.image, args.frame)
            if (img is None):
                return indata
            base_landmarks = landmark_utils.get_landmarks(img)
        else:
            img, base_landmarks = utilities.get_base_face(args.image)
        if (args.display_size[0] > 0 and args.display_size[1] > 0):
            im_size = (args.display_size[0], args.display_size[1])
            mapped_data = face_mapper.map_fixation_data(fix_data, landmarks, base_landmarks, True)
            fixations = mapped_data.get_raw_fixations()
            xoff = (im_size[0] - img.shape[1]) / 2
            yoff = (im_size[1] - img.shape[0]) / 2
            for i in range(len(fixations)):
                x = fixations[i][0] + xoff
                y = fixations[i][1] + yoff
                fixations[i] = (int(x), int(y), fixations[i][2])
        else:
            im_size = (img.shape[1], img.shape[0])
            mapped_data = face_mapper.map_fixation_data(fix_data, landmarks, base_landmarks, True)
            fixations = mapped_data.get_raw_fixations()
    else:
        print("Heatmap is generated without background image and landmarks. This is not adviced since fixations of multiple videos are not comparable this way.")
        if (args.display_size[0] > 0 and args.display_size[1] > 0):
            im_size = args.display_size
        else:
            im_size = [config.VIDEO_SIZE["width"], config.VIDEO_SIZE["height"]]
        if (args.only_landmarks is True):
            print("The flag '--only-landmarks' can only be used in conjunction with '--image'.")
            print("failed")
            return indata
        for data in fix_data.data_set:
            fixations += data.fixations
        fixations = list(map(lambda x: (x[0], x[1], x[3]-x[2]), fixations))

    if ("standard_deviation" not in vars(args)):
        std = None
    else:
        std = args.standard_deviation

    if (base_landmarks is not None and args.only_landmarks):
        background = None
    else:
        background = img

    heatmap = heatmap_generator.generate_heatmap(fixations,
                                      im_size,
                                      background=background,
                                      alpha=args.alpha,
                                      gaussianwh=int(args.gaussianwh),
                                      gaussiansd=std)
    if (base_landmarks is not None and args.only_landmarks):
        color = config.LANDMARK_COLOR
        color = (color[0], color[1], color[2], 255)
        for lm in base_landmarks:
            p = lm
            if (args.display_size[0] > 0 and args.display_size[1] > 0):
                p = (int(p[0] + xoff), int(p[1] + yoff))
            cv2.circle(heatmap, p, config.LANDMARK_SIZE, color, -1)
    cv2.imwrite(args.save_file, heatmap)

    print("done")
    return indata


def init_visualization(args, indata):
    """
    Unified function that dependent on the input chooses the right funtions to produced the desired visualization.
    """
    print("Starting visualization...")
    if (not os.path.exists(args.image) or not os.path.isfile(args.image)):
        print(f"The first parameter must be a valid path to file containing either an image or a video. (\'{args.image}\')")
        return indata
    input_is_video = (os.path.splitext(args.image)[1][1:] in config.VIDEO_FORMATS)
    if (input_is_video and not args.frame >= 0):
        args.video = True
    vis = config.VISUALIZATIONS[args.visualization]
    vis_types = ["draw_aois", "draw_landmarks", "draw_fixations", "draw_arrows"]
    vis_types = list(filter(lambda k: k in vis and vis[k], vis_types)) # visualizations to draw
    fixation_idx = None
    landmark_idx = None

    # test which visualizations are to be drawn
    if ("draw_aois" in vis_types):
        vis_types.append("draw_aoi")
        if (landmark_idx is None and args.video):
            data = utilities.test_dtypes(indata, ["landmarks"])
            if (data is False):
                print("Landmarks are needed to visualize aois.")
                return indata
            landmark_idx = data
    if ("draw_landmarks" in vis_types):
        vis_types.append("draw_landmarks")
        if (landmark_idx is None and args.video):
            data = utilities.test_dtypes(indata, ["landmarks"])
            if (data is False):
                print("Landmark data is needed to visualize landmarks.")
                return indata
            landmark_idx = data
    if ("draw_fixations" in vis_types):
        vis_types.append("draw_fixations")
        if (fixation_idx is None):
            data = utilities.test_dtypes(indata, ["fixations"])
            if (data is False):
                print("Fixation data is needed to visualize fixations.")
                return indata
            fixation_idx = data
        if (args.map_data and landmark_idx is None):
            data = utilities.test_dtypes(indata, ["landmarks"])
            if (data is False):
                print("Landmarks are needed to map fixations.")
                return indata
            landmark_idx = data
    if ("draw_arrows" in vis_types):
        vis_types.append("draw_arrows")
        if (fixation_idx is None):
            data = utilities.test_dtypes(indata, ["fixations"])
            if (data is False):
                print("Fixation data is needed to visualize scanpaths.")
                return indata
            fixation_idx = data
        if (args.map_data and landmark_idx is None):
            data = utilities.test_dtypes(indata, ["landmarks"])
            if (data is False):
                print("Landmarks are needed to map scanpaths.")
                return indata
            landmark_idx = data
    fixations = indata[fixation_idx[0]] if fixation_idx is not None else None
    landmarks = indata[landmark_idx[0]] if landmark_idx is not None else None

    file_name = os.path.splitext(os.path.split(args.image)[1])[0]
    if (args.output == ""):
        if (not os.path.exists("visualizations/")):
            os.mkdir("visualizations/")
        if (not args.video):
            args.output = f"visualizations/visualization_{file_name}_{args.visualization}.png"
        else:
            args.output = f"visualizations/visualization_{file_name}_{args.visualization}.avi"

    # test what the output of the visulization should be. A video of a still image a single image or an animated video
    if (not args.video):
        image = None
        if (input_is_video):
            image = utilities.get_frame_of_video(args.image, args.frame)
            if (image is None):
                return indata
        else:
            image = cv2.imread(args.image)
        if (image is None):
            print(f"Could not open the image \'{args.image}\'.")
        overlay = image.copy()
        visualizations.visualize_image(overlay, fixations, landmarks, vis, map_data=args.map_data, grouping=config.AOI_GROUPINGS[args.grouping], start_frame=args.start_frame, end_frame=args.end_frame)
        if (args.alpha is not None and args.alpha < 1):
            overlay = cv2.addWeighted(image, 1-args.alpha, overlay, args.alpha, 0)
        cv2.imwrite(args.output, overlay)
    else:
        init_visualization_video(args, fixations, landmarks)
    print("Done")
    return indata



def init_visualization_video(args, fixations, landmarks):
    """
    Creates a visualization for a video as background using the given arguments.
    Fixations and landmarks might be None depending on the visualization that is to be generated.
    """
    vis = config.VISUALIZATIONS[args.visualization]
    image, video = None, None # current frame and video capture
    base_landmarks = None # landmarks for each frame of the visualized video
    is_video = (os.path.splitext(args.image)[1][1:] in config.VIDEO_FORMATS)
    file_name = os.path.splitext(os.path.split(args.image)[1])[0]
    if (not is_video):
        # use image and animate the visualization
        image, base_landmarks = utilities.get_base_face(args.image)
        is_video = False
    elif (args.frame >= 0):
        vid = cv2.VideoCapture(args.image)
        if (not vid.set(cv2.CAP_PROP_POS_FRAMES, args.frame)):
            print(f"Can't get frame no. {args.frame} of the video \'{args.image}\'.")
            return
        success, image = vid.read()
        vid.release()
        if (not success):
            print(f"Can't get frame no. {args.frame} of the video \'{args.image}\'.")
            return
        base_landmarks = landmark_utils.get_landmarks(image)
    else:
        # use a video and animate the visualization
        video = cv2.VideoCapture(args.image)
        if (not video.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)):
            print(f"Can't get frame no. {args.start_frame} of the video \'{args.image}\'.")
            return
        if (landmarks is not None):
            landmark_set = landmarks.get_landmarks(file_name)
            if (not landmark_set):
                return
    if (image is None and video is None):
        print(f"Can't open the image at: \'{args.image}\'.")
        return

    # set end frame to end of video or end of landmarks
    if (args.end_frame is None):
        if (video is not None):
            args.end_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            if (landmarks is not None):
                args.end_frame = min(args.end_frame, len(landmark_set) - 1)
        else:
            args.end_frame = max([f_data.fixations[-1].end for f_data in fixations.data_set])
    else:
        if (video is not None):
            args.end_frame = min(args.end_frame, int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1)
    if (video is not None):
        fps = video.get(cv2.CAP_PROP_FPS)
        resolution = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    else:
        fps = config.VIDEO_DEFAULT_FPS
        resolution = tuple(image.shape[:2])
        resolution = (int(resolution[1]), int(resolution[0]))

    output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"DIVX"), fps, resolution)
    p_bar = tqdm(total=args.end_frame - args.start_frame, position=0, leave=True)
    # for each frame index
    for f_idx in range(args.start_frame, args.end_frame+1):
        if (video is not None):
            success, image = video.read()
            if (not success):
                image = np.zeros((resolution[1], resolution[0], 3), np.uint8)
                video.set(cv2.CAP_PROP_POS_FRAMES, f_idx+1)
            if (landmarks is not None):
                base_landmarks = landmark_set[f_idx]
        overlay = image.copy()
        visualizations.visualize_image(overlay, fixations, landmarks, vis, image_landmarks=base_landmarks, map_data=args.map_data, grouping=config.AOI_GROUPINGS[args.grouping], start_frame=args.start_frame, end_frame=f_idx)
        if (args.alpha < 1):
            overlay = cv2.addWeighted(image, 1-args.alpha, overlay, args.alpha, 0)
        output.write(overlay)
        p_bar.update(1)
    output.release()
    if (video is not None):
        video.release()




class MainParser():
    """
    The parser used to parse sys.argv and split the arguments into chained
    sub commands.
    """

    def __init__(self):
        self.namespaces = []
        self.parser = argparse.ArgumentParser(description='Parameters required for processing.')
        self.subparsers = self.parser.add_subparsers(title="commands", dest="command", metavar="<command>")

        # preprocessing
        preproc = self.subparsers.add_parser("preprocess",
                                             aliases=["prep", "PREPROCESS"],
                                             help="Loads raw data and applies basic preprocessing.")
        preproc.add_argument("input", nargs="+", type=utilities.check_file)
        preproc.add_argument("-ca", "--calibrator",
                             choices=list(config.CALIBRATION_FUNCTIONS["CALIBRATE"].keys()) + ["NONE"],
                             default="NONE",
                             help="The function used to calculate correction data.")
        preproc.add_argument("-co", "--correct",
                             choices=list(config.CALIBRATION_FUNCTIONS["CORRECT"].keys()) + ["NONE"],
                             default=config.SELECTED_CALIBRATOR["CORRECT"],
                             help="The function used to correct the data.")
        preproc.add_argument("-ds", "--display",
                             nargs=2,
                             type=int,
                             default=[config.SCREEN_SIZE["width"], config.SCREEN_SIZE["height"]],
                             help="The size of the display the videos were viewed on.")
        preproc.add_argument("-d", "--delimiter",
                             type=utilities.check_char,
                             help="The delimiter used in the input file.")
        preproc.add_argument("-hd", "--header",
                             choices=list(config.ET_FILE_HEADERS.keys()),
                             default="default",
                             help="The structure of the header of the ET file")
        preproc.add_argument("-v", "--video-directory",
                             type=utilities.check_existing_dir,
                             default=".",
                             help="The directory which contains the videos.")
        preproc.add_argument("-l", "--list",
                             action="store_true",
                             default=False,
                             help="If set uses the given files as lists of files that match ET data with the corresponding videos and participants.")
        preproc.set_defaults(func=preprocess)
        preproc.set_defaults(intype="any")
        preproc.set_defaults(next=["save", "preprocessed/"])

        # save data
        save_parse = self.subparsers.add_parser("save",
                                                aliases=["SAVE"],
                                                help="Saves the data to the given directory or file.")
        save_parse.add_argument("output",
                                type=str,
                                default="output/",
                                help="Specifies where the result is saved.")
        save_parse.add_argument("-d", "--delimiter",
                                type=utilities.check_char,
                                default=",",
                                help="The delimiter to be used in the output file.")
        save_parse.set_defaults(func=save)
        save_parse.set_defaults(intype="any")

        # loading
        load_parse = self.subparsers.add_parser("load",
                                                aliases=["LOAD"],
                                                help="Loads files into data.")
        load_parse.add_argument("files", nargs="+", type=utilities.check_file, help="Files to load.")
        load_parse.add_argument("-t", "--type",
                                choices=list(config.LOADING_FUNCTIONS.keys()) + ["AUTO"],
                                default="AUTO",
                                help="The type of data that is loaded.")
        load_parse.add_argument("-d", "--delimiter",
                                type=utilities.check_char,
                                default=",",
                                help="The delimiter used in the input file.")
        load_parse.set_defaults(func=load)
        load_parse.set_defaults(intype="any")

        # landmark generation
        landmark_create_parse = self.subparsers.add_parser("create-landmarks",
                                                           aliases=["landmarks", "createlandmarks"],
                                                           help="Calculates landmarks from the given data for each frame of the videos.")
        landmark_create_parse.add_argument("videos", nargs="+",
                                           type=utilities.check_file,
                                           help="Video to calculate the landmarks for.")
        landmark_create_parse.add_argument("-g", "--generator",
                                           choices=list(config.AOI_MANAGERS["GENERATOR"].keys()),
                                           default=config.SELECTED_AOI_MANAGER["GENERATOR"],
                                           help="The function used to generate the landmarks.")
        landmark_create_parse.add_argument("-i", "--ignore-filter",
                                           action="store_true",
                                           default=False,
                                           help="If set ignores the fixation data and generates landmarks for all videos.")
        landmark_create_parse.set_defaults(func=create_landmarks)
        landmark_create_parse.set_defaults(intype="any")
        landmark_create_parse.set_defaults(next=["save", "landmarks/"])

        # aoi getter
        aoi_getter_parse = self.subparsers.add_parser("get-aoi",
                                                      aliases=["getaoi", "GETAOI", "get-aois", "getaois"],
                                                      help="Gets the corresponding AOI for each fixation.")
        aoi_getter_parse.add_argument("-g", "--getter",
                                      choices=list(config.AOI_MANAGERS["GETTER"].keys()),
                                      default=config.SELECTED_AOI_MANAGER["GETTER"],
                                      help="The function used to get the AOI for a given fixation.")
        aoi_getter_parse.add_argument("-p", "--grouping",
                                      choices=list(config.AOI_GROUPINGS.keys()),
                                      default=config.SELECTED_GROUPING,
                                      help="The grouping of the landmarks. Only used if supported by getter function.")
        aoi_getter_parse.add_argument("-v", "--visualize",
                                      choices=list(config.VISUALIZATIONS.keys()),
                                      default=None,
                                      help="Visualizes the AOIs for each frame using the selected visualization.")
        aoi_getter_parse.set_defaults(func=get_aoi)
        aoi_getter_parse.set_defaults(intype=["landmarks", "fixations"])
        aoi_getter_parse.set_defaults(next=["save", "aoi_fixations/"])

        # Levenshtein distance
        levenshtein_parse = self.subparsers.add_parser(
            "levenshtein",
            aliases=["levenshtein-distance"],
            help="Calculates the Levenshtein distance for a given base scanpath and the rest of the scanpaths. A list of fixated aois is used."
        )
        levenshtein_parse.add_argument("video", type=str,
                                       help="The base video to which all other data is compared.")
        levenshtein_parse.add_argument("participant", type=str,
                                       help="The participant to wich all other data is compared.")
        levenshtein_parse.add_argument("-s", "--start-frame", type=int,
                                       default=0,
                                       help="Frame of the video from which onwards to start the analysis.")
        levenshtein_parse.add_argument("-e", "--end-frame", type=int,
                                       default=None,
                                       help="Frame of the video till which the analysis is performed.")
        levenshtein_parse.set_defaults(func=levenshtein_func)
        levenshtein_parse.set_defaults(intype=["aois"])
        levenshtein_parse.set_defaults(next=["save", "analysis/"])

        # Mannan distance
        mannan_parse = self.subparsers.add_parser(
            "mannan",
            aliases=["mannan-distance"],
            help="Calculates the Mannan distance for a given base scanpath and the rest of the scanpaths."
        )
        mannan_parse.add_argument("video", type=str,
                                  help="The base video to which all other data is compared.")
        mannan_parse.add_argument("participant", type=str,
                                  help="The participant to wich all other data is compared.")
        mannan_parse.add_argument("-v", "--video-size",
                                  nargs=2,
                                  type=int,
                                  default=[1920, 1080],
                                  help="The size of the video needed for normalization.")
        mannan_parse.add_argument("-s", "--seed",
                                  type=int,
                                  default=2,
                                  help="The seed used to generate the random scanpath to compare with.")
        mannan_parse.add_argument("-b", "--start-frame", type=int,
                                  default=0,
                                  help="Frame of the video from which onwards to start the analysis.")
        mannan_parse.add_argument("-e", "--end-frame", type=int,
                                  default=None,
                                  help="Frame of the video till which the analysis is performed.")
        mannan_parse.set_defaults(func=mannan_func)
        mannan_parse.set_defaults(intype=["fixations"])
        mannan_parse.set_defaults(next=["save", "analysis/"])

        # AOI fixation count
        aoi_fix_count_parse = self.subparsers.add_parser(
            "count-aoi-fixations",
            aliases=["fixationcount", "fixcount", "aoifixcount"],
            help="Counts the fixations for each aoi and saves the data to a FixationCounts object."
        )
        aoi_fix_count_parse.add_argument("-s", "--start-frame", type=int,
                                         default=0,
                                         help="Frame of the video from which onwards to start the analysis.")
        aoi_fix_count_parse.add_argument("-e", "--end-frame", type=int,
                                         default=None,
                                         help="Frame of the video till which the analysis is performed.")
        aoi_fix_count_parse.add_argument("-avg", "--average",
                                         action="store_true",
                                         default=False,
                                         help="If set appends the average of the counts to the file.")
        aoi_fix_count_parse.set_defaults(func=aoi_count_func)
        aoi_fix_count_parse.set_defaults(intype=["aois"])
        aoi_fix_count_parse.set_defaults(next=["save", "statistics/"])

        # AOI dwelltime calculation
        aoi_dwelltime_parse = self.subparsers.add_parser(
            "calculate-dwelltimes",
            aliases=["dwelltimes", "dwelltime"],
            help="Calculates the total dwell time for each aoi and saves the data to a Dwelltimes object."
        )
        aoi_dwelltime_parse.add_argument("-s", "--start-frame", type=int,
                                         default=0,
                                         help="Frame of the video from which onwards to start the analysis.")
        aoi_dwelltime_parse.add_argument("-e", "--end-frame", type=int,
                                         default=None,
                                         help="Frame of the video till which the analysis is performed.")
        aoi_dwelltime_parse.add_argument("-avg", "--average",
                                         action="store_true",
                                         default=False,
                                         help="If set appends the average of the dwell times to the file.")
        aoi_dwelltime_parse.set_defaults(func=dwelltime_func)
        aoi_dwelltime_parse.set_defaults(intype=["aois"])
        aoi_dwelltime_parse.set_defaults(next=["save", "statistics/"])

        # saccades between AOIs
        aoi_saccades_parse = self.subparsers.add_parser(
            "count-aoi-saccades",
            aliases=["saccadecount", "aoisaccades", "aoisaccadecount"],
            help="Counts the saccades between aois and saves the data to a AOISaccadesData object."
        )
        aoi_saccades_parse.add_argument("-i", "--ignore-doubles",
                                        action="store_true",
                                        default=False,
                                        help="If set ignores saccades that have the same starting and end aoi.")
        aoi_saccades_parse.add_argument("-s", "--start-frame", type=int,
                                        default=0,
                                        help="Frame of the video from which onwards to start the analysis.")
        aoi_saccades_parse.add_argument("-e", "--end-frame", type=int,
                                        default=None,
                                        help="Frame of the video till which the analysis is performed.")
        aoi_saccades_parse.set_defaults(func=aoi_saccades_func)
        aoi_saccades_parse.set_defaults(intype=["aois"])
        aoi_saccades_parse.set_defaults(next=["save", "statistics/"])

        # vector based analysis
        vector_based_parse = self.subparsers.add_parser(
            "vector-based-analysis",
            aliases=["vectoranalysis", "vectorbased"],
            help="Analyses the scanpath by comparing the fixation vectors."
        )
        vector_based_parse.add_argument("video", type=str,
                                        help="The base video to which all other data is compared.")
        vector_based_parse.add_argument("participant", type=str,
                                        help="The participant to which all other data is compared.")
        vector_based_parse.add_argument("-diff-func", "-f",
                                        choices=list(config.VECTOR_DIFF_FUNCTIONS.keys()),
                                        default="DURATION")
        vector_based_parse.add_argument("-s", "--start-frame", type=int,
                                        default=0,
                                        help="Frame of the video from which onwards to start the analysis.")
        vector_based_parse.add_argument("-e", "--end-frame", type=int,
                                        default=None,
                                        help="Frame of the video till which the analysis is performed.")
        vector_based_parse.set_defaults(func=vector_based_func)
        vector_based_parse.set_defaults(intype=["fixations"])
        vector_based_parse.set_defaults(next=["save", "analysis/"])

        # heatmaps
        gen_heatmap_parse = self.subparsers.add_parser(
            "generate-heatmap",
            aliases=["heatmap", "genheatmap"],
            help="Generates an averaged heatmap for the current fixation data"
        )
        gen_heatmap_parse.add_argument("save_file", metavar="save-file",
                                       help="Name of the file the generated heatmap is saved to.")
        gen_heatmap_parse.add_argument("-d", "--display-size",
                                       nargs=2,
                                       type=int,
                                       default=[-1, -1],
                                       help="Size of the display.")
        gen_heatmap_parse.add_argument("-i", "--image",
                                       type=utilities.check_file,
                                       default=None,
                                       help="An image over which the heatmap is to be over laid.")
        gen_heatmap_parse.add_argument("-f", "--frame",
                                       type=int,
                                       default=0,
                                       help="Number of the frame that is used for the heatmap. Only used if the input is a video.")
        gen_heatmap_parse.add_argument("-a", "--alpha",
                                       type=float,
                                       default="0.5",
                                       help="A float between 0 and 1 indicating the transparency of the heatmap.")
        gen_heatmap_parse.add_argument("-gw", "--gaussianwh",
                                       type=float,
                                       default=200,
                                       help="Size of the used gaussian blur.")
        gen_heatmap_parse.add_argument("-sd", "--standard-deviation",
                                       type=float,
                                       help="Standard deviation of the used gaussian blur.")
        gen_heatmap_parse.add_argument("-lm", "--only-landmarks",
                                       action="store_true",
                                       default=False,
                                       help="If set, a visulaization of the landmarks is used as a background image.")
        gen_heatmap_parse.set_defaults(func=gen_heatmap_func)
        gen_heatmap_parse.set_defaults(intype=["fixations"])
        gen_heatmap_parse.set_defaults(next=["end"])

        # visualize grouping and save it as an image
        visualize_parse = self.subparsers.add_parser(
            "visualize",
            aliases=["VISUALIZE"],
            help="Visualizes a given grouping using a given visualization for a given image or video."
        )
        visualize_parse.add_argument("visualization",
                                       choices=list(config.VISUALIZATIONS.keys()),
                                       help="The visualization that is used.")
        visualize_parse.add_argument("image",
                                       help="Name of the image or video that is used for the visualization.")
        visualize_parse.add_argument("-o", "--output",
                                       type=str,
                                       default="",
                                       help="Name of the output file. If none is given the file is named using the image name and the grouping.")
        visualize_parse.add_argument("-g", "--grouping",
                                       choices=list(config.AOI_GROUPINGS.keys()),
                                       default=config.SELECTED_GROUPING,
                                       help="The grouping which is used for the visualization.")
        visualize_parse.add_argument("-f", "--frame",
                                       type=int,
                                       default=-1,
                                       help="Number of the frame that is used for the visualization. Only used if the input is a video.")
        visualize_parse.add_argument("-s", "--start-frame",
                                       type=int,
                                       default=0,
                                       help="Frame of the video from which onwards to start the visualization.")
        visualize_parse.add_argument("-e", "--end-frame",
                                       type=int,
                                       default=None,
                                       help="Frame of the video till which the scanpath is visualized.")
        visualize_parse.add_argument("-a", "--alpha",
                                       type=float,
                                       default=1,
                                       help="If less than 1, aplha is used as the amount of transparency for the overlayed visualization.")
        visualize_parse.add_argument("-m", "--map-data",
                                        action="store_true",
                                        default=False,
                                        help="If set maps all fixations that are on the face to the face in the image.")
        visualize_parse.add_argument("-v", "--video",
                                       action="store_true",
                                       default=False,
                                       help="If set, the visualization is rendered as a video.")
        visualize_parse.set_defaults(func=init_visualization)
        visualize_parse.set_defaults(intype="any")
        visualize_parse.set_defaults(next=["end"])

        # clear data
        clear_parse = self.subparsers.add_parser("clear",
                                                 aliases=["CLEAR", "delete", "DELETE"],
                                                 help="Clears the current data.")
        clear_parse.set_defaults(intype="any")
        clear_parse.set_defaults(func=lambda *args: [])

        # end of command chain
        end_parse = self.subparsers.add_parser("end",
                                               aliases=["END", "exit", "EXIT"],
                                               help="Exits the programm. Usefull if it is not intended to save the last produced data.")
        end_parse.set_defaults(intype="any")
        end_parse.set_defaults(func=lambda *args: None)

    def parse(self):
        """
        Splits the arguments in argv by subcommands and parses each subcommand
        and the main command.
        Based on:
        https://stackoverflow.com/questions/10448200/how-to-parse-multiple-nested-sub-commands-using-python-argparse
        """
        if (len(self.namespaces) > 0):
            return self.namespaces
        # split args by different sub commands
        split_args = [[]]
        for arg in sys.argv[1:]:
            if (arg in self.subparsers.choices):
                split_args.append([arg])
            else:
                split_args[-1].append(arg)
        self.parser.parse_args(split_args[0])
        # parse args and save results into namespaces
        self.namespaces = []
        for argv in split_args[1:]:
            self.namespaces.append(self.parser.parse_args(argv))
        if (len(self.namespaces) == 0):
            self.parser.print_help()
            print("\nAt least one command needs to be provided.\nExiting...")
            sys.exit(1)
        # add possible default action at the end
        if (hasattr(self.namespaces[-1], "next") and self.namespaces[-1].next is not None):
            self.namespaces.append(self.parser.parse_args(self.namespaces[-1].next))
        return self.namespaces

    def visualize(self):
        """
        Prints as visualization of the different processing steps that are
        chained together.
        """
        tasks = ""
        if (self.namespaces is None):
            self.parse()
        task_count = len(self.namespaces)
        for i, ns in enumerate(self.namespaces):
            tasks += ns.command
            if (i < task_count - 1):
                tasks += " -> "
        print(tasks)
