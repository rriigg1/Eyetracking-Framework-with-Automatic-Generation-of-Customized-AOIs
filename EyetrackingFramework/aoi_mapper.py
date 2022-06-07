import cv2
from tqdm import tqdm
import inspect

# framework imports
import config
import result_classes
import utilities
import visualizations


def get_all_aois(indata, getter=config.SELECTED_AOI_MANAGER["GETTER"], grouping=config.SELECTED_GROUPING):
    """
    For a given dict containing preprocessed data of fixations and aois
    calculates for each fixation the corresponding aoi
    by using the given getter function.

    Args:
        indata : list
            A list containing a FixationDataset object and a Landmarks object.
        getter : str
            Name of the fuction used to calculate to which aoi a given fixation corresponds.
        grouping : str
            Name of the selected grouping which represents the used aois.

    Returns:
        dict
            A dict containig data about which aoi is fixated for each frame.

    """
    AOI_GETTER = config.AOI_MANAGERS["GETTER"][getter]
    grouping_name = grouping
    grouping = config.AOI_GROUPINGS[grouping]
    # find needed data
    data_indices = utilities.test_dtypes(indata, ["fixations", "landmarks"])
    if (data_indices):
        prep_idx, landmark_idx = data_indices
    else:
        raise ValueError("Not all data types needed for processing were found. (fixations, landmarks)")

    landmark_set = indata[landmark_idx]
    fixation_set = indata[prep_idx]

    return_data = None
    for video_idx, video in enumerate(landmark_set.videos):
        fix_data = fixation_set.get_video_data(video)
        if (len(fix_data) <= 0):
            print(f"No fixation data for video \'{video}\'")
            continue
        output_buffer = [[] for k in range(len(fix_data))]
        aoi_buffer = []
        progress = tqdm(total=len(landmark_set.landmarks[video_idx]), leave=True, position=0)
        last_idx = [0] * len(fix_data)
        for frame_count, landmarks in enumerate(landmark_set.landmarks[video_idx]):
            for k, data in enumerate(fix_data):
                fixation = None
                if (frame_count <= data.fixations[-1].end):
                    # get the first fixations that took place during the current frame of the video
                    row, fixation = next(((r, row) for r, row in enumerate(data.fixations[last_idx[k]:]) if row.start <= frame_count <= row.end), (0, None))
                    last_idx[k] += row
                if (fixation is not None and fixation.x >= 0):
                    if (inspect.getargspec(AOI_GETTER).keywords is not None):
                        output_buffer[k].append((frame_count, AOI_GETTER(landmarks, (fixation.x, fixation.y), **{"grouping": grouping})))
                    else:
                        output_buffer[k].append((frame_count, AOI_GETTER(landmarks, (fixation.x, fixation.y))))
            progress.update(1)
        if (getter != "CLOSEST"):
            name_data = list(grouping.keys())
        else:
            name_data = []
        for k, data in enumerate(fix_data):
            aoi_buffer.append(result_classes.FixatedAOIs(output_buffer[k], indata[landmark_idx].video_files[video_idx],
                                                         participant=data.participant,
                                                         source_name=data.filename,
                                                         aoi_names=name_data,
                                                         info=data.info,
                                                         grouping=grouping_name))
        if (return_data is None):
            return_data = result_classes.FixatedAOIsData(aoi_buffer)
        else:
            return_data.append_aois(aoi_buffer)
        progress.close()
    if (return_data is None):
        return_data = result_classes.FixatedAOIsData([])
        print("No data produced.")
    return return_data


def visualize_aois(indata,
                   getter=config.SELECTED_AOI_MANAGER["GETTER"],
                   grouping=config.SELECTED_GROUPING,
                   random_colors=False,
                   label_aois=True,
                   draw_fixations=True,
                   label_fixations=True,
                   fill_polygons=False,
                   alpha=0.6):
    """
    For a given dict containing preprocessed data of fixations and aois
    calculates for each fixation the corresponding aoi
    by using the given getter function.

    Args:
        indata : list
            A list containing a FixationsDataset object and a Landmarks object.
        getter : str
            Name of the fuction used to calculate to which aoi a given fixation corresponds.
        grouping : str
            Name of the used grouping.
        random_colors : int
            Whether to use random colors for the different aois.
            If not false it is used as the seed.
        label_aois : bool
            Whether to label the aois.
        label_fixations : bool
            Whether to label the fixations.
        fill_polygons : bool
            Whether to fill the polygons or to only draw their outlines.
        alpha : float
            A value between 0 and 1 which determines the transparency of the drawn aois.
            Zero means fully transparent and one is equivalent to fully opaque.

    Returns:
        dict
            A dict containig data about which aoi is fixated for each frame.

    """
    AOI_GETTER = config.AOI_MANAGERS["GETTER"][getter]
    grouping_name = grouping
    grouping = config.AOI_GROUPINGS[grouping]
    # find needed data
    data_indices = utilities.test_dtypes(indata, ["fixations", "landmarks"])
    if (data_indices):
        prep_idx, landmark_idx = data_indices
    else:
        print(indata)
        raise ValueError("Not all data types needed for processing were found. (fixations, landmarks)")

    landmark_set = indata[landmark_idx]
    fixation_set = indata[prep_idx]

    stop_processing = False
    return_data = None
    for video_idx, video in enumerate(landmark_set.videos):
        fix_data = fixation_set.get_video_data(video)
        if (len(fix_data) <= 0):
            print(f"No fixation data for video \'{video}\'")
            continue
        video_capture = cv2.VideoCapture(landmark_set.video_files[video_idx])
        success, image = video_capture.read()
        if (not success):
            print("Unable to open video: \'{}\'".format(landmark_set.video_files[video_idx]))
        frame_count = 0
        output_buffer = [[] for k in range(len(fix_data))]
        aoi_buffer = []
        progress = tqdm(total=len(landmark_set.landmarks[video_idx]), leave=True, position=0)
        last_idx = [0] * len(fix_data)
        while success:
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                stop_processing = True
                break
            landmarks = landmark_set.landmarks[video_idx][frame_count]
            # visualize
            overlay = image.copy()
            visualizations.draw_grouping(overlay, landmarks, grouping, random_colors, label_aois, fill_polygons)
            image = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
            # for each fixation calculate closest aoi
            for k, data in enumerate(fix_data):
                fixation = None
                if (frame_count <= data.fixations[-1][3]):
                    # get first fixations that took place during the current frame of the video
                    row, fixation = next(((r, row) for r, row in enumerate(data.fixations[last_idx[k]:]) if row.start <= frame_count <= row.end), (0, None))
                    last_idx[k] += row
                if fixation is not None and fixation.x >= 0:
                    if (draw_fixations):
                        cv2.circle(image, (fixation.x, fixation.y), 5, (255, 0, 0))
                    if (inspect.getargspec(AOI_GETTER).keywords is not None):
                        # call the aoi getter functions with the specified args
                        output_buffer[k].append((frame_count, AOI_GETTER(landmarks, (fixation.x, fixation.y), **{"grouping": grouping})))
                    else:
                        # call the aoi getter function without any args
                        output_buffer[k].append((frame_count, AOI_GETTER(landmarks, (fixation.x, fixation.y))))
                    if (draw_fixations and label_fixations):
                        cv2.putText(image, str(output_buffer[k][-1]), (fixation.x, fixation.y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 255))
            cv2.imshow(video, image)
            # read next frame
            success, image = video_capture.read()
            frame_count += 1
            progress.update(1)
        video_capture.release()
        cv2.destroyAllWindows()
        progress.close()
        if (stop_processing):
            break
        if (getter != "CLOSEST"):
            name_data = list(grouping.keys())
        else:
            name_data = []
        for k, data in enumerate(fix_data):
            aoi_buffer.append(result_classes.FixatedAOIs(output_buffer[k], indata[landmark_idx].video_files[video_idx],
                                                         participant=data.participant,
                                                         source_name=data.filename,
                                                         aoi_names=name_data,
                                                         info=data.info,
                                                         grouping=grouping_name))
        if (return_data is None):
            return_data = result_classes.FixatedAOIsData(aoi_buffer)
        else:
            return_data.append_aois(aoi_buffer)
    return return_data
