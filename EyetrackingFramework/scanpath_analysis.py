import editdistance as ed
import numpy as np
from collections import deque
import types
import result_classes
import config
import os
import math


def get_fixations_edit_distances(video, participant, aoi_data, start_frame=0, end_frame=None):
    """
    For a given FixatedAOIsData object and a base video and participant
    calculates the Levenshtein distances between each scanpath and the given
    base path.
    """
    vid_name = os.path.splitext(os.path.split(video)[1])[0]
    vid_data = aoi_data.get_video_data(vid_name)
    base_idx, base_data = next(((i, a) for (i, a) in enumerate(vid_data) if (a.participant == str(participant))), (None, None))
    if (not base_data):
        print("No data found for video: {} and participant: {}".format(video, participant))
        return
    comp_data = aoi_data.aoi_data[:]
    if (start_frame > 0):
        i = 0
        while (i < len(base_data.aois) and base_data.aois[i][0] < start_frame):
            i += 1
        base_data.aois = base_data.aois[i:]
        for data in comp_data:
            i = 0
            while (i < len(data.aois) and data.aois[i][0] < start_frame):
                i += 1
            data.aois = data.aois[i:]
    if (end_frame is not None):
        i = len(base_data.aois) - 1
        while (i >= 0 and base_data.aois[i][0] > end_frame):
            i -= 1
        base_data.aois = base_data.aois[:i+1]
        for data in comp_data:
            i = len(data.aois) - 1
            while (i >= 0 and data.aois[i][0] > end_frame):
                i -= 1
            data.aois = data.aois[:i+1]
    base_data = (base_data.video, base_data.participant, base_data.info, base_data.grouping, list(map(lambda r: r[1], base_data.aois)))
    new_comp_data = []
    for i, data in enumerate(comp_data):
        if (base_data[0] == data.video and base_data[1] == data.participant and base_data[3] == data.grouping):
            continue
        data = [data.video, data.participant, data.info, data.grouping, list(map(lambda r: r[1], data.aois))]
        new_comp_data.append(data)
    scores = get_edit_distances(base_data[-1], [d[-1] for d in new_comp_data])
    assert (len(scores) == len(new_comp_data))
    for i in range(len(new_comp_data)):
        new_comp_data[i][-1] = scores[i]
    result = result_classes.AOIScanpathDistances(new_comp_data,
                                                 base_data[0],
                                                 base_data[1],
                                                 info=base_data[2],
                                                 grouping=base_data[3],
                                                 analysis="LEVENSHTEIN")
    return result


def get_edit_distance(aoi_fix1, aoi_fix2, start_frame=0, end_frame=None):
    """
    For two given lists of aoi fixations calulates the Levenshtein distance
    and returns a normalized score.
    Args:
        aoi_fix1 : list
            A list of fixated aois.
        aoi_fix2 : list
            Second list of fixated aois.
    Returns:
        float
            Returns a score between 1 and 0 where 1 means that the scanpath
            are identical.
    """
    if (end_frame is None and start_frame == 0):
        ad = [r[0] for r in aoi_fix1]
        bd = [r[0] for r in aoi_fix2]
    elif(end_frame is None):
        ad = [r[0] for r in aoi_fix1[start_frame:]]
        bd = [r[0] for r in aoi_fix2[start_frame:]]
    else:
        ad = [r[0] for r in aoi_fix1[start_frame:end_frame]]
        bd = [r[0] for r in aoi_fix2[start_frame:end_frame]]

    mlen = max(len(ad), len(bd))
    score = 1 - (ed.eval(ad, bd) / mlen)
    return score


def get_edit_distances(aoi_fix, fix_list):
    """
    For a given list of fixated aois and a list of multiple scanpaths
    calulates the average Levenshtein distance and returns a normalized score.
    ### Args:
        aoi_fix : list
            A list of fixated aois.
        fix_list : list
            A list of lists of fixated aois.
    ### Returns:
        list
            Returns a list of distances
    """
    scores = []
    for i, s in enumerate(fix_list):
        mlen = max(len(aoi_fix), len(s))
        score = 1 - (ed.eval(aoi_fix, s) / mlen)
        scores.append(score)
    return scores


def get_fixations_mannan_distance(video, participant, fixation_data, video_size=[1920, 1080], seed=2, clamp=True, start_frame=0, end_frame=None):
    """
    Calculates multiples Mannan indices. The scanpath described by the given
    video and participant is compared to every other scanpath in fixation_data.
    ### Args:
        video : string
            Path to the video of scanpath to which all other scanpaths are compared.
        participant : string
            Name / ID of the participant of the scanpath to which all other
            scanpaths are compared.
        fixation_data : FixationsDataset
            Data containign all scanpaths for the comparison.
        video_size : lits / tuple
            Width and height of the screen which was used when capturing the
            fixation data.
        seed : int
            Seed used to seed the random number generator for calculating the
            Mannan Index.
        clamp : bool
            If set to true, the values of the Mannan Index are restricted to
            be between 0 and 100.
        start_frame : int
            First frame from which onwards each of the scanpath are compared.
        end_frame : int
            Last frame until which each of the scanpath are compared.
    ### Returns:
        ScanpathDistances
            Return as ScanpathDistances object containing information about the
            performaed comparisons and their respective calculated Mannan indices.
    """
    vid_name = os.path.splitext(os.path.split(video)[1])[0]
    vid_data = fixation_data.get_video_data(vid_name)
    base_data = next((a for a in vid_data if (a.participant == str(participant))), None)
    if (not base_data):
        print("No data found for video: {} and participant: {}".format(video, participant))
        return
    # remove base data from fixations data
    comp_data = list(filter(lambda x: x.participant != base_data.participant or x.video != base_data.video, fixation_data.data_set))
    # trim data before start_frame
    if (start_frame > 0 or end_frame is not None):
        base_scanpath = base_data[slice(start_frame, end_frame)]
        comp_scanpaths = []
        for data in comp_data:
            comp_scanpaths.append(data[slice(start_frame, end_frame)])
    else:
        base_scanpath = base_data[slice(start_frame, end_frame)]
        comp_scanpaths = []
        for data in comp_data:
            comp_scanpaths.append(data.fixations)
    # remove data that has no fixations during the analyzed interval
    filtered_data = []
    filtered_scanpaths = []
    for i in range(len(comp_data)):
        if (len(comp_scanpaths[i]) > 0):
            filtered_scanpaths.append([(r[0], r[1]) for r in comp_scanpaths[i]])
            filtered_data.append(comp_data[i])
    if (len(base_scanpath) <= 0 or len(filtered_data) <= 0):
        raise ValueError("No data in specified interval.")
    # remove time information from fixations
    base_scanpath = list(map(lambda r: (r[0], r[1]), base_data.fixations))
    comp_scanpaths = filtered_scanpaths
    comp_info = list(map(lambda a: [a.video, a.participant, a.info], filtered_data))
    scores = get_mannan_indices(base_scanpath, comp_scanpaths, video_size=video_size, seed=seed, clamp=clamp)
    assert (len(scores) == len(comp_info))
    distances = [comp_info[i] + [scores[i]] for i in range(len(scores))]
    result = result_classes.ScanpathDistances(distances,
                                              base_data.video,
                                              base_data.participant,
                                              info=base_data.info,
                                              analysis="MANNAN")
    return result


def mannan_distance(scanpath1, scanpath2, video_size):
    """
    Calculates the Mannan distance between two given scanpath.
    ### Args:
        scanpath1 : list
            List of coordinates as tuples of the first scanpath.
        scanpath2 : list
            List of coordinates as tuples of the second scanpath.
        video_size : tuple
            Size of the video the participants watched.
    ### Return:
        Returns the Mannan distance between two given scanpath.
    """
    d1 = lambda p: get_sqrd_min_dist(p, scanpath2)
    d2 = lambda p: get_sqrd_min_dist(p, scanpath1)
    sum1 = sum(map(d1, scanpath1)) / len(scanpath1)
    sum2 = sum(map(d2, scanpath2)) / len(scanpath2)
    denom = 2 * (video_size[0]*video_size[0] + video_size[1]*video_size[1])
    return math.sqrt((sum1 + sum2) / denom)


def get_mannan_indices(scanpath, scan_list, video_size=[1920, 1080], seed=2, clamp=False):
    """
    Calculates the Mannan index for comparisons between the given scanpath and
    each of the scanpaths contained in scan_list.
    ### Args:
        scanpath : list
            List of fixations representing the scanpath to which all other
            scanpaths are compared.
        scan_list : list
            List of scanpaths to compared to.
        video_size : list / tuple
            Size of the video the participants watched.
        seed : int
            number used to seed the random number generator.
        clamp : bool
            If set to true, the values of the Mannan Index are restricted to
            be between 0 and 100.
    ### Returns:
        list
            Returns a list of indices indicating how closely each of the scanpath match.
    """
    np.random.seed(seed)
    idxs = []
    ran_1 = np.random.random((len(scanpath), 2)) * video_size
    # random scanpath is generated once with the maximum length in scan_list
    # in ordner to minimize randomness of the results
    maxlen = max(scan_list, key=lambda p: len(p))
    ran_2 = np.random.random((len(maxlen), 2)) * video_size
    for path in scan_list:
        cur_ran = ran_2[:len(path)]
        dr = mannan_distance(ran_1, cur_ran, video_size)
        d = mannan_distance(scanpath, path, video_size)
        idx = (1 - d / dr) * 100
        if (clamp):
            if idx < 0:
                idx = 0
            if idx > 100:
                idx = 100
        idxs.append(idx)
    return idxs


def get_mannan_index(scanpath1, scanpath2, video_size):
    """
    Calculates an index comparing the Mannan distance of two scanpaths with
    the distance of two random scanpaths.
    ### Args:
        scanpath1 : list
            List of fixations in the first scanpath.
        scanpath3 : list
            List of fixations in the second scanpath.
        video_size : tuple
            Dimensions of the video (width, height).
    ### Returns:
        float
            A number between 0 meaning random and 1 meaning the scanpaths are identical.
    """
    np.random.seed(0)
    ran_1 = np.random.random((len(scanpath1), 2)) * video_size
    ran_2 = np.random.random((len(scanpath2), 2)) * video_size
    dr = mannan_distance(ran_1, ran_2, video_size)
    d = mannan_distance(scanpath1, scanpath2, video_size)
    print("d: {}".format(d))
    print("dr: {}".format(dr))
    idx = (1 - d / dr) * 100
    if idx < 0:
        idx = 0
    if idx > 100:
        idx = 100
    return idx


def get_sqrd_min_dist(point, scanpath):
    """
    For a given point and a list of points returns the smallest squared distance
    between the first point and any of the points in the list.
    ### Args:
        point : tuple
            Tuples containing "x" and "y" coordinates.
        scanpath : list
            List of points given as tuples.
    ### Returns:
        float
            Returns the smallest squared distance between point and any
            of the points in scanpath.
    """
    x, y = point[0], point[1]
    dx, dy = scanpath[0][0] - x, scanpath[0][1] - y
    min_dist = dx * dx + dy * dy
    for p in scanpath:
        dx, dy = p[0] - x, p[1] - y
        dist = dx * dx + dy * dy
        if (dist < min_dist):
            min_dist = dist
    return min_dist


def get_fixation_pos_diff(scanpath1, scanpath2, fixation1, fixation2):
    """
    Returns the distance between the two fixations.
    ### Args:
        scanpath1 : list
            List of fixations in scanpath 1.
        scanpath2 : list
            List of fixations in scanpath 2.
        fixation1 : int
            Index of fixation 1.
        fixation2 : int
            Index of fixation 2.
    ### Returns:
        float
            Returns the distance between the two fixations. The distance is
            normalized by the screen diagonal. Two fixations in opposite corners
            of the screen would return 1.0.
    """
    fixation1 = scanpath1[fixation1]
    fixation2 = scanpath2[fixation2]
    d1 = fixation2[0] - fixation1[0]
    d2 = fixation2[1] - fixation1[1]
    v = list(config.VIDEO_SIZE.values())
    return math.hypot(d1, d2)/math.hypot(v[0], v[1])


def get_fixation_duration_diff(scanpath1, scanpath2, fixation1, fixation2):
    """
    Returns the absolute difference in fixation duration.
    ### Args:
        scanpath1 : list
            List of fixations in scanpath 1.
        scanpath2 : list
            List of fixations in scanpath 2.
        fixation1 : int
            Index of fixation 1.
        fixation2 : int
            Index of fixation 2.
    ### Returns:
        int
            Returns the difference in fixation duration in number of frames
            between the two fixations.
    """
    fixation1 = scanpath1[fixation1]
    fixation2 = scanpath2[fixation2]
    duration1 = fixation1[3] - fixation1[2] + 1
    duration2 = fixation2[3] - fixation2[2] + 1
    return abs(duration2 - duration1) / max(duration1, duration2)


def get_saccade_length_diff(scanpath1, scanpath2, saccade1, saccade2):
    """
    Returns the difference in saccade length between two saccades.
    ### Args:
        scanpath1 : list
            List of fixations in scanpath 1.
        scanpath2 : list
            List of fixations in scanpath 2.
        saccade1 : int
            Index of saccade 1.
        saccade2 : int
            Index of saccade 2.
    ### Returns:
        int
            Returns the difference in the length of saccades.
    """
    if ((saccade1 >= len(scanpath1) - 1) and (saccade2 >= len(scanpath2) - 1)):
        # scanpath 1 and 2 have no saccades starting from the last fixation
        return 0
    elif (saccade1 >= len(scanpath1) - 1):
        # scanpath 1 has no saccade starting from the last fixation
        p1 = scanpath2[saccade2]
        p2 = scanpath2[saccade2 + 1]
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    elif (saccade2 >= len(scanpath2) - 1):
        # scanpath 2 has no saccade starting from the last fixation
        p1 = scanpath1[saccade1]
        p2 = scanpath1[saccade1 + 1]
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    else:
        p1_1 = scanpath1[saccade1]
        p1_2 = scanpath1[saccade1 + 1]
        p2_1 = scanpath2[saccade2]
        p2_2 = scanpath2[saccade2 + 1]
        l1 = math.hypot(p1_2[0] - p1_1[0], p1_2[1] - p1_1[1])
        l2 = math.hypot(p2_2[0] - p2_1[0], p2_2[1] - p2_1[1])
        return (1-(l2 / l1)) if (l1 > l2) else (1 - (l1 / l2))


def align_scanpath_fixations(fixation_data1, fixation_data2, diff_func=get_fixation_pos_diff, start_frame=0, end_frame=0):
    """
    Given two FixationData objects, returns a series of comparison tuples that
    minimizes the difference of the compared fixation but still compares each
    fixation of one scanpath with at least on fixation of the second scanpath.
    The method creates a matrix containing possible comparisons between fixations
    and to which a cost is assigned. In this matrix a shortest path is than
    calculated which corresponds to a comparison whith minimal difference.
    ### Args:
        fixation_data1 : FixationData
            Fixation data representing the first of the two scanpath.
        fixation_data2 : FixationData
            Fixation data representing the second of the two scanpath.
        diff_func : function
            A function taking two fixations as input which returns a
            difference/distance between these fixations.
        start_frame : int
            First frame from which onwards to perform the comparison.
        end_frame : int
            Last frame until which the comparison is performed.
    ### Returns:
        list
            Returns a list containing tuples of fixations to compare to achieve
            minimal difference.
    """
    assert isinstance(fixation_data1, result_classes.FixationData)
    assert isinstance(fixation_data2, result_classes.FixationData)
    assert isinstance(diff_func, types.FunctionType)
    scanpath1 = fixation_data1[slice(start_frame, end_frame)]
    scanpath2 = fixation_data2[slice(start_frame, end_frame)]
    dist = diff_func(scanpath1, scanpath2, 0, 0)
    # stores tuples (distance, x, y)
    # used to backtrack the shortest path
    comp_mat = {(0, 0): [dist, None, None]}
    current = deque()
    current.append((0, 0))
    end_reached = False
    # basicly Dijkstra on a comparison matrix
    while current:
        selected = current.popleft()
        x, y = selected
        if (end_reached and comp_mat[selected][0] >= comp_mat[(len(scanpath1) - 1, len(scanpath2) - 1)][0]):
            continue
        if (x + 1 < len(scanpath1)):
            dist = diff_func(scanpath1, scanpath2, x + 1, y)
            # coparison not done yet or achieved more expensive
            if ((x + 1, y) not in comp_mat
                    or comp_mat[selected][0] + dist < comp_mat[(x + 1, y)][0]):
                comp_mat[(x + 1, y)] = [comp_mat[selected][0] + dist, selected[0], selected[1]]
                current.append((x + 1, y))
            if (y + 1 < len(scanpath2)):
                dist = diff_func(scanpath1, scanpath2, x + 1, y + 1)
                # coparison not done yet or achieved more expensive
                if ((x + 1, y + 1) not in comp_mat
                        or comp_mat[selected][0] + dist < comp_mat[(x + 1, y + 1)][0]):
                    comp_mat[(x + 1, y + 1)] = [comp_mat[selected][0] + dist, selected[0], selected[1]]
                    current.append((x + 1, y + 1))
        if (y + 1 < len(scanpath2)):
            dist = diff_func(scanpath1, scanpath2, x, y + 1)
            # coparison not done yet or achieved more expensive
            if ((x, y + 1) not in comp_mat
                    or comp_mat[selected][0] + dist < comp_mat[(x, y + 1)][0]):
                comp_mat[(x, y + 1)] = [comp_mat[selected][0] + dist, selected[0], selected[1]]
                current.append((x, y + 1))
        if (not end_reached and (len(scanpath1)-1, len(scanpath2) -1) in comp_mat):
            end_reached = True
    # a shortest path to the end should be calculated when the algorithm ends
    assert ((len(scanpath1) - 1, len(scanpath2) - 1) in comp_mat)
    current = (len(scanpath1) - 1, len(scanpath2) - 1)
    comparisons = [current]
    while (current != (0, 0)):
        d, x, y = tuple(comp_mat[current])
        current = (x, y)
        comparisons.append(current)
    return comparisons


def get_vector_based_fixation_diff(fixation_data1, fixation_data2, alignment=None, diff_func=get_fixation_pos_diff, start_frame=0, end_frame=None):
    """
    Given two FixationData objects and an alignment and a diff_function
    calculates the difference of the two scanpath by using the comparisons
    given in alignment. The difference for individual fixations is calculated
    using the diff_func.
    ### Args:
        fixation_data1 : FixationData
            Fixation data representing the first of the two scanpath.
        fixation_data2 : FixationData
            Fixation data representing the second of the two scanpath.
        alignment : list
            A list of tuples containing two fixations to compare.
            If None is given the diff_func is used to calculate an alignment
            that results in the minimal difference between the two scanpath.
        diff_func : function
            A function taking two fixations as input which returns a
            difference/distance between these fixations.
        start_frame : int
            First frame from which onwards to perform the comparison.
        end_frame : int
            Last frame until which the comparison is performed.
    ### Returns:
        float
            Returns a difference of two given scanpaths. Higher values indicate
            more differences between the two scanpaths.
    """
    assert isinstance(fixation_data1, result_classes.FixationData)
    assert isinstance(fixation_data2, result_classes.FixationData)
    assert isinstance(diff_func, types.FunctionType)
    if (alignment is None):
        alignment = align_scanpath_fixations(fixation_data1, fixation_data2, diff_func=diff_func, start_frame=start_frame, end_frame=end_frame)
    scanpath1 = fixation_data1[slice(start_frame, end_frame)]
    scanpath2 = fixation_data2[slice(start_frame, end_frame)]
    difference = 0
    for comp in alignment:
        difference += diff_func(scanpath1, scanpath2, comp[0], comp[1])
    difference /= len(alignment)
    return difference
