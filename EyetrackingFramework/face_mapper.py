from scipy.spatial import Delaunay
import dlib
import cv2
import numpy as np

# framework imports
import config
import landmark_utils
import result_classes
import utilities
import error_logger

def get_barycentric(triangle, point):
    """
    For given triangle and cartesian coordinates returns the respective barycentric coordinates.
    Args:
        triangle : list
            A list containing the three vertices of a triangle.
        point : tuple
            A tuple x,y describing a two dimensional point.
    Returns
        tuple
            Returns a tuple u,v,w describing the barycentric coordinates or None if the triangle is coplanar.
    """
    a, b, c = tuple(triangle)
    p = point
    denom = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
    if (denom == 0):
        return None
    u = ((b[1]-c[1])*(p[0]-c[0])+(c[0]-b[0])*(p[1]-c[1])) / denom
    v = ((c[1]-a[1])*(p[0]-c[0])+(a[0]-c[0])*(p[1]-c[1])) / denom
    w = 1 - u - v
    return u, v, w


def get_cartesian(triangle, barycentric):
    """
    Calculates the cartesian coordinates of point given by its barycentric coordinates for a given triangle.
    Args:
        triangle : list
            A list containing the three vertices of a triangle.
        barycentric : tuple
                A tuple u,v,w describing a point by a combination of the vertices of the given triangle.
    Returns:
        tuple
            A tuple x,y describing the cartesian coordinates of the given point.
    """
    u, v, w = barycentric
    a, b, c = triangle
    x = a[0]*u + b[0]*v + c[0]*w
    y = a[1]*u + b[1]*v + c[1]*w
    return x, y


def test_triangle(triangle, point):
    """
    Tests whether the given cartesian coordinates lie within the given triangle.
    Args:
        triangle : list
            A list containing the three vertices of a triangle.
        point : tuple
            A tuple x,y describing a two dimensional point.
    Returns:
        bool
            True if the given cartesian coordinates lie within the given triangle.
    """
    u, v, w = get_barycentric(triangle, point)
    return (0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1)


def test_barycentric(point, epsilon=0):
    """
    Tests whether the given barycentric coordinates lie within a triangle.
    Args:
        point : tuple
            A tuple u,v,w describing a point by a combination of the vertices of the given triangle.
        epsilon : float
            If greater than zero allows for the coordinates to lie slightly
            outside of the triangle and still be counted as inside.
    Returns:
        bool
            True if the given barycentric coordinates lie within the given triangle.
    """
    u, v, w = point
    epsilon = abs(epsilon)
    return (-epsilon <= u <= 1+epsilon and -epsilon <= v <= 1+epsilon and -epsilon <= w <= 1+epsilon)


def map_to_baseface(base_face, cur_face, point):
    """
    Maps the given fixation for a given face that was looked at to a given base face.
    This allows for fixation coordinates in videos to be movement invariant.
    And allows to compare coordinates for different videos.
    Args:
        base_face : list
            List of landmarks in the base face.
        cur_face : list
            List of landmarks in the face the participant looked at.
        point : tuple
            The x and y coordinate of a fixation.
    Returns:
        The coordinates for the given fixation mapped to the base face.
        If the background (not the face) was fixated None is returned.
    """
    if (cur_face is None or base_face is None):
        return None
    assert len(base_face) == len(cur_face)
    tris = Delaunay(base_face).vertices
    for tri in tris:
        bary = get_barycentric([cur_face[i] for i in tri], point)
        if (bary is not None and test_barycentric(bary, 0.1)):
            cartesian = get_cartesian([base_face[i] for i in tri], bary)
            return int(cartesian[0]), int(cartesian[1])
    else:
        return None


def map_fixations(fixations: list, landmarks: list, base_landmarks: list, delete_background: bool=False):
    """
    Maps the given list of fixations using the given base_landmarks to a base face.
    This allows for fixation coordinates in videos to be movement invariant.
    And allows to compare coordinates for different videos.
    Args:
        fixations : list
            A scanpath consisting of fixations.
        landmarks : list
            A landmarks object storing landmarks for each frame of the video.
        base_landmarks : list
            The landmarks the fixations are mapped to.
        delete_background : bool
            Whether fixation that lie outside the face are discarded or not.
    Returns:
        A list of fixations mapped to the base face.
    """
    mapped_fixations = []
    for fix in fixations:
        if (fix[2] >= len(landmarks)):
            if (not delete_background):
                mapped_fixations.append(result_classes.Fixation(fix.x, fix.y, fix.start, fix.end))
                continue
            else:
                continue
        new_fix = map_to_baseface(base_landmarks, landmarks[fix.start], (fix.x, fix.y))
        if (new_fix is not None):
            mapped_fixations.append(result_classes.Fixation(new_fix[0], new_fix[1], fix.start, fix.end))
        elif (not delete_background):
            mapped_fixations.append(result_classes.Fixation(fix.x, fix.y, fix.start, fix.end))
    return mapped_fixations


def map_fixation_data(fixation_data: result_classes.FixationsDataset, landmarks: result_classes.Landmarks, base_landmarks: list, delete_background=False):
    """
    Maps the given fixation data using landmarks to a given base face.
    This allows for fixation coordinates in videos to be movement invariant.
    And allows to compare coordinates for different videos.
    Args:
        fixation_data : FixationsDataset
            A scanpath consisting of fixations.
        landmarks : Landmarks
            A landmarks object storing landmarks for each frame of the video.
        base_image : path
            The image the fixations are mapped to.
        delete_background : bool
            Whether fixation that lie outside the face are discarded or not.
    Returns:
        A new FixationsDataset containing the mapped fixations.
    """
    if (not isinstance(fixation_data, result_classes.FixationsDataset) or not isinstance(landmarks, result_classes.Landmarks)):
        raise ValueError("Expected FixationDataset and Landmarks objects.")
        return None

    mapped_data_set = []
    for data in fixation_data.data_set:
        lm = landmarks.get_landmarks(data.video)
        if (lm is None):
            error_logger.log_error(f"No landmarks found for \'{data.video}\'.")
            continue
        fixations = []
        for fix in data.fixations:
            if (fix.start >= len(lm)):
                if (not delete_background):
                    fixations.append(result_classes.Fixation(fix.x, fix.y, fix.start, fix.end))
                    continue
                else:
                    break
            new_fix = map_to_baseface(base_landmarks, lm[fix.start], (fix.x, fix.y))
            if (new_fix is not None):
                fixations.append(result_classes.Fixation(new_fix[0], new_fix[1], fix.start, fix.end))
            elif (not delete_background):
                fixations.append(result_classes.Fixation(fix.x, fix.y, fix.start, fix.end))
        mapped_data = result_classes.FixationData(fixations, data.video, data.participant, data.filename, data.info)
        mapped_data_set.append(mapped_data)
    new_data_set = result_classes.FixationsDataset(mapped_data_set)
    return new_data_set
