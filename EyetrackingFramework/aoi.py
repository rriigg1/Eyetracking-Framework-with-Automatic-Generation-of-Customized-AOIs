import numpy as np
import math
import config
import os
import csv
import result_classes
import time
import cv2


def shape_to_parts(shape):
    """
    Converts a dlib.full_object_detection to a list of coordinates.
    """
    return list(shape.parts())


def shape_to_tuples(shape):
    """
    Converts a dlib.full_object_detection object to a list of tuples
    resambling the coordinates.
    """
    return [(p.x, p.y) for p in shape.parts()]


def shape_to_string(shape):
    """
    Converts a dlib.full_object_detection to a list of coordinates as a string.
    """
    parts = list(shape.parts())
    return str(parts)


def in_polygon(polygon, coord, padding=0):
    """
    Test if the given coordinate lies inside the given polygon.
    If padding is greater than zero, coordinates that are closer than padding
    to the polygon are accepted as well.
    Args:
        polygon : list
            A list of tuples containing the x and y coordinate of a points
            describing the vertices of the polygon.
        coord : tuple
            A tuple containing the x and y coordinate of a point.
            The point fow which to test whether it lies within the convex hull.
        padding : float
            Maximum distance the given coordinate is allowed to have to the hull,
            to be counted as within its bounds.
    Returns:
        Returns the distance to the boundary of the polygon.
        If the point lies outside of the polygon -1 is returned.
    """
    mindist = on_polyline(polygon, coord, max_distance=float("inf"), closed=True)
    # test for padding
    if (0 <= mindist <= padding):
        return mindist
    # Count the intersections of the polygons boundaries with a ray eminating
    # from coord towards positive x.
    # If this count is odd the coord lies within the polygon.
    intersection_count = 0
    for i in range(len(polygon)):
        p1 = polygon[i-1]
        p2 = polygon[i]
        if (p1 == coord):
            return 0
        d = [p2[0] - p1[0], p2[1] - p1[1]]
        if (d[0] == 0):
            continue
        if (p1[1] == coord[1] and p1[0] > coord[0]):
            # Test for lower corner or upper corner
            p3 = polygon[i-2]
            if ((p2[1] - coord[1]) * (p3[1] - coord[1]) < 0):
                intersection_count += 1
        elif (p1[0] >= coord[0] or p2[0] >= coord[0]):
            # Test if the line between p1 and p2 crosses the y-value of coord.
            if ((p1[1] - coord[1]) * (p2[1] - coord[1]) < 0):
                length = (coord[1] - p1[1]) / d[1]
                xintersection = length * d[0] + p1[0]
                if (xintersection >= coord[0]):
                    intersection_count += 1
    if (intersection_count % 2 == 1):
        return mindist
    else:
        return -1


def in_hull(points, coord, padding=0):
    """
    Calculates the convex hull for a given list of points and tests whether a
    given coordinate lies within its bounds.
    Args:
        points : list
            A list of tuples containing the x and y coordinate of a point.
        coord : tuple
            A tuple containing the x and y coordinate of a point.
            The point fow which to test whether it lies within the convex hull.
        padding : float
            Maximum distance the given coordinate is allowed to have to the hull,
            to be counted as within its bounds.
    """
    hull = cv2.convexHull(np.array(points))
    nhull = hull.reshape(-1, 2).tolist()
    return in_polygon(nhull, coord, padding)


def distance_to_line(line, coord):
    """
    Calculates the distance between the given line and coord.

    Returns:
        Returns the distance of coord to the given line.
    """
    p1 = line[0]
    p2 = line[1]
    if (p1 == p2):
        return math.hypot(p1[0]-coord[0], p1[1]-coord[1])
    d1 = [p2[0] - p1[0], p2[1] - p1[1]] # direction of the line
    d2 = [coord[0] - p1[0], coord[1] - p1[1]] # vector from coord to p1
    l1 = math.hypot(d1[0], d1[1]) # length of the line
    l2 = math.hypot(d2[0], d2[1]) # distance to p1
    offset = (d1[0] * d2[0] + d1[1] * d2[1]) / l1
    if (0 < offset < l1):
        if (-1e-8 < (l2 - offset) < 1e-8):
            return 0
        return math.sqrt(l2 * l2 - offset * offset)
    else:
        ds1 = math.hypot(p1[0]-coord[0], p1[1]-coord[1])
        ds2 = math.hypot(p2[0]-coord[0], p2[1]-coord[1])
        return min(ds1, ds2)


def on_polyline(polyline, coord, max_distance=20, closed=False):
    """
    Calculates whether coord lies within the distance of max_distance to
    the given polyline and returns the minimal distance.

    Returns:
        Returns the distance of coord to the polyline if less than max_distance
        else -1 is returned.
    """
    if (closed):
        idx = range(len(polyline))
    else:
        idx = range(1, len(polyline))

    mindist = max_distance + 1
    for i in idx:
        p1 = (polyline[i - 1][0], polyline[i - 1][1])
        p2 = (polyline[i][0], polyline[i][1])
        dist = distance_to_line((p1, p2), coord)
        mindist = min(mindist, dist)
    if (mindist < max_distance):
        return mindist
    else:
        return -1


def near_point(points, coord, max_distance=50):
    """
    Finds the point in points that is nearest to coord and
    returns the distance to it if it's less than max_distance.

    Returns:
        Returns the minimal distance of coord to a point in points if less
        than max_distance else -1 is returned.
    """
    a = coord
    mindist = max_distance + 1
    for pt in points:
        p = pt
        pa = [p[0] - a[0], p[1] - a[1]]
        dist = math.hypot(pa[0], pa[1])
        mindist = min(mindist, dist)
    if (mindist <= max_distance):
        return mindist
    else:
        return -1


def in_hessels_voronoi(points, coord, max_distance=None):
    """
    Averages the given coordinates and returns the distance to the center
    with a maximum of max_distance. If max_distance is set to None,
    the distance is not limited.

    Returns:
        Returns the distance of coord to the average of points if less
        than max_distance or max_distance is None else -1 is returned.
    """
    average = [sum(x)/len(x) for x in zip(*points)]
    d = [average[0] - coord[0], average[1] - coord[1]]
    distance = math.hypot(d[0], d[1])
    if (max_distance is None or max_distance <= 0 or distance <= max_distance):
        return distance
    else:
        return -1


def get_closest_aoi(landmarks, fixation, **kwargs):
    """
    Finds the AOI in SELECTED_GROUPING with the lowest distance
    to the fixation.

    Args:
        landmarks : list
            List of 2d-landmarks.

        fixation : tuple
            Tuple describing the looked at point.

    Returns:
        Name of the AOI with the lowest distance.
        Returns NONE if the fixation is too far away from any of the AOIs.
    """
    if (landmarks is None):
        return config.AOI_NO_FACE
    grouping = config.SELECTED_GROUPING
    if ("grouping" in kwargs):
        grouping = kwargs["grouping"]
    min_dist = float("inf")
    min_aoi = config.AOI_NONE
    for name, aoi in grouping.items():
        for part in aoi:
            pts = [landmarks[i] for i in part[0]]
            dist = config.DISTANCE_FUNCTIONS[part[1]][0](pts, fixation, *part[2])
            if (dist >= 0 and dist < min_dist):
                min_dist = dist
                min_aoi = name
    return min_aoi


def get_aoi(landmarks, fixation, **kwargs):
    """
    Finds the AOI in SELECTED_GROUPING which is fixated respecting the order of AOIs.

    Args:
        landmarks : list
            List of 2d-landmarks.

        fixation : tuple
            Tuple describing the looked at point.

    Returns:
        Name of the AOI that was fixated.
        Returns NONE if the fixation is too far away from any of the AOIs.
    """
    if (landmarks is None):
        return config.AOI_NO_FACE
    grouping = config.SELECTED_GROUPING
    if ("grouping" in kwargs):
        grouping = kwargs["grouping"]
    for name, aoi in grouping.items():
        for part in aoi:
            pts = [landmarks[i] for i in part[0]]
            dist = config.DISTANCE_FUNCTIONS[part[1]][0](pts, fixation, *part[2])
            if (dist >= 0):
                return name
    return config.AOI_NONE


def convex_hull(points):
    """
    For a given set of points calculates a minimal convex polygon containing all the points.
    Args:
        points : list
            A list of tuples containing the x and y coordinates of the points.
    Returns:
        list
            A list of points forming a minimal convex polygon containing all the points.
    """
    # find bottom left
    points = np.array(points)
    origin = points[0]
    origin_idx = 0
    for i, p in enumerate(points):
        if (p[1] < origin[1]):
            origin = p
            origin_idx = i
        elif (p[1] == origin[1] and p[0] < origin[0]):
            origin = p
            origin_idx = i
    # subtract origin and reshape arrays
    hull = [(0, 0)]
    points = np.delete(points, origin_idx, 0)
    points = points.reshape(-1, 2)
    points = points - origin
    # sort points by angle with origin
    points = sorted(points, key=lambda x: (math.acos(x[0] / math.sqrt(x[0] * x[0] + x[1] * x[1])), math.fabs(x[0] + x[1])))
    points = np.array(points)
    hull.append(points[0])
    a = (0, 0)
    c = points[0]
    points = points[1:]
    # march along the edge of the polygon
    for bi, b in enumerate(points):
        bi += 2
        while (test_side(a, b, c)):
            hull.pop()
            c = hull[-1]
            a = hull[-2]
        else:
            hull.append(b)
            a = c
            c = b
    hull = hull + origin
    return hull


def test_side(a, b, c):
    """
    Tests whether c lies on left side of a line connecting a and b.
    If this is the case True is returned.
    Args:
        a : tuple
            Start point of the line.
        b : tuple
            End point of the line.
        c : tuple
            Single point for which is tested whether it is on the left or right side of the line.
    Returns:
        bool
            True if c is on left side of a line connecting a to b.
    """
    t = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    if (t > 0):
        return True
    else:
        return False


def get_closest_point(points, coord, max_distance=150):
    """
    Gets the closest point to coord within points with a set maximum distance.
    """
    if (points is None):
        return -1
    mindist = float("inf")
    min_idx = 0
    for i, point in enumerate(points):
        pdiff = [point[0] - coord[0], point[1] - coord[1]]
        dist = math.hypot(pdiff[0], pdiff[1])
        if (dist < mindist):
            mindist = dist
            min_idx = i
    if (max_distance is None or mindist <= max_distance):
        return min_idx
    else:
        return -1
