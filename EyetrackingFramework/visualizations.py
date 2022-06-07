import random
import math
import numpy as np
import cv2
import scipy.spatial as spatial
import inspect

# framework imports
import config
import aoi
import utilities
import landmark_utils
import face_mapper
import error_logger


def visualize_image(image, fixations, landmarks, visualization_options, image_landmarks=None, map_data=False, grouping=config.AOI_GROUPINGS[config.SELECTED_GROUPING], start_frame=0, end_frame=None):
    """
    Creates a visulaizations with the given parameters outputting a single image.
    """
    vis_types = ["draw_aois", "draw_landmarks", "draw_fixations", "draw_arrows"]
    vis_types = list(filter(lambda k: k in visualization_options and visualization_options[k], vis_types))
    if (image_landmarks is None and ("draw_aois" in vis_types or "draw_landmarks" in vis_types or map_data)):
        image_landmarks = landmark_utils.get_landmarks(image)
    if ("draw_aois" in vis_types):
        arg_list = inspect.getargspec(draw_grouping).args
        cur_vis = dict(filter(lambda k: k[0] in arg_list, visualization_options.items()))
        draw_grouping(image, image_landmarks, grouping, **cur_vis)
    if ("draw_landmarks" in vis_types):
        arg_list = inspect.getargspec(draw_landmarks).args
        cur_vis = dict(filter(lambda k: k[0] in arg_list, visualization_options.items()))
        draw_landmarks(image, image_landmarks, **cur_vis)

    # handle visualization of fixations
    if (fixations is not None):
        # map and slice landmarks
        fixation_data = []
        for data in fixations.data_set:
            # slice to only relevant data
            if ((start_frame is not None and start_frame > 0) or end_frame is not None):
                f_data = data[slice(start_frame, end_frame)]
            else:
                f_data = data.fixations
            if ("fixation_count" in visualization_options and visualization_options["fixation_count"] is not None and visualization_options["fixation_count"] > 0):
                f_data = f_data[-visualization_options["fixation_count"]:]
            if (map_data):
                # get landmarks for the video and map the fixations to the base_landmarks
                l_data = landmarks.get_landmarks(data.video)
                if (l_data is None):
                    error_logger.log_error(f"No landmarks for video \'{data.video}\' found. Unable to map fixations.")
                    continue
                f_data = face_mapper.map_fixations(f_data, l_data, image_landmarks, delete_background=False)
            if (len(f_data) < 1):
                # no fixations are within the specified time interval
                continue
            fixation_data.append(f_data)

        # draw fixation data
        if ("draw_arrows" in vis_types):
            arg_list = inspect.getargspec(draw_arrows).args
            cur_vis = dict(filter(lambda k: k[0] in arg_list, visualization_options.items()))
            draw_arrows(image, fixation_data, **cur_vis)
        if ("draw_fixations" in vis_types):
            arg_list = inspect.getargspec(draw_fixations).args
            cur_vis = dict(filter(lambda k: k[0] in arg_list, visualization_options.items()))
            draw_fixations(image, fixation_data, ignore_last_fixation=("draw_arrows" in vis_types), **cur_vis)


def draw_landmarks(image, landmarks, color=config.LANDMARK_COLOR, radius=config.LANDMARK_SIZE, label_landmarks=False):
    """
    Visualizes the landmarks in the image
    """
    for idx, landmark in enumerate(landmarks):
        cv2.circle(image, landmark, radius, color, -1)
        if (label_landmarks):
            cv2.putText(image, str(idx), landmark, cv2.FONT_HERSHEY_PLAIN, 0.6, color)


def draw_fixations(image, fixations, radius=config.FIXATION_SIZE, random_colors=False, label_fixations=False, fixation_count=5, ignore_last_fixation=False):
    """
    Visualizes the given landmarks as circles on the image.
    """
    dot_colors = []
    color_offset = utilities.LAST_COLOR_COUNT + 1
    for i in range(len(fixations)):
        dot_colors.append(utilities.get_color(i+color_offset) if random_colors == -1 else utilities.get_color(random_colors))
    for i, data in enumerate(fixations):
        sliced_data = data[-fixation_count:] if fixation_count is not None and fixation_count > 0 else data
        if (ignore_last_fixation):
            sliced_data = sliced_data[:-1]
        for fix in sliced_data:
            point = (fix.x, fix.y)
            cv2.circle(image, point, radius, dot_colors[i], int(radius/5)+1)
    if (label_fixations):
        for i, data in enumerate(fixations):
            point = (data[-1].x, data[-1].y)
            cv2.putText(image, str(i), point, cv2.FONT_HERSHEY_PLAIN, 0.6, dot_colors[i])


def draw_arrows(image, fixations, thickness=config.ARROW_THICKNESS, random_colors=False, fixation_count=5):
    """
    Visualizes the given landmarks as circles on the image.
    """
    arrow_colors = []
    color_offset = utilities.LAST_COLOR_COUNT + 1
    for i in range(len(fixations)):
        arrow_colors.append(utilities.get_color(i+color_offset) if random_colors == -1 else utilities.get_color(random_colors))
    for i, data in enumerate(fixations):
        sliced_data = data[-fixation_count:] if fixation_count is not None and fixation_count > 0  else data
        for f_idx in range(len(sliced_data) - 1):
            a = (sliced_data[f_idx].x, sliced_data[f_idx].y)
            b = (sliced_data[f_idx+1].x, sliced_data[f_idx+1].y)
            tl = config.ARROW_TIP_SIZE / math.hypot(a[0]-b[0], a[1]-b[1])
            cv2.arrowedLine(image, a, b, arrow_colors[i], thickness, tipLength=tl)



def draw_grouping(image, landmarks, grouping, random_colors=False, label_aois=True, fill_polygons=False):
    """
    Draws the given grouping using given landmarks to a given image.
    This method is intended to run fast and to only give a quick overview of
    the current grouping.
    """
    if (landmarks is None or grouping is None):
        return
    if (random_colors is not False):
        random.seed(random_colors)
    count = 0
    for group, parts in reversed(list(grouping.items())):
        if (random_colors is not False and (random_colors is True or random_colors >= 0)):
            color = utilities.get_color(random_color=True)
        elif(random_colors == -1):
            color = utilities.get_color(random_color=count)
            count += 1
        else:
            color = utilities.get_color(random_color=False)
        for part in parts:
            # no points -> no drawing
            if (len(part[0]) <= 0):
                continue
            pts = np.array([landmarks[i] for i in part[0]])
            # get center of the part used for certain type of aois and labels
            center = [sum(d)/len(d) for d in zip(*pts)]
            center = (int(center[0]), int(center[1]))
            if (part[1] == "LINE"):
                thickness = int(part[2][0]*2) if (len(part[2]) > 0) else 20
                closed = part[2][1] if (len(part[2]) > 1) else False
                if (random_colors is False):
                    color = (50, 190, 220)
                cv2.polylines(image, [pts], closed, color, thickness=thickness)
            elif (part[1] == "POLYGON"):
                thickness = int(part[2][0]*2) if (len(part[2]) > 0) else 1
                if (random_colors is False):
                    color = (0, 255, 0)
                cv2.polylines(image, [pts], True, color, thickness=thickness)
                if (fill_polygons):
                    cv2.fillPoly(image, [pts], color)
            elif (part[1] == "HULL"):
                thickness = int(part[2][0]*2) if (len(part[2]) > 0) else 1
                pts = cv2.convexHull(np.array(pts))
                pts = pts.reshape((-1, 2))
                if (random_colors is False):
                    color = (255, 255, 0)
                cv2.polylines(image, [pts], True, color, thickness=thickness)
                if (fill_polygons):
                    cv2.fillPoly(image, [pts], color)
            elif (part[1] == "VORONOI-HESSELS"):
                radius = max(math.hypot(point[0]-center[0], point[1]-center[1]) for point in pts)
                if (radius < 20):
                    radius = 20
                if (random_colors is not False):
                    cv2.circle(image, center, 5, color, thickness=-1)
                    cv2.circle(image, center, int(radius), color, thickness=1)
            else:
                radius = int(part[2][0]) if (len(part[2]) > 0) else 1
                if (random_colors is False):
                    color = (0, 255, 0)
                for pt in pts:
                    cv2.circle(image, tuple(pt), radius, color, thickness=-1)
            if (label_aois is True):
                text_color = (255-color[0], 255-color[1], 255-color[2])
                if (part[1] == "LINE"):
                    cv2.putText(image, str(group), (pts[int(len(pts)/2)][0], pts[int(len(pts)/2)][1]), cv2.FONT_HERSHEY_PLAIN, 0.6, text_color)
                else:
                    cv2.putText(image, str(group), center, cv2.FONT_HERSHEY_PLAIN, 0.6, text_color)
    draw_hessels_voronoi(image, landmarks, grouping, draw_centers=(random_colors is False), fill_polygons=fill_polygons, draw_range=True)


def draw_hessels_voronoi(image, landmarks, grouping, draw_centers=False, fill_polygons=False, draw_range=False):
    """
    Draws a voronoi tesselation for all aois of type "VORONOI-HESSELS".
    The visualization is similar to the one shown in the paper of Hessels et. al.
    """
    if (landmarks is None or grouping is None):
        return
    voronoi_cell_centers = []
    max_dist = 50
    for group, parts in grouping.items():
        for part in parts:
            if (part[1] == "VORONOI-HESSELS" and len(part[0]) >= 1):
                pts = np.array([list(landmarks[i]) for i in part[0]])
                center = [sum(d)/len(d) for d in zip(*pts)]
                center = (int(center[0]), int(center[1]))
                voronoi_cell_centers.append(center)
                if (part[2][0] is not None and part[2][0] > max_dist):
                    max_dist = part[2][0]

    if (len(voronoi_cell_centers) <= 2):
        return
    color = (0, 0, 255)
    if (draw_centers):
        for point in voronoi_cell_centers:
            cv2.circle(image, point, 5, color, thickness=-1)
    vor = spatial.Voronoi(np.array(voronoi_cell_centers))
    augmented_points = {}  # stores coordinates of the augmented points for each vertex
    for i, idx in enumerate(vor.ridge_vertices):
        if (idx[0] >= 0 and idx[1] >= 0):
            p1 = (int(vor.vertices[idx[0]][0]), int(vor.vertices[idx[0]][1]))
            p2 = (int(vor.vertices[idx[1]][0]), int(vor.vertices[idx[1]][1]))
            # cv2.circle(image, (int(p[0]), int(p[1])), 2, color, thickness=-1)
            cv2.line(image, p1, p2, color, thickness=3)
        else:
            # draw a ridge between two cells that point towards the edge of the image
            if (idx[0] >= 0):
                p1 = (int(vor.vertices[idx[0]][0]), int(vor.vertices[idx[0]][1]))
            else:
                p1 = (int(vor.vertices[idx[1]][0]), int(vor.vertices[idx[1]][1]))
            # get point on line by combining the centers of the neighbouring cells
            ridge_pts = vor.ridge_points[i]
            ridge1 = (int(vor.points[ridge_pts[0]][0]), int(vor.points[ridge_pts[0]][1]))
            ridge2 = (int(vor.points[ridge_pts[1]][0]), int(vor.points[ridge_pts[1]][1]))
            dir = (ridge2[0]-ridge1[0], ridge2[1]-ridge1[1])
            p2 = (ridge1[0] + 0.5*dir[0], ridge1[1] + 0.5*dir[1])
            # get normal of vector between ridge points (same as direction of the voronoi edge)
            dir = (dir[1], -dir[0])
            # fit the vector to max length
            length = math.hypot(dir[0], dir[1])
            fac = max_dist / length
            dir = (dir[0]*fac, dir[1]*fac)
            p2 = (p1[0] + dir[0], p1[1] + dir[1])
            # test if the line is pointing in the right direction
            closest_point = aoi.get_closest_point(vor.points, p2, None)
            if (closest_point not in ridge_pts):
                dir = (-dir[0], -dir[1])
            p2 = (int(p1[0] + dir[0]), int(p1[1] + dir[1]))
            augmented_points[idx[1] if idx[0] < 0 else idx[0]] = p2
            cv2.line(image, p1, p2, color, thickness=3)

    if (draw_range):
        # draw arc segments
        for i, fi in enumerate(vor.point_region):
            f = vor.regions[fi]
            for vi in range(len(f)):
                if (f[vi] == -1):
                    v1 = augmented_points[f[(vi+1) % len(f)]]
                else:
                    v1 = (int(vor.vertices[f[vi]][0]), int(vor.vertices[f[vi]][1]))
                if (f[(vi+1) % len(f)] == -1):
                    v2 = augmented_points[f[vi]]
                    v2_2 = augmented_points[f[(vi+2) % len(f)]]
                else:
                    v2 = (int(vor.vertices[f[(vi+1) % len(f)]][0]), int(vor.vertices[f[(vi+1) % len(f)]][1]))
                    v2_2 = None
                seg = get_arc_segments(vor.points[i], v1, v2, max_dist)
                for s in seg:
                    start = s[0]
                    delta = (s[1]-s[0])%360
                    cv2.ellipse(image, (int(vor.points[i][0]), int(vor.points[i][1])), (int(max_dist), int(max_dist)), start, 0, delta, color, 4)
                if (v2_2 is not None):
                    a1 = math.atan2(v2[1]-vor.points[i][1], v2[0]-vor.points[i][0])/math.pi*180%360
                    a2 = math.atan2(v2_2[1]-vor.points[i][1], v2_2[0]-vor.points[i][0])/math.pi*180%360
                    mi = f[0] if f[0] > -1 else f[1]
                    a3 = math.atan2(vor.vertices[mi][1]-vor.points[i][1], vor.vertices[mi][0]-vor.points[i][0])/math.pi*180%360
                    if (test_if_angle_in_middle(a3, a1, a2)):
                        a1, a2 = a2, a1
                    start = a1
                    delta = (a2 - a1) % 360
                    cv2.ellipse(image, (int(vor.points[i][0]), int(vor.points[i][1])), (int(max_dist), int(max_dist)), start, 0, delta, color, 4)


def test_if_angle_in_middle(middle, left, right):
    """
    Returns True if the angle described by middle lies in between the angles left and right.
    Otherwise False is returned.
    """
    return ((middle-left)%360 < (right-left)%360)


def get_arc_segments(center, p1, p2, radius):
    """
    For a given line segment a center of a circle and a radius calculates the
    intersection of the line segment and the circle and returns the resulting
    arcsegments, which are closer to the center of the circle than the line.
    The arc segments are returned as intervals of angles.
    """
    cp1 = (center[0]-p1[0], center[1]-p1[1])
    d1 = math.hypot(cp1[1], cp1[0])  # distance center - p1
    d2 = math.hypot(p2[1]-center[1], p2[0]-center[0])  # distance center - p2
    if (d1 <= radius and d2 <= radius):
        return []
    a1 = math.atan2(-cp1[1], -cp1[0])/math.pi*180 % 360  # angle p1
    a2 = math.atan2(p2[1]-center[1], p2[0]-center[0])/math.pi*180 % 360  # angle p2
    ldir = (p2[0]-p1[0], p2[1]-p1[1])
    length = math.hypot(ldir[0], ldir[1])  # length of the line segment
    d_x = (ldir[0]*cp1[0] + ldir[1]*cp1[1]) / length  # distance along the line
    if ((d_x <= 0 and d1 >= radius) or (d_x >= length and d2 >= radius)):
        return [(a1, a2) if (a2-a1) % 360 < 180 else (a2, a1)]
    d_y = math.sqrt(d1*d1 - d_x*d_x)  # distance to the line
    if (d_y >= radius):
        return [(a1, a2) if (a2-a1) % 360 < 180 else (a2, a1)]
    delta = math.sqrt(radius*radius - d_y*d_y)  # distance of circle intersections
    segments = []
    if (d1 >= radius):
        # calculate first intersection and append the arc segement
        intersection = (p1[0] + ldir[0]*(d_x-delta)/length, p1[1] + ldir[1]*(d_x-delta)/length)
        a_i = math.atan2(intersection[1]-center[1], intersection[0]-center[0])/math.pi*180 % 360
        segments.append((a1, a_i) if (a_i-a1) % 360 < 180 else (a_i, a1))
    if (d2 >= radius):
        # calculate second intersection and append the arc segement
        intersection = (p1[0] + ldir[0]*(d_x+delta)/length, p1[1] + ldir[1]*(d_x+delta)/length)
        a_i = math.atan2(intersection[1]-center[1], intersection[0]-center[0])/math.pi*180 % 360
        segments.append((a_i, a2) if (a2-a_i) % 360 < 180 else (a2, a_i))
    return segments


def draw_scanpath(image, fixations, frame_num=None, fixation_count=10, arrow_thickness=4, dot_radius=7, arrow_color=None, dot_color=None):
    """
    Draws a scanpath to the given image. The scanpath is given as a list
    of fixations. For each fixation an arrow is drawn pointing toward the
    next fixation.
    """
    # get random colors if neeeded
    if (arrow_color is None):
        arrow_color = utilities.get_color(True)
    if (dot_color is None):
        dot_color = utilities.get_color(True)
    # set fixation count and frame number if neeeded
    if (fixation_count is None or fixation_count == 0):
        fixation_count = len(fixations)
    if (frame_num is None):
        fix_idx = len(fixations)
    else:
        # get the last fixation that occured before or during the given frame.
        fix_idx = 0
        while (fix_idx < len(fixations) and fixations[fix_idx].end < frame_num):
            fix_idx += 1
    start_idx = max(fix_idx - fixation_count, 0)
    fix_points = fixations[start_idx: fix_idx]
    p1 = (fix_points[0][0], fix_points[0][1])
    if (len(fix_points) == 1):
        cv2.circle(image, p1, dot_radius, dot_color, thickness=3)
        return
    # for all fixations except the last one draw an arrow to next fixation
    # in the list
    for i in range(len(fix_points)-1):
        p2 = (fix_points[i+1][0], fix_points[i+1][1])
        # draw the tip of the arrow 1.3 times as large as the radius of the dots
        tl = dot_radius / math.hypot(p1[0]-p2[0], p1[1]-p2[1]) * 1.3
        cv2.arrowedLine(image, p1, p2, arrow_color, arrow_thickness, tipLength=tl)
        cv2.circle(image, p1, dot_radius, dot_color, thickness=3)
        p1 = p2
