import dlib
import csv
import os
import numpy as np

# framework imports
import config


class Landmarks():
    """
    For each frame of a video stores landmarks in that frame.

    ### Attributes:
        videos : list
            Names of the videos the landmarks where generated for.
        video_files : list
            Names of the corresponding video files.
        landmarks : list
            A list of landmarks for each video.
            The landmarks are stored as "TUPLES" if possible.
    """

    def get_aoi_type(landmarks):
        if (not isinstance(landmarks, list) or len(landmarks) <= 0):
            print("No list")
            return None
        if (isinstance(landmarks[0], dlib.full_object_detection)):
            aoi_type = "SHAPE"
        elif (isinstance(landmarks[0], dlib.points)):
            aoi_type = "PARTS"
        elif (isinstance(landmarks[0], list) and isinstance(landmarks[0][0], tuple)):
            aoi_type = "TUPLES"
        else:
            aoi_type = None
        return aoi_type

    def __init__(self, video_files, landmarks):
        self.videos = [os.path.splitext(os.path.split(file)[1])[0] for file in video_files]
        self.video_files = video_files
        if (not isinstance(landmarks, list)):
            self.landmarks = landmarks
            print("Type of landmarks not supported: {}".format(type(landmarks)))
            return
        aoi_type = Landmarks.get_aoi_type(landmarks[0])
        if (aoi_type != "TUPLES"):
            if ((aoi_type, "TUPLES") not in config.AOI_CONVERTER):
                print("Aois of type {} can't be converted to tuples".format(aoi_type))
            else:
                converter = config.AOI_CONVERTER[(aoi_type, "TUPLES")]
                self.landmarks = [[converter(r) for r in f] for f in landmarks]
        else:
            self.landmarks = landmarks

    def add_video(self, video_file, landmarks):
        self.videos.append(os.path.splitext(os.path.split(video_file)[1])[0])
        self.video_files.append(video_file)
        aoi_type = Landmarks.get_aoi_type(landmarks)
        if (aoi_type != "TUPLES"):
            if ((aoi_type, "TUPLES") not in config.AOI_CONVERTER):
                print("Aois of type {} can't be converted to tuples".format(aoi_type))
            else:
                converter = config.AOI_CONVERTER[(aoi_type, "TUPLES")]
                self.landmarks.append([converter(r) for r in landmarks])
        else:
            self.landmarks.append(landmarks)

    def get_landmarks(self, video):
        for i, v in enumerate(self.videos):
            if (v == video):
                return self.landmarks[i]
        return None

    def save(self, dir="landmarks/", delimiter=","):
        """
        Saves the content to a file in the given directory.

        ### Args:
            dir : path
                The directory where the data is saved to.
            delimiter : char
                The delimiter to use in the produced files.
        """
        if (not os.path.exists(dir)):
            os.mkdir(dir)
        output_files = []
        for i, v in enumerate(self.videos):
            fmt_data = [np.array(row).flatten() for row in self.landmarks[i]]
            file = os.path.join(dir, "{}_landmarks.csv".format(v))
            with open(file, "w", newline="") as fh:
                wr = csv.writer(fh, delimiter=delimiter)
                wr.writerow(["#video"])
                wr.writerow(["#" + self.video_files[i]])
                wr.writerows(fmt_data)
            output_files.append(file)
        return output_files

    def load(filenames, delimiter=","):
        landmark_list = []
        video_names = []
        for fn in filenames:
            with open(fn, "r") as f:
                file_data = [row for row in f]
                info_data = None
                if (file_data[0][0] == "#" and file_data[1][0] == "#"):
                    info_reader = csv.DictReader(map(lambda r: r[1:], file_data[:2]), delimiter=",")
                    info_data = [row for row in info_reader][0]
                else:
                    print(f"No header found for file {fn}. Needed to determine video name.")
                    continue
                if ("video" not in info_data):
                    print(f"Header is either missing or incomplete in file {fn}")
                    continue
                csvfile = csv.reader(filter(lambda r: r[0] != "#", file_data), delimiter=delimiter)
                fmt_data = []
                for row in csvfile:
                    if (len(row) > 1):
                        fmt_data.append([(int(coord[0]), int(coord[1])) for coord in np.array(row).reshape((-1, 2))])
                    else:
                        fmt_data.append(None)
                landmark_list.append(fmt_data)
                video_names.append(info_data["video"])
        return Landmarks(video_names, landmark_list)
