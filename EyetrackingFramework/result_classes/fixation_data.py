import math
import csv
import os


class Fixation:
    """
    Stores the coordinates of a fixation and the frame it started in and the frame that was displayed while it ended.
    """
    def __init__(self, x, y, start_frame, end_frame):
        self.x = x
        self.y = y
        self.start = start_frame
        self.end = end_frame

    # deprecated use x, y, start, end instead
    def __getitem__(self, key):
        """
        Gets one or more values of the fixation.
        For example fixation[:2] gets only the coordinates of the fixation.
        """
        if(isinstance(key, int)):
            if (key == 0):
                return self.x
            elif (key == 1):
                return self.y
            elif (key == 2):
                return self.start
            elif (key == 3):
                return self.end
            else:
                return None
        elif(isinstance(key, slice)):
            data = [self.x, self.y, self.start, self.end]
            return data[slice(key.start, key.stop, key.step)]

    # deprecated use x, y, start, end instead
    def __setitem__(self, key, value):
        """
        Sets a single value of the fixation.
        """
        if(isinstance(key, int)):
            if (key == 0):
                self.x = value
            elif (key == 1):
                self.y = value
            elif (key == 2):
                self.start = value
            elif (key == 3):
                self.end = value
            else:
                return False
        else:
            return False
        return True

    def __iter__(self):
        """
        Iterator used in saving.
        """
        data = [self.x, self.y, self.start, self.end]
        return iter(data)

    def duration(self):
        """
        Returns the duration of the fixation in frames.
        """
        return self.end-self.start


class FixationData():
    """
    A class that stores fixation data.

    ### Attributes:
        fixations : list
            A list of tuples of format (x, y, start_frame, end_frame)
        video : str
            Name of the corresponding video.
        participant : str
            Number of the participant.
        file : str
            File from which the data was generated.
        info: str
            Additional information that can be used in further processing.
    """

    def __init__(self, fixations, video, participant="0", file="data.csv", info=None):
        self.fixations = fixations
        self.video = video
        self.participant = participant
        self.filename = os.path.splitext(os.path.split(file)[1])[0]
        self.info = info

    def save(self, dir="preprocessed/", delimiter=","):
        file_name = "{}_{}_{}_fixations.csv".format(self.filename, self.video, self.participant)
        file_path = os.path.join(dir, file_name)
        with open(file_path, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            if (self.info is not None):
                writer.writerow(["#video","participant","info","source"])
                writer.writerow(["#" + self.video, self.participant, self.info, self.filename])
            else:
                writer.writerow(["#video","participant","source"])
                writer.writerow(["#" + self.video, self.participant, self.filename])
            writer.writerow(["#x y start_frame end_frame"])
            writer.writerows(self.fixations)
        return file_path

    def simplify(self, distance_threshold=30, direction_threshold=10):
        """
        Simplifies the scanpath by merging fixations that are
        closer together than distance_threshold and merging saccades which
        direction deviates by less than direction_threshold degrees.
        """
        direction_threshold = math.cos(direction_threshold/180*math.pi)
        last_fix = self.fixations[0]
        nscanpath = [last_fix]
        for fix in self.fixations[1:]:
            last_fix = nscanpath[-1]
            # cluster close together fixations
            d0 = fix.x-last_fix.x
            d1 = fix.y-last_fix.y
            if (d0*d0 + d1*d1 < distance_threshold*distance_threshold):
                frames = fix.end - last_fix.start
                first_frames = last_fix.duration()
                second_frames = frames - first_frames
                nscanpath[-1].x = (last_fix.x*first_frames + fix.x*second_frames) / frames
                nscanpath[-1].y = (last_fix.y*first_frames + fix.y*second_frames) / frames
                nscanpath[-1].end = fix.end
            # cluster direction-wise similar saccades
            elif (len(nscanpath) > 1):
                last_dir = (nscanpath[-1].x-nscanpath[-2].x, nscanpath[-1].y-nscanpath[-2].y)
                denom = math.hypot(last_dir[0], last_dir[1])
                last_dir = (last_dir[0] / denom, last_dir[1] / denom)
                new_dir = (fix.x-last_fix.x, fix.y-last_fix.y)
                denom = math.hypot(new_dir[0], new_dir[1])
                new_dir = (new_dir[0] / denom, new_dir[1] / denom)
                # compare dot product to not have to calculate a cosine
                if (last_dir[0]*new_dir[0]+last_dir[1]*new_dir[1] > direction_threshold):
                    nscanpath[-1] = fix
                else:
                    nscanpath.append(fix)
            else:
                nscanpath.append(fix)
        self.fixations = nscanpath

    def __getitem__(self, key):
        """
        Returns a slice of the fixations list.
        """
        if (not isinstance(key, slice)):
            raise TypeError("Only slicing supported.")
        start = key.start
        stop = key.stop
        if key.step is not None and key.step != 1:
            raise ValueError("Step needs to be one.")
        if (start is not None):
            i = 0
            while (i <= len(self.fixations) and self.fixations[i].end < start):
                i += 1
            start = i

        if (stop is not None):
            i = len(self.fixations) - 1
            while (i >= 0 and self.fixations[i].start > stop):
                i -= 1
            stop = i + 1
        return self.fixations[slice(start, stop)]

    def get_name():
        """
        Returns a name for the fixation data generated from the name of the video and participant
        """
        return f"{self.video}_{self.participant}"

class FixationsDataset():
    """
    A class that stores and handles preprocessed data in form of fixation data.

    ### Attributes:
        videos : dict
            A dictionary which for each video stores the corresponding.
            fixation data.
        data_set : list
            A list of FixationData.
    """

    def __init__(self, data_set):
        self.videos = {}
        self.data_set = []
        for i, data in enumerate(data_set):
            self.data_set.append(data)
            self.add_video(data.video, i)

    def save(self, dir="preprocessed/", delimiter=","):
        """
        Save all contained fixation data to the given directory using the
        given delimiter.
        ### Args:
            dir : path
                The directory where the data is saved to.
            delimiter : char
                The delimiter to use in the produced files.
        ### Returns:
            list
                Returns a list of the created files.
        """
        if (not os.path.exists(dir)):
            os.mkdir(dir)
        output_files = []
        for data in self.data_set:
            output_files.append(data.save(dir, delimiter))
        return output_files

    def add_video(self, video, idx):
        """
        Adds data to the video dictionary.
        """
        if (video in self.videos):
            self.videos[video].append(idx)
        else:
            self.videos[video] = [idx]

    def get_video_data(self, video):
        """
        Returns all the fixation data for a given video.
        """
        if (video not in self.videos):
            # print("No fixation data for video \'{}\' found.".format(video))
            return []
        return [self.data_set[i] for i in self.videos[video]]

    def append_prep(self, preprocessed_data):
        """
        Appends new preprocessed data to the existing one and creates entries
        in the dictionary if necessary.
        """
        length = len(self.data_set)
        for i, data in enumerate(preprocessed_data.data_set):
            self.data_set.append(data)
            self.add_video(data.video, length + i)

    def append_data(self, data_set):
        """
        Appends new fixation data to the existing one and creates entries
        in the dictionary if necessary.
        """
        length = len(self.data_set)
        for i, data in enumerate(data_set):
            self.data_set.append(data)
            self.add_video(data.video, length + i)

    def load(filenames, delimiter=","):
        """
        Loads fixations from files given as a list of filesnames.
        A delimiter that was used in the csv files can be specified.
        """
        data_set = []
        for fn in filenames:
            with open(fn, "r") as f:
                file_data = [row for row in f]
                info_data = None
                if (file_data[0][0] == "#" and file_data[1][0] == "#"):
                    info_reader = csv.DictReader(map(lambda r: r[1:], file_data[:2]), delimiter=",")
                    info_data = [row for row in info_reader][0]
                else:
                    print("No header found. Needed to determine video name, participant number and source file.")
                    continue
                if ("video" not in info_data or "participant" not in info_data or "source" not in info_data):
                    print("Header of \'{}\' is missing collumns.".format(fn))
                    continue
                csvfile = csv.reader(filter(lambda r: r[0] != "#", file_data), delimiter=delimiter)
                raw_data = [r for r in csvfile]
                fmt_data = [Fixation(*[int(value) for value in row]) for row in raw_data]
                if ("info" not in info_data):
                    info_data["info"] = None
                fix_data = FixationData(fmt_data, info_data["video"], info_data["participant"], info_data["source"], info_data["info"])
                data_set.append(fix_data)
        return FixationsDataset(data_set)

    def simplify(self, distance_threshold=30, direction_threshold=10):
        """
        Simplyfies the fixation data by merging fixations that are closer to
        each other than a given threshold. Or saccades which differ in direction
        by less than a given threshold.
        """
        for data in self.data_set:
            data.simplify(distance_threshold=distance_threshold, direction_threshold=direction_threshold)

    def get_raw_fixations(self):
        """
        Returns the concatenated raw fixation data from all participants and videos.
        Only x and y position and duration are kept.
        """
        fixations = []
        for data in self.data_set:
            for fix in data.fixations:
                fixations.append((fix.x, fix.y, fix.duration()))
        return fixations
