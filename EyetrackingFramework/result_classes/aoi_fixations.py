import csv
import os


class FixatedAOIs:
    """
    Stores the fixated aois for a given video and participant.

    ### Attributes:
        aois : list
            Names or index of the aoi fixated at a given video frame.
        video : str
            Name of the video the participant has watched.
        participant : str
            Name or number of the participant.
        source_name : str
            Name of the file the original data came from.
    """

    def __init__(self, aois, video_file, participant="0", source_name="data", aoi_names=[], info=None, grouping="UNKNOWN"):
        self.aois = aois
        self.video = os.path.splitext(os.path.split(video_file)[1])[0]
        self.video_file = video_file
        self.participant = participant
        self.source = source_name
        self.aoi_names = aoi_names
        self.info = info
        self.grouping = grouping

    def save(self, dir="fixated_aois/", delimiter=","):
        """
        Saves the content to a file in the given directory.
        ### Args:
            dir : path
                The directory where the data is saved to.
            delimiter : char
                The delimiter to use in the produced files.
        """
        file_name = "{}_{}_{}_fixated_aois.csv".format(self.source, self.video, self.participant)
        file_path = os.path.join(dir, file_name)
        with open(file_path, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            if (self.info is not None):
                writer.writerow(["#video", "participant", "info", "source", "grouping"])
                writer.writerow(["#" + self.video_file, self.participant, self.info, self.source, self.grouping])
            else:
                writer.writerow(["#video", "participant", "source", "grouping"])
                writer.writerow(["#" + self.video_file, self.participant, self.source, self.grouping])
            if (len(self.aoi_names) > 0):
                writer.writerow(["#aoi_names"])
                writer.writerow(["#" + self.aoi_names[0]] + self.aoi_names[1:])
            writer.writerow(["frame", "aoi"])
            for row in self.aois:
                writer.writerow(row)
        return file_path

    def __getitem__(self, key):
        """
        Returns a slice of the aois list.
        """
        if (not isinstance(key, slice)):
            raise TypeError("Only slicing supported.")
        start = key.start
        stop = key.stop
        if key.step is not None and key.step != 1:
            raise ValueError("Step needs to be one.")
        if (start is not None):
            i = 0
            while (i < len(self.aois) and self.aois[i][0] < start):
                i += 1
            start = i
        if (stop is not None):
            i = len(self.aois) - 1
            while (i >= 0 and self.aois[i][0] > stop):
                i -= 1
            stop = i + 1
        return self.aois[slice(start, stop)]


class FixatedAOIsData:
    """
    Stores FixatedAOIs objects for multiple videos.
    The data can be accessed via a dictionary using the video name as a key.
    ### Attributes:
        videos : dict
            A dictionary using the video names as a key and storing
            a list of indices as a value.
        aoi_data : list
            A list of FixatedAOIs objects.
    """

    def __init__(self, fixated_aois):
        self.videos = {}
        self.aoi_data = []
        for i, aoi in enumerate(fixated_aois):
            self.aoi_data.append(aoi)
            self.add_video(aoi.video, i)

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
        Returns all the aoi data for a given video.
        """
        if (video not in self.videos):
            raise ValueError("No data for video \'{}\' found.".format(video))
        return [self.aoi_data[i] for i in self.videos[video]]

    def append_fixated_aois(self, fix_data):
        """
        Appends a given FixatedAOIsData object the existing one and creates
        entries in the dictionary if necessary.
        """
        length = len(self.aoi_data)
        for i, data in enumerate(fix_data.aoi_data):
            self.aoi_data.append(data)
            self.add_video(data.video, length + i)

    def append_aois(self, aoi_data):
        """
        Appends a list of FixatedAOIs objects to the existing aoi data and
        creates entries in the dictionary if necessary.
        """
        length = len(self.aoi_data)
        for i, data in enumerate(aoi_data):
            self.aoi_data.append(data)
            self.add_video(data.video, length + i)

    def save(self, dir="fixated_aois/", delimiter=","):
        """
        Save all contained fixation data to the given directory using the
        given delimiter.
        ### Args:
            dir : path
                The directory where the data is saved to.
            delimiter : char
                The delimiter to use in the produced files.
        """
        if (not os.path.exists(dir)):
            os.mkdir(dir)
        output_files = []
        for data in self.aoi_data:
            output_files.append(data.save(dir, delimiter))
        return output_files

    def load(filenames, delimiter=","):
        data_set = []
        for fn in filenames:
            with open(fn, "r") as f:
                file_data = [row for row in f]
                info_data = None
                # test info data for consistency
                if (file_data[0][0] == "#" and file_data[1][0] == "#"):
                    # read two rows and remove the # infront
                    info_reader = csv.DictReader(map(lambda r: r[1:], file_data[:2]), delimiter=",")
                    info_data = [row for row in info_reader][0]
                else:
                    print("No header found. Needed to determine video name, participant number and source file.")
                    continue
                if ("video" not in info_data or "participant" not in info_data or "source" not in info_data):
                    print("Header is either missing or incomplete in file {}".format(fn))
                    continue
                if ("#aoi_names" in file_data[2] and file_data[3][0] == "#"):
                    # read one row containing the aoi names and remove the # infront
                    name_reader = csv.reader([file_data[3][1:]], delimiter=",")
                    name_data = [row for row in name_reader][0]
                else:
                    print("No aoi names found.")
                    name_data = []
                # read all rows that are no comment and use the first row as a header
                csvfile = csv.DictReader(filter(lambda r: r[0] != "#", file_data), delimiter=delimiter)
                raw_data = [r for r in csvfile]
                if ("aoi" not in raw_data[0] or "frame" not in raw_data[0]):
                    print("Header is either missing or incomplete in file {}".format(fn))
                    continue
                else:
                    raw_data = list(map(lambda r: (int(r["frame"]), r["aoi"]), raw_data))
                # load additional info if found
                if ("info" in info_data):
                    info = info_data["info"]
                elif ("message" in info_data):
                    info = info_data["message"]
                else:
                    info = None

                if ("grouping" in info_data):
                    grouping = info_data["grouping"]
                else:
                    grouping = "UNKNOWN"
                aoi_data = FixatedAOIs(raw_data, info_data["video"],
                                       participant=info_data["participant"],
                                       source_name=info_data["source"],
                                       aoi_names=name_data,
                                       info=info,
                                       grouping=grouping)
                data_set.append(aoi_data)
        return FixatedAOIsData(data_set)
