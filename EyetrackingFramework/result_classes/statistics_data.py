import csv
import os

# framework imports
import config
import utilities


class FixationCount:
    """
    Saves the number of fixations per AOI.
    """
    def __init__(self, counts, video_file, participant, source, info=None, grouping="unknown"):
        self.video = os.path.splitext(os.path.split(video_file)[1])[0]
        self.video_file = video_file
        self.participant = participant
        self.source = source
        self.data = counts
        self.info = info
        self.grouping = grouping

    def save(self, dir="statistics/", delimiter=","):
        """
        Saves the content to a file in the given directory.
        Args:
            dir : path
                The directory where the data is saved to.
            delimiter : char
                The delimiter to use in the produced files.
        """
        file_name = "{}_{}_{}_fixation_count.csv".format(self.source, self.video, self.participant)
        file_path = os.path.join(dir, file_name)
        with open(file_path, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            if (self.info is not None):
                writer.writerow(["#video", "participant", "source", "info", "grouping"])
                writer.writerow(["#" + self.video_file, self.participant, self.source, self.info, self.grouping])
            else:
                writer.writerow(["#video", "participant", "source", "grouping"])
                writer.writerow(["#" + self.video_file, self.participant, self.info, self.grouping])
            writer.writerow(["aoi_name", "count"])
            for aoi, count in self.data.items():
                writer.writerow([aoi, count])
        return file_path


class FixationCounts:
    """
    Saves the number of fixations per AOI for multiple scanpaths.
    """
    def __init__(self):
        self.videos = {}
        self.data = []
        self.has_info = False
        self.average_value = None

    file_suffix = "fixation_counts"

    def append_count(self, counts, video_file, participant="0", source="data", info=None, grouping="unknown"):
        """
        Creates a FixationCount object and appends it to data.
        """
        if (info is not None):
            self.has_info = True
        fix_count = FixationCount(counts, video_file, participant, source, info, grouping)
        if (fix_count.video in self.videos):
            self.videos[fix_count.video].append(len(self.data))
        else:
            self.videos[fix_count.video] = [len(self.data)]
        self.data.append(fix_count)

    def append_data(self, fix_count):
        """
        Appends a given FixationCount object to data.
        """
        if (fix_count.info is not None):
            self.has_info = True
        if (fix_count.video in self.videos):
            self.videos[fix_count.video].append(len(self.data))
        else:
            self.videos[fix_count.video] = [len(self.data)]
        self.data.append(fix_count)

    def average(self):
        """
        Calculates the average of the fixationcount data.
        """
        counts = {}
        for d in self.data:
            for k,v in d.data.items():
                if (k in counts):
                    counts[k] += v
                else:
                    counts[k] = v
        for k in counts.keys():
            counts[k] = counts[k] / len(self.data)
        self.average_value = FixationCount(counts, "average", "average", "average", "average", "average")

    def get_video_data(self, video):
        """
        Returns all the data for a given video.
        """
        video_name = os.path.splitext(os.path.split(video)[1])[0]
        if (video_name not in self.videos):
            return []
        out_data = []
        for i in self.videos[video_name]:
            out_data.append(self.data[i])
        return out_data

    def save(self, dir="statistics/", delimiter=","):
        """
        Saves the fixation counts to a csv file in the given directory.
        """
        if (not os.path.exists(dir)):
            os.mkdir(dir)
        # find all aoi names
        aoi_names = set()
        grouping_names = set()
        for data in self.data:
            if (data.grouping not in grouping_names):
                if (data.grouping in config.AOI_GROUPINGS):
                    grouping_names.add(data.grouping)
                    for k in config.AOI_GROUPINGS[data.grouping].keys():
                        aoi_names.add(k)
                else:
                    for aoi in data.data.keys():
                        aoi_names.add(k)

        if (self.has_info):
            header = ["participant", "video", "info", "grouping"] + list(aoi_names)
        else:
            header = ["participant", "video", "grouping"] + list(aoi_names)
        # format data
        rows = []
        full_data = self.data
        full_data.append(self.average_value)
        for data in full_data:
            if (self.has_info):
                if (data.info is not None):
                    row = [data.participant, data.video, data.info, data.grouping]
                else:
                    row = [data.participant, data.video, "", data.grouping]
            else:
                row = [data.participant, data.video, data.grouping]

            for aoi in aoi_names:
                if (aoi in data.data):
                    row.append(data.data[aoi])
                else:
                    row.append(0)
            rows.append(row)
        # save data to file
        file_name = "{}_{}_{}.csv".format(self.data[0].grouping, len(self.data), self.file_suffix)
        file_path = os.path.join(dir, file_name)
        with open(file_path, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            writer.writerow(header)
            writer.writerows(rows)
        return [file_name]


class Dwelltimes(FixationCounts):
    """
    Similar to FixationCounts but saves the dwelltime in ms per aoi instead of
    the number of frames.
    """
    def __init__(self, fix_count):
        self.videos = fix_count.videos
        self.data = fix_count.data
        self.has_info = fix_count.has_info
        self.average_value = fix_count.average_value
        self.file_suffix = "dwelltime"

    def save(self, dir="statistics/", delimiter=","):
        """
        Saves the dwelltimes to a csv file in the given directory.
        """
        if (not os.path.exists(dir)):
            os.mkdir(dir)
        # find all aoi names
        aoi_names = set()
        for data in self.data:
            for aoi in data.data:
                aoi_names.add(aoi)
        if (self.has_info):
            header = ["participant", "video", "info", "grouping"] + list(aoi_names)
        else:
            header = ["participant", "video", "grouping"] + list(aoi_names)
        # format data
        rows = []
        full_data = self.data
        full_data.append(self.average_value)
        for data in full_data:
            if (data.info is not None):
                row = [data.participant, data.video, data.info, data.grouping]
            else:
                row = [data.participant, data.video, "", data.grouping]
            if (not os.path.exists(data.video_file)):
                print("Video: \'{}\' not found.".format(data.video_file))
                frame_time = 1000 / config.VIDEO_DEFAULT_FPS
            else:
                frame_time = 1000 / utilities.get_fps(data.video_file)
            for aoi in aoi_names:
                if (aoi in data.data):
                    row.append(int(data.data[aoi] * frame_time))
                else:
                    row.append(0)
            rows.append(row)
        # save data to file
        file_name = "{}_{}_{}.csv".format(self.data[0].grouping, len(self.data), self.file_suffix)
        file_path = os.path.join(dir, file_name)
        with open(file_path, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            writer.writerow(header)
            writer.writerows(rows)
        return [file_name]


class AOISaccades(FixationCount):
    def save(self, dir="statistics/", delimiter=","):
        """
        Saves the content to a file in the given directory.
        Args:
            dir : path
                The directory where the data is saved to.
            delimiter : char
                The delimiter to use in the produced files.
        """
        file_name = "{}_{}_{}_saccade_count.csv".format(self.source, self.video, self.participant)
        file_path = os.path.join(dir, file_name)
        with open(file_path, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            if (self.info is not None):
                writer.writerow(["#video", "participant", "source", "info", "grouping"])
                writer.writerow(["#" + self.video_file, self.participant, self.source, self.info, self.grouping])
            else:
                writer.writerow(["#video", "participant", "source", "grouping"])
                writer.writerow(["#" + self.video_file, self.participant, self.source, self.grouping])
            writer.writerow(["aoi_start", "aoi_end", "count"])
            for aoi, count in self.data.items():
                writer.writerow([aoi[0], aoi[1], count])
        return file_path


class AOISaccadesData(FixationCounts):
    file_suffix = "saccade_counts"

    def append_count(self, counts, video_file, participant="0", source="data", info=None, grouping="unknown"):
        """
        Creates a AOISaccades object and appends it to data.
        """
        saccades_count = AOISaccades(counts, video_file, participant, source, info, grouping)
        if (saccades_count.video in self.videos):
            self.videos[saccades_count.video].append(len(self.data))
        else:
            self.videos[saccades_count.video] = [len(self.data)]
        self.data.append(saccades_count)

    def save(self, dir="statistics/", delimiter=","):
        return FixationCounts.save(self, dir, delimiter)
