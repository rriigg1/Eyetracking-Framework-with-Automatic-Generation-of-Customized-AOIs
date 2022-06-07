import csv
import os


class ScanpathDistances(list):
    """
    For a base scanpath and multiple other scanpaths saves the results of a analysis.
    This class is basicly a wrapper for a list, but allows to save the data by calling save().
    """

    def __init__(self, distances, base_video, participant, info="real", analysis="scanpath"):
        list.__init__(self, distances)
        self.video = base_video
        self.video_name = os.path.splitext(os.path.split(base_video)[1])[0]
        self.participant = participant
        self.analysis = analysis
        self.info = info

    def save(self, dir="analysis/", delimiter=","):
        """
        Saves the list of distances to a csv file in the given directory.
        """
        if (not os.path.exists(dir)):
            os.mkdir(dir)
        filename = os.path.join(dir, "{}_{}_{}_distance.csv".format(self.video_name, self.participant, self.analysis))
        with open(filename, "w", newline="") as fh:
            csvwr = csv.writer(fh, delimiter=delimiter)
            csvwr.writerow(["#video", "participant", "info", "type"])
            csvwr.writerow(["#" + self.video, self.participant, self.info, self.analysis])
            csvwr.writerow(["video", "participant", "info", "distance"])
            for dist in self:
                csvwr.writerow(list(dist))
        return filename

    def get_distribution(self):
        """
        Returns the average of the distances and the standard error of the mean.
        """
        mean = sum(d[-1] for d in self) / len(self)
        variance = sum((d[-1]-mean)**2 for d in self) / len(self)
        std_err = (variance**0.5) / (len(self)**0.5)
        return (mean, std_err)

    def __add__(self, x):
        # make sure video and participant are equal in order to concatenate the data.
        if (isinstance(x, ScanpathDistances)):
            if (x.video != self.video or x.participant != self.participant or self.analysis != x.analysis):
                raise ValueError("Video and participant need to be the same to concatenate the distances.")
                return
            else:
                return list.__add__(self, x)
        else:
            return list.__add__(self, x)


class AOIScanpathDistances(ScanpathDistances):
    """
    For a base series of fixated AOIs and multiple other series saves the results of a analysis.
    This class is basicly a wrapper for a list, but allows to save the data by calling save().
    """

    def __init__(self, distances, base_video, participant, info="real", grouping="DEFAULT", analysis="scanpath"):
        list.__init__(self, distances)
        self.video = base_video
        self.video_name = os.path.splitext(os.path.split(base_video)[1])[0]
        self.participant = participant
        self.analysis = analysis
        self.info = info
        self.grouping = grouping

    def save(self, dir="analysis/", delimiter=","):
        """
        Saves the list of distances to a csv file in the given directory.
        """
        if (not os.path.exists(dir)):
            os.mkdir(dir)
        filename = os.path.join(dir, "{}_{}_{}_{}_distance.csv".format(self.video_name, self.participant, self.grouping, self.analysis))
        with open(filename, "w", newline="") as fh:
            csvwr = csv.writer(fh, delimiter=delimiter)
            csvwr.writerow(["#video", "participant", "info", "grouping", "type"])
            csvwr.writerow(["#" + self.video, self.participant, self.info, self.grouping, self.analysis])
            csvwr.writerow(["video", "participant", "info", "grouping", "distance"])
            for dist in self:
                csvwr.writerow(list(dist))
        return filename

    def __add__(self, x):
        # make sure video, participant and grouping are equal in order to concatenate the data.
        if (isinstance(x, AOIScanpathDistances)):
            if (x.video != self.video or x.participant != self.participant or self.analysis != x.analysis or self.grouping != x.grouping):
                raise ValueError("Video, participant and grouping need to be the same to concatenate the distances.")
                return
            else:
                return list.__add__(self, x)
        else:
            return list.__add__(self, x)
