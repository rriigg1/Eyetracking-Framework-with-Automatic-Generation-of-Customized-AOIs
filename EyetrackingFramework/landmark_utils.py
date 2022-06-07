import cv2
from tqdm import tqdm
import dlib
import os
import threading
from multiprocessing import Process, Pipe
import time

# framework imports
import config


def get_default_landmarks(img_size=config.DEFAULT_LANDMARK_SIZE):
    """
    Gets the default landmarks from the config and maps them to the desired image size.
    """
    return list(map(lambda l: (l[0] * (img_size[0] / config.DEFAULT_LANDMARK_SIZE[0]), l[1] * (img_size[1] / config.DEFAULT_LANDMARK_SIZE[1])), config.DEFAULT_LANDMARKS))

def get_landmarks(image):
    """
    For a given image in form of a numpy array returns the landmarks of the
    face present in the image.
    """
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR)
    except Exception as e:
        print("Unable to open the trained shape predictor model. Make sure the "
        + "trained model is located at:\n\t\'{}\'\n Or update the path in the config file.".format(os.path.abspath(config.SHAPE_PREDICTOR)))
        exit(1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image, 0)
    if (len(dets) < 1):
        print("No face was found.")
        return None
    if (len(dets) > 1):
        print("More than one face was found.")
    shape = predictor(image, dets[0])
    return [(p.x, p.y) for p in shape.parts()]


def generate_landmarks_simple(predictor, image, face):
    """
    Simple landmark generator that returns a list of coordinates as 2d tuples.
    """
    shape = predictor(image, face)
    return [(p.x, p.y) for p in shape.parts()]


def generate_landmarks_and_background(predictor, image, face, background_landmarks=[]):
    """
    Generates landmarks from a given image and appends the background_landmarks
    to the list of coordinates.
    """
    shape = predictor(image, face)
    landmarks = [(p.x, p.y) for p in shape.parts()]
    landmarks += background_landmarks
    return landmarks


class LandmarkGeneratorThread(Process):
    """
    Generates landmarks for a given video using the given generator.
    Attributes:
        video : str
            Name of the video this thead processes.
        generator : function
            Function that is used to generate the AOIs.
        result : list
            A list contaning the generated landmarks after the thread has finished.
        file : str
            The path to the video file.
    """
    def __init__(self, video, generator, pipe_connector=None):
        Process.__init__(self)
        self.file = video
        self.video = os.path.splitext(os.path.split(video)[1])[0]
        self.generator = generator
        self.connector = pipe_connector
        self.result = None

    def get_result(self):
        return self.result

    def run(self):
        self.generate_landmarks()

    def generate_landmarks(self):
        video_capture = cv2.VideoCapture(self.file)
        detector = dlib.get_frontal_face_detector()
        try:
            predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR)
        except Exception as e:
            print("Unable to open the trained shape predictor model. Make sure the "
            + "trained model is located at:\n\t\'{}\'\n Or update the path in the config file.".format(os.path.abspath(config.SHAPE_PREDICTOR)))
            exit(1)
        target = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        landmark_list = []
        current = 0
        success = True
        image = None
        if (self.connector is not None):
            self.connector.send((current, target))

        face_warning_printed = False
        no_landmarks_frames = 0
        faulty_frames = 0
        frame_count = 0

        while current < target:
            success, image = video_capture.read()
            current += 1
            if (not success):
                landmark_list.append(None)
                faulty_frames += 1
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, current)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets = detector(image, 0)
            if dets:
                if (len(dets) > 1 and not face_warning_printed):
                    print(f"Multiple faces in one video currently not supported ({self.file}). Using only first face of {len(dets)}.")
                    face_warning_printed = True
                d = dets[0]
                landmarks = self.generator(predictor, image, d)
                landmark_list.append(landmarks)
            else:
                no_landmarks_frames += 1
                landmark_list.append(None)
            if (self.connector is not None and current%10==0):
                self.connector.send((current, target))
        if (no_landmarks_frames > 0):
            print(f"Failed to find landmarks for {no_landmarks_frames} of {target} frames for video \'{self.file}\'")
        if (faulty_frames > 0):
            print(f"Failed to read {faulty_frames} frames of video \'{self.file}\'")
        self.connector.send((self.file, landmark_list))


class Observer(threading.Thread):
    """
    Collects the data from a list of pipes and maintains a progress bar that
    shows the combined progress of all processes.
    """
    def __init__(self, threads, connectors):
        threading.Thread.__init__(self)
        self.threads = threads
        self.cons = connectors
        self.results = []
        self.total_elements = [1 for i in range(len(self.cons))]
        self.current_elements = [1 for i in range(len(self.cons))]
        self.bar = tqdm(total=sum(self.total_elements), leave=True, position=0)
        self.update()

    def update(self):
        fin = True
        for i, c in enumerate(self.cons):
            while (self.threads[i].is_alive() and c.poll(1)):
                fin = False
                msg = c.recv()
                if (isinstance(msg, tuple) and isinstance(msg[0], str)):
                    self.results.append(msg)
                else:
                    self.total_elements[i] = msg[1]
                    self.current_elements[i] = msg[0]

        self.bar.total = sum(self.total_elements)
        self.bar.n = sum(self.current_elements)
        self.bar.refresh()
        return fin

    def run(self):
        fin = False
        while (not fin):
            fin = self.update()
        for t in self.threads:
            t.join()
        self.bar.close()


def generate_all_landmarks(videos, generator):
    """
    For each video in the given list starts a thread and
    generates landmarks using the given generator.
    ### Args:
        videos : list
            A list of videos to generate landmarks for.
        generator : function
            Function used to generate the landmarks.
    ### Returns:
        observer : Observer
        An observer that can be used to monitor the progress of the landmark creation.
        The join() function can be used to wait for all threads to finish.
    """
    thread_list = []
    connectors = []
    print(f"Generating landmarks for {len(videos)} videos.")
    for video in videos:
        # create thread and pipe to send status and result
        parrent_con, child_con = Pipe()
        vid_thread = LandmarkGeneratorThread(video, generator, pipe_connector=child_con)
        vid_thread.start()
        connectors.append(parrent_con)
        thread_list.append(vid_thread)
    # the observer controls the threads and collects the status data and results
    observer = Observer(thread_list, connectors)
    observer.start()
    return observer
