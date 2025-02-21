import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt

class video_reader:
    """
    class written to get frame from video source.
    wrote this mainly to be able to get specific frames from camera and to have a nice wrapper around cv2.VideoCapture.

    Args:
        - video_source (Path | str | int): Source for video
    """

    def __init__(self, video_source: Path | str | int=0):
        self.video = cv2.VideoCapture(video_source)
        if not self.video.isOpened():
            raise RuntimeError("Unable to read frames from video...")

        self.is_live = isinstance(video_source, int)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else None
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.video.read()
        if not ret:
            print("Unable to read frames from video")
            self.video.release()
            raise StopIteration
        
        return frame

    def __len__(self) -> int:
        return int(self.total_frames) # initially float type

    @property
    def frame_number(self) -> int:
        return self.video.get(cv2.CAP_PROP_POS_FRAMES)

    def get_frame(self, index: int) -> np.ndarray:
        # raise error if trying to get specific frame from live camera
        if self.is_live:
            raise RuntimeError("Running from live feed..., unable to retrieve specific frames")
        
        # raise error if provided error is out of range
        if (index < 0) or (index >= self.total_frames):
            raise IndexError(f"Frame index {index} out of range (0 to {self.total_frames-1}).")

        # move to the specific frame
        self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.video.read()
        if not ret:
            raise RuntimeError("Failed to retrieve the requested frame.")
        
        # reset index, back to the start. But this can be useful to set iteration start
        # self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame


def image_preprocessing(frame: np.ndarray) -> np.ndarray:
    # reduce resolution by half
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # updates frame to have the right color space (blue gloves not orange)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
    return resized_frame


def houghcircle_detections(frame: np.ndarray) -> Tuple[np.ndarray, Optional[List[float]]]:
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # circles is a list of (x, y, r) extra dims added
    circles = cv2.HoughCircles(greyscale_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=72,
                                param1=15, param2=20, minRadius=34, maxRadius=36)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    return frame, circles

# TODO: Generic function to draw bounding box using detection from the different detectors
def draw_bounding_box():
    pass

"""
Different opencv detectors to consider
- cv2.HoughCircles (with min/max radius), with edge detection. key parameters
    - min/max radius (35px with +-1 error)
    - minDist between center coordinate is diameter of max of radius
    
    - dp: Inverse ratio of the accumulator resolution
    (high value (> 1.0) -> much faster but may miss smaller circle, lower,
    lower value -> more precise but can be computationally expensive)

    - param1: Canny edge detection threshold
    (high value -> detects fewer edges, while lower value detects more edges)

    - param2: Accumulator threshold for circle detection.
    (high value -> fewer but more accurate circle,
    low value -> more circle detected but possibly more false positives)

- cv2 colour based segmentation (colour thresholding to detect the gloves in the image.
- cv2 template matcher or possible other crafted features to detect the bottle of chemical.

"""

video = video_reader(r"AICandidateTest-FINAL.mp4")
frame = video.get_frame(1875) # also sets the starting frame

# iterate through the video
for frame in video:
    # preprocess the individual frame
    frame = image_preprocessing(frame)

    # apply houghcircle detection function
    frame, _ = houghcircle_detections(frame)

    # display the frame with drawn circle
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break