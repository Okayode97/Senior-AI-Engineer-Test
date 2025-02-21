import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass


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
            print("Unable to read frames from video....")
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


@dataclass
class bbox:
# bbox in the form xyhw
    x: int
    y: int
    height: int
    width: int

    def draw_bbox_onto_frame(self, frame: np.ndarray) -> np.ndarray:
        # rectangle drawn using top left cornor to bottom right cornor
        # color in BGR
        frame = cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), (0, 255, 0), 2)
        return frame

def image_preprocessing(frame: np.ndarray) -> np.ndarray:
    # reduce resolution by half
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # updates frame to have the right color space (blue gloves not orange)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
    return resized_frame


def get_minimum_bbox_that_fits_each_circle(hough_circles: np.ndarray) -> List[bbox]:
    list_of_bbox = []
    for circles in hough_circles:
        x = circles[0] - circles[2]
        y = circles[1] - circles[2]
        h = circles[2] *2
        w = circles[2] *2
        list_of_bbox.append(bbox(x, y, h, w))
    return list_of_bbox
    

def houghcircle_detections(frame: np.ndarray) -> List[bbox]:
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # circles is a list of (x, y, r) extra dims added
    circles = cv2.HoughCircles(greyscale_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=72,
                                param1=15, param2=20, minRadius=34, maxRadius=36)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = np.rint(circles).astype(int) # round to nearest int
        return get_minimum_bbox_that_fits_each_circle(circles[0]) #indexed to account for first dim
    return []


def draw_bounding_box(frame: np.ndarray, list_of_bbox: List[bbox]) -> np.ndarray:
    for box in list_of_bbox:
        # draw bounding box onto frame
        frame = box.draw_bbox_onto_frame(frame)
    return frame


def colour_based_segmentation(frame: np.ndarray, min_contour_area: int = 500) -> List[bbox]:
    # convert to hsv colour space
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define hsv range for blue
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # apply color thresholding onto the individual frames in the video
    # with the lower and upper range of hsv values.
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # find the contours within the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours by area, fit a bounding box onto it and append it to list
    list_of_bbox: list[bbox] = []
    for contour in contours:
        # filter based on area
        if cv2.contourArea(contour) > min_contour_area:

            # fit a bounding box onto the contour
            x, y, w, h = cv2.boundingRect(contour)
            list_of_bbox.append(bbox(x, y, h, w))
    return list_of_bbox


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
    -
- Possibly other crafted features to detect the bottle of chemical.

"""

video = video_reader(r"AICandidateTest-FINAL.mp4")
frame = video.get_frame(2678) # also sets the starting frame

for frame in video:
    frame = image_preprocessing(frame)

    list_of_houghcircle_bbox = houghcircle_detections(frame)
    list_of_color_based_bbox = colour_based_segmentation(frame, 500)

    frame = draw_bounding_box(frame, list_of_color_based_bbox+list_of_houghcircle_bbox)

    # Display result
    cv2.imshow("Lab technician object detection and tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
