import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import time


class video_reader:
    """
    class written to get frame from video source.
    wrote this mainly to be able to get specific frames from camera and to have a nice wrapper around
    cv2.VideoCapture.

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
        
        # raise error if provided index is out of range
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

    def draw_bbox_onto_frame(self, frame: np.ndarray, colour: tuple[int]=(0, 255, 0)) -> np.ndarray:
        # rectangle drawn using top left cornor to bottom right cornor
        # color in BGR
        frame = cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height),
                colour, 2)
        return frame

def image_preprocessing(frame: np.ndarray) -> np.ndarray:
    # reduce resolution by half
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # updates frame to have the right color space (blue gloves not orange)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
    return resized_frame


def get_minimum_bbox_that_fits_each_circle(hough_circles: np.ndarray) -> list[bbox]:
    list_of_bbox = []
    for circles in hough_circles:
        x = circles[0] - circles[2]
        y = circles[1] - circles[2]
        h = circles[2] *2
        w = circles[2] *2
        list_of_bbox.append(bbox(x, y, h, w))
    return list_of_bbox
    

def houghcircle_detections(frame: np.ndarray) -> list[bbox]:
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # circles is a list of (x, y, r) extra dims added
    circles = cv2.HoughCircles(greyscale_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=72,
                                param1=15, param2=20, minRadius=34, maxRadius=36)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = np.rint(circles).astype(int) # round to nearest int
        return get_minimum_bbox_that_fits_each_circle(circles[0]) #indexed to account for first dim
    return []


def draw_bounding_box(frame: np.ndarray, list_of_bbox: list[bbox], colour: tuple[int]) -> np.ndarray:
    for box in list_of_bbox:
        # draw bounding box onto frame
        frame = box.draw_bbox_onto_frame(frame, colour)
    return frame


def colour_based_segmentation(frame: np.ndarray, min_contour_area: int = 500) -> list[bbox]:
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


def roi_selector(frame: np.ndarray):
    # contains y, x, w, h
    roi = cv2.selectROI("select ROI", frame, False)

    # if valid roi is returned write the cropped image to file
    if roi != (0, 0, 0, 0):
        cropped_image = frame[int(roi[1]):int(roi[1]+roi[3]),  
                            int(roi[0]):int(roi[0]+roi[2])] 
        
        cv2.imwrite("target_img.png", cropped_image)


def instantiate_sift_and_feature_matcher():
    # sift feature extractor
    sift = cv2.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return sift, flann


def experimenting_with_feature_matching(src_img: np.ndarray, tgt_img: np.ndarray,
                                        sift, flann) -> Optional[np.ndarray]:

    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    min_match_count = 10
    kp1, des1 = sift.detectAndCompute(src_img, None)
    kp2, des2 = sift.detectAndCompute(tgt_img, None)

    matches = flann.knnMatch(des1, des2, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    if len(good)>min_match_count:
        src_pts = np.rint([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2).astype(int)
        return src_pts


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

- cv2 colour based segmentation (colour thresholding to detect the gloves in the image).
    -
- Possibly other crafted features to detect the bottle of chemical.
    - considered using template matcher, which tries to find a template image from the live image,
      this approach works well for static image but is limited once the object starts to move.
    - experimenting with using crafted features and feature matching. Similar to the considered 
      approach with template matcher, it requires providing the target image with features to extract
      and then find and extract it from the live image.

"""

video = video_reader(r"AICandidateTest-FINAL.mp4")
frame = video.get_frame(0) # also sets the starting frame

sift, flann_matcher = instantiate_sift_and_feature_matcher()

# target image
target_img = cv2.imread("target_img.png", cv2.IMREAD_GRAYSCALE)

for frame in video:
    frame = image_preprocessing(frame)
 
    start_time = time.time()
    list_of_houghcircle_bbox = houghcircle_detections(frame)
    list_of_color_based_bbox = colour_based_segmentation(frame, 500)

    duration = time.time() - start_time
    if duration > 0:
        fps = 1/duration
        print(f"Detectors fps: {int(fps)} | frame processed: {video.frame_number}/{len(video)}",
        f"| number of detections: {len(list_of_houghcircle_bbox)+len(list_of_color_based_bbox)}")

    frame = draw_bounding_box(frame, list_of_color_based_bbox, (255, 0, 0))
    frame = draw_bounding_box(frame, list_of_houghcircle_bbox, (0, 255, 0))

    # Display result
    cv2.imshow("Lab technician object detection and tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
