import cv2
import numpy as np
from pathlib import Path
from typing import Optional

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

    def get_frame(self, index) -> np.ndarray:
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


# video = video_reader(r"C:\Users\oluka\Desktop\Job Application 2025\Reach industries\code\AICandidateTest-FINAL.mp4")
# frame = video.get_frame(1100)
# resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
# cv2.imshow("Frame", resized_frame)
# cv2.waitKey(0)


# for frame in video:
#     print(f"Frame number: {video.frame_number}")
#     resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#     cv2.imshow("Frame", resized_frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

"""
Different opencv detectors to consider
- cv2.HoughCircles (with min/max radius)

"""