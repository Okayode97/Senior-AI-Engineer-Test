import torch
import torchvision
import numpy as np
import cv2
import time
from pathlib import Path
import torchvision.transforms as transforms
from torch.nn import Module
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


NUM_CLASSES = 4  # 0 - background, 1 - petri_dish, 2 gloves, 3 chemical bottle
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODEL_WEIGHT = r".\.dataset_v2\faster_rcnn_custom_4.pth"


class video_reader:
    """
    class written to get frame from video source.
    wrote this mainly to be able to get specific frames from camera and to have a nice wrapper around
    cv2.VideoCapture.

    Args:
        - video_source (Path | str | int): Source for video
    """

    def __init__(self, video_source: Path | str | int = 0):
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
        return int(self.total_frames)  # initially float type

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

        return frame


def image_preprocessing(frame: np.ndarray) -> torch.tensor:
    # reduce resolution by half
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # updates frame to have the right color space (blue gloves not orange)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # add color dimension first, normalise range and expand dims to include batch dimension
    frame = frame.astype(np.float32)
    frame = frame / 255
    frame = frame.transpose((2, 0, 1))

    torch_tensor = torch.from_numpy(frame)
    torch_tensor = torch.unsqueeze(torch_tensor, 0)
    return torch_tensor  # set type to float


# load pre-trained model and set num of clas
def get_model() -> torchvision.models.detection.faster_rcnn.FasterRCNN:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def top_n_detection(predictions: dict[torch.Tensor], confidence: float = 0.5) -> list[dict]:

    # select n predictions with highest score
    scores = predictions["scores"].detach().numpy()
    bboxes = predictions["boxes"].detach().numpy()
    labels = predictions["labels"].detach().numpy()

    # selected detections
    selected_detections = []

    # get predictions with scores greater than confidence
    idxs = np.argwhere(scores > confidence)
    selected_scores = scores[idxs]
    selected_boxs = bboxes[idxs]
    selected_labels = labels[idxs]

    return [
        {"score": score, "box": box, "label": label}
        for score, box, label in zip(selected_scores, selected_boxs, selected_labels)
    ]


label_to_color = {1: (0, 255, 0), 2: (255, 0, 0)}

# define the model
model = get_model().to(DEVICE)

# load the model weights and set it to evaluation mode
model.load_state_dict(torch.load(MODEL_WEIGHT, map_location=DEVICE, weights_only=True))
model.eval()

# instaniate video reader
video = video_reader(r"AICandidateTest-FINAL.mp4")
_ = video.get_frame(1000)

for frame in video:
    if video.frame_number % 30 == 0:
        # preprocess each frame
        torch_tensor = image_preprocessing(frame)

        start_time = time.time()
        prediction = model(torch_tensor)[0]  # account for batch dim
        duration = time.time() - start_time

        if duration > 0:
            fps = 1 / duration
            print(f"Model fps: {fps} | frame processed: {video.frame_number}/{len(video)}")

            frame = cv2.putText(
                frame, f"fps: {round(fps, 2)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )

        # get predictions with score greater than given confidence
        selected_detections = top_n_detection(prediction, 0.7)

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for detection in selected_detections:
            box = detection["box"][0]
            label = detection["label"][0]
            score = detection["score"]

            frame = cv2.rectangle(
                frame, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), label_to_color[label], 2
            )

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
