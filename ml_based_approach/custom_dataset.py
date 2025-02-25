import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import cv2
import os
from collections import defaultdict
import torchvision.transforms as transforms
import numpy as np

class LabDataset(Dataset):
    def __init__(self, dataset_folder: str | Path, annotation: str | Path, preprocessing_func=None):        
        self.dataset_folder = dataset_folder
        self.annotations = self.load_json_data(annotation)
        self.preprocess_func = preprocessing_func
        self.image_id_to_annotation = self.get_annotations_for_image_id()

    def load_json_data(self, annotation_path: str | Path):
        with open(annotation_path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.annotations["images"])

    def get_annotations_for_image_id(self):
        data = defaultdict(list)
        for idx, annotation in enumerate(self.annotations["annotations"]):
            data[annotation["image_id"]].append(idx)
        return data

    def __getitem__(self, idx):
        # get the data associated with the idx
        image_dict = self.annotations["images"][idx]
        img_path = os.path.join(self.dataset_folder, image_dict["file_name"])
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError("Image read as none")
        image_id =  image_dict["id"]

        # get associated annotations
        annotation_list = [self.annotations["annotations"][idx] for idx in self.image_id_to_annotation[image_id]]

        boxes = []
        labels = []
        for annotation in annotation_list:
            # convert bounding box format x,y,w,h to x,y,xy
            x_min, y_min, height, width = annotation['bbox']
            x_max = x_min + height
            y_max = y_min + width
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(annotation['category_id'])

        # convert lists to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros(len(labels), dtype=torch.int64)
        }

        # if there's any additional preprocessing apply it to the labels
        if self.preprocess_func:
            image = self.preprocess_func(image)
        return image, target

def image_preprocessing(frame: np.ndarray) -> torch.tensor:    
    # add color dimension first, normalise range and expand dims to include batch dimension
    frame = frame.transpose((2, 0, 1))
    frame = frame / 255
    return torch.from_numpy(frame).float() # set type to float