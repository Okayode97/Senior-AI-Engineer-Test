import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import cv2
from collections import defaultdict
import torchvision.transforms as transforms

class LabDataset(Dataset):
    def __init__(self, annotation: str | Path, preprocessing_func=None):        
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
        image = cv2.imread(image_dict["file_name"])

        transform = transforms.ToTensor()
        image = transform(image)
        image /= 255

        if image is None:
            raise ValueError("Image read as none")
        image_id =  image_dict["id"]

        # get associated annotations
        annotation_list = [self.annotations["annotations"][idx] for idx in self.image_id_to_annotation[image_id]]

        boxes = []
        labels = []
        for annotation in annotation_list:
            # convert bounding box format x,y,w,h to x,y,xy
            x_min, y_min, width, height = annotation['bbox']
            x_max = x_min + width
            y_max = y_min + height
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
            image, target = self.transforms(image, target)
        return image, target
