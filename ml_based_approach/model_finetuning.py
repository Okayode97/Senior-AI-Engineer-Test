# reference
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from custom_dataset import LabDataset


# training config
# possible TODO, turn this into a dataclass and experiment with changing different parameters
NUM_CLASSES = 3  # 0 - background, 1 - petri_dish, 2 gloves
BATCH_SIZE = 4
LEARNING_RATE = 0.005
EPOCHS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ANNOTATION_JSON_FILE = r"C:\Users\oluka\Desktop\Job Application 2025\Reach industries\Senior-AI-Engineer-Test\ml_based_approach\annotations.json"


# load pre-trained model and set num of clas
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(DEVICE)
    return model

# DATA TRANSFORMS (for training)
def get_transform(train=True):
    transforms_list = [transforms.ToTensor()]
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))  # Data Augmentation
    return transforms.Compose(transforms_list)

# collate function
def collate_fn(batch):
    return tuple(zip(*batch))

# training function
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()  # Set model to training mode
    total_loss = 0
    idx = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        print(f"sample: {idx}, total_loss: {total_loss}")
        idx += 1

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


def evaluate(model, data_loader, device):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets) 
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    print(f"Validation Loss: {total_loss:.4f}")

# load the dataset
dataset = LabDataset(ANNOTATION_JSON_FILE)

# divide the dataset into training, validation & test set
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size
dataset, validation_test = torch.utils.data.random_split(dataset, [train_size, validation_size])

# define training and test data loaders
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(validation_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
# get custom model
model = get_model()

# construct optimizers, learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 

# train the model and evaluate it using the validation set
# for epoch in range(EPOCHS):
train_one_epoch(model, optimizer, train_loader, DEVICE, 1)
evaluate(model, val_loader, DEVICE)
lr_scheduler.step()  # Adjust learning rate

# # evaluate the trained model using the test set and save the resulting weights
# evaluate(model, val_loader, DEVICE)
# torch.save(model.state_dict(), "faster_rcnn_custom.pth")

