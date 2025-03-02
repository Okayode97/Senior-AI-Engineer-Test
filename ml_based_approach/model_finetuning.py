# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from custom_dataset import LabDataset, train_transforms
from torchvision.utils import draw_bounding_boxes
import cv2

# training config
# possible TODO, turn this into a dataclass and experiment with changing different parameters
# further TODO: Add more training samples and extended dataset to include chemical bottle
# moved training to google colab
NUM_CLASSES = 3  # 0 - background, 1 - petri_dish, 2 gloves
BATCH_SIZE = 4
LEARNING_RATE = 0.005
EPOCHS = 40
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_WEIGHT_NAME = "faster_rcnn_custom.pth"


# load pre-trained model and set num of clas
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


# collate function
def collate_fn(batch):
    return tuple(zip(*batch))

def test_dataloader(data_loader):
    for images, targets in data_loader:
        for image, target in zip(images, targets):
            # set image to H, W, C
            boxes = target["boxes"]
            image = draw_bounding_boxes(image, boxes, width=5)

            # convert to numpy array and transpose back to color last
            image = image.detach().numpy().transpose((1, 2, 0))
            cv2.imshow("annotations", image)
            cv2.waitKey(0)

# training function
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()  # Set model to training mode
    total_loss = 0

    for images, targets in data_loader:
        optimizer.zero_grad()
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    print(f"Epoch {epoch}, Loss: {total_loss}")
    return total_loss


def evaluate(model, data_loader, device):
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # set to train mode to get loss
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            # set back to eval mode
            model.eval()
    
    print(f"Validation Loss: {total_loss}\n")
    return total_loss

# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# load the dataset
full_dataset = LabDataset(r".\.dataset_v2",
                          r".\.dataset_v2\annotations.json",
                          train_transforms)

# divide the dataset into training, validation & test set
train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size
dataset, validation_set = torch.utils.data.random_split(full_dataset, [train_size, validation_size])

# define training and test data loaders
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# test_dataloader(train_loader)

# get custom model
model = get_model().to(DEVICE)
model.train()

# construct optimizers, learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 

early_stopper = EarlyStopper(patience=4, min_delta=0.005)

# train the model and evaluate it using the validation set
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
    val_loss = evaluate(model, val_loader, DEVICE)
    lr_scheduler.step()  # Adjust learning rate

    if early_stopper.early_stop(val_loss):
        break

torch.save(model.state_dict(), MODEL_WEIGHT_NAME)
