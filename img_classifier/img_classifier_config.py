import os
from torchvision import transforms

BASE_DATA_PATH = r"D:\datasetbank\data"
TRAIN_PATH = os.path.join(BASE_DATA_PATH, "train")
VALID_PATH = os.path.join(BASE_DATA_PATH, "val")
TEST_PATH = os.path.join(BASE_DATA_PATH, "test")

MODEL_SAVE_PATH = "model.pth"

IMAGE_SIZE = (256, 256)
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)

BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VALID = 11
BATCH_SIZE_TEST = 10
LEARNING_RATE = 0.003
NUM_EPOCH = 4

MODEL_TYPE = "M_EFNETB0"
# Available model : M_EFNETB0, RESNET18, CNN_MOD

NUM_CLASSES = None
# Set by default to use len(train_dataset.classes) 
