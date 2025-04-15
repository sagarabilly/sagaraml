import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm

# Import the configuration file
import img_classifier_config as config


def get_version():
    import pandas as pd
    import numpy as np

    versions = {
        "sys_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    return versions


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


#  DATASET CLASS
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(root=data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def classes(self):
        return self.data.classes

    def __repr__(self):
        return f"Dataset path: {self.data.root}, Total images: {len(self.data)}"


#  MODEL ARCHITECTURES
class M_EFNETB0(nn.Module):
    def __init__(self, num_classes):
        super(M_EFNETB0, self).__init__()
        self.num_classes = num_classes
        # Load pretrained EfficientNet-B0 from timm
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.enet_out_size = 1280  # output feature size

        # Exclude the final classifier layer of the base model
        self.input_layer = nn.Sequential(*list(self.base_model.children())[:-1])
        self.hidden_layer = nn.Linear(self.enet_out_size, 320)
        self.output_layer = nn.Linear(320, self.num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(x.size(0), -1)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Classifier, self).__init__()
        # Load pre-trained ResNet18 from torchvision
        self.resnet = models.resnet18(pretrained=True)
        # Replace the final fully connected layer to match the number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
        
class CNN_MOD(nn.Module):
    def __init__(self, initial_input_size, num_classes):
        super(CNN_MOD, self).__init__()
        self.initial_input_size = initial_input_size
        self.num_classes = num_classes

        print(f"Initial input image size is: {self.initial_input_size}")

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Calculate flattened size: for 256x256 input, after three pools -> (256/2/2/2)=32
        self.flattened_size = 128 * 32 * 32

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#  TRAINER CLASS
class ModelTrainer:
    def __init__(self, model, criterion, num_epoch, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.device = get_device()

    def fit(self, train_loader, valid_loader):
        self.model.to(self.device)
        self.training_losses, self.valid_losses = [], []

        for epoch in range(self.num_epoch):
            training_loss = self.train(train_loader)
            valid_loss = self.valid(valid_loader)
            self.training_losses.append(training_loss)
            self.valid_losses.append(valid_loss)

            print(
                f"Epoch {epoch+1}/{self.num_epoch} - Train Loss: {training_loss:.4f} - Valid Loss: {valid_loss:.4f}"
            )
        print("Training Completed")
        return

    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * labels.size(0)
        return running_loss / len(train_loader.dataset)

    def valid(self, valid_loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Validation", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        return running_loss / len(valid_loader.dataset)

    def visualize_loss(self):
        plt.plot(self.training_losses, label="Training Loss")
        plt.plot(self.valid_losses, label="Validation Loss")
        plt.legend()
        plt.title("Loss over Epochs")
        plt.show()


#  PREDICTION CLASSES
class ModelPrediction:
    class SinglePredict:
        def __init__(self, model):
            self.model = model
            self.device = get_device()
            self.original_image = None
            self.image_tensor = None

        def preprocess_image(self, transform, image_path, show=False):
            self.original_image = Image.open(image_path)
            if show:
                plt.imshow(self.original_image)
                plt.axis("off")
                plt.show()
            self.image_tensor = transform(self.original_image).unsqueeze(0)

        def predict(self):
            if self.image_tensor is None:
                raise RuntimeError("Preprocess the image first.")
            self.model.eval()
            with torch.no_grad():
                self.image_tensor = self.image_tensor.to(self.device)
                output = self.model(self.image_tensor)
                probabilities = F.softmax(output, dim=1)
            return probabilities.cpu().numpy().flatten()

        def visualize_predictions(self, probabilities, class_names):
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            fig = plt.figure(figsize=(12, 10))
            ax_image = fig.add_subplot(gs[0])
            ax_image.imshow(self.original_image)
            ax_image.axis("off")
            ax_predictions = fig.add_subplot(gs[1])
            ax_predictions.barh(class_names, probabilities)
            ax_predictions.set_xlabel("Probability")
            ax_predictions.set_title("Class Prediction")
            ax_predictions.set_xlim(0, 1)
            plt.tight_layout()
            plt.show()


#  FUNCTIONALITY: COMPOSE (TRAIN) AND LOAD/PREDICT
def train_model():
    # --- Load datasets ---
    train_dataset = ImageDataset(data_dir=config.TRAIN_PATH, transform=config.transform)
    valid_dataset = ImageDataset(data_dir=config.VALID_PATH, transform=config.transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE_VALID, shuffle=False
    )

    # --- Determine number of classes ---
    num_classes = len(train_dataset.classes)
    print(f"Detected classes: {train_dataset.classes}")

    # --- Create model instance based on config ---
    if config.MODEL_TYPE == "RESNET18":
        model = ResNet18Classifier(num_classes=num_classes)
    elif config.MODEL_TYPE == "M_EFNETB0":
        model = M_EFNETB0(num_classes=num_classes)
    elif config.MODEL_TYPE == "CNN_MOD":
        model = CNN_MOD(initial_input_size=config.IMAGE_SIZE, num_classes=num_classes)
    else:
        raise ValueError("Unsupported MODEL_TYPE set in config.")

    # --- Define optimizer and loss ---
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()  # Note: includes softmax

    # --- Trainer instance ---
    trainer = ModelTrainer(model, criterion, config.NUM_EPOCH, optimizer)
    trainer.fit(train_loader, valid_loader)
    trainer.visualize_loss()

    # --- Save the trained model ---
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")
    return model, train_dataset.classes


def load_model(model_path, num_classes):
    # --- Create the model instance (it must match the trained model type) ---
    if config.MODEL_TYPE == "M_EFNETB0":
        model = M_EFNETB0(num_classes=num_classes)
    elif config.MODEL_TYPE == "CNN_MOD":
        model = CNN_MOD(initial_input_size=config.IMAGE_SIZE, num_classes=num_classes)
    else:
        raise ValueError("Unsupported MODEL_TYPE set in config.")

    model.load_state_dict(torch.load(model_path, map_location=get_device()))
    model.to(get_device())
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


def predict_image(model, image_path, class_names):
    predictor = ModelPrediction.SinglePredict(model)
    predictor.preprocess_image(
        transform=config.transform, image_path=image_path, show=True
    )
    probabilities = predictor.predict()
    print("Predicted probabilities:", probabilities)
    predictor.visualize_predictions(probabilities, class_names)


#  CLI
def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Classifier - Train or Predict")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--compose", action="store_true", help="Train and compose a new model."
    )
    group.add_argument(
        "--load",
        type=str,
        metavar="MODEL_PATH",
        help="Load a pre-trained model from the specified path.",
    )

    parser.add_argument(
        "--target",
        type=str,
        metavar="IMAGE_PATH",
        default=None,
        help="Path to the image for prediction (used with --load).",
    )
    return parser.parse_args()


#  MAIN EXECUTION
if __name__ == "__main__":
    args = parse_arguments()

    print("Version details:")
    print(get_version())
    print(f"Using device: {get_device()}")

    if args.compose:
        model, classes = train_model()
        print("Training complete.")
    elif args.load:
        temp_dataset = ImageDataset(
            data_dir=config.TRAIN_PATH, transform=config.transform
        )
        classes = temp_dataset.classes

        model = load_model(args.load, num_classes=len(classes))
        if args.target:
            predict_image(model, args.target, classes)
        else:
            print("Please provide an image path using --target to run prediction.")
