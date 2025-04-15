import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from torchinfo import summary

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from PIL import Image

import sys
import os
from tqdm import tqdm
from argparse import ArgumentParser


def get_version():
    versions = {
        "sys_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    return versions


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def custom_csv(csv_path):
    df_ori = pd.read_csv(csv_path)
    return df_ori


def data_split(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

# Helper function to load data from a folder.
# <folder>/
#   ├── images/   --> input images
#   └── masks/    --> corresponding segmentation masks
def load_data_from_folder(data_folder):
    images_folder = os.path.join(data_folder, "images")
    masks_folder = os.path.join(data_folder, "masks")

    if not os.path.isdir(images_folder) or not os.path.isdir(masks_folder):
        raise FileNotFoundError(
            "Subfolders 'images' and 'masks' must exist inside the specified folder."
        )

    image_files = sorted(
        [
            f for f in os.listdir(images_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    mask_files = sorted(
        [
            f for f in os.listdir(masks_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if len(image_files) != len(mask_files):
        print(
            "Warning: The number of images and masks do not match. Please verify your folder contents."
        )

    df = pd.DataFrame(
        {
            0: [os.path.join(images_folder, f) for f in image_files],
            1: [os.path.join(masks_folder, f) for f in mask_files],
        }
    )
    return df


# Dataset Preparation
class ImageDataset(Dataset):
    def __init__(self, df, transform=None, binary_processing=True):
        self.data_frame = df
        self.transform = transform
        self.binary_processing = binary_processing

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        image = Image.open(img_path).convert("L")
        target_path = self.data_frame.iloc[idx, 1]
        target = Image.open(target_path).convert("RGB")

        if self.binary_processing:
            target = self.process_target_binary(target)
        else:
            target = self.process_target(target)

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

    def process_target(self, target):
        r, g, b = target.split()
        black = Image.new("L", target.size)
        target_red = Image.merge("RGB", (r, black, black))
        target_gray = target_red.convert("L")
        return target_gray

    def process_target_binary(self, target):
        r, g, b = target.split()
        red_threshold = 100
        mask = (
            (np.array(r) > red_threshold)
            & (np.array(g) < red_threshold)
            & (np.array(b) < red_threshold)
        )
        target_mask = Image.new("L", target.size)
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255))
        target_mask.paste(255, mask=mask_image)
        return target_mask


def testing_loader(loader, data_index):
    test_image = None
    test_target = None

    for index, (image, target) in enumerate(loader):
        if index == data_index:
            test_target = target
            test_image = image
            break

    if test_image is not None and test_target is not None:
        test_target = test_target[0].squeeze(0)
        test_image = test_image[0].squeeze(0)

        plt.figure(figsize=(12, 8), dpi=300)
        plt.subplot(1, 2, 1)
        plt.imshow(test_image, cmap="gray")
        plt.title("Test Image")
        plt.subplot(1, 2, 2)
        plt.imshow(test_target, cmap="gray")
        plt.title("Test Target")
        plt.show()
    else:
        print(f"No data found at index {data_index}.")

    return print("Image View Done...")


# =====================================================================================
# ----------------------------------------MODELING-------------------------------------
# =====================================================================================


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same", bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1x1conv or in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)
        out += identity
        return self.relu(out)


class ResidualUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(ResidualUNet2D, self).__init__()
        features = init_features

        # Encoder
        self.encoder1 = ResidualBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ResidualBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock(features * 8, features * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = ResidualBlock((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ResidualBlock((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ResidualBlock((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = ResidualBlock(features * 2, features)

        # Output
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = torch.sigmoid(self.conv(dec1))
        return output


# =====================================================================================
# ----------------------------------------TRAINING-------------------------------------
# =====================================================================================


class ModelTrainer:
    def __init__(self, model, criterion, num_epoch, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.device = get_device()

    def fit(self, train_loader, valid_loader=None, only_train=False):
        self.model.to(self.device)
        self.training_losses, self.valid_losses = [], []

        for epoch in range(self.num_epoch):
            training_loss = self.train(train_loader)
            self.training_losses.append(training_loss)

            if valid_loader is not None:
                valid_loss = self.valid(valid_loader)
                self.valid_losses.append(valid_loss)
            else:
                valid_loss = "NA"
                self.valid_losses.append(valid_loss)

            print(
                f"Epoch {epoch}/{self.num_epoch}, Train_loss: {training_loss}, Valid_loss: {valid_loss}"
            )

        return print("Training Procedure Completed")

    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for image, target in tqdm(
            train_loader, desc="Processing Model Training...", total=len(train_loader)
        ):
            image, target = image.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * target.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        return avg_loss

    def valid(self, valid_loader):
        self.model.eval()
        running_valid_loss = 0.0

        with torch.no_grad():
            for image, target in tqdm(
                valid_loader, desc="Processing Validation...", total=len(valid_loader)
            ):
                image, target = image.to(self.device), target.to(self.device)
                output = self.model(image)
                loss = self.criterion(output, target)
                running_valid_loss += loss.item() * target.size(0)
            avg_valid_loss = running_valid_loss / len(valid_loader.dataset)
        return avg_valid_loss

    def visualize_loss(self):
        plt.plot(self.training_losses, label="Training Loss")
        plt.plot(self.valid_losses, label="Valid Loss")
        plt.legend()
        plt.title("Loss over epochs")
        plt.show()
        return


# =====================================================================================
# -------------------------------------Prediction--------------------------------------
# =====================================================================================


class ModelPrediction:
    class SinglePredict:
        def __init__(self, model):
            self.model = model
            self.original_image = None
            self.image_tensor = None
            self.device = get_device()

        def preprocess_image(self, transform, image_path, show=False):
            self.original_image = Image.open(image_path).convert("L")
            if show:
                plt.imshow(self.original_image, cmap="gray")
            self.image_tensor = transform(self.original_image).unsqueeze(0)

        def predict(self):
            if self.image_tensor is None:
                raise RuntimeError("Input image must be preprocessed first.")
            self.model.eval()
            with torch.no_grad():
                self.image_tensor = self.image_tensor.to(self.device)
                output = self.model(self.image_tensor)
                output = output.cpu().numpy()
            return self.original_image, output.squeeze()

        @staticmethod
        def visualize_prediction(original_image, prediction):
            print("Visualizing Prediction...")
            plt.figure(figsize=(14, 10), dpi=300)
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap="gray")
            plt.title("Input Image")
            plt.subplot(1, 2, 2)
            plt.imshow(prediction, cmap="gray")
            plt.title("Segmentation Result")
            plt.show()


def test_single_predict(test_path, model, binary_threshold=True):
    print("Predicting...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    singlepredict = ModelPrediction.SinglePredict(model)
    singlepredict.preprocess_image(transform=transform, image_path=test_path)
    original_image, predicted_image = singlepredict.predict()
    if binary_threshold:
        predicted_image = (predicted_image > 0.5).astype(np.float32)
    singlepredict.visualize_prediction(original_image, predicted_image)
    return predicted_image


def image_analysis(binary_image):
    white_pixel_count = np.sum(binary_image)
    black_pixel_count = np.sum(binary_image == 0)
    total_pixel_count = binary_image.size
    white_percentage = (white_pixel_count / total_pixel_count) * 100
    black_percentage = (black_pixel_count / total_pixel_count) * 100
    print(f"Black Pixel Percentage (Density): {black_percentage:.2f}%")
    print(f"White Pixel Percentage (Porosity): {white_percentage:.2f}%")
    return


def get_argument():
    parser = ArgumentParser(
        description="Fast Residual UNET2D Segmentation Model. Choose Compose (train) or Load (inference)."
    )
    parser.add_argument(
        "-l", "--load", type=str, help="Load an existing model by passing model path"
    )
    parser.add_argument(
        "-c",
        "--compose",
        type=str,
        help="Compose and train a new model by passing model name",
    )
    parser.add_argument(
        "-t", "--target", type=str, help="Target image path for inference"
    )
    parser.add_argument(
        "-f",
        "--folder",
        action="store_true",
        help="Load training data from folder instead of CSV",
    )
    # New flag --valid to indicate that the user wants to load a validation dataset (only applicable if --folder is used)
    parser.add_argument(
        "-v",
        "--valid",
        action="store_true",
        help="Load validation data from folder (only applicable with --folder option)",
    )
    args = parser.parse_args()
    return args


# =====================================================================================
# ======================================= MAIN ========================================
# =====================================================================================


def main_compose(model_name, load_from_folder, load_valid=False):
    import sr_unet_config

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    if load_from_folder:
        print("Loading training data from folder...")
        train_df = load_data_from_folder(sr_unet_config.custom_data_folder)
        if load_valid:
            print("Loading validation data from folder...")
            valid_df = load_data_from_folder(sr_unet_config.custom_valid_folder)
    else:
        print("Loading custom data from CSV...")
        train_df = pd.read_csv(sr_unet_config.custom_data_path, encoding="utf-8")
        valid_df = (
            None  # CSV-based approach currently does not handle validation separately
        )

    print("Preparing Data & Pre-Processing")
    train_dataset = ImageDataset(df=train_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    valid_loader = None
    if load_valid and load_from_folder:
        valid_dataset = ImageDataset(df=valid_df, transform=transform)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    print("Configuring setup parameters...")
    in_channels = sr_unet_config.in_channels
    out_channels = sr_unet_config.out_channels
    convol_depth = sr_unet_config.convol_depth

    model = ResidualUNet2D(
        in_channels=in_channels, out_channels=out_channels, init_features=convol_depth
    )
    learning_rate = sr_unet_config.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    num_epoch = sr_unet_config.num_epoch

    checkpoint_path = sr_unet_config.checkpoint_path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    path_save = os.path.join(checkpoint_path, model_name + ".pth")

    summary(model)

    # Training - if validation loader is provided, pass it to the trainer
    trainer = ModelTrainer(model, criterion, num_epoch, optimizer)
    trainer.fit(
        train_loader, valid_loader=valid_loader, only_train=(valid_loader is None)
    )

    torch.save(model, path_save)
    return print("Model training finished")


def main_inference(model_path, test_path):
    model_unet = torch.load(model_path)
    _ = test_single_predict(test_path, model=model_unet)
    return print("Model inference finished")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print()
    print("====================================================================")
    print("      Fast Residual UNET2D Segmentation Modified Model Rendered     ")
    print("          Powered By Torchvision Basic Convolutional Module         ")
    print("                              -SRUNET2D-                            ")
    print("====================================================================")
    print()

    print(get_version())
    print("====================================================================")
    print()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_directory)

    args = get_argument()

    if args.load and args.target:
        main_inference(args.load, args.target)
    elif args.compose:
        main_compose(args.compose, args.folder, load_valid=args.valid)
    else:
        print("Error Option Selection")
        print(
            "Ensure you pick a valid option (Compose or Load) with the appropriate arguments."
        )
