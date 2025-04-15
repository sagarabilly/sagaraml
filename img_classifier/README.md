# Simple Image Classifier  

A sinple image classifier using pytorch with 3 sample architectures:  
- **M_EFNETB0** – based on EfficientNet-B0 (using `timm`)  
- **RESNET18** – from torchvision deep residual learning   
- **CNN_MOD** – a simple Convolutional Neural Network (added just for you to try or modify)  

It is currently using Pytorch ImageFolder for folder structure.    
The configuration parameters are separated in the `img_classifier_config.py` file, and the main code is executable through the terminal using `argparse`.     

## Overview  

- **CLI Interface:** Using command-line arguments (`--compose` for making the model, `--load`, and `--target` for loading the model and inferencing an input image)  
- **Seperate Configuration:** seperate configuration file to adjust dataset paths, training hyperparameters, and model type in `img_classifier_config.py`.  

## Usage

Please always make sure you have installed the required libraries.  
Please adjust the parameter and the data path in the img_classifier_config.py`.  

To use it, you can execute it from the terminal.   
The main script supports the following command-line options:  

--compose: Train a new model from scratch.  
--load MODEL_PATH: Load an existing model (provide the model file path).  
--target IMAGE_PATH: Provide an image file path for prediction (used with --load).  

To compose and train a new model, run:  
```bash
python main.py --compose
```

If you already have a trained model via ```--compose```, you can load the model and predict the class of a target image.   
```bash
python main.py --load model.pth --target path/to/your/image.jpg
```
This command will load the saved model, preprocess the image using the transforms, Output and visualize the predicted class probabilities.  

## Contributions  
As always, contributions, improvements, and bug fixes are welcome.  

