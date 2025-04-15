# Residual UNet2D for Image Segmentation

A simple fast image segmentation model based on a Residual UNet2D architecture implemented in PyTorch.   
It is designed to be used via the command line, providing easy options for training a new model or performing inference on a single image.  

## Overview  

The Residual UNet2D model uses residual blocks and a U-Net style encoder-decoder network for efficient segmentation tasks. It supports:  
- **Training** a segmentation model using either a CSV file listing image-mask pairs or by loading training images directly from a folder.  
- **Validation**: An option to load a separate validation dataset (when using folder mode) to evaluate model performance during training.  
- **Inference**: Running a trained model to generate segmentation predictions on new images.  
- **Visualization**: Plotting and comparing input images with their segmentation results, plus analyzing simple statistics like pixel density and porosity.  

## Folder Structure

If you want to use --folder option, please make sure the folder structure consist of /images and /masks for both training and valid folder.  

```bash
train/                       
├── images/  
└── masks/ 
valid/                       
├── images/ 
└── masks/ 
```

## Usage  
Please djust the parameter and the input path in the sr_unet_config.py  
The code supports two main modes of operation: training (compose) and inference (load).  

### Training (Compose)  
You can train a new model using one of the following options:  

1. CSV-based Training (Default)  
If your training data is listed in a CSV file (with two columns: image path and mask path):  
```bash
python sr_unet.py --compose model_name
```
  
2. Folder-based Training  
To load your training images directly from a folder structure, use the --folder flag. If you have validation data in a separate folder, add the --valid flag:  
```bash
python sr_unet.py --compose model_name --folder
```
  
With validation data (folder-based mode only):  
```bash
python sr_unet.py --compose model_name --folder --valid
```
  
### Inference (Load)  
For running a previously trained model for segmentation on a new image:  
```bash
python sr_unet.py --load path/to/model.pth --target path/to/test/image.jpg
```

##Contributions  
As always, contributions, improvements, and bug fixes are welcome.  



