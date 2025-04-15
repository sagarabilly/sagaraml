# SagaraML

A collection of simple machine learning / deep learning code in the focus of simplicity and ease of use.
Most of them are built using the PyTorch framework and are designed to be executed via commannd line terminal for both training and inference.
You could also use it as a starting template for your code. 

## Fatures

As of right now, it only includes 3 application features:  
- **img_classifier** Image classification with 3 sample architectures: Efficientnet B0, Resnet18, Simple CNN.  
- **SR_UNET**: Residual Unet2D (ResUNet2D) model architecture for medical or microscopic image segmentation.  
- **TPML**: Traditional Predictive Machine Learning, predictive model using traditional machine learning method designed for tabular dataset. Includes KFold cross-validation and models such as LGBM Regressor and more.  

## Usage and Requirements

1. Most of them have a seperate configuration file that you could adjust and modify.    
2. Make sure you have installed the required libraries and adjust the version based on your current installed cuda GPU. (read more at [PyTorch](https://pytorch.org/))  
3. For how to use it, please check each of the separate README.md file.   

Clone this repository by:
```bash
git clone https://github.com/sagarabilly/sagaraml.git
```

## Contributions

As always, contributions, improvements, and bug fixes are welcome.  

