
# Machine Learning Model Composer

This repository contains a Python-based machine learning model composer that allows users to create, train, and evaluate various machine learning models on their inputted tabular dataset. <br>
<br>Contains 2 CLI-executable code [compose.py and predict.py] and 1 small separate lib code [nn_method.py]. <br>
The compose.py provides flexibility to choose between different algorithms, perform dimensionality reduction, visualize the data, and save the generated model. <br>
<br>The script also supports neural networks for modern machine learning techniques. <br>
While predict.py will generate a prediction based on the inputted data and the choosen model that has been created. 

## Features

- **Dataset Input**: Supports CSV, Excel, or Pickle format datasets.
- **Model Selection**:
  - LightGBM Regressor
  - Linear Regression
  - SGD Regressor
  - Feedforward Neural Network (FNN)
  - Recurrent Neural Network (RNN)
- **Data Visualization**: Option to visualize the dataset using seaborn's pairplot.
- **Dimensionality Reduction**: Option to apply PCA (Principal Component Analysis).
- **Cross-Validation**: Supports K-Fold cross-validation and Train-Test Split.
- **Model Saving**: Automatically saves the trained model for future use.
- **Metrics**: Computes various performance metrics such as R², MSE, MAE, MAPE, and RMSE.
- **Prediction**: Create a Prediction based on the inputted data, designated target parameter, choosen model, and batching options.

## Requirements

- Python 3.x
- Required libraries:
  - `scikit-learn`
  - `pandas`
  - `seaborn`
  - `matplotlib`
  - `lightgbm`
  - `tensorflow` (for neural network models)
  - `torch` (for neural network models)
  - `gc` (Garbage Collection)
  - `rich` (for progress bar)
  - `numpy`

You can install the required libraries using pip:
```pip install -r requirements.txt```

## Use Case Example

1. Run with LightGBM Regressor and save the model:<br>
```python compose.py -p /path/to/dataset.csv -t target_column --lgbm -s```

2. Run with Linear Regression and visualize data:<br>
```python compose.py -p /path/to/dataset.csv -t target_column --linear --visualize```

3. Run with Feedforward Neural Network with PCA and save model:<br>
```python compose.py -p /path/to/dataset.csv -t target_column --fnn --pca -s```

4. Run with Recurrent Neural Network (RNN):<br>
```python compose.py -p /path/to/dataset.csv -t target_column --rnn```

5. Make Prediction:<br>
```python predict.py -d /path/to/dataset.csv -t target_column -m /path/to/trained/model```
 
