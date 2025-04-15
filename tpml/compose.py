#Python 3.11.5 : UTF-08

#Data Structure
import pandas as pd
import numpy as np

#Visual Inspections
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning
from sklearn import decomposition
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import (r2_score, 
                             mean_squared_error,
                             mean_absolute_error, 
                             mean_absolute_percentage_error,
                             root_mean_squared_error
                             )
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import joblib
import torch

import lightgbm as lgbm

#util and tools
import os
import time 
start_time = time.time()
from rich.progress import Progress 
import gc
from argparse import ArgumentParser

# outside lib
os.chdir(r"D:\tpml")
from nn_method import (
    convert_tensor, prepare_sequences, batching,
    RNNRegressor, FNNRegressor,
    nn_train, nn_predict, nn_save_model)

#-----------------------------------------------------------------------

def visual_inspect(df):
    #merging 2 categorical parameters, ONLY FOR VISUAL INSPECTIONS 
    df['cgroup'] = df["f4g_fmdg_mdgen"].astype(str) + "-" + df["f4g_fmds_mdf"].astype(str)
    df['cgroup'] = df['cgroup'].astype('category')
    
    #visualizatons
    sns.set_theme(rc={'figure.figsize':(80,80)})
    visualization_features = list(df.columns)
    visualization_features.remove("f4g_fmds_mdf")
    visualization_features.remove("f4g_fmdg_mdgen")
    
    inspect_pplot = sns.pairplot(df[visualization_features], hue="cgroup", palette="viridis", plot_kws={"s": 3})
    plt.show()
    return visualization_features

def data_preparation(df, target:list, features=None):
    if features :
        assert isinstance(features, list), "Inputted features must in a form of list"    
        X = df[features].values
        y = df[target[0]].values
    else:
        X = df.drop(columns=target).values
        y = df[target[0]].values
    return X, y

def pca_bar(pca:object):
    sns.set_theme(rc={'figure.figsize':(10,5)})
    plt.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1, len(pca.explained_variance_)+1),
             np.cumsum(pca.explained_variance_), c='red',
             label="Cumulative Explained Variance")
    plt.legend(loc='upper left')
    return

def make_score_data(score_data:list):
    column_names = ["R2", "MSE", "MAE", "MAPE", "RMSE"] 
    new_index = [f'Test{i}' for i in range(1, len(score_data[0]) + 1)]
    score_df = pd.DataFrame(score_data, columns=column_names, index=new_index)  
    score_df.loc['Average'] = score_df.mean()
    return score_df.round(5)

def save_model(model, modern_ml, algorithm):
    if modern_ml == False: 
        if algorithm == "lgbm":
            model.booster_.save_model('lgbm_r1.txt')
        else:
            joblib.dump(model, 'model.pkl')
    else:
        nn_save_model(model)
    return print("Model is Saved Successfully")    

def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.xlsx' or ext == '.xls':
        return pd.read_excel(file_path)
    elif ext == '.pkl' or ext == '.gzip':
        return pd.read_pickle(file_path, compression='gzip')
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def get_args():
    parser = ArgumentParser(description='Compose or create a machine learning model based on your inputted data and decided target parameter. Created by: sagara_billy')

    parser.add_argument('-p','--path', help='Path to the dataset file (CSV, Excel, or Pickle format)')
    parser.add_argument('-t','--target', help='The target column for prediction')

    # Features arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--lgbm', action='store_true', help='LightGBM Regressor model')
    model_group.add_argument('--linear', action='store_true', help='Linear Regression model')
    model_group.add_argument('--sgd', action='store_true', help='SGD Regressor model')
    model_group.add_argument('--fnn', action='store_true', help='Feedforward Neural Network model')
    model_group.add_argument('--rnn', action='store_true', help='Recurrent Neural Network model')
    model_group.add_argument('--visualize', action='store_true', help='PreVisualize data using pairplot')

    parser.add_argument('--pca', action='store_true', help='PCA Dimensionality Reduction')
    parser.add_argument('-s', '--save', action='store_true', help='Save generated model automatically')
    parser.add_argument('--version', action='store_true', help="Show version")

    args = parser.parse_args()
    
    return args

#------------------------------------------------------------------------------

def main(algorithm:str, modern_ml=False, save_mod=True, visual=False, dimred=False):
    #path assignment and initialization
    print(f"detecting data : {path}")
    
    df = load_data(path)
    df_desc = df.describe()
    print(df_desc)
    print(df.info())
    
    #-----PairPlot Visualization------
    if visual:
        threshold_range = 100
        step_cal = len(df)
        downsample_df = df.iloc[::step_cal, :]
    
        #pairplot visualization
        sns.pairplot(downsample_df)
        plt.show()

    #-----Preparation and Scaling------
    X, y = data_preparation(df, target)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #-----Dimensionality Reduction-----
    ratio_reduction = 0.05
    if dimred == False:
        ratio_reduction = 0
    
    print("Processing Dimensionality Reduction ...")
    pca_init = decomposition.PCA(1-ratio_reduction) 
    X_pca_init = pca_init.fit_transform(X)
    print("Base PCA Explained Variance Ratio:")
    print(pca_init.explained_variance_ratio_)
    pca_bar(pca_init)
    
    pca = decomposition.PCA(n_components=len(pca_init.explained_variance_ratio_) ,svd_solver='randomized', 
                            whiten=True, random_state=529)
    X_pca = pca.fit_transform(X)
    loadings_df = pd.DataFrame(pca.components_, columns=feature_names)

    #-----KFold cross-validation-----
    kf = KFold(n_splits=5, shuffle=True, random_state=529)   
    
    #-----metrics-----
    r2, mse, mae, mape, rmse = [], [], [], [], [] 
    
    #============================= ALGORITHM SELECTION =============================
    #modern ml utilizes neural network in which the code is written in nn_method.py
    
    if modern_ml :
        #Modern / Neural Network Algorithm
        
        num_epochs = 10
        batch_size = 10
        algorithm = 'fnn'
        crossval = "tts"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Model selections (Define Param & Initialize)
        if algorithm == 'rnn':
            print('Selected Model RNN')
            model = RNNRegressor(input_size=len(pca_init.explained_variance_ratio_), 
                                 hidden_sizes=[64, 32], 
                                 output_size=1, 
                                 learning_rate=0.005)
            #Sequencing 
            seq_len = 100
            X_pca, y = prepare_sequences(X_pca, y, seq_len)
            use_rnn = True
            
        elif algorithm == 'fnn':
            print('Selected Model FNN')
            model = FNNRegressor(input_size=len(pca_init.explained_variance_ratio_), 
                                  hidden_sizes=[64, 32], 
                                  output_size=1,
                                  learning_rate=0.005)
            use_rnn = False
            
        else:
            print("Model Not Selected. Error")
           
        model = model.to(device)
        
        print("Model-Making and Fitting Execution...")
        
        if crossval == "kfold" :
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing Fold Iteration", total=kf.get_n_splits())
                
                for train_index, test_index in kf.split(X_pca):
                    X_train, X_test = X_pca[train_index], X_pca[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    print('Convert to tensor...')
                    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_tensor(X_train, X_test, y_train, y_test, use_rnn=use_rnn, device=device)
                    
                    print('Batching and using Data Loader')
                    train_loader = batching(X_train_tensor, y_train_tensor, batch_size)
                    test_loader = batching(X_test_tensor, y_test_tensor, batch_size)
                    
                    print('Training Model...')
                    trained_model = nn_train(num_epochs, model, train_loader, val_loader=None, device=device)
                    
                    print('Predicting...')
                    y_pred = nn_predict(trained_model, test_loader, device=device)
                    
                    # Calculate and store the accuracy
                    r2.append(r2_score(y_test, y_pred))
                    mse.append(mean_squared_error(y_test, y_pred))
                    mae.append(mean_absolute_error(y_test, y_pred))   
                    mape.append(mean_absolute_percentage_error(y_test, y_pred))
                    rmse.append(root_mean_squared_error(y_test, y_pred))
                    
                    progress.update(task, advance=1)
        
        elif crossval == "tts":
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=529)
            
            print('Convert to tensor...')
            X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_tensor(X_train, X_test, y_train, y_test, use_rnn=use_rnn, device=device)
            
            print('Batching and using Data Loader')
            train_loader = batching(X_train_tensor, y_train_tensor, batch_size)
            test_loader = batching(X_test_tensor, y_test_tensor, batch_size)
            
            print('Training Model...')
            trained_model = nn_train(num_epochs, model, train_loader, val_loader=None, device=device)
            
            print('Predicting...')
            y_pred = nn_predict(trained_model, test_loader, device=device)
            
            # Calculate and store the accuracy
            r2.append(r2_score(y_test, y_pred))
            mse.append(mean_squared_error(y_test, y_pred))
            mae.append(mean_absolute_error(y_test, y_pred))   
            mape.append(mean_absolute_percentage_error(y_test, y_pred))
            rmse.append(root_mean_squared_error(y_test, y_pred))
            
        else:
            raise Exception('Error: Options is not available yet')
            
    else :
        #Traditional Algorithm (Default Option) 
        
        if algorithm == "linear" :
            model = LinearRegression()
        
        elif algorithm == "SGD" : 
            model = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling',
                                 penalty='l2', alpha=0.0001, random_state=529)
        
        elif algorithm == "lgbm" :
            model = lgbm.LGBMRegressor(n_estimators=1000, learning_rate=0.05) #Current Best Model Selected
        
        else :
            raise Exception('Error: Selected model is not an option')
            
        gc.collect()
        
        print("Model-Making and Fitting Execution...")
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing Fold Iteration", total=kf.get_n_splits())
            
            for train_index, test_index in kf.split(X_pca):
                X_train, X_test = X_pca[train_index], X_pca[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                print('Training/Fitting ...')
                model.fit(X_train, y_train)
                
                print('Predicting ...')
                y_pred = model.predict(X_test)
                
                # Calculate and store the accuracy
                r2.append(r2_score(y_test, y_pred))
                mse.append(mean_squared_error(y_test, y_pred))
                mae.append(mean_absolute_error(y_test, y_pred))   
                mape.append(mean_absolute_percentage_error(y_test, y_pred))
                rmse.append(root_mean_squared_error(y_test, y_pred))
                
                progress.update(task, advance=1)
            
    acc_score_data = [r2, mse, mae, mape, rmse]
    acc_score_data = np.array(acc_score_data).T.tolist()
    score_df = make_score_data(acc_score_data)
    
    print("---> Accuracy Result:")
    print("=================================================================")
    print(score_df)
    print("-----------------------------------------------------------------")
    
    #save option
    if save_mod:
        save_model(model, modern_ml, algorithm)
        
    #End Note
    print(f"-Time Elapsed : {time.time() - start_time}")

#---------------------------------------------------------------------------
if __name__ == "__main__": 
    args = get_args()
    
    #Default setting
    modern_ml = False
    algorithm = 'lgbm'
    visual = False
    save_mod = False
    dimred = False

    if args.fnn or args.rnn:
        modern_ml = True

    if args.lgbm:
        print("Selected model: LightGBM Regressor")
        algorithm = "lgbm"
    elif args.linear:
        print("Selected model: Linear Regression")
        algorithm = "linear"
    elif args.sgd:
        print("Selected model: SGD Regressor")
        algorithm = "sgd"
    elif args.fnn:
        print("Selected model: Feedforward Neural Network")
        algorithm = "fnn"
    elif args.rnn:
        print("Selected model: Recurrent Neural Network")
        algorithm = "rnn"
    elif args.visualize:
        print("Selected action: Visualize data using seaborn pairplot")
        visual = True
    if args.pca:
        print("Using PCA for dimensionality reduction")
        dimred = True
    if args.save:
        print('Model will be saved automatically')
        save_mod = True
    
    global path, target
    path = args.path
    target = [args.target] 

    #EXECUTE
    if not args.version:
        print(f"Loading data from {args.path}")
        df = load_data(args.path)
        
        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in the data")
        
        # Print basic description of the dataset
        print(df.describe())
        print(df.info())

        main(algorithm, modern_ml, save_mod, visual, dimred)

    else:
        print("Version 1.0: Updated January 2025 : Unstable")

