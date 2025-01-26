
import argparse
import pandas as pd
import joblib
import lightgbm as lgb
import torch
import numpy as np
from nn_method import nn_predict  # Import your nn_predict function

def main(args):
    data = pd.read_csv(args.data)

    if args.target in data.columns:
        print(f"Target column '{args.target}' already exists. Predictions will be filled in.")

    model = None
    predictions = None

    if args.model.endswith('.txt'):
        model = lgb.Booster(model_file=args.model)
        predictions = model.predict(data.drop(columns=[args.target], errors='ignore'))

    elif args.model.endswith('.pkl'):
        model = joblib.load(args.model)
        predictions = model.predict(data.drop(columns=[args.target], errors='ignore'))

    elif args.model.endswith('.pth'):
        model = torch.load(args.model)
        input_tensor = torch.FloatTensor(data.drop(columns=[args.target], errors='ignore').values)

        if args.batch_size:
            # Use DataLoader for batching
            test_loader = torch.utils.data.DataLoader(input_tensor, batch_size=args.batch_size, shuffle=False)
            predictions = nn_predict(model, test_loader)
        else:
            predictions = nn_predict(model, input_tensor.unsqueeze(0))  # Add batch dimension

    else:
        raise ValueError("Unsupported model format. Please provide a .txt, .pkl, or .pth file.")

    # Fill in the predictions in the target column
    data[args.target] = predictions

    # Save the result to a new CSV file
    output_file = args.data.replace('.csv', '_predictions.csv')
    data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the target parameter using a saved machine learning model from compose.py. Sagara_Z625034")
    parser.add_argument('-d', '--data', required=True, help='Path to the input data CSV file.')
    parser.add_argument('-m', '--model', required=True, help='Path to the trained model file (.txt, .pkl, .pth).')
    parser.add_argument('-t', '--target', required=True, help='Target column name for predictions.')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size for predictions. If not specified, predictions will be done without batching.')

    args = parser.parse_args()
    main(args)
