import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def convert_tensor(X_train, X_test, y_train, y_test, use_rnn=False, device='cpu'):
    if use_rnn:
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    else:
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train)
        y_test_tensor = torch.FloatTensor(y_test)

    return (X_train_tensor.to(device), X_test_tensor.to(device),
            y_train_tensor.to(device), y_test_tensor.to(device))

class FNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        super(FNNRegressor, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        for i, hidden in enumerate(hidden_sizes):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden))
            else:
                self.hidden_layers.append(nn.Linear(hidden_sizes[i - 1], hidden))
                
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)
        self.learning_rate = learning_rate

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.fc_out(x)
        return x

class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_sizes[0], batch_first=True)
        self.fc_layers = nn.ModuleList()
        
        for i in range(len(hidden_sizes) - 1):
            self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)
        self.learning_rate = learning_rate

    def forward(self, x):
        out, _ = self.rnn(x)
        # Take output from the last time step
        out = out[:, -1, :]
        for fc in self.fc_layers:
            out = torch.relu(fc(out))
        out = self.fc_out(out)
        return out

def nn_train(num_epochs, model, train_loader, val_loader=None,
             early_stopping_patience=5, early_stopping_min_delta=1e-4,
             device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in tqdm(range(num_epochs), desc="Training Per Epoch Progress"):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Ensure data is on the correct device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    y_val_pred = model(X_val)
                    running_val_loss += loss_fn(y_val_pred, y_val).item() * X_val.size(0)
            val_loss = running_val_loss / len(val_loader.dataset)
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping logic
            if val_loss + early_stopping_min_delta < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f}")

    return model

def nn_predict(model, test_loader, device='cpu'):
    model.eval() 
    all_predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            # Move back to CPU for numpy conversion
            all_predictions.append(y_pred.cpu().numpy())
    return np.concatenate(all_predictions)

#if you want to determine the sequence
def prepare_sequences(X, y, seq_len):
    num_sequences = len(X) // seq_len
    
    #Reshaping the data
    X_sequences = X[:num_sequences * seq_len].reshape(num_sequences, seq_len, X.shape[1])
    y_sequences = y[:num_sequences * seq_len].reshape(num_sequences, seq_len, 1)[:, -1, :]
    return X_sequences, y_sequences

def batching(X_tensor, y_tensor, batch_size):
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
