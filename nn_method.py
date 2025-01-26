
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def convert_tensor(X_train, X_test, y_train, y_test, use_rnn=False):    
    if use_rnn == True:
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)    
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  
    else:
        X_train_tensor = torch.FloatTensor(X_train)  
        X_test_tensor = torch.FloatTensor(X_test)    
        y_train_tensor = torch.FloatTensor(y_train)
        y_test_tensor = torch.FloatTensor(y_test)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

class FNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        super(FNNRegressor, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))  
        x = self.fc_out(x)  
        return x

class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_sizes[0], batch_first=True)
        self.hidden_sizes = hidden_sizes
        self.fc_layers = nn.ModuleList()
        
        # Create additional hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  
        for fc in self.fc_layers:
            out = torch.relu(fc(out))
        out = self.fc_out(out)
        return out

def nn_train(num_epochs, model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Define optimizer here
    loss_fn = nn.MSELoss()  # Define loss function

    for epoch in range(num_epochs):
        model.train()  
        
        for X_batch, y_batch in train_loader:  # Iterate through batches
            optimizer.zero_grad()  # Zero the gradients
            
            # Forward pass
            y_pred = model(X_batch)
            
            # Compute loss
            loss = loss_fn(y_pred, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 1 == 0:  # Print every 1 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}] ------------------- Loss: {loss.item():.4f}')
    
    return model

def nn_predict(model, test_loader):
    model.eval()  # Set to evaluation
    all_predictions = []
    
    with torch.no_grad():  # Disable gradient calculation
        for X_batch, _ in test_loader:  # Ignore y_batch for prediction
            y_pred = model(X_batch)
            all_predictions.append(y_pred.numpy())  

    return np.concatenate(all_predictions)  

#if you want to determine the sequence
def prepare_sequences(X, y, seq_len):
    # Total number of sequences
    num_sequences = len(X) // seq_len
    
    # Reshape the data
    X_sequences = X[:num_sequences * seq_len].reshape(num_sequences, seq_len, X.shape[1])
    y_sequences = y[:num_sequences * seq_len].reshape(num_sequences, seq_len, 1)[:, -1, :]  # Take the last target value
    
    return X_sequences, y_sequences

def batching(X_tensor, y_tensor, batch_size):
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Choose an appropriate batch size
    return dataloader
