import torch
import torch.nn as nn
torch.manual_seed(42)
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import joblib


def make_lag_df(df, feature_cols, window=14):
    out = {}
    for col in feature_cols:
        s = df[col]
        for lag in range(window, 0, -1):
            out[f"{col}_t-{lag}"] = s.shift(lag)
    df_seq = pd.DataFrame(out, index=df.index).dropna()
    return df_seq

def df_to_loader(df_seq, feature_cols, lagged_feature_cols, target_col="TargetDelta",
                 window=14, batch_size=128, shuffle=False):

    # Create input feature sequences X with shape (N, T, F)
    Xflat = torch.tensor(df_seq[lagged_feature_cols].to_numpy(), dtype=torch.float32) 
    N = Xflat.shape[0]
    T = window
    F = len(features)
    X = Xflat.view(N, T, F)

    # Create targets y with shape (N, 1)
    y = torch.tensor(df_seq[target_col].to_numpy(), dtype=torch.float32).unsqueeze(1) 

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# Load testing set
spy_test_df = pd.read_csv("datasets/spy_test.csv", index_col="Date")

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'sma_10', 'ema_20', 'rsi_14',
       'macd', 'macd_signal', 'macd_hist', 'atr_14', 'bb_upper', 'bb_lower']

spy_test_seq_df = make_lag_df(spy_test_df, features, window=24)
lagged_features = spy_test_seq_df.columns

scaler = joblib.load("models/scaler_spy.pkl")
spy_test_seq_df_scaled = pd.DataFrame(scaler.transform(spy_test_seq_df), 
                                      index=spy_test_seq_df.index, 
                                      columns=spy_test_seq_df.columns)

spy_test_seq_df_scaled["TargetDelta"] = spy_test_df.loc[spy_test_seq_df_scaled.index, "TargetDelta"].astype("float32")

test_loader = df_to_loader(spy_test_seq_df_scaled, 
                            feature_cols=features,
                            lagged_feature_cols=lagged_features,
                            target_col="TargetDelta",
                            window=24,
                            batch_size=128, 
                            shuffle=False,)

spy_test_df.round(2).head()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleRNN(nn.Module):
    def __init__(self, input_size=14, hidden=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden, 
                          num_layers=num_layers, 
                          batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)           
        return self.fc(out[:, -1, :])   
        
## YOUR SOLUTION HERE ##
class SimpleGRU(nn.Module):
    def __init__(self, input_size=14, hidden=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)          
        return self.fc(out[:, -1, :]) 

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=14, hidden=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)         
        return self.fc(out[:, -1, :]) 

# Instantiate models + pre-trained weights
RNN_PATH = "models/rnn_spy_state.pt"
GRU_PATH = "models/gru_spy_state.pt"
LSTM_PATH = "models/lstm_spy_state.pt"

rnn = SimpleRNN(14,128,2)
rnn_state = torch.load(RNN_PATH,weights_only=True)
rnn.load_state_dict(rnn_state)
rnn.to(device)

gru  = SimpleGRU(14,128,2)
gru_state = torch.load(GRU_PATH,weights_only=True)
gru.load_state_dict(gru_state)
gru.to(device)

lstm = SimpleLSTM(14,128,2)
lstm_state = torch.load(LSTM_PATH,weights_only=True)
lstm.load_state_dict(lstm_state)
lstm.to(device)




def predict(model, dataloader):
    model.eval()
    batch_preds = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            batch_pred = rnn(X_batch.to(device)) 
            batch_pred = batch_pred.cpu().detach().numpy().ravel()
            batch_preds.append(batch_pred)
    y_preds = np.concatenate(batch_preds)
    return y_preds

rnn_preds = predict(rnn,test_loader)
gru_preds = predict(gru,test_loader)
lstm_preds = predict(lstm,test_loader)

# Show output - Test set MSE performance
from sklearn.metrics import mean_squared_error
y_true = spy_test_df["TargetDelta"][:-24].values
mse_rnn  = mean_squared_error(y_true, rnn_preds)
mse_gru  = mean_squared_error(y_true, gru_preds)
mse_lstm = mean_squared_error(y_true, lstm_preds)

print("=== Test Performance ===")
print(f"RNN  MSE: {mse_rnn:.6f}")
print(f"GRU  MSE: {mse_gru:.6f}")
print(f"LSTM MSE: {mse_lstm:.6f}")






# Obtain the most recent prediction
pred_delta_rnn  = rnn_preds[-1]
pred_delta_gru  = gru_preds[-1]
pred_delta_lstm = lstm_preds[-1]

# Use delta + prev close price to calculate the new predicted close price
prev_close = float(spy_test_df["Close"].iloc[-1])  
actual_close = float(spy_test_df["Target"].iloc[-1]) 

## YOUR SOLUTION HERE ##
pred_next_close_rnn  = prev_close + pred_delta_rnn
pred_next_close_gru  = prev_close + pred_delta_gru
pred_next_close_lstm = prev_close + pred_delta_lstm


# Show output - Predicted next-day close
print(f"Previous Close:       ${prev_close:.2f}")
print(f"RNN:  Pred Δ Close:    {pred_delta_rnn:+.2f}  | Next Close: ${pred_next_close_rnn:.2f}")
print(f"GRU:  Pred Δ Close:    {pred_delta_gru:+.2f}  | Next Close: ${pred_next_close_gru:.2f}")
print(f"LSTM: Pred Δ Close:    {pred_delta_lstm:+.2f} | Next Close: ${pred_next_close_lstm:.2f}")
print(f"Actual Close:         ${actual_close:.2f}")








# Introduction to Neural Network Architectures
# Recurrent Neural Networks: RNN, GRU, and LSTM
# 29 min
# Recurrent Neural Networks (RNNs) are traditional methods for modeling sequential data. The key idea behind RNNs is recurrence, where the model maintains a hidden state that serves as its “memory,” which is passed from one step to the next.

# This recurrence connection helps the network remember past information to predict future outputs (e.g., predicting the next value in a time series based on previous values).

# We’ll implement an RNN architecture containing an RNN layer with the following class:

# import torch
# import torch.nn as nn

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=2):
#         super().__init__()
#         self.rnn = nn.RNN(input_size=input_size, 
#                           hidden_size=hidden_size, 
#                           num_layers=num_layers, 
#                           batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         out, hidden = self.rnn(x)           
#         return self.fc(out[:, -1, :])   

# # Initialize model (adjust input_size based on your features)
# rnn = SimpleRNN(input_size=1).to(device)

# Copy to Clipboard

# input_size: the number of features per timestep in each sequence
# hidden_size: the size of the hidden state
# num_layers: the number of recurrent layers to stack — deeper isn’t always better due to vanishing/exploding gradients
# batch_first: Use True if the input sequences are batched with the shape (batch, sequence, features).
# The forward pass output out contains the hidden states maintained across all time steps. The final hidden state out[:, -1, :] is passed into the last linear layer to predict the final output.

# Specifically, this is referred to as a many-to-one architecture because we utilize a sequence with multiple timesteps to predict a single output (TargetDelta in our example).

# Here’s how we generate the RNN prediction loop:

# rnn.eval()
# batch_preds = []
# with torch.no_grad():
#     for X_batch, y_batch in loader:
#         batch_pred = rnn(X_batch.to(device)) 
#         batch_pred = batch_pred.cpu().detach().numpy().ravel()
#         batch_preds.append(batch_pred)
# y_preds = np.concatenate(batch_preds)

# Copy to Clipboard

# We move the predictions back to 
# CPU
# Preview: Docs Loading link description
#  and convert them to NumPy arrays using .cpu().detach().numpy().ravel() to collect each batch’s predictions.
# RNN Variants — GRUs and LSTMs
# While powerful, standard RNNs suffer from the vanishing and exploding gradient problems:

# Vanishing Gradient: Gradients shrink exponentially during backpropagation through time, causing the network to “forget” information in longer sequences.
# Exploding Gradient: Gradients grow extremely large during backpropagation, leading to unstable weight updates.
# To address these issues, AI researchers developed variants with gated architectures:

# Long Short-Term Memory (LSTM) Networks: Introduced gates (input, forget, and output) to control information flow across sequential steps through the hidden state and an additional memory cell state:

# Forget Gate: decides what information to discard from the cell state
# Input Gate: decides what new information to add to the cell state
# Output Gate: decides which part of the updated cell state is used to compute the hidden state
# Gated Recurrent Units (GRU): a simplified LSTM with fewer gates (reset and update gates), resulting in faster training

# We can implement GRU and LSTM variants in PyTorch using the following layers:

# gru = nn.GRU(input_size=1, hidden_size=32, num_layers=1, batch_first=True)

# lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)

# Copy to Clipboard

# Model	Advantages	Disadvantages
# RNN	Simple, fast, foundational	Struggles with long-term dependencies (vanishing/exploding gradients)
# LSTM	Handles long-term memory best, widely used	More parameters, slower training
# GRU	Faster training, fewer parameters	Sometimes slightly less accurate than LSTMs on longer sequences
# Let’s implement a GRU and an LSTM similarly to the RNN from before and compare their performance on the testing set.

# Disclaimer: This is for educational purposes only. This is not financial advice, and the models we build are not intended for real trading or investment decisions. Always consult a financial professional before making investment choices, and trade at your own risk.