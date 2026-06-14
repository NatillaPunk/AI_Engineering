import torch
import torch.nn as nn
torch.manual_seed(42)
import pandas as pd
import numpy as np
import pandas as pd


spy_train_df = pd.read_csv("datasets/spy_train.csv", index_col="Date")
spy_train_df.round(2).head()






## YOUR SOLUTION HERE ##
def make_lag_df(df, feature_cols, window=14):
    out = {}
    for col in feature_cols:
        s = df[col]
        for lag in range(window, 0, -1):
            out[f"{col}_t-{lag}"] = s.shift(lag)
    df_seq = pd.DataFrame(out, index=df.index).dropna()
    return df_seq

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'sma_10', 'ema_20', 'rsi_14',  'macd', 'macd_signal', 'macd_hist', 'atr_14', 'bb_upper', 'bb_lower']

spy_train_seq_df = make_lag_df(spy_train_df, features, window=24)

lagged_features = list(spy_train_seq_df.columns)

# Show output
print("Number of lagged features:", len(lagged_features))
spy_train_seq_df.head()








## YOUR SOLUTION HERE ##
spy_train_seq_df["TargetDelta"] = spy_train_df.loc[spy_train_seq_df.index,"TargetDelta"].astype('float32')

# Show output
spy_train_seq_df["TargetDelta"].head()






import torch
from torch.utils.data import TensorDataset, DataLoader

## YOUR SOLUTION HERE ##
def df_to_loader(df_seq, feature_cols, lagged_feature_cols, target_col="TargetDelta",
                 window=14, batch_size=128, shuffle=False):

    # Create input feature sequences X with shape (N, T, F)
    Xflat = torch.tensor(df_seq[lagged_feature_cols].to_numpy(),dtype=torch.float32)
    N = Xflat.shape[0]
    T = window
    F = len(feature_cols)
    X = Xflat.view(N, T, F)

    # Create targets y with shape (N, 1)
    y = torch.tensor(df_seq[target_col].to_numpy(), dtype=torch.float32).unsqueeze(1)

    # Create TensorDataset and DataLoader
    dataset =  TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

train_loader = df_to_loader(
    spy_train_seq_df, features, lagged_features,
    target_col="TargetDelta", window=24, batch_size=128, shuffle=True
)


















# Introduction to Neural Network Architectures
# Recurrent Neural Networks: Sequential Data
# 37 min
# Next, we’ll examine an architecture designed for learning patterns within sequential data, where order and temporal relationships are present.

# Examples of sequential data include:

# Text: a sequence of words or characters
# Time series: stock prices, weather, or sensor readings
# Audio: sound waves captured at regular intervals
# Preprocessing Data Into Sequences
# When working with sequential models, such as RNNs, the format of the input data is crucial. Unlike in tabular tasks, where each observation (row) is treated independently, we’ll need to transform the data into a lagged sliding window format.

# For example, in this exercise, we’ll apply sequential modeling techniques to financial time series by attempting to predict the next day’s closing price of the S&P 500 ETF ($SPY) using historical price data:

# Date	Open	Close	TargetDelta
# 2025-09-16	659.64	658.18	-0.82
# 2025-09-17	658.19	657.36	3.07
# 2025-09-18	660.06	660.43	3.27
# 2025-09-19	662.33	663.70	2.91
# Each row contains pricing information for a single day. We’ll predict TargetDelta, which is the difference between the closing price and the previous day’s closing price.

# Feeding this directly will not allow the model to learn the temporal order of values, so we’ll need to restructure the data where each row is a “window” of past inputs.

# For example, we can use the closing prices in the previous three days as our input to predict the TargetDelta on 2025-09-19 by preprocessing the data into the following sequence:

# Date (t)	Close_t-3	Close_t-2	Close_t-1	TargetDelta
# 2025-09-19	658.18	657.36	660.43	2.91
# Target y: The TargetDelta value on 2025-09-19 is 2.91
# Features X: sliding window containing the previous three closing prices. For example, Close_t-1 is the previous day’s closing price on 2025-09-18.
# Note: We need to be careful not to include the current day’s closing price Close_t in our features, as this would result in data leakage.

# To transform our data, we’ll create the following loop to create a three-day lagged window:

# feature_cols = ["Close"]
# window = 3
# out = {}
# for col in feature_cols:
#     s = df[col]
#     for lag in range(window, 0, -1):
#         out[f"{col}_t-{lag}"] = s.shift(lag)
        
# df_seq = pd.DataFrame(out, index=df.index).dropna()
# lagged_features = list(df_seq.columns)

# Copy to Clipboard

# out helps create the new lagged columns.
# This code loops over each feature column (e.g., Close, but can also include Open, etc.).
# for lag in range(window, 0, -1) creates lagged versions of each feature based on how many windows are specified.
# s.shift(lag) moves values down by lag rows and helps name each new column in the format feature_t-lag.
# df_seq builds the final DataFrame, keeping the same 
# index
# Preview: Docs Loading link description
#  (dates) and dropping rows with missing values.
# lagged_features contains the list of lagged feature column names.
# Next, we’ll re-attach the target column to the new lagged DataFrame, making sure to align the indices:

# df_seq["TargetDelta"] = df.loc[df_seq.index, "TargetDelta"].astype("float32")

# Copy to Clipboard

# Next, we’ll format our input feature sequences X and targets y with the following:

# Xflat = torch.tensor(df_seq[lagged_features].to_numpy(), dtype=torch.float32)
# N = Xflat.shape[0]
# T = window
# F = len(feature_cols)
# X = Xflat.view(N, T, F)

# y = torch.tensor(df_seq["TargetDelta"].to_numpy(), dtype=torch.float32).unsqueeze(1)

# Copy to Clipboard

# Xflat converts the lagged feature values into a tensor with datatype float32.
# N is the number of sequences.
# T is the number of timesteps (window size).
# F is the number of features.
# X is reshaped into (N, T, F) for model input.
# y converts the target values into a tensor with the datatype float32 and shape (N, 1).
# Lastly, we’ll batch X and y by loading them into dataloaders:

# from torch.utils.data import TensorDataset, DataLoader
# dataset = TensorDataset(X, y)
# loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Copy to Clipboard

# We should be cautious when shuffling a time series dataset with shuffle=True to prevent data leakage. Generally, we should only shuffle the training data if:

# Each window sequence is independent.
# Each window sequence is ordered internally.
# You want to randomize the order to improve training.
# Instructions
# Checkpoint 1 Passed
# 1.
# Let’s create a function named make_lag_df that creates lagged sequences with a specified window.

# A. The function should have the following inputs:

# df: the original DataFrame of $SPY prices and technical indicators
# feature_cols: list of (unlagged) feature names
# window: specifies the window size (set default to 14)
# B. The function should return df_seq, which is the lagged DataFrame where each row is a lagged sequence for each date. Be sure to set the Date column as the index for the lagged DataFrame.

# C. Use the function to create a lagged DataFrame with a 24-day window and save it to the variable spy_train_seq_df.

# Use the following features: ['Open', 'High', 'Low', 'Close', 'Volume', 'sma_10', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'atr_14', 'bb_upper', 'bb_lower'] and save the list to the variable features.
# Save the new lagged columns to the variable lagged_features.
# D. Re-assign the target column TargetDelta to the lagged DataFrame.