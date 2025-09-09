import numpy as np                                      # For numerical operations with arrays
import matplotlib.pyplot as plt                         # For plotting graphs
import pandas as pd                                     # For data manipulation and analysis
import yfinance as yf                                   # For fetching financial data from Yahoo Finance API

import torch                                            # For building and training neural networks
import torch.nn as nn           
import torch.optim as optim

from sklearn.preprocessing import StandardScaler        # For feature scaling
from sklearn.metrics import root_mean_squared_error     # For evaluating model performance





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
print(f"Using device: {device}")

ticker = '^NDX'  # NASDAQ 100 index
start_date = '2020-01-01'  # Start date for historical data

df = yf.download(ticker, start=start_date)  

df.Close.plot(title=f"{ticker} Close Price")  # Plot the closing prices

scaler = StandardScaler()  

df['Close'] = scaler.fit_transform(df['Close'])  

seq_length = 30 
data = []

for i in range(len(df) - seq_length):
    data.append(df.Close[i:i + seq_length])

data = np.array(data)

train_size = int(len(data) * 0.8)

X_train = torch.from_numpy(data[:train_size, :-1, :])
Y_train = torch.from_numpy(data[:train_size, -1, :])    #the last element

X_test = torch.from_numpy(data[train_size:, :-1, :])
Y_test = torch.from_numpy(data[train_size:, -1, :])

class LSTMModel(nn.Module):

    def __init__(self,input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
    