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

X_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
Y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
Y_test = torch.from_numpy(data[train_size:, -1, :]).type

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
    
model = LSTMModel(input_dim=1, hidden_dim=50, num_layers=2, output_dim=1).to(device) 

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

for i in range(num_epochs):
    y_train_pred = model(X_train)

    loss = criterion(y_train_pred, Y_train)

    if i % 25 == 0:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
y_test_pred = model(X_test)

y_train_pred = scaler.inverse_transform(y_train_pred.cpu().detach().numpy())
Y_train = scaler.inverse_transform(Y_train.cpu().detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.cpu().detach().numpy())
Y_test = scaler.inverse_transform(Y_test.cpu().detach().numpy())

train_rmse =  root_mean_squared_error(Y_train[:,0], y_train_pred[:,0])
test_rmse = root_mean_squared_error(Y_test[:,0], y_test_pred[:,0])

fig = plt.figure(figsize=(12,10))

gs = fig.add_gridspec(4, 1)

ax1 = fig.add_subplot(gs[0:3, 0])

ax1.plot(df.iloc[len(y_test):].index, y_test, color = 'purple', label='Actual Price')
ax1.plot(df.iloc[len(y_test):].index, y_test_pred, color = 'cyan', label='Predicted Price')
ax1.legend()
plt.title(f"{ticker} Price Prediction")
plt.xlabel('Date')
plt.ylabel('Price')