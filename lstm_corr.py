import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import talib
import warnings

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning)


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed
random_seed = 10
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Define the ticker symbol
ticker_symbol = "AMD"

def give_data(sym):
    # Calculate the start and end dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=3000)
    # Fetch data
    data = yf.download(sym, start=start_date, end=end_date)
    # Add a date column with year-month-day format
    data['Date'] = data.index.strftime('%Y-%m-%d')

    data.reset_index(drop=True, inplace=True)
    # Calculate MACD, RSI, and MA
    # data['MACD'], data['MACD Signal'], _ = talib.MACD(data['Close'])
    # data['RSI'] = talib.RSI(data['Close'])
    # data['MA'] = data['Close'].rolling(window=20).mean()
    return data[['Close','Date']]

amd = give_data('AMD')
intel = give_data('INTC')
nvidia = give_data('NVDA')

data = amd.merge(intel, on='Date', suffixes=('_AMD', '_INTC')).merge(nvidia, on='Date', suffixes=('_AMD', '_NVDA'))

# Split data into train and test sets
train = data.iloc[:int(len(data) * 0.9)]
test = data.iloc[int(len(data) * 0.9):]

train.dropna(inplace=True)
test.dropna(inplace=True)

inp_columns = ['Close_AMD','Close_INTC', 'Close']
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_features = scaler.fit_transform(train[inp_columns].values)
test_features = scaler.transform(test[inp_columns].values)

# Define function to create sequences for LSTM
def create_sequences(data, lookback):
    x = []
    y = []
    for i in range(lookback, data.shape[0]):
        x.append(data[i - lookback: i])
        y.append(data[i, 0])
    return torch.tensor(x, device=device), torch.tensor(y, device=device)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, fcc, output_size, num_layers, dropout_rate, l2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, fcc)
        self.fc2 = nn.Linear(fcc, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.l2 = l2

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# # Define hyperparameters to search over
# param_grid = {
#     'lookback': [10, 20,50,100],
#     'dropout_rate': [0.1],
#     'epochs': [100],
#     'batch_size': [64],
#     'l2': [0.005],
#     'fc' : [10,20,30,50,100],
#     'units' : [5,10,15],
#     'layers' : [1,2]
# }
#
# best_loss = float('inf')
# best_params = None
#
# # Loop over each combination of hyperparameters
# for fc in param_grid['fc']:
#     for layer in param_grid['layers']:
#         for units in param_grid['units']:
#             for lookback in param_grid['lookback']:
#                 for dropout_rate in param_grid['dropout_rate']:
#                     for epochs in param_grid['epochs']:
#                         for batch_size in param_grid['batch_size']:
#                             for l2 in param_grid['l2']:
#                                 # Create the model
#                                 model = LSTMModel(input_size=len(inp_columns), hidden_size=units, fcc=fc, output_size=1, num_layers=layer,
#                                                   dropout_rate=dropout_rate, l2=l2).to(device)
#
#                                 # Define optimizer and loss function
#                                 optimizer = optim.Adam(model.parameters())
#                                 criterion = nn.MSELoss()
#
#                                 # Prepare data with the current lookback
#                                 x_train, y_train = create_sequences(train_features, lookback)
#                                 x_test, y_test = create_sequences(test_features, lookback)
#
#                                 # Initialize lists to store training and validation loss
#                                 val_loss_history = []
#
#                                 # Training loop
#                                 model.train()
#                                 for epoch in tqdm(range(epochs), desc=f"fc={fc}, layer={layer}, Unit={units}, lookback={lookback}, dropout={dropout_rate}, epochs={epochs}, batch_size={batch_size}, l2={l2}"):
#                                     train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
#                                     running_train_loss = 0.0
#                                     for inputs, targets in train_loader:
#                                         optimizer.zero_grad()
#                                         outputs = model(inputs.float())
#                                         loss = criterion(outputs.squeeze(), targets.float())
#                                         loss.backward()
#                                         optimizer.step()
#                                         running_train_loss += loss.item() * inputs.size(0)
#                                     epoch_train_loss = running_train_loss / len(train_loader.dataset)
#
#                                     # Validation
#                                     model.eval()
#                                     with torch.no_grad():
#                                         outputs = model(x_test.float())
#                                         val_loss = criterion(outputs.squeeze(), y_test.float())
#                                         val_loss_history.append(val_loss.item())
#                                     model.train()
#
#                                 # Check if this model is the best so far
#                                 avg_val_loss = np.mean(val_loss_history)
#                                 print(avg_val_loss)
#                                 if avg_val_loss < best_loss:
#                                     best_loss = avg_val_loss
#                                     best_params = {
#                                         'lookback': lookback,
#                                         'dropout_rate': dropout_rate,
#                                         'epochs': epochs,
#                                         'batch_size': batch_size,
#                                         'l2': l2,
#                                         'unit' : units,
#                                         'layers' : layer
#                                     }
#
# # Print the best hyperparameters
# print("Best Hyperparameters:", best_params)

#{fc:100 'lookback': 20, 'dropout_rate': 0.1, 'epochs': 100, 'batch_size': 12, 'l2': 0.005, 'unit': 15, 'layers': 2}



##################################################
# Find the best model based on the test loss
best_model_params = {'lookback': 100, 'dropout_rate': 0.1, 'epochs': 100, 'batch_size': 8, 'l2': 0.001}

# Create the best model
best_model = LSTMModel(input_size=len(inp_columns), hidden_size=5, fcc=50, output_size=1, num_layers=1,
                       dropout_rate=best_model_params['dropout_rate'], l2=best_model_params['l2']).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(best_model.parameters())
criterion = nn.MSELoss()

# Prepare data with the best lookback
x_train_best, y_train_best = create_sequences(train_features, best_model_params['lookback'])
train_dataset = TensorDataset(x_train_best, y_train_best)
train_loader = DataLoader(train_dataset, batch_size=best_model_params['batch_size'], shuffle=True)
x_test, y_test = create_sequences(test_features, best_model_params['lookback'])

# Initialize lists to store training and validation loss
train_loss_history = []
val_loss_history = []

# Training loop for the best model
best_model.train()
for epoch in tqdm(range(best_model_params['epochs'])):
    running_train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = best_model(inputs.float())
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        running_train_loss += loss.item() * inputs.size(0)

    # Calculate average training loss for the epoch
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_loss_history.append(epoch_train_loss)

    # Validation
    best_model.eval()
    with torch.no_grad():
        outputs = best_model(x_test.float())
        val_loss = criterion(outputs.squeeze(), y_test.float())
        val_loss_history.append(val_loss.item())
    best_model.train()

# # Plot training and validation loss
# plt.plot(train_loss_history, label='Training Loss')
# plt.plot(val_loss_history, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

def inverse_process(data,cols):
    temp = []
    for d in data:
        d = list(d)
        for i in range(cols-1):
            d = d + [0]
        temp.append(d)
    return temp

# Make predictions using the best model
best_model.eval()
with torch.no_grad():
    y_pred_best = best_model(torch.tensor(x_test, dtype=torch.float, device=device))
    y_pred_best = y_pred_best.cpu().numpy().reshape(-1, 1)
    y_test = y_test.cpu().numpy().reshape(-1, 1)  # Move y_test to CPU memory
    y_pred_best = inverse_process(y_pred_best,len(inp_columns))
    y_test = inverse_process(y_test,len(inp_columns))
    y_pred_rescaled_best = scaler.inverse_transform(y_pred_best)[:,0]
    y_test_rescaled = scaler.inverse_transform(y_test)[:,0]

# # Plot the rescaled original and predicted prices
# plt.figure(figsize=(12, 6))
# plt.plot(y_test_rescaled, 'b', label="Original Price")
# plt.plot(y_pred_rescaled_best, 'r', label="Predicted Price")
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# plt.show()

for i, j in zip(y_pred_rescaled_best[-5:], y_test_rescaled[-5:]):
    print(i, j)

loss = 0
for i, j in zip(y_pred_rescaled_best[-5:], y_test_rescaled[-5:]):
    loss += abs(i-j)
print('5 days : ',round(loss,3))

loss = 0
for i, j in zip(y_pred_rescaled_best[-10:], y_test_rescaled[-10:]):
    loss += abs(i-j)
print('10 days : ',round(loss,3))
loss = 0

for i, j in zip(y_pred_rescaled_best[-15:], y_test_rescaled[-15:]):
    loss += abs(i-j)
print('15 days : ',round(loss,3))

import pandas as pd
def data_process(temp):
    #temp = pd.DataFrame(temp['Close'],columns=['Close'])
    # Calculate MACD, RSI, and MA
    temp['MACD'], _, _ = talib.MACD(temp['Close'])
    temp['RSI'] = talib.RSI(temp['Close'])
    temp['MA'] = temp['Close'].rolling(window=20).mean()
    temp.dropna(inplace=True)
    temp.reset_index(inplace=True,drop=True)
    return temp


look_main = 0
main_series = data[look_main:].copy()
main_series = main_series[inp_columns]
predictions_temp = []
for i in range(24):
    temp = data_process(main_series)
    future_pred = scaler.transform(temp[inp_columns].values)
    future_pred = future_pred[look_main:]

    f_x_test, f_pred = create_sequences(future_pred, best_model_params['lookback'])
    f_x_test = np.array(f_x_test[-1].cpu())

    temp_pred = scaler.inverse_transform(f_x_test[-1:].reshape(1,-1))
    print(i+1,temp_pred[0][0],sep='  -> ')
    temp = torch.tensor(f_x_test, dtype=torch.float,device=device).reshape(1,best_model_params['lookback'],len(inp_columns))  # Convert to tensor and add batch dimension
    with torch.no_grad():
        temp_pred = best_model(temp).cpu().numpy()  # Predict using the trained model
    temp_pred = inverse_process(temp_pred,len(inp_columns))
    temp_pred_rescaled = scaler.inverse_transform(temp_pred)  # Inverse transform the prediction
    predictions_temp.append(temp_pred_rescaled[0][0])
    #temp = scaler.inverse_transform(temp.cpu())
    last_row = np.array(inverse_process([[temp_pred_rescaled[0][0]]],len(inp_columns))[0])
    last_row = pd.DataFrame(last_row.reshape(1,len(inp_columns)),columns=inp_columns)
    main_series = pd.concat([main_series , last_row],ignore_index=True)

