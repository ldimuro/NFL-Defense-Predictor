import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import get_data
import stat_encodings

class Passing3rdDown(nn.Module):

    def __init__(self):
        super(Passing3rdDown, self).__init__()

        # self.games_data = games_data

        self.input_features = ['homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'receiverAlignment', 'passLength', 'passResult', 'targetX', 'targetY', 'yardsGained', 'pff_passCoverage', 'pff_manZone']

        # self.train_x, self.train_y, self.test_x, self.test_y = self.process_data(plays_data, self.input_features)

        # Layers
        self.linear_layer1 = nn.Linear(12, 32)
        self.relu1 = nn.ReLU()
        self.linear_layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.linear_layer3 = nn.Linear(16, 8)
        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Linear(8, 2)

    def process_data(self, data, input_features, normalize=True):
        data = pd.DataFrame(data)

        # Get 2022 Games data
        games_data = get_data.games_2022()

        # Append Home and Visitor values from Game dataset to each play
        data = data.merge(games_data[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']], on='gameId', how='right')

        # Filter Play data by 3rd Down Passing plays
        third_down_play_data = data[(data['down']) == 3 & (data['passResult'].notna())]

        # Remove all columns except input_features
        third_down_play_data = third_down_play_data.filter(items=input_features)

        # with pd.option_context('display.max_rows', None):
        print('3rd Down Play Data\n', third_down_play_data)

        # Convert preSnapHomeScore and preSnapVisitorScore to possessionTeamScoreDiff
        third_down_play_data['possessionTeamScoreDiff'] = third_down_play_data.apply(stat_encodings.get_possessionTeamScoreDiff, axis=1)
        
        # Encode yardsToGo
        third_down_play_data['yardsToGo'] = third_down_play_data['yardsToGo'].apply(stat_encodings.encode_yardsToGo)

        # Encode passLength
        third_down_play_data['passLength'] = third_down_play_data['passLength'].apply(stat_encodings.encode_passLength)

        # Encode offenseFormation
        third_down_play_data['offenseFormation'] = third_down_play_data['offenseFormation'].apply(stat_encodings.encode_offenseFormation)

        # Encode receiverAlignment
        third_down_play_data['receiverAlignment'] = third_down_play_data['receiverAlignment'].apply(stat_encodings.encode_receiverAlignment)

        # Encode passCoverage
        third_down_play_data['pff_passCoverage'] = third_down_play_data['pff_passCoverage'].apply(stat_encodings.encode_passCoverage)

        # Encode manZone
        third_down_play_data['pff_manZone'] = third_down_play_data['pff_manZone'].apply(stat_encodings.encode_manZone)

        # Encode targetY
        third_down_play_data['targetY'] = third_down_play_data['targetY'].apply(stat_encodings.encode_targetY)

        # Encode passResult
        third_down_play_data['passResult'] = third_down_play_data['passResult'].apply(stat_encodings.encode_passResult) 

        # Remove leftover text columns
        third_down_play_data.drop(columns=['preSnapHomeScore', 'preSnapVisitorScore', 'homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam', 'targetX'], inplace=True)

        # Remove all NaNs
        third_down_play_data = third_down_play_data.replace([np.inf, -np.inf], np.nan).fillna(-1).astype(int)

        # Add "firstDown" (True = 1/False = 0) column as y-label
        third_down_play_data['firstDown'] = (third_down_play_data['yardsGained'] >= third_down_play_data['yardsToGo']).astype(int)

        print('Encoded 3rd Down Play Data\n', third_down_play_data)

        # Convert to Torch Tensor
        third_down_play_tensor = torch.tensor(third_down_play_data.values, dtype=torch.float32)

        # Split into Tensor_X and Tensor_Y
        tensor_x = third_down_play_tensor[:, :-1] # All rows, all columns except last
        tensor_y = third_down_play_tensor[:, -1]  # All rows, last column only

        maj_class_count = torch.bincount(tensor_y.to(torch.int)).max().item()
        print(f'Baseline Accuracy:\t{np.round((maj_class_count/tensor_y.shape[0])*100, 2)}%')

        # Normalize numerical features
        if normalize:
            mean = tensor_x.mean(dim=0)
            std = tensor_x.std(dim=0)
            tensor_x = (tensor_x - mean) / std

        # Divide into Train set and Test set
        split_index = int(0.8 * tensor_x.shape[0])
        train_x = tensor_x[:split_index]
        train_y = tensor_y[:split_index]
        test_x = tensor_x[split_index:]
        test_y = tensor_y[split_index:]

        train_y = train_y.long()
        test_y = test_y.long()

        return train_x, train_y, test_x, test_y
    

    def forward(self, x):
        x = self.relu1(self.linear_layer1(x))
        x = self.relu2(self.linear_layer2(x))
        x = self.sigmoid(self.linear_layer3(x))
        x = self.output_layer(x)
        return x
    

    def train_model(self):
        plays_data = get_data.plays_2022()
        train_x, train_y, test_x, test_y = self.process_data(plays_data, self.input_features)

        num_epochs = 10
        batch_size = 32
        dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = Passing3rdDown()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()  # Zero gradients

                # Forward pass with the entire training dataset
                y_pred = model(batch_x) # Predict for all training data
                loss = loss_fn(y_pred.squeeze(), batch_y) # Compute loss for all data

                loss.backward() # Backward pass
                optimizer.step() # Update weights

                # Calculate accuracy for the training set
                with torch.no_grad():
                    correct = (torch.argmax(y_pred, dim=1) == batch_y).sum().item()
                    train_accuracy = correct / batch_y.shape[0]

                print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {train_accuracy:.2%}")

        model.eval()  # Switch to evaluation mode
        with torch.no_grad():  # Disable gradients for evaluation
            y_test_pred = model(test_x)
            test_correct = (torch.argmax(y_test_pred, dim=1) == test_y).sum().item()
            test_accuracy = test_correct / test_y.shape[0]

        print(f"Test Accuracy FNN:\t{test_accuracy:.2%} (Train {train_accuracy:.2%})")

