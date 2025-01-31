import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import get_data
import stat_encodings

class Passing3rdDown(nn.Module):

    def __init__(self):
        super(Passing3rdDown, self).__init__()

        # self.games_data = games_data

        self.input_features = ['down', 'homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'receiverAlignment', 'passLength', 'passResult', 'targetX', 'targetY', 'pff_passCoverage', 'pff_manZone', 'yardsGained', 'nonPossessionTeam', 'full_team_name', 'defense_Rk', 'defense_Cmp%', 'defense_TD%', 'defense_PD', 'defense_Int%', 'defense_ANY/A', 'defense_Rate', 'defense_QBHits', 'defense_TFL', 'defense_Sk%', 'defense_EXP', 'offense_Rk', 'offense_Cmp%', 'offense_TD%', 'offense_Int%', 'offense_ANY/A', 'offense_Rate', 'offense_Sk%', 'offense_EXP']

        # self.train_x, self.train_y, self.test_x, self.test_y = self.process_data(plays_data, self.input_features)

        # Layers
        self.linear_layer1 = nn.Linear(11, 32)
        self.relu1 = nn.ReLU()
        self.linear_layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.linear_layer3 = nn.Linear(16, 8)
        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Linear(8, 10)

    def process_data(self, data, input_features, normalize=True):
        data = pd.DataFrame(data)

        # Get 2022 Games data
        games_data = get_data.games_2022()

        # Append Home and Visitor values from Game dataset to each play
        data = data.merge(games_data[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']], on='gameId', how='right')

        # Add defensive team
        data['nonPossessionTeam'] = data.apply(stat_encodings.get_defensive_team, axis=1)

        # Get offensive and defensive data
        defense_data, passing_defense_data = get_data.defense_2022()
        offense_data, passing_offense_data = get_data.offense_2022()

        # Add defensive stats for nonPossessionTeam
        defense_columns = ['Rk', 'Cmp%', 'TD%', 'PD', 'Int%', 'ANY/A', 'Rate', 'QBHits', 'TFL', 'Sk%', 'EXP']
        defense_stat_cols = {col: f'defense_{col}' for col in defense_columns}
        data['full_team_name'] = data['nonPossessionTeam'].map(stat_encodings.team_abbr_to_name)
        data = data.merge(passing_defense_data[['Tm'] + defense_columns].rename(columns={'Tm': 'full_team_name'}), on='full_team_name', how='left')
        data.rename(columns=defense_stat_cols, inplace=True)

        # Add offensive stats for possessionTeam
        offense_columns = ['Rk', 'Cmp%', 'TD%', 'Int%', 'ANY/A', 'Rate', 'Sk%', 'EXP']
        offense_stat_cols = {col: f'offense_{col}' for col in offense_columns}
        data['full_team_name'] = data['possessionTeam'].map(stat_encodings.team_abbr_to_name)
        data = data.merge(passing_offense_data[['Tm'] + offense_columns].rename(columns={'Tm': 'full_team_name'}), on='full_team_name', how='left')
        data.rename(columns=offense_stat_cols, inplace=True)

        

        # Filter Play data by 3rd Down Passing plays
        # third_down_play_data = data[(data['down']) == 3 & (data['passResult'].notna())]
        third_down_play_data = data[data['passResult'].notna()]
        # third_down_play_data = data

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


        # Encode offensive and defensive stats
        off_def_stats = ['offense_Cmp%', 'defense_Cmp%', 'offense_TD%', 'defense_TD%', 'offense_Int%', 'defense_Int%', 'offense_ANY/A', 'defense_ANY/A', 'offense_Rate', 'defense_Rate', 'offense_Sk%', 'defense_Sk%', 'offense_EXP', 'defense_EXP']
        for each in off_def_stats:
            third_down_play_data[each] = third_down_play_data[each].astype(float)
            third_down_play_data[each] = pd.qcut(third_down_play_data[each], q=4, labels=[0, 1, 2, 3]).astype(int)


        # Remove leftover text columns
        third_down_play_data.drop(columns=['preSnapHomeScore', 'preSnapVisitorScore', 'homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam', 'targetX', 'nonPossessionTeam', 'full_team_name'], inplace=True)

        # Remove all NaNs
        third_down_play_data = third_down_play_data.replace([np.inf, -np.inf], np.nan).fillna(-1).astype(int)

        

        # Add "firstDown" (True = 1/False = 0) column as y-label
        third_down_play_data['firstDown'] = (third_down_play_data['yardsGained'] >= third_down_play_data['yardsToGo']).astype(int)

        # Move stats to the end of the DataFrame
        # third_down_play_data['targetY'] = third_down_play_data.pop('targetY')
        # third_down_play_data['passLength'] = third_down_play_data.pop('passLength')
        # third_down_play_data['pff_manZone'] = third_down_play_data['pff_manZone'].replace(-1, 2)
        # third_down_play_data['pff_passCoverage'] = third_down_play_data.pop('pff_passCoverage')
        # third_down_play_data['pff_passCoverage'] = third_down_play_data['pff_passCoverage'].replace(-1, 2)

        # Encode passLocation using targetY and passLength
        third_down_play_data['passLocation'] = third_down_play_data.apply(stat_encodings.encode_passLocation, axis=1)
        third_down_play_data['passLocation'] = third_down_play_data['passLocation'].replace(-1, 9)
        

        

        # third_down_play_data = third_down_play_data.drop(columns=['yardsGained', 'passLength', 'targetY', 'passResult'])




        print('Encoded 3rd Down Play Data\n', third_down_play_data)

        # Convert to Torch Tensor
        third_down_play_tensor = torch.tensor(third_down_play_data.values, dtype=torch.float32)

        # Split into Tensor_X and Tensor_Y
        tensor_x = third_down_play_tensor[:, :-1] # All rows, all columns except last
        tensor_y = third_down_play_tensor[:, -1]  # All rows, last column only

        print('tensor_y:', tensor_y.shape)

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

        print('train_x:', train_x.shape)
        print('train_y:', train_y.shape)

        num_epochs = 1
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

