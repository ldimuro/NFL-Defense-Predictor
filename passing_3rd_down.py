import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import stat_encodings

class Passing3rdDown(nn.Module):

    def __init__(self, plays_data, games_data):
        super(Passing3rdDown, self).__init__()

        self.games_data = games_data

        self.input_features = ['homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'receiverAlignment', 'passLength', 'passResult', 'targetX', 'targetY', 'yardsGained', 'pff_passCoverage', 'pff_manZone']

        self.train_x, self.train_y, self.test_x, self.test_y = self.process_data(plays_data, self.input_features)

    def process_data(self, data, input_features, normalize=True):
        data = pd.DataFrame(data)

        # Append Home and Visitor values from Game dataset to each play
        data = data.merge(self.games_data[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']], on='gameId', how='right')

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
