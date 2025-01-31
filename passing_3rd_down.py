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

        self.input_features = ['down', 'homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'receiverAlignment', 'passLength', 'passResult', 'targetX', 'targetY', 'pff_passCoverage', 'pff_manZone', 'yardsGained', 'nonPossessionTeam', 'full_team_name', 'defense_Rk', 'defense_Cmp%', 'defense_TD%', 'defense_PD', 'defense_Int%', 'defense_ANY/A', 'defense_Rate', 'defense_QBHits', 'defense_TFL', 'defense_Sk%', 'defense_EXP', 'offense_Rk', 'offense_Cmp%', 'offense_TD%', 'offense_Int%', 'offense_ANY/A', 'offense_Rate', 'offense_Sk%', 'offense_EXP', 'firstDown']

        # self.train_x, self.train_y, self.test_x, self.test_y = self.process_data(plays_data, self.input_features)

        # Layers
        self.linear_layer1 = nn.Linear(28, 32)
        self.relu1 = nn.ReLU()
        self.linear_layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.linear_layer3 = nn.Linear(16, 8)
        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Linear(8, 1)

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
        # play_data = data[(data['down']) == 3 & (data['passResult'].notna())]
        play_data = data[data['passResult'].notna()]
        # play_data = data

        # Remove all columns except input_features
        play_data = play_data.filter(items=input_features)

        # with pd.option_context('display.max_rows', None):
        print('3rd Down Play Data\n', play_data)

        # Convert preSnapHomeScore and preSnapVisitorScore to possessionTeamScoreDiff
        play_data['possessionTeamScoreDiff'] = play_data.apply(stat_encodings.get_possessionTeamScoreDiff, axis=1)
        
        # Encode yardsToGo
        play_data['yardsToGo'] = play_data['yardsToGo'].apply(stat_encodings.encode_yardsToGo)

        # Encode passLength
        play_data['passLength'] = play_data['passLength'].apply(stat_encodings.encode_passLength)

        # Encode offenseFormation
        play_data['offenseFormation'] = play_data['offenseFormation'].apply(stat_encodings.encode_offenseFormation)

        # Encode receiverAlignment
        play_data['receiverAlignment'] = play_data['receiverAlignment'].apply(stat_encodings.encode_receiverAlignment)

        # Encode passCoverage
        play_data['pff_passCoverage'] = play_data['pff_passCoverage'].apply(stat_encodings.encode_passCoverage)

        # Encode manZone
        play_data['pff_manZone'] = play_data['pff_manZone'].apply(stat_encodings.encode_manZone)

        # Encode targetY
        play_data['targetY'] = play_data['targetY'].apply(stat_encodings.encode_targetY)

        # Encode passResult
        play_data['passResult'] = play_data['passResult'].apply(stat_encodings.encode_passResult)


        # Encode offensive and defensive stats
        off_def_stats = ['offense_Cmp%', 'defense_Cmp%', 'offense_TD%', 'defense_TD%', 'offense_Int%', 'defense_Int%', 'offense_ANY/A', 'defense_ANY/A', 'offense_Rate', 'defense_Rate', 'offense_Sk%', 'defense_Sk%', 'offense_EXP', 'defense_EXP']
        for each in off_def_stats:
            play_data[each] = play_data[each].astype(float)
            play_data[each] = pd.qcut(play_data[each], q=4, labels=[0, 1, 2, 3]).astype(int)


        # Remove leftover text columns
        play_data.drop(columns=['preSnapHomeScore', 'preSnapVisitorScore', 'homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam', 'targetX', 'nonPossessionTeam', 'full_team_name'], inplace=True)

        # Remove all NaNs
        play_data = play_data.replace([np.inf, -np.inf], np.nan).fillna(-1).astype(int)


        # Encode passLocation using targetY and passLength
        play_data['passLocation'] = play_data.apply(stat_encodings.encode_passLocation, axis=1)
        play_data['passLocation'] = play_data['passLocation'].replace(-1, 9)
        

        # Add "firstDown" (True = 1/False = 0) column as y-label
        play_data['firstDown'] = (play_data['yardsGained'] >= play_data['yardsToGo']).astype(int)

        # Move stats to the end of the DataFrame
        # play_data['targetY'] = play_data.pop('targetY')
        # play_data['passLength'] = play_data.pop('passLength')
        # play_data['pff_manZone'] = play_data['pff_manZone'].replace(-1, 2)
        # play_data['pff_passCoverage'] = play_data.pop('pff_passCoverage')
        # play_data['pff_passCoverage'] = play_data['pff_passCoverage'].replace(-1, 2)

        

        # play_data = play_data.drop(columns=['yardsGained', 'passLength', 'targetY', 'passResult'])



        # features = ['down', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone', 'possessionTeamScoreDiff',
        #             'defense_Rk', 'defense_Cmp%', 'defense_TD%', 'defense_PD', 'defense_Int%', 'defense_ANY/A', 'defense_Rate', 'defense_QBHits', 'defense_TFL', 'defense_Sk%', 
        #             'defense_EXP', 'offense_Rk', 'offense_Cmp%', 'offense_TD%', 'offense_Int%', 'offense_ANY/A', 'offense_Rate', 'offense_Sk%', 'offense_EXP', 'passLocation']
        # features = ['down', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone', 'possessionTeamScoreDiff',
        #             'defense_Rk', 'defense_Cmp%', 'defense_TD%', 'defense_PD', 'defense_Int%', 'defense_ANY/A', 'defense_Rate', 'defense_QBHits', 'defense_TFL', 'defense_Sk%', 
        #             'defense_EXP', 'offense_Rk', 'offense_Cmp%', 'offense_TD%', 'offense_Int%', 'offense_ANY/A', 'offense_Rate', 'offense_Sk%', 'offense_EXP', 'firstDown']
        # play_data = play_data.filter(features)





        print('Encoded 3rd Down Play Data\n', play_data)
        print(play_data.columns)


        # Convert to Torch Tensor
        third_down_play_tensor = torch.tensor(play_data.values, dtype=torch.float32)

        # Split into Tensor_X and Tensor_Y
        tensor_x = third_down_play_tensor[:, :-1] # All rows, all columns except last
        tensor_y = third_down_play_tensor[:, -1]  # All rows, last column only

        print('tensor_y:', tensor_y.shape)

        maj_class_count = torch.bincount(tensor_y.to(torch.int)).max().item()
        print(maj_class_count)
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


        # Defensive Coverage Effectiveness: Which Coverages Minimize Yards Gained?
        # 
        #   

        '''
        Defensive Coverage Effectiveness: Which Coverages Minimize Yards Gained?
        ================================================================================
        Goal: Identify which defensive coverage types (pff_passCoverage, pff_manZone) are most effective at limiting passing yards.
        Features: pff_passCoverage, pff_manZone, passLength, passResult, targetY, yardsGained, defense stats
        - Train a regression model (Neural Network, XGBoost, Random Forest) to predict yardsGained based on coverage.
        - Evaluate if certain defensive strategies reduce passing yardage more effectively.
        - Compare performance across different offensive formations
        Does Cover-2 or Cover-3 limit passing yards better than Man coverage?
        Are certain coverages more vulnerable against deep passes?
        How does offensive scheme influence defensive coverage effectiveness?
        '''

        '''
        Optimize Pass Location for First Down Probability
        ================================================================================
        Goal: Determine the best pass location (short, medium, deep, left, middle, right) for a given game situation to maximize first down probability.
        Features: All features except passLocation, Train a model to predict the best pass location instead of using actual data.
        - Train a classification model (Neural Net, XGBoost, Decision Tree, etc.) with passLocation as the label.
        - Compare predicted best locations vs. actual thrown locations.
        - Perform counterfactual analysis: What if the QB had thrown to a different location?
        Are certain locations systematically underused despite high success rates?
        Which defensive schemes (pff_passCoverage) make certain locations more vulnerable?
        '''
        features = ['down', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone', 'possessionTeamScoreDiff',
                    'defense_Rk', 'defense_Cmp%', 'defense_TD%', 'defense_PD', 'defense_Int%', 'defense_ANY/A', 'defense_Rate', 'defense_QBHits', 'defense_TFL', 'defense_Sk%', 
                    'defense_EXP', 'offense_Rk', 'offense_Cmp%', 'offense_TD%', 'offense_Int%', 'offense_ANY/A', 'offense_Rate', 'offense_Sk%', 'offense_EXP', 'passLocation']

        print('train_x:', train_x)
        print('train_y:', train_y)

        '''
        Predict the Probability of a First Down (firstDown)
        ================================================================================
        Goal: Build a classification model to predict whether a pass play will result in a first down.
        Features: down, yardsToGo, absoluteYardlineNumber, quarter offenseFormation, receiverAlignment, passLength, passResult
                pff_passCoverage, pff_manZone, possessionTeamScoreDiff, passLocation, offense & defense metrics
        - Train a Feedforward Neural Network (FNN) with softmax or sigmoid output.
        - Use logistic regression, random forest, or XGBoost for comparison.
        - Interpret feature importance: Which features matter most for getting a first down?
        - Run SHAP analysis to understand how different inputs influence predictions.
        Does pass location (short left, deep middle, etc.) influence success?
        Which defensive coverages are most vulnerable to giving up first downs?
        How does offensive ranking impact success probability?
        '''
        features = ['down', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone', 'possessionTeamScoreDiff',
                    'defense_Rk', 'defense_Cmp%', 'defense_TD%', 'defense_PD', 'defense_Int%', 'defense_ANY/A', 'defense_Rate', 'defense_QBHits', 'defense_TFL', 'defense_Sk%', 
                    'defense_EXP', 'offense_Rk', 'offense_Cmp%', 'offense_TD%', 'offense_Int%', 'offense_ANY/A', 'offense_Rate', 'offense_Sk%', 'offense_EXP']

        num_epochs = 1
        batch_size = 32
        dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = Passing3rdDown()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
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

    
    def RandomForest(self):
        plays_data = get_data.plays_2022()
        train_x, train_y, test_x, test_y = self.process_data(plays_data, self.input_features)

        print('train_x:', train_x.shape)
        print('train_y:', train_y.shape)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(train_x, train_y)

        y_pred_rf = rf.predict(test_x)

        print("Random Forest Accuracy:", accuracy_score(test_y, y_pred_rf))
        print(classification_report(test_y, y_pred_rf))

