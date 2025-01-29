import torch
import torch.nn as nn
import pandas as pd

class Passing3rdDown(nn.Module):

    def __init__(self, play_data):
        super(Passing3rdDown, self).__init__()

        # # Filter Play data by 3rd Down Passing plays
        # self.data = play_data[(play_data['down']) == 3 & (play_data['passResult'].notna())]

        self.input_features = ['yardsToGo', 'absoluteYardlineNumber', 'quarter', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'receiverAlignment', 'passLength', 'passResult', 'targetX', 'targetY', 'yardsGained', 'pff_passCoverage', 'pff_manZone']
        self.play_data = self.process_data(play_data, self.input_features)
        print(self.play_data)

    def process_data(self, data, input_features):
        data = pd.DataFrame(data)

        # Filter Play data by 3rd Down Passing plays
        third_down_play_data = data[(data['down']) == 3 & (data['passResult'].notna())]

        # Remove all columns except input_features
        third_down_play_data = third_down_play_data.filter(items=input_features)

        # Add "firstDown" (True/False) column
        third_down_play_data['firstDown'] = third_down_play_data['yardsGained'] >= third_down_play_data['yardsToGo']

        print(third_down_play_data)

        # Discretize yardsToGo
        third_down_play_data['yardsToGo'] = third_down_play_data['yardsToGo'].apply(self.discretize_yardsToGo)

        # Discretize passLength
        third_down_play_data['passLength'] = third_down_play_data['passLength'].apply(self.discretize_passLength)

        # Discretize offenseFormation
        third_down_play_data['offenseFormation'] = third_down_play_data['offenseFormation'].apply(self.discretize_offenseFormation)

        # Discreteize receiverAlignment
        third_down_play_data['receiverAlignment'] = third_down_play_data['receiverAlignment'].apply(self.discretize_receiverAlignment)

        # Discretize passCoverage
        third_down_play_data['pff_passCoverage'] = third_down_play_data['pff_passCoverage'].apply(self.discretize_passCoverage)

        # Discretize manZone
        third_down_play_data['pff_manZone'] = third_down_play_data['pff_manZone'].apply(self.discretize_manZone)

        # Discretize targetX
        third_down_play_data['targetX'] = third_down_play_data['targetX'].apply(self.discretize_targetX)


        return third_down_play_data
    
    def discretize_yardsToGo(self, yards_to_go):
        # Inches= 1
        # Short = 2-5
        # Med   = 6-9
        # Long  = 10+

        # Inches
        if yards_to_go == 1:
            return 0
        # Short
        elif yards_to_go > 1 and yards_to_go <= 5:
            return 1
        # Medium
        elif yards_to_go > 5 and yards_to_go <= 9:
            return 2
        # Deep
        else:
            return 3
        
    def discretize_passLength(self, pass_length):
        # Short = 0-7
        # Med   = 8-15
        # Deep  = 16+
        
        if pd.isna(pass_length):
            return -1
        # Short
        elif pass_length <= 7:
            return 0
        # Medium
        elif pass_length > 7 and pass_length <= 15:
            return 1
        # Deep
        else:
            return 2
        
    def discretize_offenseFormation(self, formation):
        if formation == 'SHOTGUN':
            return 0
        elif formation == 'SINGLEBACK':
            return 1
        elif formation == 'EMPTY':
            return 2
        elif formation == 'I_FORM':
            return 3
        elif formation == 'PISTOL':
            return 4
        elif formation == 'JUMBO':
            return 5
        elif formation == 'WILDCAT':
            return 6
        
    def discretize_receiverAlignment(self, alignment):
        if alignment == '2x2':
            return 0
        elif alignment == '3x1':
            return 1
        elif alignment == '2x1':
            return 2
        elif alignment == '3x2':
            return 3
        elif alignment == '4x1':
            return 4
        elif alignment == '1x1':
            return 5
        elif alignment == '2x0':
            return 6
        
    def discretize_passCoverage(self, coverage):
        if pd.isna(coverage):
            return -1
        
        if 'Cover-1' in coverage:
            return 0
        elif 'Cover-2' in coverage:
            return 1
        elif 'Cover-3' in coverage:
            return 2
        elif 'Cover 6' in coverage:
            return 3
        elif coverage == 'Quarters':
            return 4
        elif coverage == 'Red Zone':
            return 5
        elif coverage == 'Cover-0':
            return 6
        elif coverage == '2-Man':
            return 7
        elif coverage == 'Prevent':
            return 8
        elif coverage == 'Goal Line':
            return 9
        elif coverage == 'Bracket':
            return 10
        elif coverage == 'Miscellaneous':
            return 11
        
    def discretize_manZone(self, val):
        if val == 'Man':
            return 0
        elif val == 'Zone':
            return 1
        else:
            return 2
        
    def discretize_targetX(self, targetX):
        # Middle     = area between the hash marks
        # Left hash  = 23.9 yards from sideline
        # Hash width = 6.2 yards

        if targetX <= 23.9:
            return 0
        elif targetX > 23.9 and targetX <= 29.4:
            return 1
        else:
            return 2
