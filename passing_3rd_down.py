import torch
import torch.nn as nn
import pandas as pd

class Passing3rdDown(nn.Module):

    def __init__(self, play_data):
        super(Passing3rdDown, self).__init__()

        # Filter Play data by 3rd Down Passing plays
        self.data = play_data[(play_data['down']) == 3 & (play_data['passResult'].notna())]
        
        self.input_features = ['yardsToGo', 'yardlineNumber', 'quarter', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'receiverAlignment', 'passLength', 'targetX', 'targetY', 'yardsGained', 'pff_passCoverage', 'pff_manZone']

    def test(self):
        print('data:', self.data)