import stat_encodings
import get_data
import pandas as pd
import numpy as np
import torch

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


def estimated_rushers_on_play(self):
    player_plays_data = get_data.player_play_2022()
    play_data = get_data.plays_2022()
    game_data = get_data.games_2022()
    player_data = get_data.players_2022()

    # Filter Play data by Passing plays only
    play_data = play_data[play_data['passResult'].notna()]
    print('plays:', play_data)

    # blitz = 4387, 4396, 4439

    # All-out Blitz examples:
    # 4439: https://youtu.be/3PAFAYNi3mA?si=7M3f01dS8XBsYLvZ&t=447
    # 4262: https://youtu.be/gUvHlA1-JWQ?si=3h3o05MGlAbRwLKK&t=217

    start = 235 #2678, 2690
    
    # blitzes = 0
    # for i in range(0, 5000):

    # Get Game State
    player_play_data = player_plays_data[['getOffTimeAsPassRusher', 'causedPressure', 'gameId', 'playId', 'nflId']][start*22:(start*22)+22]
    game_id = player_play_data['gameId'].iloc[0]
    play_id = player_play_data['playId'].iloc[0]
    get_data.print_game_state(play_id, game_id, game_data, play_data)

    # Get defensive players whose getOffTimeAsPassRusher is not Nan or causedPressure
    estimate_rushers = player_play_data[pd.notna(player_play_data['getOffTimeAsPassRusher']) | player_play_data['causedPressure'] == True]
    estimated_rushers_count = estimate_rushers.shape[0]
    print(f'Estimated rushers {"(BLITZ🚨) " if estimated_rushers_count >= 5 else " "}({estimated_rushers_count}):')
    for i,rusher in estimate_rushers.iterrows():
        player_id = int(rusher['nflId'])
        player = player_data[player_data['nflId'] == player_id].iloc[0]
        getOffTimeAsPassRusher = estimate_rushers[estimate_rushers['nflId'] == player_id]['getOffTimeAsPassRusher'].iloc[0]
        causedPressure = estimate_rushers[estimate_rushers['nflId'] == player_id]['causedPressure'].iloc[0]
        print(f"{player['position']}\t{player['displayName'].split(' ')[0][0]}. {player['displayName'].split(' ')[1]}\t{np.round(getOffTimeAsPassRusher, 6)} getOffTimeAsPassRusher\tcausedPressure: {causedPressure}")
            
        # if estimated_rushers_count >= 5:
        #     blitzes += 1

    # print(f'Blitz Percentage: {np.round((blitzes/5000)*100, 3)}%')