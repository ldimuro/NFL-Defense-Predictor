import torch
import pandas as pd
import numpy as np

import get_data
import stat_encodings

class PassingDown():

    def __init__(self):
        # super(PassingDown, self).__init__()

        # self.games_data = games_data

        self.input_features = ['down', 'homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'receiverAlignment', 'passLength', 'passResult', 'targetX', 'targetY', 'pff_passCoverage', 'pff_manZone', 'yardsGained', 'nonPossessionTeam', 'full_team_name', 'defense_Rk', 'defense_Cmp%', 'defense_TD%', 'defense_PD', 'defense_Int%', 'defense_ANY/A', 'defense_Rate', 'defense_QBHits', 'defense_TFL', 'defense_Sk%', 'defense_EXP', 'offense_Rk', 'offense_Cmp%', 'offense_TD%', 'offense_Int%', 'offense_ANY/A', 'offense_Rate', 'offense_Sk%', 'offense_EXP', 'firstDown']

        # self.train_x, self.train_y, self.test_x, self.test_y = self.process_data(plays_data, self.input_features)


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
        self.print_game_state(play_id, game_id, game_data, play_data)

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

    def get_defensive_features_at_snap(self, play_id, game_id, player_plays_data, passing_play_data, game_data, player_data, tracking_data):
        start = play_id #327
        temp = player_plays_data[(player_plays_data['playId'] == play_id) & (player_plays_data['gameId'] == game_id)]
        # print('XXX:', temp)
        first_row_play_index = temp.index[0]
        start = first_row_play_index
        # print(first_row_play_index)

        # FEATURES TO EXTRACT (to predict pre-snap defensive shell)
        # - Each defender's distance to LOS (11 features)
        # - Each defender's y-coordinate (11 features)
        # - Deepest safety yardage (1 feature)
        # - 2nd deepest safety yardage (1 feature)
        # - Avg Defender depths (1 feature)
        # - Std Dev of defender depths (how spread out the defense is) (1 feature)
        # - Avg CB depth (1 feature)
        # - Min CB depth (1 feature)
        # - # of defenders inside hashmarks (1 feature)
        # - # of defenders outside numbers (1 feature)
        # - # of defenders within 1.5 yards of LOS (1 feature)
        # - # of defenders in the box (within 5 Yards of LOS & Inside Hashmarks) (1 feature)
        # POTENTIAL OTHER FEATURES:
        # - defender movement during offensive motion (1 feature)
        # - Safeties' Horizontal Spread (1 feature)
        

        print('getting state')

        # print(player_plays_data[['getOffTimeAsPassRusher', 'causedPressure', 'gameId', 'playId', 'nflId', 'teamAbbr']])

        # Get Game State
        # start = 327
        print('BEFORE\n:', player_plays_data)
        player_play_data = player_plays_data[['gameId', 'playId', 'nflId', 'teamAbbr']][start:start+22]

        print('start:', start)

        print(f'all 22 players on play_{play_id}:\n{player_play_data}')

        self.print_game_state(play_id, game_id, game_data, passing_play_data)

        play = passing_play_data[(passing_play_data['playId'] == play_id) & (passing_play_data['gameId'] == game_id)].iloc[0]

        # Extract Center position to get y-axis of ball placement at the snap
        print(player_play_data)
        offensive_ids = player_play_data[player_play_data['teamAbbr'] == play['possessionTeam']]['nflId'].to_list()
        print('offensive_ids:', offensive_ids)
        center_id = 0
        for id in offensive_ids:
            player = player_data[player_data['nflId'] == id].iloc[0]
            if player['position'] == 'C':
                center_id = id
                break

        print('center:', center_id)
        print(tracking_data)

        center_tracking_data = tracking_data[(tracking_data['gameId'] == game_id) & 
                    (tracking_data['playId'] == play_id) & 
                    (tracking_data['nflId'] == center_id) & 
                    (tracking_data['frameType'] == 'SNAP')]
        print('center_tracking_data:\n', center_tracking_data)
        ball_y_coord = center_tracking_data['y'].iloc[0]
        print('ball position (y-axis):', ball_y_coord)

        defensive_players_ids = player_play_data[player_play_data['teamAbbr'] != play['possessionTeam']]['nflId'].to_list()

        features = {}

        positions_analyzed = []

        deepest_safety_depth = -1.0
        next_deepest_safety_depth = -1.0

        all_depths = []
        cb_depths = []

        # Count how many players are inside/outside 10 yards of the ball's y-axis
        middle_field_count = 0
        outside_field_count = 0
        middle_thresh = 10

        near_los_thresh = 1.5 # Near LoS = within 1.5 yards of LoS
        in_box_x_thresh = 5 # In box = within 5 yards of LoS, within 10 yards on either side of ball
        players_near_los = 0
        players_in_box = 0

        for i,player_id in enumerate(defensive_players_ids):
            player_tracking_data = tracking_data[(tracking_data['gameId'] == game_id) & 
                    (tracking_data['playId'] == play_id) & 
                    (tracking_data['nflId'] == player_id) & 
                    (tracking_data['frameType'] == 'SNAP')]
            
            player = player_data[player_data['nflId'] == player_id].iloc[0]
            player_x_coord_at_snap = player_tracking_data['x'].iloc[0]
            player_y_coord_at_snap = player_tracking_data['y'].iloc[0]
            player_jersey_num = int(player_tracking_data['jerseyNumber'].iloc[0])
            player_position = player['position']
            player_dist_to_los = np.round(np.abs(play['absoluteYardlineNumber'] - player_x_coord_at_snap), 4)

            # Get depths of all defenders
            all_depths.append(player_dist_to_los)

            # Get depths of Safeties
            if player_position == 'FS' or player_position == 'SS':
                if player_dist_to_los > deepest_safety_depth:
                    if deepest_safety_depth == -1.0:
                        deepest_safety_depth = player_dist_to_los
                    else:
                        next_deepest_safety_depth = deepest_safety_depth
                        deepest_safety_depth = player_dist_to_los
                elif player_dist_to_los > next_deepest_safety_depth:
                    next_deepest_safety_depth = player_dist_to_los

            # Get CB depths
            if player_position == 'CB':
                cb_depths.append(player_dist_to_los)

            # Get # of players in middle of the field and outside
            if np.round(np.abs(player_y_coord_at_snap - ball_y_coord), 4) <= middle_thresh:
                middle_field_count += 1
            else:
                outside_field_count += 1

            # Get # of players near the line of scrimmage
            if player_dist_to_los <= near_los_thresh:
                players_near_los += 1

            # Get # of players in the box (within 5 yards of LoS and within 10 yards of the ball placement (y-axis))
            if player_dist_to_los <= in_box_x_thresh and np.round(np.abs(player_y_coord_at_snap - ball_y_coord), 4) <= middle_thresh:
                players_in_box += 1


            positions_analyzed.append(player_position)


            # Record player coordinates into features
            features[f'defender{i+1}_x'] = player_dist_to_los
            features[f'defender{i+1}_y'] = player_y_coord_at_snap
            # position_count = positions_analyzed.count(player_position)
            # features[f'{player_position}{position_count}_x'] = player_dist_to_los
            # features[f'{player_position}{position_count}_y'] = player_y_coord_at_snap

            print(f"#{player_jersey_num} {player_position}\tdist from LoS: {player_dist_to_los},\ty:{player_y_coord_at_snap}")

            
            
        # Combine all input features
        features['deepest_safety_depth'] = deepest_safety_depth
        features['next_deepest_safety_depth'] = next_deepest_safety_depth
        features['middle_field_count'] = middle_field_count
        features['outside_field_count'] = outside_field_count
        features['players_near_los'] = players_near_los
        features['players_in_box'] = players_in_box
        features['avg_defender_depth'] = np.round(np.mean(all_depths), 4)
        features['std_defender_depth'] = np.round(np.std(all_depths), 4)
        features['avg_cb_depth'] = np.round(np.mean(cb_depths), 4)
        features['min_cb_depth'] = np.min(cb_depths)

        # print('ESTIMATED RUSHERS:', estimated_rushers_count)
        print(features)
        print('Avg defender depth:', np.round(np.mean(all_depths), 4))
        print('Std defender depth:', np.round(np.std(all_depths), 4))
        print('Deepest safety depth:', deepest_safety_depth)
        print('Next deepest safety:', next_deepest_safety_depth)
        print('Players in middle of the field:', middle_field_count)
        print('Players in outside of the field:', outside_field_count)
        print('Players near LoS:', players_near_los)
        print('Players in box:', players_in_box)
        print('Avg CB depth:', np.mean(cb_depths))
        print('Min CB depth:', np.min(cb_depths))
        


        # print(player_play_data)


    def get_defensive_features_for_passing_plays(self):
        print('getting data')
        player_plays_data = get_data.player_play_2022()
        play_data = get_data.plays_2022()
        game_data = get_data.games_2022()
        player_data = get_data.players_2022()
        tracking_data = get_data.get_tracking_data_week_7()

        # Filter Play data by Passing plays only
        passing_play_data = play_data[play_data['passResult'].notna()]

        z = 0
        for i,passing_play in passing_play_data.iterrows():
            z += 1
            if z > 6:
                break

            play_id = passing_play['playId']
            game_id = passing_play['gameId']
            print('play_id:', play_id)
            print('game_id:', game_id)

            self.get_defensive_features_at_snap(play_id, game_id, player_plays_data, passing_play_data, game_data, player_data, tracking_data)


    def print_game_state(self, play_id, game_id, game_data, play_data):
        print('======================================================================================================')
        game = game_data[game_data['gameId'] == game_id].iloc[0]
        play = play_data[(play_data['playId'] == play_id) & (play_data['gameId'] == game_id)].iloc[0]
        print('GAME ID:', game_id)
        print('PLAY_ID:', play_id)
        print(f"{game['gameDate']} | Week {game['week']} {game['season']} | {game['gameTimeEastern']} EST")
        print(f"{game['homeTeamAbbr']} {play['preSnapHomeScore']} - {play['preSnapVisitorScore']} {game['visitorTeamAbbr']} | {play['absoluteYardlineNumber']} yd line | {play['possessionTeam']}'s ball")
        print(f"Q{play['quarter']} [{play['down']}&{play['yardsToGo']}] {play['playDescription']}")
        print('======================================================================================================')



        

