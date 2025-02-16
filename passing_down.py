import torch
import pandas as pd
import numpy as np
import os
import time
import dask.dataframe as dd

import get_data
import stat_encodings

class PassingDown():

    def __init__(self):
        pass


    def get_defensive_features_at_snap(self, play_id, game_id, player_plays_data, passing_play_data, game_data, player_data, tracking_data, verbose=False):

        try:

            # if verbose:
            #     get_data.print_game_state(play_id, game_id, game_data, passing_play_data)

            # Get starting index to obtain all 22 players for a specific play within 'player_plays_data'
            start = play_id
            temp = player_plays_data[(player_plays_data['playId'] == play_id) & (player_plays_data['gameId'] == game_id)]
            first_row_play_index = temp.index[0]
            start = first_row_play_index

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

            player_play_data = player_plays_data[['gameId', 'playId', 'nflId', 'teamAbbr']][start:start+22]

            play = passing_play_data[(passing_play_data['playId'] == play_id) & (passing_play_data['gameId'] == game_id)].iloc[0]

            # EXTRACT pff_passCoverage TO USE AS TARGET, if it's not a Top coverage type, ignore
            target_y = stat_encodings.encode_passCoverage(play['pff_passCoverage'])
            if target_y == -1:
                print('\tExcluded')
                return torch.zeros(54)
            
            # Extract pff_manZone
            man_zone = stat_encodings.encode_manZone(play['pff_manZone'])

            # Extract receiverAlignment
            alignment = stat_encodings.encode_receiverAlignment(play['receiverAlignment'])

            # Extract possessionTeamScoreDiff
            play_temp = pd.DataFrame([play])
            home_team = game_data['homeTeamAbbr']
            visitor_team = game_data['visitorTeamAbbr']
            play_temp['homeTeamAbbr'] = home_team
            play_temp['visitorTeamAbbr'] = visitor_team
            play_temp['possessionTeamScoreDiff'] = play_temp.apply(stat_encodings.get_possessionTeamScoreDiff, axis=1)
            possessionTeamScoreDiff = play_temp['possessionTeamScoreDiff'].iloc[0]

            # Extract Quarter
            quarter = play['quarter']

            # Extract Down and Distance, convert to a single value
            down = play['down']
            yards_to_go = stat_encodings.encode_yardsToGo(play['yardsToGo'])

            # Extract Center position to get y-axis of ball placement at the snap
            offensive_ids = player_play_data[player_play_data['teamAbbr'] == play['possessionTeam']]['nflId'].to_list()
            center_id = 0
            for id in offensive_ids:
                player = player_data[player_data['nflId'] == id].iloc[0]
                if player['position'] == 'C':
                    center_id = id
                    break
            center_tracking_data = tracking_data[(tracking_data['gameId'] == game_id) & 
                (tracking_data['playId'] == play_id) & 
                (tracking_data['nflId'] == center_id) & 
                (tracking_data['frameType'] == 'SNAP')
            ]
            ball_y_coord = center_tracking_data['y'].iloc[0]


            
            defensive_players_ids = player_play_data[player_play_data['teamAbbr'] != play['possessionTeam']]['nflId'].to_list()

            features = {}

            positions_analyzed = []

            deepest_safety_depth = -1.0
            next_deepest_safety_depth = -1.0

            all_depths = []
            all_y_coords = []

            lb_depths = []
            lb_y_coords = []

            dbs = ['FS', 'SS', 'CB']
            db_depths = []
            db_y_coords = []

            cb_depths = []

            safeties = ['FS', 'SS']
            safety_depths = []
            safety_y_coords = []

            # Count how many defenders are inside/outside 10 yards of the ball's y-axis
            middle_field_count = 0
            outside_field_count = 0
            middle_thresh = 10

            near_los_thresh = 1.5 # Near LoS = within 1.5 yards of LoS
            in_box_x_thresh = 5 # In box = within 5 yards of LoS, within 10 yards on either side of ball
            defenders_near_los = 0
            defenders_in_box = 0

            # If both safeties are 10+ yards away from LoS, Middle of Field is open
            mof_open_thresh = 10


            for i,player_id in enumerate(defensive_players_ids):
                player_tracking_data = tracking_data.query(
                    'gameId == @game_id and playId == @play_id and nflId == @player_id and frameType == "SNAP"'
                )

                player = player_data[player_data['nflId'] == player_id].iloc[0]
                player_x_coord_at_snap = player_tracking_data['x'].iloc[0]
                player_y_coord_at_snap = player_tracking_data['y'].iloc[0]
                player_jersey_num = int(player_tracking_data['jerseyNumber'].iloc[0])
                player_position = player['position']
                player_dist_to_los = np.round(np.abs(play['absoluteYardlineNumber'] - player_x_coord_at_snap), 4)


                # Get depths of all defenders
                all_depths.append(player_dist_to_los)

                # Get y-axes of all defenders
                all_y_coords.append(player_y_coord_at_snap)

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

                # Get LB depths and spreads
                if 'LB' in player_position:
                    lb_depths.append(player_dist_to_los)
                    lb_y_coords.append(player_y_coord_at_snap)

                 # Get Safety depths and spreads
                if player_position in safeties:
                    safety_depths.append(player_dist_to_los)
                    safety_y_coords.append(player_y_coord_at_snap)

                # Get DB depths and spreads
                if player_position in dbs:
                    db_depths.append(player_dist_to_los)
                    db_y_coords.append(player_y_coord_at_snap)
                
                # Get # of defenders in middle of the field and outside
                if np.round(np.abs(player_y_coord_at_snap - ball_y_coord), 4) <= middle_thresh:
                    middle_field_count += 1
                else:
                    outside_field_count += 1

                # Get # of defenders near the line of scrimmage
                if player_dist_to_los <= near_los_thresh:
                    defenders_near_los += 1

                # Get # of defenders in the box (within 5 yards of LoS and within 10 yards of the ball placement (y-axis))
                if player_dist_to_los <= in_box_x_thresh and np.round(np.abs(player_y_coord_at_snap - ball_y_coord), 4) <= middle_thresh:
                    defenders_in_box += 1


                positions_analyzed.append(player_position)

                # Record player coordinates into features
                features[f'defender{i+1}_x'] = player_dist_to_los
                features[f'defender{i+1}_y'] = player_y_coord_at_snap


                if verbose:
                    print(f"#{player_jersey_num} {player_position}\tdist from LoS: {player_dist_to_los},\ty:{player_y_coord_at_snap}")

                
            # Average defender-to-deepest-safety distance
            defender_safety_distances = [abs(d_x - deepest_safety_depth) for d_x in all_depths]
            avg_defender_to_deepest_safety_depth = np.round(np.mean(defender_safety_distances), 4)

            # Lateral spread of defenders
            defender_lateral_spread = np.round(max(all_y_coords) - min(all_y_coords), 4)

            # Lateral spread of LBs
            lb_lateral_spread = np.round(max(lb_y_coords) - min(lb_y_coords), 4)

            # Lateral spread of Safeties
            safety_lateral_spread = np.round(max(safety_y_coords) - min(safety_y_coords), 4)

            # Middle of Field open?
            mof_open = ((deepest_safety_depth > mof_open_thresh) & (next_deepest_safety_depth > mof_open_thresh)).astype(int)

                
            # Combine all input features
            features['offensive_alignment'] = alignment
            features['possessionTeamScoreDiff'] = possessionTeamScoreDiff
            features['quarter'] = quarter
            features['down'] = down
            features['yards_to_go'] = yards_to_go
            features['deepest_safety_depth'] = deepest_safety_depth
            features['next_deepest_safety_depth'] = next_deepest_safety_depth
            features['middle_field_count'] = middle_field_count
            features['outside_field_count'] = outside_field_count
            features['defenders_near_los'] = defenders_near_los
            features['defenders_in_box'] = defenders_in_box
            features['avg_defender_depth'] = np.round(np.mean(all_depths), 4)
            features['std_defender_depth'] = np.round(np.std(all_depths), 4)
            features['avg_cb_depth'] = np.round(np.mean(cb_depths), 4)
            features['min_cb_depth'] = np.min(cb_depths)
            features['max_cb_depth'] = np.max(cb_depths)
            features['avg_defender_to_deepest_safety_depth'] = np.round(avg_defender_to_deepest_safety_depth, 4)
            features['defender_lateral_spread'] = defender_lateral_spread
            features['std_lb_depth'] = np.round(np.std(lb_depths), 4)
            features['lb_lateral_spread'] = lb_lateral_spread
            features['avg_lb_depth'] = np.round(np.mean(lb_depths), 4)
            features['min_lb_depth'] = np.min(lb_depths)
            features['max_lb_depth'] = np.max(lb_depths)
            features['std_db_depth'] = np.round(np.std(db_depths), 4)
            features['avg_db_depth'] = np.round(np.mean(db_depths), 4)
            features['safety_lateral_spread'] = safety_lateral_spread
            features['std_safety_depth'] = np.round(np.std(safety_depths), 4)
            features['avg_safety_depth'] = np.round(np.mean(safety_depths), 4)
            features['safety_depth_diff'] = np.round(deepest_safety_depth - next_deepest_safety_depth, 4)
            features['mof_open'] = mof_open
            features['man_zone'] = man_zone
            features['target_y'] = target_y

            if verbose:
                print('TARGET_Y:\t\t\t', target_y)
                print('Man or Zone:\t\t\t', man_zone)
                print('offensive_alignment:\t\t', alignment)
                print('possessionTeamScoreDiff:\t', possessionTeamScoreDiff)
                print('quarter:\t\t\t', quarter)
                print('down:\t\t\t\t', down)
                print('yards_to_go:\t\t\t', yards_to_go)
                print('ball position (y-axis):\t\t', ball_y_coord)
                print('Avg defender depth:\t\t', np.round(np.mean(all_depths), 4))
                print('Std defender depth:\t\t', np.round(np.std(all_depths), 4))
                print('Deepest safety depth:\t\t', deepest_safety_depth)
                print('Next deepest safety:\t\t', next_deepest_safety_depth)
                print('Defenders in middle of the field:', middle_field_count)
                print('Defenders in outside of the field:', outside_field_count)
                print('Defenders near LoS:\t\t', defenders_near_los)
                print('Defenders in box:\t\t\t', defenders_in_box)
                print('Avg CB depth:\t\t\t', np.round(np.mean(cb_depths), 4))
                print('Min CB depth:\t\t\t', np.min(cb_depths))
                print('Max CB depth:\t\t\t', np.min(cb_depths))
                print('Avg LB depth:\t\t\t', np.round(np.mean(lb_depths), 4))
                print('Std LB depth:\t\t\t', np.round(np.std(lb_depths), 4))
                print('Min LB depth:\t\t\t', np.min(lb_depths))
                print('Max LB depth:\t\t\t', np.min(lb_depths))
                print('LB lateral spread:\t\t', lb_lateral_spread)
                print('Std DB depth:\t\t\t', np.round(np.std(db_depths), 4))
                print('Avg DB depth:\t\t\t', np.round(np.mean(db_depths), 4))
                print('Avg safety depth:\t\t', np.round(np.mean(safety_depths), 4))
                print('Std safety depth:\t\t', np.round(np.std(safety_depths), 4))
                print('Safety lateral spread:\t\t', safety_lateral_spread)
                print('Safety depth diff:\t\t', np.round(deepest_safety_depth - next_deepest_safety_depth), 4)
                print('Mof Open:\t\t\t', mof_open)
                print('Avg defender depth to deepest safety:', np.round(avg_defender_to_deepest_safety_depth), 4)
                print('defender lateral spread:\t', defender_lateral_spread)
            

            features_list = list(features.values())
            features_tensor = torch.tensor(features_list)

            if verbose:
                print(features_tensor)

            return features_tensor
        
        except Exception as e:
            print('\tError processing:', e)
            return torch.zeros(54)



    def get_defensive_features_for_passing_plays(self):
        print('getting data')
        player_plays_data = get_data.player_play_2022()
        play_data = get_data.plays_2022()
        game_data = get_data.games_2022()
        player_data = get_data.players_2022()
        # tracking_data = get_data.get_tracking_data_week_7()

        tracking_week_1, tracking_week_2, tracking_week_3, tracking_week_4, tracking_week_5, tracking_week_6, tracking_week_7, tracking_week_8, tracking_week_9 = get_data.load_tracking_data()
        # tracking_data = get_data.get_tracking_data_week_7()

        all_play_features = []

        print('all plays length:', len(play_data))

        # Filter Play data by Passing plays only (9736 total rows)
        passing_play_data = play_data[play_data['passResult'].notna()]
        print('all passing play length:', len(passing_play_data))

        count = 0
        start = 0
        limit = 2500
        for i,passing_play in passing_play_data.iterrows():
            count += 1
            if count <= start:
                # print('skipping line', count)
                continue  # Skip first 2500 rows
            if count > limit:
                break

            print(f'analyzing play {count}/{limit}')

            play_id = passing_play['playId']
            game_id = passing_play['gameId']
            game = game_data[game_data['gameId'] == game_id].iloc[0]
            week = game['week']

            # get_data.print_game_state(play_id, game_id, game_data, passing_play_data)

            match week:
                case 1:
                    tracking_data = tracking_week_1
                    # print('Using Tracking Data Week 1')
                case 2:
                    tracking_data = tracking_week_2
                    # print('Using Tracking Data Week 2')
                case 3:
                    tracking_data = tracking_week_3
                    # print('Using Tracking Data Week 3')
                case 4:
                    tracking_data = tracking_week_4
                    # print('Using Tracking Data Week 4')
                case 5:
                    tracking_data = tracking_week_5
                    # print('Using Tracking Data Week 5')
                case 6:
                    tracking_data = tracking_week_6
                    # print('Using Tracking Data Week 6')
                case 7:
                    tracking_data = tracking_week_7
                    # print('Using Tracking Data Week 7')
                case 8:
                    tracking_data = tracking_week_8
                    # print('Using Tracking Data Week 8')
                case 9:
                    tracking_data = tracking_week_9
                    # print('Using Tracking Data Week 9')
                case _:
                    print('Error - could not find Tracking Data')

            all_play_features.append(self.get_defensive_features_at_snap(play_id, game_id, player_plays_data, passing_play_data, game, player_data, tracking_data))

        all_play_features_tensor = torch.stack(all_play_features)
        print('FINAL OUTPUT:', all_play_features_tensor.shape)

        # Save output
        numpy_array = all_play_features_tensor.numpy()
        columns = ['defender1_x', 'defender1_y', 'defender2_x', 'defender2_y', 'defender3_x', 'defender3_y', 
        'defender4_x', 'defender4_y', 'defender5_x', 'defender5_y', 'defender6_x', 'defender6_y',
        'defender7_x', 'defender7_y', 'defender8_x', 'defender8_y', 'defender9_x', 'defender9_y', 
        'defender10_x', 'defender10_y', 'defender11_x', 'defender11_y', 'offensive_alignment', 
        'possessionTeamScoreDiff', 'quarter', 'down', 'yards_to_go', 'deepest_safety_depth', 
        'next_deepest_safety_depth', 'middle_field_count', 'outside_field_count', 'players_near_los', 
        'players_in_box', 'avg_defender_depth', 'std_defender_depth', 'avg_cb_depth', 'min_cb_depth', 'max_cb_depth',
        'avg_defender_to_deepest_safety_depth', 'defender_lateral_spread', 'std_lb_depth', 'lb_lateral_spread', 
        'avg_lb_depth', 'min_lb_depth', 'max_lb_depth', 'std_db_depth', 'avg_db_depth', 'safety_lateral_spread', 'std_safety_depth', 
        'avg_safety_depth', 'safety_depth_diff', 'mof_open', 'man_zone', 'target_y']
        df = pd.DataFrame(numpy_array, columns=columns)
        df.to_csv(f'features/play_features_pffCoverage_{len(columns)}features_{start}-{limit}.csv', index=False)


