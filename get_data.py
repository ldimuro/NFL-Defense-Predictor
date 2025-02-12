import pandas as pd
import torch
import stat_encodings
import dask.dataframe as dd
# from pro_football_reference_web_scraper import player_game_log as nfl_player
# from pro_football_reference_web_scraper import team_game_log as nfl_team

def print_game_state(play_id, game_id, game_data, play_data):
        print('======================================================================================================')
        game = game_data[game_data['gameId'] == game_id].iloc[0]
        play = play_data[(play_data['playId'] == play_id) & (play_data['gameId'] == game_id)].iloc[0]
        print('GAME ID:', game_id)
        print('PLAY_ID:', play_id)
        print(f"{game['gameDate']} | Week {game['week']} | {game['gameTimeEastern']} EST")
        print(f"{game['homeTeamAbbr']} {play['preSnapHomeScore']} - {play['preSnapVisitorScore']} {game['visitorTeamAbbr']} | {play['absoluteYardlineNumber']} yd line | {play['possessionTeam']}'s ball")
        print(f"Q{play['quarter']} [{play['down']}&{play['yardsToGo']}] {play['playDescription']}")
        print('======================================================================================================')

def plays_2022():
    file_path = 'data/plays_2022.csv'
    data = pd.read_csv(file_path)
    return data

def games_2022():
    file_path = 'data/games_2022.csv'
    data = pd.read_csv(file_path)
    return data

def defense_2022():
    file_path = 'data/defense_2022.csv'
    defense = pd.read_csv(file_path)

    file_path = 'data/passing_defense_2022.csv'
    passing_defense = pd.read_csv(file_path)

    # file_path = 'data/advanced_defense_2022.csv'
    # advanced_defense = pd.read_csv(file_path)[defensive_team_full]

    return defense, passing_defense#, advanced_defense

def offense_2022():
    file_path = 'data/offense_2022.csv'
    offense = pd.read_csv(file_path)

    file_path = 'data/passing_offense_2022.csv'
    passing_offense = pd.read_csv(file_path)

    return offense, passing_offense

def player_play_2022():
    file_path = 'data/player_play_2022.csv'
    player_play = pd.read_csv(file_path)
    return player_play

def players_2022():
    file_path = 'data/players_2022.csv'
    players = pd.read_csv(file_path)
    return players

def get_tracking_data_week_1():
    file_path = 'data/tracking_week_1.csv'
    tracking_data = pd.read_csv(file_path)
    return tracking_data

def get_tracking_data_week_7():
    file_path = 'data//tracking_data/tracking_week_7.csv'
    tracking_data = pd.read_csv(file_path)
    return tracking_data

def load_tracking_data():
    file_path = 'data/tracking_data/tracking_week_1.csv'
    tracking_week_1 = pd.read_csv(file_path)
    print('Loaded tracking_week_1.csv')

    file_path = 'data/tracking_data/tracking_week_2.csv'
    tracking_week_2 = pd.read_csv(file_path)
    print('Loaded tracking_week_2.csv')

    file_path = 'data/tracking_data/tracking_week_3.csv'
    tracking_week_3 = pd.read_csv(file_path)
    print('Loaded tracking_week_3.csv')

    file_path = 'data/tracking_data/tracking_week_4.csv'
    tracking_week_4 = pd.read_csv(file_path)
    print('Loaded tracking_week_4.csv')

    file_path = 'data/tracking_data/tracking_week_5.csv'
    tracking_week_5 = pd.read_csv(file_path)
    print('Loaded tracking_week_5.csv')

    file_path = 'data/tracking_data/tracking_week_6.csv'
    tracking_week_6 = pd.read_csv(file_path)
    print('Loaded tracking_week_6.csv')

    file_path = 'data/tracking_data/tracking_week_7.csv'
    tracking_week_7 = pd.read_csv(file_path)
    print('Loaded tracking_week_7.csv')

    file_path = 'data/tracking_data/tracking_week_8.csv'
    tracking_week_8 = pd.read_csv(file_path)
    print('Loaded tracking_week_8.csv')

    file_path = 'data/tracking_data/tracking_week_9.csv'
    tracking_week_9 = pd.read_csv(file_path)
    print('Loaded tracking_week_9.csv')

    return tracking_week_1, tracking_week_2, tracking_week_3, tracking_week_4, tracking_week_5, tracking_week_6, tracking_week_7, tracking_week_8, tracking_week_9


def get_tracking_data(self, week, play_id, game_id):
    chunk_size = 100000
    file_path = f'data/tracking_week_{week}.csv'
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Filter the chunk based on play_id and game_id
        filtered_chunk = chunk[(chunk['playId'] == play_id) & (chunk['gameId'] == game_id)]
        
        if not filtered_chunk.empty:
            return filtered_chunk  # Return the rows where the play_id and game_id match
    
    return None  # Return None if no match was found


def play_by_play():
    # Load the CSV file into a DataFrame
    file_path = 'data/play_by_play_2024.csv'  # Replace with the actual file path
    df = pd.read_csv(file_path)

    # Display the first few rows to ensure it loaded correctly
    # print(df.head())

    # List all column values in the first row
    print("\nColumn values in the first row:")
    first_row = df.iloc[0]
    for column, value in first_row.items():
        print(f"{column}: {value}")
    

# def pro_football_player_data(player, position, season):
#     game_log = nfl_player.get_player_game_log(player = player, position = position, season = season)
#     return game_log

# def pro_football_team_data(team, season):
#     game_log = nfl_team.get_team_game_log(team = team, season = season)
#     return game_log