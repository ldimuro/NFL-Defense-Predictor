import pandas as pd
import torch
import stat_encodings
# from pro_football_reference_web_scraper import player_game_log as nfl_player
# from pro_football_reference_web_scraper import team_game_log as nfl_team

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