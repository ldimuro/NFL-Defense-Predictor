import pandas as pd
import torch
from pro_football_reference_web_scraper import player_game_log as nfl_player
from pro_football_reference_web_scraper import team_game_log as nfl_team

def plays():
    # Load the CSV file into a DataFrame
    file_path = 'plays_2022.csv'  # Replace with the actual file path
    data = pd.read_csv(file_path)

    return data


def play_by_play():
    # Load the CSV file into a DataFrame
    file_path = 'play_by_play_2024.csv'  # Replace with the actual file path
    df = pd.read_csv(file_path)

    # Display the first few rows to ensure it loaded correctly
    # print(df.head())

    # List all column values in the first row
    print("\nColumn values in the first row:")
    first_row = df.iloc[0]
    for column, value in first_row.items():
        print(f"{column}: {value}")
    

def pro_football_player_data(player, position, season):
    game_log = nfl_player.get_player_game_log(player = player, position = position, season = season)
    return game_log

def pro_football_team_data(team, season):
    game_log = nfl_team.get_team_game_log(team = team, season = season)
    return game_log