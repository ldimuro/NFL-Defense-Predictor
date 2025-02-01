import numpy as np
import get_data
from passing_3rd_down import Passing3rdDown

def main():
    # PRO FOOTBALL REFERENCE
    # kyler_murray_stats = pro_football_player_data('Kyler Murray', 'QB', 2024)
    # cardinals_2008_stats = pro_football_team_data('Arizona Cardinals', 2008)
    # print(kyler_murray_stats)
    # print(cardinals_2008_stats)

    plays_data = get_data.plays_2022()
    # games_data = get_data.games_2022()

    passing_3rd_down_model = Passing3rdDown()
    # passing_3rd_down_model.train_model()
    # passing_3rd_down_model.RandomForest()
    # passing_3rd_down_model.estimated_rushers_on_play()
    passing_3rd_down_model.get_defense_coordinates_at_snap()

    # List all column values in the first row
    # 0 = Pass
    # 32 = Run
    # first_row = plays_data.iloc[0]
    # for column, value in first_row.items():
    #     print(f'{column}: {value}')

    print('======================================================================================================')

    # IDEAS:
    # - Correlation between QB Height and batted passes?
    # - Predictive model for 3rd Down playcalls
    # - Predicting Optimal Pass Location (using targetX, targetY, and passLength)
    #       - pair with opposing defense's stats to (e.g. teams with bad secondaries should be attacked more)
    #       - Example: "On 3rd and 6 against Cover-3 Zone, your best option is targeting the short-middle, where similar plays have a 65% conversion success rate."

    # third_down_plays = plays_data[(plays_data['down']) == 3 & (plays_data['passResult'].notna())]
    # print('3rd down:', third_down_plays)

    # average_to_go = third_down_plays['yardsToGo'].mean()
    # print(f'Average Yards to Go on 3rd Down {average_to_go}')

    # average_yards_gained = third_down_plays['yardsGained'].mean()
    # print(f'Average Yards Gained on 3rd Down: {average_yards_gained}')

    # average_time_to_throw = third_down_plays['timeToThrow'].mean()
    # print(f'Average Time to Throw on 3rd Down: {average_time_to_throw}')

    # average_pass_length = third_down_plays['passLength'].mean()
    # print(f'Average Pass Length on 3rd Down: {average_pass_length}')

    # average_target_x = third_down_plays['targetX'].mean()
    # print(f'Average Target X of Pass: {average_target_x}')
    # average_target_y = third_down_plays['targetY'].median()
    # print(f'Average Target Y of Pass: {average_target_y}')

    # pass_coverage_counts = plays_data['pff_passCoverage'].value_counts()
    # print('pff_passCoverage on 3rd Down:', pass_coverage_counts)

    # pass_location_type = third_down_plays['passLocationType'].value_counts()
    # print('passLocationType on 3rd Down:', pass_location_type)

    # man_zone_counts = third_down_plays['pff_manZone'].value_counts()
    # print('pff_manZone on 3rd Down:', man_zone_counts)

    # receiver_alignments = third_down_plays['receiverAlignment'].value_counts()
    # print('receiver_alignments on 3rd Down:', receiver_alignments)

    # offense_formation = plays_data['offenseFormation'].value_counts()
    # print('offense_formation on 3rd Down:', offense_formation)



if __name__ == "__main__":
    main()