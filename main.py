import numpy as np
import get_data
from passing_down import PassingDown
import random_tree
import fnn
import time
import pandas as pd
from fnn import FNN
import torch
import random
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    check_random_state(seed_value)


def main():

    set_seed()

    passing_down_model = PassingDown()
    # passing_down_model.train_model()
    # passing_down_model.RandomForest()
    # passing_down_model.estimated_rushers_on_play()

    # start_time = time.perf_counter()
    # passing_down_model.get_defensive_features_for_passing_plays()
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Function took {elapsed_time} seconds to complete.")



    # RUN MODELS
    # data1 = pd.read_csv('play_features_pffCoverage_40features_0-2500.csv')
    # data2 = pd.read_csv('play_features_pffCoverage_40features_2500-5500.csv')
    # data3 = pd.read_csv('play_features_pffCoverage_40features_5500-8500.csv')
    # data4 = pd.read_csv('play_features_pffCoverage_40features_8500-9736.csv')
    # data = pd.concat([data1, data2, data3, data4], ignore_index=True)
    # data.to_csv(f'play_features_pffCoverage_40features_0-9736.csv', index=False)
    data = pd.read_csv('play_features_pffCoverage_40features_0-8500.csv')

    # Remove all defender coordinate data
    x_coord = ['defender1_x', 'defender2_x', 'defender3_x', 'defender4_x', 'defender5_x', 'defender6_x', 'defender7_x', 'defender8_x', 'defender9_x', 'defender10_x', 'defender11_x']
    y_coord = ['defender1_y', 'defender2_y', 'defender3_y', 'defender4_y', 'defender5_y', 'defender6_y', 'defender7_y', 'defender8_y', 'defender9_y', 'defender10_y', 'defender11_y']
    columns_to_drop = ['defender1_x', 'defender2_x', 'defender3_x', 'defender4_x', 'defender5_x', 'defender6_x', 'defender7_x',
                       'defender8_x', 'defender9_x', 'defender10_x', 'defender11_x', 'defender1_y', 'defender2_y', 'defender3_y', 
                       'defender4_y', 'defender5_y', 'defender6_y', 'defender7_y', 'defender8_y', 'defender9_y', 'defender10_y', 
                       'defender11_y']#, 'deepest_safety_depth','next_deepest_safety_depth', 'avg_safety_distance']#'middle_field_count','outside_field_count','players_near_los','players_in_box','avg_defender_depth','std_defender_depth','avg_cb_depth','min_cb_depth','avg_safety_distance','lateral_spread']
    data = data.drop(columns=x_coord)

    print('data:', data.shape, data.columns)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    y_distribution = y.value_counts()
    print(y_distribution)

    majority_class_count = y_distribution.max()
    print(f'BASELINE ACCURACY: {(majority_class_count / y.shape[0])*100:.2f}%')

    rt = random_tree.RandomForest(x, y)



    net = FNN()
    net.train_model(x, y)




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