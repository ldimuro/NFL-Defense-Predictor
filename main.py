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

    is_training = False
    
    if is_training:
        passing_down_model = PassingDown()
        start_time = time.perf_counter()
        passing_down_model.get_defensive_features_for_passing_plays()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function took {elapsed_time} seconds to complete.")

    # Obtain input features
    data = pd.read_csv('features/play_features_pffCoverage_21features_passplays0-8500.csv')

    # Column types
    x_coords = ['defender1_x', 'defender2_x', 'defender3_x', 'defender4_x', 'defender5_x', 'defender6_x', 'defender7_x', 'defender8_x', 'defender9_x', 'defender10_x', 'defender11_x']
    y_coords = ['defender1_y', 'defender2_y', 'defender3_y', 'defender4_y', 'defender5_y', 'defender6_y', 'defender7_y', 'defender8_y', 'defender9_y', 'defender10_y', 'defender11_y']
    game_state = ['offensive_alignment', 'quarter', 'down', 'yards_to_go', 'possessionTeamScoreDiff']
    depth_data = ['deepest_safety_depth', 'next_deepest_safety_depth', 'avg_defender_depth', 'std_defender_depth', 'avg_cb_depth', 'min_cb_depth', 'avg_defender_to_deepest_safety_depth', 'std_lb_depth', 'avg_lb_depth', 'std_db_depth', 'avg_db_depth', 'safety_depth_diff', 'std_safety_depth', 'avg_safety_depth']
    lateral_data = ['defender_lateral_spread', 'lb_lateral_spread']
    std_data = ['std_db_depth', 'std_lb_depth', 'std_defender_depth', 'std_safety_depth']
    avg_data = ['avg_defender_depth', 'avg_cb_depth', 'avg_defender_to_deepest_safety_depth', 'avg_lb_depth', 'avg_db_depth']
    db_data = ['std_db_depth', 'avg_db_depth']
    lb_data = ['std_lb_depth', 'lb_lateral_spread', 'avg_lb_depth']
    safety_data = ['safety_depth_diff', 'std_safety_depth', 'avg_safety_depth']
    test = ['avg_defender_to_deepest_safety_depth']

    # BEST RESULTS
    data = data.drop(columns=(x_coords + y_coords + db_data + safety_data + test)) # test = 'avg_defender_to_deepest_safety_depth'
    # data = data.drop(columns=(x_coords))


    # Add Man/Zone into Coverage (e.g. Cover-1 Man, Cover-1 Zone, etc.)
    # data['target_y'] = data['target_y'] * 2 + data['man_zone']
    data = data.drop(columns=['man_zone'])


    print('======================================================================================================')
    print('INPUT')
    print('data:', data.shape, data.columns)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    y_distribution = y.value_counts()
    print(y_distribution)

    rt = random_tree.RandomForest(x, y, data)



    # net = FNN()
    # net.train_model(x, y)




    print('======================================================================================================')

    # IDEAS:
    # - Correlation between QB Height and batted passes?
    # - Predictive model for 3rd Down playcalls
    # - Predicting Optimal Pass Location (using targetX, targetY, and passLength)
    #       - pair with opposing defense's stats to (e.g. teams with bad secondaries should be attacked more)
    #       - Example: "On 3rd and 6 against Cover-3 Zone, your best option is targeting the short-middle, where similar plays have a 65% conversion success rate."

if __name__ == "__main__":
    main()