import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import get_data

class FNN(nn.Module):

    def __init__(self):
        super(FNN, self).__init__()

        # FFN Layers
        self.linear_layer1 = nn.Linear(28, 256)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        # self.dropout = nn.Dropout(0.2)

        self.linear_layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.linear_layer3 = nn.Linear(128, 64)

        self.output_layer = nn.Linear(64, 6)
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        # x = self.relu(self.linear_layer1(x))
        # x = self.relu(self.linear_layer2(x))
        # x = self.softmax(self.output_layer(x))

        x = self.relu(self.bn1(self.linear_layer1(x)))
        # x = self.dropout(x)
        x = self.relu(self.bn2(self.linear_layer2(x)))
        x = self.relu(self.linear_layer3(x))
        x = self.softmax(self.output_layer(x))
        return x
    

    def train_model(self, x, y):

        # Normalize features
        scalar = StandardScaler()
        x = scalar.fit_transform(x)

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

        # train_x = train_x.to_numpy()
        # test_x = test_x.to_numpy()
        # train_y = train_y.to_numpy()
        # test_y = test_y.to_numpy()

        train_x = torch.tensor(train_x, dtype=torch.float32)#.long()
        test_x = torch.tensor(test_x, dtype=torch.float32)#.long()
        train_y = torch.tensor(train_y, dtype=torch.float32).long()
        test_y = torch.tensor(test_y.values, dtype=torch.float32).long()

        print('train_x:', train_x.shape)
        print('train_y:', train_y.shape, train_y)


        # Defensive Coverage Effectiveness: Which Coverages Minimize Yards Gained?
        # 
        #   

        '''
        Defensive Coverage Effectiveness: Which Coverages Minimize Yards Gained?
        ================================================================================
        Goal: Identify which defensive coverage types (pff_passCoverage, pff_manZone) are most effective at limiting passing yards.
        Features: pff_passCoverage, pff_manZone, passLength, passResult, targetY, yardsGained, defense stats
        - Train a regression model (Neural Network, XGBoost, Random Forest) to predict yardsGained based on coverage.
        - Evaluate if certain defensive strategies reduce passing yardage more effectively.
        - Compare performance across different offensive formations
        Does Cover-2 or Cover-3 limit passing yards better than Man coverage?
        Are certain coverages more vulnerable against deep passes?
        How does offensive scheme influence defensive coverage effectiveness?
        '''

        '''
        Optimize Pass Location for First Down Probability
        ================================================================================
        Goal: Determine the best pass location (short, medium, deep, left, middle, right) for a given game situation to maximize first down probability.
        Features: All features except passLocation, Train a model to predict the best pass location instead of using actual data.
        - Train a classification model (Neural Net, XGBoost, Decision Tree, etc.) with passLocation as the label.
        - Compare predicted best locations vs. actual thrown locations.
        - Perform counterfactual analysis: What if the QB had thrown to a different location?
        Are certain locations systematically underused despite high success rates?
        Which defensive schemes (pff_passCoverage) make certain locations more vulnerable?
        '''
        features = ['down', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone', 'possessionTeamScoreDiff',
                    'defense_Rk', 'defense_Cmp%', 'defense_TD%', 'defense_PD', 'defense_Int%', 'defense_ANY/A', 'defense_Rate', 'defense_QBHits', 'defense_TFL', 'defense_Sk%', 
                    'defense_EXP', 'offense_Rk', 'offense_Cmp%', 'offense_TD%', 'offense_Int%', 'offense_ANY/A', 'offense_Rate', 'offense_Sk%', 'offense_EXP', 'passLocation']

        # print('train_x:', train_x)
        # print('train_y:', train_y)

        '''
        Predict the Probability of a First Down (firstDown)
        ================================================================================
        Goal: Build a classification model to predict whether a pass play will result in a first down.
        Features: down, yardsToGo, absoluteYardlineNumber, quarter offenseFormation, receiverAlignment, passLength, passResult
                pff_passCoverage, pff_manZone, possessionTeamScoreDiff, passLocation, offense & defense metrics
        - Train a Feedforward Neural Network (FNN) with softmax or sigmoid output.
        - Use logistic regression, random forest, or XGBoost for comparison.
        - Interpret feature importance: Which features matter most for getting a first down?
        - Run SHAP analysis to understand how different inputs influence predictions.
        Does pass location (short left, deep middle, etc.) influence success?
        Which defensive coverages are most vulnerable to giving up first downs?
        How does offensive ranking impact success probability?
        '''
        features = ['down', 'yardsToGo', 'absoluteYardlineNumber', 'quarter', 'offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone', 'possessionTeamScoreDiff',
                    'defense_Rk', 'defense_Cmp%', 'defense_TD%', 'defense_PD', 'defense_Int%', 'defense_ANY/A', 'defense_Rate', 'defense_QBHits', 'defense_TFL', 'defense_Sk%', 
                    'defense_EXP', 'offense_Rk', 'offense_Cmp%', 'offense_TD%', 'offense_Int%', 'offense_ANY/A', 'offense_Rate', 'offense_Sk%', 'offense_EXP']
        

        '''
        Predicting Sack Probability Given Pre-Snap Factors
        ================================================================================
        Goal: QB pressure drives passing success. This project predicts whether a pass will result in a sack based on game context
        Features: [Offense Formation, Dropback Type, Down, YardsToGo, PossessionTeamScoreDiff, Defensive Stats (QB Hits, TFL, Sack Rate)]
        Binary Classification (1 = Sack, 0 = No Sack)

        '''
        
        num_epochs = 100
        batch_size = 64#32
        best_acc = 0
        patience = 10
        patience_counter = 0
        # dataset = TensorDataset(train_x, train_y)
        # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = FNN()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for i in range(0, len(train_x), batch_size):
                batch_x = train_x[i:i+batch_size]
                batch_y = train_y[i:i+batch_size]

                optimizer.zero_grad()  # Zero gradients

                # Forward pass with the entire training dataset
                y_pred = model(batch_x) # Predict for all training data
                loss = loss_fn(y_pred.squeeze(), batch_y) # Compute loss for all data

                loss.backward() # Backward pass
                optimizer.step() # Update weights

                # Calculate accuracy for the training set
                # with torch.no_grad():
                #     correct = (torch.argmax(y_pred, dim=1) == train_y).sum().item()
                #     train_accuracy = correct / train_y.shape[0]

                    # print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {train_accuracy:.2%}")

            # Validation accuracy
            model.eval()
            with torch.no_grad():
                val_outputs = model(test_x)
                val_preds = torch.argmax(val_outputs, dim=1)
                acc = (val_preds == test_y).float().mean().item()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Acc: {acc*100:.2f}%')

            # Early stopping
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0  # Reset counter if accuracy improves
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print('Early stopping triggered.')
                break

        print(f'BEST VAL ACCURACY: {best_acc*100:.2f}%')



        # print(f"Test Accuracy FNN:\t{test_accuracy:.2%} (Train {train_accuracy:.2%})")