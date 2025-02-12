import get_data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import torch
import pandas as pd

def RandomForest(x, y):
    # plays_data = get_data.plays_2022()
    # train_x, train_y, test_x, test_y = self.process_data(plays_data, self.input_features)

    # Normalize numerical features
    # mean = tensor_x.mean(dim=0)
    # std = tensor_x.std(dim=0)
    # std[std == 0] = 1.0
    # tensor_x = (tensor_x - mean) / std

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape, train_y.value_counts())

    # Apply SMOTE only on training data
    # smote = SMOTE(sampling_strategy='auto', random_state=42)  # 'auto' balances minority classes
    # train_x_smote, train_y_smote = smote.fit_resample(train_x, train_y)

    # print("Class distribution AFTER SMOTE:")
    # print(pd.Series(train_y_smote).value_counts())

    # train_x_np = train_x.to_numpy()
    # test_x_np = train_x.to_numpy()
    # train_y_np = train_y.to_numpy()
    # test_y_np = test_y.to_numpy()

    # scaler = StandardScaler()
    # train_x_normalized = scaler.fit_transform(train_x)
    # test_x_normalized = scaler.transform(test_x)

    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [10, 20, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt']
    # }

    # rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    # grid_search.fit(train_x, train_y)

    # print(grid_search.best_params_)



    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(train_x, train_y)
    y_pred_rf = rf.predict(test_x)
    print('==============================================================================')
    print(f'Random Forest Accuracy: {accuracy_score(test_y, y_pred_rf)*100:.2f}%')
    print(classification_report(test_y, y_pred_rf))



    # xgb_model = XGBClassifier(
    #     n_estimators=50,       # Number of boosting rounds (trees)
    #     max_depth=5,            # Controls tree depth (prevents overfitting)
    #     learning_rate=0.1,      # Step size shrinkage to prevent overfitting
    #     # subsample=0.8,          # Randomly samples 80% of data for training (helps generalization)
    #     # colsample_bytree=0.8,   # Randomly samples 80% of features per tree
    #     # objective='multi:softmax',  # Multi-class classification
    #     # num_class=len(y.unique()),  # Number of coverage types (classes)
    #     random_state=42
    # )
    # xgb_model.fit(train_x, train_y)
    # y_pred_xgb = xgb_model.predict(test_x)
    # accuracy = accuracy_score(test_y, y_pred_xgb)
    # print('==============================================================================')
    # print(f'XGBoost Model Accuracy: {accuracy:.2%}')
    # print(classification_report(test_y, y_pred_xgb))

