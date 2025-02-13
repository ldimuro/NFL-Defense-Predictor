import get_data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from xgboost import XGBClassifier
# import lightgbm as lgb
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def RandomForest(x, y):

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # class_weights = {2.0: 1, 0.0: 1.2, 1.0: 1.5, 3.0: 2, 4.0: 1.3, 5.0: 1.5}
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42) # 42
    rf.fit(train_x, train_y)
    y_pred_rf = rf.predict(test_x)
    print('==============================================================================')
    print(f'Random Forest Accuracy: {accuracy_score(test_y, y_pred_rf)*100:.2f}%')
    print(classification_report(test_y, y_pred_rf))

    feature_importance = rf.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    features = np.array(train_x.columns)[sorted_idx]

    plt.figure(figsize=(10, 5))
    plt.barh(features[:10], feature_importance[sorted_idx][:10])
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Top 10 Most Important Features for Defensive Coverage Prediction")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')

    sorted_feature_importance = feature_importance[sorted_idx]

    # Print out all features and their corresponding scores
    print("Feature Importance Scores:")
    for feature, score in zip(features, sorted_feature_importance):
        print(f"{feature}: {score:.4f}")


    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 4, 6],
        'max_features': ['auto', 'sqrt']
    }

    # rf = RandomForestClassifier(random_state=42)
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    # grid_search.fit(train_x, train_y)

    # print(grid_search.best_params_)


