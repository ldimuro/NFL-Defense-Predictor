import get_data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def RandomForest(x, y):

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # class_weights = {2.0: 1, 0.0: 1.2, 1.0: 1.5, 3.0: 2, 4.0: 1.3, 5.0: 1.5}
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42) # 42
    rf.fit(train_x, train_y)
    y_pred_rf = rf.predict(test_x)
    print('==============================================================================')
    print(f'Random Forest Accuracy: {accuracy_score(test_y, y_pred_rf)*100:.2f}%')
    print(classification_report(test_y, y_pred_rf))

    # FEATURE IMPORTANCE
    feature_importance = rf.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    features = np.array(train_x.columns)[sorted_idx]
    plt.figure(figsize=(10, 5))
    plt.barh(features[:10], feature_importance[sorted_idx][:10])
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Top 10 Most Important Features for Defensive Coverage Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('diagrams/rf_feature_importance.png')

    sorted_feature_importance = feature_importance[sorted_idx]

    # Print out all features and their corresponding scores
    print('FEATURE IMPORTANCE SCORES')
    print('==============================================================================')
    for feature, score in zip(features, sorted_feature_importance):
        print(f'{feature}: {score:.4f}')


    # CONFUSION MATRIX
    cm = confusion_matrix(y_pred_rf, test_y)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row (true labels)
    classes = ['Cover-1', 'Cover-2', 'Cover-3', 'Cover-6', 'Cover-4', 'Other']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('diagrams/rf_confusion_matrix.png')


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


