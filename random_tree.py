import get_data
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import time

def RandomForest(x, y, dataframe):

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # CALCULATE BEST n_estimator
    # n_estimators_range = range(10, 401, 10)
    # n_accuracy_scores = []
    # best_n = 0
    # best_acc = 0
    # for n in n_estimators_range:
    #     print('n:', n)
    #     rf = RandomForestClassifier(n_estimators=n, max_depth=20, random_state=42, class_weight='balanced') # 42
    #     rf.fit(train_x, train_y)
    #     y_pred = rf.predict(test_x)

    #     accuracy = accuracy_score(test_y, y_pred)
    #     if accuracy > best_acc:
    #         best_acc = accuracy
    #         best_n = n

    #     n_accuracy_scores.append(accuracy)

    # plt.figure(figsize=(10, 6))
    # plt.plot(n_estimators_range, n_accuracy_scores, marker='o')
    # plt.title('Effect of n_estimators on Random Forest Performance')
    # plt.xlabel('Number of Trees (n_estimators)')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # plt.savefig('diagrams/n_estimators.png')

    # print('BEST n_estimator:', best_n)

    # CALCULATE BEST max_depth_range
    # max_depth_range = range(1, 41)
    # depth_accuracy_scores = []
    # best_depth = 0
    # best_acc = 0
    # for depth in max_depth_range:
    #     print('depth:', depth)
    #     rf = RandomForestClassifier(n_estimators=180, max_depth=depth, random_state=42, class_weight='balanced') # 42
    #     rf.fit(train_x, train_y)
    #     y_pred = rf.predict(test_x)

    #     accuracy = accuracy_score(test_y, y_pred)
    #     if accuracy > best_acc:
    #         best_acc = accuracy
    #         best_depth = depth

    #     depth_accuracy_scores.append(accuracy)

    # plt.figure(figsize=(10, 6))
    # plt.plot(max_depth_range, depth_accuracy_scores, marker='o')
    # plt.title('Effect of max_depth on Random Forest Performance')
    # plt.xlabel('Max Depth')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # plt.savefig('diagrams/max_depths.png')

    # print('BEST max_depth:', best_depth)
    

    best_n_estimator = 180
    best_max_depth = 20


    # Best results:  n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'
    rf = RandomForestClassifier(n_estimators=best_n_estimator, max_depth=best_max_depth, random_state=42, class_weight='balanced') # 42
    rf.fit(train_x, train_y)
    y_pred_rf = rf.predict(test_x)








    majority_class = np.bincount(train_y).argmax()
    baseline_preds = np.full_like(test_y, majority_class)
    baseline_accuracy = accuracy_score(test_y, baseline_preds)

    precision_baseline = precision_score(test_y, baseline_preds, average='weighted', zero_division=0)
    recall_baseline = recall_score(test_y, baseline_preds, average='weighted')
    f1_baseline = f1_score(test_y, baseline_preds, average='weighted')

    print('======================================================================================================')
    print(f'Random Forest Accuracy: {accuracy_score(test_y, y_pred_rf)*100:.2f}% (~0.02 seconds)')
    print(f'Baseline Accuracy:\t{baseline_accuracy*100:.2f}%')
    print(f'Baseline Precision:\t{precision_baseline*100:.2f}%')
    print(f'Baseline Recall:\t{recall_baseline*100:.2f}%')
    print(f'Baseline F1-Score:\t{f1_baseline*100:.2f}%')
    print(classification_report(test_y, y_pred_rf))



    # DIAGRAMS
    ######################################################################################################
    classes = ['Cover-1', 'Cover-2', 'Cover-3', 'Cover-6', 'Cover-4', 'Other']
    # classes = ['C-1 Man', 'C-2']
    cardinals_red = '#972440'

    # FEATURE IMPORTANCE
    feature_importance = rf.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    features = np.array(train_x.columns)[sorted_idx]
    plt.figure(figsize=(10, 5))
    plt.barh(features[:20], feature_importance[sorted_idx][:20], color=cardinals_red)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Top 10 Most Important Features for Defensive Pass Coverage Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('diagrams/rf_feature_importance.png')

    sorted_feature_importance = feature_importance[sorted_idx]

    # Print out all features and their corresponding scores
    print('======================================================================================================')
    print('FEATURE IMPORTANCE SCORES')
    for feature, score in zip(features, sorted_feature_importance):
        print(f'{feature}: {score:.4f}')


    # CONFUSION MATRIX
    print('======================================================================================================')
    cm = confusion_matrix(y_pred_rf, test_y)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row (true labels)
    print('CONFUSION MATRIX:\n', cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Reds', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('diagrams/rf_confusion_matrix_51features.png')



    # dataframe_temp = dataframe[dataframe['down'] != 4.0] # Remove 'Other' coverage class and 4th-Down plays
    coverage_dist = dataframe.groupby(['down', 'target_y']).size().unstack()
    coverage_dist = coverage_dist.div(coverage_dist.sum(axis=1), axis=0) # Normalize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    downs = [1, 2, 3, 4]
    suffix = ['st', 'nd', 'rd', 'th']
    colors = ['#fc7060', '#db44d4', '#f0872b', '#7dcc7f', '#ffe046', '#1a76bc']
    for i, down in enumerate(downs):
        ax = axes[i//2, i%2]  # Calculate position for a 2x2 grid
        ax.pie(coverage_dist.loc[down], labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Coverage Distribution - {down}{suffix[i]}-Down')

    plt.tight_layout()
    plt.savefig('diagrams/coverage_distribution_pie.png')



    # Create a subset of features relevant to coverage prediction
    dataframe_temp = dataframe[dataframe['target_y'] != 5.0] # Remove 'Other' coverage class
    relevant_features = dataframe_temp[['deepest_safety_depth', 'next_deepest_safety_depth', 'target_y']]
    sns.pairplot(relevant_features, hue='target_y', palette='Set1')
    # pairplot._legend.remove()
    # plt.legend(title='Coverage Type', labels=classes, bbox_to_anchor=(1.9, 1), loc='upper right')
    # plt.subplots_adjust(right=0.85)
    # plt.tight_layout()
    plt.savefig('diagrams/key_features_pairwise.png')


    dataframe_temp = dataframe[dataframe['target_y'] != 5.0] # Remove 'Other' coverage class
    # dataframe_temp['targey_y'] = dataframe_temp['target_y'].map(lambda x: classes[x]) # Map target_y class names
    plt.figure(figsize=(10, 6))
    # sns.boxplot(x='target_y', y='next_deepest_safety_depth', data=dataframe_temp, palette='Set2')
    sns.boxplot(x='target_y', y='next_deepest_safety_depth', data=dataframe_temp, hue='target_y', palette='Set2', legend=False)
    plt.title('Distribution of "next_deepest_safety_depth" by Coverage Type')
    plt.xlabel('Coverage Type')
    plt.ylabel('Next Deepest Safety Depth')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('diagrams/next_deepest_safety_depth_distribution.png')


    # Create boxplots for avg_depth features across coverage types
    avg_depth_features = ['avg_cb_depth', 'avg_db_depth', 'avg_defender_depth','avg_lb_depth']
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(avg_depth_features, 1):
        plt.subplot(3, 4, i)  # Create a 3x4 grid of subplots
        sns.boxplot(x='target_y', y=feature, data=dataframe_temp, hue='target_y', palette='Set1', legend=False)
        plt.title(f'{feature}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('diagrams/avg_depth_boxplots.png')


    # sns.pairplot(dataframe[depth_features + ['target_y']], hue='target_y', palette='Set1')
    # plt.suptitle('Pairplot of Depth Features by Coverage Type', y=1.02)
    # plt.savefig('diagrams/depth_pairplots.png')


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


