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

def RandomForest(x, y, dataframe):

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # class_weights = {2.0: 1, 0.0: 1.2, 1.0: 1.5, 3.0: 2, 4.0: 1.3, 5.0: 1.5}
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42) # 42
    rf.fit(train_x, train_y)
    y_pred_rf = rf.predict(test_x)
    print('==============================================================================')
    print(f'Random Forest Accuracy: {accuracy_score(test_y, y_pred_rf)*100:.2f}%')
    print(classification_report(test_y, y_pred_rf))



    # DIAGRAMS
    ######################################################################################################
    classes = ['Cover-1', 'Cover-2', 'Cover-3', 'Cover-6', 'Cover-4', 'Other']
    cardinals_red = '#972440'

    # FEATURE IMPORTANCE
    feature_importance = rf.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    features = np.array(train_x.columns)[sorted_idx]
    plt.figure(figsize=(10, 5))
    plt.barh(features[:10], feature_importance[sorted_idx][:10], color=cardinals_red)
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
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Reds', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('diagrams/rf_confusion_matrix.png')



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

    # Display the plot
    plt.tight_layout()
    plt.savefig('diagrams/coverage_distribution_pie.png')

    # coverage_dist.plot(kind='bar', stacked=True, figsize=(10, 6))
    # plt.title('Distribution of Coverage Types Across Downs')
    # plt.xlabel('Down')
    # plt.ylabel('Frequency')
    # plt.legend([classes[int(i)] for i in range(len(classes))], title='Coverage Type')
    # plt.savefig('diagrams/coverage_distribution.png')



    # Create a subset of features relevant to coverage prediction
    dataframe_temp = dataframe[dataframe['target_y'] != 5.0] # Remove 'Other' coverage class
    relevant_features = dataframe_temp[['deepest_safety_depth', 'next_deepest_safety_depth', 'target_y']]
    sns.pairplot(relevant_features, hue='target_y', palette='Set1')
    plt.title('Pairplot of Key Features')
    plt.savefig('diagrams/key_features_pairwise.png')


    dataframe_temp = dataframe[dataframe['target_y'] != 5.0] # Remove 'Other' coverage class
    # dataframe_temp['targey_y'] = dataframe_temp['target_y'].map(lambda x: classes[x]) # Map target_y class names
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target_y', y='next_deepest_safety_depth', data=dataframe_temp, palette='Set2')
    plt.title('Distribution of "next_deepest_safety_depth" by Coverage Type')
    plt.xlabel('Coverage Type')
    plt.ylabel('Next Deepest Safety Depth')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('diagrams/next_deepest_safety_depth_distribution.png')




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


