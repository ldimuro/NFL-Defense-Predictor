import get_data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def RandomForest(self):
    plays_data = get_data.plays_2022()
    train_x, train_y, test_x, test_y = self.process_data(plays_data, self.input_features)

    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(train_x, train_y)

    y_pred_rf = rf.predict(test_x)

    print('Random Forest Accuracy:', accuracy_score(test_y, y_pred_rf))
    print(classification_report(test_y, y_pred_rf))