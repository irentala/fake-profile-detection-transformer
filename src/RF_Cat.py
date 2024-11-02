import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the dataset
file_path = '/Users/irentala/PycharmProjects/fake-profile-detection-transformer/data/cleaned2.csv'
df = pd.read_csv(file_path, sep=',')

# Feature extraction: Selecting relevant features for modeling
features = ['press_time', 'release_time']
x = df[features]
y = df['user_ids']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Random Forest Classifier with hyperparameter tuning to reduce overfitting
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred_rf = rf_classifier.predict(x_test)
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=1))

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_classifier, x, y, cv=5)
print("Random Forest Cross-Validation Accuracy: {} +/- {}".format(np.mean(rf_cv_scores), np.std(rf_cv_scores)))

# CatBoost Classifier with hyperparameter tuning to reduce overfitting
catboost_classifier = CatBoostClassifier(iterations=50, learning_rate=0.05, depth=4, l2_leaf_reg=3, verbose=0)
catboost_classifier.fit(x_train, y_train)
y_pred_catboost = catboost_classifier.predict(x_test)
print("CatBoost Classifier Accuracy:", accuracy_score(y_test, y_pred_catboost))
print("CatBoost Classification Report:\n", classification_report(y_test, y_pred_catboost, zero_division=1))

# Cross-validation for CatBoost
catboost_cv_scores = cross_val_score(catboost_classifier, x, y, cv=5)
print("CatBoost Cross-Validation Accuracy: {} +/- {}".format(np.mean(catboost_cv_scores), np.std(catboost_cv_scores)))
