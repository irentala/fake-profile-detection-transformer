import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Load the dataset
file_path = '/Users/irentala/PycharmProjects/fake-profile-detection-transformer/data/cleansed_50.csv'
df = pd.read_csv(file_path, sep=',')

# Feature extraction: Creating new features for modeling
# Compute Hold Time as (release_time - press_time)
df['hold_time'] = df['release_time'] - df['press_time']

# Compute Flight Time (if applicable)
# Assuming we have the same user session sorted by keystrokes, compute flight time between keystrokes
df['flight_time'] = df['press_time'].diff().fillna(0)  # Calculate the difference between press times
df.loc[df['session_id'] != df['session_id'].shift(), 'flight_time'] = 0  # Reset flight time at session boundaries

# Compute Preceding Flight Time (time between release of previous key and press of next key)
df['preceding_flight_time'] = df['press_time'] - df['release_time'].shift().fillna(0)
df.loc[df['session_id'] != df['session_id'].shift(), 'preceding_flight_time'] = 0  # Reset preceding flight time at session boundaries

# Compute Following Flight Time (time between release of current key and press of next key)
df['following_flight_time'] = df['press_time'].shift(-1) - df['release_time']
df['following_flight_time'] = df['following_flight_time'].fillna(0)
df.loc[df['session_id'] != df['session_id'].shift(-1), 'following_flight_time'] = 0  # Reset following flight time at session boundaries

# Feature selection: Adding hold time and flight time to features
features = ['hold_time', 'flight_time', 'preceding_flight_time', 'following_flight_time']
x = df[features]
y = df['user_ids']

# Apply StandardScaler
standard_scaler = StandardScaler()
x_standard = standard_scaler.fit_transform(x)

# Apply MinMaxScaler
minmax_scaler = MinMaxScaler()
x_minmax = minmax_scaler.fit_transform(x)

# Apply Extended MinMaxScaler (scaling to a custom range)
extended_minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
x_extended_minmax = extended_minmax_scaler.fit_transform(x)


# Define a function to train and evaluate classifiers with different scalers
def evaluate_model(x_scaled, scaler_name):
    print(f"\nResults for {scaler_name}:\n")

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Random Forest Classifier with hyperparameter tuning to reduce overfitting
    rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=3,
                                           random_state=42)
    rf_classifier.fit(x_train, y_train)
    y_pred_rf = rf_classifier.predict(x_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print("Random Forest Classifier Accuracy:", rf_accuracy)
    rf_report = classification_report(y_test, y_pred_rf, zero_division=1, output_dict=True)
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=1))

    # Cross-validation for Random Forest
    rf_cv_scores = cross_val_score(rf_classifier, x_scaled, y, cv=5)
    print("Random Forest Cross-Validation Accuracy: {} +/- {}".format(np.mean(rf_cv_scores), np.std(rf_cv_scores)))

    # CatBoost Classifier with hyperparameter tuning to reduce overfitting
    catboost_classifier = CatBoostClassifier(iterations=50, learning_rate=0.05, depth=4, l2_leaf_reg=3, verbose=0)
    catboost_classifier.fit(x_train, y_train)
    y_pred_catboost = catboost_classifier.predict(x_test)
    catboost_accuracy = accuracy_score(y_test, y_pred_catboost)
    print("CatBoost Classifier Accuracy:", catboost_accuracy)
    catboost_report = classification_report(y_test, y_pred_catboost, zero_division=1, output_dict=True)
    print("CatBoost Classification Report:\n", classification_report(y_test, y_pred_catboost, zero_division=1))

    # Cross-validation for CatBoost
    catboost_cv_scores = cross_val_score(catboost_classifier, x_scaled, y, cv=5)
    print(
        "CatBoost Cross-Validation Accuracy: {} +/- {}".format(np.mean(catboost_cv_scores), np.std(catboost_cv_scores)))

    return (rf_accuracy, rf_report['weighted avg']), (catboost_accuracy, catboost_report['weighted avg'])


# Evaluate models for each scaler
weighted_avgs = []
weighted_avgs.append(evaluate_model(x_standard, "StandardScaler"))
weighted_avgs.append(evaluate_model(x_minmax, "MinMaxScaler"))
weighted_avgs.append(evaluate_model(x_extended_minmax, "Extended MinMaxScaler"))

# Prepare results for Excel format
excel_data = {
    "Scaler": ["StandardScaler", "MinMaxScaler", "Extended MinMaxScaler"],
    "Random Forest": [
        f"Accuracy: {weighted_avgs[0][0][0]:.2f}, Precision: {weighted_avgs[0][0][1]['precision']:.2f}, Recall: {weighted_avgs[0][0][1]['recall']:.2f}, F1-score: {weighted_avgs[0][0][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[1][0][0]:.2f}, Precision: {weighted_avgs[1][0][1]['precision']:.2f}, Recall: {weighted_avgs[1][0][1]['recall']:.2f}, F1-score: {weighted_avgs[1][0][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[2][0][0]:.2f}, Precision: {weighted_avgs[2][0][1]['precision']:.2f}, Recall: {weighted_avgs[2][0][1]['recall']:.2f}, F1-score: {weighted_avgs[2][0][1]['f1-score']:.2f}"
    ],
    "CatBoost": [
        f"Accuracy: {weighted_avgs[0][1][0]:.2f}, Precision: {weighted_avgs[0][1][1]['precision']:.2f}, Recall: {weighted_avgs[0][1][1]['recall']:.2f}, F1-score: {weighted_avgs[0][1][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[1][1][0]:.2f}, Precision: {weighted_avgs[1][1][1]['precision']:.2f}, Recall: {weighted_avgs[1][1][1]['recall']:.2f}, F1-score: {weighted_avgs[1][1][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[2][1][0]:.2f}, Precision: {weighted_avgs[2][1][1]['precision']:.2f}, Recall: {weighted_avgs[2][1][1]['recall']:.2f}, F1-score: {weighted_avgs[2][1][1]['f1-score']:.2f}"
    ]
}

# Create DataFrame and print in Excel format
excel_df = pd.DataFrame(excel_data)
print("\nResults in Excel Format:\n")
print(excel_df.to_string(index=False))
