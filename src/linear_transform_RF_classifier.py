import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Load the dataset
file_path = '/Users/irentala/PycharmProjects/fake-profile-detection-transformer/data/cleansed_data_25.csv'
df = pd.read_csv(file_path, sep=',')

# Feature extraction: Selecting relevant features for modeling
features = ['press_time', 'release_time']
x = df[features]
y = df['user_ids']

# Standardize the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Apply Linear Projection
input_dim = x.shape[1]
projection_layer = Dense(32, activation='linear')
x = projection_layer(tf.convert_to_tensor(x)).numpy()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42)
rf_classifier.fit(x_train, y_train)

# Predict and evaluate
y_pred_rf = rf_classifier.predict(x_test)
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=1))

# Feature Importance Analysis
importances = rf_classifier.feature_importances_
feature_names = ['Linear Projected Feature {}'.format(i+1) for i in range(x.shape[1])]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Analysis for Random Forest Classifier')
plt.show()
