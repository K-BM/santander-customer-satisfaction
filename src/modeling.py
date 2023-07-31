from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

import seaborn as sns
import matplotlib.pyplot as plt

def train_model(X_train_data, X_test_data, y_train_data, y_test_data):
    # Train model on upsampled data
    # Define the parameter grid
#     param_grid = {'n_estimators': [10, 50, 100, 200],
#                   'max_depth': [None, 5, 10, 20],
#                   'min_samples_split': [2, 5, 10],
#                   'min_samples_leaf': [1, 2, 4]}
    
    # Create the random forest classifier
    rf = RandomForestClassifier()
    
    # Create the grid search
#     grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    
    # Fit the grid search to the data
#     grid_search.fit(X_train_data, y_train_data)
    rf.fit(X_train_data, y_train_data)
    
    # Get the best estimator
#     best_estimator = grid_search.best_estimator_
    
    # Predict the test set labels
#     y_pred = best_estimator.predict(X_test_data)
    y_pred = rf.predict(X_test_data)
    
    # Calculate the scores
    recall = recall_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)
    
    return rf, recall, precision, f1