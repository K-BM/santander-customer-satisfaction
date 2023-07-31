import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

from src.processing import *
from src.modeling import *

import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv(r'data/train.csv')
df_test = pd.read_csv(r'data/test.csv')
sample_submission = pd.read_csv(r'data/sample_submission.csv')

selected_features = find_highly_correlated_features(df_train.drop(['ID', 'TARGET'], axis=1), threshold=0.7)


df_train = find_best_features(df_train, selected_features)


df_train, non_constant_features = remove_low_variance_features(df_train, threshold=0.01)


# Assuming non_constant_features is a pandas Index or a list of column names
non_constant_features = non_constant_features.drop('TARGET', errors='ignore')

df_test = df_test[non_constant_features]

X, y = handle_unbalanced_target(np.array(df_train.drop(['TARGET'], axis=1)), df_train['TARGET'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_estimator, recall, precision, f1 = train_model(X_train, X_test, y_train, y_test)

print("best_estimator :", best_estimator)
print("recall :", recall)
print("precision :", precision)
print("f1 :", f1)

predictions = best_estimator.predict(df_test)
data = predictions.tolist()
sample_submission['TARGET']=data
sample_submission.to_csv("data/submission.csv", index=False)

df_submission = pd.read_csv(r'data/submission.csv')
print(df_submission.head())











