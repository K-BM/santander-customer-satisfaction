import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import resample

def find_highly_correlated_features(df, threshold):
    # Loop over the features and replace 0 values with NaN
    for column in df.columns:
        df[column] = df[column].replace(0, np.nan)
    
    corr_matrix = df.corr().abs()
    correlated_features = []

    for i in range(len(corr_matrix.columns)):
        feature = corr_matrix.columns[i]
        target_group = None

        for group in correlated_features:
            correlations = [corr_matrix.loc[feature, other_feature] for other_feature in group]
            max_correlation = max(correlations) if correlations else 0

            if max_correlation > threshold:
                target_group = group
                break

        if target_group is not None:
            target_group.append(feature)
        else:
            correlated_features.append([feature])

    return correlated_features


def find_best_features(df_train, selected_features):
    """
    Find the best feature with the least amount of missing data from each internal list in selected_features.

    Parameters:
        df_train (DataFrame): The DataFrame containing the data.
        selected_features (list of lists): A list of lists, where each internal list contains feature names.

    Returns:
        list: A list containing the best feature with the least amount of missing data from each internal list.
    """
    # Initialize a list to store the best features from each internal list
    best_features = []

    # Loop through each list of features in selected_features
    for features_list in selected_features:
        # Calculate the number of missing values for each feature
        missing_data_counts = df_train[features_list].isnull().sum()
        # Calculate the total number of missing values for the current features_list
        total_missing_data = missing_data_counts.sum()

        # Find the feature with the least amount of missing data in the current features_list
        best_feature = missing_data_counts.idxmin()
        best_features.append(best_feature)
        
    df_train = df_train[best_features + ['TARGET']]
    return df_train

def remove_low_variance_features(df_train, threshold=0.01):
    """
    Remove low-variance features from the DataFrame.

    Parameters:
        df_train (DataFrame): The DataFrame containing the data.
        threshold (float, optional): The threshold to identify low-variance features. Default is 0.01.

    Returns:
        DataFrame: A new DataFrame containing only the non-constant features.
    """
    # Create a VarianceThreshold object with the desired threshold
    variance_threshold = VarianceThreshold(threshold=threshold)

    # Fit the VarianceThreshold object to your data to identify low-variance features
    variance_threshold.fit(df_train)

    # Get the indices of the non-constant features
    non_constant_feature_indices = variance_threshold.get_support(indices=True)

    # Get the names of the non-constant features
    non_constant_features = df_train.columns[non_constant_feature_indices]

    # Create a new DataFrame with only the non-constant features
    df_train = df_train[non_constant_features]

    return df_train, non_constant_features


def handle_unbalanced_target(X, y):
    # Separate majority and minority classes
    majority_class_indices = y[y==0].index
    minority_class_indices = y[y==1].index

    # Upsample minority class
    minority_class_X = X[minority_class_indices]
    minority_class_y = y[minority_class_indices]
    minority_class_X_upsampled, minority_class_y_upsampled = resample(minority_class_X, minority_class_y, 
                                                                    replace=True, n_samples=len(majority_class_indices), 
                                                                    random_state=123)

    # Combine majority class with upsampled minority class
    X_upsampled = np.concatenate([X[majority_class_indices], minority_class_X_upsampled])
    y_upsampled = pd.concat([y.loc[majority_class_indices], minority_class_y_upsampled])
    
    return X_upsampled, y_upsampled

