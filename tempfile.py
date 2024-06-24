# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:11:51 2024

@author: mzing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import joblib


train_original = pd.read_csv('dataset/train.csv')

####################### Classes used to preprocess the data ##############################

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=None):
        """
        Custom transformer to remove outliers from specified features in a DataFrame.

        Parameters:
            feat_with_outliers (list or None): List of feature names with potential outliers.
        """
        self.feat_with_outliers = feat_with_outliers or []

    def fit(self, X, y=None):
        """
        No fitting required. Returns self.

        Parameters:
            X (pd.DataFrame): Input data (ignored).
            y (pd.Series or None): Target labels (ignored).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Removes outliers from specified features using the IQR method.

        Parameters:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with outliers removed.
        """
        missing_features = set(self.feat_with_outliers) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        # Calculate 25% and 75% quantiles
        Q1 = X[self.feat_with_outliers].quantile(0.25)
        Q3 = X[self.feat_with_outliers].quantile(0.75)
        IQR = Q3 - Q1

        # Keep data within 3 IQR
        X = X[~((X[self.feat_with_outliers] < (Q1 - 3 * IQR)) | (X[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)]
        return X
    
class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=None):
        """
        Custom transformer to drop specified features from a DataFrame.

        Parameters:
            feature_to_drop (list or None): List of feature names to drop.
        """
        self.feature_to_drop = feature_to_drop or []

    def fit(self, X, y=None):
        """
        No fitting required. Returns self.

        Parameters:
            X (pd.DataFrame): Input data (ignored).
            y (pd.Series or None): Target labels (ignored).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Drops specified features from the DataFrame.

        Parameters:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with specified features dropped.
        """
        missing_features = set(self.feature_to_drop) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")
        return X.drop(columns=self.feature_to_drop)
    
class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=None):
        """
        Custom transformer to apply Min-Max scaling to specified features in a DataFrame.

        Parameters:
            min_max_scaler_ft (list or None): List of feature names to scale.
        """
        self.min_max_scaler_ft = min_max_scaler_ft or []

    def fit(self, X, y=None):
        """
        No fitting required. Returns self.

        Parameters:
            X (pd.DataFrame): Input data (ignored).
            y (pd.Series or None): Target labels (ignored).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Applies Min-Max scaling to specified features in the DataFrame.

        Parameters:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with specified features scaled.
        """
        missing_features = set(self.min_max_scaler_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        min_max_enc = MinMaxScaler()
        X[self.min_max_scaler_ft] = min_max_enc.fit_transform(X[self.min_max_scaler_ft])
        return X
    
class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=None):
        """
        Custom transformer to perform one-hot encoding on specified features in a DataFrame.

        Parameters:
            one_hot_enc_ft (list or None): List of feature names to one-hot encode.
        """
        self.one_hot_enc_ft = one_hot_enc_ft or ['AGE_GROUP', 'BP_Cat', 'GENDER', 'MODE OF ARRIVAL', 'MARITAL STATUS', 'TYPE OF ADMSN']
        self.one_hot_enc = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        """
        Fit the one-hot encoder to the specified features.

        Parameters:
            X (pd.DataFrame): Input data.
            y (pd.Series or None): Target labels (ignored).

        Returns:
            self
        """
        self.one_hot_enc.fit(X[self.one_hot_enc_ft])
        self.feat_names_one_hot_enc = self.one_hot_enc.get_feature_names_out(self.one_hot_enc_ft)
        return self

    def transform(self, X):
        """
        Applies one-hot encoding to specified features and concatenates with the remaining features.

        Parameters:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with one-hot encoded features.
        """
        missing_features = set(self.one_hot_enc_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        one_hot_enc_df = pd.DataFrame(self.one_hot_enc.transform(X[self.one_hot_enc_ft]).toarray(), columns=self.feat_names_one_hot_enc, index=X.index)
        rest_of_features = [ft for ft in X.columns if ft not in self.one_hot_enc_ft]
        df_concat = pd.concat([one_hot_enc_df, X[rest_of_features]], axis=1)
        return df_concat
    
class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, col_with_skewness=None):
        """
        Custom transformer to handle skewness in specified features using cubic root transformation.

        Parameters:
            col_with_skewness (list or None): List of feature names with skewness.
        """
        self.col_with_skewness = col_with_skewness or []

    def fit(self, X, y=None):
        """
        No fitting required. Returns self.

        Parameters:
            X (pd.DataFrame): Input data (ignored).
            y (pd.Series or None): Target labels (ignored).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Applies cubic root transformation to specified features.

        Parameters:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with skewness handled.
        """
        missing_features = set(self.col_with_skewness) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        X[self.col_with_skewness] = np.cbrt(X[self.col_with_skewness])
        return X
    

# Create the pipeline
pipeline = Pipeline([
    ('outlier_remover', OutlierRemover(feat_with_outliers=['bmi'])),
    ('drop features', DropFeatures(feature_to_drop=['bmi_cat'])),
    ('skewness handler', SkewnessHandler(col_with_skewness=['age', 'bmi'])),
    ('min max scaler', MinMaxWithFeatNames(min_max_scaler_ft=['age', 'bmi'])),
    ('one hot encoder', OneHotWithFeatNames(one_hot_enc_ft=['sex', 'smoker', 'region']))
])


# Apply the pipeline to the train dataset
pd.options.mode.chained_assignment = None
insurance_amt_train = pipeline.fit_transform(train_original)


X_insurance_amt_train, y_insurance_amt_train = insurance_amt_train.loc[:, insurance_amt_train.columns != 'charges'], insurance_amt_train['charges']

# Gradient Boosting Regressor

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_insurance_amt_train, y_insurance_amt_train)

scores = cross_val_score(gb_model, X_insurance_amt_train, y_insurance_amt_train, scoring= 'neg_mean_squared_error', cv=3)
rmse_scores = np.sqrt(-scores)

print('Score: {}'.format(rmse_scores))
print('Mean score: {}'.format(rmse_scores.mean()))
print('Standard deviation: {}'.format(rmse_scores.std()))

# save the model to disk
joblib.dump(gb_model, "gb_model.sav")