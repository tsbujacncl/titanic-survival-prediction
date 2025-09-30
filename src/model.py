"""
Model training utilities for Titanic survival prediction.

This module provides helper functions for training and evaluating
the Random Forest classifier.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


def create_preprocessing_pipeline():
    """
    Create the full preprocessing pipeline.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Complete preprocessing pipeline
    """
    from .preprocessing import AgeImputer, FeatureEncoder, FeatureDropper

    return Pipeline([
        ("ageimputer", AgeImputer()),
        ("featureencoder", FeatureEncoder()),
        ("featuredropper", FeatureDropper())
    ])


def prepare_data(data, target_col='Survived', scale=True):
    """
    Prepare data for model training.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed dataframe
    target_col : str, default='Survived'
        Name of the target column
    scale : bool, default=True
        Whether to apply standard scaling

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray or None
        Target vector (None if target_col not in data)
    scaler : StandardScaler or None
        Fitted scaler object (None if scale=False)
    """
    if target_col in data.columns:
        X = data.drop([target_col], axis=1)
        y = data[target_col].to_numpy()
    else:
        X = data
        y = None

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.to_numpy()

    return X, y, scaler


def train_random_forest(X_train, y_train, param_grid=None, cv=3):
    """
    Train a Random Forest classifier with hyperparameter tuning.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    param_grid : dict or None
        Hyperparameter grid for GridSearchCV
    cv : int, default=3
        Number of cross-validation folds

    Returns
    -------
    best_model : RandomForestClassifier
        Best model from grid search
    grid_search : GridSearchCV
        Full grid search object with results
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [10, 100, 200, 500],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 3, 4]
        }

    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        clf,
        param_grid,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels

    Returns
    -------
    accuracy : float
        Test set accuracy
    predictions : np.ndarray
        Predicted labels
    """
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)

    return accuracy, predictions


def generate_submission(model, test_data, passenger_ids, output_path='predictions.csv'):
    """
    Generate Kaggle submission file.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    test_data : np.ndarray
        Preprocessed test features
    passenger_ids : pd.Series or np.ndarray
        PassengerId values
    output_path : str, default='predictions.csv'
        Path to save submission file

    Returns
    -------
    submission : pd.DataFrame
        Submission dataframe with PassengerId and Survived
    """
    predictions = model.predict(test_data)

    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return submission