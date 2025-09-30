"""
Custom preprocessing transformers for Titanic dataset.

This module contains custom Scikit-learn compatible transformers for:
- Imputing missing Age values
- One-hot encoding categorical features
- Dropping unnecessary features
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class AgeImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to impute missing Age values using mean strategy.

    This is necessary because ~20% of Age values are missing, and age is
    an important predictor of survival (children had priority in lifeboats).

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy. Options: 'mean', 'median', 'most_frequent'

    Attributes
    ----------
    imputer_ : SimpleImputer
        The fitted imputer object
    """

    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def fit(self, X, y=None):
        """Fit the imputer on the data."""
        return self

    def transform(self, X):
        """
        Impute missing Age values.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with Age column

        Returns
        -------
        X : pd.DataFrame
            Dataframe with imputed Age values
        """
        imputer = SimpleImputer(strategy=self.strategy)
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to one-hot encode categorical variables.

    Converts:
    - Embarked (C/S/Q/N) into 4 binary columns
    - Sex (Female/Male) into 2 binary columns

    One-hot encoding prevents the model from assuming ordinal relationships
    in categorical data.

    Attributes
    ----------
    encoder_ : OneHotEncoder
        The fitted encoder object
    """

    def fit(self, X, y=None):
        """Fit the encoder on the data."""
        return self

    def transform(self, X):
        """
        One-hot encode categorical features.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with Embarked and Sex columns

        Returns
        -------
        X : pd.DataFrame
            Dataframe with encoded features as new columns
        """
        encoder = OneHotEncoder()

        # Encode Embarked port (C=Cherbourg, S=Southampton, Q=Queenstown)
        matrix = encoder.fit_transform(X[['Embarked']]).toarray()
        column_names = ["C", "S", "Q", "N"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

        # Encode Sex (Female/Male)
        matrix = encoder.fit_transform(X[['Sex']]).toarray()
        column_names = ["Female", "Male"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

        return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop unnecessary features.

    Drops:
    - Embarked, Sex: Already encoded as binary columns
    - Name, Ticket, Cabin: High cardinality, not useful for this model
    - N: Placeholder column from encoding

    Parameters
    ----------
    columns : list, optional
        Additional columns to drop beyond the defaults
    """

    def __init__(self, columns=None):
        self.columns = columns or ["Embarked", "Name", "Ticket", "Cabin", "Sex", "N"]

    def fit(self, X, y=None):
        """Fit (no-op for this transformer)."""
        return self

    def transform(self, X):
        """
        Drop unnecessary columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe

        Returns
        -------
        X : pd.DataFrame
            Dataframe with specified columns removed
        """
        return X.drop(self.columns, axis=1, errors="ignore")