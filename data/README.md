# Titanic Dataset

## Overview
This directory contains the Titanic dataset from Kaggle's famous ["Titanic - Machine Learning from Disaster"](https://www.kaggle.com/c/titanic) competition.

## Files

### `train.csv`
- **891 rows** (passengers)
- **12 columns** (features + target)
- Used for training and validation

### `test.csv`
- **418 rows** (passengers)
- **11 columns** (no 'Survived' column)
- Used for generating Kaggle competition predictions

### `predictions.csv`
- Model predictions for the test set
- Kaggle submission format (PassengerId, Survived)

## Data Dictionary

| Variable    | Definition                                  | Key                                            |
|-------------|---------------------------------------------|------------------------------------------------|
| PassengerId | Unique passenger identifier                 | Integer                                        |
| Survived    | Survival status (target variable)           | 0 = No, 1 = Yes                                |
| Pclass      | Ticket class                                | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| Name        | Passenger name                              | String                                         |
| Sex         | Gender                                      | male, female                                   |
| Age         | Age in years                                | Float (fractional if less than 1)              |
| SibSp       | # of siblings/spouses aboard                | Integer                                        |
| Parch       | # of parents/children aboard                | Integer                                        |
| Ticket      | Ticket number                               | String                                         |
| Fare        | Passenger fare                              | Float                                          |
| Cabin       | Cabin number                                | String (many missing)                          |
| Embarked    | Port of embarkation                         | C = Cherbourg, Q = Queenstown, S = Southampton |

## Data Quality Issues

### Missing Values (Training Set)
- **Age**: 177 missing (~20%)
- **Cabin**: 687 missing (~77%)
- **Embarked**: 2 missing (<1%)

### Preprocessing Applied
1. **Age**: Imputed using mean strategy
2. **Cabin**: Dropped (too many missing values)
3. **Embarked**: Forward-filled or imputed
4. **Categorical features**: One-hot encoded (Sex, Embarked)
5. **Unnecessary features**: Dropped (Name, Ticket)

## Dataset Source
Downloaded from: https://www.kaggle.com/c/titanic/data

**Note**: The raw CSV files are not included in this repository due to Kaggle's terms of service.
Download them directly from Kaggle and place them in this directory.