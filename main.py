#!/usr/bin/env python3
"""
Titanic Survival Prediction - Main Script

This script demonstrates the complete workflow:
1. Load data
2. Preprocess with custom pipeline
3. Train Random Forest model
4. Evaluate and generate predictions

Usage:
    python main.py
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.preprocessing import AgeImputer, FeatureEncoder, FeatureDropper
from src.model import (
    create_preprocessing_pipeline,
    prepare_data,
    train_random_forest,
    evaluate_model,
    generate_submission
)


def main():
    """Main execution function."""

    print("=" * 60)
    print("🚢 TITANIC SURVIVAL PREDICTION")
    print("=" * 60)

    # Load data
    print("\n📂 Loading data...")
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    print(f"   Training set: {train_data.shape}")
    print(f"   Test set: {test_data.shape}")

    # Create preprocessing pipeline
    print("\n🔧 Creating preprocessing pipeline...")
    pipeline = create_preprocessing_pipeline()

    # Preprocess training data
    print("   Preprocessing training data...")
    train_processed = pipeline.fit_transform(train_data)

    # Prepare data for training
    print("   Preparing features and labels...")
    X_train, y_train, scaler = prepare_data(train_processed, target_col='Survived', scale=True)
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Samples: {X_train.shape[0]}")

    # Train model with hyperparameter tuning
    print("\n🤖 Training Random Forest with GridSearchCV...")
    print("   This may take a few minutes...")
    best_model, grid_search = train_random_forest(X_train, y_train, cv=3)

    print(f"\n   ✅ Best Parameters: {grid_search.best_params_}")
    print(f"   ✅ Best CV Score: {grid_search.best_score_:.4f}")

    # Evaluate on training data (cross-validation score)
    train_accuracy = best_model.score(X_train, y_train)
    print(f"   ✅ Training Accuracy: {train_accuracy:.4f}")

    # Process test data
    print("\n📊 Generating predictions for test set...")
    test_processed = pipeline.fit_transform(test_data)
    X_test, _, test_scaler = prepare_data(test_processed, target_col='Survived', scale=True)

    # Generate predictions
    passenger_ids = test_data['PassengerId']
    submission = generate_submission(
        best_model,
        X_test,
        passenger_ids,
        output_path='data/predictions.csv'
    )

    print(f"\n   ✅ Generated {len(submission)} predictions")
    print(f"   ✅ Predicted survivors: {submission['Survived'].sum()}")
    print(f"   ✅ Predicted casualties: {(1 - submission['Survived']).sum()}")

    print("\n" + "=" * 60)
    print("✨ COMPLETE! Check data/predictions.csv for results.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure train.csv and test.csv are in the data/ directory.")
        print("   Download them from: https://www.kaggle.com/c/titanic/data")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)