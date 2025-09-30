# 🚢 Titanic Survival Prediction

A machine learning project predicting passenger survival on the Titanic using Python, Pandas, and Scikit-learn. Built with custom preprocessing pipelines and Random Forest classification.

**Achieved 78.8% accuracy** on the validation set using feature engineering and hyperparameter tuning.

---

## 📊 Project Overview

This project uses the famous Kaggle Titanic dataset to predict which passengers survived the disaster. The solution includes:

- **Custom preprocessing pipeline** with Scikit-learn transformers
- **Feature engineering**: One-hot encoding, mean imputation, standardization
- **Model optimization**: GridSearchCV for hyperparameter tuning
- **Production-ready code**: Modular, reusable components in `/src`

### Key Results
- **Model**: Random Forest Classifier
- **Accuracy**: 78.8% on test set
- **Features Used**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- **Techniques**: Stratified splitting, cross-validation, feature scaling

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning (Random Forest, GridSearchCV, pipelines)
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter** - Interactive notebooks

---

## 📁 Project Structure

```
titanic/
├── notebooks/
│   └── titanic_survial_prediction.ipynb   # Main analysis notebook
├── src/
│   ├── __init__.py                        # Package initialization
│   ├── preprocessing.py                   # Custom transformers
│   └── model.py                           # Training utilities
├── data/
│   ├── train.csv                          # Training dataset
│   ├── test.csv                           # Test dataset
│   ├── predictions.csv                    # Model predictions
│   └── README.md                          # Data documentation
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/tsbujacncl/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the Kaggle Titanic dataset and place `train.csv` and `test.csv` in the `data/` directory:
- [Kaggle Competition Page](https://www.kaggle.com/c/titanic/data)

### 4. Run the Notebook
```bash
jupyter notebook notebooks/titanic_survial_prediction.ipynb
```

Run all cells to:
- Load and explore the data
- Preprocess features
- Train the Random Forest model
- Generate predictions

### 5. (Optional) Use the Python Modules
```python
from src.preprocessing import AgeImputer, FeatureEncoder, FeatureDropper
from src.model import create_preprocessing_pipeline, train_random_forest

# Create pipeline
pipeline = create_preprocessing_pipeline()

# Train model
best_model, grid_search = train_random_forest(X_train, y_train)
```

---

## 📈 Key Insights & Visualizations

### Feature Correlation Matrix
![Correlation Heatmap](https://via.placeholder.com/600x400.png?text=Add+Your+Correlation+Heatmap)

**Key Findings**:
- **Pclass** has a negative correlation with survival (-0.34) - lower class = lower survival
- **Fare** has a positive correlation with survival (0.26) - higher fare = higher survival
- **Age** has a slight negative correlation with survival

### Survival Rate by Gender
- **Female**: ~74% survival rate
- **Male**: ~19% survival rate

Gender was the strongest predictor of survival ("women and children first").

### Survival Rate by Passenger Class
- **1st Class**: ~63% survival
- **2nd Class**: ~47% survival
- **3rd Class**: ~24% survival

Higher-class passengers had better access to lifeboats.

---

## 🧪 Model Performance

### Hyperparameters (Best Model)
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=3,
    random_state=42
)
```

### Evaluation Metrics
- **Accuracy**: 78.8%
- **Cross-Validation**: 3-fold stratified CV
- **Train-Test Split**: 80/20 with stratification on Survived, Pclass, Sex

### Feature Importance (Top 5)
1. Sex (Male/Female)
2. Pclass
3. Fare
4. Age
5. SibSp (Siblings/Spouses)

---

## 🔧 Preprocessing Pipeline

The custom pipeline handles:

1. **Age Imputation**: Fills missing values with mean (~20% missing)
2. **Categorical Encoding**: One-hot encodes Sex and Embarked
3. **Feature Dropping**: Removes Name, Ticket, Cabin (high cardinality/missing data)
4. **Standardization**: Scales features to mean=0, std=1

All transformers are Scikit-learn compatible and located in `src/preprocessing.py`.

---

## 📝 Future Improvements

- [ ] Feature engineering: Extract titles from names (Mr., Mrs., Dr., etc.)
- [ ] Create family size feature (SibSp + Parch + 1)
- [ ] Try ensemble methods: XGBoost, Gradient Boosting, Voting Classifier
- [ ] Add SMOTE for handling class imbalance
- [ ] Implement learning curves and ROC-AUC analysis
- [ ] Add confusion matrix and classification report
- [ ] Deploy model with Flask/Streamlit web interface

---

## 📚 Resources & References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- Dataset based on: [NeuralNine's Tutorial](https://www.youtube.com/watch?v=fATVVQfFyU0)

---

## 👨‍💻 Author

**Tyrbujac**
- GitHub: [@tsbujacncl](https://github.com/tsbujacncl)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Kaggle for the dataset
- Scikit-learn community for excellent ML tools
- NeuralNine for the educational foundation

---

**⭐ If you found this project helpful, please give it a star!**