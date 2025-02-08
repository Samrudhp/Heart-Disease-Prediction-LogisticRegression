# Heart Disease Prediction Model

## Overview
This project implements a machine learning model to predict the presence of heart disease using logistic regression with advanced optimization techniques. The model achieves 87% accuracy in predicting heart disease cases.

## Dataset Features

### Numerical Features:
- **age**: Age of the patient
- **resting_bp_s**: Resting blood pressure (systolic) in mm/Hg
- **cholesterol**: Serum cholesterol in mg/dl
- **max_heart_rate**: Maximum heart rate achieved during exercise
- **oldpeak**: ST depression induced by exercise relative to rest

### Categorical Features:
- **sex**: Gender of the patient
- **chest_pain_type**: Type of chest pain (1: Typical angina, 2: Atypical angina, 3: Non-anginal pain, 4: Asymptomatic)
- **fasting_blood_sugar**: Blood sugar level > 120 mg/dl (1: True, 0: False)
- **resting_ecg**: Resting electrocardiographic results
- **exercise_angina**: Exercise-induced angina (1: Yes, 0: No)
- **st_slope**: Slope of the peak exercise ST segment

## Advanced Optimization Techniques

### 1. Data Preprocessing Pipeline
```python
Pipeline(steps=[
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])),
    ('classifier', LogisticRegression())
])
```

### 2. Hyperparameter Optimization
- **GridSearchCV** implementation with:
  - 5-fold cross-validation
  - Parameter grid:
    - C: [0.01, 0.1, 1, 10, 100]
    - Solver: ['liblinear', 'saga']
- Best Parameters Found:
  - C: 0.1
  - Solver: 'liblinear'

### 3. Model Performance Metrics
- **Accuracy**: 87%
- **ROC AUC Score**: 0.867
- **Detailed Classification Metrics**:
  - Precision: 0.87
  - Recall: 0.87
  - F1-score: 0.87

## Visualization Tools
1. **Confusion Matrix Heatmap**
   - Visual representation of model predictions
   - Uses seaborn for enhanced visualization

2. **ROC Curve Analysis**
   - AUC score visualization
   - False Positive vs True Positive Rate analysis

## Technical Implementation

### Dependencies
```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### Model Pipeline Components
1. **Data Preprocessing**
   - StandardScaler for numerical features
   - OneHotEncoder for categorical variables
   - ColumnTransformer for unified preprocessing

2. **Model Training**
   - Logistic Regression with increased max_iter
   - Cross-validation for robust evaluation
   - Hyperparameter tuning using GridSearchCV

## Usage Instructions

1. Install required packages:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

2. Load and prepare your data:
```python
data = pd.read_csv('heart-disease-dataset.csv')
X = data.drop('target', axis=1)
y = data['target']
```

3. Train the model:
```python
grid_search.fit(X_train, y_train)
```

4. Make predictions:
```python
predictions = best_model.predict(X_test)
```

## Results
- The model successfully identifies heart disease cases with 87% accuracy
- Balanced performance across both classes (heart disease present/absent)
- Robust against overfitting due to cross-validation and regularization

## Future Improvements
1. Feature engineering for better predictive power
2. Ensemble methods exploration
3. Deep learning implementation for complex patterns
4. Real-time prediction API development
