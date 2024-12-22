# Employee Attrition Classification Analysis

This repository contains a classification analysis project aimed at predicting employee attrition (whether an employee will leave the organization or stay) based on various demographic, job satisfaction, and performance attributes.

## Problem Statement

The goal of this project is to predict employee attrition using classification algorithms. We aim to build a model that can predict whether an employee will leave the organization (attrition) or stay, based on their characteristics such as demographics, job satisfaction, and performance metrics.

## Dataset

The dataset used in this project is related to employee data and contains features such as:

- Age
- Job role
- Salary
- Years at the company
- Satisfaction levels
- Distance from home, etc.

The target variable is **Attrition**, which indicates whether the employee left the organization (`Yes`) or stayed (`No`).

## Key Features

- **Age**: Age of the employee.
- **Attrition**: Target variable indicating employee attrition.
- **BusinessTravel**: Frequency of business travel.
- **DailyRate**: Daily rate of the employee.
- **DistanceFromHome**: Distance of the employee’s residence from the office.
- **Education**: Level of education.
- **EmployeeCount**: Number of employees in the company.
- **JobSatisfaction**: Job satisfaction level of the employee.
- **YearsAtCompany**: Number of years the employee has worked at the company.

## Approach

### 1. **Data Preprocessing**
   - Handle class imbalance using class weights.
   - Normalize numerical features.
   - Split the dataset into training and testing sets.

### 2. **Exploratory Data Analysis (EDA)**
   - Visualize feature relationships with the target variable.
   - Identify key features influencing employee attrition.
   - Detect and address outliers.

### 3. **Model Implementation**
   - Support Vector Machine (SVM)
   - Logistic Regression
   - Decision Trees
   - Random Forest

### 4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 score
   - ROC-AUC

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2. Navigate to the project directory:

    ```bash
    cd Employee_Attrition
    ```

3. Install the required libraries:

    ```bash
    using pip install pandas numpy seaborn matplotlib statsmodels scikit-learn
    ```

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Import the dataset:

    ```python
    df = pd.read_csv('employee_attrition.csv')
    ```

2. Perform preprocessing and feature encoding:

    ```python
    # Change target column 'Attrition' to numerical values
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    ```

3. Split the data into training and testing sets:

    ```python
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    ```

4. Train and evaluate models (e.g., SVM):

    ```python
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM model
    svm_model = SVC(kernel='linear', probability=True, class_weight='balanced')
    svm_model.fit(X_train, y_train)

    # Make predictions and evaluate metrics
    y_pred = svm_model.predict(X_test)
    ```

5. Generate evaluation metrics:

    ```python
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
    print(f'Precision: {precision_score(y_test, y_pred):.3f}')
    print(f'Recall: {recall_score(y_test, y_pred):.3f}')
    print(f'F1 Score: {f1_score(y_test, y_pred):.3f}')
    print(f'ROC-AUC: {roc_auc_score(y_test, svm_model.predict_proba(X_test)[:,1]):.3f}')
    ```
