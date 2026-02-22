# Hybrid ML System for Multi-Class Rating Prediction

## Problem Statement

The objective of this project is to predict user ratings (0–5 scale) for recipes using structured engagement data and textual reviews.

The dataset includes user interaction metrics such as reputation, replies, and votes, along with review text. The main challenges include:

* Imbalanced target distribution
* Weak direct correlations between features and ratings
* High-dimensional text features
* Multi-class classification

The goal was to design a robust model that generalizes well across rating levels.

---

## Approach

### 1 Exploratory Data Analysis (EDA)

* Analyzed rating distribution and identified class imbalance
* Studied feature correlations and skewness
* Explored relationships between engagement metrics and ratings
* Identified need for non-linear modeling approaches

---

### 2️Data Preprocessing

* Removed high-cardinality and non-informative columns
* Handled missing values
* Scaled numerical features using `StandardScaler`
* Used stratified train-test splitting to preserve class proportions

---

### 3️Feature Engineering

* Engineered structured engagement features
* Converted text reviews into numerical vectors using **TF-IDF**
* Applied mutual information–based feature selection
* Used dimensionality reduction (PCA) to manage high-dimensional inputs

Final feature space combined tabular and text representations.

---

### 4️Model Training & Hyperparameter Tuning

Trained and compared multiple supervised learning models:

* Logistic Regression
* Random Forest
* HistGradientBoostingClassifier
* SGDClassifier
* XGBoost

Performed **hyperparameter tuning using GridSearchCV** and evaluated models using stratified validation.

Evaluation methods included:

* Accuracy
* Confusion matrix analysis
* Class-level performance metrics

---

## Results

* HistGradientBoostingClassifier achieved the best overall generalization performance.
* Confusion matrix analysis helped identify misclassification patterns across rating levels.
* Incorporating text features improved performance compared to tabular-only models.
* Stratified validation ensured reliable evaluation under imbalanced class distribution.

---

## Tech Stack

* Python
* NumPy, Pandas
* scikit-learn
* XGBoost
* Matplotlib, Seaborn

---

## Key Learnings

* Combining structured and text features improves predictive performance.
* Non-linear models outperform linear models when feature-target correlations are weak.
* Hyperparameter tuning is critical for improving generalization.
* Proper validation strategies are essential in imbalanced multi-class problems.
* Error analysis using confusion matrices provides deeper insight than accuracy alone.
