# Heart Disease Prediction

This project predicts the likelihood of heart disease using machine learning models. It includes data preprocessing, visualization, and classification models such as Logistic Regression and Random Forest.

---

## Features
- Data Preprocessing: Handling missing values and scaling numerical data.
- Exploratory Data Analysis (EDA): Insights through visualizations.
- Classification Models: Logistic Regression and Random Forest Classifier.
- Custom Prediction: Accepts user inputs to predict the likelihood of heart disease.

---

## Technologies Used
- Python
- Libraries:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn

---

## Dataset
The dataset used is `heart.csv`, containing the following features:
- **age**: Age of the patient.
- **sex**: Gender (1 = Male, 0 = Female).
- **cp**: Chest pain type (0-3).
- **trestbps**: Resting blood pressure (mm Hg).
- **chol**: Cholesterol level (mg/dl).
- **fbs**: Fasting blood sugar (>120 mg/dl, 1 = True, 0 = False).
- **restecg**: Resting electrocardiographic results (0-2).
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise-induced angina (1 = Yes, 0 = No).
- **oldpeak**: ST depression induced by exercise.
- **slope**: Slope of the peak exercise ST segment (0-2).
- **ca**: Number of major vessels colored by fluoroscopy (0-3).
- **thal**: Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect).
- **target**: Output variable (1 = Heart Disease, 0 = No Heart Disease).

---

## How to Use
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
