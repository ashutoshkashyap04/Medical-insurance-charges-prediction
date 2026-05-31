# Medical Insurance Charges Prediction

## 📌 Project Overview
This project focuses on predicting **medical insurance charges** based on personal and lifestyle attributes. The goal is to build a regression model that can estimate insurance costs using historical data.

The project follows a complete **end-to-end Machine Learning workflow**, including data preprocessing, feature engineering, statistical analysis, model building, and evaluation.

---

## 📊 Dataset Information

The dataset contains the following features:

- **age** → Age of the individual  
- **sex** → Gender of the individual  
- **bmi** → Body Mass Index  
- **children** → Number of dependents  
- **smoker** → Smoking status  
- **region** → Residential area  
- **charges** → Medical insurance cost (**Target Variable**)  

---

## 🧹 Exploratory Data Analysis, Data Preprocessing & Feature Engineering

The following steps were performed:

- Converted categorical variables using **Label Encoding**
  - `sex` → converted to `isfemale`
  - `smoker` → converted to `issmoker`

- Created a new feature:
  - `bmi_category` from BMI values

- Applied **One-Hot Encoding** using `pd.get_dummies()` for:
  - `region`
  - `bmi_category`

- Converted encoded columns into numeric format

- Performed **Pearson Correlation** analysis for numerical features to understand relationships with the target variable.
- Applied **Chi-Square Test of Independence (`chi2_contingency`)** for categorical features to check statistical association with target bins.


---



## 🤖 Model Building

- Applied **Regression Models** to predict insurance charges
- Trained model using processed dataset
- Split data into training and testing sets

---

## 📉 Model Evaluation

The model was evaluated using:

- **R² Score**
- **Adjusted R² Score**

### 📌 Results:
- R² Score ≈ **0.80**
- Adjusted R² Score ≈ **0.80**

This indicates that the model explains approximately **80% of the variance** in medical insurance charges.

---

## 🧠 Key Insights

- Smoking status has a strong impact on insurance charges
- BMI and age are significant predictors
- Feature engineering improved model performance
- Statistical tests helped in better feature selection

---

## 📁 Project Structure
Medical-insurance-charges-prediction/

│

├── insurance-charges-prediction.ipynb

├── insurance-charges.xls

└── README.md




---

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

## 📌 Conclusion

This project demonstrates a complete machine learning pipeline for regression, from data preprocessing to model evaluation. The final model achieves good predictive performance with an R² score of approximately 80%.

---

## 👨‍💻 Author

- Ashutosh Kashyap

---
