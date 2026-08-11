# Medical Insurance Charges Prediction

A machine learning regression project that predicts **medical insurance charges** based on demographic and lifestyle information such as age, BMI, number of children, gender, smoking status, and region.

The project covers the complete machine learning workflow:

**Data Loading → EDA → Train/Test Split → Preprocessing → Model Training → Model Comparison → Model Saving → FastAPI Backend → Cloud Deployment → Streamlit Frontend**

---

## 📌 Project Overview

Medical insurance charges depend on several factors, including age, BMI, smoking habits, family size, gender, and geographical region.

The objective of this project is to build a machine learning regression model that can predict an individual's **medical insurance charges** using these features.

Multiple regression algorithms were trained and evaluated, and **Gradient Boosting Regressor** achieved the best performance.

The trained model was then integrated with a **FastAPI backend** and a **Streamlit frontend** to create a deployable prediction application.

---

## 🎯 Objectives

* Perform exploratory data analysis on the insurance dataset.
* Clean and prepare the raw dataset.
* Split the data into training and testing sets.
* Apply appropriate preprocessing techniques.
* Compare multiple regression algorithms.
* Select the best-performing model using evaluation metrics.
* Save the trained model and preprocessing scaler.
* Build a REST API using FastAPI.
* Deploy the backend using Render.
* Build an interactive frontend using Streamlit.
* Deploy the frontend using Streamlit Cloud.

---

## 📊 Dataset

The project uses a medical insurance dataset containing information about individuals and their corresponding insurance charges.

### Features

| Feature    | Description                                 |
| ---------- | ------------------------------------------- |
| `age`      | Age of the individual                       |
| `sex`      | Gender of the individual                    |
| `bmi`      | Body Mass Index                             |
| `children` | Number of children/dependents               |
| `smoker`   | Whether the individual is a smoker          |
| `region`   | Residential region                          |
| `charges`  | Medical insurance charges — target variable |

### Target Variable

`charges`

This is a **regression problem** because the target variable is a continuous numerical value.

---

# 🔄 Machine Learning Workflow

## 1. Data Loading and EDA

The initial data loading, basic cleaning, exploratory data analysis, and train-test split were performed in:

```text
notebooks/01_data_loading_and_eda.ipynb
```

The raw dataset was divided into training and testing datasets.

The resulting files were saved as:

```text
data/raw/train.csv
data/raw/test.csv
```

The raw dataset is preserved separately so that the original split remains available.

---

## 2. Data Preprocessing

Preprocessing was performed in:

```text
notebooks/02_data_preprocessing.ipynb
```

The following transformations were applied.

### Gender Encoding

The `sex` column was converted into a numerical feature:

```text
sex → is_female
```

This was performed using one-hot encoding.

### Smoker Encoding

The `smoker` column was converted into:

```text
smoker → is_smoker
```

using one-hot encoding.

### Region Encoding

The categorical `region` feature was transformed using one-hot encoding so that it could be used by the machine learning models.

### Feature Scaling

Standard scaling was applied to:

```text
age
bmi
children
```

The fitted scaler was saved for use during inference:

```text
models/scaler.pkl
```

Saving the scaler is important because the same preprocessing transformation must be applied to new input data before making predictions.

### Pearson Correlation

Pearson correlation was calculated between the numerical features and the target variable `charges` to understand the linear relationship between the features and insurance charges.

---

## 3. Processed Dataset

After preprocessing, the processed datasets were saved as:

```text
data/processed/train.csv
data/processed/test.csv
```

These processed datasets were then used for model training and evaluation.

---

# 🤖 Model Training

Model training and evaluation were performed in:

```text
notebooks/03_model_training.ipynb
```

The processed data was separated into:

```text
X_train
X_test
y_train
y_test
```

Multiple regression algorithms were trained and compared.

### Models Evaluated

```python
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}
```

---

# 📈 Model Comparison

The models were evaluated using:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**
* **R² Score**
* **Adjusted R² Score**

### Results

| Model                 |         MAE |        RMSE |     R² Score |  Adjusted R² |
| --------------------- | ----------: | ----------: | -----------: | -----------: |
| **Gradient Boosting** | **2533.11** | **4270.18** | **0.900768** | **0.899925** |
| Random Forest         |     2559.49 |     4650.83 |     0.882289 |     0.881288 |
| Linear Regression     |     4177.05 |     5956.34 |     0.806929 |     0.805288 |
| Lasso Regression      |     4177.87 |     5957.62 |     0.806846 |     0.805205 |
| Ridge Regression      |     4193.83 |     5971.69 |     0.805933 |     0.804283 |
| Decision Tree         |     2958.57 |     6402.96 |     0.776890 |     0.774993 |

### 🏆 Best Model

**Gradient Boosting Regressor** achieved the best overall performance.

It obtained:

```text
MAE          = 2533.11
RMSE         = 4270.18
R² Score     = 0.900768
Adjusted R²  = 0.899925
```

Based on these evaluation metrics, Gradient Boosting was selected as the final model.

The trained model was saved as:

```text
models/gradient_boosting_model.pkl
```

---

# 💾 Saved ML Artifacts

The trained model and preprocessing object are stored in the `models/` directory.

```text
models/
├── gradient_boosting_model.pkl
└── scaler.pkl
```

### `gradient_boosting_model.pkl`

Contains the trained Gradient Boosting regression model used for predicting insurance charges.

### `scaler.pkl`

Contains the fitted StandardScaler used during preprocessing.

Both artifacts are required during inference.

---

# 🚀 Deployment Architecture

The project uses separate frontend and backend components.

```text
                    User
                      │
                      ▼
              Streamlit Frontend
                      │
                      │ HTTP Request
                      ▼
                FastAPI Backend
                      │
                      ▼
             Preprocessing / Scaler
                      │
                      ▼
          Gradient Boosting Model
                      │
                      ▼
              Predicted Charges
                      │
                      ▼
              Streamlit Frontend
```

### Technologies Used

| Component            | Technology      |
| -------------------- | --------------- |
| Programming Language | Python          |
| Data Manipulation    | Pandas, NumPy   |
| Machine Learning     | Scikit-learn    |
| API Development      | FastAPI         |
| API Server           | Uvicorn         |
| Frontend             | Streamlit       |
| Backend Deployment   | Render          |
| Frontend Deployment  | Streamlit Cloud |
| Model Serialization  | Pickle          |
| Version Control      | Git & GitHub    |

---

# 🔌 FastAPI Backend

The backend was developed using **FastAPI**.

Backend files:

```text
backend/
├── app.py
└── schemas.py
```

### `schemas.py`

Defines the input schema using Pydantic.

The API accepts the following information:

```text
age
bmi
children
is_female
is_smoker
region
```

### `app.py`

The FastAPI application:

1. Receives user input.
2. Validates the input using Pydantic.
3. Applies the saved scaler.
4. Loads the trained Gradient Boosting model.
5. Generates the insurance charge prediction.
6. Returns the prediction through the API.

---

# 🌐 Backend Deployment

The FastAPI backend was deployed using **Render**.

The deployed backend provides an API endpoint that the frontend can send prediction requests to.

The backend and frontend are deployed separately.

```text
Streamlit Cloud
      │
      │ API Request
      ▼
Render
      │
      ▼
FastAPI
      │
      ▼
ML Model
```

---

# 🖥️ Streamlit Frontend

The frontend was developed using Streamlit.

Frontend file:

```text
frontend/
└── main.py
```

The Streamlit application provides an interactive interface where users can enter:

* Age
* BMI
* Number of children
* Gender
* Smoking status
* Region

The frontend sends the input data to the deployed FastAPI backend and displays the predicted medical insurance charges.

---

# 📁 Project Structure

```text
Medical-insurance-charges-prediction/
│
├── backend/
│   ├── app.py
│   └── schemas.py
│
├── data/
│   ├── processed/
│   │   ├── test.csv
│   │   └── train.csv
│   │
│   └── raw/
│       ├── insurance-charges.xls
│       ├── test.csv
│       └── train.csv
│
├── frontend/
│   └── main.py
│
├── models/
│   ├── gradient_boosting_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ashutoshkashyap04/Medical-insurance-charges-prediction.git
```

Navigate into the project directory:

```bash
cd Medical-insurance-charges-prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/Scripts/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Backend Locally

From the project root:

```bash
uvicorn backend.app:app --reload
```

The FastAPI server will start locally.

The API documentation can then be accessed through FastAPI's automatically generated Swagger UI.

```text
/docs
```

---

# ▶️ Running the Frontend Locally

Open another terminal and run:

```bash
streamlit run frontend/main.py
```

The Streamlit application will open in your browser.

---

# 🔗 API Input Example

The backend accepts input containing:

```json
{
    "age": 30,
    "bmi": 25.5,
    "children": 2,
    "is_female": 1,
    "is_smoker": 0,
    "region": "southeast"
}
```

The API processes these features and returns the predicted insurance charges.

---

# 🧠 Key Machine Learning Concepts Used

This project demonstrates practical implementation of:

* Exploratory Data Analysis
* Train-Test Split
* Categorical Feature Encoding
* One-Hot Encoding
* Feature Scaling
* Pearson Correlation
* Regression
* Model Evaluation
* Model Comparison
* Gradient Boosting
* Model Serialization
* REST API Development
* Cloud Deployment
* Frontend-Backend Integration

---

# 📚 Project Pipeline

The complete pipeline can be summarized as:

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Train/Test Split
     │
     ▼
Feature Encoding
     │
     ▼
Feature Scaling
     │
     ▼
Correlation Analysis
     │
     ▼
Processed Dataset
     │
     ▼
Train Multiple Regression Models
     │
     ▼
Model Evaluation
     │
     ▼
Select Gradient Boosting
     │
     ▼
Save Model + Scaler
     │
     ▼
FastAPI Backend
     │
     ▼
Render Deployment
     │
     ▼
Streamlit Frontend
     │
     ▼
Streamlit Cloud Deployment
```

---

# 📌 Key Observation

Among the evaluated models, ensemble tree-based methods performed better than the linear models for this dataset.

**Gradient Boosting Regressor** achieved the highest R² score and the lowest RMSE among the tested models, making it the final selected model.

```text
Gradient Boosting
R² = 0.900768
RMSE = 4270.18
MAE = 2533.11
```

This indicates that the model was able to explain approximately **90% of the variance** in the insurance charges on the test set.

---

# 🔮 Future Improvements

Possible improvements for future versions include:

* Hyperparameter tuning of the Gradient Boosting model.
* Cross-validation for more robust model evaluation.
* Feature importance visualization.
* Prediction confidence/uncertainty estimation.
* Improved frontend UI/UX.
* Input validation and better error handling.
* Dockerizing the FastAPI backend.
* Adding automated CI/CD.
* Monitoring model performance after deployment.
* Experiment tracking using an MLOps tool.

---

# 👨‍💻 Author

**Ashutosh Kashyap**

BS-MS in Artificial Intelligence and Cyber Security
IIT Patna

---

# ⭐ Acknowledgements

This project was developed as a practical implementation of a complete machine learning workflow, from data preprocessing and model training to API development and cloud deployment.
