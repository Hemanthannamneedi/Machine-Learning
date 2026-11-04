# Machine Learning Internship — Day 41 to Day 60

This repository documents my Machine Learning internship work from **Day 41 to Day 60**. During this period, I practised data analysis with NumPy and Pandas, data preprocessing, exploratory data analysis, supervised machine learning, model evaluation, feature engineering, model optimization, neural networks, deep learning, model deployment, and API development.

The work includes practical Python scripts, datasets, visualizations, trained models, evaluation reports, and FastAPI applications.

## Learning Roadmap

| Day | Topic / Project | Key Work |
|---|---|---|
| 41 | NumPy Basics | NumPy arrays, array arithmetic, slicing, reshaping, and broadcasting. |
| 42 | Pandas Basics | Loading data, inspecting rows and columns, checking shape and data types, and filtering records. |
| 43 | Data Cleaning & Categorical Encoding | Handling missing values using `dropna()` and `fillna()`, applying `LabelEncoder`, and using `pd.get_dummies()`. |
| 44 | Data Visualization | Creating and saving data visualizations using Python plotting libraries. |
| 45 | Exploratory Data Analysis (EDA) | Exploring datasets, understanding distributions and relationships, and identifying patterns. |
| 46 | Regression Fundamentals | Building and evaluating regression models. |
| 47 | Regression Evaluation & Cross-Validation | Comparing regression performance using evaluation metrics and cross-validation. |
| 48 | Classification Fundamentals | Building Logistic Regression and Decision Tree classification models using a customer churn dataset. |
| 49 | Random Forest & Classification Metrics | Training Random Forest classifiers and evaluating classification performance. |
| 50 | House Price Prediction — Complete ML Pipeline | Building a regression project with preprocessing, feature engineering, Linear Regression, evaluation metrics, cross-validation, and model persistence. |
| 51 | Feature Engineering | Creating useful features and identifying outliers using IQR-based methods. |
| 52 | Feature Scaling, KNN & SVM | Applying feature scaling and practising K-Nearest Neighbors and Support Vector Machine classification. |
| 53 | ML Pipelines & Imbalanced Data | Building preprocessing pipelines and comparing baseline, class-weighted, and SMOTE-based approaches. |
| 54 | Hyperparameter Tuning, Gradient Boosting & XGBoost | Using GridSearchCV and cross-validation to tune and compare machine learning models. |
| 55 | Advanced Customer Churn Prediction | Developing an end-to-end churn prediction project with feature engineering, multiple classifiers, tuning, and model saving. |
| 56 | Artificial Neural Networks (ANN) Basics | Building and comparing neural networks with different hidden-layer neuron configurations using TensorFlow/Keras. |
| 57 | Convolutional Neural Networks (CNN) | Training a CNN on the MNIST handwritten-digit dataset and evaluating predictions. |
| 58 | Recurrent Neural Networks (RNN) | Building SimpleRNN models to predict the next value in numerical sequences. |
| 59 | Model Saving, Loading & API Integration | Saving trained models and creating single and batch prediction endpoints using FastAPI. |
| 60 | Month 3 Capstone — AI-Powered Application | Building a complete customer churn prediction application using a trained ML model and FastAPI. |

## Technologies & Libraries

- **Programming Language:** Python
- **Data Processing:** NumPy, Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Imbalanced Learning:** imbalanced-learn
- **Gradient Boosting:** XGBoost
- **Deep Learning:** TensorFlow, Keras
- **Model Persistence:** Joblib
- **API Development:** FastAPI, Uvicorn, Pydantic
- **Development Tools:** Jupyter Notebook, Git, GitHub

## Project Highlights

### 1. House Price Prediction

**Project:** `house-price-ml` — Day 50

- Built a complete regression workflow using a house-price dataset.
- Applied data preprocessing and feature engineering.
- Trained a Linear Regression model.
- Evaluated the model using MAE, MSE, RMSE, and R².
- Applied 5-fold cross-validation.
- Saved and loaded the trained model using Joblib.
- Created actual-versus-predicted visualizations.

### 2. Customer Churn Prediction

The customer churn dataset introduced during the classification exercises was reused in later machine learning projects.

- Implemented Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, and XGBoost models across different exercises.
- Practised data preprocessing, feature engineering, and feature scaling.
- Explored class weighting and SMOTE for imbalanced datasets.
- Evaluated models using classification metrics and comparison reports.
- Saved trained models for future predictions.

### 3. Neural Network & Deep Learning Exercises

- **ANN:** Built and compared neural network configurations using the customer churn dataset.
- **CNN:** Classified handwritten digits using the MNIST dataset.
- **RNN:** Practised sequence modelling by predicting the next value in generated numerical sequences.

### 4. FastAPI Model Deployment

Days 59 and 60 introduced model serving through FastAPI.

- Loaded saved machine learning pipelines.
- Created input validation schemas using Pydantic.
- Implemented single-record and batch prediction endpoints.
- Added health and model-information endpoints.
- Implemented prediction caching and cache management.
- Tested APIs through the automatically generated Swagger UI.

## Repository Structure

The exercises are organized by day, with separate folders or repositories for standalone projects.

```text
Machine Learning/
│
├── day41/
├── day42/
├── day43/
├── ...
├── day59/
│
├── house-price-ml/
│
├── churn-prediction-advanced/
│
├── ai-powered-app/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── model_service.py
│   │
│   ├── data/
│   │   └── churn.csv
│   │
│   ├── models/
│   ├── training/
│   │   └── train_model.py
│   │
│   ├── requirements.txt
│   ├── README.md
│   └── assignment.txt
│
└── README.md
```

## Running the Projects

Each project may contain its own `requirements.txt` file.

Open Command Prompt and navigate to the required project directory.

```cmd
cd path\to\project

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

Run the relevant Python script using the filename provided in that day's folder.

For notebook-based exercises, open the notebook using Jupyter Notebook.

### Running the Day 60 Capstone

Navigate to the `ai-powered-app` directory.

Install the dependencies:

```cmd
pip install -r requirements.txt
```

Train the machine learning model:

```cmd
python training\train_model.py
```

Start the FastAPI application:

```cmd
python -m uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

The model must be trained before starting the API because the application loads the saved model during startup.

## Notes & Limitations

- The customer churn dataset used in these exercises is small and intended for learning and experimentation. Its evaluation scores should not be considered production-level performance.
- Model results may vary depending on the train/test split, preprocessing, random state, and model configuration.
- SMOTE and other resampling techniques should only be applied to training data.
- For mixed numerical and categorical data, SMOTENC may be more appropriate than applying ordinary SMOTE after one-hot encoding.
- The MNIST dataset may need to be downloaded during the first CNN execution.
- TensorFlow and XGBoost installation compatibility may depend on the Python version and operating system.

## Learning Outcomes

By completing Days 41–60, I gained practical experience in:

- Working with numerical and tabular data using NumPy and Pandas.
- Cleaning datasets and handling missing values.
- Performing exploratory data analysis and creating visualizations.
- Building regression and classification models.
- Evaluating models and applying cross-validation.
- Performing feature engineering and feature scaling.
- Handling imbalanced datasets.
- Applying hyperparameter tuning and comparing algorithms.
- Building ANN, CNN, and RNN models using TensorFlow/Keras.
- Saving, loading, and reusing trained machine learning models.
- Developing prediction APIs using FastAPI and Pydantic.
- Organizing projects and maintaining code using Git and GitHub.

## Final Capstone — Day 60

**Project Name:** AI-Powered Application

The final project combines machine learning and backend API development.

It includes:

- A trained Random Forest classification pipeline.
- Model saving and versioning.
- FastAPI backend integration.
- Single and batch prediction endpoints.
- Input validation and error handling.
- Prediction caching.
- API documentation using Swagger UI.
- Project documentation and GitHub version control.

**GitHub Release Tag:**

```text
v3.0-ml-fundamentals
```

---

**Author:** Hemanth Annamneedi  
**Focus:** Machine Learning | Python | Data Analysis | Deep Learning | FastAPI
