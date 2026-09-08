<p align="center">
  <img src="Assets/Banner.png" alt="Customer Churn Uplift Modeling Banner" width="100%">
</p>

---

<h1 align="center">📊 CUSTOMER CHURN UPLIFT MODELING</h1>
<p align="center"><b>End-to-End Machine Learning & Uplift Modeling Project</b></p>

End-to-end machine learning project for estimating the incremental impact of treatment and identifying observations most likely to benefit from an intervention.

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Uplift-Modeling-7C3AED?style=for-the-badge">
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-Production%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white">
  <img src="https://img.shields.io/badge/Pytest-Testing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white">
  <img src="https://img.shields.io/badge/GitHub%20Actions-CI-2088FF?style=for-the-badge&logo=githubactions&logoColor=white">
</p>

<h1 align="center">✨ Customer Churn Uplift Modeling</h1>

<p align="center">
<b>Predict Incremental Impact • Target Smarter • Deploy Reliably</b>
</p>

---

# 💡 About This Project

- This repository contains an end-to-end machine learning workflow for **uplift modeling and treatment-effect estimation**.

- The project focuses on identifying observations that are most likely to benefit from a treatment or intervention rather than simply predicting who is likely to convert.

- The complete workflow covers data understanding, data quality, exploratory analysis, feature engineering, baseline modeling, uplift modeling, evaluation, treatment targeting, production ML, experiment tracking, model registry, API serving, testing, Docker, and CI.

---

# 🎯 Project Objectives

- 📊 Understand the dataset
- 🧹 Perform data quality analysis
- 🔍 Conduct exploratory data analysis
- 🛠 Prepare modeling features
- 🤖 Build baseline models
- 🧠 Implement T-Learner
- 🧠 Implement S-Learner
- 🧠 Implement X-Learner
- 📈 Compare uplift models
- 📊 Evaluate uplift performance
- 🎯 Build treatment targeting policies
- 🏭 Create a production-oriented ML pipeline
- 🧪 Track experiments with MLflow
- 🗂 Register model versions
- 🚀 Serve predictions using FastAPI
- 🧪 Test the API automatically
- 🐳 Containerize the application
- ⚙️ Run automated CI with GitHub Actions

---

# 💼 Business Problem

Traditional machine learning answers:

```text
Who is likely to convert?

Uplift modeling answers:
Who is likely to convert because of the treatment?

                  Customer Population
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    Convert anyway   Treatment helps   Little impact
          │               │               │
          ▼               ▼               ▼
       Lower           High-value       Lower
      priority         target           priority

🧠 What Is Uplift Modeling?
Uplift modeling estimates the difference between the expected outcome under treatment and the expected outcome under control.

Expected Outcome with Treatment
              -
Expected Outcome without Treatment
              =
     Predicted Uplift

Mathematically:

Uplift(X)
=
P(Y = 1 | X, T = 1)
        -
P(Y = 1 | X, T = 0)

Where:

X = Input Features
T = Treatment Assignment
Y = Outcome

Interpretation:

     Positive Uplift
            ↓
     Treatment is expected to improve the outcome

     Near-Zero Uplift
            ↓
     Treatment has little expected incremental effect

     Negative Uplift
            ↓
     Treatment may not be beneficial


### 4. Dataset
```markdown
# 📊 Dataset

This project uses the official:

## Criteo Uplift Prediction Dataset v2.1

The dataset is designed for **uplift modeling and incrementality analysis**.

Official dataset source:

https://ailab.criteo.com/criteo-uplift-prediction-dataset/

The dataset contains approximately 14 million observations.

---

# 🧾 Dataset Columns

```text
f0
f1
f2
f3
f4
f5
f6
f7
f8
f9
f10
f11
treatment
conversion
visit
exposure


### 5. Data Understanding + Quality + EDA
```markdown
# 🔍 Data Understanding

The data understanding stage examines:

- Dataset dimensions
- Column names
- Data types
- Treatment assignment
- Outcome variables
- Feature distributions
- Basic statistics
- Treatment/control structure

---

# 🧹 Data Quality

The data quality stage checks:

- Missing values
- Duplicate observations
- Data types
- Treatment distribution
- Outcome distribution
- Feature statistics
- Treatment/control balance
- Potential data leakage

The workflow avoids using post-treatment information as predictive model input.

---

# 📊 Exploratory Data Analysis

The EDA stage investigates:

- Treatment/control balance
- Conversion rate
- Visit rate
- Exposure rate
- Feature distributions
- Feature correlations
- Treatment differences
- Outcome relationships

The analysis provides the foundation for feature engineering and uplift modeling.

---

# 🛠 Feature Engineering

The feature engineering stage prepares the available Criteo features for machine learning.

Primary modeling features:

```text
f0 - f11


### 6. ML Methodology
```markdown
# 🤖 Machine Learning Methodology

The project implements and compares three major uplift modeling approaches:

```text
T-Learner
S-Learner
X-Learner

1️⃣ T-Learner

The T-Learner trains separate models for the treatment and control groups.

                         Dataset
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
             Control               Treatment
              Group                  Group
                 │                     │
                 ▼                     ▼
          Control Model        Treatment Model
                 │                     │
                 ▼                     ▼
        P(Y | X, T=0)          P(Y | X, T=1)
                 │                     │
                 └──────────┬──────────┘
                            │
                            ▼
                   Treatment Effect
                            │
                            ▼
                    Predicted Uplift

2️⃣ S-Learner

The S-Learner uses a single model with treatment assignment included as a feature.
                      Features + Treatment
          │
          ▼
     Single Model
          │
     ┌────┴────┐
     │         │
     ▼         ▼
Treatment=1  Treatment=0
     │         │
     ▼         ▼
Prediction  Prediction
     │         │
     └────┬────┘
          ▼
     Uplift Score

3️⃣ X-Learner

The X-Learner estimates treatment effects using outcome models and treatment-effect models.
             Dataset
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
   Treatment           Control
     Group               Group
       │                 │
       ▼                 ▼
 Outcome Model       Outcome Model
       │                 │
       ▼                 ▼
Treatment Effect    Treatment Effect
   Estimation          Estimation
       │                 │
       └────────┬────────┘
                ▼
          Final Uplift


### 7. Evaluation
```markdown
# 📈 Uplift Model Evaluation

Uplift models require evaluation methods that measure treatment-effect ranking rather than only conventional classification performance.

The project evaluates models using:

- Qini Curve
- Qini Coefficient
- Uplift@K
- Treatment targeting performance

---

# 📊 Qini Curve

The Qini curve evaluates how effectively the model ranks observations according to expected incremental treatment effect.

```text
Cumulative Incremental Gain
          │
          │                    /
          │                  /
          │                /
          │             __/
          │          __/
          │       __/
          │______/________________
                 Targeted Population


### 8. Treatment Policy
```markdown
# 🎯 Treatment Targeting Policy

The predicted uplift scores are converted into a treatment targeting policy.

```text
Predicted Uplift
       │
       ▼
Rank Observations
       │
       ▼
Highest Uplift First
       │
       ▼
Select Top K%
       │
       ▼
Treatment Policy


### 9. Production ML + MLflow
```markdown
# 🏭 Production ML Pipeline

The production-oriented pipeline performs:

```text
Load Data
    ↓
Select Features
    ↓
Separate Treatment / Control
    ↓
Train Control Model
    ↓
Train Treatment Model
    ↓
Generate Uplift Predictions
    ↓
Evaluate Model
    ↓
Save Model Artifacts
    ↓
Save Metrics


### 10. FastAPI
```markdown
# 🚀 FastAPI Model Serving

The trained production T-Learner is exposed through a REST API using FastAPI.

The API loads the trained model artifacts:

```text
models/
├── t_learner_control_model.joblib
└── t_learner_treatment_model.joblib


### 11. Testing + Docker + CI
```markdown
# 🧪 Automated Testing

The API is tested using:

```text
pytest


### 12. Complete project structure
```markdown
# 📂 Repository Structure

```text
customer-churn-uplift-modeling/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── api/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── t_learner_control_model.joblib
│   ├── t_learner_treatment_model.joblib
│   ├── t_learner_model_metadata.json
│   └── t_learner_registry_package.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_modeling.ipynb
│   ├── 06_uplift_modeling_t_learner.ipynb
│   ├── 07_uplift_evaluation.ipynb
│   ├── 08_uplift_model_comparison.ipynb
│   ├── 09_treatment_targeting_policy.ipynb
│   ├── 10_production_ml_pipeline.ipynb
│   ├── 11_experiment_tracking.ipynb
│   ├── 12_model_registry.ipynb
│   ├── 13_api_model_serving.ipynb
│   └── 14_api_testing.ipynb
│
├── reports/
│   └── production_model_metrics.csv
│
├── src/
│   └── uplift/
│       ├── __init__.py
│       └── t_learner.py
│
├── tests/
│   ├── conftest.py
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt


### 13. Notebook roadmap
```markdown
# 📓 Notebook Workflow

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_understanding.ipynb` | Dataset structure and initial understanding |
| 02 | `02_data_quality.ipynb` | Data quality analysis and validation |
| 03 | `03_eda.ipynb` | Exploratory data analysis |
| 04 | `04_feature_engineering.ipynb` | Feature preparation |
| 05 | `05_baseline_modeling.ipynb` | Baseline machine learning |
| 06 | `06_uplift_modeling_t_learner.ipynb` | T-Learner implementation |
| 07 | `07_uplift_evaluation.ipynb` | Uplift evaluation |
| 08 | `08_uplift_model_comparison.ipynb` | T/S/X Learner comparison |
| 09 | `09_treatment_targeting_policy.ipynb` | Treatment targeting strategy |
| 10 | `10_production_ml_pipeline.ipynb` | Production-oriented ML pipeline |
| 11 | `11_experiment_tracking.ipynb` | MLflow experiment tracking |
| 12 | `12_model_registry.ipynb` | MLflow model registry |
| 13 | `13_api_model_serving.ipynb` | FastAPI model serving |
| 14 | `14_api_testing.ipynb` | API testing |

---

# 📌 End-to-End Workflow

```text
Data Understanding
        │
        ▼
Data Quality
        │
        ▼
EDA
        │
        ▼
Feature Engineering
        │
        ▼
Baseline Modeling
        │
        ▼
Uplift Modeling
        │
   ┌────┼────┐
   ▼    ▼    ▼
   T    S    X
   │    │    │
   └────┼────┘
        ▼
Model Comparison
        │
        ▼
Uplift Evaluation
        │
        ▼
Treatment Targeting
        │
        ▼
Production Pipeline
        │
   ┌────┴────┐
   ▼         ▼
 MLflow    FastAPI
   │         │
   ▼         ▼
Registry   Docker
             │
             ▼
       Docker Compose
             │
             ▼
       Automated Tests
             │
             ▼
      GitHub Actions


### 14. Installation + running
```markdown
# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/potnuruteja115/customer-churn-uplift-modeling.git


### 15. Tech stack + outcomes + future
```markdown
# 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning |
| Random Forest | Outcome and treatment-effect modeling |
| Jupyter Notebook | Data science experimentation |
| MLflow | Experiment tracking and model registry |
| FastAPI | REST API model serving |
| Pydantic | API validation |
| Joblib | Model serialization |
| Pytest | Automated testing |
| Docker | Containerization |
| Docker Compose | Local container orchestration |
| Git | Version control |
| GitHub | Source control and collaboration |
| GitHub Actions | Continuous integration |

---

# 📦 Deliverables

- ✅ Data Understanding
- ✅ Data Quality Analysis
- ✅ Exploratory Data Analysis
- ✅ Feature Engineering
- ✅ Baseline Modeling
- ✅ T-Learner
- ✅ S-Learner
- ✅ X-Learner
- ✅ Uplift Model Comparison
- ✅ Qini Evaluation
- ✅ Uplift@K Analysis
- ✅ Treatment Targeting Policy
- ✅ Production ML Pipeline
- ✅ MLflow Experiment Tracking
- ✅ Model Registry
- ✅ FastAPI REST API
- ✅ Automated API Tests
- ✅ Docker Image
- ✅ Docker Compose
- ✅ GitHub Actions CI
- ✅ Technical Documentation

---

# 🎓 Learning Outcomes

This project demonstrates practical experience in:

```text
Machine Learning
       +
Treatment Effect Estimation
       +
Uplift Modeling
       +
Model Evaluation
       +
Targeting Strategy
       +
Experiment Tracking
       +
Model Registry
       +
Production ML
       +
REST API Development
       +
Containerization
       +
Automated Testing
       +
Continuous Integration


### 16. Final closing
```markdown
# 🏁 Conclusion

This project demonstrates the complete journey from exploratory data science to a production-oriented uplift modeling system.

The final workflow combines:

```text
Data Science
      +
Uplift Modeling
      +
Treatment Effect Estimation
      +
Model Evaluation
      +
Treatment Targeting
      +
Experiment Tracking
      +
Model Registry
      +
FastAPI
      +
Docker
      +
Automated Testing
      +
GitHub Actions
     


