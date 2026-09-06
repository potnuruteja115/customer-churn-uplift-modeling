# Customer Churn Uplift Modeling

An end-to-end machine learning project for estimating the incremental impact of a treatment on customer conversion and identifying customers who are most likely to respond positively to an intervention.

> **Important:** This project uses the official Criteo Uplift Prediction Dataset. The dataset is designed for uplift and incrementality modeling rather than literal customer churn prediction. Therefore, this project focuses on treatment effect estimation, uplift modeling, and targeted campaign decision-making.

---

## Project Overview

Traditional machine learning models predict whether a customer will convert.

Uplift modeling answers a more useful business question:

> **"Will this customer convert because of the treatment?"**

The objective of this project is to estimate the individual treatment effect and use those estimates to create an effective treatment targeting policy.

The project covers the complete machine learning workflow:

- Data understanding
- Data quality analysis
- Exploratory data analysis
- Feature engineering
- Baseline modeling
- Uplift modeling
- Uplift model comparison
- Uplift evaluation
- Treatment targeting policy
- Production ML pipeline
- MLflow experiment tracking
- Model registry
- FastAPI model serving
- Automated API testing
- Docker containerization
- Docker Compose deployment
- GitHub Actions CI

---

## Business Problem

Suppose a company wants to run a marketing campaign.

If the company contacts everyone:

- Some customers would convert anyway.
- Some customers convert because of the campaign.
- Some customers are unaffected.
- Some customers may respond negatively.

A conventional conversion model does not distinguish these groups.

Uplift modeling estimates the incremental effect of treatment:

```text
Expected outcome with treatment
             -
Expected outcome without treatment
             =
Predicted uplift

---

## FastAPI Model Serving

The project provides a FastAPI service for serving the trained T-Learner model.

The API loads the trained control and treatment models from the `models/` directory and calculates the predicted uplift for a given set of features.

### Start the API

```bash
uvicorn api.app:app --reload