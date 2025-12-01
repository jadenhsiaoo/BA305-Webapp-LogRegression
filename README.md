# 🚀 Loan Default Prediction Web App

**Optimized Logistic Regression + Flask UI**

This project is a full-stack machine learning web application that predicts **loan default probability** using an optimized logistic regression model.  
Users enter loan-related features through a simple web interface, and the model returns the predicted probability of default.

---

## ✨ Features

### 🔹 Optimized Logistic Regression Model

- Elastic Net regularization (L1 + L2)
- Polynomial interaction features
- Class balancing for imbalanced data
- Automatic feature selection (L1-based)
- Standard scaling + mean imputation
- Missing inputs automatically filled with dataset feature averages

### 🔹 Interactive Flask Web UI

- Clean feature input form
- Real-time prediction
- Graceful handling of missing form inputs

### 🔹 Automated Development Workflow

- `make install` — create venv + install dependencies
- `make train` — train the ML model
- `make run` — launch Flask app

---

## 📁 Project Structure

```text
BA305-Webapp-LogRegression/
│
├── app.py                 # Flask web server
├── target.py              # Model training + prediction logic
├── test_model.py          # Standalone model testing
├── requirements.txt       # Python dependencies
├── Makefile               # install / train / run commands
│
├── templates/
│   └── index.html         # Web UI template
│
├── static/
│   ├── script.js          # Form handling + fetch to /predict
│   └── style.css          # UI styling
│
├── .gitignore             # Keeps repo clean (ignores venv, data, pkls, etc.)
└── README.md              # This documentation
```
