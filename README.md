# Breast Cancer Prediction Web Application using Machine Learning

## 📌 Project Overview
This project is an end-to-end **Machine Learning–based Breast Cancer Prediction Web Application** developed using **Python, Flask, and Support Vector Machine (SVM)**.  
The application predicts breast cancer risk levels based on clinical features and provides probability-based results through a web interface.

---

## 🎯 Objective
The primary goals of this project are to:
- Build a reliable classification model for breast cancer prediction
- Optimize model performance using hyperparameter tuning
- Deploy the trained model as a Flask web application
- Enable real-time prediction with confidence scores

---

## 🧠 Machine Learning Model
- **Algorithm Used:** Support Vector Machine (SVM)
- **Kernel Options:** Linear, RBF, Polynomial
- **Hyperparameter Tuning:** GridSearchCV
- **Cross-Validation:** Stratified K-Fold (5 folds)
- **Library:** scikit-learn

---

## 📊 Dataset Description
- **Dataset Source:** scikit-learn Breast Cancer Wisconsin Dataset
- **Total Features:** 30 numerical features derived from digitized cell images
- **Target Classes:**
  - 0 → Malignant (High Risk)
  - 1 → Benign (Low Risk)

---

## ⚙️ Data Preprocessing
- Feature scaling performed using **StandardScaler**
- Ensures optimal performance of distance-based SVM models
- Data split into training and testing sets (70:30 ratio)

---

## 🧪 Model Training and Evaluation
- Best hyperparameters selected using GridSearchCV
- Stratified K-Fold ensures balanced class distribution during training
- Model outputs both:
  - Class prediction
  - Prediction probability (confidence score)

---

## 🌐 Web Application (Flask)
The Flask application allows users to:
- Enter key clinical measurements via a web form
- Receive real-time breast cancer risk predictions
- View prediction confidence as a probability percentage

### Flask Routes:
- `/` → Input form page
- `/predict` → Handles model inference and displays results

---

## 🖥️ Tech Stack Used
- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** scikit-learn
- **Data Processing:** NumPy
- **Model Optimization:** GridSearchCV
- **Feature Scaling:** StandardScaler
- **Frontend:** HTML (Jinja2 Templates)

---

## 📁 Project Structure
- breast-cancer-prediction-flask-ml/
- │
- ├── app.py # Flask application
- ├── templates/
- │ ├── index.html # Input form
- │ └── result.html # Prediction output
- ├── static/
- │ └──style.css # Webpage structure
- └── README.md # Project documentation
