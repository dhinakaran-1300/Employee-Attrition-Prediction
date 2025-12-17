# Employee Attrition Prediction System

## Project Overview
This project predicts whether an employee is at risk of attrition using machine learning techniques.
It provides an interactive Streamlit dashboard that allows HR teams to input employee details and receive real-time attrition risk predictions.

---

## Problem Statement
Employee attrition leads to increased recruitment costs, productivity loss, and organizational instability.
This project aims to assist HR teams in identifying at-risk employees early and enabling data-driven retention strategies.

---

## Solution Approach
- Data preprocessing and feature encoding
- Handling class imbalance during training
- Model training and evaluation
- Deployment using Streamlit
- Pickle-based model loading for inference

---

## Machine Learning Model
- Model Used: RandomForestClassifier
- Reason for Selection:
  The Random Forest model achieved the highest performance among the tested models while maintaining balanced recall and robustness against overfitting.

---

## Features Used
- Age
- Monthly Income
- Total Working Years
- Years at Company
- Years in Current Role
- Years with Current Manager
- OverTime
- Job Role
- Job Level
- Environment Satisfaction
- Work Life Balance

---

## Project Structure
Employee-Attrition-Prediction/
│
├── app.py
├── best_model.pkl
├── employee_attrition_analysis.ipynb
├── app_test.ipynb
├── requirements.txt
├── README.md
└── .gitignore

---

## How to Run the Application

### 1. Clone the Repository
git clone https://github.com/<your-username>/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run Streamlit App
streamlit run app.py

---

## Output
- Low Risk of Attrition
- High Risk of Attrition (with probability score)

---

## Key Notes
- Feature encoding during prediction is aligned exactly with training data.
- Model accuracy cannot be calculated for a single employee input.
- Predictions are intended to support HR decisions, not replace them.

---

## Future Enhancements
- Add feature importance or SHAP explanations
- Integrate database support
- Deploy on Streamlit Cloud or AWS
- Add batch prediction functionality

---

## Author
Dhinakaran S
