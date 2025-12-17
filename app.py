# streamlit app for employee attrition prediction

import streamlit as st
import pandas as pd
import pickle

# load trained model
file_path = r"D:\Guvi\Project\Employee_Attrition\best_model.pkl"

with open(file_path, "rb") as f:
    pickle_model = pickle.load(f)
# --------------------------------------------------

# page config
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("Employee Attrition Prediction Dashboard")
st.write(
    "This dashboard helps HR teams identify **at-risk employees** "
    "and supports **data-driven retention strategies**."
)
# --------------------------------------------------

# input section
st.header("Enter Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    monthly_income = st.number_input("Monthly Income", min_value=1000, value=30000)
    total_working_years = st.number_input("Total Working Years", min_value=0, value=8)

with col2:
    years_at_company = st.number_input("Years at Company", min_value=0, value=5)
    years_in_current_role = st.number_input("Years in Current Role", min_value=0, value=3)
    years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, value=3)

with col3:
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative",
            "Manager", "Sales Representative", "Research Director", "Human Resources"
        ]
    )
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
    work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])

# --------------------------------------------------

# prepare input dataframe
input = pd.DataFrame([{
    "Age": age,
    "MonthlyIncome": monthly_income,
    "TotalWorkingYears": total_working_years,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": years_in_current_role,
    "YearsWithCurrManager": years_with_curr_manager,
    "OverTime": overtime,
    "JobRole": job_role,
    "JobLevel": job_level,
    "EnvironmentSatisfaction": environment_satisfaction,
    "WorkLifeBalance": work_life_balance
}])

# --------------------------------------------------

# encoding ine inputs
input_encoded = input.copy()

input_encoded["OverTime"] = input_encoded["OverTime"].map({"Yes": 0, "No": 1})

# One-hot encoding
input_encoded = pd.get_dummies(input_encoded,
                               columns=["JobRole", "JobLevel", "EnvironmentSatisfaction", "WorkLifeBalance"],
                               drop_first=False,
                               dtype=int
                              )

# Align columns like trained model
input_encoded = input_encoded.reindex(
                columns=pickle_model.feature_names_in_,
                fill_value=0
                )


# --------------------------------------------------


# predicting the input

st.header("Prediction Result")

if st.button("Predict Attrition Risk"):
    prediction = pickle_model.predict(input_encoded)[0]
    prediction_prob = pickle_model.predict_proba(input_encoded)[0, 1]

    if prediction == 1:
        st.error(f"High Risk of Attrition (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"Low Risk of Attrition (Probability: {prediction_prob:.2f})")

    st.subheader("How to Use This Insight")
    st.write(
        "- Focus retention efforts on high-risk employees\n"
        "- Review workload, role satisfaction, and work-life balance\n"
        "- Proactively engage managers and HR partners"
    )

# --------------------------------------------------
# model info section
# --------------------------------------------------
st.header("Model Information")

st.write(
    "• The prediction is generated using a trained machine learning model\n"
    "• Feature encoding is aligned exactly with training data\n"
    "• Metrics were evaluated offline using validation data"
)

st.info(
    "Note: Prediction accuracy for a single employee cannot be computed in real time "
    "because the true attrition outcome is unknown."
)