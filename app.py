import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -----------------------------
# Load Model and Scaler
# -----------------------------

@st.cache_resource
def load_model_and_scaler():
    model = pickle.load(open('model_gridrf.pkl', 'rb'))
    scaler = pickle.load(open('sc.pkl', 'rb'))
    return model, scaler

model, scaler = load_model_and_scaler()

# -----------------------------
# Helper Functions
# -----------------------------

def read_file():
    df = pd.read_csv("data.csv")  # Ensure this CSV is in your project folder
    X = df.drop("PerformanceRating", axis=1)
    Y = df["PerformanceRating"]
    return X, Y

def map_cat_col(df):
    return {
        "EmpDepartment": {val: idx for idx, val in enumerate(df["EmpDepartment"].unique())},
        "OverTime": {val: idx for idx, val in enumerate(df["OverTime"].unique())},
        "EmpJobRole": {val: idx for idx, val in enumerate(df["EmpJobRole"].unique())},
    }

def get_key(val, mapping):
    return mapping.get(val, 0)

def prep_input(df):
    return df.drop(['EmployeeID'], axis=1, errors='ignore')

# -----------------------------
# Streamlit UI
# -----------------------------


st.set_page_config(page_title="Employee Performance Predictor", layout="centered")

st.title("🔍 Employee Performance Prediction App")

# Load sample data to extract category mappings
X_sample, _ = read_file()
X1 = X_sample.copy()
mappings = map_cat_col(X1)

# Prediction Form
with st.form("prediction_form"):
    st.header("Enter Employee Information")

    col1, col2 = st.columns(2)
    with col1:
        EnvSat = st.number_input("Environment Satisfaction (1-4)", min_value=1, max_value=4)
        SalaryHike = st.number_input("Last Salary Hike (%)", min_value=0)
        WorkLife = st.number_input("Work Life Balance (1-4)", min_value=1, max_value=4)
        ExpRole = st.number_input("Years in Current Role", min_value=0)

    with col2:
        SincePromo = st.number_input("Years Since Last Promotion", min_value=0)
        CurrMgr = st.number_input("Years With Current Manager", min_value=0)
        ExpCompany = st.number_input("Years at This Company", min_value=0)

        Dept = st.selectbox("Department", list(mappings['EmpDepartment'].keys()))
        Overtime = st.selectbox("OverTime", list(mappings['OverTime'].keys()))
        JobRole = st.selectbox("Job Role", list(mappings['EmpJobRole'].keys()))

    submitted = st.form_submit_button("Predict")

# Prediction Logic
if submitted:
    dept_val = get_key(Dept, mappings["EmpDepartment"])
    ot_val = get_key(Overtime, mappings["OverTime"])
    role_val = get_key(JobRole, mappings["EmpJobRole"])

    input_data = [EnvSat, SalaryHike, WorkLife, ExpRole,
                  SincePromo, CurrMgr, ExpCompany, dept_val, ot_val, role_val]

    # Append to sample data to match format
    X_sample.loc[len(X_sample)] = input_data
    user_input = prep_input(X_sample).iloc[[-1]]

    # Scale and predict
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)

    st.success(f"🎯 Predicted Employee Performance Rating: {prediction[0]}")
