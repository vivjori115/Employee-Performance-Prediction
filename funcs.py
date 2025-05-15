import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model and scaler
model_pkl = pickle.load(open('model_gridrf.pkl', 'rb'))
sc_pkl = pickle.load(open('sc.pkl', 'rb'))

# -----------------------------
# Embedded helper functions
# -----------------------------

# Read data from Excel and extract features + label
def read_file():
    file = r'EP.xls'
    data = pd.read_excel(file)
    final_cols = ['EmpEnvironmentSatisfaction','EmpLastSalaryHikePercent','EmpWorkLifeBalance',
                  'ExperienceYearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager',
                  'ExperienceYearsAtThisCompany','EmpDepartment','OverTime','EmpJobRole']
    X = data[final_cols]
    Y = data.loc[:,['PerformanceRating']]
    return X, Y

# Label encode categorical columns and store mappings
def map_cat_col(data):
    labelEncoder = LabelEncoder()
    category_col = ['EmpDepartment','OverTime','EmpJobRole']
    mapping_dict = {}
    for col in category_col:
        data[col] = labelEncoder.fit_transform(data[col])
        le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
        mapping_dict[col] = le_name_mapping
    return mapping_dict

# Mapping helpers
def get_key(val, col, mapping_dict):
    return mapping_dict[col].get(val, 0)

# Prepare input (already encoded, just return as is)
def prep_input(df):
    return df

# -----------------------------
# Streamlit App Starts Here
# -----------------------------

st.title("Employee Performance Prediction")

# Load and process data
X, Y = read_file()
X1 = X.copy()
mapping = map_cat_col(X1)

# UI inputs
env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
last_salary_hike = st.slider("Last Salary Hike (%)", 0, 25, 10)
work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
years_current_role = st.slider("Years in Current Role", 0, 20, 2)
years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
years_with_manager = st.slider("Years with Current Manager", 0, 20, 3)
years_at_company = st.slider("Years at Company", 0, 40, 5)

emp_department = st.selectbox("Department", list(mapping['EmpDepartment'].keys()))
overtime = st.selectbox("OverTime", list(mapping['OverTime'].keys()))
job_role = st.selectbox("Job Role", list(mapping['EmpJobRole'].keys()))

if st.button("Predict Performance Rating"):
    # Encode categorical inputs
    encoded_department = get_key(emp_department, 'EmpDepartment', mapping)
    encoded_overtime = get_key(overtime, 'OverTime', mapping)
    encoded_jobrole = get_key(job_role, 'EmpJobRole', mapping)

    # Collect input
    input_data = [
        env_satisfaction,
        last_salary_hike,
        work_life_balance,
        years_current_role,
        years_since_promotion,
        years_with_manager,
        years_at_company,
        encoded_department,
        encoded_overtime,
        encoded_jobrole
    ]

    # Append to dataframe
    X.loc[len(X)] = input_data

    # Predict
    y = prep_input(X)
    y = y.loc[[len(X) - 1]]
    y_scaled = sc_pkl.transform(y)
    output = model_pkl.predict(y_scaled)

    st.success(f"Predicted Employee Performance Rating: {output[0]}")
