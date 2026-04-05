import streamlit as st
import joblib
import pandas as pd


st.set_page_config(page_title="Bank Churn Predictor", layout="centered")

# Load the saved pipeline 
@st.cache_resource 
def load_model():
    return joblib.load('churn_model_pipeline.pkl')

model = load_model()

expected_columns = [
    'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
    'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

st.title(" Customer Attrition Risk Predictor")
st.markdown("Adjust the core drivers below to see how customer behavior affects churn risk.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Behavior")
    total_trans_ct = st.slider("Total Transaction Count", min_value=10, max_value=150, value=65)
    total_trans_amt = st.number_input("Total Transaction Amount ($)", min_value=500, max_value=20000, value=4500)
    total_ct_chng = st.slider("Change in Trans Count (Q4 vs Q1)", min_value=0.0, max_value=4.0, value=0.7)

with col2:
    st.subheader("Account Details")
    revolving_bal = st.slider("Total Revolving Balance ($)", min_value=0, max_value=3000, value=1100)
    relationship_count = st.selectbox("Total Relationship Count", options=[1, 2, 3, 4, 5, 6], index=2)
    months_inactive = st.slider("Months Inactive (Last 12 Mon)", min_value=0, max_value=6, value=2)

st.divider()

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    
    full_input_dict = {
        'Customer_Age': 46,                         # Baseline
        'Gender': 0,                              # Baseline
        'Dependent_count': 2,                       # Baseline
        'Education_Level': 'Graduate',              # Baseline
        'Marital_Status': 'Married',                # Baseline
        'Income_Category': 'Less than $40K',           # Baseline
        'Card_Category': 'Blue',                    # Baseline
        'Months_on_book': 36,                       # Baseline
        'Total_Relationship_Count': relationship_count, # From UI
        'Months_Inactive_12_mon': months_inactive,      # From UI
        'Contacts_Count_12_mon': 2,                 # Baseline
        'Credit_Limit': 4549.0,                     # Baseline
        'Total_Revolving_Bal': revolving_bal,           # From UI
        'Avg_Open_To_Buy': 7400.0,                  # Baseline
        'Total_Amt_Chng_Q4_Q1': 0.736,               # Baseline
        'Total_Trans_Amt': total_trans_amt,             # From UI
        'Total_Trans_Ct': total_trans_ct,               # From UI
        'Total_Ct_Chng_Q4_Q1': total_ct_chng,           # From UI
        'Avg_Utilization_Ratio': 0.176               # Baseline
    }

    input_data = pd.DataFrame([full_input_dict])

    input_data = input_data[expected_columns]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of Class 1 (Attrited)

    # Display results 
    if prediction == 1:
        st.error(f"###  High Risk of Churn")
        st.write(f"The model predicts an **{probability:.1%}** probability that this customer will close their account.")
    else:
        st.success(f"### Low Risk (Existing Customer)")
        st.write(f"The model predicts an **{probability:.1%}** probability of churn. This account is stable.")