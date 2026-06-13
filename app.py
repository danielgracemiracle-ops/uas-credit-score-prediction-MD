import streamlit as st
import pandas as pd
import joblib

# =========================
# Load Model
# =========================

model = joblib.load("best_model.pkl")

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Score Prediction")
st.write(
    "Predict customer credit score using Random Forest Model"
)

# =========================
# Input Features
# =========================

col1, col2, col3 = st.columns(3)

with col1:

    Month = st.number_input("Month", value=0.0)

    Age = st.number_input("Age", value=30.0)

    Occupation = st.number_input("Occupation", value=0.0)

    Annual_Income = st.number_input(
        "Annual Income",
        value=50000.0
    )

    Monthly_Inhand_Salary = st.number_input(
        "Monthly Inhand Salary",
        value=4000.0
    )

    Num_Bank_Accounts = st.number_input(
        "Number of Bank Accounts",
        value=5.0
    )

    Num_Credit_Card = st.number_input(
        "Number of Credit Cards",
        value=5.0
    )

    Interest_Rate = st.number_input(
        "Interest Rate",
        value=10.0
    )

with col2:

    Num_of_Loan = st.number_input(
        "Number of Loans",
        value=2.0
    )

    Type_of_Loan = st.number_input(
        "Type of Loan",
        value=0.0
    )

    Delay_from_due_date = st.number_input(
        "Delay from Due Date",
        value=10.0
    )

    Num_of_Delayed_Payment = st.number_input(
        "Number of Delayed Payments",
        value=5.0
    )

    Changed_Credit_Limit = st.number_input(
        "Changed Credit Limit",
        value=5.0
    )

    Num_Credit_Inquiries = st.number_input(
        "Number of Credit Inquiries",
        value=5.0
    )

    Credit_Mix = st.number_input(
        "Credit Mix",
        value=0.0
    )

    Outstanding_Debt = st.number_input(
        "Outstanding Debt",
        value=1000.0
    )

with col3:

    Credit_Utilization_Ratio = st.number_input(
        "Credit Utilization Ratio",
        value=30.0
    )

    Payment_of_Min_Amount = st.number_input(
        "Payment of Min Amount",
        value=1.0
    )

    Total_EMI_per_month = st.number_input(
        "Total EMI per Month",
        value=100.0
    )

    Amount_invested_monthly = st.number_input(
        "Amount Invested Monthly",
        value=200.0
    )

    Payment_Behaviour = st.number_input(
        "Payment Behaviour",
        value=0.0
    )

    Monthly_Balance = st.number_input(
        "Monthly Balance",
        value=300.0
    )

    Credit_History_Months = st.number_input(
        "Credit History Months",
        value=200.0
    )

    Debt_to_Income_Ratio = st.number_input(
        "Debt to Income Ratio",
        value=0.05
    )

# =========================
# Prediction
# =========================

if st.button("Predict Credit Score"):

    input_data = pd.DataFrame({

        "Month": [Month],
        "Age": [Age],
        "Occupation": [Occupation],
        "Annual_Income": [Annual_Income],
        "Monthly_Inhand_Salary": [Monthly_Inhand_Salary],
        "Num_Bank_Accounts": [Num_Bank_Accounts],
        "Num_Credit_Card": [Num_Credit_Card],
        "Interest_Rate": [Interest_Rate],
        "Num_of_Loan": [Num_of_Loan],
        "Type_of_Loan": [Type_of_Loan],
        "Delay_from_due_date": [Delay_from_due_date],
        "Num_of_Delayed_Payment": [Num_of_Delayed_Payment],
        "Changed_Credit_Limit": [Changed_Credit_Limit],
        "Num_Credit_Inquiries": [Num_Credit_Inquiries],
        "Credit_Mix": [Credit_Mix],
        "Outstanding_Debt": [Outstanding_Debt],
        "Credit_Utilization_Ratio": [Credit_Utilization_Ratio],
        "Payment_of_Min_Amount": [Payment_of_Min_Amount],
        "Total_EMI_per_month": [Total_EMI_per_month],
        "Amount_invested_monthly": [Amount_invested_monthly],
        "Payment_Behaviour": [Payment_Behaviour],
        "Monthly_Balance": [Monthly_Balance],
        "Credit_History_Months": [Credit_History_Months],
        "Debt_to_Income_Ratio": [Debt_to_Income_Ratio]

    })

    prediction = model.predict(
        input_data
    )[0]
    
    probabilities = model.predict_proba(
        input_data
    )[0]
    
    prediction_map = {
    
        0: "Good",
        1: "Poor",
        2: "Standard"
    
    }
    
    st.success(
        f"Predicted Credit Score: {prediction_map[prediction]}"
    )
    
    st.subheader("Prediction Confidence")
    
    st.write(
        f"🟢 Good : {probabilities[0]*100:.2f}%"
    )
    
    st.write(
        f"🔴 Poor : {probabilities[1]*100:.2f}%"
    )
    
    st.write(
        f"🟡 Standard : {probabilities[2]*100:.2f}%"
    )
    
    st.progress(
        float(probabilities[prediction])
    )