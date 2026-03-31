import streamlit as st
import joblib
import pandas as pd

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Churn AI Assistant", layout="wide")

# ==============================
# LOAD FILES
# ==============================
model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ IMPORTANT: Define numerical columns (same as training)
numerical_cols = [
    'Age', 'Avg_Session_Time', 'Pages_Visited',
    'Last_Activity_Days', 'Email_Click_Rate',
    'Previous_Purchases', 'Support_Tickets',
    'Satisfaction_Score'
]

# ==============================
# PREPARE INPUT
# ==============================
def prepare_input(age, gender, avg_time, pages, sub_type,
                  last_activity, email_click, purchases,
                  tickets, satisfaction, city, device, course):

    input_dict = dict.fromkeys(features, 0)

    # Numerical
    input_dict['Age'] = age
    input_dict['Avg_Session_Time'] = avg_time
    input_dict['Pages_Visited'] = pages
    input_dict['Last_Activity_Days'] = last_activity
    input_dict['Email_Click_Rate'] = email_click
    input_dict['Previous_Purchases'] = purchases
    input_dict['Support_Tickets'] = tickets
    input_dict['Satisfaction_Score'] = satisfaction

    # Binary
    if 'Gender' in input_dict:
        input_dict['Gender'] = 1 if gender == "Male" else 0

    if 'Subscription_Type' in input_dict:
        input_dict['Subscription_Type'] = 1 if sub_type == "Paid" else 0

    # One-hot encoding (STRICT MATCH)
    for col in features:
        if col == f"City_{city}":
            input_dict[col] = 1
        elif col == f"Device_Type_{device}":
            input_dict[col] = 1
        elif col == f"Course_Category_{course}":
            input_dict[col] = 1

    return pd.DataFrame([input_dict])

# ==============================
# PREDICTION (FIXED)
# ==============================
def predict_churn(input_df):

    input_df = input_df[features]

    # ✅ SCALE ONLY NUMERICAL COLUMNS
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    prob = model.predict_proba(input_df)[0][1]

    if prob < 0.3:
        risk = "Low"
    elif prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return prob, risk

# ==============================
# RECOMMENDATIONS (STABLE)
# ==============================
def get_recommendation(satisfaction, last_activity, risk):

    recommendations = []

    if satisfaction <= 2:
        recommendations.append("Offer personalized discount based on user dissatisfaction")

    if last_activity > 20:
        recommendations.append("Send re-engagement email with relevant course suggestions")

    if risk == "High":
        recommendations.append("Provide limited-time premium access or special offer")

    if len(recommendations) < 2:
        recommendations.append("Recommend trending courses to increase engagement")

    return recommendations[:2]

# ==============================
# UI
# ==============================
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AI Churn Prediction Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    avg_time = st.number_input("Avg Session Time", 1.0, 300.0, 30.0)
    pages = st.number_input("Pages Visited", 1, 50, 10)
    sub_type = st.selectbox("Subscription", ["Free", "Paid"])

with col2:
    last_activity = st.number_input("Last Activity Days", 0, 100, 10)
    email_click = st.slider("Email Click Rate", 0.0, 1.0, 0.3)
    purchases = st.number_input("Previous Purchases", 0, 20, 1)
    tickets = st.number_input("Support Tickets", 0, 10, 0)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)

city = st.selectbox("City", ["Chennai", "Delhi", "Hyderabad", "Mumbai"])
device = st.selectbox("Device", ["Mobile", "Tablet"])
course = st.selectbox("Course", ["Business", "Data Science", "Design", "Web Dev"])

# ==============================
# BUTTON
# ==============================
if st.button("🚀 Predict Churn"):

    input_df = prepare_input(
        age, gender, avg_time, pages, sub_type,
        last_activity, email_click, purchases,
        tickets, satisfaction, city, device, course
    )

    prob, risk = predict_churn(input_df)

    st.markdown("## 📊 Prediction Result")
    st.success(f"Churn Probability: {prob}")
    st.info(f"Risk Level: {risk}")

    st.subheader("📌 Recommendations")

    recs = get_recommendation(satisfaction, last_activity, risk)

    for i, rec in enumerate(recs, 1):
        st.markdown(f"**{i}. {rec}**")