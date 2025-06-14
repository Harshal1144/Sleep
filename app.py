import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set Streamlit page configuration
st.set_page_config(page_title="Sleep Disorder Predictor", layout="wide")

# Load and preprocess the uploaded CSV
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/data/Sleep.csv")
    df.dropna(inplace=True)

    # Split Blood Pressure
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)

    # Encode categorical columns
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    le_bmi = LabelEncoder()
    le_sleep = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Occupation'] = le_occupation.fit_transform(df['Occupation'])
    df['BMI Category'] = le_bmi.fit_transform(df['BMI Category'])
    df['Sleep Disorder'] = le_sleep.fit_transform(df['Sleep Disorder'])

    X = df.drop('Sleep Disorder', axis=1)
    y = df['Sleep Disorder']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return df, X, y, model, scaler, le_gender, le_occupation, le_bmi, le_sleep, X_train, X_test, y_train, y_test

# Load everything
df, X, y, model, scaler, le_gender, le_occupation, le_bmi, le_sleep, X_train, X_test, y_train, y_test = load_data()

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📊 Dataset Dashboard", "📈 Visual Insights", "🧠 Predict Sleep Disorder"])

# 📊 Dataset Overview
if page == "📊 Dataset Dashboard":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📊 Sleep Disorder Dataset Overview</h1>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.info("Data preview shows the first few entries after cleaning and transformation.")

# 📈 Visual Analytics Page
elif page == "📈 Visual Insights":
    st.markdown("<h1 style='text-align: center; color: #F48C06;'>📈 Visual Analysis of Sleep Disorders</h1>", unsafe_allow_html=True)

    # Pie Chart of Sleep Disorders
    st.subheader("🧩 Sleep Disorder Distribution")
    disorder_counts = df['Sleep Disorder'].value_counts()
    disorder_labels = le_sleep.inverse_transform(disorder_counts.index)
    fig1 = px.pie(values=disorder_counts.values, names=disorder_labels, title='Sleep Disorder Classes')
    st.plotly_chart(fig1, use_container_width=True)

    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix (Test Data)")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_sleep.classes_, yticklabels=le_sleep.classes_, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

# 🧠 Prediction Page
elif page == "🧠 Predict Sleep Disorder":
    st.markdown("<h1 style='text-align: center; color: #3A86FF;'>🧠 Sleep Disorder Prediction</h1>", unsafe_allow_html=True)
    st.markdown("### 🔽 Enter Patient Details:")

    # Input Fields
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", le_gender.classes_)
        age = st.slider("Age", 10, 100, 30)
        occupation = st.selectbox("Occupation", le_occupation.classes_)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)

    with col2:
        bmi_category = st.selectbox("BMI Category", le_bmi.classes_)
        heart_rate = st.slider("Heart Rate", 40, 130, 70)
        sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 6.5)
        systolic_bp = st.number_input("Systolic BP", min_value=90, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic BP", min_value=60, max_value=130, value=80)

    # Prediction Button
    if st.button("🔍 Predict Sleep Disorder"):
        input_dict = {
            "Gender": le_gender.transform([gender])[0],
            "Age": age,
            "Occupation": le_occupation.transform([occupation])[0],
            "BMI": bmi,
            "BMI Category": le_bmi.transform([bmi_category])[0],
            "Heart Rate": heart_rate,
            "Sleep Duration": sleep_duration,
            "Systolic_BP": systolic_bp,
            "Diastolic_BP": diastolic_bp
        }

        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        predicted_label = le_sleep.inverse_transform([prediction])[0]

        st.success(f"🩺 Predicted Sleep Disorder: **{predicted_label}**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>🚀 Created with ❤️ using Streamlit | by Chetana</p>", unsafe_allow_html=True)
