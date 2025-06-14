import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Page config
st.set_page_config(page_title="Sleep Disorder Predictor", layout="centered")

# Load and preprocess data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Harshal1144/Sleep/main/Sleep.csv"
    df = pd.read_csv(url)
    df.dropna(inplace=True)

    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)

    # Encode categorical variables
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    le_bmi = LabelEncoder()
    le_sleep = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Occupation'] = le_occupation.fit_transform(df['Occupation'])
    df['BMI Category'] = le_bmi.fit_transform(df['BMI Category'])
    df['Sleep Disorder'] = le_sleep.fit_transform(df['Sleep Disorder'])

    X = df.drop(['Person ID', 'Sleep Disorder'], axis=1)
    y = df['Sleep Disorder']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return df, X, y, model, scaler, le_gender, le_occupation, le_bmi, le_sleep, X_train, X_test, y_train, y_test

# Load data
df, X, y, model, scaler, le_gender, le_occupation, le_bmi, le_sleep, X_train, X_test, y_train, y_test = load_data()

# Sidebar
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Overview", "🧠 Predict Sleep Disorder"])

# 📊 Data Overview
if page == "📊 Data Overview":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📊 Sleep Disorder Dataset Overview</h1>", unsafe_allow_html=True)

    st.subheader("🔹 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("📈 Show Sleep Disorder Distribution"):
        st.subheader("Sleep Disorder Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Sleep Disorder', data=df, ax=ax)
        ax.set_title("Sleep Disorder Classes")
        st.pyplot(fig)

    if st.button("🧩 Show Confusion Matrix"):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)

# 🧠 Predictor Page
elif page == "🧠 Predict Sleep Disorder":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Sleep Disorder Predictor</h1>", unsafe_allow_html=True)
    st.markdown("### Enter patient data below:")

    with st.form("prediction_form"):
        person_id = st.text_input("Person ID")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", le_gender.classes_)
            age = st.number_input("Age", min_value=10, max_value=100, value=30)
            occupation = st.selectbox("Occupation", le_occupation.classes_)
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
            bmi_category = st.selectbox("BMI Category", le_bmi.classes_)
            systolic_bp = st.number_input("Systolic BP", min_value=90, max_value=200, value=120)
        with col2:
            sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 6.5)
            quality_of_sleep = st.slider("Quality of Sleep (1–10)", 1, 10, 6)
            physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 300, 60)
            stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
            heart_rate = st.number_input("Heart Rate", min_value=40, max_value=130, value=70)
            daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=7000)
            diastolic_bp = st.number_input("Diastolic BP", min_value=60, max_value=130, value=80)

        submitted = st.form_submit_button("🔍 Predict")

        if submitted:
            input_dict = {
                "Gender": le_gender.transform([gender])[0],
                "Age": age,
                "Occupation": le_occupation.transform([occupation])[0],
                "Sleep Duration": sleep_duration,
                "Quality of Sleep": quality_of_sleep,
                "Physical Activity Level": physical_activity,
                "Stress Level": stress_level,
                "BMI": bmi,
                "BMI Category": le_bmi.transform([bmi_category])[0],
                "Heart Rate": heart_rate,
                "Daily Steps": daily_steps,
                "Systolic_BP": systolic_bp,
                "Diastolic_BP": diastolic_bp
            }

            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            predicted_label = le_sleep.inverse_transform([prediction])[0]

            st.success(f"🩺 Person ID: **{person_id}** — Predicted Sleep Disorder: **{predicted_label}**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
