import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Set page config
st.set_page_config(page_title="Sleep Disorder Predictor", layout="centered")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Sleep.csv")  # or use your own local/online CSV path
    df.dropna(inplace=True)

    # Split Blood Pressure
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop(columns=['Blood Pressure', 'Person ID'], inplace=True)

    # Encode categorical variables
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

    return df, X.columns.tolist(), model, scaler, le_gender, le_occupation, le_bmi, le_sleep, X_test, y_test

# Load model and data
df, feature_cols, model, scaler, le_gender, le_occupation, le_bmi, le_sleep, X_test, y_test = load_data()

# Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Overview", "🧠 Predict Sleep Disorder"])

# Data Overview Page
if page == "📊 Data Overview":
    st.title("📊 Sleep Disorder Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("📈 Show Sleep Disorder Distribution"):
        st.subheader("Sleep Disorder Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Sleep Disorder', data=df, ax=ax)
        st.pyplot(fig)

    if st.button("🧩 Show Confusion Matrix"):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)

# Prediction Page
elif page == "🧠 Predict Sleep Disorder":
    st.title("🧠 Sleep Disorder Predictor")
    st.markdown("### Enter patient data:")

    gender = st.selectbox("Gender", le_gender.classes_)
    age = st.slider("Age", 10, 100, 30)
    occupation = st.selectbox("Occupation", le_occupation.classes_)
    sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 6.5)
    quality_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
    physical_activity = st.slider("Physical Activity Level", 0, 10, 5)
    stress = st.slider("Stress Level", 0, 10, 5)
    bmi_category = st.selectbox("BMI Category", le_bmi.classes_)
    heart_rate = st.slider("Heart Rate", 40, 130, 75)
    daily_steps = st.number_input("Daily Steps", 1000, 30000, 7000)
    systolic_bp = st.number_input("Systolic BP", 90, 200, 120)
    diastolic_bp = st.number_input("Diastolic BP", 60, 130, 80)

    if st.button("🔍 Predict"):
        input_dict = {
            'Gender': le_gender.transform([gender])[0],
            'Age': age,
            'Occupation': le_occupation.transform([occupation])[0],
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_sleep,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress,
            'BMI Category': le_bmi.transform([bmi_category])[0],
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic_BP': systolic_bp,
            'Diastolic_BP': diastolic_bp
        }

        input_df = pd.DataFrame([input_dict], columns=feature_cols)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        predicted_label = le_sleep.inverse_transform([prediction])[0]

        st.success(f"🩺 Predicted Sleep Disorder: **{predicted_label}**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
