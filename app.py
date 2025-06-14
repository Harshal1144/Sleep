import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Set page config
st.set_page_config(page_title="Sleep Disorder Predictor", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/data/Sleep.csv")
    df.dropna(inplace=True)
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)

    # Label encoding
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

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📊 Dataset Overview", "📉 Visual Insights", "🧠 Predict Sleep Disorder"])

# 📊 Dataset Overview
if page == "📊 Dataset Overview":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📊 Sleep Disorder Dataset Overview</h1>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

# 📉 Visual Insights
elif page == "📉 Visual Insights":
    st.markdown("<h1 style='text-align: center; color: #F48C06;'>📉 Visual Insights</h1>", unsafe_allow_html=True)

    # Bar chart for sleep disorders
    st.subheader("🧩 Sleep Disorder Distribution")
    disorder_counts = df['Sleep Disorder'].value_counts()
    disorder_labels = le_sleep.inverse_transform(disorder_counts.index)

    fig1, ax1 = plt.subplots()
    sns.barplot(x=disorder_labels, y=disorder_counts.values, palette="Set2", ax=ax1)
    ax1.set_title("Sleep Disorder Distribution")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Sleep Disorder")
    st.pyplot(fig1)

    # Confusion matrix
    st.subheader("📌 Confusion Matrix")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_sleep.classes_, yticklabels=le_sleep.classes_, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Model Performance (Test Set)")
    st.pyplot(fig2)

# 🧠 Predictor
elif page == "🧠 Predict Sleep Disorder":
    st.markdown("<h1 style='text-align: center; color: #3A86FF;'>🧠 Sleep Disorder Prediction</h1>", unsafe_allow_html=True)
    st.markdown("### 🧾 Enter Patient Details")

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
