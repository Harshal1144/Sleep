import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page config
st.set_page_config(page_title="Sleep Disorder Prediction", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🛌 Sleep Disorder Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze sleep disorder data and predict disorder type using Logistic Regression.</p>", unsafe_allow_html=True)

# Navigation
option = st.sidebar.radio("Navigate", ["Dataset Overview", "Model Performance", "Predict"])

# Load CSV from GitHub
DATA_URL = "https://raw.githubusercontent.com/Harshal1144/Sleep/main/Sleep.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.dropna(inplace=True)
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)
    le = LabelEncoder()
    for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
        df[col] = le.fit_transform(df[col])
    return df

df = load_data(DATA_URL)

# Prepare data
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Dataset Overview Section
if option == "Dataset Overview":
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📊 Sleep Disorder Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Sleep Disorder', data=df, ax=ax)
    ax.set_title("Sleep Disorder Classes")
    st.pyplot(fig)

# Model Performance Section
elif option == "Model Performance":
    st.subheader("📈 Model Evaluation")
    st.markdown(f"- **Accuracy**: `{accuracy_score(y_test, y_pred):.2f}`")
    st.markdown("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

# Prediction Section
elif option == "Predict":
    st.subheader("🧠 Predict Sleep Disorder")
    st.write("Fill in the details below to predict the type of sleep disorder.")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 10, 100, 25)
    occupation = st.selectbox("Occupation", ["Doctor", "Engineer", "Nurse", "Lawyer", "Teacher", "Accountant", "Salesperson", "Scientist"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 70)
    systolic_bp = st.number_input("Systolic Blood Pressure", 80, 200, 120)
    diastolic_bp = st.number_input("Diastolic Blood Pressure", 50, 130, 80)
    sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 6.5)

    if st.button("Predict"):
        # Encode inputs
        input_df = pd.DataFrame([[
            1 if gender == "Male" else 0,
            age,
            occupation,
            bmi,
            1 if bmi_category == "Normal" else 2 if bmi_category == "Overweight" else 0,
            heart_rate,
            sleep_duration,
            systolic_bp,
            diastolic_bp
        ]], columns=X.columns)

        # Predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Decode label
        disorder_labels = {0: "Insomnia", 1: "No Disorder", 2: "Sleep Apnea"}
        result = disorder_labels.get(prediction, "Unknown")

        st.success(f"🩺 Predicted Sleep Disorder: **{result}**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
