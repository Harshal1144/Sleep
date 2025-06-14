import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page Configuration
st.set_page_config(page_title="Sleep Disorder Prediction", layout="wide")

# Navigation Bar
st.sidebar.title("🔍 Navigation")
option = st.sidebar.radio("Go to", ["Home", "Predict"])

# Data URL
DATA_URL = "https://raw.githubusercontent.com/Harshal1144/Sleep/main/Sleep.csv"

# Caching Data Load
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.dropna(inplace=True)
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)

    # Save original for encoding later
    original_df = df.copy()

    # Encode features
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    le_bmi = LabelEncoder()
    le_sleep = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Occupation'] = le_occupation.fit_transform(df['Occupation'])
    df['BMI Category'] = le_bmi.fit_transform(df['BMI Category'])
    df['Sleep Disorder'] = le_sleep.fit_transform(df['Sleep Disorder'])

    return df, original_df, le_gender, le_occupation, le_bmi, le_sleep

df, original_df, le_gender, le_occupation, le_bmi, le_sleep = load_data(DATA_URL)

# Split features and target
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Home Page
if option == "Home":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🛌 Sleep Disorder Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Using logistic regression to predict sleep disorders based on health and lifestyle data.</p>", unsafe_allow_html=True)

    st.subheader("📋 Dataset Preview")
    st.dataframe(original_df.head(), use_container_width=True)

    st.subheader("📊 Sleep Disorder Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Sleep Disorder', data=df, ax=ax)
    ax.set_title("Sleep Disorder Classes")
    st.pyplot(fig)

    st.subheader("📈 Model Performance")
    st.markdown(f"- **Accuracy**: `{accuracy_score(y_test, y_pred):.2f}`")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

# Prediction Page
elif option == "Predict":
    st.markdown("## 🧠 Predict Sleep Disorder")
    st.write("Fill in the patient information:")

    gender = st.selectbox("Gender", le_gender.classes_)
    age = st.slider("Age", 10, 100, 25)
    occupation = st.selectbox("Occupation", le_occupation.classes_)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
    bmi_category = st.selectbox("BMI Category", le_bmi.classes_)
    heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 70)
    sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 6.5)
    systolic_bp = st.number_input("Systolic Blood Pressure", 80, 200, 120)
    diastolic_bp = st.number_input("Diastolic Blood Pressure", 50, 130, 80)

    if st.button("🔍 Predict"):
        gender_encoded = le_gender.transform([gender])[0]
        occupation_encoded = le_occupation.transform([occupation])[0]
        bmi_cat_encoded = le_bmi.transform([bmi_category])[0]

        input_data = pd.DataFrame([[
            gender_encoded,
            age,
            occupation_encoded,
            bmi,
            bmi_cat_encoded,
            heart_rate,
            sleep_duration,
            systolic_bp,
            diastolic_bp
        ]], columns=X.columns)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        predicted_label = le_sleep.inverse_transform([prediction])[0]

        st.success(f"🩺 Predicted Sleep Disorder: **{predicted_label}**")
