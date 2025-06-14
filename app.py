import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Custom Streamlit Page Configuration
st.set_page_config(page_title="Sleep Disorder Prediction", layout="centered")

# App Title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🛌 Sleep Disorder Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This app uses Logistic Regression to predict sleep disorders based on health and lifestyle data.</p>", unsafe_allow_html=True)

# Load CSV directly from GitHub
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

# Dataset Preview
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# Distribution Plot
st.subheader("📊 Sleep Disorder Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Sleep Disorder', data=df, ax=ax)
ax.set_title("Sleep Disorder Classes")
st.pyplot(fig)

# Features and Target
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Model Evaluation
st.subheader("📈 Model Performance")
st.markdown(f"- **Accuracy**: `{accuracy_score(y_test, y_pred):.2f}`")
st.markdown("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("🧩 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
