import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# App title
st.title("🛌 Sleep Disorder Prediction")
st.write("This app uses logistic regression to predict sleep disorders based on health and lifestyle data.")

# File uploader
uploaded_file = st.file_uploader("https://raw.githubusercontent.com/Harshal1144/Sleep/refs/heads/main/Sleep.csv", type=["csv"])

if uploaded_file is not None:
    # Load and clean dataset
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)

    # Split 'Blood Pressure' into systolic and diastolic
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Visualize class distribution
    st.subheader("📊 Sleep Disorder Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Sleep Disorder', data=df, ax=ax)
    st.pyplot(fig)

    # Feature/target split
    X = df.drop('Sleep Disorder', axis=1)
    y = df['Sleep Disorder']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Show model performance
    st.subheader("📈 Model Performance")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    st.pyplot(fig2)

else:
    st.info("👆 Please upload a CSV file to begin.")
