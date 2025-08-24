import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Student Performance Predictor")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("student_performance.csv")

df = load_data()
#st.write("### Data Preview", df.head())

# Data Cleaning: keep only numeric columns for correlation
#numeric_cols = df.select_dtypes(include=np.number).columns
#corr = df[numeric_cols].corr()

#st.write("### Correlation Heatmap")
#fig, ax = plt.subplots(figsize=(12, 8))
#sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
#st.pyplot(fig)

# Feature Engineering
df['test_avg'] = (df['test1_score'] + df['test2_score']) / 2

# Model Training (simple for demo: use important features)
feature_cols = ['test1_score', 'test2_score', 'test_avg', 'study_hours_per_week', 'attendance_percentage']
X = df[feature_cols]
y = df['final_exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

st.write("## Predict Final Exam Score")
with st.form(key='prediction_form'):
    test1 = st.number_input("Test 1 Score", min_value=0.0, max_value=100.0)
    test2 = st.number_input("Test 2 Score", min_value=0.0, max_value=100.0)
    study_hours = st.number_input("Study Hours/Week", min_value=0.0)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
    submit = st.form_submit_button("Predict")
    if submit:
        test_avg = (test1 + test2) / 2
        data = np.array([[test1, test2, test_avg, study_hours, attendance]])
        pred = model.predict(data)
        st.success(f"Predicted Final Exam Score: {pred[0]:.2f}")

# Additional visualization
#st.write("### Study Hours vs Final Exam Score")
#fig2, ax2 = plt.subplots(figsize=(8, 6))
#sns.scatterplot(x='study_hours_per_week', y='final_exam_score', data=df, ax=ax2)
#st.pyplot(fig2)

#st.write("### Attendance vs Final Exam Score")
#fig3, ax3 = plt.subplots(figsize=(8, 6))
#sns.scatterplot(x='attendance_percentage', y='final_exam_score', data=df, ax=ax3)
#st.pyplot(fig3)

#st.write("### Model Feature Coefficients")
#coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_})
#st.dataframe(coef_df)