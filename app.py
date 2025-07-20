
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Smart Health Risk Profiler", layout="wide")

st.title("🧠 Smart Health Risk Profiler")
st.markdown("Predict and visualize health risks using real NHANES data.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_nhanes.csv")

df = load_data()

# Display raw data
with st.expander("🔍 View Raw Data"):
    st.dataframe(df)

# Visualization
st.subheader("📊 BMI Distribution by Smoking Status")

try:
    # Check if the necessary columns exist
    if "SMQ020" in df.columns and "BMI" in df.columns:
        fig = px.box(
            df,
            x="SMQ020",
            y="BMI",
            labels={"SMQ020": "Smoking Status", "BMI": "Body Mass Index"},
            title="BMI Distribution by Smoking Status"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ Columns 'SMQ020' and/or 'BMI' not found in the uploaded data.")
        st.write("Available columns:", df.columns.tolist())
except Exception as e:
    st.error(f"❌ An error occurred while creating the BMI box plot: {e}")


# Predictive model section
st.subheader("🧪 Predict Risk of High Blood Pressure")

features = ["age", "BMI", "RIDAGEYR", "INDFMPIR"]
df_model = df.dropna(subset=features + ["BPQ020"])
X = df_model[features]
y = df_model["BPQ020"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Show metrics
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
st.json(report)

# User input
st.markdown("### Try It Yourself")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 35)
    bmi = st.slider("BMI", 15, 50, 25)

with col2:
    income = st.slider("Income-to-poverty ratio", 0.0, 5.0, 1.0)
    age_exact = st.number_input("Exact Age in Years", 18, 80, 35)

input_data = pd.DataFrame([[age, bmi, age_exact, income]], columns=features)
risk = clf.predict(input_data)[0]

st.markdown(f"### 🧠 Predicted High BP Risk: {'Yes' if risk else 'No'}")
