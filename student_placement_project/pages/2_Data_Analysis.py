import streamlit as st
import matplotlib.pyplot as plt

from utils import apply_style, sidebar, get_project_data

st.set_page_config(
    page_title="Data Analysis",
    layout="wide"
)

apply_style()
sidebar()

try:
    raw, df, encoders, models, predictions, X_train, X_test, y_train, y_test, metrics_df = get_project_data()
except FileNotFoundError:
    st.error("Not Found .csv")
    st.stop()

st.markdown("""
<div class="main-title">
    <h1>Data Analysis</h1>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(raw):,}")
c2.metric("Columns", f"{len(raw.columns)}")
c3.metric("Missing Values", f"{int(raw.isnull().sum().sum())}")
c4.metric("Duplicate Rows", f"{int(raw.duplicated().sum())}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("CGPA Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw["cgpa"], bins=20)
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Students")
    ax.set_title("CGPA Distribution")
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("Salary Package Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw["salary_package_lpa"], bins=20)
    ax.set_xlabel("Salary Package (LPA)")
    ax.set_ylabel("Students")
    ax.set_title("Salary Distribution")
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("CGPA vs Salary")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        raw["cgpa"],
        raw["salary_package_lpa"],
        alpha=0.20
    )
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Salary Package (LPA)")
    ax.set_title("CGPA vs Salary")
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("Study Hours vs Salary")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        raw["study_hours_per_day"],
        raw["salary_package_lpa"],
        alpha=0.20
    )
    ax.set_xlabel("Study Hours Per Day")
    ax.set_ylabel("Salary Package (LPA)")
    ax.set_title("Study Hours vs Salary")
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

st.subheader("Correlation with Salary")

numeric = raw.select_dtypes(include="number")
corr = numeric.corr()["salary_package_lpa"].sort_values(ascending=False)

st.dataframe(
    corr.to_frame("Correlation with Salary"),
    use_container_width=True
)

st.subheader("Numerical Dataset Summary")
st.dataframe(
    raw.describe().T.round(2),
    use_container_width=True
)
