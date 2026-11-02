import streamlit as st
import matplotlib.pyplot as plt

from utils import apply_style, sidebar, get_project_data

st.set_page_config(
    page_title="Student Placement Dashboard",
    page_icon="",
    layout="wide"
)

apply_style()
sidebar()

try:
    raw, df, encoders, models, predictions, X_train, X_test, y_train, y_test, metrics_df = get_project_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.markdown("""
<div class="main-title">
    <h1> Student Placement & Salary Dashboard</h1>
</div>
""", unsafe_allow_html=True)

st.markdown('<h3 class="section-title">Project Overview</h3>', unsafe_allow_html=True)

placed = (raw["placement_status"].astype(str).str.strip().str.lower() == "placed").sum()
not_placed = len(raw) - placed
placement_rate = placed / len(raw) * 100
avg_package = raw["salary_package_lpa"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Students", f"{len(raw):,}")
c2.metric("Placed Students", f"{placed:,}")
c3.metric("Placement Rate", f"{placement_rate:.2f}%")
c4.metric("Average Package", f"{avg_package:.2f} LPA")

st.markdown("---")

left, right = st.columns(2)

with left:
    st.subheader("Placement Status")
    counts = raw["placement_status"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Placement Status")
    ax.set_ylabel("Students")
    ax.set_title("Placed vs Not Placed")
    plt.xticks(rotation=15)
    st.pyplot(fig, use_container_width=True)

with right:
    st.subheader("Average Salary by Branch")
    branch_salary = raw.groupby("branch")["salary_package_lpa"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    branch_salary.plot(kind="bar", ax=ax)
    ax.set_xlabel("Branch")
    ax.set_ylabel("Average Package (LPA)")
    ax.set_title("Average Package by Branch")
    plt.xticks(rotation=40)
    st.pyplot(fig, use_container_width=True)

st.markdown("---")
st.subheader("Dataset Preview")
st.dataframe(raw.head(10), use_container_width=True)

st.caption(
    "Dataset: 100,000 rows and 26 columns. "
    "Target variable: salary_package_lpa."
)