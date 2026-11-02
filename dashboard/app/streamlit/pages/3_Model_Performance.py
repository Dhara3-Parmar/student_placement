import streamlit as st
import matplotlib.pyplot as plt

from utils import apply_style, sidebar, get_project_data

st.set_page_config(
    page_title="Model Performance",
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
    <h1>Model Performance</h1>
</div>
""", unsafe_allow_html=True)

# These are the values obtained in the supplied Jupyter notebook.
notebook_metrics = {
    "Linear Regression": {
        "MAE": 0.685,
        "MSE": 0.9204049403465818,
        "RMSE": 0.9593773711874707,
        "R2 Score": 0.9796589749531985,
        "Type": "Linear regression"
    },
    "Decision Tree": {
        "MAE": 0.632742,
        "MSE": 1.1541259,
        "RMSE": 1.074302517915694,
        "R2 Score": 0.9744937224802137,
        "Type": "Tree-based regression"
    },
    "Random Forest": {
        "MAE": 0.44850436499999996,
        "MSE": 0.5789051576945,
        "RMSE": 0.760858171865493,
        "R2 Score": 0.9872061482982128,
        "Type": "Ensemble regression"
    }
}

st.subheader("Complete Regression Model Details")

rows = []
for model_name, values in notebook_metrics.items():
    rows.append({
        "Model": model_name,
        "Type": values["Type"],
        "MAE": round(values["MAE"], 4),
        "MSE": round(values["MSE"], 4),
        "RMSE": round(values["RMSE"], 4),
        "R2 Score": round(values["R2 Score"], 4)
    })

details = st.dataframe(
    rows,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

c1, c2, c3 = st.columns(3)

for col, name in zip(
    [c1, c2, c3],
    ["Linear Regression", "Decision Tree", "Random Forest"]
):
    with col:
        st.markdown(f"### {name}")
        st.metric("R² Score", f"{notebook_metrics[name]['R2 Score']:.4f}")
        st.metric("MAE", f"{notebook_metrics[name]['MAE']:.4f}")
        st.metric("RMSE", f"{notebook_metrics[name]['RMSE']:.4f}")

st.markdown("---")

st.subheader("R² Score Comparison")

fig, ax = plt.subplots(figsize=(9, 4))
names = list(notebook_metrics.keys())
r2_values = [notebook_metrics[x]["R2 Score"] for x in names]

ax.bar(names, r2_values)
ax.set_ylabel("R² Score")
ax.set_ylim(0.90, 1.00)
ax.set_title("Regression Model R² Score")
for i, value in enumerate(r2_values):
    ax.text(i, value + 0.001, f"{value:.4f}", ha="center")
st.pyplot(fig, use_container_width=True)

st.markdown("---")

best_name = max(
    notebook_metrics,
    key=lambda x: notebook_metrics[x]["R2 Score"]
)

st.success(
    f"Best Model: {best_name} | "
    f"R² Score = {notebook_metrics[best_name]['R2 Score']:.4f}"
)

