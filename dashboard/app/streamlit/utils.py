import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Project folder structure:
#   <project root>/datasets/studentplacement.csv
#   <project root>/models/*.pkl            (notebook 05 me train hoke save hote hain)
#   <project root>/dashboard/app/streamlit/utils.py   <- ye file
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV_FILE = str(PROJECT_ROOT / "datasets" / "studentplacement.csv")
TARGET = "salary_package_lpa"

# Folder jahan trained models (.pkl) aur encoders save hain
MODEL_FOLDER = str(PROJECT_ROOT / "models")



SELECTED_FEATURES = [
    "cgpa",
    "branch",
    "college_tier",
    "internships_count",
    "projects_count",
    "certifications_count",
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "mock_interview_score",
    "attendance_percentage",
    "backlogs"
]



@st.cache_data
def load_raw_data():

    return pd.read_csv(CSV_FILE)



@st.cache_data
def prepare_data(raw):

    df = raw.copy()

    # Fill numerical missing values with mean
    df.fillna(
        df.mean(numeric_only=True),
        inplace=True
    )

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Dictionary to store encoders
    encoders = {}

    # Convert categorical columns into numbers
    for col in df.select_dtypes(include=["object", "string"]).columns:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = le

    return df, encoders



def load_pickle(file_name):

    file_path = os.path.join(
        MODEL_FOLDER,
        file_name
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"{file_name} nahi mili. Pehle notebook folder me "
            "01 se 05 tak ke notebooks run karo "
            "(models/ folder me .pkl files ban jayengi)."
        )

    with open(file_path, "rb") as file:

        return pickle.load(file)



@st.cache_resource
def load_saved_models(
    model_version="v3"
):

    # Dashboard sirf Random Forest use karta hai (prediction pages).
    # Baaki models (Extra Trees ~1.7 GB) dashboard me load nahi hote,
    # unka evaluation notebook 06 me hota hai.
    return {
        "Random Forest": load_pickle("random_forest.pkl")
    }



def get_salary_split(df, encoders):

    # Same split jo notebook 04 me hua tha (80/20, random_state=42)
    placed_value = encoders["placement_status"].transform(
        ["Placed"]
    )[0]

    salary_df = df[
        df["placement_status"] == placed_value
    ].copy()

    X = salary_df[SELECTED_FEATURES].copy()

    y = salary_df[TARGET].copy()

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


def apply_style():

    st.markdown("""
    <style>

    .stApp {
        background: #f5f8fc;
        color: #26384a;
    }

    .stApp p,
    .stApp li,
    .stApp span,
    .stApp label {
        color: #26384a;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #263238 !important;
    }

    .stMarkdown p,
    .stMarkdown li {
        color: #26384a !important;
    }

    [data-testid="stSidebar"] {
        background: #102a43;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #ffffff !important;
    }

    .main-title {

        background: linear-gradient(
            90deg,
            #2f6f9f,
            #3d91bd
        );

        padding: 22px 28px;

        border-radius: 14px;

        margin-bottom: 22px;
    }

    .main-title h1 {

        color: #263238 !important;

        margin: 0;

        font-size: 32px;

        font-weight: 700;
    }

    .main-title p {

        color: #ffffff !important;

        margin: 6px 0 0 0;

        font-size: 15px;
    }

    div[data-testid="stMetric"] {

        background: white;

        border-left: 5px solid #1976a8;

        padding: 12px;

        border-radius: 10px;

        box-shadow:
            0 2px 8px rgba(0,0,0,0.06);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p {

        color: #52667a !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] div {

        color: #123c69 !important;
    }

    div[data-testid="stMetricDelta"] {

        color: #1976a8 !important;
    }

    .section-title {

        color: #263238 !important;

        font-weight: 700;

        margin-top: 8px;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    .stSelectbox label,
    .stNumberInput label,
    .stTextInput label,
    .stSlider label,
    .stRadio label,
    .stCheckbox label {

        color: #26384a !important;

        font-weight: 600;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox input {

        color: #ffffff !important;

        -webkit-text-fill-color: #ffffff !important;
    }

    .stSelectbox [data-baseweb="select"] {

        background-color: #262733 !important;
    }

    .stSelectbox [data-baseweb="select"] * {

        color: #ffffff !important;

        -webkit-text-fill-color: #ffffff !important;
    }

    [data-baseweb="popover"] {

        background-color: white !important;
    }

    [data-baseweb="popover"] * {

        color: #26384a !important;
    }

    .stNumberInput div[data-baseweb="input"] {

        background-color: #262733 !important;
    }

    .stNumberInput div[data-baseweb="input"] input {

        color: #ffffff !important;

        -webkit-text-fill-color: #ffffff !important;
    }

    .stNumberInput button {

        color: #ffffff !important;

        background-color: #262733 !important;
    }

    .stTextInput div[data-baseweb="input"] {

        background-color: #262733 !important;
    }

    .stTextInput div[data-baseweb="input"] input {

        color: #ffffff !important;

        -webkit-text-fill-color: #ffffff !important;
    }

    .info-card {

        background: white;

        padding: 18px;

        border-radius: 12px;

        border: 1px solid #dbe7f2;

        margin-bottom: 15px;
    }

    .prediction-card {

        background: linear-gradient(
            135deg,
            #2f6f9f,
            #3d91bd
        );

        padding: 24px;

        border-radius: 14px;

        color: white;

        text-align: center;

        margin-top: 18px;
    }

    .prediction-card h2 {

        color: white !important;

        margin: 0;

        font-size: 34px;
    }

    .prediction-card p {

        color: white !important;
    }

    .small-note {

        color: #52667a !important;

        font-size: 13px;
    }

    [data-testid="stDataFrame"] {

        border-radius: 10px;
    }

    .stButton > button,
    .stFormSubmitButton > button {

        background: #1976a8;

        color: white !important;

        border: none;

        border-radius: 8px;

        font-weight: 600;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {

        background: #123c69;

        color: white !important;
    }

    [data-testid="stAlert"] p {

        color: #123c69 !important;
    }

    </style>
    """, unsafe_allow_html=True)



def sidebar():

    st.sidebar.markdown(
        "## Student Dashboard"
    )


def get_project_data():

    raw = load_raw_data()

    df, encoders = prepare_data(raw)

    models = load_saved_models("v3")

    X_train, X_test, y_train, y_test = get_salary_split(
        df,
        encoders
    )

    predictions = {}

    rows = []

    for name, model in models.items():

        pred = model.predict(X_test)

        predictions[name] = pred

        mse = mean_squared_error(
            y_test,
            pred
        )

        rows.append({

            "Model": name,

            "MAE": mean_absolute_error(
                y_test,
                pred
            ),

            "MSE": mse,

            "RMSE": np.sqrt(mse),

            "R2 Score": r2_score(
                y_test,
                pred
            )
        })

    metrics_df = pd.DataFrame(rows)

    return (

        raw,

        df,

        encoders,

        models,

        predictions,

        X_train,

        X_test,

        y_train,

        y_test,

        metrics_df
    )
