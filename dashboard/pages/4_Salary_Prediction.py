import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import os
import pickle

from utils import (
    apply_style,
    sidebar,
    get_project_data,
    SELECTED_FEATURES,
    MODEL_FOLDER
)


st.set_page_config(
    page_title="Salary Prediction",
    page_icon="",
    layout="wide"
)


apply_style()
sidebar()


try:

    (
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
    ) = get_project_data()

except FileNotFoundError as e:

    st.error(str(e))

    st.stop()


model_path = os.path.join(
    MODEL_FOLDER,
    "random_forest.pkl"
)

with open(model_path, "rb") as file:
    rf = pickle.load(file)


st.markdown(
    """
    <div class="main-title">
        <h1>Salary Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <p style="
        color:#52667a;
        font-size:16px;
        margin-bottom:25px;
    ">
        Enter basic personal information and important
        academic & skill details to estimate salary package.
    </p>
    """,
    unsafe_allow_html=True
)


st.subheader("Personal Details")


with st.form("personal_details_form"):

    p1, p2, p3 = st.columns(3)

    with p1:

        student_id = st.number_input(
            "Student ID",
            min_value=int(raw["student_id"].min()),
            max_value=int(raw["student_id"].max()),
            value=int(raw["student_id"].median()),
            step=1
        )

    with p2:

        age = st.number_input(
            "Age",
            min_value=int(raw["age"].min()),
            max_value=int(raw["age"].max()),
            value=int(raw["age"].median()),
            step=1
        )

    with p3:

        gender = st.selectbox(
            "Gender",
            sorted(
                raw["gender"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

    personal_submit = st.form_submit_button(
        "Save Personal Details"
    )


if personal_submit:

    st.session_state["personal_details"] = {
        "student_id": student_id,
        "age": age,
        "gender": gender
    }

    st.success("Personal details saved successfully.")


if "personal_details" in st.session_state:

    st.caption(
        "Personal details saved for this prediction session."
    )


st.markdown("---")


st.subheader(" Salary Prediction Details")


st.caption(
    "Ye fields salary prediction ke liye model mein use hongi."
)


with st.form("salary_prediction_form"):

    c1, c2, c3 = st.columns(3)

    with c1:

        cgpa = st.number_input(
            "CGPA",
            min_value=float(raw["cgpa"].min()),
            max_value=float(raw["cgpa"].max()),
            value=float(raw["cgpa"].median()),
            step=0.01
        )


        branch = st.selectbox(
            "Branch",
            sorted(
                raw["branch"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


        college_tier = st.selectbox(
            "College Tier",
            sorted(
                raw["college_tier"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


        internships_count = st.number_input(
            "Internships Count",
            min_value=int(
                raw["internships_count"].min()
            ),
            max_value=int(
                raw["internships_count"].max()
            ),
            value=int(
                raw["internships_count"].median()
            ),
            step=1
        )


    with c2:

        projects_count = st.number_input(
            "Projects Count",
            min_value=int(
                raw["projects_count"].min()
            ),
            max_value=int(
                raw["projects_count"].max()
            ),
            value=int(
                raw["projects_count"].median()
            ),
            step=1
        )


        certifications_count = st.number_input(
            "Certifications Count",
            min_value=int(
                raw["certifications_count"].min()
            ),
            max_value=int(
                raw["certifications_count"].max()
            ),
            value=int(
                raw["certifications_count"].median()
            ),
            step=1
        )


        coding_skill_score = st.number_input(
            "Coding Skill Score",
            min_value=float(
                raw["coding_skill_score"].min()
            ),
            max_value=float(
                raw["coding_skill_score"].max()
            ),
            value=float(
                raw["coding_skill_score"].median()
            ),
            step=0.01
        )


        aptitude_score = st.number_input(
            "Aptitude Score",
            min_value=float(
                raw["aptitude_score"].min()
            ),
            max_value=float(
                raw["aptitude_score"].max()
            ),
            value=float(
                raw["aptitude_score"].median()
            ),
            step=0.01
        )


    with c3:

        communication_skill_score = st.number_input(
            "Communication Skill Score",
            min_value=float(
                raw["communication_skill_score"].min()
            ),
            max_value=float(
                raw["communication_skill_score"].max()
            ),
            value=float(
                raw["communication_skill_score"].median()
            ),
            step=0.01
        )


        mock_interview_score = st.number_input(
            "Mock Interview Score",
            min_value=float(
                raw["mock_interview_score"].min()
            ),
            max_value=float(
                raw["mock_interview_score"].max()
            ),
            value=float(
                raw["mock_interview_score"].median()
            ),
            step=0.01
        )


        attendance_percentage = st.number_input(
            "Attendance Percentage",
            min_value=float(
                raw["attendance_percentage"].min()
            ),
            max_value=float(
                raw["attendance_percentage"].max()
            ),
            value=float(
                raw["attendance_percentage"].median()
            ),
            step=0.01
        )


        backlogs = st.number_input(
            "Backlogs",
            min_value=int(
                raw["backlogs"].min()
            ),
            max_value=int(
                raw["backlogs"].max()
            ),
            value=int(
                raw["backlogs"].median()
            ),
            step=1
        )


    st.markdown("---")


    predict_button = st.form_submit_button(
        " Predict Salary"
    )


if predict_button:

    st.session_state["personal_details"] = {
        "student_id": student_id,
        "age": age,
        "gender": gender
    }


    input_data = pd.DataFrame([{

        "cgpa": cgpa,

        "branch": branch,

        "college_tier": college_tier,

        "internships_count": internships_count,

        "projects_count": projects_count,

        "certifications_count": certifications_count,

        "coding_skill_score": coding_skill_score,

        "aptitude_score": aptitude_score,

        "communication_skill_score":
            communication_skill_score,

        "mock_interview_score":
            mock_interview_score,

        "attendance_percentage":
            attendance_percentage,

        "backlogs": backlogs

    }])


    for col, le in encoders.items():

        if col in input_data.columns:

            value = str(
                input_data.loc[0, col]
            )

            if value not in le.classes_:

                st.error(
                    f"{col} ke liye unknown category mili: {value}"
                )

                st.stop()

            input_data[col] = le.transform(
                [value]
            )[0]


    input_data = input_data[
        SELECTED_FEATURES
    ]


    predicted_salary = float(
        rf.predict(input_data)[0]
    )


    predicted_salary = max(
        0.0,
        predicted_salary
    )


    st.markdown("---")


    st.subheader(" Prediction Result")


    r1, r2 = st.columns(2)


    r1.metric(
        "Predicted Salary",
        f"{predicted_salary:.2f} LPA"
    )


    r2.metric(
        "Model",
        "Random Forest"
    )


    st.markdown(
        f"""
        <div class="prediction-card">
            <p>Estimated Salary Package</p>
            <h2>{predicted_salary:.2f} LPA</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("---")


    st.subheader("Personal Details")


    personal_display = pd.DataFrame(
        [st.session_state["personal_details"]]
    )


    st.dataframe(
        personal_display,
        use_container_width=True,
        hide_index=True
    )


    st.subheader("Salary Prediction Details")


    display_data = pd.DataFrame([{

        "CGPA": cgpa,

        "Branch": branch,

        "College Tier": college_tier,

        "Internships Count":
            internships_count,

        "Projects Count":
            projects_count,

        "Certifications Count":
            certifications_count,

        "Coding Skill Score":
            coding_skill_score,

        "Aptitude Score":
            aptitude_score,

        "Communication Skill Score":
            communication_skill_score,

        "Mock Interview Score":
            mock_interview_score,

        "Attendance Percentage":
            attendance_percentage,

        "Backlogs":
            backlogs

    }])


    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "Ye prediction trained Random Forest model ka "
        "estimated result hai. Actual salary company, "
        "job role, skills aur placement process par "
        "depend kar sakti hai."
    )