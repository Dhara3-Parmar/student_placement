import streamlit as st
import pandas as pd
import os
import pickle

from utils import (
    apply_style,
    sidebar,
    SELECTED_FEATURES,
    MODEL_FOLDER
)


st.set_page_config(
    page_title="Bulk Salary Prediction",
    page_icon="",
    layout="wide"
)

apply_style()
sidebar()


model_path = os.path.join(
    MODEL_FOLDER,
    "random_forest.pkl"
)

encoder_path = os.path.join(
    MODEL_FOLDER,
    "encoders.pkl"
)


try:

    with open(model_path, "rb") as file:
        rf = pickle.load(file)

    with open(encoder_path, "rb") as file:
        encoders = pickle.load(file)

except FileNotFoundError:

    st.error(
        "Saved model files nahi mili. "
        "models folder check karo (notebooks 03 aur 05 run karo)."
    )

    st.stop()


st.markdown(
    """
    <div class="main-title">
        <h1>Bulk Salary Prediction</h1>
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
        Upload multiple student details in CSV format
        and predict salary for all students at once.
    </p>
    """,
    unsafe_allow_html=True
)


st.subheader("Upload Student CSV")


uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)


    st.success(
        "CSV file uploaded successfully."
    )


    st.write(
        "Total Students:",
        len(data)
    )


    missing_columns = []

    for column in SELECTED_FEATURES:

        if column not in data.columns:

            missing_columns.append(column)


    if len(missing_columns) > 0:

        st.error(
            "CSV me required columns missing hain:"
        )

        for column in missing_columns:

            st.write(
                "-",
                column
            )

        st.stop()


    st.subheader("Uploaded Student Data")

    st.dataframe(
        data.head(100),
        use_container_width=True,
        hide_index=True
    )

    if len(data) > 100:

        st.caption(
            "Preview ke liye first 100 students dikhaye gaye hain."
        )



    st.markdown("---")


    predict_button = st.button(
        "Predict Salary for All Students"
    )


    if predict_button:

        # Make a copy
        input_data = data.copy()



        for col, le in encoders.items():

            if col in input_data.columns:

                # Convert values to string
                input_data[col] = (
                    input_data[col]
                    .astype(str)
                )


                # Create mapping
                encoding_map = dict(
                    zip(
                        le.classes_,
                        le.transform(
                            le.classes_
                        )
                    )
                )


                # Check unknown values
                unknown_values = []

                for value in input_data[col].unique():

                    if value not in encoding_map:

                        unknown_values.append(value)


                if len(unknown_values) > 0:

                    st.error(
                        f"{col} me unknown category mili: "
                        f"{unknown_values}"
                    )

                    st.stop()


                # Apply encoding
                input_data[col] = (
                    input_data[col]
                    .map(encoding_map)
                )


        input_data = input_data[
            SELECTED_FEATURES
        ]



        with st.spinner(
            "Predicting salary for all students..."
        ):

            predictions = rf.predict(
                input_data
            )


        result_data = data.copy()


        result_data[
            "Predicted Salary (LPA)"
        ] = predictions


        # Avoid negative salary
        result_data[
            "Predicted Salary (LPA)"
        ] = result_data[
            "Predicted Salary (LPA)"
        ].clip(
            lower=0
        )


        st.markdown("---")

        st.subheader(
            "Bulk Prediction Result"
        )


        st.dataframe(
            result_data,
            use_container_width=True,
            hide_index=True
        )


        st.markdown("---")

        st.subheader(
            "Prediction Summary"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Students",
                len(result_data)
            )


        with col2:

            average_salary = (
                result_data[
                    "Predicted Salary (LPA)"
                ].mean()
            )

            st.metric(
                "Average Predicted Salary",
                f"{average_salary:.2f} LPA"
            )


        with col3:

            highest_salary = (
                result_data[
                    "Predicted Salary (LPA)"
                ].max()
            )

            st.metric(
                "Highest Predicted Salary",
                f"{highest_salary:.2f} LPA"
            )


        st.markdown("---")

        st.subheader(
            "Download Prediction Result"
        )


        csv_data = result_data.to_csv(
            index=False
        )


        st.download_button(
            label="Download Predicted CSV",
            data=csv_data,
            file_name="bulk_salary_predictions.csv",
            mime="text/csv"
        )


        st.success(
            "All student salaries predicted successfully."
        )


        st.info(
            "Ye predictions saved Random Forest model "
            "ke basis par generate ki gayi hain. "
            "Actual salary company, job role, skills "
            "aur placement process par depend kar sakti hai."
        )