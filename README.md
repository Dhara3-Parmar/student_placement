# Student Placement & Salary Prediction

A data science project that analyses 100,000 student records, predicts the **salary package (LPA)** of a placed student with machine learning models, and shows everything in a **Streamlit dashboard**.

## 1. Project overview

- **Data:** 100,000 students, 26 columns (academic scores, skills, experience, lifestyle), no missing values, no duplicates.
- **Targets:** `placement_status` (Placed / Not Placed, 54.46% placed) and `salary_package_lpa` (0 for Not Placed students).
- **Main task:** regression, predict the salary of a **Placed** student from 12 features (`cgpa`, `branch`, `college_tier`, `internships_count`, `projects_count`, `certifications_count`, `coding_skill_score`, `aptitude_score`, `communication_skill_score`, `mock_interview_score`, `attendance_percentage`, `backlogs`).
- **Models:** Linear Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting.
- **Dashboard:** 5 pages (overview, data analysis, model performance, single salary prediction, bulk CSV prediction).

## 2. Folder structure

```
student_placement_project/
|-- datasets/
|   |-- studentplacement.csv          raw dataset
|   `-- processed/                    encoded data and train/test splits (created by notebooks 03 and 04)
|-- models/                           trained models, encoders (created by notebooks 03, 04, 05, 06)
|-- notebook/
|   |-- 01_data_understanding.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_data_preprocessing.ipynb
|   |-- 04_feature_engineering.ipynb
|   |-- 05_model_training.ipynb
|   `-- 06_model_evaluation.ipynb
|-- dashboard/
|   `-- app/
|       `-- streamlit/
|           |-- main.py                           dashboard home page
|           |-- utils.py                          data loading, model loading, styling
|           `-- pages/
|               |-- 2_Data_Analysis.py
|               |-- 3_Model_Performance.py
|               |-- 4_Salary_Prediction.py
|               `-- 5_Bulk_Salary_Prediction.py
|-- requirements.txt
`-- README.md
```

## 3. Setup

1. Install Python 3.9 or newer.
2. (Optional) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Mac / Linux
   ```
3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

## 4. How to run

**Step 1: run the notebooks in order** (this creates the processed data and the `.pkl` files in `models/`).
Open Jupyter from the project folder, go to `notebook/`, and run `01` to `06` from top to bottom ("Run All"):

```
jupyter notebook
```

| Notebook | What it does |
|---|---|
| 01_data_understanding | shape, columns, data types, missing values, duplicates, targets |
| 02_eda | charts and correlations |
| 03_data_preprocessing | missing values, duplicates, label encoding, saves `models/encoders.pkl` |
| 04_feature_engineering | selects Placed students and 12 features, feature importance, train/test split |
| 05_model_training | trains 5 models (about 2 minutes) and saves them as `.pkl` in `models/` |
| 06_model_evaluation | MAE, MSE, RMSE, R2 on the test set, plots, best model |

**Step 2: start the dashboard** from the project root folder:

```
streamlit run dashboard/app/streamlit/main.py
```

It opens at http://localhost:8501. The dashboard needs `models/random_forest.pkl` and `models/encoders.pkl`, so Step 1 (at least notebooks 03 and 05) must be done first.

## 5. Results (test set, 10,892 placed students)

| Model | MAE (LPA) | RMSE (LPA) | R2 |
|---|---|---|---|
| Linear Regression | 0.798 | 0.999 | 0.606 |
| Gradient Boosting | 0.801 | 1.003 | 0.603 |
| Random Forest | 0.820 | 1.024 | 0.586 |
| Extra Trees | 0.827 | 1.033 | 0.579 |
| Decision Tree | 0.981 | 1.229 | 0.404 |

Key findings from the EDA: salary depends mainly on `cgpa`, `coding_skill_score` and `internships_count`; most other features (including branch and college tier) show almost no relation with salary or placement.

## 6. Notes and limitations

- The dashboard predicts with **Random Forest** (R2 0.586), slightly below Linear Regression (0.606).
- The **Model Performance** dashboard page shows metrics that are written directly in the code (R2 0.97 to 0.99). They do not match the results above, which come from the saved models on this dataset. Notebook 06 compares both.
- Salary models are trained on Placed students only, because Not Placed students have a salary of 0.
- One train/test split (80/20, `random_state=42`), no cross-validation and no hyperparameter tuning.
- Random Forest (~850 MB) and Extra Trees (~1.7 GB) `.pkl` files are large; the dashboard loads only Random Forest.
- Pickle files should be loaded with the same scikit-learn version that created them, so create them on your own machine by running notebook 05.
