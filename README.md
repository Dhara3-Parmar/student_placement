# models/

Trained models and encoders are saved here by the notebooks:

| File | Created by | Used by |
|---|---|---|
| `encoders.pkl` | Part 3 (Data Preprocessing) | dashboard (Salary Prediction, Bulk Salary Prediction) |
| `feature_columns.json` | Part 4 (Feature Engineering) | reference (feature order) |
| `linear_regression.pkl`, `decision_tree.pkl`, `random_forest.pkl`, `extra_trees.pkl`, `gradient_boosting.pkl` | Part 5 (Model Training) | Part 6 (Model Evaluation); dashboard uses `random_forest.pkl` |
| `model_metrics.csv` | Part 6 (Model Evaluation) | reference |

No scaler is saved because the project does not use one (tree models and Linear Regression do not need feature scaling).

`.pkl` files are not shipped with the zip: Random Forest is about 850 MB and Extra Trees about 1.7 GB, and pickles should be created with the same scikit-learn version that loads them. Run Parts 3 and 5 of `notebook/student_placement_project.ipynb` to create them.
