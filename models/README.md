# models/

Trained models and encoders are saved here by the notebooks:

| File | Created by | Used by |
|---|---|---|
| `encoders.pkl` | `03_data_preprocessing.ipynb` | dashboard (Salary Prediction, Bulk Salary Prediction) |
| `feature_columns.json` | `04_feature_engineering.ipynb` | reference (feature order) |
| `linear_regression.pkl`, `decision_tree.pkl`, `random_forest.pkl`, `extra_trees.pkl`, `gradient_boosting.pkl` | `05_model_training.ipynb` | `06_model_evaluation.ipynb`; dashboard uses `random_forest.pkl` |
| `model_metrics.csv` | `06_model_evaluation.ipynb` | reference |

No scaler is saved because the project does not use one (tree models and Linear Regression do not need feature scaling).

`.pkl` files are not shipped with the zip: Random Forest is about 850 MB and Extra Trees about 1.7 GB, and pickles should be created with the same scikit-learn version that loads them. Run notebooks 03 and 05 to create them.
